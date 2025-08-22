from datetime import date
import pickle
import tempfile
import requests
import os

import numpy as np
import pandas as pd
from panel_imputer import PanelImputer
from rich.progress import Progress
import rioxarray as rxr
from tqdm import tqdm
import xarray as xr

from base.objects import Dataset, GlobalBaseGrid, ConfigParser
from base.datasets.wpp import WPPData
from utils.data_processing import create_custom_data_structure, slice_tuples
from utils.index import get_quarter
from utils.spatial_operations import pgid_to_coords, s_ceil, s_floor


class WorldPopData(Dataset):
    """Handles downloading and preprocessing of WorldPop population data.

    Implements `load_data()` to download annual country-level population GeoTIFFs
    from the WorldPop hub and a global pixel area raster, storing them as
    compressed NetCDFs in the processing directory.
    Implements `load_worldpop_areas()` to calculate grid cell areas based on the
    pixel values, caching the result.
    Implements `process_yearly_grid_aggregates()` to aggregate the downloaded
    country-level population counts to the grid. This includes adjustment to UN
    estimates and extrapolating population for recent years using UN WPP's
    estimated growth rates.
    Implements `prep_scale_factors()` to calculate a set of multipliers to apply
    to countryies' WorldPop layers to adjust to UN estimates and (past 2020)
    extrapolate with WPP growth estimates.
    Implements `get_quarterly_interpolations()` to interpolate the annual gridded
    population counts to a quarterly frequency.

    The class relies on an instance of `WPPData` for UN WPP figures used for
    country-level adjustments and extrapolation.

    Attributes:
        data_key (str): Set to "worldpop".
        local (bool): Set to False, as data is sourced via https.
        wpp (WPPData): An instance of WPPData for accessing UN population figures.
        wp_files (dict[int, list[str]]): Dictionary mapping years to the list of
            corresponding WorldPop NetCDF filenames in processing storage.
        data_loaded (bool): Flag indicating whether all available data is downloaded.
            Checked at the start of preprocessing.
        file_pixel_areas (str | None): Filepath to the cached NetCDF of global
            pixel areas, set after successful download in `load_data()`.

    """

    data_key: str = "worldpop"
    local: bool = False

    def __init__(self, config: ConfigParser):
        """Initializes WorldPopData.

        Sets up target years, checks storage for existing files, and initializes
        WPPData and flags.

        Args:
            config (ConfigParser): An initialized ConfigParser instance.
        """
        super().__init__(config=config)
        # wpp instance to
        self.wpp = WPPData(config=config)
        # check which years are already complete in storage
        # worldpop currently ends with 2020
        years = np.arange(self.global_config["start_year"], 2021)
        self.wp_files = {
            year: [
                f
                for f in sorted(os.listdir(self.storage.storage_paths["processing"]))
                if f.endswith(".nc") and str(year) in f
            ]
            for year in years
        }
        if self.regenerate["data"]:
            self.wp_files = {year: [] for year in years}
            all_available = self._check_files_available()
            self.wp_files_missing = {
                year: [f.replace(".tif", ".nc") for f in all_available[year]] for year in years
            }
        else:
            self.wp_files_missing = self._check_files_missing()
        self.data_loaded = False
        self.file_pixel_areas = None

    @staticmethod
    def _query_available():
        url = "https://hub.worldpop.org/rest/data/pop/wpgp"
        response = requests.get(url)
        wp_list = response.json()["data"]
        return wp_list

    def _check_files_missing(self) -> dict[int, list[str]]:
        wp_list = self._query_available()
        files_missing = {}
        for year in self.wp_files:
            # entries should cover the same scope for all years!
            entries = [e for e in wp_list if e["popyear"] == str(year)]
            iso3s = sorted([e["iso3"] for e in entries])
            filenames = [f"{iso3.lower()}_ppp_{year}.nc" for iso3 in iso3s]
            files_missing[year] = [f for f in filenames if f not in self.wp_files[year]]
        return files_missing

    def _check_files_available(self) -> dict[int, list[str]]:
        wp_list = self._query_available()
        files_available = {}
        for year in self.wp_files:
            # entries should cover the same scope for all years!
            entries = [e for e in wp_list if e["popyear"] == str(year)]
            iso3s = sorted([e["iso3"] for e in entries])
            files_available[year] = [f"{iso3.lower()}_ppp_{year}.tif" for iso3 in iso3s]
        return files_available

    def load_data(self):
        """Downloads WorldPop country-level population rasters and global pixel area.

        Iterates through years from the configured `start_year` to 2020. For each year:
        1. Fetches a list of available country datasets from the WorldPop API.
        2. For each country, constructs the download URL for its population GeoTIFF.
        3. Calls `_download_worldpop_file` to download the GeoTIFF, convert it to
           NetCDF, and store it, skipping if the NetCDF already exists and
           regeneration is not forced. Updates `self.wp_files`.
        After processing all years, downloads the global pixel area GeoTIFF,
        converts it to NetCDF, and stores its path in `self.file_pixel_areas`.
        Sets `self.data_loaded` to True upon successful completion of all downloads.
        Updates the global regeneration config for the 'data' stage of 'worldpop'.

        This method populates the processing storage and does not return data directly.
        """
        # Start wiuth land areas since this is outside the other logic
        url = "https://data.worldpop.org/GIS/Pixel_area/Global_2000_2020/0_Mosaicked/global_px_area_1km.tif"
        filename = url[url.rfind("/") + 1 :].replace(".tif", ".nc")
        fp = self.storage.build_filepath("processing", filename, filetype="")
        if not os.path.exists(fp) or self.regenerate["data"]:
            self.console.print("Downloading global pixel areas...")
            self._download_worldpop_file(url, fp, layer_name="land_area")
        self.file_pixel_areas = fp

        if (
            all(len(self.wp_files_missing[year]) == 0 for year in self.wp_files_missing)
            and not self.regenerate["data"]
        ):
            self.console.print("All required WorldPop country files already downloaded.")
        else:
            self.console.print(
                "Downloading WorldPop takes multiple hours per year if not in storage."
            )
            missing = sum([len(self.wp_files_missing[year]) for year in self.wp_files_missing])
            if self.regenerate["data"]:
                self.console.print(
                    'Re-downloading WorldPop completely, since "worldpop" key in regenerate["data"] in the global config.'
                )
            else:
                self.console.print(f"Currently {missing} files missing from storage.")

            with Progress(console=self.console) as progress:
                task_download = progress.add_task("[yellow]Downloading WorldPop ...", total=missing)
                for year in self.wp_files_missing:
                    filenames = [f for f in self.wp_files_missing[year]]
                    task_download_year = progress.add_task(
                        f"[blue]Downloading {year}...", total=len(filenames)
                    )
                    for f in filenames:
                        iso3 = f[: f.find("_")].upper()
                        url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020/{year}/{iso3}/{f.replace('.nc', '.tif')}"
                        fp = self.storage.build_filepath("processing", f, filetype="")
                        self._download_worldpop_file(url, fp)
                        self.wp_files[year].append(fp)
                        progress.update(task_download_year, advance=1)
                    progress.remove_task(task_download_year)
                progress.update(task_download, advance=1)
                self.console.print("Downloading WorldPop... DONE.")
        self.data_loaded = True
        self.config.set_regenerated_globally(self.data_key, "data")
        return

    def _download_worldpop_file(self, url, fp, layer_name="pop_count") -> None:
        """Downloads a WorldPop GeoTIFF, converts to NetCDF, and saves it.

        Streams the GeoTIFF from the given `url`. Saves it to a temporary local
        GeoTIFF file. Then, opens this temporary GeoTIFF, names the data array
        to `layer_name`, and saves it as a compressed NetCDF file. Handles large
        files by using Dask for loading and writing in chunks to limit memory
        requirements. Cleans up the temporary GeoTIFF file.

        Args:
            url (str): The URL of the WorldPop GeoTIFF file to download.
            fp (str): The local filepath for storing the data.
            layer_name (str, optional): The name to assign to data layer.
                Defaults to "pop_count".

        """
        response = requests.get(url, stream=True)
        response.raise_for_status()  # we want an error if download doesnt work
        with tempfile.NamedTemporaryFile("wb", suffix=".tif", delete=False) as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
            f.flush()
            fp_temp = f.name
        # read and store as nc for better compression
        da = (
            rxr.open_rasterio(fp_temp, chunks={"x": 10000, "y": 10000})
            .squeeze()  # type: ignore
            .drop_vars(["band"])
        )
        if da.size < 1000000000:
            da = da.load()  # no need for dask
            da.name = layer_name
            da.to_netcdf(
                fp, engine="netcdf4", encoding={layer_name: {"zlib": True, "complevel": 3}}
            )
        else:
            da.name = layer_name
            da.to_netcdf(
                fp,
                engine="netcdf4",
                encoding={
                    layer_name: {
                        "zlib": True,
                        "complevel": 3,
                        "chunksizes": (10000, 10000),
                    }
                },
            )
        da.close()
        # cleanup
        if fp_temp and os.path.exists(fp_temp):
            os.remove(fp_temp)
        return

    def load_worldpop_areas(self, grid: GlobalBaseGrid) -> pd.DataFrame:
        """Loads or calculates land areas for each 0.5-degree grid cell.

        Asserts that `self.data_loaded` is True. Loads the static 'wp_land.parquet'
        file if it has already been generated. Otherwise, or if
        `self.regenerate['preprocessing']` is True, calculates the grid cell land
        areas in km² by summing the areas of all pixels from the WorldPop area raster
        that fall within that cell's boundaries. The source file only contains
        information on land pixels, so no further filtering is required. Saves
        the resulting DataFrame as 'wp_land.parquet'.

        Args:
            grid (GlobalBaseGrid): The initialized GlobalBaseGrid instance.

        Returns:
            pd.DataFrame: A DataFrame indexed by 'pgid' with a 'land_area' column.
        """

        def aggregate_cell(lat: float, lon: float) -> float:
            cell = da.sel(x=slice(lon - 0.25, lon + 0.25), y=slice(lat + 0.25, lat - 0.25))
            return cell.sum().item()

        # load_data needs to be called to check for required files
        assert self.data_loaded
        fp = self.storage.build_filepath("processing", "wp_land")

        if self.regenerate["preprocessing"] or not os.path.exists(fp):
            da = xr.open_dataset(self.file_pixel_areas)["land_area"]  # type: ignore
            df = grid.load()
            df["land_area"] = df.apply(lambda x: aggregate_cell(x.lat, x.lon), axis=1)  # type: ignore
            df["land_area"] = df["land_area"] / 1e6  # convert to km²
            self.storage.save(df, "processing", "wp_land")
        else:
            df = self.storage.load("processing", "wp_land")
        return df

    def get_quarterly_interpolations(
        self, df_wp: pd.DataFrame, grid: GlobalBaseGrid, update: bool = True
    ) -> pd.DataFrame:
        """Interpolates annual gridded population data to quarterly frequency.

        If `update` is False, it first attempts to load previously interpolated
        quarterly data ('wp_index_quarterly.parquet') from processing. If found
        and up-to-date with the input `df_wp_yearly`, it's returned. Otherwise,
        it creates the standardized indicator data structure, imputes the
        missing values using linear interpolation and saves the result.

        Args:
            df_wp_yearly (pd.DataFrame): Annually aggregated WorldPop data from
                `process_yearly_grid_aggregates()`.
            grid (GlobalBaseGrid): The GlobalBaseGrid instance, used to create
                the target quarterly data structure.
            update (bool, optional): Flag whether to force regeneration. Since this
                is not part of all preprocessing steps, handle this separately
                from the standard regenerate logic. Defaults to True.

        Returns:
            pd.DataFrame: DataFrame indexed by ('pgid', 'year', 'quarter') with
                quarterly 'land_area', 'pop_count', and 'pop_density'.
        """
        if not update:
            try:
                df_out = self.storage.load("processing", "wp_index_quarterly")
                assert (
                    df_wp.index.get_level_values("year").max()
                    == df_out.index.get_level_values("year").max()
                )
                self.console.print(
                    "Quarterly interpolated population successfully loaded from storage."
                )
                return df_out
            except FileNotFoundError:
                self.console.print("No stored quarterly population found.")
            except AssertionError:
                self.console.print(
                    "Stored quarterly population does not yet cover the latest year."
                )

        self.console.print("Generating quarterly interpolated population data...")
        df_base = create_custom_data_structure(
            grid.load(), self.global_config["start_year"], get_quarter("last").year, quarterly=True
        )
        df_wp["quarter"] = 4
        df_wp = df_wp.set_index("quarter", append=True).sort_index()
        df_out = df_base.merge(
            df_wp[["land_area", "pop_count", "pop_density"]],
            how="left",
            left_index=True,
            right_index=True,
        )
        imputer = PanelImputer(
            location_index="pgid",
            time_index=["year", "quarter"],
            imputation_method="interpolate",
            interp_method="slinear",
            tail_behavior=["fill", "extrapolate"],
            parallelize=True,
            parallel_kwargs={"n_jobs": -2, "verbose": 1},
        )
        df_out.update(
            imputer.fit_transform(df_out[["land_area", "pop_count", "pop_density"]]),  # type: ignore
            overwrite=False,
        )
        self.storage.save(df_out, "processing", "wp_index_quarterly")
        self.console.print("Generating quarterly interpolated population data... DONE!")
        return df_out

    def process_yearly_grid_aggregates(self, grid: GlobalBaseGrid) -> pd.DataFrame:
        """Aggregates and extrapolates WorldPop data to the grid-year level.

        This is the core processing method for WorldPop population data, orchestrating
        the preprocessing.

        Checks for and loads existing cached yearly aggregates ('wp_index.parquet')
        through `_prep_base_df()`. Also loads grid cell land areas via
        `load_worldpop_areas()` and UN WPP data through `self.wpp`. Unless full
        regeneration is desired (self.regenerate["processing"] = True), iterates
        through missing years, calculates a scaling factor adjusting the data to
        UN WPP estimates and potentially extrapolating the population counts
        based on country-level UN WPP growth estimates (after 2020, no WorldPop
        data), and sums population counts per grid cell. Scale factors for
        adjusting/extrapolating are generated/ loaded via `prep_scale_factors()`.
        Calculates population density based on the land areas. Saves the yearly
        aggregated DataFrame as `wp_index.parquet` and updated global regenerate
        settings.

        Args:
            grid (GlobalBaseGrid): The initialized GlobalBaseGrid instance.

        Returns:
            pd.DataFrame: A DataFrame indexed by ('pgid', 'year') containing
                'pop_count', 'pop_density', and 'land_area' columns.
        """

        def aggregate_cell(pgid: int, adjustment_factor: float, da: xr.DataArray) -> float:
            """Helper function to calculate pixel sums within single grid cells."""
            lat, lon = pgid_to_coords(pgid)
            cell = da.sel(x=slice(lon - 0.25, lon + 0.25), y=slice(lat + 0.25, lat - 0.25))
            if np.isnan(cell).all():
                return 0
            else:
                agg = cell.sum().item() * adjustment_factor
                return agg

        land_areas = self.load_worldpop_areas(grid)
        df, years = self._prep_base_df(grid, land_areas)
        # This means everything needed already in the stored version, skip further processing
        if len(years) == 0:
            return df

        df_wpp = self.wpp.load_data()
        df_wpp = self.wpp.preprocess_wpp(df_wpp)
        df_wpp = df_wpp.reset_index()
        df_wpp["iso3"] = df_wpp["iso3"].str.replace("XKX", "KOS")
        df_wpp = df_wpp.set_index(["iso3", "year"]).sort_index()
        scale_factors = self.prep_scale_factors(df_wpp)
        # reshape grid-based land areas for use to pre-filter pgids
        df_filter = land_areas.reset_index().set_index(["lat", "lon"]).sort_index()[["pgid"]]
        with Progress(console=self.console) as progress:
            agg_task = progress.add_task("Aggregating WorldPop to grid...", total=len(years))
            for year in years:
                if year <= 2020:
                    files = self.wp_files[year]
                else:  # year > 2020 (max_year)
                    files = self.wp_files[2020]
                year_task = progress.add_task(f"Processing {year}...", total=len(files))
                for f in files:
                    iso3 = f[f.rfind("/") + 1 : f.rfind("/") + 4].upper()
                    fp = self.storage.build_filepath("processing", f, filetype="")
                    da = xr.open_dataset(fp)["pop_count"]
                    max_lat = float(s_ceil(da.y.max().item()))
                    max_lon = float(s_ceil(da.x.max().item()))
                    min_lat = float(s_floor(da.y.min().item()))
                    min_lon = float(s_floor(da.x.min().item()))
                    scale_factor = scale_factors[year][iso3]
                    # 300.000.000 limits to < 64 GB RAM - can be adjusted depending on the available resources.
                    # Chunking is only really essential for files covering large NA areas like RUS and USA
                    limit = 300000000
                    # Either load da into memory directly, or chunk and load the chunks
                    # Having the DataArray in memory speeds up the many small sum operations on it
                    if da.size > limit:
                        agg_results = {}
                        ys = np.append(np.arange(max_lat, min_lat, -10), min_lat)
                        xs = np.append(np.arange(min_lon, max_lon, 10), max_lon)
                        for i, j in slice_tuples(ys, xs):
                            if i == len(ys) - 1 or j == len(xs) - 1:
                                continue
                            da_temp = da.sel(
                                y=slice(ys[i], ys[i + 1]), x=slice(xs[j], xs[j + 1])
                            ).load()
                            prefiltered_pgids = df_filter.loc[
                                (slice(ys[i + 1], ys[i]), slice(xs[j], xs[j + 1])), "pgid"
                            ].values
                            # this shortens USA and RUS processing significantly (and some others)
                            # both have tons of NA due to crossing the date boundary and therefore covering lon -180 to 180
                            if np.isnan(da_temp).all():
                                temp_result = {pgid: 0 for pgid in prefiltered_pgids}
                            else:
                                temp_result = {
                                    pgid: aggregate_cell(pgid, scale_factor, da_temp)
                                    for pgid in prefiltered_pgids
                                }
                            agg_results = {**agg_results, **temp_result}
                            da_temp.close()
                    else:
                        if da.size < limit:
                            da.load()
                        prefiltered_pgids = df_filter.loc[
                            (slice(min_lat, max_lat), slice(min_lon, max_lon)), "pgid"
                        ].values

                        agg_results = {
                            pgid: aggregate_cell(pgid, scale_factor, da)
                            for pgid in prefiltered_pgids
                        }
                    da.close()
                    for pgid in agg_results:
                        df.at[(pgid, year), "pop_count"] += agg_results[pgid]
                    progress.update(year_task, advance=1)
                # calculate density
                df.loc[(slice(None), year), "pop_density"] = (
                    df.loc[(slice(None), year), "pop_count"]
                    / df.loc[(slice(None), year), "land_area"]
                )
                progress.update(agg_task, advance=1)
                progress.remove_task(year_task)
        self.storage.save(df, "processing", "wp_index")
        self.config.set_regenerated_globally(self.data_key, "preprocessing")
        return df

    def _prep_base_df(
        self, grid: GlobalBaseGrid, land_areas: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[int]]:
        """Prepares base yearly DataFrame and identifies years needing processing.

        Creates the yearly data structure and adds land areas. If not
        `self.regenerate["preprocessing"]`, it loads existing aggregated data
        ('wp_index.parquet'), updates the base DataFrame with existing pop_count
        and pop_density values, and determines which years still require processing.

        Args:
            grid (GlobalBaseGrid): The GlobalBaseGrid instance.
            land_areas (pd.DataFrame): Land areas DataFrame from `load_worldpop_areas()`.

        Returns:
            tuple[pd.DataFrame, list[int]]:
                - The prepared base DataFrame for population aggregation, indexed
                  by ('pgid', 'year'), containing 'land_area', 'pop_count' and 'pop_density'
                  columns.
                - A list of years that still need to be processed.
        """
        years = np.arange(self.global_config["start_year"], get_quarter("last").year + 1)
        filename = "wp_index"
        # data is only yearly
        df_base = create_custom_data_structure(
            grid.load(), self.global_config["start_year"], get_quarter("last").year, quarterly=False
        )
        df = df_base.reset_index().merge(
            land_areas["land_area"], how="left", left_on="pgid", right_on="pgid"
        )
        df = df.set_index(["pgid", "year"])
        df["pop_count"] = np.nan
        df["pop_density"] = np.nan
        if not self.regenerate["preprocessing"]:
            try:
                # load pre-existing processing and update df_base with it
                df_wp = self.storage.load("processing", filename)
                years = [y for y in years if y not in df_wp.index.get_level_values("year").unique()]
                df.update(df_wp, overwrite=False)
            except FileNotFoundError:
                pass
        # set NA to 0 for future aggregation
        df["pop_count"] = df.pop_count.apply(lambda x: 0 if pd.isna(x) else x)
        return df.sort_index(), years

    def prep_scale_factors(self, df_wpp: pd.DataFrame) -> dict[int, dict[str, float]]:
        """Pre-calculates and caches scale factors for UN adjustment and extrapolation.

        This method generates a dictionary of scaling factors used to adjust raw
        WorldPop country-level population rasters. For each target year and for
        each country:
        1.  **UN Adjustment (Years <= 2020):** It calculates a factor to scale the
            raw WorldPop raster sum to match the UN WPP total population for that
            country and year (via `_get_un_adjustment()`).
        2.  **Extrapolation (Years > 2020):** It uses the 2020 WorldPop raster as a
            base. It first calculates the UN adjustment factor for 2020. Then, it
            multiplies this by a cumulative growth rate (from 2020 to the target
            year) obtained from UN WPP data (via `self.wpp.get_wpp_multiplier_2020()`).
        If a country is not found in the provided UN WPP data, the adjustment/
        growth factor defaults to 1 for that country-year.

        The resulting dictionary, mapping `year -> iso3 -> scale_factor`, is
        cached to a pickle file in the processing directory as `scale_factors.pickle`.
        If the file exists and is up to date, it is loaded from disk instead.

        Args:
            df_wpp (pd.DataFrame): Preprocessed UN WPP data, indexed by
                ('iso3', 'year'), containing 'pop_total' and 'pop_change'.

        Returns:
            dict[int, dict[str, float]]: A nested dictionary where outer keys are
                years, inner keys are ISO3 country codes, and values are the
                combined scale factors.
        """

        def get_scale_factors():
            self.console.print(
                "Calculating multipliers to adjust raw WorldPop to UN estimates and extrapolate past 2020..."
            )
            scale_factor_dict = {}
            for year in years:
                scale_factor_dict[year] = {}
                if year <= 2020:
                    files = self.wp_files[year]
                else:  # year > 2020 (max_year)
                    files = self.wp_files[2020]
                for f in tqdm(files, desc=f"{year}"):
                    iso3 = f[f.rfind("/") + 1 : f.rfind("/") + 4].upper()
                    fp = self.storage.build_filepath("processing", f, filetype="")
                    da = xr.open_dataset(fp)["pop_count"]
                    # calculate scale factor from raw worldpop
                    if year <= 2020:
                        growth_rate = 1
                        try:
                            un_total = float(df_wpp.loc[(iso3, year), "pop_total"])  # type: ignore
                            un_adjust_factor = self._get_un_adjustment(da, un_total, iso3)
                        except KeyError:
                            un_adjust_factor = 1  # defaults to 1
                            self.console.log(
                                f"{iso3} not found in 'df_wpp', no UN adjustment performed and no growth assumed."
                            )
                    else:
                        growth_rate = self.wpp.get_wpp_multiplier_2020(df_wpp, year, iso3, False)
                        try:
                            un_total = float(df_wpp.loc[(iso3, 2020), "pop_total"])  # type: ignore
                            un_adjust_factor = self._get_un_adjustment(da, un_total, iso3)
                        except KeyError:
                            un_adjust_factor = 1  # defaults to 1
                            self.console.log(
                                f"{iso3} not found in 'df_wpp', no UN adjustment performed and no growth assumed."
                            )
                    scale_factor = un_adjust_factor * growth_rate
                    scale_factor_dict[year][iso3] = scale_factor
            # store
            with open(fp_scale_factors, "wb") as f:
                pickle.dump(scale_factor_dict, f)
            return scale_factor_dict

        years = list(np.arange(self.global_config["start_year"], date.today().year + 1))
        fp_scale_factors = self.storage.build_filepath(
            "processing", "scale_factors", filetype=".pickle"
        )
        if not self.regenerate["preprocessing"] and os.path.exists(fp_scale_factors):
            with open(fp_scale_factors, "rb") as f:
                scale_factor_dict = pickle.load(f)
            if max(years) in scale_factor_dict.keys():
                return scale_factor_dict
            else:
                scale_factor_dict = get_scale_factors()
        else:
            scale_factor_dict = get_scale_factors()
        return scale_factor_dict

    def _get_un_adjustment(self, da: xr.DataArray, un_total: float, iso3: str) -> float:
        """Adjusts WorldPop population counts to match a UN estimates.

        Calculates a scaling factor by comparing the sum of the input DataArray
        to the UN WPP estimate and applies this factor to the DataArray.

        Args:
            da (xr.DataArray): Input WorldPop population DataArray for a country/year.
            un_total (float): Target total population from UN WPP data.
            iso3 (str): Current country, used for warning message.

        Returns:
            xr.DataArray: The DataArray with values scaled to sum to `un_total`.
        """
        if da.size > 1000000000:  # limit xr.where to 10-15GB RAM
            # memory efficiency: chunk the calcualtion if the size of the country is too big
            max_y = float(s_ceil(da.y.max().item()))
            max_x = float(s_ceil(da.x.max().item()))
            min_y = float(s_floor(da.y.min().item()))
            min_x = float(s_floor(da.x.min().item()))
            ys = np.append(np.arange(max_y, min_y, -10), min_y)
            xs = np.append(np.arange(min_x, max_x, 10), max_x)
            sums = []
            for i, j in slice_tuples(ys, xs):
                if i == len(ys) - 1 or j == len(xs) - 1:
                    continue
                da_temp = da.sel(y=slice(ys[i], ys[i + 1]), x=slice(xs[j], xs[j + 1]))
                partial_sum = da_temp.sum().item()
                sums.append(partial_sum)
            wp_total = sum(sums)
        else:
            wp_total = da.sum().item()
        try:
            factor = un_total / wp_total
        except ZeroDivisionError:
            self.console.print(f"WordPop data for {iso3} sums to 0, no adjustment.")
            factor = 1
        return factor
