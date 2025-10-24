from datetime import date
import gzip
import json
import numpy as np
import pandas as pd
import re
import requests
import os


from dotenv import load_dotenv
from rich.progress import Progress
import rioxarray as rxr
import xarray as xr

from base.objects import ConfigParser, Dataset, GlobalBaseGrid
from utils.data_processing import create_custom_data_structure
from utils.index import get_quarter
from utils.spatial_operations import coords_to_pgid, s_ceil, s_floor


class NTLDataError(Exception):
    """Custom Exception for Nighttime Lights (NTL) data-specific errors to handle
    related problems distinctly.
    """

    pass


class NTLData(Dataset):
    """Handles downloading and preprocessing of Nighttime Lights (NTL) data.

    Implements `load_data()` to download annual harmonized NTL data (mean, median)
    and corresponding lit masks from the Colorado School of Mines Earth Observation Group.
    Implements `preprocess_yearly()` to aggregate the high-resolution NTL data
    to the base grid annually and cache the result.
    Implements `get_masked_median_data()` to load and return masked median NTL data
    for a specific year as an xarray Dataset.

    Attributes:
        data_key (str): Set to "ntl".
        local (bool): Set to False, as data is sourced via https.
        years (list[int]): List of years for which NTL data is expected,
            going from global config's `start_year` up to the last complete year.
        ntl_files (list[str]): List of NTL filenames found in the processing
            storage, set during initialization (set to empty if `regenerate['data']`
            is True) and updated during `load_data()`.
        years_missing (list[int]): Years for which NTL data files (mean, median,
            mask) are not all present in storage.
        years_unavailable (list[int]): Years for which data is not (yet) available
            up to the latest year - used for preprocessing. Initialized with the
            current year and extended during `load_data()` if applicable.
        data_loaded (bool): Flag indicating whether all available data is downloaded.
            Checked at the start of preprocessing.
        auth_headers (dict[str, str] | None): Authentication headers for EOGdata API,
            set by `_eog_authenticate()`. Initialized to None.
    """

    data_key: str = "ntl"
    local: bool = False

    def __init__(self, config: ConfigParser):
        """Initializes the NTLData instance.

        Sets up year ranges, checks processing storage for existing NTL files
        to determine which years are missing, and initializes status flags.

        Args:
            config (ConfigParser): An initialized ConfigParser instance.
        """
        super().__init__(config=config)
        # we care about complete years from the start year onwards
        self.years: list[int] = np.arange(
            self.global_config["start_year"], date.today().year
        ).tolist()
        # check which years are already complete in storage
        self.ntl_files = [
            f for f in os.listdir(self.storage.storage_paths["processing"]) if f.endswith(".tif")
        ]
        if self.regenerate["data"]:
            self.ntl_files = []
        self.years_missing = self._check_processing_storage()
        self.years_unavailable = [date.today().year]
        self.data_loaded = False
        self.auth_headers = None

    def _check_processing_storage(self) -> list[int]:
        """Checks processing storage for complete sets of NTL files for each year.

        For each year in `self.years`, it verifies if the corresponding 'lit_mask',
        'average_masked', and 'median_masked' GeoTIFF files are present in
        `self.ntl_files`.

        Returns:
            list[int]: A list of years for which one or more NTL file versions
                are missing from storage.
        """
        years_missing = []
        for year in self.years:
            files_year = [f for f in self.ntl_files if f"_{year}_" in f]
            if not all(
                [
                    any([version in f for f in files_year])
                    for version in ["lit_mask", "average_masked", "median_masked"]
                ]
            ):
                years_missing.append(year)
        return years_missing

    def load_data(self) -> None:
        """Downloads annual NTL data (mean, median, mask) for missing years.

        Determines which years require data download based on `self.years_missing`
        or `self.regenerate['data']`. If all necessary files are present and
        regeneration is not forced, it skips downloads.
        Otherwise, it authenticates using `_eog_authenticate()`. For each target year:
        1. Scrapes the EOG annual file directory (URL constructed based on year
           and NTL version v21/v22) to find filenames for the global
           'median_masked', 'average_masked', and 'lit_mask' GeoTIFF.gz files
           using regular expressions.
        2. Calls `_download_decompress_gz` to download and decompress each
           required file into the processing directory, skipping files that
           already exist unless regeneration is forced.
        Handles HTTP errors during directory scraping signifying a year's data is
        not yet available and updates `self.years_unavailable` accordingly.
        Tracks download progress, successes, skips, and failures and outputs the
        summary to the console after completion. Sets `self.data_loaded` to True
        if all downloads for the targeted years succeed without error. Updates
        the global regeneration config for the 'data' stage of 'ntl' upon
        successful completion.

        This method does not return data directly but populates the processing storage.
        """
        if self.regenerate["data"]:
            years = self.years
        else:
            years = self.years_missing

        # skip everything if data in storage
        if len(years) == 0:
            self.console.print("All NTL files already downloaded.")
            self.data_loaded = True
        else:
            with Progress(console=self.console) as progress:
                self._eog_authenticate()
                task_download = progress.add_task(
                    "[red]Downloading NTL...",
                    total=len(self.years) * 3,
                    completed=(len(self.years) - len(years)) * 3,
                )
                downloaded, skipped, failed = 0, 0, 0
                for year in years:
                    if year <= 2021:  # from 2022 onwards the version is 2.2
                        version = "v21"
                    else:
                        version = "v22"
                    url = f"https://eogdata.mines.edu/nighttime_light/annual/{version}/{year}/"
                    files = []
                    try:
                        response = requests.get(url)
                        response.raise_for_status()
                        # the regex might break in the future, the naming is unfortunately not the most consistent
                        ntl_median = re.search(
                            r"VNL_.{30,60}.median_masked\.dat\.tif\.gz<\/a>", response.text
                        )
                        if ntl_median is not None:
                            files.append(ntl_median.group(0)[:-4])
                        else:
                            progress.console.print(
                                f":warning: NTL median filename not found for year {year}. Doublecheck source and/or regex search term."
                            )
                            failed += 1
                        ntl_mean = re.search(
                            r"VNL_.{30,60}.average_masked\.dat\.tif\.gz<\/a>", response.text
                        )
                        if ntl_mean is not None:
                            files.append(ntl_mean.group(0)[:-4])
                        else:
                            progress.console.print(
                                f":warning: NTL median filename not found for year {year}. Doublecheck source and/or regex search term."
                            )
                            failed += 1
                        mask = re.search(r"VNL_.{30,60}.lit_mask\.dat\.tif\.gz<\/a>", response.text)
                        if mask is not None:
                            files.append(mask.group(0)[:-4])
                        else:
                            progress.console.print(
                                f":warning: NTL lit mask filename not found for year {year}. Doublecheck source and/or regex search term."
                            )
                            failed += 1
                        progress.update(task_download, advance=0.3)

                    except requests.HTTPError as e:
                        progress.console.print(
                            f"Error in accessing data. :warning: Year {year} failed. Continuing with next year."
                        )
                        print(e)
                        failed += 3
                        # last year may not be available, yet
                        if year == date.today().year - 1:
                            self.years_unavailable.append(year)
                            try:
                                self.years.remove(year)
                            except ValueError:
                                pass
                            try:
                                self.years_missing.remove(year)
                            except ValueError:
                                pass
                        progress.update(task_download, advance=3)
                        continue

                    for f in files:
                        filename = f[:-3]
                        # filetype already included in filename
                        fp = self.storage.build_filepath("processing", filename, filetype="")
                        if os.path.exists(fp) and not self.regenerate["data"]:
                            progress.console.print(
                                f'File "{filename}" already exists, skipping download.'
                            )
                            skipped += 1
                        else:
                            url_file = url + f
                            success = self._download_decompress_gz(url_file, fp)
                            if success:
                                downloaded += 1
                                self.ntl_files.append(filename)
                            else:
                                failed += 1
                        progress.update(task_download, advance=0.9)
                progress.console.print(
                    f"{downloaded} files newly downloaded, {skipped} already existing files skipped, {failed} errors."
                )
            if failed == 0:
                self.data_loaded = True
                self.regenerate["data"] = False
        return

    def _download_decompress_gz(self, url: str, fp: str, chunk_size: int = 2048) -> bool:
        """Downloads a .gz file from a URL, decompresses it, and saves it to a path.

        Args:
            url (str): The URL of the .gz file to download.
            fp (str): The local filepath where the decompressed file will be saved.
            chunk_size (int, optional): Size of chunks for streaming download and
                decompression. Defaults to 2048.

        Returns:
            bool: True if download and decompression were successful, False otherwise.
        """
        try:
            with requests.get(url, headers=self.auth_headers, stream=True) as r:
                r.raise_for_status()
                with gzip.GzipFile(fileobj=r.raw) as decompressed:
                    with open(fp, "wb") as fout:
                        for chunk in iter(lambda: decompressed.read(chunk_size), b""):
                            fout.write(chunk)
            return True
        except requests.HTTPError as e:
            print(e)
            return False

    def _eog_authenticate(self) -> None:
        """Authenticates with the EOG data service and stores authorization headers.

        If `self.auth_headers` is not already set, this method loads EOG user
        and password from .env, requests and acccess token and create the headers
        for the GET request to download NTL files.

        Code copied from https://eogdata.mines.edu/products/register/

        This method is typically called by `load_data()` before starting downloads.
        """
        if self.auth_headers is None:
            load_dotenv()
            username = os.getenv("EOG_USER")
            assert username is not None, "EOG_USER missing from .env or .env not loaded correctly."
            password = os.getenv("EOG_PASSWORD")
            assert password is not None, (
                "EOG_PASSWORD missing from .env or .env not loaded correctly."
            )
            # Authentication code from: https://eogdata.mines.edu/products/register/
            params = {
                "client_id": "eogdata_oidc",
                "client_secret": "2677ad81-521b-4869-8480-6d05b9e57d48",
                "username": username,
                "password": password,
                "grant_type": "password",
            }
            token_url = "https://eogauth.mines.edu/auth/realms/master/protocol/openid-connect/token"
            response = requests.post(token_url, data=params)
            access_token_dict = json.loads(response.text)
            access_token = access_token_dict.get("access_token")
            # Submit request with token bearer
            ## Change data_url variable to the file you want to download
            auth = "Bearer " + access_token
            headers = {"Authorization": auth}
            self.auth_headers = headers
        return

    def preprocess_yearly(self, grid: GlobalBaseGrid) -> pd.DataFrame:
        """Aggregates high-resolution NTL data to a yearly 0.5-degree grid.

        This method processes downloaded annual NTL GeoTIFFs (mean, median, and
        lit mask versions) for specified years. It attempts to load previously
        aggregated yearly results from `processing/ntl/ntl_yearly.parquet`,
        processing only missing years or all years if regeneration is forced.

        For each year, it loads the corresponding NTL files, applies the lit area
        mask, filters out negative (likely invalid) NTL values, and then spatially
        aggregates (mean) both the median and mean NTL values to the project's
        standard 0.5-degree grid. The aggregated results are combined in a
        pandas DataFrame indexed by ('pgid', 'year'). Remaining NaNs in the aggregated
        data are filled with 0. The DataFrame is saved as `ntl_yearly.parquet`
        in the processing folder and the global regeneration config is updated.

        Args:
            grid (GlobalBaseGrid): The initialized GlobalBaseGrid instance, used
                to load the grid and create the base yearly data structure.

        Returns:
            pd.DataFrame: A DataFrame indexed by ('pgid', 'year') containing
                'ntl_mean' and 'ntl_median' columns with values aggregated to the
                0.5-degree grid for all processed years.
        """
        assert self.data_loaded, (
            "The NTL data has not been downloaded fully, please rerun load_data()."
        )
        with Progress(console=self.console) as progress:
            years = [y for y in self.years if y not in self.years_unavailable]
            if not self.regenerate["preprocessing"]:
                try:
                    df_ntl = self.storage.load("processing", "ntl_yearly")
                    if len(self.years_missing) == 0:
                        return df_ntl
                    else:
                        years = [y for y in years if y not in df_ntl.index.get_level_values("year")]
                except FileNotFoundError:
                    df_ntl = None
            else:
                df_ntl = None

            task_agg = progress.add_task(
                "[blue]Aggregating NTL...",
                total=len(self.years),
                completed=len(self.years) - len(years),
            )
            # filetype already included in filename
            files = [
                self.storage.build_filepath("processing", f, filetype="") for f in self.ntl_files
            ]

            df_grid = grid.load()
            df_yearly = create_custom_data_structure(
                base_grid=df_grid,
                year_start=self.global_config["start_year"],
                year_end=get_quarter("last").year,
                quarterly=False,
            )
            df_yearly["ntl_mean"] = np.nan
            df_yearly["ntl_median"] = np.nan
            # if there already is some data, use that so we don't need to preprocess again
            if df_ntl is not None:
                df_yearly.update(df_ntl)

            for year in years:
                fp_mask = [f for f in files if f"{year}_global" in f and "lit_mask" in f][0]
                fp_ntl_median = [
                    f for f in files if f"{year}_global" in f and "median_masked" in f
                ][0]
                fp_ntl_mean = [f for f in files if f"{year}_global" in f and "average_masked" in f][
                    0
                ]

                mask = rxr.open_rasterio(fp_mask).squeeze().drop_vars(["band"])  # type: ignore

                ntl_median: xr.DataArray = (
                    rxr.open_rasterio(fp_ntl_median).squeeze().drop_vars(["band"])  # type: ignore
                )  # type: ignore
                ntl_median = ntl_median.where(mask)
                # If I correctly understand, negative values result from very few/no valid observations for a pixel - we ignore those for the aggregation
                ntl_median = ntl_median.where(ntl_median >= 0)

                ntl_mean: xr.DataArray = (
                    rxr.open_rasterio(fp_ntl_mean).squeeze().drop_vars(["band"])  # type: ignore
                )  # type: ignore
                ntl_mean = ntl_mean.where(mask)
                # If I correctly understand, negative values result from very few/no valid observations for a pixel - we ignore those for the aggregation
                ntl_mean = ntl_mean.where(ntl_mean >= 0)

                ntl_median_agg = self._aggregate_to_grid(ntl_median, progress, year, "ntl_median")
                progress.update(task_agg, advance=0.4)
                ntl_mean_agg = self._aggregate_to_grid(ntl_mean, progress, year, "ntl_mean")
                progress.update(task_agg, advance=0.4)
                df_ntl = ntl_median_agg.to_dataframe().reset_index()
                df_ntl["ntl_mean"] = ntl_mean_agg.to_dataframe()["ntl_mean"].to_list()
                df_ntl["pgid"] = df_ntl.apply(lambda x: coords_to_pgid(x.lat, x.lon), axis=1)
                df_ntl["year"] = year
                df_ntl = (
                    df_ntl.set_index(["pgid", "year"]).drop(columns=["lat", "lon"]).sort_index()
                )
                df_ntl = df_ntl.fillna(
                    0
                )  # there should not be any coverage issues within the available range so just assigning 0
                df_yearly.update(df_ntl, overwrite=False)
                progress.update(task_agg, advance=0.2)
            self.storage.save(df_yearly, "processing", "ntl_yearly")
            self.regenerate["preprocessing"] = False
        return df_yearly

    def _aggregate_to_grid(
        self, source_da: xr.DataArray, progress: Progress, year: int, name: str = "ntl"
    ) -> xr.DataArray:
        """Performs the mean aggregation from the NTL raster data to the grid cells for a year.

        Calculates the grid boundaries covering the spatial extent of the input
        `source_da`. Iterates through the boundaries to select the relevant data
        and calculate the mean of all pixel values from `source_da` in the
        selection.

        Adds a task to the `rich.progress` progress instance to track aggregation
        progress, which is removed again after completion.

        Args:
            source_da (xr.DataArray): The input NTL DataArray. Expected to have
                'x' and 'y' coordinates.
            progress (Progress): A `rich.progress.Progress` object for managing and
                displaying nested progress bars.
            year (int): The year of the NTL data, used for labeling the progress task.
            name (str, optional): Name for the resulting data layer, also used for
                progress bar labeling. Defaults to "ntl".

        Returns:
            xr.DataArray: A grid-resolution xarray DataArray with dimensions
                ('lat', 'lon') with the aggregated values. Named according to
                the `name` argument. Coordinates represent the grid cell centers.
        """
        # create 0.5° grid DataArray based on source bounds
        x_min = s_floor(float(source_da.x.min())) + 0.25
        if x_min < -179.75:
            x_min = -179.75
        y_min = s_floor(float(source_da.y.min())) + 0.25
        if y_min < -89.75:
            y_min = -89.75
        x_max = s_ceil(float(source_da.x.max())) - 0.25
        if x_max > 179.75:
            x_max = 179.75
        y_max = s_ceil(float(source_da.y.max())) - 0.25
        if y_max > 89.75:
            y_max = 89.75
        xs = np.arange(float(x_min), x_max + 0.5, 0.5)
        ys = np.arange(float(y_min), y_max + 0.5, 0.5)
        data = np.zeros((len(ys), len(xs)))
        task_agg_yearly = progress.add_task(
            f"[yellow]Aggregate {year} {name} data...", total=len(xs) * len(ys)
        )
        # loop through grid and aggregate via mean
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                data[j, i] = (
                    source_da.sel(x=slice(x - 0.25, x + 0.25), y=slice(y + 0.25, y - 0.25))
                    .mean()
                    .values
                )
                progress.update(task_agg_yearly, advance=1)
        # build new array
        da_grid = xr.DataArray(
            data=data,
            coords={"lat": ys, "lon": xs},
            name=name,
        )
        progress.remove_task(task_agg_yearly)
        return da_grid

    def get_masked_median_data(self, year: int) -> xr.Dataset:
        """Loads masked median NTL data for a specific year as an xarray Dataset.

        Loads the 'median_masked' and 'lit_mask' GeoTIFF files for the given
        `year`, applies the 'lit_mask' to the median NTL data (setting
        values to NaN where the mask is false/zero), and returns the result as
        an xarray Dataset where the primary data variable is named 'ntl'.

        Args:
            year (int): The year for which to retrieve the masked NTL data.

        Returns:
            xr.Dataset: An xarray Dataset containing the masked median NTL data
                for the specified year.

        Raises:
            NTLDataError: If the NTL files for the specified year are not found
                in the processing storage.
        """
        files = [f for f in self.ntl_files if f"_{year}_" in f]
        try:
            assert len(files) == 3  # we have 3 versions in storage for each year
        except AssertionError as e:
            if len(files) == 0:
                raise NTLDataError(f"NTL files for year {year} not available in storage.")
            else:
                print(files)
                raise e
        fp_ntl = self.storage.build_filepath(
            "processing", [f for f in files if "median_masked" in f][0], filetype=""
        )
        fp_mask = self.storage.build_filepath(
            "processing", [f for f in files if "lit_mask" in f][0], filetype=""
        )
        self.console.print("Masking NTL data with lit mask...")
        ntl = rxr.open_rasterio(fp_ntl).squeeze()  # type: ignore
        mask = rxr.open_rasterio(fp_mask).squeeze().drop_vars(["band"])  # type: ignore
        ntl = ntl.where(mask)
        mask.close()
        del mask
        ntl = ntl.to_dataset(name="ntl")
        return ntl
