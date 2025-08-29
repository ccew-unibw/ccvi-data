import ast
import os
import warnings
import geopandas as gpd
import numpy as np
import pandas as pd

from base.objects import Dataset, ConfigParser, GlobalBaseGrid
from utils.spatial_operations import assign_areas_to_grid


class AqueductData(Dataset):
    """Handles loading, and preprocessing of Aqueduct water stress data.

    Implements `load_data()` to read baseline and future projected data.
    Implements `preprocess_data_waterstress()` to clean, merge, and standardize
    the baseline and future data based on hydrological basins.
    Implements `match_aqueduct_grid()` to create a spatial lookup between the
    hydrological basins and the project's grid cells, caching the result.
    Implements `calculate_grid_values()` to compute the final water stress values
    for each grid cell for specific years (by default 2019 and 2030), creating
    overlap area weighted means for cells that overlap multiple basins.

    Attributes:
        data_key (str): Set to "aqueduct".
    """

    data_key: str = "aqueduct"

    def load_data(self) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
        """Loads the Aqueduct baseline and future projections.

        Loads the baseline and future data layers and selects relevant
        water stress columns for debugging and merging. Ignores poloygon-related
        RuntimeWarnings during reading the file.

        Returns:
            tuple[pd.DataFrame, gpd.GeoDataFrame]: Tuple of (Geo)DataFrames
                with Aqueduct baseline and future projections.
        """
        # ignore pygrio RuntimeWarnings from reading the geofiles:
        # "organizePolygons() received a polygon with more than 100 parts..."
        with warnings.catch_warnings(category=RuntimeWarning):
            warnings.simplefilter("ignore")
            # baseline
            df_base = gpd.read_file(self.data_config[self.data_key], layer="baseline_annual")
            # future
            df_fut = gpd.read_file(self.data_config[self.data_key], layer="future_annual")

        df_base = df_base[["pfaf_id"] + [c for c in df_base.columns if c.startswith("bws_")]]
        df_fut = df_fut[["pfaf_id", "geometry"] + [c for c in df_fut.columns if "bau30_ws_" in c]]
        return df_base, df_fut

    def preprocess_data_waterstress(
        self, df_base: pd.DataFrame, df_fut: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Preprocesses and merges the baseline and future Aqueduct data.

        This method cleans no-data values, clips data to valid ranges,
        standardizes column names between the two datasets, and merges them into
        a single GeoDataFrame indexed by the hydrological basin ID (`pfaf_id`),
        with distinct columns for 2019 and 2030 water stress.

        Args:
            df_base (pd.DataFrame): The raw baseline data from `load_data()`.
            df_fut (gpd.GeoDataFrame): The raw future projection data from `load_data()`.

        Returns:
            gpd.GeoDataFrame: A unified GeoDataFrame with raw version water stress
                values for different years, indexed by `pfaf_id`.
        """
        df_base = df_base.replace(-9999, np.nan)
        df_base["bws_raw"] = df_base["bws_raw"].clip(upper=1)
        df_fut["bau30_ws_x_r"] = df_fut["bau30_ws_x_r"].clip(upper=1)
        # match colum names
        df_fut.columns = df_fut.columns.str.replace("bau30_ws_x", "bws").str.replace("_r", "_raw")
        # since the baseline is matched to gadm but actually on the hydrological basin level,
        # we can match back to the basin geometries the future comes in
        # assert check to make sure the values per pfaf_id are all the same or nan
        assert df_base.groupby("pfaf_id")["bws_raw"].nunique().le(1).all()
        # we only need bws_raw for the indicator
        df_base = df_base.groupby("pfaf_id")["bws_raw"].mean()  # type: ignore
        df_fut = df_fut[["pfaf_id", "geometry", "bws_raw"]]
        gdf = pd.merge(df_fut, df_base, on="pfaf_id", how="left", suffixes=["_2030", "_2019"])
        gdf = gdf.set_index("pfaf_id").sort_index()
        return gdf  # type: ignore

    def match_aqueduct_grid(self, gdf_aqueduct: gpd.GeoDataFrame, grid: GlobalBaseGrid):
        """Creates and caches a lookup table mapping grid cells to Aqueduct basins.

        This method generates a spatial mapping between the grid cell IDs
        (`pgid`) and the Aqueduct hydrological basin IDs (`pfaf_id`). If
        regeneration is not forced, checks for a cached version of the lookup
        file in the processing storage and loads this, otherwise performs the
        spatial join to determine the overlap and saves the result for future use.

        Args:
            gdf_aqueduct (gpd.GeoDataFrame): The preprocessed Aqueduct data with
                basin geometries, indexed by `pfaf_id`.
            grid (GlobalBaseGrid): An initialized GlobalBaseGrid instance.

        Returns:
            pd.DataFrame: A DataFrame indexed by `pgid` containing the
                corresponding `pfaf_id`(s) for each grid cell. For cells
                overlapping multiple basins, this is a dictionary of pfaf_id: area_weight.
        """
        fp = self.storage.build_filepath("processing", "pgid_pfafid_lookup.csv", filetype="")
        if os.path.exists(fp) and not self.regenerate["preprocessing"]:
            df = pd.read_csv(fp, index_col="pgid", converters={"pfaf_id": ast.literal_eval})
            self.console.print("Existing Aqueduct-Grid matching lookup file loaded from storage.")
        else:
            self.console.print("Generating Aqueduct-Grid matching lookup file...")
            gdf_grid = grid.load(return_gdf=True)
            gdf_grid = assign_areas_to_grid(gdf_grid, gdf_aqueduct, "pfaf_id", "pfaf_id")
            df: pd.DataFrame = gdf_grid[["pfaf_id"]]
            df.to_csv(fp)
            self.config.set_regenerated_globally(self.data_key, "preprocessing")
        return df

    def calculate_grid_values(
        self,
        gdf: gpd.GeoDataFrame,
        grid: GlobalBaseGrid,
        df_matching: pd.DataFrame,
        years: tuple[int, int] = (2019, 2030),
    ):
        """Calculates the water stress values for each grid cell for specified years.

        Using the `pgid`-to-`pfaf_id` lookup table from `match_aqueduct_grid()`,
        this method assigns water stress values to each grid cell for each year.
        For cells that overlap with multiple hydrological basins, it calculates
        an overlap area weighted average of the water stress values within the cell.

        Args:
            gdf_aqueduct (gpd.GeoDataFrame): The preprocessed Aqueduct data,
                indexed by `pfaf_id`, containing water stress columns for different years.
            grid (GlobalBaseGrid): An initialized GlobalBaseGrid instance.
            df_matching (pd.DataFrame): The lookup table DataFrame from
                `match_aqueduct_grid`, indexed by `pgid`.
            years (tuple[int, int], optional): The years for which to calculate
                grid values, not hardcoded in case of data updates. Defaults to (2019, 2030).

        Returns:
            pd.DataFrame: A DataFrame indexed by ('pgid', 'year')
                containing the calculated 'bws_raw' water stress values.
        """

        def get_val(pgid: int) -> float:
            """Reads/calculates value based on column col for a given pgid."""
            if pgid not in df_matching.index:
                return np.nan
            # here is the actual matching
            else:
                match = df_matching.loc[pgid, "pfaf_id"]
                col = f"bws_raw_{year}"
                if type(match) is dict:
                    vals: pd.Series = gdf.loc[match.keys(), col]  # type: ignore
                    val = np.average(vals.values.astype(float), weights=list(match.values()))
                else:
                    val = gdf.loc[match, col]  # type: ignore
                return float(val)  # type: ignore

        dfs = []
        for year in years:
            df = grid.load()
            df["year"] = year
            df["bws_raw"] = df.reset_index()["pgid"].apply(get_val).to_list()
            dfs.append(df)
        df = pd.concat(dfs)
        df = df.set_index("year", append=True).sort_index()
        return df


if __name__ == "__main__":
    config = ConfigParser()
    grid = GlobalBaseGrid(config)
    aqueduct = AqueductData(config)
    dfs = aqueduct.load_data()
    gdf = aqueduct.preprocess_data_waterstress(*dfs)
    df_matching = aqueduct.match_aqueduct_grid(gdf, grid)
    df = aqueduct.calculate_grid_values(gdf, grid, df_matching)
