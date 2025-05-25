from functools import cache
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import Point
from tqdm import tqdm

from base.objects import Dataset
from utils.data_processing import make_iso3_column


class EPRData(Dataset):
    """Handles loading, and preprocessing of Ethnic Power Relations (EPR) data.

    Implements `load_data()` to read core EPR data and spatial GeoEPR data.
    Implements `preprocess_data()` to standardize country codes, reshapes time-period
    data into a yearly format, and merge the EPR data with GeoEPR geometries.
    Implements `calculate_excluded_groups`, which determines the number of
    politically excluded ethnic groups in the vicinity of a grid cell.

    Attributes:
        data_key (str): Set to "epr".
        data_keys (list[str]): Specifies required config keys: "epr" and "geoepr".
        # needs_storage (bool): Consider setting to False if no intermediate
        # results are cached by this class itself to its processing subfolder.
    """

    data_key: str = "epr"
    data_keys: list[str] = ["epr", "geoepr"]

    def load_data(self) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
        """Loads the core EPR and GeoEPR datasets.

        Reads the EPR and GeoEPR data, and filters both datasets to the timeframe
        from `self.global_config['start_year']` onwards. Performs basic geometry
        cleaning on the GeoEPR data and returns both DataFrames.

        Returns:
            tuple[pd.DataFrame, gpd.GeoDataFrame]: A tuple containing:
                - df_epr: DataFrame with raw EPR data
                - gdf_epr: GeoDataFrame with raw GeoEPR data.
        """
        # EPR
        df_epr = pd.read_csv(self.data_config["epr"])
        # only relevante time periods
        df_epr = df_epr[df_epr.to >= self.global_config["start_year"]]
        # GeoEPR
        gdf_epr = gpd.read_file(self.data_config["geoepr"])
        gdf_epr = gdf_epr[gdf_epr.to >= self.global_config["start_year"]]
        # clean geometries - basic checks were done to ensure this doesn't change them
        gdf_epr.loc[~gdf_epr.is_valid, "geometry"] = gdf_epr.loc[
            ~gdf_epr.is_valid, "geometry"
        ].buffer(0)
        return df_epr, gdf_epr

    def preprocess_data(self, df_epr: pd.DataFrame, gdf_epr: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Preprocesses and merges EPR and GeoEPR data into a country panel.

        Creates standardized country codes, reshapes data from a period-based format
        (with 'from' and 'to' years) to a yearly format, and merges EPR and
        GeoEPR DataFrames. Does not set the standard index, as it would not be unique
        at this point, which is exploited by `calculate_excluded_groups()`.

        Args:
            df_epr (pd.DataFrame): The raw EPR data from `load_data()`.
            gdf_epr (gpd.GeoDataFrame): The raw GeoEPR data from `load_data()`.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing EPR data merged with
                GeoEPR spatial information for each year, country, and ethnic group.
        """
        # countries
        df_epr["iso3"] = make_iso3_column(df_epr, "statename")
        # fix "not found" iso code assignment
        df_epr.loc[df_epr.statename == "German Federal Republic", "iso3"] = "DEU"
        df_epr.loc[df_epr.statename == "Vietnam, Democratic Republic of", "iso3"] = "VNM"
        df_epr["year"] = df_epr.apply(lambda x: np.arange(x["from"], x["to"] + 1), axis=1)
        # reshape to yearly data
        df_epr = df_epr.explode("year")
        df_epr = df_epr.set_index(["iso3", "year"]).drop(columns=["to", "from"])

        # reshape to yearly data
        gdf_epr["year"] = gdf_epr.apply(lambda x: np.arange(x["from"], x["to"] + 1), axis=1)
        gdf_epr = gdf_epr.explode("year")

        df = pd.merge(
            df_epr.reset_index(),
            gdf_epr[["gwgroupid", "sqkm", "type", "geometry", "year"]],
            on=["gwgroupid", "year"],
            how="left",
        )
        gdf = gpd.GeoDataFrame(df, geometry=df.geometry, crs="epsg:4326")
        return gdf

    def calculate_excluded_groups(
        self, df_base: pd.DataFrame, gdf_epr: gpd.GeoDataFrame, buffer_size: float
    ) -> pd.DataFrame:
        """Calculates the number of politically excluded ethnic groups near each grid cell.

        For each grid cell and time point in `df_base`, this method identifies
        the number of politically excluded groups present in a circular area
        around the grid cell's center. The core calculation for each cell-year
        (`epr_excluded_groups` helper) is cached to avoid redundant computations.

        Args:
            df_base (pd.DataFrame): A standardized indicator DataFrame, indexed by
                ('pgid', 'year', 'quarter'), and containing 'lat' and 'lon'
                columns representing grid cell centers.
            gdf_epr (gpd.GeoDataFrame): The preprocessed and merged GeoEPR data,
                as returned by `self.preprocess_data()`.

        Returns:
            pd.DataFrame: The input `df_base` DataFrame augmented with the new
                column 'excluded_number'.
        """

        @cache
        def epr_excluded_groups(year: int, lat: float, lon: float) -> float | int:
            """
            Calculates number of excluded groups near a single grid cell based on a
            circle around the grid cell center. Using cache since every year appears
            up to 4 times in index structure but data is only yearly.
            """
            if year > gdf_epr.index.get_level_values("year").max():
                excluded_number = np.nan
            else:
                cell_center = Point(lon, lat)
                buffer = cell_center.buffer(buffer_size)
                shapes = gdf_epr.xs(year, level="year").clip(buffer, keep_geom_type=True)  # type: ignore
                if len(shapes) == 0:
                    excluded_number = 0
                else:
                    excluded = shapes[shapes.status.isin(["POWERLESS", "DISCRIMINATED"])]
                    excluded_number = len(excluded)

            return excluded_number

        df = df_base.copy()
        # set index for further processing - this is not a unique index which we exploit for data selection in the for loop below
        gdf_epr = gdf_epr.set_index(["iso3", "year"]).sort_index()
        # shapefile-based variables
        tqdm.pandas()
        self.console.print("Calculating politically excluded groups in the vicinity...")
        df["excluded_number"] = (
            df.reset_index()
            .progress_apply(lambda x: epr_excluded_groups(x.year, x.lat, x.lon), axis=1)  # type: ignore
            .to_list()
        )

        return df
