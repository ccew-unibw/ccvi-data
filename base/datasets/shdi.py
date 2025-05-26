import ast
import os

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm import tqdm

from base.objects import ConfigParser, Dataset, GlobalBaseGrid
from utils.data_processing import default_impute
from utils.spatial_operations import assign_areas_to_grid


class SubnationalHDIData(Dataset):
    """Handles loading, preprocessing, and grid-matching of subnational HDI data.

    Implements `load_data()` to read and filter Subnational Human Development Index
    (SHDI) data from a CSV.
    Implements `preprocess_data()` to orchestrate the matching of SHDI regions
    to the grid cells and merge the SHDI values, using area-weighted
    averages for cells overlapping multiple regions, and imputation of quarterly
    values.

    Attributes:
        data_key (str): Set to "shdi".
        data_keys (list[str]): Specifies required input data keys: "shdi"
            (for SHDI data CSV) and "shdi_shapes" (for GDL admin1 shapes).
    """

    data_key: str = "shdi"

    def __init__(self, config: ConfigParser):
        """Initializes the SubnationalHDIData instance.

        Sets `self.data_keys` to include both "shdi" (for the data) and
        "shdi_shapes" (for the GDL administrative boundaries). Then calls the
        parent `Dataset` initializer.

        Args:
            config (ConfigParser): An initialized ConfigParser instance.
        """
        self.data_keys = [self.data_key, "shdi_shapes"]
        super().__init__(config)

    def load_data(self, shdi_cols: list[str]) -> pd.DataFrame:
        """Loads and performs initial filtering on the Subnational HDI data CSV.

        Reads the SHDI CSV specified by `self.data_config["shdi"]`, filters data
        to include years from five years before `self.global_config['start_year']`
        onwards, selects only subnational level data, and keeps only the
        specified `shdi_cols`.

        Args:
            shdi_cols (list[str]): A list of column names to select from the SHDI
                dataset.

        Returns:
            pd.DataFrame: The loaded and initially filtered SHDI data, indexed by
                ('gdlcode', 'year').
        """
        df_shdi = pd.read_csv(self.data_config[self.data_key], low_memory=False, na_values=" ")
        df_shdi = df_shdi.rename(columns={"GDLCODE": "gdlcode"})
        df_shdi = df_shdi[df_shdi.year >= self.global_config["start_year"] - 5]
        df_shdi = df_shdi[df_shdi.level == "Subnat"]
        df_shdi = df_shdi.set_index(["gdlcode", "year"])[shdi_cols]
        return df_shdi.sort_index()

    def preprocess_data(
        self, df_shdi: pd.DataFrame, df_base: pd.DataFrame, grid: GlobalBaseGrid
    ) -> pd.DataFrame:
        """Preprocesses SHDI data by matching it to the project grid and merging.

        This method orchestrates two main steps:
        1. `_hdi_grid_match()`: Creates or loads a lookup table matching grid cell
           IDs ('pgid') to GDL region codes ('gdlcode').
        2. `_hdi_merge()`: Merges the yearly SHDI data (`df_shdi`) onto the
           quarterly base DataFrame (`df_base`) using the lookup table, assigning
           SHDI values only to the fourth quarter of each year. Handles grid cells
           that span multiple SHDI regions by area-weighted averaging.
           Missing values after merging are linearly imputed.

        Args:
            df_shdi (pd.DataFrame): The DataFrame from `load_data()`.
            df_base (pd.DataFrame): The standardized indicator base DataFrame
                indexed by ('pgid', 'year', 'quarter').
            grid (GlobalBaseGrid): The initialized GlobalBaseGrid instance, used
                by `_hdi_grid_match()` to load grid geometries.

        Returns:
            pd.DataFrame: A DataFrame with the same index as `df_base`, containing
                the merged and imputed SHDI columns. Original columns from `df_base`
                (like 'iso3') are dropped.
        """
        df_matching = self._hdi_grid_match(grid)
        df = self._hdi_merge(df_base, df_shdi, df_matching)
        df = df.drop(columns=df_base.columns)
        return df

    def _hdi_grid_match(self, grid: GlobalBaseGrid) -> pd.DataFrame:
        """Creates or loads a lookup table matching grid cells (pgid) to GDL codes.

        If a cached lookup file (`pgid_gdl_lookup.csv`) exists and preprocessing
        regeneration is not forced, it's loaded and returned.

        Otherwise, it loads the grid with geometries and assigns GDL region codes
        ('gdlcode') from the 'shdi_shapes' file (admin level 1 geometries) to
        each grid cell using the `assign_areas_to_grid` utility. In cases where a
        grid cell covers multiple GDL regions, a dictionary of GDL codes and
        their respective overlap areas is used. The resulting lookup table
        (pgid to gdlcode/gdlcode_dict) is stored as CSV file and the global
        regeneration config is updated.

        Args:
            grid (GlobalBaseGrid): The initialized GlobalBaseGrid instance.

        Returns:
            pd.DataFrame: A DataFrame indexed by 'pgid', with a 'gdlcode' column
                that contains either a single GDL code string or a dictionary
                of GDL codes and their overlap areas for cells spanning multiple
                regions.
        """
        fp = self.storage.build_filepath("processing", "pgid_gdl_lookup.csv")
        if os.path.exists(fp) and not self.regenerate["preprocessing"]:
            df = pd.read_csv(fp)
            df.gdlcode = df.gdlcode.apply(lambda x: ast.literal_eval(x) if x[0] == "{" else x)
        else:
            self.console.print("Generating SHDI-Grid matching lookup file...")
            gdf_grid = grid.load(return_gdf=True)
            gdf_grid = assign_areas_to_grid(
                gdf_grid, self.data_config["shdi_shapes"], "gdlcode", "gdlcode"
            )
            df = gdf_grid.drop(columns=["geometry", "iso3"])
            df.to_csv(fp)
            self.config.set_regenerated_globally(self.data_key, "preprocessing")
        return df

    def _hdi_merge(
        self, df_base: pd.DataFrame, df_shdi: pd.DataFrame, df_matching: pd.DataFrame
    ) -> pd.DataFrame:
        """Merges yearly SHDI data onto the quarterly base grid structure.

        For each column in `df_shdi`, it assigns values to the `df_base` DataFrame.
        The annual SHDI values are assigned only to the fourth quarter of each
        year in `df_base`. The matching between grid cells (`pgid` from
        `df_base`) and SHDI regions (`gdlcode` from `df_shdi`) is done using
        the `df_matching` lookup table.
        - If a grid cell maps to a single GDL code, the corresponding SHDI value is used.
        - If a grid cell maps to multiple GDL codes, an area-weighted average of
          the SHDI values from those overlapping GDL regions is calculated.
        After merging, `default_impute` is applied to linarly impute the values
        of the other quarters in the year.

        Args:
            df_base (pd.DataFrame): The indicator base DataFrame, indexed by
                ('pgid', 'year', 'quarter').
            df_shdi (pd.DataFrame): The Subnational HDI data, indexed by
                ('gdlcode', 'year'), with SHDI indicators as columns.
            df_matching (pd.DataFrame): The lookup table from `_hdi_grid_match`,
                indexed by 'pgid', with a 'gdlcode' column.

        Returns:
            pd.DataFrame: The `df_base` DataFrame with the merged and imputed
                SHDI columns added.
        """

        def get_shdi_val(pgid: int, year: int, quarter: int, col: str) -> float:
            """Reads/calculates shdi value based on column col for a given year,
            quarter and pgid.
            """
            # all outside of shdi data
            if year > df_shdi.index.get_level_values("year").max():
                return np.nan
            elif pgid not in df_matching.index:
                return np.nan
            elif quarter != 4:  # only assign to end of year
                return np.nan
            # here is the actual matching
            else:
                try:
                    match = df_matching.loc[pgid, "gdlcode"]
                    if type(match) is dict:
                        temp = pd.DataFrame.from_dict(match, orient="index")
                        vals = df_shdi.loc[(temp.index, year), col]
                        val = np.average(vals.values.astype(float), weights=temp.values.flatten())
                    else:
                        val = df_shdi.loc[(match, year), col]
                    return float(val)  # type: ignore
                except Exception:
                    return np.nan

        def parallel_col(col):
            data = list(
                df_base.reset_index().apply(
                    lambda x: get_shdi_val(x.pgid, x.year, x.quarter, col), axis=1
                )
            )
            series = pd.Series(index=df_base.index, data=data, name=col).to_frame()
            return series

        df = df_base.copy()
        if len(df_shdi.columns) > 1:  # do it in parallel only if we have more than one
            dfs = Parallel(n_jobs=len(df_shdi.columns), verbose=1)(
                delayed(parallel_col)(col) for col in df_shdi.columns
            )
            df_processed = pd.concat(dfs, axis=1)
            df = pd.concat([df, df_processed], axis=1)

        else:  # in case of one we can do an individual progress bar
            tqdm.pandas()
            for col in df_shdi.columns:
                df[col] = list(
                    df_base.reset_index().progress_apply(
                        lambda x: get_shdi_val(x.pgid, x.year, x.quarter, col), axis=1
                    )  # type: ignore
                )  # type: ignore
        for col in df_shdi.columns:
            df[col] = default_impute(df[col])
        return df
