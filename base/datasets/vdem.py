from base.objects import Dataset

import pandas as pd
import pyreadr


class VDemData(Dataset):
    """Handles loading and preprocessing of Varieties of Democracy (V-Dem) data.

    Implements `load_data()` to read V-Dem data from an R data file (.rds),
    selecting specified variables.
    Implements `preprocess_data()` for data cleaning, including filtering by year,
    specific country/territory exclusions, and standardizing the data structure
    to a country-year panel.

    Attributes:
        data_key (str): Set to "vdem".
        needs_storage (bool): Set to False.
    """

    data_key: str = "vdem"
    needs_storage: bool = False

    def load_data(self, variables: list[str] | str):
        """Loads specified variables from the V-Dem data file.

        Reads the V-Dem .rds file (path from `self.data_config[self.data_key]`).
        It extracts the 'country_text_id', 'year', and the variables requested
        in the `variables` argument.

        Args:
            variables (list[str] | str): A single variable name or a list
                of variable names to load from the V-Dem dataset.

        Returns:
            pd.DataFrame: A DataFrame containing country, year, and the
                specified V-Dem variables.
        """
        if isinstance(variables, str):
            variables = [variables]
        r_frame = pyreadr.read_r(self.data_config[self.data_key])
        df_vdem = r_frame[None]
        df_vdem = df_vdem[["country_text_id", "year"] + variables]
        return df_vdem

    def preprocess_data(self, df_vdem: pd.DataFrame):
        """Preprocesses and standardizes the raw V-Dem data.

        Filters out specific entries and retains data from the configured `start_year`
        onwards. Renames everything to the standard country-year panel

        Args:
            df_vdem (pd.DataFrame): The raw V-Dem DataFrame from `load_data()`.

        Returns:
            pd.DataFrame: The preprocessed V-Dem data, indexed by ('iso3', 'year').
        """
        # 2 different Palestine areas in vdem with same iso code processed, but gaza does not appear in the grid (too small)
        df_vdem = df_vdem[~df_vdem["country_text_id"].isin(["Palestine/Gaza"])]
        df_vdem = df_vdem[df_vdem.year >= self.global_config["start_year"]].copy()
        # prep for merging to grid
        df_vdem["year"] = df_vdem["year"].astype(int)
        df_vdem = (
            df_vdem.rename(columns={"country_text_id": "iso3"})
            .set_index(["iso3", "year"])
            .sort_index()
        )
        return df_vdem
