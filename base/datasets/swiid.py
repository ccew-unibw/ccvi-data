from base.objects import Dataset

import pandas as pd

from utils.data_processing import make_iso3_column


class SWIIDData(Dataset):
    """Handles loading and preprocessing of Standardized World Income Inequality Database (SWIID) data.

    Implements `load_data()` to read SWIID data.
    Implements `preprocess_data()` to standardize the data format and adjustment the scale.

    Attributes:
        data_key (str): Set to "swiid".
        needs_storage (bool): Set to False.
    """

    data_key: str = "swiid"
    needs_storage: bool = False

    def load_data(self):
        """Loads SWIID data from the configured CSV file.

        Returns:
            pd.DataFrame: The raw SWIID data loaded from the CSV.
        """
        return pd.read_csv(self.data_config[self.data_key])

    def preprocess_data(self, df_swiid: pd.DataFrame):
        """Preprocesses the raw SWIID data.

        Converts country names to iso3, sets a MultiIndex ('iso3', 'year') and adjusts gini
        scale.

        Args:
            df_swiid (pd.DataFrame): The DataFrame loaded by `load_data()`.

        Returns:
            pd.DataFrame: The preprocessed DataFrame, indexed by ('iso3', 'year').
        """
        df_swiid["iso3"] = make_iso3_column(df_swiid, "country")
        df_swiid = df_swiid.set_index(["iso3", "year"]).sort_index().drop(columns=["country"])
        df_swiid = df_swiid.drop(index="not found")
        df_swiid["gini_disp"] = df_swiid["gini_disp"] / 100  # adjust scale
        return df_swiid[["gini_disp"]]
