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

    Implements `load_data()` to read core EPR data.
    Implements `preprocess_data()` to standardize country codes, and reshapes 
    time-period data into a yearly format.

    Attributes:
        data_key (str): Set to "epr".
    """

    data_key: str = "epr"

    def load_data(self) -> pd.DataFrame:
        """Loads the core EPR and GeoEPR datasets.

        Reads the EPR data and filters the datasets to the timeframe from 
        `self.global_config['start_year']` onwards.
        
        Returns:
            pd.DataFrame: DataFrame with raw EPR data
        """
        # EPR
        df_epr = pd.read_csv(self.data_config["epr"])
        # only relevant time periods
        df_epr = df_epr[df_epr.to >= self.global_config["start_year"]]
        return df_epr

    def preprocess_data(self, df_epr: pd.DataFrame) -> pd.DataFrame:
        """Preprocesses EPR data, reshaping it into a country panel.

        Creates standardized country codes, and reshapes data from a period-based 
        format (with 'from' and 'to' years) to a yearly format. Does not set the
        standard ['iso3', 'year'] index, as it would not be unique  at this 
        point.

        Args:
            df_epr (pd.DataFrame): The raw EPR data from `load_data()`.

        Returns:
            pd.DataFrame: A DataFrame containing EPR data for each year, 
                country, and ethnic group.
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
        return df_epr