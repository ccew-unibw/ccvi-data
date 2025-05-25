import json
import requests

import pandas as pd

from base.objects import Dataset
from utils.data_processing import make_iso3_column


class IMFData(Dataset):
    """Handles downloading and preprocessing of International Monetary Fund (IMF) data.

    Implements `load_data()` to download specified yearly economic indicators
    from the IMF DataMapper API.
    Implements `preprocess_data()` to filter data to countries, matches the
    format with the standard country-year panel, and apply numerical scaling to
    adjust data values.

    Attributes:
        data_key (str): Set to "imf".
        local (bool): Set to False, as data is sourced from the IMF API.
        needs_storage (bool): Set to False.
    """

    data_key: str = "imf"
    local: bool = False
    needs_storage: bool = False

    def load_data(self, indicators: dict[str, str]) -> pd.DataFrame:
        """Downloads yearly data for multiple indicators from the IMF DataMapper API.

        For each indicator code provided, it constructs the API URL, fetches the
        data, parses the JSON response, and reshapes the values into a pandas Series
        indexed by ('year', 'iso3'). All Series are then concatenated into a single
        DataFrame. The 'iso3' codes used here are still IMF's country codes.

        Args:
            indicators (dict[str, str]): A dictionary mapping IMF indicator
                codes to desired column names for these indicators in the output
                DataFrame.

        Returns:
            pd.DataFrame: A DataFrame indexed by ('year', 'iso3') containing
                all specified indicators.
        """
        s_list = []
        # get IMF data from API
        for i in indicators:
            url = f"https://www.imf.org/external/datamapper/api/v1/{i}"
            response = requests.get(url)
            data = json.loads(response.text)["values"][i]

            series = pd.DataFrame(data).stack()
            series.name = indicators[i]
            series.index.names = ["year", "iso3"]
            s_list.append(series.sort_index())
        df_imf = pd.concat(s_list, axis=1)
        return df_imf

    def preprocess_data(self, df_imf: pd.DataFrame, scaling_factor: int) -> pd.DataFrame:
        """Preprocesses the downloaded IMF data.

        This method filters the data to only countries using IMF's country API,
        dropping aggregates. It standardizes the ISO3 code to the ones used in
        the index and adjust indicator values to absolute values.

        Args:
            df_imf (pd.DataFrame): The DataFrame from `load_data()`.
            scaling_factor (int): Factor by which to multiply the numerical
                indicator values.

        Returns:
            pd.DataFrame: Preprocessed IMF data indexed by ('iso3', 'year').
        """
        # get country list to filter region and for easy iso3 conversion
        url = "https://www.imf.org/external/datamapper/api/v1/countries"
        response = requests.get(url)
        response.raise_for_status()
        country_df = pd.DataFrame(json.loads(response.text)["countries"]).T
        # probably only changes Kosovo, could do this manually, but this makes sure others are caught as well
        country_df["iso3"] = make_iso3_column(country_df, "label")

        df_imf = df_imf.loc[
            list(df_imf.reset_index().iso3.isin(list(country_df.index)))
        ].reset_index()
        df_imf["iso3"] = df_imf.iso3.apply(lambda x: country_df.loc[x, "iso3"])
        df_imf["year"] = df_imf["year"].astype(int)
        df_imf = df_imf.set_index(["iso3", "year"]).sort_index()
        df_imf = df_imf * scaling_factor
        return df_imf
