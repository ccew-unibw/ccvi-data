import io
import requests

import pandas as pd

from base.objects import Dataset
from utils.data_processing import make_iso3_column
from utils.index import get_quarter


class SDGData(Dataset):
    """Handles downloading and preprocessing of UN SDG indicator data.

    Implements `load_data()` to download yearly data for a list of specified
    SDG series codes from the UNStats SDG API.
    Implements `preprocess_data()` to clean the downloaded data and create a
    standardized country-year panel.

    Attributes:
        data_key (str): Set to "sdg".
        local (bool): Set to False, as data is sourced from the UNStats API.
        needs_storage (bool): Set to False.
    """

    data_key: str = "sdg"
    local: bool = False
    needs_storage: bool = False

    def load_data(self, indicators: list[str]) -> pd.DataFrame:
        """Downloads specified yearly SDG indicator data from the UNStats API.

        Fetches data for specified indicators starting from the year 2000. It
        selects relevant columns and ensures data are comatible for concatenation,
        before combining everything in one DataFrame.

        Args:
            indicators (list[str]): A list of SDG indicator codes to download.

        Returns:
            pd.DataFrame: A concatenated DataFrame containing the downloaded SDG
                indicators.
        """
        dfs = []
        url = "https://unstats.un.org/SDGAPI/v1/sdg/Series/DataCSV"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/octet-stream",
        }
        for series in indicators:
            data = {"seriesCodes": series, "timePeriodStart": "2000"}
            response = requests.post(url, headers=headers, data=data)
            if response.ok:
                df_series = pd.read_csv(io.StringIO(response.content.decode()))
                df_series = df_series.dropna(how="all")
                if not df_series["[Sex]"].isna().all():
                    df_series = df_series.loc[df_series["[Sex]"] == "BOTHSEX"]
                dfs.append(df_series[["SeriesCode", "GeoAreaName", "TimePeriod", "Value"]])
            else:
                raise Exception(f"Request failed with status code: {response.status_code}.")

        df = pd.concat(dfs)
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocesses the downloaded UN SDG data into a standardized country-year panel.

        Standardizes column names, adds iso3 codes, and removes regional
        aggregates and entries with non-standard ISO3 codes. The data is then
        reshaped to have SDG indicator values in columns merged to a standardized
        data structure covering all years and iso3s.

        Args:
            df (pd.DataFrame): The raw SDG data DataFrame from `load_data()`.

        Returns:
            pd.DataFrame: Preprocessed SDG data indexed by ('iso3', 'year').
        """
        df = df.rename(columns={"GeoAreaName": "country", "TimePeriod": "year"})
        df["year"] = df.year.astype(int)
        df["iso3"] = make_iso3_column(df, "country")
        # cleanup - remove grouped estimates
        df = df[df.iso3 != "not found"]
        df = df[[type(iso3) is str for iso3 in df.iso3]]
        df = df[df.country != "Southern Africa"]
        # to be safe create a data structure with all products of iso3 and years relevant
        df_out = pd.DataFrame(data=df.reset_index().iso3.unique(), columns=["iso3"])
        df_out["year"] = [
            list(range(2000, get_quarter(which="last").year + 1)) for i in range(len(df_out))
        ]
        df_out = df_out.explode("year")
        df_out = df_out.set_index(["iso3", "year"]).sort_index()
        # reshape df and merge to df_out
        df = df.set_index(["iso3", "year", "SeriesCode"]).drop(columns="country")
        df = df.unstack(level="SeriesCode").droplevel(0, axis="columns")  # type: ignore
        df_out = df_out.merge(df, how="left", left_index=True, right_index=True)
        for col in df_out.columns:
            if df_out[col].dtype == object:
                # there can be string dtypes due to <x values returned - assign them their max value
                df_out[col] = df_out[col].str.replace("<", "").astype(float)
        return df_out
