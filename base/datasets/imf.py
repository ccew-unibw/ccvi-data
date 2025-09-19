import pandas as pd
import sdmx

from base.objects import ConfigParser, Dataset
from utils.data_processing import make_iso3_column


class IMFGDPData(Dataset):
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

    def __init__(self, config: ConfigParser):
        super().__init__(config)
        self.client = sdmx.Client("IMF_DATA")

    def load_data(self) -> pd.DataFrame:
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
        # resource_id: Data provider, Dataset
        # key: All countries (wildcard), GDP PPP, Annual
        data = self.client.get(
            "data",
            resource_id="IMF.RES,WEO",
            key=".PPPGDP.A",
            params={"startPeriod": self.global_config["start_year"]},
        )
        df_imf = sdmx.to_pandas(data)
        return df_imf

    def preprocess_data(self, df_imf: pd.DataFrame) -> pd.DataFrame:
        """Preprocesses the downloaded IMF data.

        This method filters the data to only countries using IMF's country API,
        dropping aggregates. It standardizes the ISO3 code to the ones used in
        the index and adjust indicator values to absolute values.

        Args:
            df_imf (pd.DataFrame): The DataFrame from `load_data()`.

        Returns:
            pd.DataFrame: Preprocessed IMF data indexed by ('iso3', 'year').
        """
        df = df_imf.reset_index()
        # read country list from api - agency_id is needed to overwrite the default "all"
        data = self.client.get("codelist", resource_id="CL_ITS_COUNTRY", agency_id="IMF.RES")
        countries = sdmx.to_pandas(data)["codelist"]["CL_ITS_COUNTRY"].reset_index()
        countries.columns = ["COUNTRY", "COUNTRY_NAME"]
        df = df.merge(countries, on="COUNTRY", how="left")
        # probably only changes Kosovo, could do this manually, but this makes sure others are caught as well
        df["iso3"] = make_iso3_column(df, "COUNTRY_NAME")
        # some cleaning
        df["iso3"] = df["iso3"].apply(lambda x: x[0] if type(x) is list else x)
        df = df.rename(columns={"TIME_PERIOD": "year", "value": "gdp_ppp"})
        df = df.set_index(["iso3", "year"]).sort_index()[["gdp_ppp"]]
        return df
