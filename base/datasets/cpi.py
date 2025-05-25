# suppress clean-text import log message about missing unidecode GPL package
import logging

logger = logging.getLogger()
original_level = logger.level
logger.setLevel(logging.ERROR)
from cleantext import clean

logger.setLevel(original_level)
import pandas as pd

from base.objects import Dataset


class CPIData(Dataset):
    """Handles loading and preprocessing of Transparency International's CPI data.

    Implements `load_data()` to read Corruption Perceptions Index (CPI) scores
    from an Excel file.
    Implements `preprocess_data()` to reshape the data from wide to long format,
    clean column names, and standardize country codes.

    Attributes:
        data_key (str): Set to "cpi".
        needs_storage (bool): Set to False.
    """

    data_key: str = "cpi"
    needs_storage: bool = False

    def load_data(self) -> pd.DataFrame:
        """Loads CPI data from the specified Excel file .

        Returns:
            pd.DataFrame: The raw CPI data loaded from the Excel sheet.
        """
        df_cpi = pd.read_excel(self.data_config[self.data_key], sheet_name=1, header=2)
        return df_cpi

    def preprocess_data(self, df_cpi: pd.DataFrame) -> pd.DataFrame:
        """Preprocesses the raw CPI data into a standardized long format.

        Cleans column names, unnests yearly CPI scores into a long format
        and standardizes iso3 country codes.

        Args:
            df_cpi (pd.DataFrame): The raw CPI DataFrame from `load_data()`.

        Returns:
            pd.DataFrame: Preprocessed CPI data indexed by ('iso3', 'year').
        """
        df_cpi.columns = [
            clean(col, no_line_breaks=True, no_punct=True).replace(" ", "_")
            for col in df_cpi.columns
        ]
        df_cpi = df_cpi.drop("region", axis=1)

        cpi_scores = [var for var in df_cpi.columns if "cpi_score" in var]

        def melt_cpi(var, col_list):
            df = df_cpi.melt(
                id_vars=["country_territory", "iso3"],
                value_vars=col_list,
                var_name="year",
                value_name=var,
            )
            df["year"] = df.year.apply(lambda x: int(x[-4:]))
            return df.set_index(["iso3", "year"]).sort_values(["iso3", "year"])

        df_cpi = melt_cpi("cpi_score", cpi_scores).drop(columns="country_territory")
        df_cpi.index = df_cpi.index.set_levels(  # type: ignore
            df_cpi.index.levels[0].str.replace("KSV", "XKX"),
            level=0,  # type: ignore
        )  # Kosovo
        return df_cpi
