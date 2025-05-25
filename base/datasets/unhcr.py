import io
import zipfile
import requests

import pandas as pd

from base.objects import Dataset


class UNHCRData(Dataset):
    """Handles downloading and preprocessing of UNHCR statistics.

    Implements `load_data()` to download population data (including refugees,
    asylum-seekers, IDPs) from the UNHCR API as a ZIP archive, extract it,
    and load the relevant CSV.
    Implements `preprocess_data()` to rename columns and calculate a total
    'forcibly_displaced' count.

    Attributes:
        data_key (str): Set to "unhcr".
        local (bool): Set to False, as data is sourced from the UNHCR API.
    """

    data_key = "unhcr"
    local = False

    def load_data(self) -> pd.DataFrame:
        """Downloads and extracts UNHCR population data.

        Fetches a ZIP archive containing population statistics from the UNHCR API,
        starting from the project's `start_year`. The archive is extracted to
        the dataset's processing directory, and the data CSV (typically named
        'population.csv') is then loaded into a pandas DataFrame.

        Returns:
            pd.DataFrame: The raw UNHCR population data loaded from the extracted CSV.
        """
        # no regenerate, since its so little data and takes no time to download new
        year_start = self.global_config["start_year"]
        url = f"https://api.unhcr.org/population/v1/population/?year_from={year_start}&download=true&coa_all=true&cf_type=ISO"
        response = requests.get(url)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(self.storage.storage_paths["processing"])

        fp = self.storage.build_filepath("processing", filename="population", filetype=".csv")
        df_unhcr = pd.read_csv(fp, na_values="-")
        return df_unhcr

    def preprocess_data(self, df_unhcr: pd.DataFrame) -> pd.DataFrame:
        """Preprocesses the raw UNHCR population data.

        Renames columns and selects relevant populations (refugees, asylum-seekers,
        other people in need of international protection, IDPs). It then calculates
        a summary column 'forcibly_displaced' by summing the selected population
        figures.

        Args:
            df_unhcr (pd.DataFrame): The raw UNHCR data DataFrame from `load_data()`.

        Returns:
            pd.DataFrame: Preprocessed UNHCR data, indexed by ('iso3', 'year'),
                containing selected population figures and the calculated
                'forcibly_displaced' total sum.
        """
        cols_of_interest = {
            "Country of asylum (ISO)": "iso3",
            "Year": "year",
            "Refugees under UNHCR's mandate": "refugees",
            "Asylum-seekers": "asylum_seekers",
            "Other people in need of international protection": "others_need_of_protection",
            "IDPs of concern to UNHCR": "idps",
        }
        df_unhcr = df_unhcr.rename(columns=cols_of_interest)
        df_unhcr = df_unhcr[list(cols_of_interest.values())]
        df_unhcr = df_unhcr.set_index(["iso3", "year"]).sort_index()
        df_unhcr["forcibly_displaced"] = df_unhcr.sum(axis=1)
        return df_unhcr
