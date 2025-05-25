import country_converter as coco
import pandas as pd

from base.objects import Dataset


class FHData(Dataset):
    """Handles loading and preprocessing of Freedom House's Freedom in the World data.

    Implements `load_data()` to read scores from an Excel file, renaming columns
    and adjusting years.
    Implements `preprocess_data()` to filter locations, assign ISO3 country
    codes, and normalize scores to percentages.

    Attributes:
        data_key (str): Set to "freedomhouse".
        needs_storage (bool): Set to False.
    """

    data_key = "freedomhouse"
    needs_storage: bool = False

    def load_data(self) -> pd.DataFrame:
        """Loads Freedom in the World data from the specified Excel file.

        Reads a the data, selects relevant columns, renames them to standardized
        names, and adjusts the 'year' column (as FH Edition refers to the
        previous year's assessment).

        Returns:
            pd.DataFrame: The raw Freedom House data with standardized column names.
        """
        rename_cols = {
            "Country/Territory": "country",
            "Edition": "year",  # needs to be -1 in Data
            "PR": "political_rights",  # 40 max
            "CL": "civil_liberties",  # 60 max
            "Total": "fh_score",
        }
        # read data
        df_free = pd.read_excel(
            self.data_config[self.data_key],
            sheet_name=1,
            header=1,
            usecols=list(rename_cols.keys()),
        ).rename(columns=rename_cols)
        df_free["year"] = df_free.year - 1  # FH Edition is always for the previous year
        return df_free

    def preprocess_data(self, df_free: pd.DataFrame) -> pd.DataFrame:
        """Preprocesses the raw Freedom House data.

        Filters out specific contested territories that do not appear in the
        standardized grid-iso3 matching. Converts country names to ISO3 codes
        using the `country_converter` library. Normalizes scores by calculating
        their percentage of their respective maximum possible scores.

        Args:
            df_free (pd.DataFrame): The raw Freedom House DataFrame from `load_data()`.

        Returns:
            pd.DataFrame: Preprocessed Freedom House data indexed by ('iso3', 'year'),
                with original scores and normalized percentage scores.
        """
        # explicitly excluding contested territories or sub-regions that are not differentiated in the grid / other data sources for clarity
        df_free = df_free.loc[
            ~df_free.country.isin(
                [
                    "Pakistani Kashmir",
                    "Indian Kashmir",
                    "Somaliland",
                    "Northern Cyprus",
                    "Gaza Strip",
                    "Puerto Rico",
                    "Abkhazia",
                    "Crimea",
                    "Eastern Donbas",
                    "Nagorno-Karabakh",
                    "South Ossetia",
                    "Tibet",
                    "Transnistria",
                ]
            )
        ].copy()
        # assign iso3 code
        c_dict = dict(
            zip(df_free.country.unique(), coco.convert(names=df_free.country.unique(), to="ISO3"))
        )
        df_free["iso3"] = df_free.country.apply(lambda x: c_dict[x])
        # new 2024, hard to deal with, dropping for now
        df_free = df_free.loc[df_free.country != "Russian-Occupied Territories of Ukraine"]

        # some further processing
        df_free = df_free.set_index(["iso3", "year"]).drop(columns="country")
        # normalized by turning score into percentage of max points
        max_points = {"political_rights": 40, "civil_liberties": 60, "fh_score": 100}
        for col in max_points:
            df_free[f"{col}_percent"] = df_free[col] / max_points[col]
        return df_free
