from datetime import date
import math
from typing import Literal
import pandas as pd

from base.objects import Dataset


class WPPData(Dataset):
    """Handles loading, preprocessing, and utility calculations for UN WPP data.

    Implements `load_data()` to load either the main WPP data or WPP age group
    data from specified Excel sheets.
    Implements `preprocess_wpp()` to perform basic preprocessing and standardization
    of the data. Also applies a scaling factor to get absolute values.
    Implements `get_wpp_multiplier()` to calculates a cumulative population growth
    factor from 2020 to a target year for a given country based on WPP data.

    Attributes:
        data_key (str): Set to "wpp".
        data_keys (list[str]): Specifies required config keys: "wpp" and "wpp_agegroups".
        local (bool): Set to True, expecting local Excel files.
        # needs_storage (bool): Should be explicitly set if different from Dataset default.
    """

    data_key: str = "wpp"
    data_keys: list[str] = ["wpp", "wpp_agegroups"]
    needs_storage: bool = False

    def load_data(self, variant: Literal["wpp", "wpp_agegroups"] = "wpp") -> pd.DataFrame:
        """Processes wpp excel files, combining estimates with projections."""
        assert variant in self.data_keys
        fp_unwpp = self.data_config[variant]
        if variant == "wpp":
            cols_dict = {
                "Type": "type",
                "Region, subregion, country or area *": "country",
                "ISO3 Alpha-code": "iso3",
                "Year": "year",
                "Total Population, as of 1 January (thousands)": "pop_total",
                "Population Growth Rate (percentage)": "pop_change",
            }
        else:
            cols_dict = {
                "Type": "type",
                "Region, subregion, country or area *": "country",
                "ISO3 Alpha-code": "iso3",
                "Year": "year",
                "0-14": "pop_youth",
                "15-64": "pop_working",
                "65+": "pop_old",
            }
        df_wpp = pd.read_excel(
            fp_unwpp,
            "Estimates",
            header=16,
            usecols=list(cols_dict.keys()),
            na_values="...",
        )
        df_wpp_proj = pd.read_excel(
            fp_unwpp,
            "Medium variant",
            header=16,
            usecols=list(cols_dict.keys()),
            na_values="...",
        )
        df_wpp = df_wpp.rename(columns=cols_dict)
        df_wpp_proj = df_wpp_proj.rename(columns=cols_dict)
        df_wpp = pd.concat(
            [
                df_wpp.query('type == "Country/Area"'),
                df_wpp_proj.query('type == "Country/Area"'),
            ]
        )
        return df_wpp

    def preprocess_wpp(self, df_wpp: pd.DataFrame) -> pd.DataFrame:
        """Preprocesses loaded UN WPP data into a standardized country-year panel.

        This method refines the raw WPP data by filtering to a relevant time
        period, converting  population figures to absolute numbers, and adjusting
        "as of 1 January" population totals to represent values for the previous
        year.

        Args:
            df_wpp (pd.DataFrame): The raw WPP DataFrame loaded by `load_data()`.

        Returns:
            pd.DataFrame: The preprocessed WPP data, indexed by ('iso3', 'year').
        """
        # Restrict to timeframe 2000 - current year + 5
        # should be far enough into the future for anything we need with revised datasets requiring regeneration every 2 years anyways
        df_wpp = df_wpp.query(f"year >= 2000 and year <= {date.today().year + 5}").copy()
        # TODO: remove if no issues
        # # Reset dtypes due to issues with joblib parallel; weird bug with read_excel failing when not using np dtypes
        # df_wpp['year'] = df_wpp['year'].astype(int)
        df_wpp = df_wpp.set_index(["iso3", "year"]).drop(columns=["country", "type"]).sort_index()
        for col in ["pop_total", "pop_youth", "pop_working", "pop_old"]:
            if col in df_wpp.columns:
                df_wpp[col] = df_wpp[col] * 1000  # absolute numbers
        if "pop_total" in df_wpp.columns:
            # move pop_total to prior year since its January 1st values so end of last year
            for iso3 in df_wpp.index.get_level_values("iso3").unique():
                df_wpp.loc[(iso3, slice(None)), "pop_total"] = df_wpp.loc[
                    (iso3, slice(None)), "pop_total"
                ].shift(-1)
        return df_wpp

    def get_wpp_multiplier_2020(
        self, df_wpp: pd.DataFrame, year: int, iso3: str, iso3_errors: bool = True
    ) -> float:
        """Calculates a cumulative population growth multiplier from 2020 to a target year.

        Uses the population growth rate from the preprocessed `df_wpp` DataFrame
        to calculate a cumulative growth factor. This factor represents how much
        the population is projected to have grown from 2020 to the target year.
        `year`. If an iso3 is not found, returns 1 if iso3_errors=False, else
        raises the error. Used for extrapolating population figures beyond
        WorldPop's native data range.

        Args:
            df_wpp (pd.DataFrame): The preprocessed WPP DataFrame, indexed by
                ('iso3', 'year'), containing a 'pop_change' column.
            year (int): The target year for which the growth multiplier is needed.
            iso3 (str): The ISO3 country code for which to calculate the multiplier.
            iso3_errors (bool): Whether to raise errors caused by passed iso3s not
                in the df_wpp. Defaults to True.

        Returns:
            float: The cumulative growth rate multiplier from 2020 up to the
                target `year`. For example, if `year` is 2022, it
                calculates (1 + growth_2020) * (1 + growth_2021).
        """
        # iso3 from WorldPop for Kosovo is KOS, df_wpp uses CCVI standard XKX
        if iso3 == "KOS":
            iso3 = "XKX"
        try:
            yearly_growth_rates = [
                1 + df_wpp.loc[(iso3, y), "pop_change"] / 100  # type: ignore
                for y in range(2020, year)
            ]
            growth_rate = math.prod(yearly_growth_rates)
        except KeyError:
            if iso3_errors:
                raise
            else:
                growth_rate = 1
        return growth_rate  # type: ignore
