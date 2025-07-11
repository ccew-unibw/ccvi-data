from collections.abc import Callable
from datetime import date
from functools import partial
import gzip

import numpy as np
import pandas as pd
import requests

from base.objects import Dataset


class ILOData(Dataset):
    """Handles downloading and preprocessing of International Labour Organisation (ILO) data.

    Implements `load_data()` to download data for multiple specified ILO indicators
    from the ILO API.
    Implements `preproces_data_agrdep()` to orchestrate the specific preprocessing
    for different ILO indicators, standardize them to a yearly panel, and infill
    data gps using ILO model estimates.

    Attributes:
        data_key (str): Set to "ilo".
        local (bool): Set to False, as data is sourced from the ILO API.
        needs_storage (bool): Set to False.
    """

    data_key: str = "ilo"
    local: bool = False
    needs_storage: bool = False

    def load_data(self, indicators: dict[str, str]) -> dict[str, dict]:
        """Downloads data for multiple ILO indicators from the ILO API.

        For each indicator ID provided, it fetch data from 5 years before the
        global config's start_year up to the present. The downloaded data and the
        provided output name for the indicator are returned in a nested dictionary
        as input for the processing method.

        Args:
            indicators (dict[str, str]): A dictionary where keys are ILO
                indicator codes and values are the desired output names for these
                indicators.

        Returns:
            dict[str, dict[str, Any]]: A nested dictionary. Outer keys are ILO
                indicator codes. Inner dictionary contains 'df' (the downloaded
                DataFrame for that indicator) and 'name' (the desired output name).
        """
        df_dict = {i: {} for i in indicators}
        year_min = self.global_config["start_year"] - 5
        # downloading indicators from ILO web service (https://rplumber.ilo.org/__docs__/)
        for i in indicators:
            url = f"https://rplumber.ilo.org/data/indicator/?id={i}&timefrom={year_min}&format=.csv"
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                with gzip.GzipFile(fileobj=response.raw) as decompressed_file:
                    df_i = pd.read_csv(decompressed_file, low_memory=True)    # type: ignore
            df_i = df_i.drop_duplicates()  # input data seems not quite clean
            df_dict[i]["df"] = df_i
            df_dict[i]["name"] = indicators[i]
        return df_dict

    def preproces_data_agrdep(self, df_dict: dict[str, dict]) -> pd.DataFrame:
        """Preprocesses and combines ILO datasets for the agricultural dependency indicator.

        This method orchestrates the preprocessing of different ILO indicators
        downloaded by `load_data`. It applies indicator-specific preprocessing
        methods. The preprocessed data for each indicator is then standardized
        to a ('iso3', 'year') format via `_prep_yearly_ilo_data` and combined.
        Finally, missing data is potentially infilled using ILO model estimates.

        Args:
            df_dict (dict[str, dict[str, Any]]): The nested dictionary returned
                by `load_data`.

        Returns:
            pd.DataFrame: A DataFrame indexed by ('iso3', 'year') containing
                the processed and combined ILO indicators for the agricultural
                dependency CCVI indicator.
        """
        # different reshaping functions depending on the indicator
        functions = {
            i: partial(self._prep_ilo_sex_age_data, output_name=df_dict[i]["name"])
            if "SEX_AGE" in i
            else partial(self._prep_ilo_agri_sector, output_name=df_dict[i]["name"])
            for i in df_dict
        }
        df_list = []
        names = []
        for i in df_dict:
            df_i = self._prep_yearly_ilo_data(df_dict[i]["df"], functions[i])
            df_list.append(df_i)
            names.append(df_dict[i]["name"])
        df_ilo = pd.concat(df_list, axis=1)
        df_ilo = self._fill_ilo_agr_data_model(df_ilo, [n for n in names if "_model" not in n])
        return df_ilo

    def _prep_yearly_ilo_data(
        self, df: pd.DataFrame, function: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> pd.DataFrame:
        """Applies indicator-specific preprocessing and standardizes data structure.

        First, this method applies the provided `function` to the `df` to transform
        it into a ('iso3', 'year') format with an indicator value column. Then, it
        creates a complete yearly panel structure for all countries found in the
        processed data, merges the indicator to it and cleans the iso3 code.

        Args:
            df (pd.DataFrame): The raw DataFrame for a single ILO indicator as
                returned with the output from `load_data`.
            function (Callable[[pd.DataFrame], pd.DataFrame]):
                A function that takes the raw indicator DataFrame and returns a
                processed DataFrame indexed by ('iso3', 'year') and containing
                the indicator column(s).

        Returns:
            pd.DataFrame: The indicators combined in a standardized ('iso3', 'year')
                dataframe, covering all relevant years.
        """
        df_merge = function(df)
        # prep output df
        countries = df_merge.index.get_level_values("iso3").unique()
        # remove regional aggregates
        countries = [country for country in countries if country[0] != "X"]
        years = np.arange(self.global_config["start_year"], date.today().year + 1)
        df_out = pd.DataFrame(
            data=zip(countries, [years for c in countries]), columns=["iso3", "year"]
        )
        df_out = df_out.explode("year").set_index(["iso3", "year"]).sort_index()
        df_out = df_out.merge(df_merge, how="left", left_index=True, right_index=True)
        # clean up Kosovo
        df_out.index = df_out.index.set_levels(  # type: ignore
            df_out.index.levels[0].str.replace("KOS", "XKX"),  # type: ignore
            level=0,
        )
        return df_out

    def _prep_ilo_agri_sector(self, df: pd.DataFrame, output_name: str) -> pd.DataFrame:
        """Preprocesses ILO employment by sector data to calculate agricultural share.

        Specifically tailored for 'EMP_TEMP_SEX_ECO_NB_A' indicator.
        It renames columns, filters for relevant sectors ('TOTAL', 'AGR') and
        total sex ('SEX_T'), unnests the sector data, calculates the agricultural
        employment share (`agr / total_employed`), and returns this share under
        the specified `output_name`.

        Args:
            df (pd.DataFrame): Raw DataFrame for an ILO employment by sector
                indicator from `load_data`.
            output_name (str): The desired column name for the calculated
                agricultural sector share.

        Returns:
            pd.DataFrame: A DataFrame indexed by ('iso3', 'year') with a single
                column named `output_name` containing the agricultural employment share.
        """
        rename_cols = {
            "ref_area": "iso3",
            "time": "year",
            "classif1": "sector",
            "obs_value": "value",
        }
        df = df.rename(columns=rename_cols)
        df = df.drop(
            columns=[col for col in df.columns if col not in list(rename_cols.values()) + ["sex"]]
        )
        df = df[df.sector.str.contains("SECTOR")]
        df.sector = df.sector.replace(
            {"ECO_SECTOR_TOTAL": "total_employed", "ECO_SECTOR_AGR": "agr"}
        )
        df = df[df.sector.isin(["total_employed", "agr"])]
        df = df[df.sex == "SEX_T"]
        df = df.set_index(["iso3", "year", "sector"])["value"].unstack("sector").sort_index()
        df[output_name] = df.agr / df.total_employed
        return df[[output_name]]

    def _prep_ilo_sex_age_data(self, df: pd.DataFrame, output_name: str) -> pd.DataFrame:
        """Preprocesses ILO data disaggregated by sex and age.

        Renames standard columns ('ref_area' to 'iso3', 'time' to 'year'),
        filters for total sex ('SEX_T') and age groups older than 15
        ('AGE_YTHADULT_YGE15'), and sets 'iso3' and 'year' as the index.
        The observation value is renamed to `output_name`.

        Args:
            df (pd.DataFrame): Raw DataFrame for an ILO indicator disaggregated
                by sex and age from `load_data`.
            output_name (str): The desired column name for the indicator value.

        Returns:
            pd.DataFrame: A DataFrame indexed by ('iso3', 'year') with a single
                column named `output_name` containing the indicator value.
        """
        rename_cols = {
            "ref_area": "iso3",
            "time": "year",
            "classif1": "age",
            "obs_value": output_name,
        }
        df = df.rename(columns=rename_cols)
        df = df.drop(
            columns=[col for col in df.columns if col not in list(rename_cols.values()) + ["sex"]]
        )
        df = df[(df.sex == "SEX_T") & (df.age == "AGE_YTHADULT_YGE15")]
        df = df.set_index(["iso3", "year"]).sort_index()
        return df[[output_name]]

    def _fill_ilo_agr_data_model(self, df_ilo: pd.DataFrame, ilo_vars: list[str]) -> pd.DataFrame:
        """Fills gaps in data using ILO model estimates.

        For specified variables `ilo_vars`, if a country has insufficient actual
        data points (fewer than 4), this method replaces the entire time series
        for that country with data generated from from corresponding ILO model
        estimates, provided model data is available. The full replacement ensures
        internal consistency if model data is used.

        Args:
            df_ilo (pd.DataFrame): The DataFrame containing ILO-derived indicators,
                with versions from both from reported data and from corresponding
                model estimate data (e.g., 'var' and 'var_model').

        Returns:
            pd.DataFrame: The input DataFrame with specified variables potentially
                infilled using model data, returning only the final combined version.
        """
        for var in ilo_vars:
            assert var in df_ilo.columns and f"{var}_model" in df_ilo.columns
        df = df_ilo.copy()
        for iso3 in df.index.get_level_values("iso3").unique():
            # where there is no(!) or not enough data to impute for a country, use data from ilo model
            for var in ilo_vars:
                # condition is set to less than 4 data values (arbitrary)
                if sum(~df.loc[iso3, var].isna()) < 4:  # type: ignore
                    # sometimes we have cases where there is no model data then we don't want to replace incase we do have a few data points
                    if not df.loc[iso3, f"{var}_model"].isna().all():  # type: ignore
                        # if we fill, we want to use only model data for internal consistency
                        df.loc[iso3, var] = df.loc[iso3, f"{var}_model"].values  # type: ignore
        return df[ilo_vars]
