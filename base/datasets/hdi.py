import pandas as pd

from base.objects import Dataset


class HDIData(Dataset):
    """Handles loading and preprocessing of UNDP Human Development data.

    Implements `load_data()` to read specified Human Development Index (HDI) or
    Gender Inequality Index (GII) components from a CSV file.
    Implements `preprocess_data()` to reshape data from wide to long format
    (country-year panel) and calculate HDI sub-indices (Life Expectancy Index,
    Education Index) depending on the selected components.

    Attributes:
        data_key (str): Set to "hdi".
        needs_storage (bool): Set to False.
    """

    data_key: str = "hdi"
    needs_storage: bool = False

    def load_data(self, hdi_cols: list[str]) -> pd.DataFrame:
        """Loads specified components from the HDI data CSV file.

        Reads the data, selects year-based columns based on prefixes
        provided in `hdi_cols` (e.g., "gii_", "eys_"), excluding gender-specific
        or rank columns. The 'iso3' column is also selected.

        Args:
            hdi_cols (list[str]): A list of base variable names to identify and
                load relevant columns from the raw data file. Currently supported
                by preprocessing are "gii", "eys", "mys", "le".

        Returns:
            pd.DataFrame: A DataFrame containing the 'iso3' column and the
                selected raw HDI component columns (still in wide format by year).
        """
        supported_hdi_cols = ["gii", "eys", "mys", "le"]
        assert all([col in supported_hdi_cols for col in hdi_cols]), (
            f"Currently, only {supported_hdi_cols} are supported as options in 'hdi_cols'."
        )
        # Note the unusual encoding, present since the 2023-24 versions
        df_hdi = pd.read_csv(self.data_config[self.data_key], encoding="windows-1252")
        columns = [
            col
            for col in df_hdi.columns
            if col.startswith(tuple([f"{col}_" for col in hdi_cols]))
            and ("_f_" not in col and "_m_" not in col and "_rank_" not in col)
        ]
        df_hdi = df_hdi[["iso3"] + columns]
        return df_hdi

    def preprocess_data(self, df_hdi: pd.DataFrame) -> pd.DataFrame:
        """Preprocesses the raw HDI data into a standardized long format.

        Transforms the wide-format HDI data (where years are columns) into a
        long format and filters out unneccessary years and regions.

        If relevant component columns are present, it calculates:
        - 'le_index' (Life Expectancy Index)
        - 'edu_index' (Education Index).

        Args:
            df_hdi (pd.DataFrame): The raw HDI DataFrame from `load_data()`,
                containing selected component columns in wide format.

        Returns:
            pd.DataFrame: Preprocessed HDI data in long format, indexed by
                ('iso3', 'year'), including original components and calculated
                indices (le_index, edu_index if applicable).
        """

        def melt_hdi(var):
            value_vars = [col for col in df_hdi.columns if col.startswith(var)]
            if len(value_vars) == 0:
                raise ValueError(f"Var {var} not in passed df_hdi.")
            else:
                df = df_hdi.melt(
                    id_vars=["iso3"], value_vars=value_vars, var_name="year", value_name=var
                )
                df["year"] = df.year.apply(lambda x: int(x[-4:]))
                return df

        melted_dfs = []
        for var in ["gii", "eys", "mys", "le"]:
            try:
                df_melted = melt_hdi(var)
            except ValueError:
                continue
            df_melted = df_melted[df_melted.year >= self.global_config["start_year"]]
            df_melted = df_melted[~df_melted.iso3.str.contains(r"\.")]
            df_melted = df_melted.set_index(["iso3", "year"]).sort_index()
            melted_dfs.append(df_melted)
        df_hdi = pd.concat(melted_dfs, axis=1)

        if "le" in df_hdi.columns:
            self.console.print("Life expectancy limits: 85/20 years ")
            df_hdi["le_index"] = df_hdi["le"].apply(self._le_index)
        if "eys" in df_hdi.columns and "mys" in df_hdi.columns:
            self.console.print("Expected/mean years of schooling limits: 18/15 years ")
            df_hdi["edu_index"] = df_hdi.apply(
                lambda x: self._edu_index(x["eys"], x["mys"]), axis=1
            )
        return df_hdi

    # Implementation of hdi scores as of May 2025. Based on "Calculating the Indices" Excel
    # @ https://hdr.undp.org/data-center/documentation-and-downloads
    def _le_index(self, le: float) -> float:
        """Calculates the Life Expectancy Index component of the HDI."""
        if le > 85:
            le = 85
        elif le < 20:
            le = 20
        return (le - 20) / (85 - 20)

    def _edu_index(self, eys: float, mys: float) -> float:
        """Calculates the Education Index component of the HDI."""
        if eys > 18:
            eys = 18
        if mys > 15:
            mys = 15
        return (eys / 18 + mys / 15) / 2
