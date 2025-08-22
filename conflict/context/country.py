import numpy as np
import pandas as pd

from base.datasets import ACLEDData, WPPData
from base.objects import Indicator, ConfigParser, GlobalBaseGrid
from conflict.shared import NormalizationMixin
from utils.data_processing import default_impute, min_max_scaling


class ConContextCountry(Indicator, NormalizationMixin):
    def __init__(
        self,
        config: ConfigParser,
        grid: GlobalBaseGrid,
        pillar: str = "CON",
        dim: str = "context",
        id: str = "country",
    ):
        """Params defining indicator's place in index set to designed hierarchy by default"""
        self.acled = ACLEDData(config=config, grid=grid)
        self.wpp = WPPData(config=config)
        super().__init__(pillar=pillar, dim=dim, id=id, config=config, grid=grid)

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_acled = self.acled.load_data()
        df_wpp = self.wpp.load_data()
        return df_acled, df_wpp

    def preprocess_data(
        self, dfs_input: tuple[pd.DataFrame, pd.DataFrame]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_acled, df_wpp = dfs_input
        df_acled = self.acled.preprocess_data(df_acled)
        df_wpp = self.wpp.preprocess_wpp(df_wpp)
        df_wpp["quarter"] = [[1, 2, 3, 4] for i in range(len(df_wpp))]
        df_wpp = df_wpp.explode("quarter")
        df_wpp = df_wpp.set_index("quarter", append=True).sort_index()
        df_wpp.loc[(slice(None), slice(None), slice(1, 3)), slice(None)] = np.nan
        df_wpp = default_impute(df_wpp, location_index="iso3")
        return df_acled, df_wpp

    def create_indicator(self, dfs_preprocessed: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
        df_acled, df_wpp = dfs_preprocessed
        # produce some additional history for normalization
        df_base = self.create_base_df(self.global_config["start_year"] - 3)
        df_gridded = self.acled.create_grid_quarter_aggregates(df_base, df_acled)
        # country-level fatalities/pop
        df_countries = self._conflict_density_country(df_wpp, df_acled)
        # country affectedness based on grid-quarter aggregates
        country_affectedness = df_gridded.groupby(["iso3", "year", "quarter"])[
            "acled_fatalities"
        ].agg(self._country_extent)
        country_affectedness = country_affectedness.rename("country_affectedness")
        # combine data
        df_countries = df_countries.merge(
            country_affectedness, how="left", on=["iso3", "year", "quarter"]
        )
        df_indicator = df_base.reset_index().merge(
            df_countries, how="left", on=["iso3", "year", "quarter"]
        )
        df_indicator = df_indicator.set_index(["pgid", "year", "quarter"]).sort_index()
        # re-apply acled coverage information
        df_indicator["acled_coverage"] = df_gridded["acled_coverage"]
        for col in ["fatalities_pop", "country_affectedness"]:
            df_indicator.loc[df_indicator.acled_coverage, col] = df_indicator[col].fillna(0)
            df_indicator.loc[~df_indicator.acled_coverage, col] = np.nan
        # clean & normalize fatalities/pop for combination
        df_indicator[f"{self.composite_id}_raw1"] = df_indicator["fatalities_pop"].apply(np.log1p)
        df_indicator["fatalities_pop_log"] = self.conflict_normalize(
            df_indicator,
            f"{self.composite_id}_raw1",
            self.indicator_config["normalization_quantile"],
        )
        df_indicator = df_indicator.drop(columns=[f"{self.composite_id}_raw1"])
        # manual time limit since normalization only applies to one component
        df_indicator = df_indicator.loc[
            (slice(None), slice(self.global_config["start_year"], None), slice(None)), slice(None)
        ]
        # combine to indicator
        df_indicator[self.composite_id] = df_indicator[
            ["fatalities_pop_log", "country_affectedness"]
        ].mean(axis=1)
        return df_indicator

    def normalize(self, df_indicator: pd.DataFrame) -> pd.DataFrame:
        """Fatalities already normalized, so only stardard winsorization done here"""
        q = self.indicator_config["normalization_quantile"]
        threshold = df_indicator[self.composite_id].quantile(q)
        df_indicator[self.composite_id] = min_max_scaling(
            df_indicator[self.composite_id], maxv=threshold
        )
        return df_indicator[[c for c in df_indicator.columns if self.composite_id in c]]

    def _conflict_density_country(
        self,
        df_wpp: pd.DataFrame,
        df_acled: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame:
        """

        acled_grouped = df_acled.groupby(["iso3", "year", "quarter"])["fatalities"].sum()
        df_wpp = df_wpp.loc[
            (slice(None), slice(self.global_config["start_year"], None), slice(None)), slice(None)
        ]
        df = pd.merge(df_wpp, acled_grouped, on=["iso3", "year", "quarter"], how="left")
        df["fatalities_pop"] = df["fatalities"] / df["pop_total"]
        return df[["fatalities_pop"]]

    @staticmethod
    def _country_extent(a, t=1) -> float:
        mask = a >= t
        if mask.sum() == 0:
            return 0
        else:
            return mask.sum() / len(a)


# this is possible by adding the root folder as the PYTHONPATH var in .env
if __name__ == "__main__":
    config = ConfigParser()
    grid = GlobalBaseGrid(config)
    indicator = ConContextCountry(config=config, grid=grid)
    indicator.run()
