import numpy as np
import pandas as pd

from base.datasets import UCDPData
from base.datasets.wpp import WPPData
from base.objects import Indicator, ConfigParser, GlobalBaseGrid
from conflict.shared import NormalizationMixin
from utils.data_processing import default_impute


class ConContextActors(Indicator, NormalizationMixin):
    def __init__(
        self,
        config: ConfigParser,
        grid: GlobalBaseGrid,
        pillar: str = "CON",
        dim: str = "context",
        id: str = "actors",
    ):
        """Params defining indicator's place in index set to designed hierarchy by default"""
        self.ucdp = UCDPData(config=config)
        self.wpp = WPPData(config=config)
        super().__init__(pillar=pillar, dim=dim, id=id, config=config, grid=grid)

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_ucdp = self.ucdp.load_data()
        df_wpp = self.wpp.load_data()
        return df_ucdp, df_wpp

    def preprocess_data(self, dfs_input: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_ucdp, df_wpp = dfs_input
        df_ucdp = self.ucdp.preprocess_data(df_ucdp, self.grid)
        country_stats = df_ucdp.groupby(["iso3", "year", "quarter"])[["event_count", "best"]].sum()
        # single entry per actor
        df_ucdp["actor"] = df_ucdp.apply(lambda x: [x["side_a"], x["side_b"]], axis=1)
        df_ucdp = df_ucdp.explode("actor")
        grouped_base = (
            df_ucdp.groupby(["iso3", "year", "quarter", "actor"])["event_count"].sum().reset_index()
        )
        # cleaning: drop civilians (only victims) and unclear actors
        grouped_base = grouped_base.loc[grouped_base.actor.str.lower() != "civilians"]
        grouped_base = grouped_base.loc[~grouped_base.actor.str.contains("XXX")]
        # prep population data
        df_wpp = self.wpp.preprocess_wpp(df_wpp)
        df_wpp["quarter"] = [[1, 2, 3, 4] for i in range(len(df_wpp))]
        df_wpp = df_wpp.explode("quarter")
        df_wpp = df_wpp.set_index("quarter", append=True).sort_index()
        df_wpp.loc[(slice(None), slice(None), slice(1, 3)), slice(None)] = np.nan
        df_wpp = default_impute(df_wpp, location_index="iso3")
        return grouped_base, country_stats, df_wpp

    def create_indicator(self, dfs_preprocessed: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
        grouped_base, country_stats, df_wpp = dfs_preprocessed
        # event threshold for actors currently set to 0, could be revisited in the future
        event_threshold = 0
        grouped = (
            grouped_base.loc[grouped_base.event_count >= event_threshold]
            .groupby(["iso3", "year", "quarter"])["actor"]
            .count()
        )
        grouped = pd.merge(grouped, country_stats, how="left", on=["iso3", "year", "quarter"])
        grouped = pd.merge(grouped, df_wpp, how="left", on=["iso3", "year", "quarter"])
        grouped[self.composite_id] = (
            grouped["actor"].apply(np.log1p) * grouped["best"].apply(np.log1p) /
            grouped["pop_total"].apply(np.log1p)
        )
        # produce some additional history for normalization
        df_base = self.create_base_df(self.global_config["start_year"] - 3)
        df_indicator = df_base.reset_index().merge(
            grouped, how="left", on=["iso3", "year", "quarter"]
        )
        df_indicator = df_indicator.set_index(["pgid", "year", "quarter"]).sort_index()
        df_indicator = df_indicator.fillna(0)
        return df_indicator

    def normalize(self, df_indicator: pd.DataFrame) -> pd.DataFrame:
        """Standardized normalization via ConflictMixin"""
        quantile = self.indicator_config["normalization_quantile"]
        start_year = self.global_config["start_year"]
        return self.conflict_normalize(df_indicator, self.composite_id, quantile, start_year)


if __name__ == "__main__":
    config = ConfigParser()
    grid = GlobalBaseGrid(config)
    indicator = ConContextActors(config=config, grid=grid)
    indicator.run()
