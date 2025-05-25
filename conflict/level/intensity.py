import numpy as np
import pandas as pd

from base.datasets import ACLEDData
from base.objects import Indicator, ConfigParser, GlobalBaseGrid
from conflict.shared import NormalizationMixin


class ConLevelIntensity(Indicator, NormalizationMixin):
    def __init__(
        self,
        config: ConfigParser,
        grid: GlobalBaseGrid,
        pillar: str = "CON",
        dim: str = "level",
        id: str = "intensity",
    ):
        """Params defining indicator's place in index set to designed hierarchy by default"""
        self.acled = ACLEDData(config=config)
        super().__init__(pillar=pillar, dim=dim, id=id, config=config, grid=grid)

    def load_data(self) -> pd.DataFrame:
        df = self.acled.load_data()
        return df

    def preprocess_data(self, df_acled: pd.DataFrame) -> pd.DataFrame:
        df_base = self.create_base_df()
        df_preprocessed = self.acled.create_grid_quarter_aggregates(df_base, df_acled)
        return df_preprocessed

    def create_indicator(self, df_preprocessed: pd.DataFrame) -> pd.DataFrame:
        # logging conflict to expand differentiation at lower levels
        df_preprocessed[self.composite_id] = df_preprocessed["armed_violence_fatalities"].apply(
            np.log1p
        )
        return df_preprocessed

    def normalize(self, df_indicator: pd.DataFrame) -> pd.DataFrame:
        """Standardized normalization via ConflictMixin"""
        quantile = self.indicator_config["normalization_quantile"]
        return self.conflict_normalize(df_indicator, self.composite_id, quantile)

    def add_raw_indicator(
        self, df_indicator: pd.DataFrame, df_preprocessed: pd.DataFrame
    ) -> pd.DataFrame:
        df_indicator[f"{self.composite_id}_raw"] = df_preprocessed["armed_violence_fatalities"]
        return df_indicator


# this is possible by adding the root folder as the PYTHONPATH var in .env
if __name__ == "__main__":
    config = ConfigParser()
    grid = GlobalBaseGrid(config)
    indicator = ConLevelIntensity(config=config, grid=grid)
    indicator.run()
