import numpy as np
import pandas as pd

from base.datasets import ACLEDData
from base.objects import Indicator, ConfigParser, GlobalBaseGrid
from conflict.shared import NormalizationMixin
from utils.spatial_operations import create_diffusion_layers


class ConLevelSurrounding(Indicator, NormalizationMixin):
    def __init__(
        self,
        config: ConfigParser,
        grid: GlobalBaseGrid,
        pillar: str = "CON",
        dim: str = "level",
        id: str = "surrounding",
    ):
        """Params defining indicator's place in index set to designed hierarchy by default"""
        self.acled = ACLEDData(config=config, grid=grid)
        super().__init__(pillar=pillar, dim=dim, id=id, config=config, grid=grid)

    def load_data(self) -> pd.DataFrame:
        df = self.acled.load_data()
        return df

    def preprocess_data(self, df_acled: pd.DataFrame) -> pd.DataFrame:
        df_base = self.create_base_df()
        df_preprocessed = self.acled.create_grid_quarter_aggregates(df_base, df_acled)
        return df_preprocessed

    def create_indicator(self, df_preprocessed: pd.DataFrame) -> pd.DataFrame:
        """Indicator and raw values both"""
        base_var = "armed_violence_fatalities"
        df_diffusion = create_diffusion_layers(df_preprocessed, [base_var])
        # sum of all fatalities in the neighborhood as raw value
        df_diffusion = df_diffusion.rename(
            columns={f"{base_var}_diffusion_sum": f"{self.composite_id}_raw"}
        )
        df = pd.concat([df_preprocessed, df_diffusion], axis=1)
        df[self.composite_id] = df["armed_violence_fatalities_diffusion_mean"].apply(np.log1p)
        return df

    def normalize(self, df_indicator: pd.DataFrame) -> pd.DataFrame:
        """Standardized normalization via ConflictMixin"""
        quantile = self.indicator_config["normalization_quantile"]
        return self.conflict_normalize(df_indicator, self.composite_id, quantile)


# this is possible by adding the root folder as the PYTHONPATH var in .env
if __name__ == "__main__":
    config = ConfigParser()
    grid = GlobalBaseGrid(config)
    indicator = ConLevelSurrounding(config=config, grid=grid)
    indicator.run()
