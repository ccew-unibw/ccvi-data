import pandas as pd

from base.objects import Indicator, ConfigParser, GlobalBaseGrid
from conflict.soctens.intensity import ConSoctensIntensity
from conflict.shared import NormalizationMixin, PersistenceMixin


class ConSoctensPersistence(Indicator, NormalizationMixin, PersistenceMixin):
    def __init__(
        self,
        config: ConfigParser,
        grid: GlobalBaseGrid,
        base_indicator: ConSoctensIntensity,
        pillar: str = "CON",
        dim: str = "soctens",
        id: str = "persistence",
    ):
        """Params defining indicator's place in index set to designed hierarchy by default"""
        self.base_indicator = base_indicator
        super().__init__(pillar=pillar, dim=dim, id=id, config=config, grid=grid)

    def load_data(self) -> pd.DataFrame:
        """Loading passed indicator as base data."""
        self.validate_indicator_input(self.base_indicator, ConSoctensIntensity)
        df_base = self.base_indicator.storage.load()
        return df_base

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """No preprocessing necessary, keeping for consistency."""
        return df

    def create_indicator(self, df_preprocessed: pd.DataFrame) -> pd.DataFrame:
        base_name = self.base_indicator.composite_id
        df_preprocessed[self.composite_id] = self.decay_based_persistence(
            df_preprocessed[base_name],
            decay_target=0.25,
            value_weight=0.8,
            timeframe=1,
            name=self.composite_id,
        )
        return df_preprocessed

    def normalize(self, df_indicator: pd.DataFrame) -> pd.DataFrame:
        """No normalization for decay-based persistence values necessary."""
        return df_indicator[[self.composite_id]]


# this is possible by adding the root folder as the PYTHONPATH var in .env
if __name__ == "__main__":
    config = ConfigParser()
    grid = GlobalBaseGrid(config)
    base_indicator = ConSoctensIntensity(config=config, grid=grid)
    indicator = ConSoctensPersistence(config=config, grid=grid, base_indicator=base_indicator)
    indicator.run()
