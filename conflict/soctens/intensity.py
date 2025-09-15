import numpy as np
import pandas as pd
from panel_imputer import PanelImputer

from base.datasets import ACLEDData, VDemData
from base.objects import Indicator, ConfigParser, GlobalBaseGrid
from conflict.shared import NormalizationMixin
from utils.data_processing import process_yearly_data


class ConSoctensIntensity(Indicator, NormalizationMixin):
    def __init__(
        self,
        config: ConfigParser,
        grid: GlobalBaseGrid,
        pillar: str = "CON",
        dim: str = "soctens",
        id: str = "intensity",
    ):
        """Params defining indicator's place in index set to designed hierarchy by default"""
        self.acled = ACLEDData(config=config, grid=grid)
        self.vdem = VDemData(config)
        super().__init__(pillar=pillar, dim=dim, id=id, config=config, grid=grid)

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_acled = self.acled.load_data()
        df_vdem = self.vdem.load_data("v2x_libdem")
        return df_acled, df_vdem

    def preprocess_data(self, input_data: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
        df_acled, df_vdem = input_data
        # produce some additional history for normalization
        df_base = self.create_base_df(self.global_config["start_year"] - 3)
        # acled preprocessing creates the data structure
        acled_preprocessed = self.acled.create_grid_quarter_aggregates(df_base, df_acled)
        df_vdem = self.vdem.preprocess_data(df_vdem)
        # match with grid
        df_preprocessed = process_yearly_data(acled_preprocessed, df_vdem, df_vdem.columns)
        # impute missing data via forward-filling
        imputer = PanelImputer(
            time_index=["year", "quarter"],
            location_index="pgid",
            imputation_method="ffill",
            parallelize=True,
        )
        df_preprocessed["v2x_libdem"] = imputer.fit_transform(df_preprocessed[["v2x_libdem"]])
        return df_preprocessed

    def create_indicator(self, df_preprocessed: pd.DataFrame) -> pd.DataFrame:
        df_preprocessed[self.composite_id] = df_preprocessed["unrest_event_count"].apply(
            np.log1p
        ) * (1 - df_preprocessed["v2x_libdem"])
        return df_preprocessed

    def normalize(self, df_indicator: pd.DataFrame) -> pd.DataFrame:
        """Standardized normalization via ConflictMixin"""
        quantile = self.indicator_config["normalization_quantile"]
        start_year = self.global_config["start_year"]
        return self.conflict_normalize(df_indicator, self.composite_id, quantile, start_year)
    
    def add_raw_value(
        self, df_indicator: pd.DataFrame, df_preprocessed: pd.DataFrame
    ) -> pd.DataFrame:
        df_indicator[f"{self.composite_id}_raw"] = df_preprocessed["unrest_event_count"]
        return df_indicator


# this is possible by adding the root folder as the PYTHONPATH var in .env
if __name__ == "__main__":
    config = ConfigParser()
    grid = GlobalBaseGrid(config)
    indicator = ConSoctensIntensity(config=config, grid=grid)
    indicator.run()
