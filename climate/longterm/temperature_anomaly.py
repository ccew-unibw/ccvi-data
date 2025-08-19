import os

import pandas as pd
import xarray as xr

from base.datasets.berkley import BERKLEYData
from base.objects import Indicator, ConfigParser, GlobalBaseGrid
from climate.shared import NormalizationMixin


class CliLongtermTemperatureAnomaly(Indicator, NormalizationMixin):
    def __init__(
        self,
        config: ConfigParser,
        grid: GlobalBaseGrid,
        pillar: str = "CLI",
        dim: str = "longterm",
        id: str = "temperature-anomaly",
    ):
        """Params defining indicator's place in index set to designed hierarchy by default"""
        self.berkley = BERKLEYData(config=config)
        super().__init__(pillar=pillar, dim=dim, id=id, config=config, grid=grid)

    def load_data(self) -> dict[str, str]:
        fps_raw = self.berkley.load_data()
        return fps_raw

    def preprocess_data(self, fps_raw: dict[str, str]) -> xr.DataArray:
        da_preprocessed = self.berkley.preprocess_data(fps_raw)
        return da_preprocessed

    def create_indicator(self, da_preprocessed: xr.DataArray) -> pd.DataFrame:
        df_berkley = self.berkley.calculate_grid_quarter_anomalies(da_preprocessed)
        df_indicator = self.create_base_df()
        df_indicator = df_indicator.merge(df_berkley, how="left", left_index=True, right_index=True)
        df_indicator.rename(
            columns={"temperature_anomaly": f"{self.composite_id}_raw"}, inplace=True
        )
        return df_indicator

    def normalize(self, df_indicator: pd.DataFrame) -> pd.DataFrame:
        """Standardized normalization via ClimateMixin"""
        df_normalized = self.climate_normalize(
            df_indicator, self.composite_id, self.indicator_config, self.global_config["start_year"]
        )
        return df_normalized


# this is possible by adding the root folder as the PYTHONPATH var in .env
if __name__ == "__main__":
    config = ConfigParser()
    grid = GlobalBaseGrid(config)
    indicator = CliLongtermTemperatureAnomaly(config=config, grid=grid)
    indicator.run()
