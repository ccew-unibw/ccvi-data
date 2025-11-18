import pandas as pd
import numpy as np
from utils.data_processing import get_quarter

from base.datasets.ecmwf_spei import CDSECMWFSPEIData
from base.objects import Indicator, ConfigParser, GlobalBaseGrid

from climate.shared import NormalizationMixin


class CliAccumulatedDrought(Indicator, NormalizationMixin):
    def __init__(
        self,
        config: ConfigParser,
        grid: GlobalBaseGrid,
        pillar: str = "CLI",
        dim: str = "accumulated",
        id: str = "drought",
    ):
        """Params defining indicator's place in index set to designed hierarchy by default"""
        self.event_data = CDSECMWFSPEIData(config=config)
        super().__init__(pillar=pillar, dim=dim, id=id, config=config, grid=grid)

    def load_data(self) -> pd.DataFrame:
        df = self.event_data.load_data()
        return df

    def preprocess_data(self, df_event_data: pd.DataFrame) -> pd.DataFrame:
        fp_preprocessed = self.event_data.storage.build_filepath(
            "processing", filename="preprocessed"
        )
        try:
            if self.regenerate["preprocessing"]:
                raise FileNotFoundError
            df_preprocessed = pd.read_parquet(fp_preprocessed)
            last_quarter_date = get_quarter("last")
            if df_preprocessed["time"].max().date() < last_quarter_date:
                raise FileNotFoundError
            return df_preprocessed

        except FileNotFoundError:
            print("-- df_base creation ...")
            df_base = self.create_base_df(start_year=df_event_data["EVENT_DATE"].dt.year.min())
            print("-- create_grid_quarter_aggregates ...")
            df_preprocessed = self.event_data.select_quarter_values(df_base, df_event_data)
            df_preprocessed.to_parquet(fp_preprocessed)
            return df_preprocessed

    def create_indicator(self, df_preprocessed: pd.DataFrame) -> pd.DataFrame:
        # Drought indicator
        # mean of negative SPEI-12 values over the past seven years for the respective quarter

        # the raw value should not be reversed and is thus negative
        # the indicator is based on values < 0, with higher values set to 0 for aggregation
        # the indicator score is reversed (so higher normalized values imply more severe drought)

        df_indicator = df_preprocessed[["pgid", "year", "quarter", "lat", "lon", "spei12"]]
        df_indicator["spei12_n"] = df_indicator["spei12"].apply(
            lambda x: x * -1 if x < 0 or np.isnan(x) else 0
        )

        # the mean of negative SPEI-12 values over the past seven years
        df_indicator = df_indicator.set_index(["pgid", "year", "quarter"]).sort_index()
        means = df_indicator.groupby(["pgid", "quarter"])["spei12_n"].rolling(window=7).mean()
        means = means.reset_index(level=[0, 1], drop=True).sort_index()
        df_indicator[self.composite_id] = means
        df_indicator[f"{self.composite_id}_raw"] = means * -1
        return df_indicator

    def normalize(self, df_indicator: pd.DataFrame) -> pd.DataFrame:
        """Standardized normalization via ClimateMixin"""
        df_normalized = self.climate_normalize(
            df_indicator,
            self.composite_id,
            self.indicator_config,
            self.global_config["start_year"],
            False,
        )
        return df_normalized


# this is possible by adding the root folder as the PYTHONPATH var in .env
if __name__ == "__main__":
    config = ConfigParser()
    grid = GlobalBaseGrid(config)
    indicator = CliAccumulatedDrought(config=config, grid=grid)
    indicator.run()
