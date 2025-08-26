import pandas as pd
import numpy as np
from utils.index import get_quarter

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
            fp_preprocessed = pd.read_parquet(fp_preprocessed)
            # compare to end, since data is monthly
            last_quarter_date = get_quarter("last", "end")
            if fp_preprocessed["time"].max().date() < last_quarter_date:
                raise FileNotFoundError
            return fp_preprocessed

        except FileNotFoundError:
            print("-- df_base creation ...")
            df_base = self.create_base_df(start_year=df_event_data["EVENT_DATE"].dt.year.min())
            print("-- create_grid_quarter_aggregates ...")
            df_preprocessed = self.event_data.create_grid_quarter_aggregates(df_base, df_event_data)

            return df_preprocessed

    def create_indicator(self, df_preprocessed: pd.DataFrame) -> pd.DataFrame:

        #Drought indicator
        #For dimension 1 it would be SPEI-12 over the past twelve months, 
        #for dimension 2 it would be the mean of negative annual SPEI-12 values over the past seven years. Else, we concluded that

        #the raw value should not be reversed and is thus negative
        #the raw value starts at < 0. Values above are set to zero
        #the indicator score is reversed (so higher normalized values imply more severe drought)
        #the data is ideally derived from ECWMF. The key question is whether the "intermediate data" is timely enough?
        #the SPEI-12 is masked over areas below certain precipitaiton thresholds. I assume that the ECWMF data comes with that mask already. Please double check.

        # make the 1y indicator aggregation
        spei_current = df_preprocessed[["pgid", "year", "quarter", "lat", "lon", "spei12"]]

        spei_current.loc[spei_current["spei12"] > 0, "spei12"] = 0

        spei_current.rename(columns={"spei12": f"{self.composite_id}_raw"}, inplace=True)

        return spei_current

    def normalize(self, df_indicator: pd.DataFrame) -> pd.DataFrame:
        """Standardized normalization via ClimateMixin"""

        df_indicator[f"{self.composite_id}_raw"] = df_indicator[f"{self.composite_id}_raw"] * -1
        
        #the mean of negative annual SPEI-12 values over the past seven years
        df_indicator[f"{self.composite_id}_raw"] = df_indicator.groupby('pgid')[f"{self.composite_id}_raw"].rolling(window=7*4).mean().reset_index(0, drop=True)

        
        df_normalized = self.climate_normalize(
            df_indicator, self.composite_id, self.indicator_config, self.global_config["start_year"]
        )
        df_normalized[f"{self.composite_id}_raw"] = df_normalized[f"{self.composite_id}_raw"] * -1

        #replace nan with 0
        df_normalized[f"{self.composite_id}_raw"].fillna(0, inplace=True)
        
        return df_normalized


# this is possible by adding the root folder as the PYTHONPATH var in .env
if __name__ == "__main__":
    config = ConfigParser()
    grid = GlobalBaseGrid(config)
    indicator = CliAccumulatedDrought(config=config, grid=grid)
    indicator.run()
