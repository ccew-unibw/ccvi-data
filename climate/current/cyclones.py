import pandas as pd
from utils.index import get_quarter

from base.datasets.ibtrak_cyclones import IBTRAKData
from base.objects import Indicator, ConfigParser, GlobalBaseGrid

from climate.shared import aggregate_periods, NormalizationMixin


class CliCurrentCyclones(Indicator, NormalizationMixin):
    def __init__(
        self,
        config: ConfigParser,
        grid: GlobalBaseGrid,
        pillar: str = "CLI",
        dim: str = "current",
        id: str = "cyclones",
    ):
        """Params defining indicator's place in index set to designed hierarchy by default"""
        self.ibtrak = IBTRAKData(local=False, config=config)
        super().__init__(pillar=pillar, dim=dim, id=id, config=config, grid=grid)

    def load_data(self) -> pd.DataFrame:
        df = self.ibtrak.load_data()
        return df

    def preprocess_data(self, df_ibtrak: pd.DataFrame) -> pd.DataFrame:
        fp_preprocessed = self.ibtrak.storage.build_filepath(
            "processing", filename="ibtrak_cyclones_preprocessed"
        )
        try:
            fp_preprocessed = pd.read_parquet(fp_preprocessed)
            last_quarter_date = get_quarter("last")

            if fp_preprocessed["time"].max().date() < last_quarter_date:
                raise FileNotFoundError
            return fp_preprocessed

        except FileNotFoundError:
            print("-- df_base creation ...")
            df_base = self.create_base_df(start_year=df_ibtrak["YEAR"].min())
            print("-- create_grid_quarter_aggregates ...")
            df_preprocessed = self.ibtrak.create_grid_quarter_aggregates(df_base, df_ibtrak)

            return df_preprocessed

    def create_indicator(self, df_preprocessed: pd.DataFrame) -> pd.DataFrame:
        # make the 1y indicator aggregation

        df_preprocessed["quarter"] = df_preprocessed["time"].dt.to_period("Q")
        df_preprocessed = aggregate_periods(df_preprocessed, self.id, "1yr")
        df_preprocessed["year"] = df_preprocessed["quarter"].dt.year
        df_preprocessed["quarter"] = df_preprocessed["quarter"].dt.quarter

        df_preprocessed.rename(columns={"count": f"{self.composite_id}_raw"}, inplace=True)

        return df_preprocessed

    def normalize(self, df_indicator: pd.DataFrame) -> pd.DataFrame:
        """Standardized normalization via ClimateMixin"""
        df_normalized = self.climate_normalize(
            df_indicator, self.composite_id, self.indicator_config, self.global_config["start_year"]
        )
        return df_normalized

    def add_raw_indicator(
        self, df_indicator: pd.DataFrame, df_preprocessed: pd.DataFrame
    ) -> pd.DataFrame:
        return df_indicator


# this is possible by adding the root folder as the PYTHONPATH var in .env
if __name__ == "__main__":
    config = ConfigParser()
    grid = GlobalBaseGrid(config)
    indicator = CliCurrentCyclones(config=config, grid=grid)
    indicator.run()
