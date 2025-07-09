import pandas as pd
from utils.index import get_quarter

from base.datasets.spei import SPEIData
from base.objects import Indicator, ConfigParser, GlobalBaseGrid

from climate.shared import NormalizationMixin


class CliCurrentDrought(Indicator, NormalizationMixin):
    def __init__(
        self,
        config: ConfigParser,
        grid: GlobalBaseGrid,
        pillar: str = "CLI",
        dim: str = "current",
        id: str = "drought",
    ):
        """Params defining indicator's place in index set to designed hierarchy by default"""
        self.event_data = SPEIData(local=False, config=config)
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
            df_base = self.create_base_df(start_year=df_event_data["date"].dt.year.min())
            print("-- create_grid_quarter_aggregates ...")
            df_preprocessed = self.event_data.create_grid_quarter_aggregates(df_base, df_event_data)

            return df_preprocessed

    def create_indicator(self, df_preprocessed: pd.DataFrame) -> pd.DataFrame:
        from base.datasets.spei import process_current_drought

        # make the 1y indicator aggregation
        storage = self.event_data.storage.storage_paths["processing"]
        spei_fp = self.event_data.storage.build_filepath("processing", filename="preprocessed")

        spei_current = process_current_drought(storage, spei_fp, "")

        df = df_preprocessed[["pgid", "lat", "lon"]].drop_duplicates()

        spei_current = spei_current.merge(df, on="pgid", how="left")

        spei_current = spei_current[["pgid", "year", "quarter", "lat", "lon", "count"]]
        spei_current.rename(columns={"count": f"{self.composite_id}_raw"}, inplace=True)

        return spei_current

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
    indicator = CliCurrentDrought(config=config, grid=grid)
    indicator.run()
