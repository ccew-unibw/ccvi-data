import pandas as pd
import numpy as np

from utils.data_processing import add_time, min_max_scaling
from utils.index import get_quarter
from utils.transformations import winsorization_normalization
from base.objects import Dimension, GlobalBaseGrid
from base.datasets import WorldPopData


class ClimateDimension(Dimension):
    """Modified Dimension for the climate pillar implementing the exposure logic.

    Adds configuration and requirements for WorldPop based population density
    exposure layer.

    Implements `add_exposure()` to replace generate the exposure layer and
    combine this with the compoment indicators.

    Attrs:
        console (Console): Shared console instance for output.
        pillar (str): The pillar ID of the dimension.
        dim (str): The ID of the dimension.
        components (List[Indicator]): List of initialized Indicator instances
            that feed into this dimension.
        has_exposure (bool): Flag indicating if an implementation of `add_exposure()`
            exists and should be called. Set to True.
        config (ConfigParser): The ConfigParser instance used for initialization.
        global_config (dict[str, Any]): Global configuration settings dictionary.
        aggregation_config (Dict[str, Any]): Dimension-specific aggregation config.
            Loaded via the ConfigParser.
        storage (StorageManager): Instance managing storage for this dimension.
        composite_id (str): Composite identifier ("pillar_dim") for this dimension.
        generated (bool): Flag indicating if the dimension output file exists and
            is up-to-date for the current/last quarter, checked at init.
        data_recency (dict[str, str]): Dictionary mapping indicator composite IDs
            to their last updated quarter string ('YYYY-Q'), set during `run()`.
        grid (GlobalBaseGrid): An initialized GlobalBaseGrid instance.
        wo (WorldPopData): An WorldPopData dataset instance
        exposure_filename (str): Filename under which the exposure layer is stored.
        regenerate_exposure (bool): Whether to regenerate the exposure layer if it
            already exists. Based on the regeneration config: preprocessing section.
        exposure_generated (bool): Flag indicating whether an up to date version
            of the exposure layer is available in storage.
    """

    def __init__(self, grid: GlobalBaseGrid, *args, **kwargs):
        """Initializes the ClimateDimension.

        Initializes the Dimension superclass, setting `has_exposure` to True,
        initializes the `WorldPopData` instance, and sets the flags used to
        determine if the exposure layer needs regeneration based on config and
        existing cached files.

        Args:
            grid (GlobalBaseGrid): An initialized GlobalBaseGrid instance.
            config (ConfigParser): An initialized ConfigParser instance.
            *args: Additional arguments passed to the parent `Dimension` constructor.
            **kwargs: Additional keyword arguments passed to the parent `Dimension`
                constructor.
        """
        super().__init__(*args, has_exposure=True, **kwargs)
        self.grid = grid
        self.wp = WorldPopData(config=self.config)
        self.exposure_filename = "climate_exposure"
        self.regenerate_exposure = self.config.get_regeneration_config(self.exposure_filename)[
            "preprocessing"
        ]
        self.exposure_generated = self.storage.check_component_generated(
            filename=self.exposure_filename
        )

    def add_exposure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replaces indicator data with versions combined with exposure.

        This method first creates or loads the exposure via `_create_exposure_layers()`.
        It then joins this exposure layer ('EXP_pop_density') to the input DataFrame
        calculates an exposure version by multiplying the indicator score with
        the exposure value, applying a log transformation and re-normalizing the
        result using winsorization and min-max scaling.

        Args:
            df (pd.DataFrame): Combined DataFrame from `load_components()`, containing
                the indicator scores for this dimension, indexed by
                ('pgid', 'year', 'quarter'). Columns are indicator composite IDs.

        Returns:
            pd.DataFrame: A DataFrame containing only the exposure-adjusted
                indicator columns, named with an '_exposure' suffix (e.g.,
                'pillar_dim_id_exposure').
        """
        exposure_layer = "EXP_pop_density"
        # setup Worldpop
        df_exp = self._create_exposure_layers()
        df = df.join(df_exp[exposure_layer], how="left")
        for i in self.components:
            composite_id = i.composite_id
            normalization_kwargs = {"limits": [0, 0.999], "ignore_zeroes_limit": True}
            # simple multiplication, log and re-winsorization_normalization
            df[f"{composite_id}_exposure"] = (df[composite_id] * df[exposure_layer]).apply(np.log1p)
            df[f"{composite_id}_exposure"] = winsorization_normalization(
                df[f"{composite_id}_exposure"], **normalization_kwargs
            )
            # again, sanity check - need to drop where exposure is na to get same results
            assert np.array_equal(
                df.dropna(subset=exposure_layer)[composite_id].isna().values,  # type: ignore
                df.dropna(subset=exposure_layer)[f"{composite_id}_exposure"].isna().values,  # type: ignore
            )
        # only return exposure columns
        df = df[[f"{i.composite_id}_exposure" for i in self.components]]
        return df

    def _create_exposure_layers(self) -> pd.DataFrame:
        """Creates or loads the cached population density exposure layer.

        If regeneration is not forced and an up-to-date cached exposure layer
        ('climate_exposure.parquet') exists in the dimension's processing storage,
        it's loaded. Otherwise, this method generates the exposure layer. Yearly
        gridded WorldPop population data is interpolated to quarterly frequency.
        Performs normalization of the exposure layer via log-transformation and
        winsorization. Saves the resulting DataFrame to the processing storage.

        Returns:
            pd.DataFrame: A DataFrame indexed by ('pgid', 'year', 'quarter')
                containing the exposure layer ('EXP_pop_density') and associated
                raw values.
        """
        if not self.regenerate_exposure and self.exposure_generated:
            df = self.storage.load(filename=self.exposure_filename)
        else:
            # Load Worldpop pop data
            self.wp.load_data()
            df = self.wp.process_yearly_grid_aggregates(self.grid)
            df = self.wp.get_quarterly_interpolations(df, self.grid, self.regenerate_exposure)
            # limit to last quarter for this output - no need to be more complex since wp is always yearly as it is an estimate
            last_quarter_start = get_quarter("last")
            # check that we have the required data
            assert df.index.get_level_values("year").max() >= last_quarter_start.year
            df = add_time(df)
            df = df.loc[df.time <= last_quarter_start]
            df["pop_density_log"] = df["pop_density"].apply(np.log1p)
            threshold = df.loc[
                (slice(None), slice(None, 2020), slice(None)), "pop_density_log"
            ].quantile(0.99)
            df["EXP_pop_density"] = min_max_scaling(df["pop_density_log"], maxv=threshold)
            df["EXP_pop_density_raw"] = df["pop_density"]
            df["EXP_pop_count_raw"] = df["pop_count"]
            df = df[[col for col in df.columns if "EXP" in col]]
            self.storage.save(df, filename=self.exposure_filename)
        return df


def get_last_completed_quarter():
    from datetime import datetime, timedelta

    # Get the current date
    current_date = datetime.now()

    # Calculate the previous quarter
    previous_quarter = current_date - timedelta(days=365 / 4)

    # Format the result as "YYYYQn"
    result = f"{previous_quarter.year}Q{((previous_quarter.month - 1) // 3) + 1}"

    return result


def check_quarter(period, df):
    if period in df.quarter.unique():
        return True
    else:
        return False


def get_current_past_quarter():
    import pandas as pd

    # Get the current date
    current_date = pd.to_datetime("today")

    # Get the current quarter
    current_quarter = pd.Period(current_date, freq="Q")

    # Get the previous quarter
    previous_quarter = pd.Period(current_date, freq="Q") - 1

    return current_quarter, previous_quarter


def get_days_between_quarters(start_quarter, end_quarter):
    import pandas as pd

    # Extract year and quarter information
    start_year, start_quarter_num = int(start_quarter[:4]), int(start_quarter[5])
    end_year, end_quarter_num = int(end_quarter[:4]), int(end_quarter[5])

    # Create start and end dates
    start_date = pd.to_datetime(f"{start_year}-01-01") + pd.offsets.QuarterBegin(
        start_quarter_num - 1
    )
    end_date = pd.to_datetime(f"{end_year}-01-01") + pd.offsets.QuarterEnd(end_quarter_num)

    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Create DataFrame
    df = pd.DataFrame({"date": date_range})

    return df


def aggregate_periods(df_anomaly, acronim, aggregation_level):
    if aggregation_level == "1yr":
        df_anomaly[f"count"] = (
            df_anomaly.groupby("pgid")["count"].rolling(window=4).sum().reset_index(0, drop=True)
        )
        df_anomaly.fillna(0, inplace=True)
        return df_anomaly

    if aggregation_level == "7yr":
        df_anomaly[f"count"] = (
            df_anomaly.groupby("pgid")["count"].rolling(window=4).sum().reset_index(0, drop=True)
        )
        df_anomaly.fillna(0, inplace=True)
        df_anomaly[f"count"] = (
            df_anomaly.groupby("pgid")["count"]
            .rolling(window=4 * 7)
            .mean()
            .reset_index(0, drop=True)
        )
        df_anomaly.fillna(0, inplace=True)

        return df_anomaly


class NormalizationMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # You can initialize self.indicator_config here if needed

    def _TRANSFORMATION_MAP(self):
        return {
            "log1p": np.log1p,
            "exp": np.exp,
            "custom_zero_if_negative": self._custom_zero_if_negative,
            None: None,
            "null": None,
        }

    @staticmethod
    def _custom_zero_if_negative(x: float) -> float:
        if x > 0 or np.isnan(x):
            return x
        else:
            return 0

    def climate_normalize(
        self,
        df_indicator: pd.DataFrame,
        composite_id: str,
        indicator_config: dict,
        start_year: int,
        normalize_raw_col: bool = True,
    ) -> pd.DataFrame:
        # load transformation config
        transformation_func = self._TRANSFORMATION_MAP().get(indicator_config.get("transformation"))
        # load normalization limit
        quantile_normalization_limit = indicator_config.get("normalization_quantile")

        kwargs = quantile_normalization_limit if quantile_normalization_limit else {}

        # Fallback to identity if no transformation
        func = transformation_func if transformation_func else lambda x: x

        # which col contains the value to use for normalization
        if normalize_raw_col:
            norm_col = f"{composite_id}_raw"
        else:
            norm_col = composite_id

        # Apply transformation and normalize
        df_indicator[f"{composite_id}"] = winsorization_normalization(
            df_indicator[norm_col].apply(func), **kwargs
        )

        # Set index
        df_indicator = (
            df_indicator.reset_index().set_index(["pgid", "year", "quarter"]).sort_index()
        )
        df_indicator = df_indicator.loc[
            (slice(None), slice(start_year, None), slice(None)), slice(None)
        ]
        return df_indicator[[col for col in df_indicator.columns if composite_id in col]].copy()
