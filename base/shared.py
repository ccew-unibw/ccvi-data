from base.datasets import WorldPopData
from base.objects import Dimension, GlobalBaseGrid
from utils.data_processing import (
    add_time,
    min_max_scaling,
    get_quarter,
    winsorization_normalization,
)


import numpy as np
import pandas as pd


# cannot live in base.objects due to circular import issues with the use of the dataset class
class ExposureDimension(Dimension):
    """Modified Dimension implementing exposure logic for use where applicable.

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
        self.exposure_filename = "exposure"
        self.regenerate_exposure = self.config.get_regeneration_config(
            self.exposure_filename, ["preprocessing"]
        )["preprocessing"]
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
