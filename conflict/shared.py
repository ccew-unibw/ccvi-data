from functools import cache
import math
import numpy as np
import pandas as pd

from base.objects import Indicator


# pillar-specific standardizations via Mixin class here
class NormalizationMixin:
    def conflict_normalize(
        self, df_indicator: pd.DataFrame, composite_id: str, quantile: float
    ) -> pd.DataFrame:
        thresholds = self._quantile_avg(df_indicator, composite_id, q=quantile)
        indicator = self._limit_col(df_indicator, composite_id, thresholds)
        years_limited = []
        for year in indicator.index.get_level_values("year").unique():
            part = indicator.xs(year, level="year", drop_level=False)
            minv_temp = part.min()
            maxv_temp = part.max()
            part_limited = (part - minv_temp) / (maxv_temp - minv_temp)
            years_limited.append(part_limited)
        indicator_normalized = pd.concat(years_limited).sort_index()
        df_indicator[composite_id] = indicator_normalized
        return df_indicator[[col for col in df_indicator.columns if composite_id in col]].copy()

    # Normalization
    def _limit_col(self, df: pd.DataFrame, column: str, thresholds: pd.Series) -> pd.Series:
        """
        Sets upper limits to data based on thresholds for the specified columns to apply the rolling window-based thresholds.
        Assigns threshold value to all datapoints with values above the threshold.
        Returns modified Series
        """
        df_temp = pd.merge(df, thresholds, how="left", left_on="time", right_on="time")
        limited = df_temp.apply(
            lambda x: x[column]
            if x[column] <= x["threshold"] or np.isnan(x[column])
            else x["threshold"],
            axis=1,
        )
        limited = pd.Series(index=df.index, data=limited.values)
        return limited

    def _quantile_avg(
        self, df: pd.DataFrame, column: str, window: int = 8, q: float = 0.99
    ) -> pd.Series:
        """
        Returns rolling mean of quantile value of a column of the df for each quarter.
        Default time window: 8 quarters == 2 years(!)
        Default quantile is set to 99% of NON-ZERO(!) observations
        """
        # q(uantile) of non-zero values means 1-(share of non-zero * (1-q))
        # we discard NAs in the calculation of the share of non-zero here - since they signify missing coverage
        q = 1 - (df[column] != 0).where(df[column].notna()).mean() * (1 - q)
        quantiles = df.groupby("time")[column].quantile(q)
        return quantiles.rolling(window=window).mean().rename("threshold")


class PersistenceMixin:
    def validate_indicator_input(self, indicator: Indicator, indicator_class: type):
        """Validate the base indicator input.

        Ensures that a base indicator used in the persistence dimension has the correct
        type and verifies that each indicator has been generated.
        Runs the indicators if it has not yet been generated or regenerate for the
        indicator is True.

        Args:
            indicator (Indicator): Indicator instance to validate.
            indicator_class (type): Required type for the indicator.
        """
        assert isinstance(indicator, indicator_class)
        if indicator.regenerate["indicator"]:
            indicator.run()
        else:
            try:
                assert indicator.generated
            except AssertionError:
                print("base indicator not yet generated, running")
                indicator.run()

    def decay_based_persistence(
        self,
        series: pd.Series,
        decay_target: float,
        timeframe: int,
        value_weight: float,
        name: str | None,
    ) -> pd.Series:
        """Applies a custom decay model representing conflict intensity, simulating
        the persistence of past conflict over time with exponential decay.

        Calculates persistence cell-by-cell and concatenates the results. For detailed
        information on the decay logic, see `utils.conflict.conflict_decay_cell`.

        Args:
            series (pd.Series): A time series of conflict intensity values between 0 and 1.
            decay_target (float): Target proportion of the original value remaining
                after the full timeframe. Must be greater than 0 and less than 1.
            timeframe (int): The number of years after which the decay should reach
                the target proportion of the original value.
            value_weight (float): A weighting factor applied to the intensity of current
                violence (`x`) if it occurs to combine the new violence's impact with
                the decayed influence of past violence. It's used in the formula:
                    `new_value = x * value_weight * (1 - decay_val) + decay_val`
            name (str): Name to assign to the output Series.

        Returns:
            pd.Series: A new Series with the same index as the input `series`
                containing the calculating persistence values.
        """
        if name is None:
            name = f"{series.name}_history_decay"
        series = series.copy().sort_index()
        series_list: list[pd.Series] = []
        for pgid in series.index.get_level_values("pgid").unique():
            df_cell = series.xs(pgid, level="pgid", drop_level=False).copy()
            cell_series = self._conflict_decay_cell(
                df_cell,
                decay_target=decay_target,
                timeframe=timeframe,
                value_weight=value_weight,
                name=name,
            )
            series_list.append(cell_series)
        series_history = pd.concat(series_list)
        return series_history

    def _conflict_decay_cell(
        self,
        series: pd.Series,
        decay_target: float,
        timeframe: int,
        value_weight: float,
        name: str = "decay",
    ) -> pd.Series:
        """
        Applies a custom decay model to a time series representing conflict intensity,
        simulating the persistence of past conflict over time with exponential decay.

        The function assumes the initial value in the series is NaN, representing the
        start of the observation window. Once data begins, conflict intensity decays
        over time unless a new non-zero value is observed, in which case the new value
        is blended with the decayed value using a weighted average. The decay rate is
        calculated such that after `timeframe` years with no new events, the value decays
        to the `decay_target` fraction of the value at the start of that period.

        Args:
            series (pd.Series): A time series of conflict intensity values between 0 and 1,
                expected to start with NaN.
            decay_target (float): Target proportion of the original value remaining
                after the full timeframe. Must be greater than 0 and less than 1.
            timeframe (int): The number of years after which the decay should reach
                the target proportion of the original value.
            value_weight (float): A weighting factor applied to the intensity of current
                violence (`x`) if it occurs to combine the new violence's impact with
                the decayed influence of past violence. It's used in the formula:
                    `new_value = x * value_weight * (1 - decay_val) + decay_val`
            name (str, optional): Name to assign to the output Series. Defaults to 'decay'.

        Returns:
            pd.Series: A new Series with the same index as the input `series`
                containing the values calculated through applying the decay logic.
        """

        @cache
        def get_decay_val(value: float, time_elapsed: int, decay_constant: float) -> float:
            """
            Exponential decay function: Nt = N0 * e^(-lambda * t)
            (https://en.wikipedia.org/wiki/Exponential_decay)

            Used nested to avoid caching self.

            Args:
                value (float): N0 (initial value)
                time_elapsed (int): t (timesteps elapsed)
                decay_constant (float): -lambda (decay constant)
            """
            decay_val = value * math.exp(time_elapsed * decay_constant)
            return decay_val

        # decay_constant chosen so the risk after time has elapsed is the target percentage of the conflict intensity
        decay_constant = -1 * math.log(1 / decay_target) / (timeframe * 4)
        # initialization
        # start value
        value = series.iloc[0]
        assert np.isnan(
            value
        )  # this is always the case with the variables, but added the check anyways in case something changes, as this would break the code below
        last_value = None
        time_since_conflict = 0
        output_list = []
        for x in series:
            if np.isnan(value):  # before the coverage starts
                value = x
                if not np.isnan(x):
                    last_value = x  # 0 if no conflict
                    if x != 0:
                        time_since_conflict = 0
            else:
                time_since_conflict += 1
                if x == 0:
                    # decay value for 0 returns 0
                    value = get_decay_val(last_value, time_since_conflict, decay_constant)
                elif x != 0:
                    decay_val = get_decay_val(last_value, time_since_conflict, decay_constant)
                    # emphasizing decay value (previous conflict) over new events
                    value = x * value_weight * (1 - decay_val) + decay_val
                    # saving the last value and resetting time var as the starting point for decay without new events
                    last_value = value
                    time_since_conflict = 0
            output_list.append(value)
        return pd.Series(data=output_list, index=series.index, name=name)
