import pandas as pd
import numpy as np

from utils.transformations import winsorization_normalization


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


def aggregate(previous_quarter_12_months, last_quarter, df_anomaly):
    df = df_anomaly.loc[
        ((df_anomaly.quarter >= previous_quarter_12_months) & (df_anomaly.quarter <= last_quarter))
    ]
    df = df[["pgid", "count"]]
    df = df.groupby(["pgid"]).sum().reset_index()
    df["quarter"] = last_quarter
    return df


def aggregate_periods(df_anomaly, acronim, aggregation_level):
    import pandas as pd

    # drop record with na
    df_anomaly = df_anomaly.dropna()

    if aggregation_level == "1yr":
        out = []
        # 1y moving windows aggregations:
        current_quarter, last_quarter = get_current_past_quarter()
        previous_quarter_x_months = last_quarter - 3
        test_quarter_exists = True
        while test_quarter_exists:
            df = aggregate(previous_quarter_x_months, last_quarter, df_anomaly)
            out.append(df)
            last_quarter = last_quarter - 1
            previous_quarter_x_months = last_quarter - 3
            test_quarter_exists = check_quarter(previous_quarter_x_months, df_anomaly)
            print(last_quarter, flush=True)

        out = pd.concat(out)

        return out

    if aggregation_level == "7yr":
        past_years = 7
        out = []
        # 7y moving windows aggregations:
        current_quarter, last_quarter = get_current_past_quarter()
        previous_quarter_x_months = last_quarter - 4 * past_years + 1
        test_quarter_exists = True
        while test_quarter_exists:
            df = aggregate(previous_quarter_x_months, last_quarter, df_anomaly)
            out.append(df)
            last_quarter = last_quarter - 1
            previous_quarter_x_months = last_quarter - 4 * past_years + 1
            test_quarter_exists = check_quarter(previous_quarter_x_months, df_anomaly)
            print(last_quarter, flush=True)

        out = pd.concat(out)
        return out


class NormalizationMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # You can initialize self.indicator_config here if needed

    def _TRANSFORMATION_MAP(self):
        return {
            "log1p": np.log1p,
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
        self, df_indicator: pd.DataFrame, composite_id: str, indicator_config: dict, start_year: int
    ) -> pd.DataFrame:
        # load transformation config
        transformation_func = self._TRANSFORMATION_MAP().get(indicator_config.get("transformation"))
        # load normalization limit
        quantile_normalization_limit = indicator_config.get("normalization_quantile")

        kwargs = quantile_normalization_limit if quantile_normalization_limit else {}

        # Fallback to identity if no transformation
        func = transformation_func if transformation_func else lambda x: x

        # Apply transformation and normalize
        df_indicator[f"{composite_id}"] = winsorization_normalization(
            df_indicator[f"{composite_id}_raw"].apply(func), **kwargs
        )

        # Set index
        df_indicator = (
            df_indicator.reset_index().set_index(["pgid", "year", "quarter"]).sort_index()
        )
        df_indicator = df_indicator.loc[
            (slice(None), slice(start_year, None), slice(None)), slice(None)
        ]
        return df_indicator[[col for col in df_indicator.columns if composite_id in col]].copy()
