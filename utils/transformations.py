import pandas as pd
import numpy as np

from scipy.stats import iqr
from typing import List
from panel_imputer import PanelImputer

import warnings


# TODO: Several of these functions have been produced independent of eacht other doing pretty much the same. We need to consolidate this!
# This is a first try - may have to be extended
def winsorization_normalization(
    data: pd.Series,
    limits: tuple[float] = None,
    ignore_zeroes_limit: bool = False,
    fixed_limits: bool = False,
) -> pd.Series:
    """
    Normalizes a pandas series to values between 0 and 1 with optional winsorization.
    """
    if limits is None:
        return min_max_scaling(data)

    if fixed_limits:
        minv, maxv = limits[0], limits[1]
    else:
        if ignore_zeroes_limit:
            limits = np.nanquantile(data[data != 0], limits)
        else:
            limits = np.nanquantile(data, limits)
        minv, maxv = limits[0], limits[1]

    return min_max_scaling(data, minv=minv, maxv=maxv)


def min_max_scaling(series: pd.Series, minv: float = None, maxv: float = None) -> pd.Series:
    """
    Normalizes a pandas series to values between 0 and 1 based on min and max values.


    """
    if minv is None:
        minv = series.min()
    if maxv is None:
        maxv = series.max()

    if minv > series.min():
        series = series.apply(lambda x: minv if x < minv else x)
        warnings.warn(
            f"minv inside series range provided. Applying threshold to series at minv={minv}",
            UserWarning,
        )
    if maxv < series.max():
        series = series.apply(lambda x: maxv if x > maxv else x)
        warnings.warn(
            f"maxv inside series range provided. Applying threshold to series at maxv={maxv}",
            UserWarning,
        )

    return (series - minv) / (maxv - minv)


def winsorize_iqr(x: pd.Series, remove_0: bool = False) -> pd.Series:
    """
    Winsorizes a pandas series using the IQR method.
    """
    if remove_0:
        x_ = x[x > 0]
    else:
        x_ = x
    iqr_ = iqr(x_)
    q1, q3 = np.quantile(x_, [0.25, 0.75])
    minv, maxv = q1 - 1.5 * iqr_, q3 + 1.5 * iqr_
    return np.clip(x, a_min=minv, a_max=maxv)


def default_impute(
    df: pd.DataFrame,
    time_index: List[str] | str = ["year", "quarter"],
    location_index: str = "pgid",
) -> pd.DataFrame:
    """
    Default data imputation for the index based on the following rules:
    - linear interpolation between data points
    - extrapolate backwards
    - don't fill forwards

    Returns imputed version of input dataframe (df)
    """
    imputer = PanelImputer(
        time_index=time_index,
        location_index=location_index,
        imputation_method="interpolate",
        interp_method="slinear",
        tail_behavior=["extrapolate", "None"],
        parallel_kwargs={"n_jobs": 16, "verbose": 1},
    )
    df_out = imputer.fit_transform(df)
    return df_out


def quantile_avg(df: pd.DataFrame, column: str, window: int = 2, q: float = 0.99) -> pd.Series:
    """
    Returns rolling mean of quantile value of a column of the df for each quarter.
    Default time window = 2 years
    Default quantile is set to 99% of NON-ZERO(!) observations
    """
    window = 4 * window
    quantiles = df.groupby("time")[column].quantile(q)
    return quantiles.rolling(window=window).mean().rename("threshold")


def process_yearly_data(
    df_base: pd.DataFrame,
    df_yearly: pd.DataFrame,
    var_list: List[str],
    impute=True,
    level="country",
) -> pd.DataFrame:
    """
    Simple left merge on the data structure to match existing data based on year and iso3 code.
    Yearly Dataframe is expected to have a multiindex (iso3, year).
    Data ist matched to 4th quarter of the year and imputation via linear interpolation backwards from the last
    observation is done to fill missing values when impute=True.
    """
    # always match yearly data with quarter 4
    df_yearly = df_yearly.copy()  # don't modify this df
    df_yearly["quarter"] = 4
    df_yearly = df_yearly.set_index("quarter", append=True)

    if level == "country":
        if impute:
            # for performance reasons, first build a quarter - year version and impute on the country-level
            df_temp = pd.DataFrame(
                index=df_yearly.index, columns=df_yearly.columns, dtype=np.float64
            ).reset_index(level="quarter")
            df_temp["quarter"] = [[1, 2, 3, 4] for i in range(len(df_temp))]
            df_temp = df_temp.explode("quarter")
            df_temp = df_temp.set_index("quarter", append=True)
            df_temp.update(df_yearly)
            df_yearly = default_impute(df_temp, location_index="iso3")

        df_out = pd.merge(
            df_base.reset_index(),
            df_yearly[var_list].reset_index(),
            how="left",
            on=["iso3", "year", "quarter"],
        )
    elif level == "grid":
        df_out = pd.merge(
            df_base.reset_index(),
            df_yearly[var_list].reset_index(),
            how="left",
            on=["iso3", "year", "quarter"],
        )

        if impute:
            df_out = default_impute(df_out)
    else:
        raise ValueError('Argument "level" needs to be one of ["country", "grid"]')

    # fix index
    df_out = df_out.set_index(df_base.index.names).sort_index()
    return df_out
