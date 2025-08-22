import pandas as pd
import numpy as np


import warnings


# TODO: Several of these functions have been produced independent of eacht other doing pretty much the same. We need to consolidate this!
# This is a first try - may have to be extended
def winsorization_normalization(
    data: pd.Series,
    limits: tuple[float, float] | None = None,
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
        minv, maxv = limits[0], limits[1] # type: ignore

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
