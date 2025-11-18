from collections.abc import Generator, Iterable
from datetime import date
import math
from typing import Literal

import country_converter as coco
import numpy as np
import pandas as pd
from panel_imputer import PanelImputer
import pendulum


def add_time(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a column with the FIRST DAY in each quarter as datetime.date to the input df.

    Args:
        df: pd.Dataframe with "year" and "quarter" in columns or index.

    Returns:
        pd.DataFrame: Input df with "time" column.
    """
    time_series = df.reset_index().apply(
        lambda x: date(x["year"], (x["quarter"] * 3 - 2), 1), axis=1
    )
    df["time"] = list(time_series)
    return df


def create_custom_data_structure(
    base_grid: pd.DataFrame, year_start: int, year_end: int, quarterly: bool = True
) -> pd.DataFrame:
    """
    Creates a simple grid-quarter or grid-year data structure. Covers all years fully.

    Args:
        base_grid (pd.DataFrame): Loaded base grid without geometries.
    """
    years = np.arange(year_start, year_end + 1)
    quarters = np.arange(1, 5)

    df = base_grid.copy()
    # assign lists and use explode to create all permutations
    df["year"] = [years for i in df.index]
    if not quarterly:
        df = df.explode("year")
        df = df.set_index("year", append=True)
    else:
        df["quarter"] = [quarters for i in df.index]
        df = df.explode("year").explode("quarter")
        df = df.set_index(["year", "quarter"], append=True)
    return df.sort_index()


def default_impute(
    df: pd.DataFrame | pd.Series,
    time_index: list[str] | str = ["year", "quarter"],
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
        parallelize=True,
        parallel_kwargs={"n_jobs": -2, "verbose": 1},
    )
    df_out: pd.DataFrame = imputer.fit_transform(df)  # type: ignore
    return df_out


def process_yearly_data(
    df_base: pd.DataFrame,
    df_yearly: pd.DataFrame,
    var_list: Iterable[str],
    impute: bool = True,
    level: Literal["country", "grid"] = "country",
) -> pd.DataFrame:
    """
    Simple left merge on the data structure to match existing data based on year and iso3 code.
    Yearly Dataframe is expected to have a multiindex (iso3, year).
    Data ist matched to 4th quarter of the year and imputation via linear interpolation backwards from the last
    observation is done to fill missing values if impute=True.
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
            df_temp = df_temp.set_index("quarter", append=True).sort_index()
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
        raise ValueError(f'Argument "level" needs to be one of ["country", "grid"], got {level}.')

    # fix index
    df_out = df_out.set_index(df_base.index.names).sort_index()
    return df_out


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
    else:
        assert len(limits) == 2

    if fixed_limits:
        minv, maxv = limits[0], limits[1]
    else:
        if ignore_zeroes_limit:
            limits = np.nanquantile(data[data != 0], limits)
        else:
            limits = np.nanquantile(data, limits)
        minv, maxv = limits[0], limits[1]  # type: ignore

    return min_max_scaling(data, minv=minv, maxv=maxv)


def min_max_scaling(
    series: pd.Series, minv: float | None = None, maxv: float | None = None
) -> pd.Series:
    """
    Normalizes a pandas series to values between 0 and 1 based on min and max values.

    Apply upper/lower limits to the data before normalization if desired.
    """
    if minv is None:
        minv = series.min()
    if maxv is None:
        maxv = series.max()

    if minv > series.min():
        series = series.apply(lambda x: minv if x < minv and not np.isnan(x) else x)
    if maxv < series.max():
        series = series.apply(lambda x: maxv if x > maxv and not np.isnan(x) else x)

    return (series - minv) / (maxv - minv)  # type:ignore


def make_iso3_column(df: pd.DataFrame, country_col: str) -> pd.Series:
    """
    Tries to automatically identify iso3 based on the value in country_col and adds as column to input dataframe (df)
    """
    # filter country
    c_dict = dict(
        zip(
            df[country_col].unique(),
            coco.convert(names=df[country_col].unique(), to="ISO3"),
        )
    )
    iso3_series = df[country_col].apply(lambda x: c_dict[x])
    return iso3_series


def slice_tuples(arr_x: np.ndarray, arr_y: np.ndarray) -> Generator[tuple[int, int]]:
    """Returns Generator of all index combinations of two arrays.

    Used to iterate through lists of lats/lons for chunked processing of xr.DataArrays.
    """
    for i in range(len(arr_x)):
        for j in range(len(arr_y)):
            yield i, j


def get_quarter(which: str | int = "current", bounds: Literal["start", "end"] = "start") -> date:
    """
    Return the beginning date or end date of today's quarter as a date object.
    """
    d = date.today()
    # beginning of current quarter
    month_begin = math.ceil(d.month / 3) * 3 - 2
    dt_begin = date(d.year, month_begin, 1)

    if which == "current":
        offset = 0
    elif which == "last":
        offset = -1
    else:
        try:
            offset = int(which)
        except Exception:
            raise ValueError(
                'Argument "which" needs to be one of "current", "last", or convertible to int.'
            )
    # add offset
    pend_begin = pendulum.parse(dt_begin.isoformat())
    pend_modified = pend_begin.add(months=3 * offset)  # type: ignore

    if bounds == "start":
        return date(pend_modified.year, pend_modified.month, pend_modified.day)
    elif bounds == "end":
        pend_end = pend_modified.add(months=3).subtract(days=1)
        return date(pend_end.year, pend_end.month, pend_end.day)
    else:
        raise ValueError(f'Argument "bounds" needs to be in ["start", "end"], got {bounds}.')
