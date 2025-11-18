import math
from functools import cache

import numpy as np
import pandas as pd
import xarray as xr


def ccvi_series_to_xarray(series: pd.Series) -> xr.DataArray:
    """
    Converts pd series with pgid for one timestep to xarray in lat/lon format.
    """
    temp = series.reset_index()
    temp["coords"] = temp.pgid.apply(pgid_to_coords)
    temp["lat"] = temp["coords"].apply(lambda x: x[0])
    temp["lon"] = temp["coords"].apply(lambda x: x[1])
    da = temp.set_index(["lat", "lon"])[series.name].to_xarray().astype(float)
    # fill missing all-NA columns leading to plotting artifacts
    lats = np.arange(da.lat.min(), da.lat.max() + 0.1, 0.5)
    lons = np.arange(da.lon.min(), da.lon.max() + 0.1, 0.5)
    da = da.reindex({"lat": lats, "lon": lons})
    return da


@cache
def pgid_to_coords(id: int) -> tuple[float, float]:
    """
    Converts PRIO-GRID cell id to cell center lat/lon coordinates.
    """
    id -= 1
    lat = math.floor(id / 720) / 2 - 89.75
    lon = id % 720 / 2 - 179.75
    return lat, lon


@cache
def coords_to_pgid(lat: float, lon: float) -> int:
    """
    Converts cell center lat/lon coordinates to PRIO-GRID cell id.
    """
    row = (lat + 90.25) * 2
    col = (lon + 180.25) * 2
    pgid = (row - 1) * 720 + col
    return int(pgid)
