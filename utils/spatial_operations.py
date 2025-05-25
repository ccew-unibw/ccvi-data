from collections.abc import Iterable
from fractions import Fraction
from functools import cache
import math

from joblib import Parallel, delayed
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
from tqdm import tqdm


def add_lat_lon(df) -> pd.DataFrame:
    """
    Adds lat and lon coordinates to a given dataframe with a "pgid" column.
    """
    assert "pgid" in df.columns, "DatFrame must have a 'pgid' column for merging."
    base_grid = pd.read_parquet("output/geo/base_grid_prio.parquet")
    return df.merge(base_grid[["lat", "lon"]], on="pgid", how="left")


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


def round_grid(x, s=0.5) -> float:
    """
    Round coordinate to nearest s°*s° grid cell centroid coordinate.
    """
    x -= s / 2
    x = round(x * pow(s, -1)) / pow(s, -1)
    return x + s / 2


def s_floor(x: float, s: float = 0.5) -> Fraction:
    """rounds down to the nearest multiple of s using Fractions to avoid floating point errors"""
    sf = Fraction(str(s))
    xf = Fraction(str(x))
    xf_ = math.floor(xf * sf**-1)
    return xf_ * sf


def s_ceil(x: float, s: float = 0.5) -> Fraction:
    """rounds up to the nearest multiple of s using Fractions to avoid floating point errors"""
    sf = Fraction(str(s))
    xf = Fraction(str(x))
    xf_ = math.ceil(xf * sf**-1)
    return xf_ * sf


# Spatial diffusion
def adjust_dateline_neighbors(
    pgid: int, simple_neighbors: list[int], grid_width: int = 720
) -> list[int]:
    """Adjusts neighbor IDs for dateline wrapping in the PRIO-GRID.

    Corrects neighbor IDs calculated via simple arithmetic when the central
    cell (`pgid`) is on the left or right edge of the grid.

    Args:
        pgid (int): The ID of the central grid cell.
        simple_neighbors (List[int]): The list of neighbor IDs generated using
            simple arithmetic (e.g., pgid +/- 1, etc.).
        grid_width (int): The number of cells along the longitude dimension.
            Defaults to the PRIO-GRID's 720.

    Returns:
        List[int]: The list of neighbor IDs with corrections applied for
            dateline wrapping.
    """

    adjusted_neighbors = []

    pgid_mod = pgid % grid_width
    if pgid_mod == 1:  # Center cell is on the LEFT edge (e.g., pgid=1, 721, ...)
        # Neighbors whose simple calculation would incorrectly land on the right edge
        # (modulo 0) need to be wrapped back to the *correct* right edge neighbor ID
        # relative to the central cell's row.
        for n in simple_neighbors:
            if n % grid_width == 0:
                adjusted_neighbors.append(n + grid_width)
            else:
                adjusted_neighbors.append(n)

    elif pgid_mod == 0:  # Center cell is on the RIGHT edge (e.g., pgid=720, 1440, ...)
        # Neighbors whose simple calculation would incorrectly land on the left edge
        # (modulo 1) need to be wrapped back to the *correct* left edge neighbor ID
        # relative to the central cell's row.
        for n in simple_neighbors:
            if n % grid_width == 1:
                adjusted_neighbors.append(n - grid_width)
            else:
                adjusted_neighbors.append(n)
    else:
        # Center cell is not on a vertical edge, everything is fine.
        adjusted_neighbors = simple_neighbors

    return adjusted_neighbors


@cache
def get_neighboring_cells(pgid: int, keep_center: bool = False) -> list[int]:
    """
    Returns all directly neighboring grid cells together with or without the cell itself.

    [ ][ ][ ][ ][ ]
    [ ][X][X][X][ ]
    [ ][X][O][X][ ]
    [ ][X][X][X][ ]
    [ ][ ][ ][ ][ ]

    Args:
        pgid (int): The priogrid id of the cell.
        keep_center (bool): Wether to include the cell itself.

    Returns:
        list(int): List of grid cell ids.
    """

    neighborhood = [
        pgid - 1,
        pgid + 1,
        pgid - 720,
        pgid + 720,
        (pgid - 720) - 1,
        (pgid - 720) + 1,
        (pgid + 720) - 1,
        (pgid + 720) + 1,
    ]
    neighborhood = adjust_dateline_neighbors(pgid, neighborhood)
    if keep_center:
        neighborhood.append(pgid)
    return neighborhood


def calculate_smoothed_value(
    pgid: int, df: pd.DataFrame, var: str, na_eq_zero: bool
) -> tuple[float, float]:
    """
    Calculates the mean and sum of a specified variable for a grid cell and its
    (up to 8) neighboring cells.

    This function calculates the individual values for spatial smoothing. It takes a
    grid cell ID and computes the mean and sum of the specified variable across
    the cell and its surrounding neighbors. If some neighboring cells are missing
    (e.g., ocean cells), they can be optionally treated as zeroes.

    Args:
        pgid (int): The ID of the central grid cell.
        df (pd.DataFrame):  DataFrame slice for a single time point, indexed by
            grid cell IDs (`pgid`), containing the variable `var`.
        var (str): The name of the column in `df` to compute a smoothed value for.
        na_eq_zero (bool): If True (default), treat missing neighbors
            within the 3x3 grid as having a value of 0 for calculations. If
            False, calculations only include cells present in `df`.

    Returns:
        tuple[float, float]: A tuple containing:
            - The mean value of the variable across the the 3x3 neighborhood.
            - The sum of the variable across the the 3x3 neighborhood.
    """
    # get cell location indices
    buffer_index = get_neighboring_cells(pgid, keep_center=True)
    # only use those we actually have in the data (accounting for oceans etc.)
    buffer_index_valid = df.index.intersection(buffer_index)
    # select data and calculate mean value
    buffer_values = df.loc[buffer_index_valid][var].values.astype(float)
    if all(np.isnan(buffer_values)) or pgid not in buffer_index_valid:
        return np.nan, np.nan
    if len(buffer_values) < 9 and na_eq_zero:
        fill_zeroes = np.zeros(9 - len(buffer_values))
        buffer_values = np.append(buffer_values, fill_zeroes)
    buffer_mean = np.nanmean(buffer_values, dtype=float)
    buffer_sum = np.nansum(buffer_values)
    return buffer_mean, buffer_sum


def create_diffusion_layers(
    df: pd.DataFrame, base_cols: Iterable[str], na_eq_zero: bool = True
) -> pd.DataFrame:
    """Computes spatial diffusion layers (mean/sum-based) for variables across time.

    For each column in `base_cols`, this function computes spatially smoothed
    versions (mean and sum) using `smoothed_diffusion`, based on the values of
    each grid cell and its surrounding neighbors at each quarterly timestep. The
    results are returned as a `df` with two columns with the suffixes
    '_diffusion_mean' and '_diffusion_sum' for each base column.

    Args:
        df (pd.DataFrame): Input DataFrame containing time series data with the
            CCVI ['pgid', 'year', 'quarter'] MultiIndex. It must also have a 'time'
            column (datetime-like) from which year and quarter can be derived.
        base_cols (List[str]): A list of column names present in `df` for which
            diffusion layers (neighborhood mean and sum) should be calculated.
        na_eq_zero (bool, optional): If True (default), treat missing neighbors
            within the 3x3 grid as having a value of 0 for calculations. If
            False, calculations only include cells present in `df`.

    Returns:
        pd.DataFrame: A DataFrame containing only the newly generated diffusion
            columns (e.g., 'var_diffusion_mean', 'var_diffusion_sum') and the
            CCVI ['pgid', 'year', 'quarter'] MultiIndex.
    """

    def par_func_diffusion(t):
        df_temp: pd.DataFrame = df.xs(
            (t.year, math.ceil(t.month / 3)), level=("year", "quarter")
        ).copy()  # type: ignore
        for var in base_cols:
            mean_sum = [
                calculate_smoothed_value(pgid, df_temp, var, na_eq_zero) for pgid in df_temp.index
            ]
            mean, sum = zip(*mean_sum)
            df_temp[f"{var}_diffusion_mean"] = mean
            df_temp[f"{var}_diffusion_sum"] = sum
        df_temp["quarter"] = math.ceil(t.month / 3)
        df_temp["year"] = t.year
        df_temp = df_temp.set_index(["year", "quarter"], append=True)
        return df_temp

    df = df.sort_index().copy()
    dfs = Parallel(n_jobs=-1)(delayed(par_func_diffusion)(t) for t in df.time.unique())
    df = pd.concat(dfs)
    return df[[col for col in df.columns if "diffusion" in col]]


def assign_areas_to_grid(
    grid: gpd.GeoDataFrame,
    fp_areas: str,
    grid_col: str,
    area_col: str,
    grid_size: float = 0.5,
    performance: bool = False,
) -> gpd.GeoDataFrame:
    """Assigns administrative area identifiers to grid cells based on spatial overlap.

    This function takes a GeoDataFrame of grid cells and a filepath to a
    GeoDataFrame with area Polygons, e.g. admin1 areas. It determines which
    area(s) each grid cell belongs to. It stores either the matching area or a
    dictionary of {area: grid cell share} in case of multiple overlapping areas
    in an additional column in the `grid` GeoDataFrame.

    Two operational modes are available:
    1.  **Precise Mode (`performance=False`, default):** Iterates through each
        grid cell and identifies overlapping geometries via `GeoDataFrame.clip()`.
        In case of multiple matches, uses a cell-centered Albers Equal Area
        projection for accurate calculation of each area's relative share.
        Cells with no overlap are marked "not_assigned".
    2.  **Performance Mode (`performance=True`):** Uses a faster, two-stage
        geospatial join and overlay approach, calculating area shares based on
        EPSG:3857. This is significantly faster at the cost of less precise
        area calculations.

    The function ensures input area geometries are valid. Finally, it removes
    grid cells marked "not_assigned" and drops 'lat' and 'lon' columns from
    the base grid.

    Args:
        grid (gpd.GeoDataFrame): Input GeoDataFrame of grid cells. If
            `performance=False`, it expects 'lat' and 'lon' columns representing
            grid cell centers (used for the reprojection to calculate areas).
        fp_areas (str): Filepath to the vector file (e.g., shapefile, GeoPackage)
            containing the area polygons. Expected to be in EPSG:4326.
        grid_col (str): The name of the new column to be created in the `grid`
            GeoDataFrame, which will store the assigned area identifier(s).
        area_col (str): The name of the column in the administrative areas
            GeoDataFrame that contains the unique area identifiers.
        grid_size (float, optional): The size of the grid cells in decimal degrees.
            Used in the precise mode (`performance=False`) for defining the
            local Albers Equal Area projection. Defaults to 0.5.
        performance (bool, optional): If True, uses the faster sjoin/overlay
            approach. If False (default), uses the iterative per-cell precise
            area calculation.

    Returns:
        gpd.GeoDataFrame: The input `grid` GeoDataFrame with the new `grid_col`
            containing area identifiers (string or dictionary of shares) for each
            matched grid cell. Cells that could not be assigned are removed.
            'lat'/'lon' columns are dropped.
    """

    def assign_area_name(
        geometry: shapely.Polygon | shapely.MultiPolygon, lat: float, lon: float
    ) -> str | float | dict[str, float]:
        """Helper function for performance = False."""
        clipped = areas.clip(geometry)
        if clipped.empty:
            # should not happen
            return "not_assigned"
        elif len(clipped) == 1:
            # easy case
            return clipped[area_col].values[0]
        else:
            # hard case
            # we save the proportions of the areas in the grid cell to use as weights in
            # the aggregation later on

            # reprojecting to albers equal area equidistant projection for area calculations
            lat_1 = lat - grid_size / 3
            lat_2 = lat + grid_size / 3
            aea_proj = pyproj.CRS(
                f'+proj=aea +ellps="WGS84" +lat_1={lat_1} +lat_2={lat_2} +lat_0={lat} +lon_0={lon} +units=m'
            )
            # get area
            clipped["area_size"] = clipped.to_crs(aea_proj).area
            clipped["area_share"] = clipped.area_size / clipped.area_size.sum()
            area_shares = dict(zip(clipped[area_col], clipped.area_share))
            return area_shares

    areas = gpd.read_file(fp_areas, crs="epsg:4326")
    if not areas.geometry.is_valid.all():
        areas.geometry = areas.geometry.make_valid()
    if performance:
        # This is way quicker but might lack precision
        grid.set_crs("epsg:4326", inplace=True)
        # First round: find all pgids that are fully within the admin1 regions (easy cases)
        partial = areas.sjoin(grid, how="right", predicate="covers").dropna()[
            list(grid.columns) + [grid_col]
        ]
        missings = grid.loc[list(set(grid.index) - set(partial.index))]
        # Second round: perform overlay to calculate area; this takes a while but is still quicker then the old approach
        missings = missings.reset_index().to_crs("3857")
        missings_overlayed = areas.to_crs("3857").overlay(missings, how="intersection")
        # Filter out uniques (i.e. only one intersection)
        uniques = list(
            missings_overlayed.value_counts("pgid")[
                missings_overlayed.value_counts("pgid").eq(1)
            ].index
        )
        partial = pd.concat(
            [
                missings_overlayed[missings_overlayed["pgid"].isin(uniques)]
                .set_index("pgid")[list(grid.columns) + [grid_col]]
                .to_crs("4326"),
                partial,
            ]
        )
        missings_overlayed = missings_overlayed[~missings_overlayed["pgid"].isin(uniques)]
        # Calculate area
        missings_overlayed["area"] = missings_overlayed.geometry.area
        results = missings_overlayed.groupby("pgid").apply(
            lambda x: dict(zip(x[area_col], x["area"] / x["area"].sum()))
        )
        missings = grid.loc[list(set(grid.index) - set(partial.index))]
        missings.loc[results.index, grid_col] = results
        # Unassigned pgids
        missings.replace(np.nan, "not_assigned", inplace=True)
        grid = pd.concat([partial, missings])  # type: ignore
    else:
        tqdm.pandas()
        grid[grid_col] = grid.progress_apply(
            lambda x: assign_area_name(x.geometry, x.lat, x.lon), axis=1
        )  # type: ignore
    # only keep assigned areas for consistency and drop lat/lon columns again
    grid = grid.loc[grid[grid_col] != "not_assigned"].drop(columns=["lat", "lon"])
    return grid
