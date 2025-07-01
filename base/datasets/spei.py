import os
import pandas as pd
import typer
import numpy as np
from datetime import date
import ee

targets = "all"


from base.objects import Dataset
from utils.index import get_quarter
from base.objects import console


from utils.gee import GEEClient
from joblib import Parallel, delayed


from datetime import timedelta

from utils.ee import get_grid_chunks, fc_to_dict

import duckdb

import subprocess

import calendar
import gc

from tqdm import tqdm
from rich import print

from pyeto import hargreaves


from utils.spatial_operations import get_neighboring_cells

try:
    subprocess.check_output(("R", "RHOME"))
    from rpy2.robjects.packages import importr
    import rpy2.robjects as ro

    r_check = True
except:
    print(
        "R not found. R related commands will not work. Execute them in a Docker container instead!"
    )


def create_spei_mask(client: GEEClient, storage: str, force: bool = False):
    fp_out = os.path.join(storage, "spei_mask.parquet")
    if os.path.exists(fp_out) and not force:
        return pd.read_parquet(fp_out)
    # We simply take the first image of the collection (2001)
    modis_landcover = ee.ImageCollection("MODIS/061/MCD12C1").first()
    modis_landcover_igbp = modis_landcover.select("Majority_Land_Cover_Type_1")
    mask_barren = modis_landcover_igbp.eq(16)  # Mask for barren land
    mask_ice = modis_landcover_igbp.eq(15)  # Mask for ice
    mask_water = modis_landcover_igbp.eq(0)  # Mask for water
    # Combine all masks into one image with multiple bands for easy feature extraction
    mask_image = (
        ee.ImageCollection.fromImages([mask_barren, mask_ice, mask_water])
        .toBands()
        .rename(["barren", "ice", "water"])
    )
    collection = mask_image.reduceRegions(
        collection=ee.FeatureCollection(os.getenv("BASE_GRID_ASSET")),
        reducer=ee.Reducer.count().combine(ee.Reducer.sum(), "", True),
    ).select(
        propertySelectors=["pgid", "barren_sum", "ice_sum", "water_sum"],
        retainGeometry=False,
    )
    # Get the dataframe
    df = pd.DataFrame(fc_to_dict(collection).getInfo())
    df.drop(columns=["system:index"], inplace=True)
    df["land_pixels"] = 100 - df["water_sum"]
    df["barren_share"] = df["barren_sum"] / df["land_pixels"]
    df["ice_share"] = df["ice_sum"] / df["land_pixels"]
    df["combined_share"] = df["ice_share"] + df["barren_share"]
    df = df[
        [
            "pgid",
            "barren_sum",
            "ice_sum",
            "water_sum",
            "land_pixels",
            "barren_share",
            "ice_share",
            "combined_share",
        ]
    ]
    df.to_parquet(fp_out)
    return


def calculate_spei(
    data,
    storage,
    time_scale: str = "1",
    quarterly: bool = False,
    parallel: bool = False,
    threads: int = 4,
    pet_method: str = "hargreaves",
    ref_period=None,
    save_memory: bool = True,
    rad: bool = False,
) -> str:
    """Calculate SPEI values for given time scales."""

    # This is a dirty fix for 1 missing pgid: just replace the values by the values from the adjacent pgid

    replacement = (
        data.query("pgid == 216028").copy().replace(216028, 215308).set_index(["date", "pgid"])
    )
    replacement["lat"] = 59.75
    data.set_index(["date", "pgid"], inplace=True)
    data.loc[replacement.index] = replacement
    data.reset_index(inplace=True)
    del replacement

    with console.status("Calculating SPEI...", spinner="runner"):
        output = os.path.join(
            storage,
            f"spei-{time_scale.replace(',', '-')}{'_quarterly' if quarterly else '_monthly'}.parquet",
        )
        # 1. Calculate PET for every observation using Hargreaves or Thornthwaite
        data["date"] = pd.to_datetime(data["date"], format="%Y-%m")
        # Filter max date to keep within current quarter
        max_date = data.date.max()
        max_quarter = max_date.quarter
        filter_date = max_date
        if max_quarter * 3 != max_date.month:
            if max_quarter == 1:
                filter_date = pd.Timestamp(f"{max_date.year - 1}-12-01")
            else:
                filter_date = pd.Timestamp(f"{max_date.year}-{(max_quarter - 1) * 3}-01")
        data.query("date <= @filter_date", inplace=True)
        data.sort_values(["pgid", "date"], inplace=True)
        # Convert Kelvin into C°
        data.loc[
            :,
            [
                "temperature_2m_min_mean",
                "temperature_2m_max_mean",
                "temperature_2m_mean",
            ],
        ] -= 273.15
        # data = data.query('date.dt.year > 1999')
        # data['pet'] = get_pet_df(data)
        get_pet_df(data, pet_method, rad, lib="pyeto")
        # data = data.merge(pet, 'left', on=['pgid', 'date']) # stupid shit but I don't care right now ;D
        # 2. Calculate SPEI for every location
        # TODO: explore more efficient parallelization by spatially chunking the workload and then run parallel processes on these chunks
        # spei = get_spei(data, scale=time_scale, parallel=parallel, threads=threads)
        # data = data.merge(spei, 'left', on=['pgid', 'date'])
        get_spei(
            data,
            scale=time_scale,
            parallel=parallel,
            threads=threads,
            ref_period=ref_period,
        )
        if save_memory:
            columns = ["pgid", "date"] + [f"spei_{s}".strip() for s in time_scale.split(",")]
        else:
            columns = ["pgid", "lat", "lon", "date", "iso3"] + [
                f"spei_{s}".strip() for s in time_scale.split(",")
            ]

        data[columns].to_parquet(output)

    console.print(":sparkles: Yeah, we are done.")
    console.print(f':floppy_disk: File stored at "{output}"')
    return data
    # return output


def get_pet_df(data, method="hargreaves", rad=False, lib="pyeto"):
    """Calculate PET using the hargreaves method from the SPEI R package."""
    spei = importr("SPEI")
    # Convert pandas dataframe into R vectors for Tmin, Tmax and lat
    # Tidy up data
    # data = data[['pgid', 'date', 'lat', 'temperature_2m_max_mean', 'temperature_2m_min_mean']]
    # data = data.sort_values(['pgid', 'date'])
    if rad and "surface_solar_radiation_downwards_sum_mean" in data.columns:
        # Calculate radiation values; ERA5 gives values as J/m^2. We will need to convert it to MJ/m²/d,
        # that is, we need to convert it to Megajoule and then divide it by the number of days in the
        # given month. This way we get the daily mega joule value per square meter.
        df = pd.DataFrame({"year": data.date.dt.year, "month": data.date.dt.month})
        data["days_in_month"] = np.vectorize(calendar.monthrange)(df["year"], df["month"])[1]
        data["ra"] = (data["surface_solar_radiation_downwards_sum_mean"] / 1e06) / data[
            "days_in_month"
        ]
        del df
    data_grouped = data.groupby("pgid")  # Group by pgid
    # pgids = data_grouped.groups.keys()
    # Convert to R matrices
    if method == "hargreaves":
        if rad and "surface_solar_radiation_downwards_sum_mean" in data.columns:
            if lib == "pyeto":
                # Using PyEto for cleaner code
                data["pet"] = (
                    np.vectorize(hargreaves)(
                        data["temperature_2m_min_mean"],
                        data["temperature_2m_max_mean"],
                        data["temperature_2m_mean"],
                        data["ra"],
                    )
                    * data["days_in_month"]
                )
            else:
                ra = (data_grouped["ra"].apply(lambda x: np.array(x))).tolist()
                ra = ro.r["matrix"](ro.FloatVector(np.concatenate(ra)), ncol=len(ra))
                tmin = (
                    data_grouped["temperature_2m_min_mean"].apply(lambda x: np.array(x))
                ).tolist()
                tmin = ro.r["matrix"](ro.FloatVector(np.concatenate(tmin)), ncol=len(tmin))
                tmax = (
                    data_grouped["temperature_2m_max_mean"].apply(lambda x: np.array(x)) - 273.15
                ).tolist()
                tmax = ro.r["matrix"](ro.FloatVector(np.concatenate(tmax)), ncol=len(tmax))
                data["pet"] = np.array(
                    spei.hargreaves(tmin, tmax, Ra=ra)
                ).T.flatten()  # return all columns as an array
        else:
            lat = data_grouped["lat"].apply(lambda x: min(x)).to_list()
            lat = ro.FloatVector(lat)
            tmin = (data_grouped["temperature_2m_min_mean"].apply(lambda x: np.array(x))).tolist()
            tmin = ro.r["matrix"](ro.FloatVector(np.concatenate(tmin)), ncol=len(tmin))
            tmax = (data_grouped["temperature_2m_max_mean"].apply(lambda x: np.array(x))).tolist()
            tmax = ro.r["matrix"](ro.FloatVector(np.concatenate(tmax)), ncol=len(tmax))
            data["pet"] = np.array(
                spei.hargreaves(tmin, tmax, lat=lat)
            ).T.flatten()  # return all columns as an array

    else:  # implicit thornthwaite
        lat = data_grouped["lat"].apply(lambda x: min(x)).to_list()
        lat = ro.FloatVector(lat)
        tave = (data_grouped["temperature_2m_mean"].apply(lambda x: np.array(x))).tolist()
        tave = ro.r["matrix"](ro.FloatVector(np.concatenate(tave)), ncol=len(tave))
        data["pet"] = np.array(
            spei.thornthwaite(tave, lat=lat)
        ).T.flatten()  # return all columns as an array

    # Clean up dataframe for memory efficiency
    data.drop(
        columns=[
            "surface_solar_radiation_downwards_sum_mean",
            "temperature_2m_max_mean",
            "temperature_2m_mean",
            "temperature_2m_min_mean",
            "days_in_month",
            "ra",
        ],
        inplace=True,
    )


def get_spei(data, scale: int = 1, parallel: bool = False, threads: int = 4, ref_period=None):
    """Calculate SPEI values using the R package"""
    ref_start = None
    ref_end = None
    if ref_period:
        ref_period = ref_period.split("-")
        ref_start = int(ref_period[0])
        ref_end = int(ref_period[1])
    if not parallel:
        spei = importr("SPEI")
    else:
        r_code = f"""
            library(parallel)
            library(SPEI)
            cores <- detectCores()
            cl <- makeCluster({threads})
            spei_fun <- function(d, s, rs, re) {{spei(d, s, ref.start=rs, ref.end=re, na.rm=T, verbose=T)$fitted}}
            spei_ref_fun <- function(d, s, rs, re) {{spei(ts(d, frequency = 12, start = c(1950, 2)), s, ref.start=rs, ref.end=re, na.rm=T, verbose=T)$fitted}}
            clusterExport(cl=cl, c('spei'))
            """
        ro.r(r_code)
    data["balance"] = data["total_precipitation_sum_mean"] * 1000 - data["pet"]
    scale = [int(i) for i in scale.split(",")]
    if len(scale) < 2:
        scale = scale[0]
    # We need to create one matrix column per grid
    # data = data.sort_values(['pgid', 'date'])
    data_grouped = data.groupby("pgid")
    pgids = data_grouped.groups.keys()
    spei_input = data_grouped["balance"].apply(lambda x: np.array(x)).tolist()
    spei_input = ro.r["matrix"](ro.FloatVector(np.concatenate(spei_input)), ncol=len(spei_input))
    if type(scale) == list:
        for s in scale:
            console.print(f":calendar: Calculating SPEI-{s}")
            if parallel:
                ro.globalenv["spei_input"] = spei_input
                # if ref_period:
                #    # If ref period is given, input data needs to be provided as tsmatrix
                #    ro.r(f'spei_input <- ts(spei_input, frequency = 12, start = c(1950, 2), end = c(2023, 6))')
                ro.r(
                    f"result <- parApply(cl, X=spei_input, MARGIN=2, FUN={'spei_ref_fun' if ref_period else 'spei_fun'}, {s}, {f'c({ref_start}, 1)' if ref_start else 'NULL'}, {f'c({ref_end}, 1)' if ref_end else 'NULL'})"
                )
                data[f"spei_{str(s)}"] = np.array(ro.r["result"]).T.flatten()
            else:
                data[f"spei_{str(s)}"] = np.array(
                    spei.spei(spei_input, scale=s).rx2("fitted")
                ).T.flatten()
        return data[["pgid", "date", *[f"spei_{s}" for s in scale]]]

    print(f":calendar: Calculating SPEI-{scale}")
    if parallel:
        ro.globalenv["spei_input"] = spei_input
        # ro.r('saveRDS(spei_input, file="raw.rds")')
        # if ref_period:
        #    print('Convert to time series')
        #    # If ref period is given, input data needs to be provided as tsmatrix
        #    ro.r(f'spei_input <- ts(spei_input, frequency = 12, start = c(1950, 2), end = c(2023, 6))')
        #    #print(ro.r('saveRDS(spei_input, file="ts.rds")'))
        ro.r(
            f"result <- parApply(cl, X=spei_input, MARGIN=2, FUN={'spei_ref_fun' if ref_period else 'spei_fun'}, {scale}, {f'c({ref_start}, 1)' if ref_end else 'NULL'}, {f'c({ref_end}, 1)' if ref_end else 'NULL'})"
        )
        data[f"spei_{str(scale)}"] = np.array(ro.r["result"]).T.flatten()
        # return np.array(ro.r['result']).T.flatten()
    else:
        r_ref_start = ro.IntVector([ref_start, 1]) if ref_start else ro.NULL
        r_ref_end = ro.IntVector([ref_end, 1]) if ref_end else ro.NULL
        data[f"spei_{scale}"] = np.array(
            spei.spei(spei_input, scale=scale, ref_start=r_ref_start, ref_end=r_ref_end).rx2(
                "fitted"
            )
        ).T.flatten()
        # return np.array(spei.spei(spei_input, scale=scale).rx2('fitted')).T.flatten()
    # return data[['pgid', 'date', f'spei_{scale}']]


def mask_spei(data: pd.DataFrame, columns: list[str], storage: str, threshold: float = 0.75):
    """
    Mask SPEI data based on land cover. Masking is applied if combined share of barren and ice land exceeds a threshold.

    """
    spei_mask_fp = os.path.join(storage, "spei_mask.parquet")
    mask_df = pd.read_parquet(spei_mask_fp)
    # data to be masked based on more than 80% barren land share
    mask_df["mask"] = mask_df["combined_share"] > threshold
    # For grid cells that need masking, I set their SPEI values to NaN
    mask_map = mask_df.set_index("pgid")["mask"].to_dict()
    for col in columns:
        data[col] = data.apply(
            lambda x: np.nan if mask_map.get(x.pgid, False) else x[col],
            axis=1,
        )
    return data


def process_current_drought(storage: str, spei_fp: str, out_fp: str) -> list:
    spei = pd.read_parquet(spei_fp)
    
    print("-- spei calculating lamda ...")
    spei["drought_count"] = spei["spei_3"].apply(lambda x: x if x < -1 else 0)
    spei["drought_count"] = spei["drought_count"] * (-1)
    spei["drought_count"] = spei["drought_count"].fillna(0)

    # Sum by quarter first and then by 7*4 quarters
    spei_dim1 = spei.groupby(["pgid", "year", "quarter"])["drought_count"].sum().to_frame()

    spei_dim1 = spei_dim1.rename(columns={"drought_count": "drought_months_quarter"})

    # keep the mean spei value just in case we still want that version in the future e.g. for testing etc. - UNUSED currently!
    spei_dim1["spei3_mean"] = spei.groupby(["pgid", "year", "quarter"])["spei_3"].mean()

    # yearly version
    spei_dim1["drought_months_year"] = list(
        spei_dim1.groupby("pgid")["drought_months_quarter"].rolling(4, min_periods=4).sum()
    )

    spei_dim1 = spei_dim1.reset_index()

    spei_dim1.query("year > 2009", inplace=True)

    # Add lat lon data
    # spei_dim1 = add_lat_lon(spei_dim1)

    final_df = spei_dim1[
        [
            "pgid",
            "year",
            "quarter",
            "drought_months_quarter",
            "drought_months_year",
            "spei3_mean",
        ]
    ].rename(
        columns={
            "drought_months_quarter": "CLI_current_drought_quarterly_raw",
            "drought_months_year": "count",
        }
    )

    print("-- Apply masking after count generation")
    final_df = mask_spei(final_df, ["count"], storage)
    return final_df


def process_accumulated_drought(storage: str, spei_fp: str, out_fp: str) -> list:
    spei = pd.read_parquet(spei_fp)
    # Count droughts
    spei["drought_count"] = (spei["spei_3"] <= -1).astype(int)
    # Sum by quarter first and then by 7*4 quarters
    spei_dim2 = (
        spei.groupby(["pgid", "year", "quarter"]).aggregate({"drought_count": "sum"}).reset_index()
    )
    spei_dim2["cumulative_drought"] = list(
        spei_dim2.groupby("pgid")["drought_count"].rolling(7 * 4, min_periods=7 * 4).sum()
    )
    spei_dim2.query("year > 2009", inplace=True)

    final_df = spei_dim2[
        [
            "pgid",
            "year",
            "quarter",
            "cumulative_drought",
        ]
    ].rename(
        columns={
            "cumulative_drought": "count",
        }
    )

    final_df = mask_spei(final_df, ["count"], storage)
    return final_df


def spei_input_parallel(
    client: GEEClient,
    year_start: int,
    year_end: int,
    asset_path: str,
    storage: str,
    time_chunk_size: int = 1,
    grid_chunks: int = 10,
    save_memory: bool = True,
    force: bool = False,
):
    """
    Parallel version of the processing task. This is done in order to not have to use GDrive for data storage. This is based on Stefano's approach.
    """
    client.init_dataset("spei_input", asset_path)
    cols = [
        "date",
        "iso3",
        "lat",
        "lon",
        "pgid",
        "surface_net_solar_radiation_sum_mean",
        "surface_net_solar_radiation_sum_sum",
        "surface_solar_radiation_downwards_sum_mean",
        "surface_solar_radiation_downwards_sum_sum",
        "temperature_2m_max_mean",
        "temperature_2m_mean",
        "temperature_2m_min_mean",
        "total_precipitation_sum_mean",
        "total_precipitation_sum_sum",
    ]
    input = os.path.join(storage, "data")
    output = os.path.join(input, f"spei_input-full_{year_start}-{year_end}.parquet")
    if os.path.exists(output) and not force:
        return output

    def get_quarter_end(year_start, year_end):
        dates = []
        for year in range(year_start, year_end + 1):
            for month in [3, 6, 9, 12]:
                dates.append(date(year, month, 1) + timedelta(days=32))
        return [d - timedelta(days=d.day) for d in dates]

    def get_processing_dict(spatial_chunks: int = 10, time_chunk_size: int = 1):
        # TODO: split this into quarters and make this end dynamically
        grid_chunks = get_grid_chunks(client, spatial_chunks)
        time_chunks = np.array_split(
            range(year_start, year_end + 1),
            len(range(year_start, year_end + 1)) / time_chunk_size,
        )
        return list(
            np.array(
                [
                    [
                        {"year_start": t.min(), "year_end": t.max(), "grid_chunk": c}
                        for c in grid_chunks
                    ]
                    for t in time_chunks
                ]
            ).flatten()
        )

    processing_dict = get_processing_dict(grid_chunks, time_chunk_size)
    result = Parallel(n_jobs=5, verbose=10)(
        delayed(par_func)(param, client, save_memory, cols, input) for param in processing_dict
    )
    if save_memory:
        combine_dataframes(result, output)
        result = output
    else:
        result = pd.concat(result)
        # Clean up NaNs
        result = result[cols].replace(-9999, np.nan)
        result.to_parquet(output)
    return result


def par_func(params, client, save_memory, cols, fp_out):
    """Parallel function called for every GEE task."""
    # We need to reinit the client, as the GEE session is not shared over multiple spawned processes.
    client._authorize()
    if not os.path.exists(fp_out):
        try:
            os.makedirs(fp_out)
        except PermissionError as e:
            print(f"Error creating directory {fp_out}: {e}")
            return None

    images = client.get_images_for_years(params["year_start"], params["year_end"] + 1)
    scale = client.get_scale()
    chunk = params["grid_chunk"]
    output_file = (
        f"spei_input_{params['year_start']}-{params['year_end']}_{chunk[0]}-{chunk[1]}.parquet"
    )
    output_path = os.path.join(fp_out, output_file)

    if os.path.exists(output_path):
        if save_memory:
            return output_path
        return pd.read_parquet(output_path)[cols]

    def process_images(image):
        def set_date_property(feature):
            defaults = {
                "total_precipitation_sum_mean": -9999,
                "total_precipitation_sum_sum": -9999,
                "temperature_2m_mean": -9999,
                "temperature_2m_min_mean": -9999,
                "temperature_2m_max_mean": -9999,
                "surface_net_solar_radiation_sum_mean": -9999,
                "surface_net_solar_radiation_sum_sum": -9999,
                "surface_solar_radiation_downwards_sum_mean": -9999,
                "surface_solar_radiation_downwards_sum_sum": -9999,
            }
            feature = feature.set(
                feature.toDictionary().combine(defaults, False)
            )  # Also keep empty results
            return feature.set("date", date)

        date = image.date().format("YYYY-MM")
        collection = image.reduceRegions(
            collection=client.asset.filter(f"pgid >= {chunk[0]} and pgid <= {chunk[1]}"),
            reducer=get_reducers(),
            scale=scale,
        )
        return collection.map(set_date_property)

    query = (
        images.map(process_images)
        .flatten()
        .select(
            propertySelectors=[
                "date",
                "iso3",
                "lat",
                "lon",
                "pgid",
                "total_precipitation_sum_mean",
                "total_precipitation_sum_sum",
                "temperature_2m_mean",
                "temperature_2m_min_mean",
                "temperature_2m_max_mean",
                "surface_net_solar_radiation_sum_mean",
                "surface_net_solar_radiation_sum_sum",
                "surface_solar_radiation_downwards_sum_mean",
                "surface_solar_radiation_downwards_sum_sum",
            ],
            retainGeometry=False,
        )
    )

    try:
        df = pd.DataFrame(fc_to_dict(query).getInfo())[cols]
        df.to_parquet(output_path)
    except PermissionError as e:
        print(f"Error writing file {output_path}: {e}")
        return None

    if save_memory:
        return output_path
    return df


def combine_dataframes(file_list: list, output) -> pd.DataFrame:
    """Combine single .parquet files into one. This is using duckdb instead of pandas, which uses much less memory."""
    con = duckdb.connect()
    con.execute(
        f"""COPY (SELECT 
                date, iso3, lat, lon, pgid, surface_net_solar_radiation_sum_mean, surface_net_solar_radiation_sum_sum, surface_solar_radiation_downwards_sum_mean, surface_solar_radiation_downwards_sum_sum, temperature_2m_max_mean, temperature_2m_mean, temperature_2m_min_mean, total_precipitation_sum_mean, total_precipitation_sum_sum
                FROM read_parquet({file_list})) TO '{output}' (FORMAT 'parquet', COMPRESSION 'ZSTD');"""
    )
    con.close()
    # Convert -9999 to np.nan values
    df = pd.read_parquet(output).replace(-9999, np.nan)
    df.to_parquet(output)


def get_reducers():
    return ee.Reducer.sum().combine(ee.Reducer.mean(), sharedInputs=True)


class SPEIData(Dataset):
    """Handles loading and processing of  event data.

    Supports loading from local preprocessed files or via API.
    Includes methods for aggregating event data to the
    grid-quarter level.

    Attributes:
        data_key (str): Set to "diva_subsidence".
        local (bool): Indicates whether to use local dumps (True) or
            download data via the API (False).
        dataset_available (bool): Flag set to True after data is
            successfully loaded or checked by `load_data`.
    """

    data_key = "spei"

    def __init__(self, local: bool = True, *args, **kwargs):
        """Initializes the data source.

        Sets the operation mode (local file vs API) and calls the Dataset
        initializer to setup config and storage.

        Args:
            config (ConfigParser): An initialized ConfigParser instance.
            local (bool, optional): Indicates whether to use local
                dumps (True) or download data via the API (False).
        """
        self.local = local
        self.dataset_available = False
        self.data_keys = [self.data_key, "countries"]
        super().__init__(*args, **kwargs)

    def download_data(self):
        """Downloads  data from the API.

        Downloads the  data from the API and saves it to the processing
        storage. The filename is based on the current date and time.

        Returns:
            str: The filename of the downloaded  data.
        """
        import os
        from dotenv import load_dotenv

        load_dotenv()
        assert os.getenv("BASE_GRID_ASSET"), "BASE_GRID_ASSET environment variable not set."

        app = typer.Typer()
        client = GEEClient()  # Load GEEClient globally to reduce numbers of parameters required for the functions below

        # TODO: add more print statements describing the current process

        sources_path = self.storage.storage_paths["processing"]
        destination = f"{sources_path}/"

        save_memory = True
        force = False

        if not os.path.exists(destination):
            os.makedirs(destination)

        """Process spei data. This function downloads the latest ERA5 data required as input to the SPEI formulas. It then estimates PET and
        ultimately SPEI scores using the R SPEI package. Running this code requires a lot of memory (~32GB)! Memory consumption can be lowered
        by reducing the number of cores used to calculate SPEI. This will come at a performance cost, though. Additionally, the memory_save flag can be
        used when generating the input data. This will combine all single parquet files using duckdb.

        Args:
            force (bool, optional): Forces a full re-run deleting all existing data. Defaults to False.
        """
        print(
            "NOTE that SPEI generation currently is not yet dynamic to the last quarter! That means data for the last year still needs"
            "to be manually deleted before a new quarter run"
        )
        try:
            storage = destination
            console.print(":droplet: Processing SPEI indicators...")
            with console.status("Aggregating and downloading ERA5 data on GEE...", spinner="earth"):
                assert targets in [
                    "all",
                    "dim1",
                    "dim2",
                ], "Invalid target. Allowed targets are 'all', 'dim1' or 'dim2'"
                # Step 1: Get ERA5 input data from the Google Earth Engine
                # TODO: turn the settings (years) into a config file
                df = spei_input_parallel(
                    client,
                    1950,
                    2025,
                    os.getenv("BASE_GRID_ASSET"),
                    storage,
                    save_memory=save_memory,
                    force=force,
                )
                console.print(
                    ":earth_africa: Aggregating and downloading ERA5 data on GEE... [bold green]DONE[/bold green]"
                )

            # Step 2: Calculate SPEI scores

            console.print(":droplet: Estimate PET and calculate SPEI-3...")
            assert type(df) in [pd.DataFrame, str], "Error in processing SPEI input data"
            if type(df) == str:
                df = pd.read_parquet(df)
            if save_memory:
                df.drop(
                    columns=[
                        "lat",
                        "lon",
                        "iso3",
                        "total_precipitation_sum_sum",
                        "surface_net_solar_radiation_sum_mean",
                        "surface_net_solar_radiation_sum_sum",
                        "surface_solar_radiation_downwards_sum_sum",
                    ],
                    inplace=True,
                )
            # TODO: implement real checks not hardcoded file exist check; this should be based on some kind of state file

            # spei_fp = os.path.join(storage, "spei-3_monthly.parquet")
            # if not os.path.exists(spei_fp) or force:
            spei_fp = calculate_spei(
                df,
                storage,
                time_scale="3",
                parallel=True,
                threads=8,
                ref_period="1951-1980",
                rad=True,
            )
            # return spei_fp
            spei_fp.to_parquet(
                f"{self.storage.storage_paths['processing']}/{self.filename}.parquet"
            )

            # Raw input data not needed anymore
            del df
            gc.collect()
            console.print(
                ":droplet: Estimate PET and calculate SPEI-3... [bold green]DONE[/bold green]"
            )
            with console.status("Process SPEI-3 ...", spinner="earth"):
                # Step 3: Process the data and store it in the output as a parquet file
                # TODO: this is also very opinionated; let's add some more config options in the future
                console.print(":droplet: Creating land cover mask for SPEI...")
                create_spei_mask(client, storage, force)
                

            console.print(":droplet: Processing SPEI indicators... [bold green]DONE[/bold green]")

        except AssertionError as e:
            console.print(f"[bold red]Error processing SPEI data: {e}[/bold red]")
            console.print(
                "[bold yellow]Skipping SPEI processing and continuing with other tasks...[/bold yellow]"
            )

    def load_data(self):
        """Loads  data, checking for cached processing files first.

        Attempts to load a local  copy from the 'processing' storage
        including the last completed quarter. If not found:
        - If `self.local` is True, loads the raw dump specified in the config.
          Raises an error if the provided  dump does not fully cover the
          latest quarter.
        - If `self.local` is False, currently raises NotImplementedError (API access TBD).
        Saves the loaded raw/dump data to the processing storage.

        Returns:
            pd.DataFrame: The loaded  event data.
        """
        self.last_quarter_date = get_quarter("last", bounds="end")
        self.filename = (
            f"spei_{self.last_quarter_date.year}_Q{int(self.last_quarter_date.month / 3)}"
        )
        self.columns = ["YEAR", "EVENT_TYPE", "LATITUDE", "LONGITUDE", "COUNT", "EVENT_DATE"]
        try:
            df_event_level = self.storage.load("processing", filename=self.filename)
        except FileNotFoundError:
            if self.local:
                df_event_level = pd.read_parquet(self.data_config[self.data_key])
                if df_event_level["EVENT_DATE"].max() < self.last_quarter_date:
                    raise Exception(
                        "preprocessed  data out of date, please provide a version up to "
                        f"{self.last_quarter_date}."
                    )
            else:
                self.download_data()
                self.dataset_available = True
                df_event_level = self.storage.load("processing", filename=self.filename)
                return df_event_level

        # Set an instance attribute for easy checking
        self.dataset_available = True
        return df_event_level

    def create_grid_quarter_aggregates(
        self,
        df_base: pd.DataFrame,
        df_event_level: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculates grid-quarter aggregates from the event level  data.

        Assigns events to grid cells (pgid), calculates quarterly aggregates
        (event counts, fatalities) for armed violence and unrest, merges with the
        base grid structure, and fills missing values based on data coverage
        information.

        Args:
            df_base (pd.DataFrame): Base data structure for indicator data.
            df_event_level (pd.DataFrame):  event-level data from `self.load_data`.

        Returns:
            pd.DataFrame: Dataframe aligned to index grid with quarterly  aggregates.
        """
        fp_preprocessed = self.storage.build_filepath("processing", filename="preprocessed")
        try:
            df = pd.read_parquet(fp_preprocessed)
            last_quarter_date = get_quarter("last")

            if df["time"].max().date() < last_quarter_date:
                raise FileNotFoundError
            return df
        except FileNotFoundError:
            print("No preprocessed  data in storage or out of date," + " processing event data...")

            # don't automatically start  download since those are separate step in the
            # indicator logic that should each be performed deliberately
            assert self.dataset_available, " download/data check has not run, check indicator logic"

            # quarter as number 1,2,3,4

            df = df_base.reset_index()[['pgid', 'lat', 'lon']]
            df = df.drop_duplicates(subset=['pgid','lat', 'lon'])
            
            spei = df_event_level
            spei['time']= spei['date']
            # We've got exposure data back to 2000. Hence, we can only calculate accumulated droughts
            # starting from 2000. For the accumulation, we need to load data from 2000 - 7 onwards.
            spei.query("time.dt.year >= 1993", inplace=True)
            spei["year"] = spei.time.dt.year
            spei["month"] = spei.time.dt.month
            spei["quarter"] = spei.time.dt.quarter
            spei = spei.replace([np.inf, -np.inf], np.nan)  # Turn inf into nan
            # Deal with missing values by replacing them with the mean of all surrounding grid cells
            missings = spei.query("spei_3.isna()")
            pgids = missings.pgid.apply(get_neighboring_cells)
            queries = list(zip(pgids, missings.time))
            #spei_unique = spei.drop_duplicates(subset=["time", "pgid"])
            #spei_xr = spei_unique.set_index(["time", "pgid"])["spei_3"].to_xarray()
            spei_xr = spei.set_index(["time", "pgid"])["spei_3"].to_xarray()
            means = [
                spei_xr.sel(time=time, pgid=spei_xr.pgid.isin(pgids)).mean(skipna=True).item()
                for pgids, time in tqdm(queries)
            ]
            spei.loc[missings.index, "spei_3"] = means
            # Make sure that the df is sorted properly
            spei = spei.sort_values(["pgid", "time"])

            spei = spei.merge(
                df,
                how="left",
                on="pgid",
            )
            
            spei.to_parquet(fp_preprocessed)
        return spei


# test class
if __name__ == "__main__":
    from base.objects import ConfigParser

    config = ConfigParser()

    # Example usage
    data = SPEIData(local=False, config=config)
    # just load the current data

    df_dataset = data.load_data()
    print(df_dataset.head())
