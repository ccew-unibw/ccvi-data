import os
import time
from datetime import datetime, timedelta
import pandas as pd
import ee
import numpy as np
import warnings
import requests
import xarray as xr

from base.objects import GlobalBaseGrid


def get_image_url(dataset):
    image = (
        dataset.filterBounds(ee.Geometry.BBox(-180, -55.5, 180, 84))
        .first()
        .clip(ee.Geometry.BBox(-180, -55.5, 180, 84))
    )
    image = image.mask(image.mask())

    path = image.getDownloadUrl(
        {
            "crs": "EPSG:4326",
            "format": "GEO_TIFF",
            "scale": dataset.first().projection().nominalScale().getInfo(),
        }
    )
    print(path)
    print(dataset.first().projection().nominalScale().getInfo())
    response = requests.get(path)
    print(response)

    return response


def get_image_to_file(dataset, outputpath):
    try:
        response = get_image_url(dataset)
        print(response)

        with open(outputpath, "wb") as fd:
            fd.write(response.content)
        return 1
    except Exception as e:
        print(e)
        return 0


def geotiff_to_pandas_over_prio_grid(geotiff, base_grid: pd.DataFrame):
    unique_pgid = base_grid.reset_index()
    ds = xr.open_dataset(geotiff, engine="rasterio")
    df = ds.to_dataframe().reset_index()
    df.rename(columns={"x": "lon", "y": "lat"}, inplace=True)

    df = df.dropna()

    def to_bin(x):
        binsize = 0.5

        if x > 180:
            x = -180 + x - 180
        return (binsize / 2) + np.floor(x / binsize) * binsize

    df["latbin"] = df.lat.map(to_bin)
    df["lonbin"] = df.lon.map(to_bin)

    df = df.drop(columns=["lat", "lon"])
    df.rename(columns={"latbin": "lat", "lonbin": "lon"}, inplace=True)
    df = df.groupby(["lat", "lon"]).mean().reset_index()

    df = df.merge(unique_pgid, on=["lat", "lon"], how="right")

    return df


def parallel_download(x):
    try:
        startDate = x["startDate"]
        endDate = x["endDate"]
        temp_files = x["temp_files"]
        credentials = x["GOOGLEClient"]
        variable = x["variable"]
        subfolder = x["subfolder"]
        base_grid = x["base_grid"]

        if not os.path.exists(f"{temp_files}/raw/{subfolder}/{startDate[:4]}"):
            os.makedirs(f"{temp_files}/raw/{subfolder}/{startDate[:4]}")

        outfile = f"{temp_files}/raw/{subfolder}/{startDate[:4]}/image_{startDate}.parquet.gzip"
        imagepath = f"{temp_files}/raw/{subfolder}/{startDate[:4]}/image_{startDate}.tif"

        if os.path.exists(outfile):
            print(f"file {outfile} already exists")
        else:
            print(f"downloading: {outfile}")

            if os.path.exists(imagepath):
                test_image_availability = 1

            else:
                credentials._authorize()
                dataset = (
                    ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
                    .filterDate(startDate, endDate)
                    .select(variable)
                )
                test_image_availability = get_image_to_file(dataset, imagepath)

            if test_image_availability == 1:
                try:
                    df = geotiff_to_pandas_over_prio_grid(imagepath, base_grid)
                    df["quarter"] = pd.to_datetime(startDate, format="%Y-%m-%d").to_period("Q")
                    df["quarter"] = df["quarter"].astype(str)
                    df["q"] = df["quarter"].str.strip().str[-2:]

                    df.to_parquet(outfile)

                    print("test output")
                    if not df.__len__() > 0:
                        print(f"problem with file {imagepath}")
                        try:
                            os.remove(outfile)
                            os.remove(imagepath)
                        except:
                            pass

                except Exception as e:
                    print(e)
                    print(f"problem with file {imagepath}")
                    try:
                        os.remove(imagepath)
                    except:
                        pass
                    raise e  # Re-raise to stop processing
            else:
                # Raise FileNotFoundError immediately when image download fails
                raise FileNotFoundError(f"Failed to download image for date {startDate}. Image not available or download failed.")
            # raise ValueError("An error occurred!")  # Example error
    except Exception as e:
        # Catch the exception and re-raise it to be caught in the main process
        raise e


def daily_ERA5_download(sources_path, maindir, variable, subfolder, grid: GlobalBaseGrid):
    from utils.gee import GEEClient

    warnings.filterwarnings("ignore")
    # Initialize the Earth Engine API

    client = GEEClient()

    # Get current date
    current_date = pd.to_datetime("today")
    # Get current quarter as Period
    current_quarter = pd.Period(current_date, freq="Q")
    previous_quarter = current_quarter.asfreq("Q", "end") - 1
    year_of_quarter = previous_quarter.year
    last_day_of_quarter = previous_quarter.end_time.day
    last_month_of_quarter = previous_quarter.end_time.month
    year = int(year_of_quarter)
    month = int(last_month_of_quarter)
    day = int(last_day_of_quarter)
    print(f"downloading up to {year} {month} {day}")

    start = time.time()
    temp_files = f"{sources_path}/{maindir}/"

    if not os.path.exists(temp_files):
        os.makedirs(temp_files)

    ####DOWNLOAD DATA FROM GEE

    params = []

    startDate = datetime(1951, 1, 1)
    endDate = datetime(year, month, day)
    endDate = endDate + timedelta(days=1)

    # Getting List of Days using pandas
    datesRange = pd.date_range(startDate, endDate - timedelta(days=1), freq="d")
    base_grid = grid.load()
    for datei in datesRange:
        endDatei = datei + timedelta(days=1)

        parx = {
            "startDate": datei.strftime("%Y-%m-%d"),
            "endDate": endDatei.strftime("%Y-%m-%d"),
            "temp_files": temp_files,
            "GOOGLEClient": client,
            "variable": variable,
            "subfolder": subfolder,
            "base_grid": base_grid
        }
        params.append(parx)

    # Use a normal for loop instead of multiprocessing
    try:
        for param in params:
            parallel_download(param)
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

    return True
