# add python path to the base directory
import pandas as pd
import numpy as np
import requests
import os
import netCDF4
import time

# archive:
# https://firms.modaps.eosdis.nasa.gov/download/
from datetime import datetime, timedelta
import zipfile
import json
import schedule
import io

import warnings

warnings.filterwarnings("ignore")


from base.objects import Dataset, console
from utils.data_processing import get_quarter


allJobsDone = False

EODIS_NASA_GOV = ""


def get_last_completed_quarter():
    current_date = pd.to_datetime("today")
    # Get current quarter as Period
    current_quarter = pd.Period(current_date, freq="Q")
    previous_quarter = current_quarter.asfreq("Q", "end") - 1

    return previous_quarter


##download archive first (2000-2022)
def download_archive(fromyear, toyear, source_path):
    print("start downloading", flush=True)
    try:
        for year in range(fromyear, toyear + 1):
            print(f"test year {year}", flush=True)
            outputfile = f"{source_path}/FIRMS/MODIS/modis_{year}_all_countries.zip"
            if not os.path.exists(outputfile):
                url = f"https://firms.modaps.eosdis.nasa.gov/data/country/zips/modis_{year}_all_countries.zip"
                response = requests.get(url)

                open(outputfile, "wb").write(response.content)
                print(f"Archive {year} downloaded", flush=True)

            if not os.path.exists(f"{source_path}/FIRMS/MODIS/modis/{year}"):
                with zipfile.ZipFile(outputfile, "r") as zip_ref:
                    zip_ref.extractall(f"{source_path}/FIRMS/MODIS")
                print(f"Archive {year} unzipped", flush=True)
    except Exception as e:
        print(e, flush=True)
        print(f"Archive missing in {year}", flush=True)
        print(f"Start downloading the SP and NRT: {year}", flush=True)

        pass


def download_latest(year, source_path, EODIS_NASA_GOV):
    # get last year world data (2023)
    def get_starting_dates(start_date, end_date, interval_days):
        starting_dates = []
        current_date = start_date

        while current_date <= end_date:
            starting_dates.append(current_date)
            current_date += timedelta(days=interval_days)

        return starting_dates

    def job():
        try:
            # get staritng date for NRL modis
            url = f"https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{EODIS_NASA_GOV}/MODIS_NRT"
            urlData = requests.get(url).content
            rawData = pd.read_csv(io.StringIO(urlData.decode("utf-8")))
            fromdate = pd.to_datetime(rawData["min_date"].values[0])

            checkapikey = f"https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY={EODIS_NASA_GOV}"
            response = requests.get(checkapikey)
            staus = response.content
            status = json.loads(staus)
            if status["current_transactions"] < 1000:
                start_date = datetime(year, 1, 1)  # Specify your start date (year, month, day)
                end_date = get_last_completed_quarter()
                # end_date = datetime(year, 12, 31)  # Specify your end date (year, month, day)
                interval_days = 1  # Specify the interval in days

                starting_dates = get_starting_dates(start_date, end_date.end_time, interval_days)

                checkapikey = f"https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY={EODIS_NASA_GOV}"
                print(starting_dates, flush=True)
                for datei in starting_dates:
                    if datei >= fromdate:
                        model = "MODIS_NRT"
                    else:
                        model = "MODIS_SP"
                    date_to_download = datei.strftime("%Y-%m-%d")
                    folder = f"{source_path}/FIRMS/MODIS/modis/{datei.year}"
                    os.makedirs(folder, exist_ok=True)
                    outputfile = f"{folder}/{date_to_download}.csv"
                    print(f"downlaoding {date_to_download} from {model}", flush=True)
                    if not os.path.exists(outputfile):
                        urlquery = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{EODIS_NASA_GOV}/{model}/-180,-90,180,90/1/{date_to_download}"
                        response = requests.get(urlquery)
                        if "Exceeding allowed transaction limit" in str(response.content):
                            print(
                                "Exceeding allowed transaction limit whait 10 minutes", flush=True
                            )
                            raise Exception(
                                "Exceeding allowed transaction limit whait 10 minutes",
                            )
                        open(outputfile, "wb").write(response.content)

                global allJobsDone
                allJobsDone = True
        except Exception as e:
            print(e)
            pass

    # Schedule the job to run every minute
    print("scheduling jobs to start every minutes")
    schedule.every(1).minutes.do(job)

    # Run the scheduled tasks

    while not allJobsDone:
        schedule.run_pending()
        time.sleep(10)  # Wait for 10 second before checking again


def daily_fire_download_nasa(sources_path):
    print("downloading data")
    from dotenv import load_dotenv

    load_dotenv()
    EODIS_NASA_GOV = os.getenv("EOSDIS_NASA_GOV")
    # Get current date
    current_date = pd.to_datetime("today")

    # Get current year as Period
    current_year = pd.Period(current_date, freq="Y")

    current_year = int(current_year.year)
    download_latest(current_year - 1, sources_path, EODIS_NASA_GOV)


def get_LC_class_4_apply(row):
    try:
        # Specify the target latitude and longitude
        target_lat = row["latitude"]
        target_lon = row["longitude"]

        # Read latitude and longitude values from the NetCDF file
        lats = nc.variables["lat"][:]
        lons = nc.variables["lon"][:]

        lat_index = np.argmin(np.abs(lats - target_lat))
        lon_index = np.argmin(np.abs(lons - target_lon))

        lc_class = nc.variables["lccs_class"][:, lat_index, lon_index]

        return int(lc_class)
    except Exception as e:
        print(f"Error: {e}")
        return -1


def to_bin(x):
    binsize = 0.5

    if x > 180:
        x = -180 + x - 180
    return (binsize / 2) + np.floor(x / binsize) * binsize


def process_data(rootdirsearch):
    df_path_list = []
    for path, subdirs, files in os.walk(rootdirsearch):
        for name in files:
            if name.endswith("csv"):
                # print(name)
                df_path_list.append([path, name])

        import random

    df_list = []
    while df_path_list:
        # Get a random element
        random_element = random.choice(df_path_list)

        # Remove the random element from the list
        df_path_list.remove(random_element)
        path = random_element[0]
        name = random_element[1]

        print(f"Processing {path}/{name}")

        df = pd.read_csv(f"{path}/{name}")

        # check if the point is not in classes 10,11,12,20,30,40
        df["latbin"] = df.latitude.map(to_bin)
        df["lonbin"] = df.longitude.map(to_bin)

        if not os.path.exists(f"{path}/{name.replace('.csv', '.csvx')}"):
            start_time = time.perf_counter()
            df["LC"] = df.apply(get_LC_class_4_apply, axis=1)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print("Elapsed time: ", elapsed_time)

            df.to_csv(f"{path}/{name.replace('.csv', '.csvx')}")
        else:
            df = pd.read_csv(f"{path}/{name.replace('.csv', '.csvx')}")

        df["count"] = 0
        if "type" in df.columns:
            df.loc[
                ((df.type == 0) & (df.confidence > 95) & (~(df.LC.isin([10, 11, 12, 20, 30, 40])))),
                "count",
            ] = 1
        else:
            df.loc[((df.confidence > 95) & (~(df.LC.isin([10, 11, 12, 20, 30, 40])))), "count"] = 1

        df["acq_date"] = pd.to_datetime(df["acq_date"], format="%Y-%m-%d")
        df["year"] = df["acq_date"].dt.year
        df["quarter"] = df["acq_date"].dt.to_period("Q")

        print(f"before dropping duplicates {df.shape}", flush=True)
        df = df.drop_duplicates(subset=["latitude", "longitude", "year"])
        print(f"after dropping duplicates {df.shape}", flush=True)

        df_list.append(df)

    df_list = pd.concat(df_list, ignore_index=True)
    return df_list


def daily_fire_match_pgid(sources_path, output_raw_directory, event_type):
    print("matching pgid")

    rootdirsearch = f"{sources_path}/FIRMS/MODIS/modis/"

    # Open the NetCDF file
    global nc

    nc = netCDF4.Dataset(
        f"{sources_path}/LANDCOVER/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc", "r"
    )

    raw = process_data(rootdirsearch)

    return raw


class MODISWildfiresData(Dataset):
    """Handles loading and processing of  event data.

    Supports loading from local preprocessed files or via API.
    Includes methods for aggregating event data to the
    grid-quarter level.

    Attributes:
        data_key (str): Set to "modis_wildfires".
        local (bool): Indicates whether to use local dumps (True) or
            download data via the API (False).
        dataset_available (bool): Flag set to True after data is
            successfully loaded or checked by `load_data`.
    """

    data_key = "modis_wildfires"

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

        sources_path = self.storage.storage_paths["processing"]
        destination = f"{sources_path}/"

        if not os.path.exists(destination):
            os.makedirs(destination)

        # step1
        print("downloading data")
        daily_fire_download_nasa(sources_path)
        # step2
        print("processing data")
        df_events = daily_fire_match_pgid(sources_path, sources_path, "wildfires")

        # columns to uppercase
        df_events.columns = [col.upper() for col in df_events.columns]
        df_events.rename(columns={"ACQ_DATE": "EVENT_DATE"}, inplace=True)
        # reanme LAT with LATITUDE and LON with LONGITUDE

        # df_events["EVENT_TYPE"] = "wildfires"

        df_events = df_events[["LATBIN", "LONBIN", "COUNT", "EVENT_DATE"]]

        df_events.rename(columns={"LATBIN": "LATITUDE", "LONBIN": "LONGITUDE"}, inplace=True)

        df_events = df_events.groupby(["LATITUDE", "LONGITUDE", "EVENT_DATE"]).sum().reset_index()

        df_events["YEAR"] = df_events["EVENT_DATE"].dt.year

        # "YEAR", "EVENT_TYPE", "LATITUDE", "LONGITUDE", "COUNT", "EVENT_DATE"
        # self.storage.save(df_events[self.columns], "processing", filename=self.filename)
        # beacuse those are the events I keep all the columns
        self.storage.save(df_events, "processing", filename=self.filename)

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
        self.filename = f"modis_wildfires_{self.last_quarter_date.year}_Q{int(self.last_quarter_date.month / 3)}"
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
            console.print(
                "No preprocessed  data in storage or out of date," + " processing event data..."
            )

            # don't automatically start  download since those are separate step in the
            # indicator logic that should each be performed deliberately
            assert self.dataset_available, " download/data check has not run, check indicator logic"

            df_event_level["QUARTER"] = df_event_level["EVENT_DATE"].dt.to_period("Q")
            df_event_level = (
                df_event_level[["LATITUDE", "LONGITUDE", "QUARTER", "COUNT"]]
                .groupby(by=["LATITUDE", "LONGITUDE", "QUARTER"])
                .sum()
                .reset_index()
            )

            df_event_level = df_event_level[["LATITUDE", "LONGITUDE", "QUARTER", "COUNT"]]

            # subset  2000 - 2010
            raw_reference = df_event_level[
                (df_event_level.QUARTER >= "2000Q1") & (df_event_level.QUARTER <= "2010Q4")
            ]
            # calcualte percentile 95 per pgid
            raw_reference = (
                raw_reference.groupby(by=["LATITUDE", "LONGITUDE"])["COUNT"]
                .quantile(0.95)
                .reset_index()
            )
            # get where the original raw  is > than the 95p
            df_event_level = df_event_level.merge(
                raw_reference, on=["LATITUDE", "LONGITUDE"], how="left", suffixes=("", "_95p")
            )
            df_event_level["COUNT_95p"] = df_event_level["COUNT_95p"].fillna(0)
            # if column count is greater than 95p put the value in count, else put 0
            df_event_level["COUNT"] = np.where(
                df_event_level["COUNT"] > df_event_level["COUNT_95p"], df_event_level["COUNT"], 0
            )
            # remove column count_95p
            df_event_level = df_event_level.drop(columns=["COUNT_95p"])

            # convert to quarter number only as 1,2,3,4
            df_event_level["YEAR"] = df_event_level["QUARTER"].dt.year
            df_event_level["QUARTER"] = df_event_level["QUARTER"].dt.quarter

            df_base = df_base.reset_index()
            df = df_base.merge(
                df_event_level,
                left_on=["year", "quarter", "lat", "lon"],
                right_on=["YEAR", "QUARTER", "LATITUDE", "LONGITUDE"],
                how="left",
            )

            df = df[["pgid", "year", "quarter", "lat", "lon", "COUNT"]]

            df["time"] = df["year"].astype(str) + "Q" + df["quarter"].astype(str)
            df["time"] = pd.PeriodIndex(df["time"], freq="Q")
            # convert to datetime
            df["time"] = df["time"].dt.to_timestamp()

            "lowercase all columns"
            df.columns = [col.lower() for col in df.columns]
            df = df.fillna(0)

            df.to_parquet(fp_preprocessed)
        return df


# test class
if __name__ == "__main__":
    from base.objects import ConfigParser

    config = ConfigParser()

    # Example usage
    data = MODISWildfiresData(local=False, config=config)
    # just load the current data

    df_modis_wildfires = data.load_data()
    print(df_modis_wildfires.head())
