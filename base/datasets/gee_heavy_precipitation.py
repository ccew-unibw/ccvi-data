# add python path to the base directory
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from base.objects import Dataset, console, GlobalBaseGrid
from utils.index import get_quarter
from utils.gee_daily import daily_ERA5_download


def get_days_between_quarters(start_quarter, end_quarter):
    # Convert the input quarters to pandas Period objects
    start_period = pd.Period(start_quarter, freq="Q")
    end_period = pd.Period(end_quarter, freq="Q")

    # Generate a date range between the start and end quarters
    date_range = pd.date_range(start=start_period.start_time, end=end_period.end_time, freq="D")

    # Create DataFrame
    df = pd.DataFrame({"date": date_range})

    return df


def get_last_completed_quarter():
    # Get the current date
    current_date = datetime.now()

    # Calculate the previous quarter
    previous_quarter = current_date - timedelta(days=365 / 4)

    # Format the result as "YYYYQn"
    result = f"{previous_quarter.year}Q{((previous_quarter.month - 1) // 3) + 1}"

    return result


resultdict = []


def log_result(result):
    resultdict.append(result)


def count_anomaly_and_magnitude(arr_temp, pgid, renge_dates, reference_pgid):
    print(pgid, flush=True)

    t_99 = []

    # building years vectors: (365/366 days) of the 90h, 75h, 25h percentile for the specific pgid
    for i in range(0, len(arr_temp)):
        # day = renge_dates[i]
        # dataref = f"{str(day.month).zfill(2)}_{str(day.day).zfill(2)}"
        # reference = reference_pgid.loc[reference_pgid.dateref==dataref]

        t_99.append(reference_pgid["band_data"].values[0])

    # vectors series of referece values
    t_99 = np.array(t_99)

    # Heatwave: period of 3 consecutive days with maximum temperature (Tmax) above the daily threshold
    # for the reference period 1981–2010. The threshold is
    # defined as the 90th percentile of daily maxima temperature, centered on a 31 day window
    anomaly = (arr_temp > t_99) & (arr_temp > (10 / 1000))
    anomaly = anomaly * 1

    # recording the position (day) of the anomaly when are >=0 cosecutive days
    heatwave_list = []
    out = []
    if (anomaly.sum()) > 0:
        for i in range(0, anomaly.shape[0]):
            if anomaly[i] == 1:
                heatwave_list.append(i)

        for i in heatwave_list:
            date_anomaly = renge_dates[i]
            precipitation = arr_temp[i]
            out.append([date_anomaly, precipitation])

    out = pd.DataFrame(out, columns=["date_anomaly", "precipitation"])
    out["pgid"] = pgid
    return out


def build_anomaly(arr, renge_dates, reference_pd):
    import multiprocessing
    from multiprocessing import Pool

    out = []
    # loop for all pgid to get and scan the temporal series
    list_of_commands = []
    for i in range(arr.shape[0]):
        print(f"processing pgid {i} on {arr.shape[0]}", flush=True)
        if arr[i, 0, :].max() == arr[i, 0, :].min():
            # identify the pgid
            pgid = int(arr[i, 0, 0])

            # identify the 366 reference values 90h, 75h, 25h for the selected pgid
            reference_pgid = reference_pd.loc[reference_pd.pgid == pgid]

            # arr[i,1,:] is the series of tmax value for the selected pgid (356/366 values)
            # pgid is pgid
            # reference_pgid is a DataFrame of all days reference values for the selected pgid
            list_of_commands.append([arr[i, 1, :], pgid, renge_dates, reference_pgid])

    print("start creating pool", flush=True)

    # set pool accoring to the number of cores
    # determine the number of cores
    num_cores = multiprocessing.cpu_count()
    print(f"Number of cores: {num_cores}")
    # Create a pool of workers
    # Use the number of cores - 1 to leave one core free

    # Create a pool of workers
    pool = Pool(num_cores - 1)
    for x_y_z in list_of_commands:
        pool.apply_async(count_anomaly_and_magnitude, args=(x_y_z), callback=log_result)

    print("pool close", flush=True)
    pool.close()
    print("pool join", flush=True)
    pool.join()

    out = pd.concat(resultdict)

    return out


def dailyprecipitation_buildthreshold(source_path):
    threshold_path = f"{source_path}/precipitation/raw/era5_precipitation_reference/era5_precipitation_1951-1980_99p.parquet.gzip"

    if os.path.exists(threshold_path):
        print(f"hystorical threshold _1951-1980_99p already exists: {threshold_path}")
    else:
        reference_directory = f"{source_path}/precipitation/raw/era5_precipitation_reference"
        if not os.path.exists(reference_directory):
            os.makedirs(reference_directory)

        dfs = []
        reference_pgid = pd.DataFrame()
        for year in range(1951, 1981):
            basepath = f"{source_path}/precipitation/raw/era5_precipitation/{year}"
            files = [f for f in os.listdir(basepath) if f.endswith(".parquet.gzip")]
            for file in files:
                print(f"Reading {file}")
                df = pd.read_parquet(f"{basepath}/{file}")
                if reference_pgid.__len__() == 0:
                    reference_pgid = df[["pgid"]]
                    reference_pgid.drop_duplicates(inplace=True)
                df = df[["pgid", "band_data"]]
                dfs.append(df.loc[df.band_data > (1 / 1000)])

        print("Concatenating")
        dfs = pd.concat(dfs)
        # Group by 'pgid' and calculate the 99th percentile of 'band_data' for each group
        print("Calculating 99p")
        result = dfs.groupby("pgid")["band_data"].quantile(0.99).reset_index()
        print("merge with reference")
        result = pd.merge(reference_pgid, result, on="pgid", how="left")
        print("Saving 99p")
        result.to_parquet(
            f"{source_path}/precipitation/raw/era5_precipitation_reference/era5_precipitation_1951-1980_99p.parquet.gzip",
            compression="gzip",
        )


def dailyprecipitation_buildhistorical(source_path):
    path_daily_prec = f"{source_path}/precipitation/raw/era5_precipitation"
    path_daily_prec_reference = f"{source_path}/precipitation/raw/era5_precipitation_reference/era5_precipitation_1951-1980_99p.parquet.gzip"
    path_year_anomaly = f"{source_path}/precipitation/raw/anomaly/"
    path_quarter_historical = f"{source_path}/precipitation/raw/anomaly/"

    if not os.path.exists(path_daily_prec):
        os.makedirs(path_daily_prec)

    if not os.path.exists(path_year_anomaly):
        os.makedirs(path_year_anomaly)

    if not os.path.exists(path_quarter_historical):
        os.makedirs(path_quarter_historical)

    fromdate = "2000Q1"
    todate = get_last_completed_quarter()

    # last year measure, most recent quarter 2023Q2
    yearquarterto = todate  # (sys.argv[1])
    yearquarterfrom = fromdate  # (sys.argv[1])

    renge_dates = get_days_between_quarters(yearquarterfrom, yearquarterto)
    print("renge_dates:")
    print(renge_dates["date"].max())
    print(renge_dates["date"].min())

    renge_dates["quarter"] = (
        pd.to_datetime(renge_dates["date"], format="%Y%m%d").dt.to_period("Q").astype(str)
    )
    renge_dates["q"] = renge_dates["quarter"].str.strip().str[-2:]
    renge_dates["year"] = renge_dates["quarter"].str.strip().str[:4]

    reference_pd = pd.read_parquet(path_daily_prec_reference)

    renge_dates_i = renge_dates
    renge_dates_i = renge_dates_i.date.to_list()

    print("Grouping the daily Temp in the selected year")
    df = []
    for day in renge_dates_i:
        print(f"Grouping day: {day}", flush=True)
        data = f"{path_daily_prec}/{day.year}/image_{day.year}-{str(day.month).zfill(2)}-{str(day.day).zfill(2)}.parquet.gzip"
        dfi = pd.read_parquet(data)[["pgid", "band_data"]].sort_values(by=["pgid"]).to_numpy()
        df.append(dfi)

    df = np.dstack(df)

    df_anomaly = build_anomaly(df, renge_dates_i, reference_pd)

    return df_anomaly


class GEEHeavyPrecipitationData(Dataset):
    """Handles loading and processing of  event data.

    Supports loading from local preprocessed files or via API.
    Includes methods for aggregating event data to the
    grid-quarter level.

    Attributes:
        data_key (str): Set to "gee_heavy-precipitation".
        local (bool): Indicates whether to use local dumps (True) or
            download data via the API (False).
        dataset_available (bool): Flag set to True after data is
            successfully loaded or checked by `load_data`.
    """

    data_key = "gee_heavy-precipitation"

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

    def download_data(self, grid: GlobalBaseGrid):
        """Downloads  data from the API.

        Downloads the  data from the API and saves it to the processing
        storage. The filename is based on the current date and time.

        Args:
            grid: GlobalBaseGrid instance

        Returns:
            str: The filename of the downloaded  data.
        """

        sources_path = self.storage.storage_paths["processing"]
        destination = f"{sources_path}/"

        if not os.path.exists(destination):
            os.makedirs(destination)

        # step1
        print("downloading data")

        out_status = daily_ERA5_download(
            sources_path, "precipitation", "total_precipitation_sum", "era5_precipitation", grid
        )
        if out_status is False:
            raise Exception("Error downloading precipitation data")
        # step2
        print("building threshold")
        dailyprecipitation_buildthreshold(sources_path)
        # step3
        print("building historical data")
        df_events = dailyprecipitation_buildhistorical(sources_path)

        # columns to uppercase
        df_events.columns = [col.upper() for col in df_events.columns]
        df_events.rename(columns={"DATE_ANOMALY": "EVENT_DATE"}, inplace=True)
        # reanme LAT with LATITUDE and LON with LONGITUDE

        df_events["YEAR"] = df_events["EVENT_DATE"].dt.year

        df_events["EVENT_TYPE"] = "heavy-precipitation"

        # "YEAR", "EVENT_TYPE", "LATITUDE", "LONGITUDE", "COUNT", "EVENT_DATE"
        # self.storage.save(df_events[self.columns], "processing", filename=self.filename)
        # beacuse those are the events I keep all the columns
        self.storage.save(df_events, "processing", filename=self.filename)

    def load_data(self, grid: GlobalBaseGrid):
        """Loads  data, checking for cached processing files first.

        Attempts to load a local  copy from the 'processing' storage
        including the last completed quarter. If not found:
        - If `self.local` is True, loads the raw dump specified in the config.
          Raises an error if the provided  dump does not fully cover the
          latest quarter.
        - If `self.local` is False, currently raises NotImplementedError (API access TBD).
        Saves the loaded raw/dump data to the processing storage.

        Args:
            grid: GlobalBaseGrid instance

        Returns:
            pd.DataFrame: The loaded  event data.
        """
        self.last_quarter_date = get_quarter("last", bounds="end")
        self.filename = f"gee_heavy-precipitation_{self.last_quarter_date.year}_Q{int(self.last_quarter_date.month / 3)}"
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
                self.download_data(grid)
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

            # quarter as number 1,2,3,4
            df_event_level["QUARTER"] = df_event_level["EVENT_DATE"].dt.quarter

            df_event_level["COUNT"] = 1

            # aggregate
            df_event_level_aggregated = (
                df_event_level[["PGID", "YEAR", "QUARTER", "COUNT"]]
                .groupby(["PGID", "YEAR", "QUARTER"])
                .sum()
            )
            df_event_level_aggregated = df_event_level_aggregated.reset_index()
            df_base = df_base.reset_index()
            df = df_base.merge(
                df_event_level_aggregated,
                left_on=["year", "quarter", "pgid"],
                right_on=["YEAR", "QUARTER", "PGID"],
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
    data = GEEHeavyPrecipitationData(local=False, config=config)
    # just load the current data

    df_gee_heavy_precipitation = data.load_data()
    print(df_gee_heavy_precipitation.head())
