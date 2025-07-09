# add python path to the base directory
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import multiprocessing
from multiprocessing import Pool


from base.objects import Dataset, GlobalBaseGrid, console
from utils.index import get_quarter
from utils.gee_daily import daily_ERA5_download


def calculate_humidex_indicator(row):
    """
    Calculate the humidex using temperature and dew point data.

    Parameters:
    row (pd.Series): Series containing temperature and dew point values in Celsius.

    Returns:
    float: Calculated humidex value.
    """
    temp = row["temp"]
    dewpoint = row["dewpoint"]

    # Humidex calculation
    humidex = (
        temp
        - 273.15
        + 0.5555
        * (6.11 * np.exp(5417.7530 * ((1 / 273.15) - (1 / (273.15 + dewpoint - 273.15)))) - 10)
    )
    humidex = humidex + 273.15

    return humidex


def calculate_humidex(temp_df, dewpoint_df):
    """
    Calculate the humidex using temperature and dew point data.

    Parameters:
    temp_df (pd.DataFrame): DataFrame containing temperature values in Celsius.
    dewpoint_df (pd.DataFrame): DataFrame containing dew point values in Celsius.

    Returns:
    pd.DataFrame: DataFrame containing the calculated humidex values.
    """

    temp_df = temp_df[
        ["lat", "lon", "band_data", "pgid", "iso3", "quarter", "q"]
    ].copy()  # Explicit copy
    temp_df.rename(columns={"band_data": "temp"}, inplace=True)

    dewpoint_df = dewpoint_df[
        ["lat", "lon", "band_data", "pgid", "iso3", "quarter", "q"]
    ].copy()  # Explicit copy
    dewpoint_df.rename(columns={"band_data": "dewpoint"}, inplace=True)

    humidex_df = pd.merge(
        temp_df, dewpoint_df, how="inner", on=["lat", "lon", "pgid", "iso3", "quarter", "q"]
    )

    # Ensure the dataframes have the same dimensions
    if temp_df.shape != dewpoint_df.shape:
        raise ValueError("Temperature and Dew Point DataFrames must have the same dimensions.")

    # Humidex calculation
    humidex_df["band_data"] = humidex_df.apply(calculate_humidex_indicator, axis=1)

    humidex_df = humidex_df[
        ["lat", "lon", "band_data", "pgid", "iso3", "quarter", "q"]
    ].copy()  # Explicit copy

    return humidex_df


def combine_to_humidex(x):
    dwp_file = x["dwp_file"]
    temp = x["temp"]
    humidex = x["humidex"]
    dwp_path = dwp_file
    dwp_path_year = dwp_path.split("/")[-2]
    dwp_path_filename = dwp_path.split("/")[-1]
    temp_path = f"{temp}/{dwp_path_year}/{dwp_path_filename}"
    humidex_path = f"{humidex}/{dwp_path_year}/{dwp_path_filename}"

    if not os.path.exists(f"{humidex}/{dwp_path_year}"):
        os.makedirs(f"{humidex}/{dwp_path_year}")

    if not os.path.exists(humidex_path):
        df_dwp = pd.read_parquet(dwp_file)
        df_temp = pd.read_parquet(temp_path)
        df_humidex = calculate_humidex(df_temp, df_dwp)
        print(humidex_path)
        df_humidex.to_parquet(humidex_path)
    else:
        print(f"File {humidex_path} already exists, skipping calculation.")


def dailytemperature_calculate_humidex(sources_path):
    dwp = f"{sources_path}/dewpoint_temperature_2m_max/raw/era5_dewpoint_temperature_2m_max"
    temp = f"{sources_path}/temperature/raw/era5_temperature_max"
    humidex = f"{sources_path}/humidex/raw/humidex"
    if not os.path.exists(humidex):
        os.makedirs(humidex)

    # Scan the dwp directory and its subdirectories looking for the *parquet* files
    dwp_files = []
    for root, dirs, files in os.walk(dwp):
        for file in files:
            if file.endswith(".parquet.gzip"):
                dwp_files.append(os.path.join(root, file))

    multiprocessing_params = []
    for dwp_file in dwp_files:
        if dwp_file.endswith(".parquet.gzip"):
            multiprocessing_params.append(
                {
                    "sources_path": sources_path,
                    "dwp_file": dwp_file,
                    "temp": temp,
                    "humidex": humidex,
                }
            )
    num_cores = multiprocessing.cpu_count()
    with Pool(num_cores - 1) as p:
        p.map(combine_to_humidex, multiprocessing_params)


def build_percentile_per_pgid(arr):
    out = []
    for i in range(arr.shape[0]):
        if arr[i, 0, :].max() == arr[i, 0, :].min():
            pgid = int(arr[i, 0, 0])
            percentile_distribution_95 = np.percentile(arr[i, 1, :], 95)
            percentile_distribution_90 = np.percentile(arr[i, 1, :], 90)
            percentile_distribution_75 = np.percentile(arr[i, 1, :], 75)
            percentile_distribution_25 = np.percentile(arr[i, 1, :], 25)
            out.append(
                [
                    pgid,
                    percentile_distribution_95,
                    percentile_distribution_90,
                    percentile_distribution_75,
                    percentile_distribution_25,
                ]
            )
        else:
            print("some errors in the pgid")
    out = pd.DataFrame(out, columns=["pgid", "95p", "90p", "75p", "25p"])
    return out


def get_ranges(centerdate, sources_path):
    r = pd.date_range(
        (centerdate - timedelta(days=15)), (centerdate + timedelta(days=15)), freq="d"
    )
    list_of_month_and_days = []
    for el in r:
        list_of_month_and_days.append([el.day, el.month])

    list_of_month_and_days = pd.DataFrame(list_of_month_and_days, columns=["day", "month"])
    list_of_month_and_days = list_of_month_and_days.drop_duplicates()
    list_of_layers = []
    for year in range(1951, 1981):
        for el in list_of_month_and_days.values:
            try:
                file = f"{sources_path}/temperature/raw/era5_temperature_max/{year}/image_{year}-{str(el[1]).zfill(2)}-{str(el[0]).zfill(2)}.parquet.gzip"
                df = pd.read_parquet(file)[["pgid", "band_data"]]
                df = df.fillna(0)
                df = df.sort_values(by=["pgid"])
                df = df.to_numpy()

                list_of_layers.append(df)
            except Exception as e:
                print(e)
                pass

    # Check all the df has the same length:
    if len(set([len(x) for x in list_of_layers])) != 1:
        print("some error in the length of the layers")
        return None
    list_of_layers = np.dstack(list_of_layers)
    df = build_percentile_per_pgid(list_of_layers)
    return df


def buildthreshold_parallel(x):
    reference_directory = x["reference_directory"]
    day = x["day"]
    sources_path = x["sources_path"]

    try:
        if not os.path.exists(
            f"{reference_directory}/{str(day.month).zfill(2)}_{str(day.day).zfill(2)}.parquet.gzip"
        ):
            try:
                reference = get_ranges(day, sources_path)

                reference.to_parquet(
                    f"{reference_directory}/{str(day.month).zfill(2)}_{str(day.day).zfill(2)}.parquet.gzip",
                    compression="gzip",
                )
            except Exception as e:
                print(e)
                pass
        else:
            print(
                f"Exists in cache !!! {reference_directory}/{str(day.month).zfill(2)}_{str(day.day).zfill(2)}.parquet.gzip"
            )
    except Exception as e:
        print(e)
        pass


def dailytemperature_buildthreshold(sources_path):
    startDate = datetime(2024, 1, 1)
    endDate = datetime(2024, 12, 31)
    renge_dates = pd.date_range(startDate, endDate, freq="d")
    reference_directory = f"{sources_path}/temperature/raw/era5_temperature_max_reference"
    if not os.path.exists(reference_directory):
        os.makedirs(reference_directory)

    params = []
    for day in renge_dates:
        parx = {
            "reference_directory": reference_directory,
            "day": day,
            "sources_path": sources_path,
        }
        params.append(parx)

    num_cores = multiprocessing.cpu_count()
    # with Pool(1) as p:
    with Pool(num_cores - 1) as p:
        try:
            p.map(buildthreshold_parallel, params)
        except Exception as e:
            # Terminate the pool and handle the error
            p.terminate()

            print(f"An error occurred: {e}")
            return False


def get_last_completed_quarter():
    from datetime import datetime, timedelta

    # Get the current date
    current_date = datetime.now()

    # Calculate the previous quarter
    previous_quarter = current_date - timedelta(days=365 / 4)

    # Format the result as "YYYYQn"
    result = f"{previous_quarter.year}Q{((previous_quarter.month - 1) // 3) + 1}"

    return result


def get_days_between_quarters(start_quarter, end_quarter):
    # Convert the input quarters to pandas Period objects
    start_period = pd.Period(start_quarter, freq="Q")
    end_period = pd.Period(end_quarter, freq="Q")

    # Generate a date range between the start and end quarters
    date_range = pd.date_range(start=start_period.start_time, end=end_period.end_time, freq="D")

    # Create DataFrame
    df = pd.DataFrame({"date": date_range})

    return df


def count_anomaly_and_magnitude(arr_temp, arr_humidex, pgid, renge_dates, reference_pgid):
    print(pgid, flush=True)

    t_95 = []

    # building years vectors: (365/366 days) of the 90h, 75h, 25h percentile for the specific pgid
    for i in range(0, len(arr_temp)):
        day = renge_dates[i]
        dataref = f"{str(day.month).zfill(2)}_{str(day.day).zfill(2)}"
        reference = reference_pgid.loc[reference_pgid.dateref == dataref]

        t_95.append(reference["95p"].values[0])

    # vectors series of referece values
    t_95 = np.array(t_95)

    # Heatwave: period of 3 consecutive days with maximum temperature (Tmax) above the daily threshold
    # for the reference period 1981–2010. The threshold is
    # defined as the 90th percentile of daily maxima temperature, centered on a 31 day window
    anomaly = (arr_temp > t_95) & ((arr_temp > (35 + 273.15)) | (arr_humidex > (40 + 273.15)))
    anomaly = anomaly * 1

    # recording the position (day) of the anomaly when are >=3 cosecutive days
    heatwave_list = []
    heatwave = []
    out = []
    for i in range(0, anomaly.shape[0]):
        if anomaly[i] == 1:
            heatwave.append(i)
        elif anomaly[i] == 0:
            if len(heatwave) >= 3:
                heatwave_list.append(heatwave)
                heatwave = []
            else:
                heatwave = []

    for iel in heatwave_list:
        for i in iel:
            date_anomaly = renge_dates[i]
            temperature = arr_temp[i]
            out.append([date_anomaly, temperature])

    out = pd.DataFrame(out, columns=["date_anomaly", "temperature"])
    out["pgid"] = pgid
    return out


resultdict = []


def log_result(result):
    resultdict.append(result)


def build_anomaly(arr, arr_h, renge_dates, reference_pd):
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
            list_of_commands.append(
                [arr[i, 1, :], arr_h[i, 1, :], pgid, renge_dates, reference_pgid]
            )

    print("start creating pool", flush=True)
    num_cores = multiprocessing.cpu_count()
    print(f"Number of cores: {num_cores}")
    pool = Pool(num_cores)
    for x_y_z in list_of_commands:
        pool.apply_async(count_anomaly_and_magnitude, args=(x_y_z), callback=log_result)

    print("pool close", flush=True)
    pool.close()
    print("pool join", flush=True)
    pool.join()

    out = pd.concat(resultdict)

    return out


def dailytemperature_buildhistorical(sources_path):
    path_daily_temp = f"{sources_path}/temperature/raw/era5_temperature_max/"
    path_daily_humidex = f"{sources_path}/humidex/raw/humidex/"
    path_daily_temp_reference = f"{sources_path}/temperature/raw/era5_temperature_max_reference/"

    path_quarter_historical = f"{sources_path}/temperature/raw/anomaly/"

    if not os.path.exists(path_daily_temp):
        os.makedirs(path_daily_temp)

    if not os.path.exists(path_daily_temp_reference):
        os.makedirs(path_daily_temp_reference)

    if not os.path.exists(path_quarter_historical):
        os.makedirs(path_quarter_historical)

    fromdate = "2000Q1"
    todate = get_last_completed_quarter()
    out_dir = path_quarter_historical

    yearquarterto = todate  # (sys.argv[1])
    yearquarterfrom = fromdate  # (sys.argv[1])

    renge_dates = get_days_between_quarters(yearquarterfrom, yearquarterto)

    print(renge_dates["date"].max())
    print(renge_dates["date"].min())

    renge_dates["quarter"] = (
        pd.to_datetime(renge_dates["date"], format="%Y%m%d").dt.to_period("Q").astype(str)
    )
    renge_dates["q"] = renge_dates["quarter"].str.strip().str[-2:]
    renge_dates["year"] = renge_dates["quarter"].str.strip().str[:4]

    reference_pd = []
    for filename in os.listdir(path_daily_temp_reference):
        # print(filename)
        f = os.path.join(path_daily_temp_reference, filename)
        df = pd.read_parquet(f)
        df = df.fillna(0)
        month = filename[0:2]
        day = filename[3:5]
        df["dateref"] = f"{month}_{day}"
        reference_pd.append(df)

    reference_pd = pd.concat(reference_pd)

    renge_dates_i = renge_dates
    renge_dates_i = renge_dates_i.date.to_list()

    # Grouping the daily Tmax in the selected year
    df = []
    for day in renge_dates_i:
        print(day)
        data = f"{path_daily_temp}/{day.year}/image_{day.year}-{str(day.month).zfill(2)}-{str(day.day).zfill(2)}.parquet.gzip"
        dfi = pd.read_parquet(data)[["pgid", "band_data"]].sort_values(by=["pgid"]).to_numpy()
        df.append(dfi)

    df = np.dstack(df)

    df_h = []
    for day in renge_dates_i:
        print(day)
        data = f"{path_daily_humidex}/{day.year}/image_{day.year}-{str(day.month).zfill(2)}-{str(day.day).zfill(2)}.parquet.gzip"
        dfi = pd.read_parquet(data)[["pgid", "band_data"]].sort_values(by=["pgid"]).to_numpy()
        df_h.append(dfi)

    df_h = np.dstack(df_h)

    df_anomaly = build_anomaly(df, df_h, renge_dates_i, reference_pd)

    return df_anomaly


class GEEHeatwaveData(Dataset):
    """Handles loading and processing of  event data.

    Supports loading from local preprocessed files or via API.
    Includes methods for aggregating event data to the
    grid-quarter level.

    Attributes:
        data_key (str): Set to "gee_heatwave".
        local (bool): Indicates whether to use local dumps (True) or
            download data via the API (False).
        dataset_available (bool): Flag set to True after data is
            successfully loaded or checked by `load_data`.
    """

    data_key = "gee_heatwave"

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
            sources_path, "temperature", "temperature_2m_max", "era5_temperature_max", grid
        )
        if out_status is False:
            raise Exception("Error downloading temperature data")
        daily_ERA5_download(
            sources_path,
            "dewpoint_temperature_2m_max",
            "dewpoint_temperature_2m_max",
            "era5_dewpoint_temperature_2m_max",
            grid,
        )
        # step2
        print("Calculate humidex")
        dailytemperature_calculate_humidex(sources_path)
        # step 3
        print("Calculate buildthreshold")
        dailytemperature_buildthreshold(sources_path)
        # step3
        print("building historical data")
        df_events = dailytemperature_buildhistorical(sources_path)

        # columns to uppercase
        df_events.columns = [col.upper() for col in df_events.columns]
        df_events.rename(columns={"DATE_ANOMALY": "EVENT_DATE"}, inplace=True)
        # reanme LAT with LATITUDE and LON with LONGITUDE

        df_events["YEAR"] = df_events["EVENT_DATE"].dt.year

        df_events["EVENT_TYPE"] = "heatwave"

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
        self.filename = (
            f"gee_heatwave_{self.last_quarter_date.year}_Q{int(self.last_quarter_date.month / 3)}"
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
    data = GEEHeatwaveData(local=False, config=config)
    # just load the current data

    df_gee_heatwave = data.load_data()
    print(df_gee_heatwave.head())
