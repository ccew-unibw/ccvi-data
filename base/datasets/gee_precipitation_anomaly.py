import os
import pandas as pd
import numpy as np
import ee
import warnings
import time
from datetime import datetime, timedelta

from base.objects import Dataset
from utils.index import get_quarter


##########################################
##########################################


warnings.filterwarnings("ignore")

# TODO: Finalize download


class ZsGEE:
    """zonal statistics from gee"""

    def __init__(
        self,
    ):
        # some init here
        self.some_var_here = ""

    def get_dataframe(self):
        startDate = self.startDate  # '2000-03-01' # time period of interest beginning date
        interval = 1  # time window length
        intervalUnit = "month"  #'month'  # unit of time e.g. 'year', 'month', 'day'
        intervalCount = 1  # 275 # number of time windows in the series
        dataset = ee.ImageCollection(self.satellite).select(self.bands)
        temporalReducer = ee.Reducer.mean()  # how to reduce images in time window
        spatialReducers = ee.Reducer.mean()  # how to reduce images in time window

        # Get time window index sequence.
        intervals = ee.List.sequence(0, intervalCount - 1, interval)

        # Map reductions over index sequence to calculate statistics for each interval.
        def a(i):
            # Calculate temporal composite.
            startRangeL = ee.Date(startDate).advance(i, intervalUnit)
            endRangeL = startRangeL.advance(interval, intervalUnit)
            temporalStat = dataset.filterDate(startRangeL, endRangeL).reduce(temporalReducer)

            # Calculate zonal statistics.
            statsL = temporalStat.reduceRegions(
                collection=self.asset,
                reducer=spatialReducers,
                scale=dataset.first().projection().nominalScale().getInfo(),
                crs=dataset.first().projection(),
            )

            # Set start date as a feature property.

            def b(feature):
                #  or 'YYYY-MM-dd'
                return feature.set({"composite_start": startRangeL.format("YYYYMMdd")})

            return statsL.map(b)

        zonalStatsL = intervals.map(a)

        zonalStatsL = ee.FeatureCollection(zonalStatsL).flatten()

        output = fc_to_dict(zonalStatsL).getInfo()
        output = pad_dict_list(output, np.nan)
        out_pd = pd.DataFrame(output)
        out_pd["quarter"] = (
            pd.to_datetime(out_pd["composite_start"], format="%Y%m%d").dt.to_period("Q").astype(str)
        )
        out_pd["q"] = out_pd["quarter"].str.strip().str[-2:]
        # if not os.path.exists("../latlon.parquet.gzip"):
        #    out_pd[["pgid","lat","lon"]].to_parquet("latlon.parquet.gzip",compression="gzip")
        return out_pd[["pgid", "composite_start", "mean", "quarter", "q"]]


def fc_to_dict(fc):
    prop_names = fc.first().propertyNames()
    prop_lists = fc.reduceColumns(
        reducer=ee.Reducer.toList().repeat(prop_names.size()), selectors=prop_names
    ).get("list")

    return ee.Dictionary.fromLists(prop_names, prop_lists)


def pad_dict_list(dict_list, padel):
    lmax = 0
    for lname in dict_list.keys():
        lmax = max(lmax, len(dict_list[lname]))
    for lname in dict_list.keys():
        ll = len(dict_list[lname])
        if ll < lmax:
            dict_list[lname] += [padel] * (lmax - ll)
    return dict_list


def f(x):
    try:
        startDate = x["startdate"]
        temp_files = x["temp_files"]
        credentials = x["GOOGLEClient"]
        asset = x["asset"]

        if not os.path.exists(f"{temp_files}/raw/era5_precipitation/{startDate[:4]}"):
            os.makedirs(f"{temp_files}/raw/era5_precipitation/{startDate[:4]}")

        outfile = (
            f"{temp_files}raw/era5_precipitation/{startDate[:4]}/image_{startDate}.parquet.gzip"
        )
        print(f"Checking this directory: {outfile}", end="\r")

        doneIt = False
        if not os.path.exists(outfile):
            print(f"downloading: {outfile}")

            # ee.Initialize(credentials)
            credentials._authorize()
            my_grid = ee.FeatureCollection(asset)

            # while not doneIt:

            myPanelData = ZsGEE()
            myPanelData.startDate = startDate
            myPanelData.satellite = "ECMWF/ERA5_LAND/MONTHLY_AGGR"
            myPanelData.bands = "total_precipitation_sum"
            myPanelData.asset = my_grid
            df = myPanelData.get_dataframe()
            doneIt = True

            df.to_parquet(outfile, compression="GZIP")

        # else:
        # print(f"Exists in cache!!: {outfile}")

        # raise ValueError("An error occurred!")  # Example error
    except Exception as e:
        # Catch the exception and re-raise it to be caught in the main process
        raise e


# TODO: Fill up the data until the current quarter.
# TODO: Save data to the right dir with correct naming convention
# TODO: Check if all columns are named correctly.


class PrecipitationAnomaly(object):
    def __init__(self, dir: str, start_year: int, start_baseline: int, end_baseline: int, lag: int):
        """
        Init object instance.

        Parameters
        ----------
            dir: str
                Directory containing the subdirectories for which all file paths should be returned.

            start_year: int
                The year from which to start the indicator calculation.

            start_baseline: int
                The start year of the reference/baseline period. Year is inclusinve.

            end_baseline: int
                The final year of the reference/baseline period. Year is inclusive.

            lag: int
                Lag in years for the calculation of the current value. Here 10 years.

            output_path: str
                Valid path and file name where the longterm precipitation anomaly
                should be saved to.

        Returns
        -------
            None
        """

        self.dir = dir

        self.start_year = start_year
        self.end_year = datetime.now().year - 1

        self.current_files = []

        self.start_baseline = start_baseline
        self.end_baseline = end_baseline

        self.baseline_files = []

        self.lag = lag

        # self.output_path = os.path.join(get_storage_location("climate_raw"), "CLI_longterm_precipitation-anomaly_raw.parquet")

        return None

    def retrieve_all_file_paths(
        self,
    ) -> None:
        """
        Read all files in a list of subdirectories.

        Parameters
        ----------
            self: object

        Returns
        -------
            self.baseline_files: list
                List of all valid files in all subdirectiries within self.dir within the baseline period.

            self.current_files: list
                List of all valid files in all subdirectiries within self.dir within the current period,
                i.e. from 2000 to the last complete year.
        """

        # Filter paths based on start and end year for baseline and for current files.
        baseline_files = [
            os.path.join(self.dir, dir)
            for dir in os.listdir(self.dir)
            if self.start_baseline <= int(os.path.basename(dir)) <= self.end_baseline
        ]
        current_files = [
            os.path.join(self.dir, dir)
            for dir in os.listdir(self.dir)
            if self.start_year <= int(os.path.basename(dir)) <= self.end_year
        ]

        # Create a list of valid, absolute paths to the files.
        for baseline_file_dir in baseline_files:
            self.baseline_files += [
                os.path.join(baseline_file_dir, file) for file in os.listdir(baseline_file_dir)
            ]

        for current_file_dir in current_files:
            self.current_files += [
                os.path.join(current_file_dir, file) for file in os.listdir(current_file_dir)
            ]

        return None

    def _add_year(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add year and quarter to a data frame that has the column composite start"""

        assert "composite_start" in data.columns.to_list(), (
            "Please make sure that the data you download from GEE for the monthly precipitation has a column called `composite_start`."
        )

        data["composite_start"] = pd.to_datetime(data["composite_start"], format="%Y%m%d")
        data["year"] = data["composite_start"].dt.year

        return data

    def aggregate_to_annual(self) -> None:
        """
        Read files for baseline and current period and aggregate from monthly to annual frequency.

        Returns
        -------
            self.baseline: pd.DataFrame
                Baseline period aggregated to year-quarter-pgid combination

            self.current: pd.DataFrame
                Current period aggreagted to year-quarter-pgid combination
        """

        # pandas.read_parquet() can handle a list of files and then handles
        # the concatination into one single dataframe.
        self.baseline = pd.read_parquet(self.baseline_files)
        self.current = pd.read_parquet(self.current_files)

        self.baseline = self._add_year(self.baseline)
        self.current = self._add_year(self.current)

        # Rename to avoid confusion when aggregating.
        self.baseline = self.baseline.rename(columns={"mean": "total_precipitation_sum"})
        self.current = self.current.rename(columns={"mean": "total_precipitation_sum"})

        # Aggregate from months to years
        self.baseline = (
            self.baseline.groupby(["pgid"])
            .agg({"total_precipitation_sum": "mean"})
            .reset_index()
        )
        self.baseline = self.baseline.rename(
            columns={"total_precipitation_sum": "total_precipitation_sum_baseline"}
        )

        self.current = (
            self.current.groupby(["pgid", "quarter", "year"])
            .agg({"total_precipitation_sum": "mean"})
            .reset_index()
        )

        # Sort the data by pgid and quarter to ensure proper order
        self.current = self.current.sort_values(['pgid', 'quarter'])

        # Calculate 12-month (4-quarter) rolling average grouped by pgid
        self.current['total_precipitation_12m_avg'] = (
            self.current.groupby('pgid')['total_precipitation_sum']
            .rolling(window=4, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )


        # Alternative method if you want to require at least 4 quarters of data
        # (this will show NaN for the first 3 quarters of each pgid)
        self.current['total_precipitation_12m_avg_strict'] = (
            self.current.groupby('pgid')['total_precipitation_sum']
            .rolling(window=4, min_periods=4)
            .mean()
            .reset_index(level=0, drop=True)
        )


        return None

    def calculate_baseline(self) -> None:
        """
        Calculate the baseline mean of sum of preciptiation per PGID.
        """

        # Calculate average over sum per year per pgid
        #self.baseline = (
        #    self.baseline.groupby("pgid").agg({"total_precipitation_sum": "mean"}).reset_index()
        #)

        #self.baseline = self.baseline.rename(
        #    columns={"total_precipitation_sum": "total_precipitation_mean_baseline"}
        #)

        return None

    def calculate_anomaly(self) -> None:
        """
        Calculate the anomaly per PGID-year combination.

        Returns
        -------
            self.anomalies: pd.DataFrame
                Data Frame with the year-pgid anomaly combinations.
        """

        # Combine the current and basline data
        # Each row will be a year-pgid combination coming from the current data.
        # The baseline is on pgid level and will thus be repeated for each year.
        results = pd.merge(left=self.current, right=self.baseline, how="left", on=["pgid"])

        # Calculate the annual anomaly per pgid
        # TODO: Change naming for all occurences
        results["CLI_longterm_precipitation-anomaly_raw"] = (
            results["total_precipitation_12m_avg"] / results["total_precipitation_sum_baseline"]
        ) - 1

        # Calculate the absolute value for the percentage anomaly
        # to ensure that more is always worse.
        results["CLI_longterm_precipitation-anomaly_raw"] = results[
            "CLI_longterm_precipitation-anomaly_raw"
        ].abs()

        # For each year-pgid combination get the values for the past 10 years.

        for i in range(1, self.lag):
            results[f"CLI_longterm_precipitation-anomaly_raw{i}"] = results.loc[
                :, "CLI_longterm_precipitation-anomaly_raw"
            ].shift(i)

        
        # Filter the data to only contain everyting from the starting year onward.
        results = results.query(f"year >= {self.start_year}")

        # Prepare dataset for row-wise mean calculation, i.e. calculation of
        # mean anomaly per year-pgid combination
        results = results.set_index(["pgid", "year", "quarter"])

        columns = [
            col for col in results.columns if "CLI_longterm_precipitation-anomaly_raw" in col
        ]
        results = results.loc[:, columns]

        ## Calculate row-wise mean as we have per row (i.e. per year-pgid) combination
        ## the values for the past 10 years.
        ## Rename the resulting column so the naming is in line with the google sheets.
        results = (
            results.agg("mean", axis=1)
            .reset_index()
            .rename(columns={0: "CLI_longterm_precipitation-anomaly_raw"})
        )

        # Add quarters to final data frame
        # TODO: Delete
        # quarters = pd.DataFrame({"quarter": [1,2,3,4]})
        # results = pd.merge(results, quarters, how = "cross")

        self.anomaly = results

        return None


##########################################
##########################################


class GEEPrecipitationAnomaly(Dataset):
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

    data_key = "gee_precipitation-anomaly"

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

        print("... Downloading GEE mmonthly Precipitation data...")
        from dotenv import load_dotenv

        load_dotenv()
        BASE_GRID_ASSET = os.getenv("BASE_GRID_ASSET")
        from utils.gee import GEEClient

        warnings.filterwarnings("ignore")

        sources_path = self.storage.storage_paths["processing"]
        destination = f"{sources_path}/"

        if not os.path.exists(destination):
            os.makedirs(destination)

        client = GEEClient()

        asset = BASE_GRID_ASSET

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

        start = time.time()
        sources_path = self.storage.storage_paths["processing"]
        temp_files = f"{sources_path}/ERA5/precipitation_monthly/"

        if not os.path.exists(temp_files):
            os.makedirs(temp_files)
        from_year = 1951

        ####DOWNLOAD DATA FROM GEE

        params = []
        startDate = datetime(1951, 1, 1)
        endDate = datetime(year, month, day)
        endDate = endDate + timedelta(days=1)

        # Getting List of Days using pandas
        datesRange = pd.date_range(startDate, endDate - timedelta(days=1), freq="MS")

        for datei in datesRange:
            parx = {
                "startdate": datei.strftime("%Y-%m-%d"),
                "temp_files": temp_files,
                "GOOGLEClient": client,
                "asset": asset,
            }
            params.append(parx)

        # Fix as parallel download seems not to be working any longer from GEE
        for x in params:
            f(x)

        ####################

        precip_anomaly = PrecipitationAnomaly(
            dir=f"{self.storage.storage_paths['processing']}/ERA5/precipitation_monthly/raw/era5_precipitation/",
            start_baseline=1950,
            end_baseline=1980,
            lag=10,
            start_year=2000,
        )

        precip_anomaly.retrieve_all_file_paths()
        precip_anomaly.aggregate_to_annual()
        precip_anomaly.calculate_baseline()
        precip_anomaly.calculate_anomaly()
        precip_anomaly.anomaly.to_parquet(
            f"{self.storage.storage_paths['processing']}/{self.filename}.parquet"
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
        self.filename = f"gee_precipitation-anomaly_{self.last_quarter_date.year}_Q{int(self.last_quarter_date.month / 3)}"
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
            #quarter to number 1,2,3
            df_event_level["quarter"] = df_event_level["quarter"].apply(
                lambda x: int(x.split("Q")[-1])
            )
            df = pd.merge(
                df_base.reset_index().set_index(["pgid", "year", "quarter"]),
                df_event_level.set_index(["pgid", "year", "quarter"]),
                how="left",
                left_index=True,
                right_index=True,
            )

            df = df.fillna(method="ffill")

            df.to_parquet(fp_preprocessed)
        return df


# test class
if __name__ == "__main__":
    from base.objects import ConfigParser

    config = ConfigParser()

    # Example usage
    data = GEEPrecipitationAnomaly(local=False, config=config)
    # just load the current data

    basedata = data.load_data()
    print(basedata.head())
