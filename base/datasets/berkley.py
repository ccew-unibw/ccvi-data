# add python path to the base directory
import os
import pandas as pd
import requests
import math

import numpy as np
from datetime import date
import xarray as xr

from base.objects import Dataset
from utils.index import get_quarter


def download_berkley(storage: str, force: bool) -> str:
    url = "https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Gridded/Land_and_Ocean_LatLong1.nc"
    dir = os.path.join(storage, "data")
    os.makedirs(dir, exist_ok=True)
    fp = os.path.join(dir, f"berkley_temp_{get_quarter().isoformat()}.nc")
    if force or not os.path.exists(
        fp
    ):  # by default data downloaded in one quarter is read, if forced newly downloaded
        response = requests.get(url)
        response.raise_for_status()  # we want an error if download doesnt work
        with open(fp, "wb") as f:
            for chunk in response.iter_content(chunk_size=2048):
                f.write(chunk)
    else:
        print(
            "... Berkley data already downloaded this quarter.",
            "If downloading a new version is required, use --force argument.",
        )
    # output some information on data...
    ds = xr.open_dataset(fp)
    max_time = ds.time.max()
    year = math.floor(max_time)
    month = math.ceil((max_time % 1) * 12)
    print(f"... Berkey data with information up to {month}-{year} in storage.")
    return fp


def convert_berkley_time(time: float) -> date:
    """converts berkley time float to date fo first day(!) of respective month"""
    year = math.floor(time)
    month = math.ceil((time % 1) * 12)
    return date(year, month, 1)


def process_temp_anomaly_data(
    fp_in: str,
    year_min: int = 2000,
    years_average: int = 10,
    baseline: tuple[int, int] = (1850, 1900),
) -> str:
    """creates the raw temp anomaly indicator from berkley data"""
    # read data, adjust baseline, select temperature
    berkley = xr.open_dataset(fp_in)
    berkley = berkley.rename({"latitude": "lat", "longitude": "lon"})
    berkley["temperature_adjusted"] = berkley.temperature - berkley.temperature.sel(
        time=slice(baseline[0], baseline[1])
    ).mean("time")
    berkley["time"] = np.vectorize(convert_berkley_time)(berkley.time)

    # limit to last full quarter in data and select adjusted temp layer
    offset = -1  # start with last quarter
    q_start = get_quarter(offset)
    while berkley.time.max() < q_start:
        if (
            offset == -3
        ):  # offset should always be either -1 or -2 - berkley data has only a few months lag
            raise ValueError(
                f"Data not sufficient to generate data for Q{q_start.month // 3 + 1}-{q_start.year}.",
                "Check the Berkley data, something is probably wrong.",
            )
        else:
            offset -= 1
            q_start = get_quarter(offset)
    berkley_end = get_quarter(offset, "end")
    berkley_temp = berkley.sel(time=slice(berkley_end)).temperature_adjusted

    # linearly interpolate data to our resolution
    print(":hot_face: Interpolating...")
    lat_new = np.arange(-89.75, 90, 0.5)
    lon_new = np.arange(-179.75, 180, 0.5)
    berkley_temp = berkley_temp.interp(lat=lat_new, lon=lon_new, method="linear")

    # create 10-year averages
    print(f":hot_face: Calculating {years_average}-year averages...")
    berkley_temp_average = berkley_temp.rolling(time=12 * years_average).mean()

    # limit to quarters - select 10-year means from last month in each quarter and convert to first month for consistency and merging
    berkley_temp_average = berkley_temp_average.sel(
        time=xr.date_range(f"{year_min}-03-01", berkley.time.values.max(), freq="3MS")
    )
    berkley_temp_average["time"] = np.vectorize(lambda x: date(x.year, x.month - 2, 1))(
        berkley_temp_average.time
    )
    df_berkley = berkley_temp_average.to_dataframe()

    return df_berkley.reset_index()


class BERKLEYData(Dataset):
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

    data_key = "berkley"

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

        print("... Downloading Berkley temperature data...")
        fp_berkley = download_berkley(sources_path, force=True)

        # Step 2: Process the data and store it in the output as a parquet file. This will be
        print("... Processing Berkley temperature data...")
        df_berkley = process_temp_anomaly_data(fp_berkley, years_average=30)

        print(
            "... Processing CLI_longterm_temperature-anomaly raw data... [bold green]DONE[/bold green]"
        )

        df_berkley.to_parquet(f"{self.storage.storage_paths['processing']}/{self.filename}.parquet")

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
            f"berkley_{self.last_quarter_date.year}_Q{int(self.last_quarter_date.month / 3)}"
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

            df = df_base
            df_berkley = df_event_level
            index = df.index.names
            df = df.reset_index().merge(df_berkley, how="left", on=["lat", "lon", "time"])
            df = df.rename(columns={"temperature_adjusted": "count"})

            df = df[["pgid", "year", "quarter", "lat", "lon", "count", "time"]]

            df["time"] = pd.to_datetime(df["time"])

            # df=df.fillna(0)

            df.to_parquet(fp_preprocessed)
        return df


# test class
if __name__ == "__main__":
    from base.objects import ConfigParser

    config = ConfigParser()

    # Example usage
    data = BERKLEYData(local=False, config=config)
    # just load the current data

    df_berkley = data.load_data()
    print(df_berkley.head())
