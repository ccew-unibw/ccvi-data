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
from utils.spatial_operations import coords_to_pgid


class BERKLEYData(Dataset):
    """Handles loading and processing of  event data.

    Supports loading from local preprocessed files or via API.
    Includes methods for aggregating event data to the
    grid-quarter level.

    Attributes:
        data_key (str): Set to "berkley".
        local (bool): Set to False, as data is downloaded from source.
        dataset_available (bool): Flag set to True after data is
            successfully loaded or checked by `load_data`.
    """

    data_key = "berkley"
    local = False

    def __init__(self, *args, **kwargs):
        """Initializes the data source.

        Sets the operation mode (local file vs API) and calls the Dataset
        initializer to setup config and storage.

        Args:
            config (ConfigParser): An initialized ConfigParser instance.
        """
        self.dataset_available = False
        super().__init__(*args, **kwargs)

    def load_data(self) -> dict[str, str]:
        """Downloads data from the API.

        Downloads the data from the API and saves it to the processing storage.
        The filename is based on the first day of the respective quarter. Sets
        `dataset_available` to True.
        """
        # the new high-res based version does not have enough coverage early on to create the preindustrial baseline, using the old version for this
        urls = {
            "baseline": "https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Gridded/Land_and_Ocean_LatLong1.nc",
            "high-res": "https://storage.googleapis.com/berkeley-earth-temperature-hr/global/gridded/Global_TAVG_Gridded_1deg.nc",
        }
        fps = {}
        for version in urls:
            url = urls[version]
            # static filename for old data just used for baseline
            filename = f"berkley_{version}"
            if version == "high-res":
                filename = filename + f"_{get_quarter('last').isoformat()}"
            fp = self.storage.build_filepath(
                "processing", filename=filename, subfolder="data", filetype=".nc"
            )
            fps[version] = fp
            # by default data downloaded in one quarter is read, unless regenerate is set
            if self.regenerate["data"] or not os.path.exists(fp):
                response = requests.get(url)
                response.raise_for_status()  # we want an error if download doesnt work
                with open(fp, "wb") as f:
                    for chunk in response.iter_content(chunk_size=2048):
                        f.write(chunk)
            else:
                self.console.print(
                    f"Berkley data (version {version}) already downloaded this quarter.",
                    "If a new download is required, set regenerate['data']=True.",
                )
        # output some information on data...
        ds = xr.open_dataset(fps["high-res"])
        max_time = ds.time.max()
        year = math.floor(max_time)
        month = math.ceil((max_time % 1) * 12)
        self.console.print(f"Berkey data with information up to {month}-{year} in storage.")
        self.dataset_available = True
        return fps

    @staticmethod
    def _convert_berkley_time(time: float) -> date:
        """converts berkley time float to date fo first day(!) of respective month"""
        year = math.floor(time)
        month = math.ceil((time % 1) * 12)
        return date(year, month, 1)

    def preprocess_data(
        self, fps_raw: dict[str, str], baseline: tuple[int, int] = (1850, 1900)
    ) -> xr.DataArray:
        """Berkley preprocessing: Calculate anomaly relative to baseline, perform time conversion and intepolation"""
        assert self.dataset_available, " download/data check has not run, check indicator logic"
        fp = self.storage.build_filepath("processing", "berkley_preprocessed", filetype=".nc")
        try:
            if self.regenerate["preprocessing"]:
                raise FileNotFoundError
            berkley_temp = xr.open_dataarray(fp)
            # check recency - if there is at least some data since last quarter start its probably
            last_quarter_date = get_quarter("last")
            berkley_end_date = date.fromisoformat(berkley_temp["time"].max().item())
            if berkley_end_date < last_quarter_date:
                raise FileNotFoundError
            if berkley_end_date.month != last_quarter_date.month + 2:
                self.console.print(
                    f"Berkley only available until {berkley_end_date}, last quarter not fully covered."
                )

        except FileNotFoundError:
            self.console.print("Calculating temperature anomalies...")
            # read data, adjust baseline, select temperature
            berkley = xr.open_dataset(fps_raw["high-res"])
            berkley_baseline = xr.open_dataset(fps_raw["baseline"])
            berkley["temperature_anomaly"] = berkley.temperature - berkley_baseline.temperature.sel(
                time=slice(baseline[0], baseline[1])
            ).mean("time")

            berkley = berkley.rename({"latitude": "lat", "longitude": "lon"})
            berkley["time"] = np.vectorize(self._convert_berkley_time)(berkley.time)

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
            berkley_temp = berkley.sel(time=slice(berkley_end)).temperature_anomaly

            # linearly interpolate data to our resolution
            self.console.print("Interpolating resolution...")
            lat_new = np.arange(-89.75, 90, 0.5)
            lon_new = np.arange(-179.75, 180, 0.5)
            berkley_temp: xr.DataArray = berkley_temp.interp(
                lat=lat_new, lon=lon_new, method="linear"
            )
            # convert time for storage
            berkley_temp["time"] = np.vectorize(lambda x: x.isoformat())(berkley_temp.time)
            berkley_temp.to_netcdf(fp)

        berkley_temp["time"] = np.vectorize(lambda x: date.fromisoformat(x))(berkley_temp.time)
        return berkley_temp

    def calculate_grid_quarter_anomalies(
        self, berkley_temp: xr.DataArray, years_average: int = 30
    ) -> pd.DataFrame:
        # create averages
        self.console.print(f"Calculating {years_average}-year averages...")
        berkley_temp = berkley_temp.sel(
            time=slice(date(self.global_config["start_year"] - (years_average + 1), 1, 1), None)
        )
        berkley_temp_average = berkley_temp.rolling(time=12 * years_average).mean()

        # limit to quarters - select 10-year means from last month in each quarter and convert to first month for consistency and merging
        berkley_temp_average = berkley_temp_average.sel(
            time=xr.date_range(
                f"{self.global_config['start_year']}-03-01",
                berkley_temp.time.values.max(),
                freq="3MS",
            )
        )
        berkley_temp_average["time"] = np.vectorize(lambda x: date(x.year, x.month - 2, 1))(
            berkley_temp_average.time
        )
        df_berkley = berkley_temp_average.to_dataframe()
        df_berkley = df_berkley.dropna(subset="temperature_anomaly")
        df_berkley = df_berkley.reset_index()
        df_berkley["year"] = df_berkley["time"].apply(lambda x: x.year)
        df_berkley["quarter"] = df_berkley["time"].apply(lambda x: math.ceil(x.month / 3))
        df_berkley["pgid"] = df_berkley.apply(lambda x: coords_to_pgid(x.lat, x.lon), axis=1)
        df_berkley = df_berkley.set_index(["pgid", "year", "quarter"]).sort_index()
        return df_berkley[["temperature_anomaly"]]


# test class
if __name__ == "__main__":
    from base.objects import ConfigParser

    config = ConfigParser()

    # Example usage
    data = BERKLEYData(local=False, config=config)
    # just load the current data

    df_berkley = data.load_data()
    print(df_berkley.head())
