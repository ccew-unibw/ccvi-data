#https://xds-preprod.ecmwf.int/datasets/derived-drought-historical-monthly?tab=download
import os
import pandas as pd
from base.objects import Dataset, console
from utils.index import get_quarter


def to_bin(x):
    import numpy as np

    binsize = 0.5
    if x > 180:
        x = -180 + x - 180
    return (binsize / 2) + np.floor(x / binsize) * binsize


def get_days_in_month(year=1990, month="01"):
    days_in_month = pd.Period(f"{year}-{month}").days_in_month
    return [str(i).zfill(2) for i in range(1, days_in_month + 1)]


def apirequest(outfilename="", year="1990", month="01"):
    import cdsapi
    from dotenv import load_dotenv

    load_dotenv()
    CDS_ECMWF_KEY = os.getenv("CDS_ECMWF_KEY")
    #up to 2022 this
    
    try:
        dataset = "derived-drought-historical-monthly"
        request = {
                "variable": ["standardised_precipitation_evapotranspiration_index"],
                "accumulation_period": ["12"],
                "version": "1_0",
                "product_type": ["reanalysis"],
                "dataset_type": "intermediate_dataset",
                "year": [year],
                "month": [month]
            }
        
        c = cdsapi.Client(url="https://xds-preprod.ecmwf.int/api", key=f"{CDS_ECMWF_KEY}")

        c.retrieve(dataset, request, outfilename)
    except Exception as e:
        print(e)
        print("try the consolidated")
        try:
            dataset = "derived-drought-historical-monthly"
            request = {
                "variable": [
                    "standardised_precipitation_evapotranspiration_index",
                    "test_for_normality_spei"
                ],
                "accumulation_period": ["12"],
                "version": "1_0",
                "product_type": ["reanalysis"],
                "dataset_type": "consolidated_dataset",
                "year": [year],
                "month": [month]
            }

            c = cdsapi.Client(url="https://xds-preprod.ecmwf.int/api", key=f"{CDS_ECMWF_KEY}")
            c.retrieve(dataset, request, outfilename)

        except Exception as e:
            print(e)
            print("failed to download the data, please check the CDS API key and the dataset availability.")
        


def process_nc(ncfilepath):
    # read nc file``
    import xarray as xr

    ds = xr.open_dataset(ncfilepath, engine="h5netcdf")

    # to pandas dataframe
    df = ds.to_dataframe().reset_index()
    #replace values smaller equal <-9999 with np.nan
    df.loc[df["SPEI12"] <= -9999, "SPEI12"] = pd.NA
    df = df[["time", "lat", "lon", "SPEI12"]]
    df["lat"] = df["lat"].round(3)
    df["lon"] = df["lon"].round(3)
    #subeset to only the spei12 negative values    
    df["spei12n"] = df["SPEI12"].where(df["SPEI12"] < 0)


    # rescale xarray ds to resolution 0.5d
    df["latbin"] = df["lat"].map(to_bin)
    df["lonbin"] = df["lon"].map(to_bin)

    # drop latbin,lonbin duplicates, and aggregate latbin,lonbin
    # do not mean the nan values, just drop them
    df = df.groupby(["latbin", "lonbin", "time"]).mean().reset_index()

    # drop latitude and longitude columns
    df.drop(columns=["lat", "lon"], inplace=True)

    # rename columns latbin, lonbin with lat lon
    df.rename(
        columns={
            "latbin": "LATITUDE",
            "lonbin": "LONGITUDE",
            "time": "EVENT_DATE",            
        },
        inplace=True,
    )

    return df


class CDSECMWFSPEIData(Dataset):
    """Handles loading and processing of EWDS event data.

    Supports loading from local preprocessed files or via API.
    Includes methods for aggregating event data to the
    grid-quarter level.

    Attributes:
        data_key (str): Set to "cds_ecmwf_spei".
        local (bool): Indicates whether to use local cds ecmwf spei dumps (True) or
            download data via the cds ecmwf spei API (False).
        dataset_available (bool): Flag set to True after data is
            successfully loaded or checked by `load_data`.
    """

    data_key = "cdsecmwf_spei"

    def __init__(self, local: bool = True, *args, **kwargs):
        """Initializes the cds ecmwf spei data source.

        Sets the operation mode (local file vs API) and calls the Dataset
        initializer to setup config and storage.

        Args:
            config (ConfigParser): An initialized ConfigParser instance.
            local (bool, optional): Indicates whether to use local ewds
                dumps (True) or download data via the ewds API (False).
        """
        self.local = local
        self.dataset_available = False
        self.data_keys = [self.data_key, "countries"]
        super().__init__(*args, **kwargs)

    def download_data(self):
        """Downloads ewds data from the API.

        Downloads the ewds data from the API and saves it to the processing
        storage. The filename is based on the current date and time.

        Returns:
            str: The filename of the downloaded ewds data.
        """
        import xarray as xr
        import zipfile

        sources_path = self.storage.storage_paths["processing"]
        destination = f"{sources_path}/CDS-ECMWF-SPEI/"

        if not os.path.exists(destination):
            os.makedirs(destination)

        # Generate dates from the start date to the last quarter
        start_date = pd.to_datetime("1990-01-01")
        today = pd.to_datetime("today")
        current_quarter = today.to_period("Q")
        previous_quarter = current_quarter - 1
        last_day_previous_quarter = previous_quarter.end_time

        date_range = pd.date_range(start=start_date, end=last_day_previous_quarter, freq="M")
        date_range = [(date.year, date.month) for date in date_range]

        for date in date_range:
            year = str(date[0])
            month = str(date[1]).zfill(2)
            outfilename = f"{destination}/cdsecmwfspei12-{year}-{month}.zip"
            print(
                f"=========================start downloading  {year}-{month}==================================",
                flush=True,
            )
            if not os.path.exists(outfilename):
                apirequest(outfilename, year, month)

        # unzip files
        for date in date_range:
            year = str(date[0])
            month = str(date[1]).zfill(2)
            zipfilepath = f"{destination}/cdsecmwfspei12-{year}-{month}.zip"
            ncfilepath = f"{destination}/cdsecmwfspei12-{year}-{month}.nc"
            print(
                f"=========================unzipping {year}-{month}==================================",
                flush=True,
            )
            if not os.path.exists(ncfilepath):
                with zipfile.ZipFile(zipfilepath, "r") as zip_ref:
                    zip_ref.extractall(ncfilepath)

        df_events = []

        for date in date_range:
            year = str(date[0])
            month = str(date[1]).zfill(2)
            ncfilepath = f"{destination}/cdsecmwfspei12-{year}-{month}.nc/SPEI12_genlogistic_global_era5_moda_ref1991to2020_{year}{month}.nc"
            if not os.path.exists(ncfilepath):
                try:
                    ncfilepath = f"{destination}/cdsecmwfspei12-{year}-{month}.nc/SPEI12_genlogistic_global_era5t_moda_ref1991to2020_{year}{month}.nc"
                    if not os.path.exists(ncfilepath):
                        raise FileNotFoundError
                except FileNotFoundError:
                    print(
                        f"File {ncfilepath} not found!!",
                        flush=True,
                    )
                    
            print(f"processing {ncfilepath}", flush=True)
            dfs = process_nc(ncfilepath)
            df_events.append(dfs)
        df_events = pd.concat(df_events)

        # columns to uppercase
        df_events.columns = [col.upper() for col in df_events.columns]

        # reanme LAT with LATITUDE and LON with LONGITUDE

        df_events["YEAR"] = df_events["EVENT_DATE"].dt.year

        df_events["EVENT_TYPE"] = "drought"

        # "YEAR", "EVENT_TYPE", "LATITUDE", "LONGITUDE", "COUNT", "EVENT_DATE"
        # self.storage.save(df_events[self.columns], "processing", filename=self.filename)
        # beacuse those are the events I keep all the columns
        self.storage.save(df_events, "processing", filename=self.filename)

    def load_data(self):
        """Loads ewds data, checking for cached processing files first.

        Attempts to load a local ewds copy from the 'processing' storage
        including the last completed quarter. If not found:
        - If `self.local` is True, loads the raw dump specified in the config.
          Raises an error if the provided ewds dump does not fully cover the
          latest quarter.
        - If `self.local` is False, currently raises NotImplementedError (API access TBD).
        Saves the loaded raw/dump data to the processing storage.

        Returns:
            pd.DataFrame: The loaded ewds event data.
        """
        self.last_quarter_date = get_quarter("last", bounds="end")
        self.filename = (
            f"cdsecmwfspei12_{self.last_quarter_date.year}_Q{int(self.last_quarter_date.month / 3)}"
        )
        self.columns = ["YEAR", "EVENT_TYPE", "LATITUDE", "LONGITUDE", "COUNT", "EVENT_DATE"]
        try:
            df_event_level = self.storage.load("processing", filename=self.filename)
        except FileNotFoundError:
            if self.local:
                df_event_level = pd.read_parquet(self.data_config[self.data_key])
                if df_event_level["EVENT_DATE"].max() < self.last_quarter_date:
                    raise Exception(
                        "preprocessed ewds data out of date, please provide a version up to "
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
        """Calculates grid-quarter aggregates from the event level ewds data.

        Assigns events to grid cells (pgid), calculates quarterly aggregates
        (event counts, fatalities) for armed violence and unrest, merges with the
        base grid structure, and fills missing values based on ewds coverage
        information.

        Args:
            df_base (pd.DataFrame): Base data structure for indicator data.
            df_event_level (pd.DataFrame): ewds event-level data from `self.load_data`.

        Returns:
            pd.DataFrame: Dataframe aligned to index grid with quarterly ewds aggregates.
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
                "No preprocessed ewds data in storage or out of date," + " processing event data..."
            )

            # don't automatically start ewds download since those are separate step in the
            # indicator logic that should each be performed deliberately
            assert self.dataset_available, (
                "ewds download/data check has not run, check indicator logic"
            )

            # quarter as number 1,2,3,4
            df_event_level["QUARTER"] = df_event_level["EVENT_DATE"].dt.quarter

            # aggregate
            df_event_level_aggregated = (
                df_event_level[["LATITUDE", "LONGITUDE", "YEAR", "QUARTER", "SPEI12", "SPEI12N"]]
                .groupby(["LATITUDE", "LONGITUDE", "YEAR", "QUARTER"])
                .mean()
            )
            df_event_level_aggregated = df_event_level_aggregated.reset_index()
            df_base = df_base.reset_index()
            df = df_base.merge(
                df_event_level_aggregated,
                left_on=["year", "quarter", "lat", "lon"],
                right_on=["YEAR", "QUARTER", "LATITUDE", "LONGITUDE"],
                how="left",
            )

            df = df[["pgid", "year", "quarter", "lat", "lon",  "SPEI12", "SPEI12N"]]

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
    cdsecmwf_data = CDSECMWFSPEIData(local=False, config=config)
    # just load the current data
    # df_ewds = ewds_data.load_data()

    df_cdsecmwf_data = cdsecmwf_data.load_data()
    print(df_cdsecmwf_data.head())
