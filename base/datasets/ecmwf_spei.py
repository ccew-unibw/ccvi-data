#https://xds-preprod.ecmwf.int/datasets/derived-drought-historical-monthly?tab=download
import os
import zipfile

import cdsapi
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import xarray as xr

from base.objects import Dataset, ConfigParser
from utils.index import get_quarter

    
def mask_spei(data: pd.DataFrame, columns: list[str], storage: str, threshold: float = 0.75):
    """
    Mask SPEI data based on land cover. Masking is applied if combined share of barren and ice land exceeds a threshold.

    """
    spei_mask_fp = f"{storage}/spei_mask.parquet"
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

def to_bin(x):
    

    binsize = 0.5
    if x > 180:
        x = -180 + x - 180
    return (binsize / 2) + np.floor(x / binsize) * binsize


def get_days_in_month(year=1990, month="01"):
    days_in_month = pd.Period(f"{year}-{month}").days_in_month
    return [str(i).zfill(2) for i in range(1, days_in_month + 1)]


def apirequest(outfilename="", year="1990", month="01"):


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
            print("failed to download the data, please check the CDS API key and the dataset availability.")
            raise e
        


def process_nc(ncfilepath):
    # read nc file``
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
        local (bool): Set to False as data is downloaded via the ecmwf API.
        dataset_available (bool): Flag set to True after data is
            successfully loaded or checked by `load_data`.
    """

    data_key = "cdsecmwf_spei"
    local = False

    def __init__(self, config: ConfigParser, *args, **kwargs):
        """Initializes the cds ecmwf spei data source.

        Sets up the availability flag and calls the Dataset initializer.

        Args:
            config (ConfigParser): An initialized ConfigParser instance.
        """
        self.dataset_available = False
        super().__init__(config=config, *args, **kwargs)

    def download_data(self):
        """Downloads ewds data from the API.

        Downloads the ewds data from the API and saves it to the processing
        storage. The filename is based on the current date and time.

        Returns:
            str: The filename of the downloaded ewds data.
        """

        subfolder = f"CDS-ECMWF-SPEI/"

        # Generate dates from the start date to the last quarter
        start_date = pd.to_datetime("1990-01-01")
        today = pd.to_datetime("today")
        current_quarter = today.to_period("Q")
        previous_quarter = current_quarter - 1
        last_day_previous_quarter = previous_quarter.end_time

        date_range = pd.date_range(start=start_date, end=last_day_previous_quarter, freq="M")

        for date in date_range:
            year = str(date.year)
            month = str(date.month).zfill(2)
            outfilename = self.storage.build_filepath("processing", f"cdsecmwfspei12-{year}-{month}", subfolder, ".zip")
            self.console.print(
                f"========================= downloading and unzipping {year}-{month}=================================="
            )
            if not os.path.exists(outfilename):
                # download
                apirequest(outfilename, year, month)
                # unzip files
                ncfolder = outfilename.removesuffix(".zip")
                
                if not os.path.exists(ncfolder):
                    with zipfile.ZipFile(outfilename, "r") as zip_ref:
                        zip_ref.extractall(ncfolder)

        df_events = []

        for date in date_range:
            year = str(date.year)
            month = str(date.month).zfill(2)
            ncfolder = self.storage.build_filepath("processing", f"cdsecmwfspei12-{year}-{month}", subfolder, "")

            ncfilepath = f"{ncfolder}/SPEI12_genlogistic_global_era5_moda_ref1991to2020_{year}{month}.nc"
            if not os.path.exists(ncfilepath):
                try:
                    ncfilepath = f"{ncfolder}/SPEI12_genlogistic_global_era5t_moda_ref1991to2020_{year}{month}.nc"
                    if not os.path.exists(ncfilepath):
                        raise FileNotFoundError
                except FileNotFoundError:
                    print(
                        f"File {ncfilepath} not found!!",
                        flush=True,
                    )
                    
            self.console.print(f"processing {ncfilepath}")
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
        including the last completed quarter. If not found, or 
        regenerate["preprocessing"] is set to True, downloads the data and 
        converts to gridded dataframe. Saves the preprocessed data to the processing storage.

        Returns:
            pd.DataFrame: The loaded ewds event data.
        """
        self.last_quarter_date = get_quarter("last", bounds="end")
        self.filename = (
            f"cdsecmwfspei12_{self.last_quarter_date.year}_Q{int(self.last_quarter_date.month / 3)}"
        )
        self.columns = ["YEAR", "EVENT_TYPE", "LATITUDE", "LONGITUDE", "COUNT", "EVENT_DATE"]
        try:
            if self.regenerate["preprocessing"]:
                raise FileNotFoundError
            df_event_level = self.storage.load("processing", filename=self.filename)
        except FileNotFoundError:
            self.download_data()
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
        # don't automatically start ewds download since those are separate step in the
        # indicator logic that should each be performed deliberately
        assert self.dataset_available, (
            "ewds download/data check has not run, check indicator logic"
        )
        fp_preprocessed = self.storage.build_filepath("processing", filename="preprocessed")
        try:
            df = pd.read_parquet(fp_preprocessed)
            last_quarter_date = get_quarter("last")

            if df["time"].max().date() < last_quarter_date:
                raise FileNotFoundError
            return df
        except FileNotFoundError:
            self.console.print(
                "No preprocessed ewds data in storage or out of date," + " processing event data..."
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
            #mask with old spei mask
            
            storage =self.storage.storage_paths['processing'] 
            df = mask_spei(df, ["spei12"], storage)
    
            df = df.fillna(0)

            df.to_parquet(fp_preprocessed)
        return df


# test class
if __name__ == "__main__":

    config = ConfigParser()

    # Example usage
    cdsecmwf_data = CDSECMWFSPEIData(config=config)
    # just load the current data

    df_cdsecmwf_data = cdsecmwf_data.load_data()
    print(df_cdsecmwf_data.head())
