# https://xds-preprod.ecmwf.int/datasets/derived-drought-historical-monthly?tab=download
import os
import zipfile

import cdsapi
from dotenv import load_dotenv
import ee
import numpy as np
from requests.exceptions import HTTPError
import pandas as pd
import xarray as xr

from base.objects import Dataset, ConfigParser
from utils.index import get_quarter
from utils.gee import GEEClient


def fc_to_dict(fc: ee.FeatureCollection) -> ee.Dictionary:
    prop_names = fc.first().propertyNames()
    prop_lists = fc.reduceColumns(
        reducer=ee.Reducer.toList().repeat(prop_names.size()), selectors=prop_names
    ).get("list")
    return ee.Dictionary.fromLists(prop_names, prop_lists)


def create_spei_mask(fp: str):
    client = GEEClient()
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
    df.to_parquet(fp)
    return


def mask_spei(data: pd.DataFrame, columns: list[str], storage: str, threshold: float = 0.75):
    """
    Mask SPEI data based on land cover. Masking is applied if combined share of barren and ice land exceeds a threshold.

    """
    spei_mask_fp = f"{storage}/spei_mask.parquet"
    if not os.path.exists(spei_mask_fp):
        create_spei_mask(spei_mask_fp)
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


def apirequest(
    outfilename: str,
    month: int,
    year: int | None = None,
    var: str = "standardised_precipitation_evapotranspiration_index",
) -> str:
    """Submits a data retrieval request to the CDS API for ECMWF SPEI data.

    This function constructs and sends a request to the Copernicus Climate Data
    Store (CDS) API to download SPEI data for a specific variable, month, and
    optional year. It is designed to handle different dataset types, first
    attempting to download the final "consolidated_dataset" and falling back
    to the "intermediate_dataset" if the former is not yet available for the
    requested period. Requires `CDS_ECMWF_KEY` to be set as an environment variable.

    Args:
        outfilename (str): The local path where the downloaded data (a .zip file)
            will be saved.
        month (int): The month for which to request data.
        year (int | None, optional): The year for which to request data. Not
            required for var="test_for_normalty_spei", therefore defaults to None.
        var (str, optional): The CDS API variable name to download. Defaults to
            "standardised_precipitation_evapotranspiration_index".

    Returns:
        str: The type of dataset that was successfully downloaded
            ("consolidated_dataset" or "intermediate_dataset").
    """
    # input checks
    assert var in ["standardised_precipitation_evapotranspiration_index", "test_for_normality_spei"]
    if var == "standardised_precipitation_evapotranspiration_index":
        assert year is not None

    load_dotenv()
    CDS_ECMWF_KEY = os.getenv("CDS_ECMWF_KEY")

    dataset = "derived-drought-historical-monthly"
    c = cdsapi.Client(url="https://xds-preprod.ecmwf.int/api", key=f"{CDS_ECMWF_KEY}")
    request = {
        "variable": [var],
        "accumulation_period": ["12"],
        "version": "1_0",
        "product_type": ["reanalysis"],
        "month": [str(month).zfill(2)],
    }
    if year is not None:
        request["year"] = str(year)
    # up to 2022 this

    try:
        request["dataset_type"] = "consolidated_dataset"
        c.retrieve(dataset, request, outfilename)
    except HTTPError as e:
        print(e)
        print("consolidated not yet available, try intermediate")
        request["dataset_type"] = "intermediate_dataset"
        try:
            c.retrieve(dataset, request, outfilename)

        except Exception as e:
            print(
                "failed to download the data, please check the CDS API key and the dataset availability."
            )
            raise e
    return request["dataset_type"]


def process_nc(spei_filepath, significance_filepath):
    """Processes raw monthly SPEI-12 and significance NetCDF files into a gridded DataFrame.

    This function reads a raw SPEI NetCDF file and a corresponding significance
    NetCDF file. It masks the SPEI data, keeping only values where the significance
    test passed. It then spatially aggregates the higher-resolution data to a
    standardized 0.5-degree grid by taking the mean of all source pixels within
    each grid cell. The function also cleans the data by removing fill values and
    standardizes column names.

    Args:
        spei_filepath (str): The file path to the raw SPEI NetCDF data for one
            year and month.
        significance_filepath (str): The file path to the corresponding significance
            NetCDF data for that month.

    Returns:
        pd.DataFrame: A DataFrame containing the processed and gridded SPEI
            data for the given month, with columns for latitude, longitude,
            time, and the SPEI-12 value.
    """
    # read nc files
    ds_spei = xr.open_dataset(spei_filepath, engine="h5netcdf")
    ds_significance = xr.open_dataset(significance_filepath, engine="h5netcdf")
    # set significance to to spei time so it matches for where masking
    ds_significance["time"] = ds_spei["time"]
    ds_spei["SPEI12"] = ds_spei["SPEI12"].where(ds_significance["significance"] == 1)

    # to pandas dataframe
    df = ds_spei.to_dataframe().reset_index()
    # replace values smaller equal <-6 with np.nan
    # the FillValue is -9999 and there are some invalid values around -8
    # no valid values should ever reach -6
    df.loc[df["SPEI12"] <= -6, "SPEI12"] = pd.NA
    df = df[["time", "lat", "lon", "SPEI12"]]
    df["lat"] = df["lat"].round(3)
    df["lon"] = df["lon"].round(3)

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
    """Handles downloading, processing, and aggregation of CDS ECMWF SPEI drought data.

    Implements `load_data()` as the main entry point, which orchestrates the
    full download and processing pipeline via `download_data()` if a cached,
    up-to-date file is not found.
    Implements `download_data()` to fetch raw monthly SPEI and significance
    NetCDF files from the CDS API and process them into a unified, gridded DataFrame.
    Implements `create_grid_quarter_aggregates()` to temporally aggregate the
    monthly data into quarterly mean values.
    Implements `select_quarter_values()` as an alternative aggregation method that
    selects the SPEI value from the last month of each quarter.

    This class requires `CDS_ECMWF_KEY` to be set as an environment variable for
    API authentication.

    Attributes:
        data_key (str): Set to "cds_ecmwf_spei".
        local (bool): Set to False, as data is downloaded via the ECMWF API.
        dataset_available (bool): Flag set to True after `load_data` confirms
            that the processed data is available.
    """

    data_key = "cds_ecmwf_spei"
    local = False

    def __init__(self, config: ConfigParser, *args, **kwargs):
        """Initializes the CDS ECMWF SPEI data source.

        Sets up the availability flag and calls the parent `Dataset` initializer
        to set up configuration and storage.

        Args:
            config (ConfigParser): An initialized ConfigParser instance.
        """
        self.dataset_available = False
        super().__init__(config=config, *args, **kwargs)

    def download_data(self) -> None:
        """Orchestrates the full download and initial processing workflow.

        This method manages the end-to-end process of generating the monthly
        gridded SPEI-12 dataset. It iterates through a historical date range,
        downloads the raw SPEI and significance data for each month from the CDS
        API, processes the downloaded NetCDF files into a standardized 0.5-degree
        grid format, and concatenates all monthly results. The final, comprehensive
        DataFrame of gridded monthly SPEI values is then cached to the processing
        directory. This method is typically called by `load_data()` when no
        up-to-date cache is found.
        """
        # Generate dates from the start date to the last quarter
        start_date = pd.to_datetime("1990-01-01")
        today = pd.to_datetime("today")
        current_quarter = today.to_period("Q")
        previous_quarter = current_quarter - 1
        last_day_previous_quarter = previous_quarter.end_time

        date_range = pd.date_range(start=start_date, end=last_day_previous_quarter, freq="M")

        def download_unzip(var: str):
            outfilenc = outfilename.replace(".zip", ".nc")
            if not os.path.exists(outfilenc):
                dataset_type = apirequest(outfilename, year=year, month=month, var=var)
                # rename in case of intermediate so intermediates get retried each time
                if dataset_type == "intermediate_dataset":
                    outfilenc = outfilenc.replace(".nc", "_intermediate.nc")
                with zipfile.ZipFile(outfilename, "r") as zip_ref:
                    # sanity check of contents
                    if var == "test_for_normality_spei":
                        file_selection = [
                            f for f in zip_ref.filelist if "speisignificance" in f.filename
                        ]
                    else:
                        file_selection = [
                            f for f in zip_ref.filelist if "speisignificance" not in f.filename
                        ]
                    assert len(file_selection) == 1
                    zip_ref.extractall(os.path.dirname(outfilename), file_selection)
                os.rename(
                    os.path.join(os.path.dirname(outfilename), file_selection[0].filename),
                    outfilenc,
                )
                os.remove(outfilename)

        # significance files - one per month
        year = None
        for month in range(1, 13):
            self.console.print(
                f"========================= downloading and unzipping significance {str(month).zfill(2)}=================================="
            )
            outfilename = self.storage.build_filepath(
                "processing",
                f"cdsecmwfspei12-significance-{str(month).zfill(2)}",
                "significance",
                ".zip",
            )
            download_unzip("test_for_normality_spei")

        for date in date_range:
            self.console.print(
                f"========================= downloading and unzipping {date.year}-{str(date.month).zfill(2)}=================================="
            )
            year = date.year
            month = date.month
            outfilename = self.storage.build_filepath(
                "processing", f"cdsecmwfspei12-{year}-{str(month).zfill(2)}", "spei12", ".zip"
            )
            download_unzip("standardised_precipitation_evapotranspiration_index")

        df_events = []

        for date in date_range:
            year = date.year
            month_str = str(date.month).zfill(2)
            spei_filepath = self.storage.build_filepath(
                "processing", f"cdsecmwfspei12-{year}-{month_str}", "spei12", ".nc"
            )
            # switch to intermediate if
            try:
                assert os.path.exists(spei_filepath)
            except AssertionError:
                spei_filepath = self.storage.build_filepath(
                    "processing", f"cdsecmwfspei12-{year}-{month_str}_intermediate", "spei12", ".nc"
                )
                assert os.path.exists(spei_filepath)
            significance_filepath = self.storage.build_filepath(
                "processing", f"cdsecmwfspei12-significance-{month_str}", "significance", ".nc"
            )
            self.console.print(f"processing {spei_filepath}")
            dfs = process_nc(spei_filepath, significance_filepath)
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
        return

    def load_data(self) -> pd.DataFrame:
        """Loads the processed monthly gridded SPEI-12 data.

        This is the primary public method for accessing SPEI data. It first
        checks for a cached data file corresponding to the last completed quarter.
        If this file is not found or if regeneration is forced via the global
        config, it triggers the full download and processing pipeline by calling
        `download_data()`.

        Returns:
            pd.DataFrame: A DataFrame of monthly SPEI-12 values, spatially aggregated
                to the grid, indexed with time and location information.
        """
        self.last_quarter_date = get_quarter("last", bounds="end")
        self.filename = (
            f"cdsecmwfspei12_{self.last_quarter_date.year}_Q{int(self.last_quarter_date.month / 3)}"
        )
        self.columns = ["YEAR", "EVENT_TYPE", "LATITUDE", "LONGITUDE", "COUNT", "EVENT_DATE"]
        try:
            if self.regenerate["data"]:
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
        """Aggregates the monthly gridded SPEI-12 data into quarterly averages.

        This method takes the monthly SPEI-12 data and calculates the mean SPEI-12
        value for each grid cell within each quarter. The results are then
        merged onto the standard data structure. It also applies a spatial mask to
        filter out barren and ice-covered grid cells.

        Args:
            df_base (pd.DataFrame): The base DataFrame structure, indexed by
                ('pgid', 'year', 'quarter').
            df_event_level (pd.DataFrame): The monthly gridded SPEI-12 data, as
                returned by `self.load_data()`.

        Returns:
            pd.DataFrame: `df_base` with the quarterly mean SPEI-12 values per grid
                cell added.
        """
        # don't automatically start ewds download since those are separate step in the
        # indicator logic that should each be performed deliberately
        assert self.dataset_available, (
            "ecmwf spei download/data check has not run, check indicator logic"
        )

        # quarter as number 1,2,3,4
        df_event_level["QUARTER"] = df_event_level["EVENT_DATE"].dt.quarter

        # aggregate
        df_event_level_aggregated = (
            df_event_level[["LATITUDE", "LONGITUDE", "YEAR", "QUARTER", "SPEI12"]]
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

        df = df[["pgid", "year", "quarter", "lat", "lon", "SPEI12"]]

        df["time"] = df["year"].astype(str) + "Q" + df["quarter"].astype(str)
        df["time"] = pd.PeriodIndex(df["time"], freq="Q")
        # convert to datetime
        df["time"] = df["time"].dt.to_timestamp()

        df.columns = [col.lower() for col in df.columns]
        # mask with old spei mask
        storage = self.storage.storage_paths["processing"]
        df = mask_spei(df, ["spei12"], storage)

        return df

    def select_quarter_values(
        self,
        df_base: pd.DataFrame,
        df_event_level: pd.DataFrame,
    ) -> pd.DataFrame:
        """Filters monthly gridded SPEI-12 data by selecting the end-of-quarter value.

        As an alternative to averaging, this method only keeps the monthly SPEI-12
        value for the last month of each quarter (March, June, September, December).
        These end-of-quarter values are then merged onto the standard data structure
        and masked to filter out barren and ice-covered grid cells.

        Args:
            df_base (pd.DataFrame): The base DataFrame structure, indexed by
                ('pgid', 'year', 'quarter').
            df_event_level (pd.DataFrame): The monthly gridded SPEI-12 data, as
                returned by `self.load_data()`.

        Returns:
            pd.DataFrame: `df_base` with the end-of-quarter SPEI-12 values per grid
                cell added.
        """
        assert self.dataset_available, (
            "ecmwf spei download/data check has not run, check indicator logic"
        )
        # select only the values for the last month in each quarter
        df_event_level_quarters = df_event_level.loc[
            df_event_level["EVENT_DATE"].dt.month.isin([3, 6, 9, 12])
        ]
        # quarter as number 1,2,3,4
        df_event_level_quarters["QUARTER"] = df_event_level_quarters["EVENT_DATE"].dt.quarter

        df = df_base.reset_index().merge(
            df_event_level_quarters[["LATITUDE", "LONGITUDE", "YEAR", "QUARTER", "SPEI12"]],
            left_on=["year", "quarter", "lat", "lon"],
            right_on=["YEAR", "QUARTER", "LATITUDE", "LONGITUDE"],
            how="left",
        )
        df = df[["pgid", "year", "quarter", "lat", "lon", "SPEI12"]]

        df["time"] = df["year"].astype(str) + "Q" + df["quarter"].astype(str)
        df["time"] = pd.PeriodIndex(df["time"], freq="Q")
        # convert to datetime
        df["time"] = df["time"].dt.to_timestamp()

        df.columns = [col.lower() for col in df.columns]
        # mask with old spei mask
        storage = self.storage.storage_paths["processing"]
        df = mask_spei(df, ["spei12"], storage)

        return df


# test class
if __name__ == "__main__":
    config = ConfigParser()

    # Example usage
    cdsecmwf_data = CDSECMWFSPEIData(config=config)
    # just load the current data

    df_cdsecmwf_data = cdsecmwf_data.load_data()
    print(df_cdsecmwf_data.head())
