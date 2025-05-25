import ee
import os

from dotenv import load_dotenv
from google.oauth2 import service_account

from typing import Union
from datetime import datetime, date


class GEEClient:
    """Very basic GEE class used for authentication and basic operations."""

    def __init__(self, credentials=""):
        self._parse_credentials(credentials)
        self._authorize()
        self.band = None
        self.dataset = None
        self.type = None
        self.asset = None

    def _parse_credentials(self, credentials):
        if isinstance(credentials, service_account.Credentials):
            self.credentials = credentials
            return
        load_dotenv()
        service_name = os.getenv("GEE_SERVICE_NAME")
        service_json = os.getenv("GEE_SERVICE_JSON")
        if service_name == "" or service_json == "":
            raise Exception(
                "Please set the .env credential vars if no Credential object is provided!"
            )

        try:
            self.credentials = ee.ServiceAccountCredentials(
                service_name,
                service_json,
            )
        except:
            raise Exception("Invalid credentials!")

    def _authorize(self):
        if not ee.data._credentials:
            ee.Initialize(self.credentials)

    def set_asset(self, asset_path: str) -> None:
        try:
            self.asset = ee.FeatureCollection(asset_path)
        except Exception as e:
            raise e

    def init_dataset(self, dataset_name: str, asset_path: str) -> None:
        """Set initial values (band, dataset) of the dataset to be queried.

        Currently only supports: gpw_land, gpw_pop, wp_pop TODO: update list
        """
        era5_bands = [
            "temperature_2m",
            "temperature_2m_min",
            "temperature_2m_max",
            "volumetric_soil_water_layer_1",
            "volumetric_soil_water_layer_2",
            "volumetric_soil_water_layer_3",
            "volumetric_soil_water_layer_4",
            "total_precipitation_sum",
            "total_precipitation_min",
            "total_precipitation_max",
        ]

        spei_input = [
            "total_precipitation_sum",
            "temperature_2m",
            "temperature_2m_min",
            "temperature_2m_max",
            "surface_net_solar_radiation_sum",
            "surface_solar_radiation_downwards_sum",
        ]

        set_dict = {
            "gpw_land": ["CIESIN/GPWv411/GPW_Land_Area", "land_area", "image"],
            "gpw_pop": [
                "CIESIN/GPWv411/GPW_UNWPP-Adjusted_Population_Count",
                "unwpp-adjusted_population_count",
                "image_collection",
            ],
            "wp_pop": ["WorldPop/GP/100m/pop", "population", "image_collection"],
            "ndvi": ["MODIS/061/MOD13Q1", "NDVI", "image_collection"],
            "ndvi_anomaly": ["MODIS/061/MOD13Q1", "NDVI", "image_collection"],
            "ntl": [
                "NOAA/VIIRS/001/VNP46A2",
                "Gap_Filled_DNB_BRDF_Corrected_NTL",
                "image_collection",
            ],
            "era5": ["ECMWF/ERA5_LAND/MONTHLY_AGGR", era5_bands, "image_collection"],
            "era5_anomaly": [
                "ECMWF/ERA5_LAND/MONTHLY_AGGR",
                era5_bands,
                "image_collection",
            ],
            "dynamic_world": ["GOOGLE/DYNAMICWORLD/V1", "label", "image_collection"],
            "spei_input": [
                "ECMWF/ERA5_LAND/MONTHLY_AGGR",
                spei_input,
                "image_collection",
            ],
            "temp_anomaly": [
                "ECMWF/ERA5_LAND/MONTHLY_AGGR",
                "temperature_2m",
                "image_collection",
            ],
        }

        if dataset_name not in set_dict.keys():  # made this dynamic
            raise ValueError(
                f"The dataset name must be one of the following: {','.join(list(set_dict.keys()))}"
            )

        self.dataset = set_dict[dataset_name][0]
        self.band = set_dict[dataset_name][1]
        self.type = set_dict[dataset_name][2]

        try:
            self.asset = ee.FeatureCollection(asset_path)
        except Exception as e:
            raise e

    def get_scale(self) -> float:
        """Get scale of ee.Image or ee.ImageCollection."""
        if (
            self.dataset is None
            or self.type is None
            or self.type not in ["image", "image_collection"]
        ):
            raise Exception("You have to initialize a dataset first!")
        if self.type == "image":
            return ee.Image(self.dataset).projection().nominalScale().getInfo()
        if self.type == "image_collection":
            return ee.ImageCollection(self.dataset).first().projection().nominalScale().getInfo()

    def get_latest_available_data(self) -> date:
        """
        Get the latest image available from the image collection or image and return the most recent date the data is available for.
        """
        if self.type == "image":
            return datetime.fromisoformat(ee.Image(self.dataset).date().format().getInfo()).date()
        if self.type == "image_collection":
            return datetime.fromisoformat(
                ee.ImageCollection(self.dataset)
                .limit(1, "system:time_end", False)
                .first()
                .date()
                .format()
                .getInfo()
            ).date()

    def get_image_for_year(self, year: int) -> ee.Image:
        if self.type == "image":
            return ee.Image(self.dataset).filterDate(str(year))
        if self.type == "image_collection":
            return ee.ImageCollection(self.dataset).filterDate(str(year)).select(self.band)

    def get_images_for_years(
        self, year_start: Union[int, str], year_end: Union[int, str]
    ) -> ee.Image:
        if self.type == "image":
            raise NotImplementedError('not implemented for type "image"')
        if self.type == "image_collection":
            return (
                ee.ImageCollection(self.dataset)
                .filterDate(str(year_start), str(year_end))
                .select(self.band)
            )

    def get_images_for_quarter(self, year: int, quarter: int) -> ee.ImageCollection:
        start_month = (quarter - 1) * 3 + 1
        end_month = quarter * 3 + 1
        start_date = f"{year}-{start_month}-01"
        if quarter != 4:
            end_date = f"{year}-{end_month}-01"
        else:
            end_date = f"{year + 1}-01-01"
        if self.type == "image":
            raise NotImplementedError('not implemented for type "image"')
        if self.type == "image_collection":
            return ee.ImageCollection(self.dataset).filterDate(start_date, end_date)

    def get_tasks(self, n: int):
        return ee.data.getTaskList()[:n]
