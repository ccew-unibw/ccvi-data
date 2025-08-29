import rioxarray as rxr
import xarray as xr

from base.objects import Dataset


class SoilErosionData(Dataset):
    """Handles loading and preprocessing of Magpie soil erosion data.

    Implements `load_data()` to load the data from input storage.

    Attributes:
        data_key (str): Set to "soil_erosion".
    """

    data_key: str = "soil_erosion"

    def load_data(self) -> xr.DataArray:
        """Loads and returns data from input storage"""
        return rxr.open_rasterio(self.data_config[self.data_key])  # type: ignore
