import xarray as xr

from base.objects import Dataset


class BIIData(Dataset):
    """Handles loading and preprocessing of Magpie BII data.

    Implements `load_data()` to load the magpie bii data from input storage.

    Attributes:
        data_key (str): Set to "bii".
    """

    data_key: str = "bii"

    def load_data(self) -> xr.Dataset:
        """Loads and returns data from input storage"""
        return xr.open_dataset(self.data_config[self.data_key], decode_times=False)
