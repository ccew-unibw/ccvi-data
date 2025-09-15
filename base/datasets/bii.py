import xarray as xr

from base.objects import Dataset


class BIIData(Dataset):
    """Handles loading and preprocessing of Magpie BII data.

    Implements `load_data()` to load the magpie bii data from the repo.

    Attributes:
        data_key (str): Set to "bii".
        local (bool): Set to False, since data is published in this repo and does
            not need to be added to input data manually.
    """

    data_key: str = "bii"
    local: bool = False

    def load_data(self) -> xr.Dataset:
        """Loads and returns data from input storage"""
        return xr.open_dataset("./bii_ccvi/cell.bii_0.5.nc", decode_times=False)
