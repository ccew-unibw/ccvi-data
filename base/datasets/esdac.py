import numpy as np
import rioxarray as rxr
import xarray as xr

from base.objects import Dataset


class ESDACData(Dataset):
    """Handles loading, and preprocessing of ESDAC below-ground carbon debt data.

    Implements `load_data()` to load the carbon debt data from input storage.
    Implements `preprocess_data()` to clean and upscale data to better match
    the 0.5 degree spatial resolution for aggregation to the grid.

    Attributes:
        data_key (str): Set to "esdac".
        upper_threshold (float): Normalization threshold (99th percentile), 
            set by `preprocess_data()` before interpolation based on the original
            input data.
    """

    data_key: str = "esdac"

    def load_data(self) -> xr.DataArray:
        """Loads and returns data from input storage"""
        return rxr.open_rasterio(self.data_config[self.data_key])  # type: ignore

    def preprocess_data(self, da: xr.DataArray) -> xr.DataArray:
        """Preprocesses the carbon debt data.

        Sets the fill value to NA, stores the 99th percentile based on the orgiginal
        resolution as class attribute for further usage, and increases the
        resolution to a multiple of the 0.5 degree grid for proper mean aggregation
        across pixel boundaries.

        Args:
            da (xr.DataArray): The data loaded through `load_data()`.

        Returns:
            xr.DataArray: The preprocessed (upscaled) data.
        """
        da = da.where(da != da._FillValue)
        # data comes in 0.4 degree resolution
        xs = np.arange(-179.95, 180, 0.1)
        ys = np.arange(83.35, -55.8, -0.1)
        # store quantile before interpolation
        self.upper_threshold = da.quantile(0.99).item()
        da_interp = da.interp(coords={"y": ys, "x": xs}, method="nearest")
        return da_interp
