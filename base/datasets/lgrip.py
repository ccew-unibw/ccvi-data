import itertools
import os
import re

import requests
import numpy as np
import pandas as pd
from rich.progress import Progress
import rioxarray as rxr

from base.objects import Dataset, ConfigParser, GlobalBaseGrid
from utils.spatial_operations import pgid_to_coords


class LGRIPData(Dataset):
    """Handles downloading and preprocessing of LGRIP30 data.

    Implements `load_data()` to download LGRIP tiles.
    Implements `create_grid_aggregates()` to calculate pixel counts per grid
    cell for each category in the data.

    Attributes:
        data_key (str): Set to "lgrip".
        local (bool): Set to False, as data is downloaded via http.
        files (list[str]): List of files for individual tiles in storage.
        data_loaded (bool): Flag to indicate whether the data is completely in 
            storage.
    """
    data_key: str = "lgrip"
    local: bool = False
    
    def __init__(self, config: ConfigParser):
        """Initializes the LGRIPData instance.

        Checks the processing directory for already downloaded GeoTIFF
        tiles and sets an initial status flag for data loading.

        Args:
            config (ConfigParser): An initialized ConfigParser instance.
        """
        super().__init__(config=config)
        self.files = [f for f in os.listdir(self.storage.storage_paths["processing"]) if f.endswith(".tif")]
        self.data_loaded = False


    def load_data(self):
        """Downloads LGRIP GeoTIFF tiles if they are missing.

        Scrapes the LGRIP data directory webpage to get a list of all available
        GeoTIFF tiles. If regeneration is not forced, it compares this to the
        list of files in storage to determine wheter files are missing. For
        the remaining (or all if regenerate["data"]=True) files iterates through
        them, and downloads them via the `_download_tile()` method, updating
        `self.files` in the process. Requires Earthdata user and password loaded
        from the .env file. Sets `self.data_loaded` to True upon successful completion.

        This method populates the processing storage with the raw LGRIP GeoTIFFs
        and does not return data directly.
        """
        url = 'https://e4ftl01.cr.usgs.gov/COMMUNITY/LGRIP30.001/2014.01.01/'
        response = requests.get(url)
        response.raise_for_status()  # we want an error if loading the site doesn't work
        web_text = response.text
        files = set(re.findall('LGRIP30_2015_.{24,26}\.tif', web_text))
        if not self.regenerate["data"]:
            # only download files not in storage
            files = [f for f in files if f not in self.files]
        else:
            # set self.files to empty list since we want to re-download
            self.files = []
        
        self.console.print("Downloading LGRIP files...")
        if len(files) == 0:
            self.console.print("All required files already in local storage.")
            self.data_loaded = True
            return
        else:                
            with Progress(console=self.console) as progress:
                task_download = progress.add_task("[green]Downloading (missing) tiles ...", total=len(files))
                session = requests.Session()
                for filename in files:
                    fp = self.storage.build_filepath("processing", filename, filetype="")
                    url_tile = url + filename
                    self._download_tile(session, fp, url_tile)
                    self.files.append(filename)
                    progress.update(task_download, advance=1)
            self.data_loaded = True
            return

    @staticmethod
    def _download_tile(session: requests.Session, fp: str, url_tile: str) -> None:
        """Static method to download a single LGRIP GeoTIFF tile.

        Uses a provided Session object to download a file from the url.
        Handles authetication via the EARTHDATA credentials loaded from .env by
        redirecting to the `urs.earthdata.nasa.gov` login page. 
        
        Args:
            session (requests.Session): A `requests.Session` object to maintain
                cookies and connection state across potential redirects.
            fp (str): The local filepath where the downloaded GeoTIFF will be saved.
            url_tile (str): The URL of the LGRIP GeoTIFF tile to download.
        """
        response = session.get(url_tile)
        # handle redirect and authentication via earthdata
        if response.status_code == 401 and 'https://urs.earthdata.nasa.gov/' in response.url:
            user = os.getenv('EARTHDATA_USER')
            pw = os.getenv('EARTHDATA_PASSWORD')
            assert user is not None and pw is not None, "Please specify 'EARTHDATA_USER' and 'EARTHDATA_PASSWORD' env variables."
            response = session.get(response.url, auth=(user, pw))
        response.raise_for_status() # we want an error if download doesnt work
        with open(fp, 'wb') as f:
            for chunk in response.iter_content(chunk_size=2048):
                f.write(chunk)
        return None

    def create_grid_aggregates(self, grid: GlobalBaseGrid):
        """Aggregates high-resolution LGRIP tile data to the 0.5-degree grid.

        If `self.regenerate['preprocessing']` is False, this method first attempts
        to load previously aggregated results from the processing directory.

        Otherwise, initializes an empty DataFrame based on the input `grid` 
        structure, iterates through downloaded LGRIP GeoTIFF tiles, and identifies 
        grid cells within the tile's spatial extent. For each grid cell, counts 
        the number of pixels belonging to each croplandcategory (0: water, 
        1: non-cropland, 2: irrigated cropland, 3: rainfed cropland) and writes 
        this to the DataFrame. Saves the DataFrame as `lgrip_grid_aggregates.parquet`.

        Args:
            grid (GlobalBaseGrid): The initialized GlobalBaseGrid instance.

        Returns:
            pd.DataFrame: A DataFrame indexed by 'pgid' containing columns for
                the pixel counts of each cropland category for every grid cell.
        """
        try:
            if self.regenerate["preprocessing"]:
                raise FileNotFoundError
            df = self.storage.load("processing", "lgrip_grid_aggregates")
        except FileNotFoundError:
            df = grid.load()
            df = df.drop(columns=df.columns)
            pgids = df.index.get_level_values('pgid').unique()
            path = self.storage.storage_paths["processing"]
            fps = [os.path.join(path, f) for f in self.files]
            with Progress(console=self.console) as progress:
                task_aggregate = progress.add_task("[green]Aggregating tiles to grid...", total=len(fps))
                for fp in fps:
                    da = rxr.open_rasterio(fp).squeeze()
                    lons = np.arange(round(da.x.min().item())+.25, da.x.max().item(), .5)
                    lats = np.arange(round(da.y.min().item())+.25, da.y.max().item(), .5)
                    coords = list(itertools.product(lats, lons))
                    pgids_tile = [pgid for pgid in pgids if pgid_to_coords(pgid) in coords]
                    task_tile = progress.add_task(f"[green]Processing tile...", total=len(pgids_tile))
                    for pgid in pgids_tile:
                        y,x = pgid_to_coords(pgid)
                        da_cell = da.sel(x=slice(x-0.25,x+0.25), y=slice(y+0.25,y-0.25))
                        df.loc[pgid, 'water'] = (da_cell.values==0).sum().item()
                        df.loc[pgid, 'non_cropland'] = (da_cell.values==1).sum().item()
                        df.loc[pgid, 'cropland_irrigated'] = (da_cell.values==2).sum().item()
                        df.loc[pgid, 'cropland_rainfed'] = (da_cell.values==3).sum().item()
                        progress.update(task_tile, advance=1)
                    progress.remove_task(task_tile)
                    progress.update(task_aggregate, advance=1)
            self.storage.save(df, "processing", "lgrip_grid_aggregates")
        return df