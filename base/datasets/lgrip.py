import itertools
import os

from dotenv import load_dotenv
import earthaccess
import numpy as np
import pendulum
from rich.progress import Progress
import rioxarray as rxr

from base.objects import Dataset, ConfigParser, GlobalBaseGrid
from utils.conversions import pgid_to_coords


class LGRIPData(Dataset):
    """Handles downloading and preprocessing of LGRIP30 data.

    Implements `load_data()` to download LGRIP tiles.
    Implements `create_grid_aggregates()` to calculate pixel counts per grid
    cell for each category in the data.

    Attributes:
        data_key (str): Set to "lgrip".
        local (bool): Set to False, as data is downloaded via NASA earthdata.
        files (list[str]): List of files for individual tiles in storage.
        data_loaded (bool): Flag to indicate whether the data is completely in
            storage.
    """

    data_key: str = "lgrip"
    local: bool = False

    def __init__(self, config: ConfigParser, name: str, version: str):
        """Initializes the LGRIPData instance.

        Checks the processing directory for already downloaded GeoTIFF
        tiles and sets an initial status flag for data loading.

        Args:
            config (ConfigParser): An initialized ConfigParser instance.
            name (str): Name of the dataset in Earthdata.
            version (str): Version of the dataset in Earthdata.
        """
        super().__init__(config=config)
        self.files = [
            f for f in os.listdir(self.storage.storage_paths["processing"]) if f.endswith(".tif")
        ]
        self.data_loaded = False
        self.name = name
        self.version = version

    def load_data(self, start_date: str, end_date: str | None = None):
        """Downloads LGRIP GeoTIFF tiles if they are missing.

        Queries NASA's earthdata for available granules. If regeneration is not
        forced, it compares them to the list of files in storage to determine
        whether files are missing. For the remaining (or all if regenerate["data"]=True)
        granules iterates through them, and downloads them, updating `self.files`
        in the process. Requires Earthdata user and password set in the .env
        file. Sets `self.data_loaded` to True upon successful completion.

        This method populates the processing storage with the raw LGRIP GeoTIFFs
        and does not return data directly.

        Args:
            start_date (str): Start date for the time window of the desired granules.
                Isoformat or parseable date string.
            end_date (str | None): End date for the time window of the desired granules.
                If not None, isoformat or parseable date string or None. If None,
                defaults to current date (i.e. all after start date).

        """
        granules = self._search_earthdata_granules(start_date, end_date)
        if self.regenerate["data"]:
            # set self.files to empty list since we want to re-download
            self.files = []
        # only download files not in storage
        granules_download = [
            g for g in granules if os.path.basename(g.data_links()[0]) not in self.files
        ]

        self.console.print("Downloading LGRIP files...")
        if len(granules_download) == 0:
            self.console.print("All required files already in local storage.")
        else:
            # login to download
            load_dotenv()
            earthaccess.login(strategy="environment")
            with Progress(console=self.console) as progress:
                task_download = progress.add_task(
                    "[green]Downloading (missing) tiles ...", total=len(granules_download)
                )
                for granule in granules_download:
                    fp = self.storage.storage_paths["processing"]
                    res = earthaccess.download(granule, local_path=fp)
                    progress.update(task_download, advance=1)
        self.data_loaded = True
        return

    def _search_earthdata_granules(
        self, start_date: str, end_date: str | None = None
    ) -> list[earthaccess.DataGranule]:
        """Queries earthdata for existing granules for a dataset, version, and time period.

        Args:
            start_date (str): Start date for the time window of the desired granules.
                Isoformat or parseable date string.
            end_date (str | None): End date for the time window of the desired granules.
                If not None, isoformat or parseable date string or None. If None,
                defaults to current date (i.e. all after start date).

        Returns:
            (list[earthaccess.results.DataGranule]): Result from earthaccess query.
        """
        # input validation: parse and convert back to isoformat
        start_date_parsed = pendulum.parse(start_date).date().isoformat()
        if end_date is None:
            end_date_parsed = pendulum.now().date().isoformat()
        else:
            end_date_parsed = pendulum.parse(end_date).date().isoformat()

        results = earthaccess.search_data(
            short_name=self.name,
            version=self.version,
            temporal=(start_date_parsed, end_date_parsed),
        )
        return results

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
            pgids = df.index.get_level_values("pgid").unique()
            path = self.storage.storage_paths["processing"]
            fps = [os.path.join(path, f) for f in self.files]
            with Progress(console=self.console) as progress:
                task_aggregate = progress.add_task(
                    "[green]Aggregating tiles to grid...", total=len(fps)
                )
                for fp in fps:
                    da = rxr.open_rasterio(fp).squeeze()
                    lons = np.arange(round(da.x.min().item()) + 0.25, da.x.max().item(), 0.5)
                    lats = np.arange(round(da.y.min().item()) + 0.25, da.y.max().item(), 0.5)
                    coords = list(itertools.product(lats, lons))
                    pgids_tile = [pgid for pgid in pgids if pgid_to_coords(pgid) in coords]
                    task_tile = progress.add_task(
                        "[green]Processing tile...", total=len(pgids_tile)
                    )
                    for pgid in pgids_tile:
                        y, x = pgid_to_coords(pgid)
                        da_cell = da.sel(x=slice(x - 0.25, x + 0.25), y=slice(y + 0.25, y - 0.25))
                        df.loc[pgid, "water"] = (da_cell.values == 0).sum().item()
                        df.loc[pgid, "non_cropland"] = (da_cell.values == 1).sum().item()
                        df.loc[pgid, "cropland_irrigated"] = (da_cell.values == 2).sum().item()
                        df.loc[pgid, "cropland_rainfed"] = (da_cell.values == 3).sum().item()
                        progress.update(task_tile, advance=1)
                    progress.remove_task(task_tile)
                    progress.update(task_aggregate, advance=1)
            self.storage.save(df, "processing", "lgrip_grid_aggregates")
        return df
