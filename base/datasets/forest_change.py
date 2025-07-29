from typing import Iterable, Any
import numpy as np
import re
import os

import boto3
import botocore

from base.objects import ConfigParser, Dataset, GlobalBaseGrid
from utils.spatial_operations import pgid_to_coords, s_ceil, s_floor


class GFCData(Dataset):
    """Handles downloading and preprocessing of Global Forest Change (GFC) data.

    Implements `load_data()` to download the current version of the GFC data
    from a Google Cloud Storage bucket.

    Attributes:
        data_key (str): Set to "gfc".
        local (bool): Set to False, as data is sourced from cloud storage.
        version (str): Current GFC version, checked and set during initialization.
        gfc_files (list[str]): List of GFC filenames found in the processing
            storage, set during initialization (set to empty if `regenerate['data']`
            is True) and updated with data downloads.
        data_loaded (bool): Flag indicating whether all available data is downloaded.
            Checked at the start of preprocessing.
    """

    data_key: str = "gfc"
    local: bool = False
    bucket_name = "earthenginepartners-hansen"

    def __init__(self, config: ConfigParser):
        """Initializes the GFCData instance.

        Args:
            config (ConfigParser): An initialized ConfigParser instance.
        """
        super().__init__(config=config)
        self.s3_client = boto3.client(
            "s3",
            endpoint_url="https://storage.googleapis.com",
            config=botocore.client.Config(signature_version=botocore.UNSIGNED),  # type: ignore
        )
        self.version = self._find_latest_forest_change_version()
        # check which files are already in storage
        self.gfc_files = self._check_version_files()
        self.data_loaded = False

    def _check_version_files(self) -> list[str]:
        files_storage = [
            f for f in os.listdir(self.storage.storage_paths["processing"]) if f.endswith(".tif")
        ]
        if self.regenerate["data"]:
            valid_files = []
        else:
            files_current_version = [f for f in files_storage if self.version in f]
            if files_current_version == files_storage:
                valid_files = files_storage
            else:
                files_different_version = [
                    f for f in files_storage if f not in files_current_version
                ]
                self.console.print(
                    f"Deleting {len(files_different_version)} files not from the current GFC version."
                )
                for f in files_different_version:
                    os.remove(self.storage.build_filepath("processing", f, filetype=""))
                valid_files = files_current_version
        return valid_files

    def _find_latest_forest_change_version(self) -> str:
        """Finds the latest version of the dataset querying the public GCS bucket.

        This function uses an anonymous boto3 client to list the top-level "directories"
        (common prefixes) in the public GCS bucket for the GFC dataset. It parses
        these prefixes to identify all unique versioned directories, sorts them, and
        returns the name of the most recent one.

        Returns:
            str: The name of the latest version (directory) (e.g., "GFC-2023-v1.11").
        """

        # Configure boto3 for anonymous (unsigned) access to a public bucket
        paginator = self.s3_client.get_paginator("list_objects_v2")
        # Use Delimiter='/' to list only top-level "directories" (prefixes)
        pages = paginator.paginate(Bucket=self.bucket_name, Delimiter="/")
        directories = []
        for page in pages:
            for prefix in page.get("CommonPrefixes", []):
                # Extract the directory name, removing the trailing '/'
                directories.append(prefix.get("Prefix", "").strip("/"))

        version_pattern = re.compile(r"GFC-(\d{4})-v(\d+)\.(\d+)")
        parsed_versions = []

        for dirname in directories:
            match = version_pattern.match(dirname)
            if match:
                parsed_versions.append(dirname)

        if not parsed_versions:
            raise FileNotFoundError("No directories matching the GFC version pattern were found.")

        parsed_versions.sort()
        latest_version_name = parsed_versions[-1]
        return latest_version_name

    @staticmethod
    def get_tiles_for_grid(pgids: Iterable[int]) -> list[str]:
        """Generates strings representing relevant 10x10 degree GFC tiles based on pgids."""
        tiles = []
        for pgid in pgids:
            lat, lon = pgid_to_coords(pgid)
            lat_boundary = s_ceil(lat, 10)
            lat_postfix = "N" if lat_boundary >= 0 else "S"
            lat_boundary_str = f"{abs(int(lat_boundary)):02d}{lat_postfix}"
            lon_boundary = s_floor(lon, 10)
            lon_postfix = "E" if lon_boundary >= 0 else "W"
            lon_boundary_str = f"{abs(int(lon_boundary)):03d}{lon_postfix}"
            tiles.append(f"{lat_boundary_str}_{lon_boundary_str}")
        return sorted(list(np.unique(tiles)))

    @staticmethod
    def get_pgids_for_tile(tile_loc: str, pgids: Iterable[int]) -> list[int] | None:
        # parse tile string to get boundaries
        max_lat = int(tile_loc[:2])
        lat_postfix = tile_loc[2]
        if lat_postfix == "S":
            max_lat = max_lat * -1
        min_lat = max_lat - 10
        min_lon = int(tile_loc[4:7])
        lon_postfix = tile_loc[7]
        if lon_postfix == "W":
            min_lon = min_lon * -1
        max_lon = min_lon + 10
        # check which pgids are in tile
        pgids_filtered = []
        for pgid in pgids:
            lat, lon = pgid_to_coords(pgid)
            if min_lat < lat < max_lat and min_lon < lon < max_lon:
                pgids_filtered.append(pgid)
        if len(pgids_filtered) == 0:
            return None
        else:
            return pgids_filtered

    @staticmethod
    def _gfc_filename(version: str, layer: str, tile_loc: str) -> str:
        return f"Hansen_{version}_{layer}_{tile_loc}.tif"

    def load_data(self, grid: GlobalBaseGrid) -> None:
        """Downloads latest version of GFC data.

        This method does not return data directly but populates the processing storage.
        """
        pgids = grid.load().reset_index()["pgid"].to_list()
        relevant_tiles = self.get_tiles_for_grid(pgids)
        required_files: list[str] = []
        for tile in relevant_tiles:
            for layer in ["datamask", "lossyear", "treecover2000"]:
                required_files.append(self._gfc_filename(self.version, layer, tile))
        files_to_download = [f for f in required_files if f not in self.gfc_files]
        # regenerate data is handled by the file check durin init
        if len(files_to_download) == 0:
            self.console.print("All required files already in storage.")
        else:
            self.console.print(
                f"{len(files_to_download)}/{len(required_files)} potential files missing in storage. Expected are 63 based on 2024."
            )
            # with Progress(console=self.console) as progress:
            #     task_download = progress.add_task("Downloading GFC", total=len(files_to_download))
            #     for filename in files_to_download:
            #         object_key = f"{self.version}/{filename}"
            #         filepath = self.storage.build_filepath("processing", filename, filetype="")
            #         try:
            #             self.s3_client.download_file(self.bucket_name, object_key, filepath)
            #         except botocore.exceptions.ClientError as e:
            #             if e.response["Error"]["Code"] == "404":
            #                 self.console.print(f"404 Error: File not found in bucket: {object_key}")
            #             else:
            #                 self.console.print(f"Error downloading tile: {e}")
            #         progress.update(task_download, advance=1)
        self.data_loaded = True
        return

    def get_dict_files(self, grid: GlobalBaseGrid) -> dict[str, Any]:
        pgids = grid.load().reset_index()["pgid"].to_list()
        relevant_tiles = self.get_tiles_for_grid(pgids)
        file_dict = {tile: {} for tile in relevant_tiles}
        layers = ["datamask", "lossyear", "treecover2000"]
        for tile in file_dict:
            for layer in layers:
                filename = self._gfc_filename(self.version, layer, tile)
                file_dict[tile][layer] = (
                    self.storage.build_filepath("processing", filename, filetype="")
                    if filename in self.gfc_files
                    else None
                )
            if all(file_dict[tile][layer] is None for layer in layers):
                file_dict[tile] = None  # type: ignore
        return file_dict
