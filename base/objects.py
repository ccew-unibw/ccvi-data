from abc import abstractmethod, ABC
import itertools
import os
from typing import Any, Literal, overload

import geopandas as gpd
import numpy as np
import pandas as pd
from panel_imputer import PanelImputer
import pyproj
from rich.console import Console
from scipy.stats import gmean, pmean
from shapely.geometry import box
from tqdm import tqdm
import yaml


from utils.data_processing import (
    add_time,
    min_max_scaling,
    create_custom_data_structure,
)
from utils.index import get_quarter
from utils.spatial_operations import coords_to_pgid, s_ceil, s_floor

console = Console(force_terminal=True, force_interactive=True)


class ConfigParser:
    """
    Parses configuration settings from a YAML file.

    This class loads a specified YAML configuration file upon initialization
    and provides methods to retrieve specific sections (global, indicator, data,
    aggregation) of the configuration.

    Attributes:
        console (Console): A console object for displaying messages.
        config_path (str): The path to the YAML configuration file used.
        all_config (dict): A dictionary holding the entire parsed content of the
            YAML file.
    """

    console: Console = console

    def __init__(self, config_path: str = "config.yaml"):
        """Initializes the ConfigParser by loading and validating the config file.

        Args:
            config_path (str, optional): The path to the YAML configuration
                file. Defaults to "config.yaml" in the repository root.
        """
        self.config_path = config_path
        self.all_config = self._load_yaml(config_path)
        self._validate_top_level_config()
        self.console.print(f"Global config: {self.get_global_config()}")

    def _validate_top_level_config(self):
        """Validates the presence and uniqueness of required top-level keys."""
        config_types = {"indicators", "aggregation", "data", "global"}
        current_keys = set(self.all_config.keys())

        if not config_types.issubset(current_keys):
            missing = config_types - current_keys
            raise ValueError(
                f"Top-level config keys missing in '{self.config_path}'. Missing: {missing}. "
                f"Required: {config_types}."
            )

    def get_indicator_config(self, pillar: str, dim: str, id: str) -> dict[str, Any] | None:
        """Retrieves the configuration for a specific indicator.

        Navigates the nested structure under the 'indicators' top-level key.

        Args:
            pillar (str): The pillar ID of the indicator.
            dim (str): The dimension ID of the indicator.
            id (str): The specific ID of the indicator.

        Returns:
            dict[str, Any] | None: The configuration dictionary for the specified
                indicator or None if no config is provided.
        """
        try:
            config = self.all_config["indicators"][pillar][dim][id]
        # Not providing a config results in a TypeError or KeyError
        except TypeError:
            config = None
        except KeyError:
            config = None
        return config

    def get_data_config(self, data_keys: str | list[str]) -> dict[str, str]:
        """Retrieves data source file paths, returning full paths.

        Looks up keys under the 'data' section and prepends the configured
        input directory path.

        Args:
            data_keys (str | list[str]): A single key or list of keys
                corresponding to entries under the 'data' section in the YAML.

        Returns:
            dict[str, str]: A dictionary mapping each requested data key to its
                full file path (absolute or relative to the repository root,
                depending on the path specified in the global config).
        """
        # data config specifies the input paths
        path = os.path.join(self.get_global_config()["storage_path"], "input")
        if isinstance(data_keys, str):
            data_keys = [data_keys]
        try:
            config = {key: os.path.join(path, self.all_config["data"][key]) for key in data_keys}
        except KeyError:
            raise KeyError(
                f"One or multiple keys from {data_keys} not listed in data config. "
                "Please doublecheck the if all keys are specified in the config YAML."
            )
        return config

    def get_global_config(self) -> dict[str, Any]:
        """Retrieves the global configuration section.

        Validates that the required keys ('storage_path', 'start_year') are present.

        Returns:
            dict[str, Any]: The dictionary containing global configuration settings.
        """
        config = self.all_config["global"]
        global_keys = {"storage_path", "start_year", "regenerate"}
        assert set(config.keys()) == global_keys and len(config) == 3, (
            "Global config keys do not match requirements. Check for missing or duplicate keys."
            f"\nRequired: {global_keys}."
        )
        return config

    def get_regeneration_config(self, id: str) -> dict[str, bool]:
        """Builds the regeneration config for a specific component based on its id.

        Determines whether regeneration is desired based on the 'regenerate'
        section in the global configuration. The 'regenerate' section is expected
        to have keys: 'indicator', 'data', 'preprocessing'. The value for each
        key can be:
        - A list of IDs of the components to regenerate.
        - The string "all", indicating regeneration for all components.
        - None or an empty list, indicating no regeneration for that stage.

        Args:
            id (str): The composite ID of the component (`composite_id`
                for an indicator, or `data_key` for a dataset) for which the
                regeneration flags are being determined.

        Returns:
            dict[str, bool]: A dictionary with keys 'indicator', 'data', and
                'preprocessing', where each value is a boolean indicating whether
                that stage should be regenerated for the given `id`.
        """
        regeneration_config = self.get_global_config()["regenerate"]
        regenerate_config_keys = {"indicator", "data", "preprocessing"}
        assert set(regeneration_config.keys()) == regenerate_config_keys
        (
            "Regenerate keys do not match requirements. Check for missing keys."
            f"\nRequired: {regenerate_config_keys}."
        )
        regeneration_config_id = {}
        for key in regeneration_config:
            if regeneration_config[key] is None:
                regeneration_config_id[key] = False
            elif "all" in regeneration_config[key]:
                regeneration_config_id[key] = True
            elif id in regeneration_config[key]:
                regeneration_config_id[key] = True
            else:
                regeneration_config_id[key] = False
        return regeneration_config_id

    def set_regenerated_globally(
        self, id: str, key: Literal["data", "preprocessing", "indicator", None] = None
    ) -> None:
        """Modifies the global config in memory to mark a component as regenerated.

        This method removes the specified `id` from the regeneration lists
        stored in `self.all_config['global']['regenerate']` if found. This prevents
        subsequent calls to `get_regeneration_config` for the same `id` (and `key`)
        from returning True, effectively ensuring that a component is not
        unnecessarily regenerated multiple times within the same session if it was
        initially flagged for regeneration.

        Note: This modifies the `self.all_config` dictionary in place. It does
        not write changes back to the `config.yaml` file.

        Args:
            id (str): The composite ID of the component that has been processed
                and should no longer be flagged for regeneration.
            key (Literal["data", "preprocessing", "indicator", None], optional):
                The specific regeneration stage to clear for the `id`.
                If "data", "preprocessing", or "indicator", only that specific
                list is modified. If None (default), the `id` is removed from
                all regeneration lists if present.
        """
        global_regenerate_config = self.all_config["global"]["regenerate"]
        if key is None:
            for key in global_regenerate_config:
                if global_regenerate_config[key] is None:
                    pass
                elif id in global_regenerate_config[key]:
                    global_regenerate_config[key].remove(id)
        elif global_regenerate_config[key] is None:
            pass
        else:
            if id in global_regenerate_config[key]:
                global_regenerate_config[key].remove(id)
        self.all_config["regenerate"] = global_regenerate_config
        return

    def get_aggregation_config(self, pillar: str, dim: str | None = None) -> dict[str, Any]:
        """Retrieves the aggreation configuration for a specific dimension / pillar.

        Args:
            pillar (str): The pillar ID.
            dim (str | None, optional): The dimension ID. Defaults to None

        Returns:
            dict[str, Any]: The configuration dictionary for the aggregation step.
        """
        if dim is not None:
            config = self.all_config["aggregation"][pillar][dim]
        else:
            config = self.all_config["aggregation"][pillar]["pillar"]
        return config

    def _load_yaml(self, path: str) -> dict[str, Any]:
        """
        Loads a YAML file from the specified path.

        Args:
            path (str): The file path to the YAML configuration file.

        Returns:
            dict[str, Any]: A dictionary representing the parsed YAML content.
        """
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return config_dict


class StorageManager:
    """Manages filepaths and storage for CCVI components.

    Handles the creation of standard storage directories (input, processing,
    output) based on a base path. Provides methods for constructing file paths,
    checking file existence, saving/loading DataFrames to/from Parquet and
    checking component generation status. Also manages the component-specific
    `composite_id` used for indicator/aggregate scores.

    Attributes:
        console (Console): A console object for displaying messages.
        storage_base_path (str): The root directory for all storage operations.
        requires_processing_storage (bool, optional): Flag indicating whether a
            processing storage folder is required by the class using this instance
            of StorageManager.
        storage_paths (dict[str, str]): A dictionary mapping processing stages
            ('input', 'processing', 'output') to their corresponding directory paths.
            Automatically generated based on the storage_base_path setting
            during initialization.
        composite_id (str, optional): A unique identifier for the component using
            this storage manager. Can be set via `set_composite_id` for indicators
            and dimensions. Used as default filename where available.
    """

    console: Console = console

    def __init__(
        self,
        storage_base_path: str,
        requires_processing_storage: bool = False,
        processing_folder: str | None = None,
    ):
        """Initializes the StorageManager and sets up directories.

        Args:
            storage_base_path (str): The root directory where 'input',
                'processing', and 'output' subdirectories will be created.
            requires_processing_storage (bool, optional): Flag indicating whether a
                processing storage folder is required by the class using this instance
                of StorageManager.
            processing_folder (str | None, optional): Name of subfolder the
                processing folder to distinguish processing output from the
                different components. Only required if processing storage indicated,
                otherwise will be ignored. If `set_composite_id()` is called after setup,
                the composite_id will be used and no argument needs to be provided.
        """
        if not storage_base_path or not isinstance(storage_base_path, str):
            raise ValueError("StorageManager requires a valid storage_base_path string.")
        self.storage_base_path = storage_base_path
        self.requires_processing_storage = requires_processing_storage
        if self.requires_processing_storage and processing_folder is None:
            self.console.print(
                "No processing folder provided but required processing storage indicated. Unless "
                "`set_composite_id()` is called from the class using the this StorageManager "
                "instance, there will be no processing folder created, which may cause problems "
                "later on."
            )
        if not self.requires_processing_storage and processing_folder is not None:
            processing_folder = None
            self.console.print(
                "Processing folder provided but required processing storage not indicated. No "
                "processing folder will be created."
            )
        self.storage_paths = self._setup_storage_paths(self.storage_base_path, processing_folder)

    def set_composite_id(
        self,
        pillar: str,
        dim: str | None = None,
        id: str | None = None,
        component_type: Literal["indicator", "dimension", "pillar"] = "indicator",
    ) -> None:
        """Sets the composite identifier for the index component.

        This ID is also used as the default filename for saving/loading indicators
        and as the processing subfolder for a component. Method prevents the use
        of underscores ('_') within individual component identifiers (pillar, dim, id).
        Updates self.storage_paths["processing"] with the composite_id as a subfolder
        and creates the folder if `self.requires_processing_storage` = True.

        Args:
            pillar (str): The pillar ID for the component.
            dim (str| None, optional): The dimension ID for the component.
            id (str | None, optional): The specific indicator ID.
                Required if `component_type` is "indicator". Defaults to None.
            component_type (str, optional): The type of component ("indicator",
                "dimension" or "pillar"), determining the structure of the composite ID.
                Defaults to "indicator".
        """
        if component_type == "indicator":
            if id is None:
                raise ValueError("An indicator composite ID requires an indicator id. Got None.")
            if dim is None:
                raise ValueError("An indicator composite ID requires a dimension id. Got None.")
            if "_" in pillar or "_" in dim or "_" in id:
                raise ValueError(
                    'Pillar, dimension, and indicator id components may not contain "_".'
                    " Please adjust accordingly."
                )
            self.composite_id = f"{pillar}_{dim}_{id}"
        elif component_type == "dimension":
            if dim is None:
                raise ValueError("A dimension composite ID requires a dimension id. Got None.")
            if "_" in pillar or "_" in dim:
                raise ValueError(
                    'Pillar and dimension id components may not contain "_".'
                    " Please adjust accordingly."
                )
            else:
                self.composite_id = f"{pillar}_{dim}"
        elif component_type == "pillar":
            if "_" in pillar:
                raise ValueError(
                    'Pillar and dimension id components may not contain "_".'
                    " Please adjust accordingly."
                )
            else:
                self.composite_id = pillar
        else:
            raise ValueError(
                f'Argument "component_type" can be one of ["indicator", "dimension", "pillar"], '
                f"got {component_type}."
            )
        # update processing path and cleanup
        self.storage_paths["processing"] = os.path.join(
            self.storage_base_path, "processing", self.composite_id
        )
        if self.requires_processing_storage:
            os.makedirs(self.storage_paths["processing"], exist_ok=True)

    def _setup_storage_paths(self, base_path: str, processing_folder: str | None) -> dict[str, str]:
        """Setup the storage paths based on a base_path.

        Creates 'input', 'processing', and 'output' subdirectories within the
        specified `storage_folder` if they don't exist, and returns the paths
        as dictionary. If `processing folder` is not None, the corresponding
        subdirectory will also be created.

        Args:
            base_path (str): The storage root directory.

        Returns:
            dict[str, str]: A dictionary mapping 'input', 'processing', 'output'
                to their created directory paths.
        """
        storage = {}
        for stage in ["input", "processing", "output"]:
            path = os.path.join(base_path, stage)
            if stage == "processing" and processing_folder is not None:
                path = os.path.join(path, processing_folder)
            os.makedirs(path, exist_ok=True)
            storage[stage] = path
        return storage

    def save(
        self,
        df: pd.DataFrame,
        mode: Literal["processing", "output"] = "output",
        filename: str | None = None,
        subfolder: str | None = None,
    ) -> None:
        """Saves a DataFrame as a Parquet file in the specified processing stage directory.

        Creates the processing subfolder if it does not already exist.

        Args:
            df (pd.DataFrame): The DataFrame to be saved.
            mode (str, optional): Whether to save to the output or processing folder.
                Can be "processing" or "output". Defaults to "output".
            filename (str, optional): Desired filename (without extension).
                If None, defaults to `self.composite_id`. Must be provided if
                `composite_id` is not set.
            subfolder(str | None, optional): Optional subfolder under the storage path
                to be created if it does not exist and used.
        """
        if mode not in ["processing", "output"]:
            raise ValueError(f'Allowed values for mode are ["processing", "output"], got {mode}')
        if mode == "processing":
            os.makedirs(self.storage_paths["processing"], exist_ok=True)
        fp = self.build_filepath(mode, filename, subfolder)
        df.to_parquet(fp, compression="brotli")

    def load(
        self,
        mode: Literal["processing", "output"] = "output",
        filename: str | None = None,
        subfolder: str | None = None,
    ) -> pd.DataFrame:
        """Loads a DataFrame from a Parquet file in the specified processing stage directory.

        Args:
            mode (str): Whether to load from the output or processing folder.
                Can be "processing" or "output". Defaults to "output".
            filname (str, optional): Filename of the stored parquet file (without
                ".parquet" extension). If None, defaults to `self.composite_id`.
                Must be provided if `composite_id` is not set.
            subfolder(str | None, optional): Optional subfolder under the storage path
                to be created if it does not exist and used.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        if mode not in ["processing", "output"]:
            raise ValueError(f'Allowed values for mode are ["processing", "output"], got {mode}')
        fp = self.build_filepath(mode, filename, subfolder)
        self.check_exists(fp)
        df = pd.read_parquet(fp)
        return df

    def check_exists(self, fp: str) -> None:
        """Checks if a file exists at the given path. Raises FileNotFoundError if not.

        Args:
            fp (str): The full file path to check.
        """
        try:
            assert os.path.exists(fp)
        except AssertionError:
            raise FileNotFoundError(f"File {fp} does not exist.")

    def check_component_generated(self, time: int = -1, **load_kwargs) -> bool:
        """Checks if component output exists and is up-to-date for a target quarter.

        Attempts to load the component's output file (using `self.load` with
        default 'output' mode and `self.composite_id` filename, unless overridden
        by `load_kwargs`). It then checks if the latest timestamp in the loaded data
        is at least as recent as the quarter specified by `time`.

        Args:
            time (int, optional): Determines the relevant quarter for the check.
                Used as the 'which' argument in `get_quarter()`. Defaults to -1
                (most recent quarter).
            **load_kwargs (Any):  Optional keyword arguments passed to the
                `self.load()` method.
        """
        generated = False
        try:
            df = self.load(**load_kwargs)
            last_quarter = get_quarter(which=time)
            year = last_quarter.year
            quarter = (last_quarter.month + 2) / 3
            data_year = df.sort_index().iloc[-1:].reset_index().year.item()
            data_quarter = df.sort_index().iloc[-1:].reset_index().quarter.item()
            if year <= data_year and quarter <= data_quarter:
                generated = True
        except FileNotFoundError:
            pass
        return generated

    def build_filepath(
        self,
        mode: Literal["input", "processing", "output"],
        filename: str | None = None,
        subfolder: str | None = None,
        filetype: str = ".parquet",
    ) -> str:
        """Constructs the full filepath for a file in a given storage stage.

        For 'input' mode, it assumes `filename` is already the full path (as
        constructed by `ConfigParser.get_data_config`). For 'processing'
        and 'output' modes, it joins the stage directory with the filename and
        adds the `filetype` extension. If desired, will add a subfolder to the
        filepath and makes sure it exists. Uses `self.composite_id` if `filename`
        is None.

        Args:
            mode (str): The processing stage. Must be in ["input", "processing", "output"].
            filename (str | None, optional): The base filename or full path
                (depending on `mode`). If None, defaults to `self.composite_id`.
                Must be provided if `self.composite_id` is not set.
            subfolder(str | None, optional): Optional subfolder under the storage path
                to be created if it does not exist and used.
            filetype (str, optional): Filetype to use when building a filepath. Used to build
                paths for storage of different data not handled through save/load. Defaults
                to 'parquet'.

        Returns:
            str: The constructed filepath.
        """
        path = self.storage_paths[mode]
        if filename is None:
            try:
                filename = self.composite_id
            except AttributeError:
                raise ValueError("Please provide a filename in an object without composite_id.")
        # for inputs the full path and filename is provided through the config
        if mode == "input":
            fp = filename
        else:
            if subfolder is not None:
                path = os.path.join(path, subfolder)
                os.makedirs(path, exist_ok=True)
            fp = os.path.join(path, f"{filename}{filetype}")
        return fp


class GlobalBaseGrid:
    """Generates and manages a global geospatial grid including country-matching.

    This class handles the creation of a base grid at a specified resolution,
    loads necessary geographical data for filtering (land and water masks), assigns
    grid cells to countries based on a majority rule, and provides methods to save
    and load the resulting grid data.

    Attributes:
        console (Console): A console object for displaying messages.
        grid_size (float): The size of the grid cells in decimal degrees.
            Defaults to 0.5 in the CCVI aligned to the PRIO-GRID.
        global_config (dict): Dictionary containing global configuration settings.
        config (dict): Dictionary containing paths to required data files
            (countries, land_mask, inland_water_mask).
        regenerate (bool): Whether regeneration of the grid is desired based on
            the global config regenerate settings. (Regeneration is possible by
            adding "base_grid" to `regenerate: preprocessing` in the config.yaml).
        storage (StorageManager): An initialized StorageManager instance for
            handling data storage based on global config's storage_path.
        basemap (gpd.GeoDataFrame): Country basemap for matching. Set via
            `create_country_basemap()` during init.
    """

    console: Console = console
    grid_size: float = 0.5

    def __init__(self, config: ConfigParser):
        """Initializes the GlobalBaseGrid instance.

        Loads global configurations and input data locations using the provided
        ConfigParser, sets up the StorageManager for data persistence, and checks
        for the existence of required input data files.

        Args:
            config (ConfigParser): An initialized ConfigParser instance used to
                retrieve configuration settings.
        """
        self.global_config = config.get_global_config()
        self.config = config.get_data_config(["countries", "land_mask", "inland_water_mask"])
        self.regenerate = config.get_regeneration_config("base_grid")["preprocessing"]
        self.storage = StorageManager(
            storage_base_path=self.global_config["storage_path"],
            requires_processing_storage=True,
            processing_folder="base_grid",
        )
        for key in self.config:
            self.storage.check_exists(self.config[key])
        self.basemap = self.create_country_basemap()

    def load_filter_data(self) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Loads and preprocessed country boundaries and land/water masks.

        Returns:
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]: A tuple
                containing the processed GeoDataFrames for countries, land mask,
                and inland water mask, respectively.
        """
        countries = self.create_country_basemap()
        land = gpd.read_file(self.config["land_mask"])
        lakes = gpd.read_file(self.config["inland_water_mask"])
        return countries, land, lakes

    def create_grid(
        self, bounds: list[float] = [-180, -90, 180, 90], xbuffer: float = 0, ybuffer: float = 0
    ) -> gpd.GeoDataFrame:
        """Creates a geospatial grid of rectangular cells.

        This method generates a grid covering a specified geographic area,
        defined by longitude and latitude boundaries. The grid cells are uniform
        squares in degrees. Optional buffers can be added, and the final
        boundaries are adjusted outwards to align with the grid cell size.

        Args:
            bounds (list[float], optional): A list defining the initial bounding
                box as `[lon_min, lat_min, lon_max, lat_max]`. Defaults to
                `[-180, -90, 180, 90]`, covering the full extent of the world.
                Note that the final grid extent may be larger due to outward
                rounding based on cell size `s`.
            xbuffer (float, optional): An additional buffer in decimal degrees to
                add to the minimum and maximum longitude values specified in
                `bounds` *before* the final boundaries are calculated by rounding.
                Defaults to 0.
            ybuffer (float, optional): An additional buffer in decimal degrees to
                add to the minimum and maximum latitude values specified in
                `bounds` *before* the final boundaries are calculated by rounding.
                Defaults to 0.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing all grid cells.
        """
        assert len(bounds) == 4, (
            'Please provide all 4 boundaries [lon_min, lat_min, lon_max, lat_max] in the "bounds" argument.'
        )
        lon_min, lat_min, lon_max, lat_max = bounds[0], bounds[1], bounds[2], bounds[3]
        s = self.grid_size
        # calculate boundaries
        lon_min = s_floor(lon_min - xbuffer, s)
        lat_min = s_floor(lat_min - ybuffer, s)
        lon_max = s_ceil(lon_max + xbuffer, s)
        lat_max = s_ceil(lat_max + ybuffer, s)

        assert lon_min >= -180 and lon_max <= 180 and lat_min >= -90 and lat_max <= 90, (
            "Grid bounds outside global bounds with chosen combination of bounds, grid_size and "
            f"buffer! Configuration results in grid bounds {float(lon_min)}, {float(lat_min)}, "
            f"{float(lon_max)}, {float(lat_max)}."
        )
        # Create grid center points and corresponding cell polygons
        lats = np.linspace(lat_min + s / 2, lat_max - s / 2, int((lat_max - lat_min) / s))
        lons = np.linspace(lon_min + s / 2, lon_max - s / 2, int((lon_max - lon_min) / s))
        midpoints = list(itertools.product(lats, lons))
        boxes = [box(p[1] - s / 2, p[0] - s / 2, p[1] + s / 2, p[0] + s / 2) for p in midpoints]
        # Combine into GeoDataFrame
        mi = pd.MultiIndex.from_tuples(midpoints, names=["lat", "lon"])
        gdf_grid = gpd.GeoDataFrame(index=mi, geometry=boxes, crs="epsg:4326")
        return gdf_grid

    def create_country_basemap(self) -> gpd.GeoDataFrame:
        """Preprocessed the raw GeoBoundaries country geometries.

        This method loads country geometries, ensures geometries are valid, and
        performs some adjustments: Merges West Bank and Gaza Strip into a single
        "PSE" entity, adjusts Western Sahara's administrative level status,
        removes Antarctica, and renames columns. It also filters disputed
        territories. Check for cached version first unless `self.regenerate`
        is True and returns the final GeoDataFrame.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing the processed country
                geometries.
        """

        fp_basemap = self.storage.build_filepath("processing", "basemap")
        if os.path.exists(fp_basemap) and not self.regenerate:
            cgaz = gpd.read_parquet(fp_basemap)
        else:
            self.console.print("Creating country basemap...")
            cgaz = gpd.read_file(self.config["countries"])
            cgaz["geometry"] = cgaz["geometry"].make_valid()
            # merge palestine and gaza as this is often not treated separately
            palestine_geom = cgaz.loc[
                cgaz["shapeName"].isin(["West Bank", "Gaza Strip"])
            ].union_all()
            palestine = gpd.GeoDataFrame(
                {
                    "shapeGroup": ["PSE"],
                    "shapeType": ["ADM0"],
                    "shapeName": [
                        "Palestine"
                    ],  # Custom name deviating from ISO 3166 ("Palestine, State of")
                    "geometry": [palestine_geom],
                }
            ).set_crs(cgaz.crs)  # type: ignore
            cgaz = pd.concat(
                [cgaz.loc[~cgaz["shapeName"].isin(["West Bank", "Gaza Strip"])], palestine],
                ignore_index=True,
            ).reset_index(drop=True)

            # Adjust Western Sahara's status (DISP --> ADM0)
            cgaz.loc[cgaz["shapeName"] == "Western Sahara", "shapeType"] = "ADM0"
            # Remove Antarctica
            cgaz.drop(cgaz[cgaz.shapeGroup == "ATA"].index, inplace=True)

            # Build a mapping country index to iso-code
            #
            # The mapping is constructed by simply enumerating the lexicographically sorted
            # list of ascending country codes
            cgaz = (
                cgaz.sort_values("shapeGroup")
                .reset_index(drop=True)  # Reset index to reflect new sort order and drop old one
                .reset_index()  # Move clean index into column
                .rename(
                    {
                        "index": "cid",
                        "shapeGroup": "cgaz",
                        "shapeType": "level",
                        "shapeName": "name",
                    },
                    axis=1,
                )
            )
            cgaz["disputed"] = cgaz["level"] != "ADM0"
            cgaz = cgaz[~cgaz.disputed]
            cgaz = cgaz.rename(columns={"cgaz": "iso3"})
            cgaz.to_parquet(fp_basemap, compression="brotli")
        return cgaz  # type: ignore

    def filter_and_match_grid(
        self,
        grid: gpd.GeoDataFrame,
        countries: gpd.GeoDataFrame,
        land: gpd.GeoDataFrame,
        lakes: gpd.GeoDataFrame,
    ):
        """Filters grid to land and assigns country ISO codes to grid cells.

        Filters the grid to keep only land cells (excluding major oceans and
        inland water bodies) Cells with less than 25% land coverage are discarded.
        For remaining cells, it assigns countries based on a majority rule using
        an Albers Equal Area re-projection centered on the cell.

        Args:
            grid (gpd.GeoDataFrame): The base grid generated by `create_grid`.
                Expected to have a MultiIndex (lat, lon).
            countries (gpd.GeoDataFrame): Country polygons with an 'iso3'
                column.
            land (gpd.GeoDataFrame): Polygons of land areas.
            lakes (gpd.GeoDataFrame): Polygons of inland water bodies.

        Returns:
            gpd.GeoDataFrame: A filtered version of the input grid containing only
                assigned land cells, with an added 'iso3' column indicating the
                assigned country code. The original MultiIndex is preserved.
        """

        def assign_area_name(geometry, lat: float, lon: float):
            """Helper function for use in .apply"""
            clipped = countries.clip(geometry)
            if clipped.empty:  # this means if not touching land - should not happen
                return "not_assigned"
            # reprojecting to albers equal area equidistant projection for area calculations
            lat_1 = lat - self.grid_size / 3
            lat_2 = lat + self.grid_size / 3
            aea_proj = pyproj.CRS(
                f'+proj=aea +ellps="WGS84" +lat_1={lat_1} +lat_2={lat_2} +lat_0={lat} +lon_0={lon} +units=m'
            )
            # get area
            clipped["area_size"] = clipped.to_crs(aea_proj).area
            # only assign country if grid cell covers more than 25% land (mostly coastal areas)
            total_area = grid.loc[[(lat, lon)], :].to_crs(aea_proj).area.iloc[0]
            if clipped.area_size.sum() / total_area < 0.25:
                return "not_assigned"
            else:
                # get index of the largest area in cell and assign its name to grid
                i = clipped.area_size.idxmax()
                return countries.loc[i]["iso3"]

        grid["lat"] = grid.index.get_level_values(0)
        grid["lon"] = grid.index.get_level_values(1)
        # remove oceans and lakes
        console.print("cropping to land... this may take some time...")
        # get land index based on natural earth land data - fine cropping is done with area assignment
        index_keep = grid.clip(land).index
        console.print("dropping grid cells with more than 75% water...")
        # get inland water index based on natural earth lakes data - remove all with more than 75% water
        index_touched_lakes = grid.clip(lakes.geometry).index
        index_remove = []
        for i in index_touched_lakes:
            cell = grid.loc[i]
            clipped = lakes.clip(cell.geometry)
            # reprojecting to albers equal area equidistant projection for area calculations
            lat_1 = cell.lat - self.grid_size / 3
            lat_2 = cell.lat + self.grid_size / 3
            aea_proj = pyproj.CRS(
                f'+proj=aea +ellps="WGS84" +lat_1={lat_1} +lat_2={lat_2} +lat_0={cell.lat} +lon_0={cell.lon} +units=m'
            )
            # get area
            clipped["area_size"] = clipped.to_crs(aea_proj).area
            # only assign country if grid cell covers more than 25% land (mostly coastal areas)
            total_area = grid.loc[[(cell.lat, cell.lon)], :].to_crs(aea_proj).area.iloc[0]
            if clipped.area_size.sum() / total_area > 0.75:
                index_remove.append(i)

        grid = grid.loc[index_keep].drop(index=index_remove)

        tqdm.pandas()  # needed for progress bar on (progress_)apply
        grid["iso3"] = grid.progress_apply(
            lambda x: assign_area_name(x.geometry, x.lat, x.lon), axis=1
        )  # type: ignore
        # only keep assigned areas for consistency and drop lat/lon columns again
        grid = grid.loc[grid["iso3"] != "not_assigned"].drop(columns=["lat", "lon"])
        return grid

    @overload
    def load(self, return_gdf: Literal[False] = False) -> pd.DataFrame: ...

    @overload
    def load(self, return_gdf: Literal[True]) -> gpd.GeoDataFrame: ...

    def load(self, return_gdf: bool = False) -> pd.DataFrame | gpd.GeoDataFrame:
        """Loads the base grid from storage.

        This wraps the StorageHandler, to provide an option to add the geometries
        back to the grid and return a GeoDataFrame. The grid is stored without
        geometries to save storage space.

        Args:
            return_gdf (bool, optional): If True, adds geometry column with polygons
                for each grid cell and returns a GeoDataFrame. If False (default),
                returns a simple pandas DataFrame.

        Returns:
            (pd.DataFrame | gpd.GeoDataFrame): The loaded base grid, either
                as a DataFrame or GeoDataFrame depending on `return_gdf`.
        """
        df = self.storage.load("output", filename="base_grid_prio")
        if not return_gdf:
            return df
        else:
            gdf = self._add_grid_geometry(df)
            return gdf

    def _add_grid_geometry(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Adds geometry polygons to a DataFrame with lat/lon information.

        Creates polygon geometries for each row based on 'lat' and 'lon' columns
        (assumed to be grid cell centers) and the class's `grid_size`.

        Args:
            df (pd.DataFrame): DataFrame containing 'lat' and 'lon' columns
               representing grid cell centers.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with added 'geometry' column.
        """
        s = self.grid_size
        gdf = gpd.GeoDataFrame(
            df,
            geometry=df.apply(
                lambda x: box(
                    x["lon"] - s / 2, x["lat"] - s / 2, x["lon"] + s / 2, x["lat"] + s / 2
                ),
                axis=1,
            ),
        )
        return gdf

    def run(self) -> None:
        """Executes the full workflow to generate and save the base grid."""
        try:
            fp = self.storage.build_filepath("output", "base_grid_prio")
            self.storage.check_exists(fp)
            # no need to update this after regeneration since there is always only 1 instance of the GlobalBaseGrid
            if self.regenerate:
                raise FileNotFoundError
            console.print("Base grid already exists. No need to re-run generation.")
        except FileNotFoundError:
            base_grid = self.create_grid()
            countries, land, lakes = self.load_filter_data()
            filtered_grid = self.filter_and_match_grid(base_grid, countries, land, lakes)
            filtered_grid = filtered_grid.reset_index()
            filtered_grid["pgid"] = filtered_grid.apply(
                lambda x: coords_to_pgid(x.lat, x.lon), axis=1
            )
            filtered_grid = filtered_grid.set_index("pgid")
            self.storage.save(
                filtered_grid.drop(columns="geometry"), filename="base_grid_prio", mode="output"
            )


class Indicator(ABC):
    """Abstract base class for computing an indicator within the CCVI.

    Provides common initialization logic for setting up identifiers, configuration,
    storage access, and a reference to the base grid. Defines the standard workflow
    methods that subclasses must implement:
    - loading data,
    - preprocessing,
    - computing the indicator,
    - normalizing values,
    Provides a method to create the indicator data structure and defines an universal
    workflow to run an indicator.

    Conventions for indicator implementations:
    - Subclasses always invoke super().__init__()
    - Indicators should set their ID components as default values for their __init__ method.
    - Indicators should set an instance of any data source classes they require as attribute
      during initialization.
    - The code included directly in an indicator class should be sufficient to understand
      the basic logic of generating the indicator.

    Attributes:
        console (Console): Shared console instance for output.
        requires_processing_storage (bool): Flag indicating whether the indicator needs processing
            storage. If False, no processing folder will be created.
        pillar (str): The pillar ID of the indicator.
        dim (str): The dimension ID of the indicator.
        id (str): The ID of the indicator.
        config (ConfigParser): The ConfigParser instance used for initialization.
        global_config (dict[str, Any]): Global configuration settings dictionary.
        indicator_config (dict[str, Any] | None): Indicator-specific configuration dictionary.
            None if no config is set.
        storage (StorageManager): Instance managing storage for this indicator.
        composite_id (str): Composite identifier ("pillar_dim_id") for this indicator.
        grid (GlobalBaseGrid): An initialized GlobalBaseGrid instance.
        generated (bool): Flag indicating if the indicator output file already exists
            and is up-to-date for the last quarter, generated through a check at init.
        regenerate (dict[str, bool]): Dictionary with regenerate settings for
            "indicator" and "preprocessing" based on the global config.
    """

    console: Console = console
    requires_processing_storage: bool = False

    def __init__(self, pillar: str, dim: str, id: str, config: ConfigParser, grid: GlobalBaseGrid):
        """Initializes the Indicator instance.

        Sets up identifiers, retrieves configurations, initializes the
        StorageManager, sets the composite ID, and stores references to the
        config parser and grid.

        Args:
            pillar (str): The pillar ID for the indicator.
            dim (str): The dimension ID for the indicator.
            id (str): The specific ID for the indicator.
            config (ConfigParser): An initialized ConfigParser instance.
            grid (GlobalBaseGrid): An initialized GlobalBaseGrid instance.
        """
        self.pillar = pillar
        self.dim = dim
        self.id = id

        # store config parser instance to pass along and set indicator-specific and global config
        self.config = config
        self.global_config = config.get_global_config()
        # typed to dict even though it can be None to prevent type checking errors
        self.indicator_config: dict[str, Any] = config.get_indicator_config(pillar, dim, id)  # type: ignore

        # storage functionality, which also takes care of the indicator ID
        self.storage = StorageManager(
            storage_base_path=self.global_config["storage_path"],
            requires_processing_storage=self.requires_processing_storage,
        )
        self.storage.set_composite_id(pillar, dim, id, component_type="indicator")
        self.composite_id = self.storage.composite_id
        self.generated = self.storage.check_component_generated()

        # add regenerate config
        regenerate_config = config.get_regeneration_config(self.composite_id)
        self.regenerate = {key: regenerate_config[key] for key in ["indicator", "preprocessing"]}
        # this allows acces to load the grid for data structures
        self.grid = grid

        # print some information
        self.console.print(
            f"Indicator {self.composite_id} initialized with \ngenerated: {self.generated} \nconfig: {self.indicator_config}"
        )

    def create_base_df(self, start_year: int | None = None) -> pd.DataFrame:
        """Creates a base DataFrame structured by grid cell, year, and quarter.

        Args:
            start_year (int | None, optional): The first year to include. If None,
                defaults to 'start_year' from the global configuration.

        Returns:
            pd.DataFrame: A DataFrame indexed by 'pgid' and time periods,
                containing columns from the base grid ('lat', 'lon', 'iso3').
        """

        if start_year is None:
            year_min = self.global_config["start_year"]
        else:
            year_min = start_year

        last_quarter = get_quarter("last")

        df_grid = self.grid.load()
        df = create_custom_data_structure(df_grid, year_min, last_quarter.year)
        # creating a datetime column for easier time cropping and time-based operations
        df = add_time(df)
        df = df.loc[df.time <= last_quarter]
        return df.sort_index()

    @abstractmethod
    def load_data(self, *args, **kwargs) -> Any:
        """Load the raw data needed to compute the indicator.

        Position in `self.run()` unless overwritten:
            - Return value is passed to self.preprocess_data().
        """
        pass

    @abstractmethod
    def preprocess_data(self, *args, **kwargs) -> Any:
        """Preprocess the loaded data for indicator computation.

        Position in `self.run()` unless overwritten:
            - Takes the return value from `self.load_data()` as argument.
            - Return value is passed to `self.preprocess_data()`.
        """
        pass

    @abstractmethod
    def create_indicator(self, *args, **kwargs) -> Any:
        """Implement the indicator formula based on preprocessed data.

        Position in `self.run()` unless overwritten:
            - Takes the return value from `self.preprocess_data()` as argument.
            - Return value is passed to `self.normalize()` and `self.add_raw_value()` if
              implemented.
        """
        pass

    @abstractmethod
    def normalize(self, *args, **kwargs) -> pd.DataFrame:
        """Normalize the computed indicator data to the CCVI range 0-1. Includes
        any transformations applied. Should return a dataframe with only the
        columns to be saved later.

        Position in `self.run()` unless overwritten:
            - Takes the return value from `self.preprocess_data()` as argument.
        """
        pass

    def add_raw_value(self, df_indicator: pd.DataFrame, df_preprocessed: Any) -> pd.DataFrame:
        """Adds the raw (un-normalized) indicator value as a separate column.

        Optional step. Subclasses can override this to extract the relevant raw
        value from the preprocessed data and add it to the final DataFrame with
        a '_raw' suffix. They may also add a raw value during a different step
        or not at all.

        Placeholder implementation returns the DataFrame unchanged.

        Args:
            df_indicator (pd.DataFrame): DataFrame returned by `normalize()`.
            df_preprocessed (Any): Return value from `preprocess_data()`. Usually
                a DataFrame.

        Returns:
            pd.DataFrame: The `df_indicator` DataFrame, potentially with an
                added raw value column (e.g., 'pillar_dim_id_raw').
        """
        return df_indicator

    def run(self) -> None:
        """Executes the standard indicator generation workflow.

        Checks `self.generated` flag; if True and the global config setting
        `regenerate_indicators` is False, skips execution. Otherwise, calls
        `load_data`, `preprocess_data`, `create_indicator`, `normalize`, and
        `add_raw_value` sequentially. Performs sanity checks on the final DataFrame
        structure and saves the result using `self.storage`. Sets `self.generated`
        to True upon success.
        """
        self.console.print(f'Generating indicator "{self.composite_id}"...')
        if self.generated and not self.regenerate["indicator"]:
            self.console.print(
                f'Indicator "{self.composite_id}" already generated for last quarter, no need to rerun. If regeneration is desired, please add {self.composite_id} to regenerate:preprocessing in the global config.'
            )
        else:
            self.console.print("Loading data...")
            df_data = self.load_data()
            self.console.print("Preprocessing components...")
            df_preprocessed = self.preprocess_data(df_data)
            self.console.print("Indicator calculation and normalization...")
            df_indicator = self.create_indicator(df_preprocessed)
            df_indicator = self.normalize(df_indicator)
            df_indicator = self.add_raw_value(df_indicator, df_preprocessed)
            # some sanity checks
            # after normalization, the indicator df should only have the indicator column and possibly a raw column
            assert all(self.composite_id in c for c in df_indicator.columns), (
                "DataFrame df_indicator should only contain columns named after the composite_id."
            )
            assert list(df_indicator.index.names) == ["pgid", "year", "quarter"], (
                "DataFrame df_indicator should have a ['pgid', 'year', 'quarter'] Multiindex."
            )
            assert (
                df_indicator.index.get_level_values("year").min()
                == self.global_config["start_year"]
            ), "DataFrame df_indicator may not contain values before the defined start_year."
            self.storage.save(df_indicator)
            # rerun generated check
            self.generated = self.storage.check_component_generated()
            assert self.generated, (
                "Indicator does not exist in storage or does not include data up to the last quarter, please doublecheck the implementation."
            )
            self.console.print(f'Indicator "{self.composite_id}" generated successfully.')
            # set regenerate to false to make sure it does not get regenerated twice in one session:
            # for the instance:
            self.regenerate["indicator"] = False
            self.regenerate["preprocessing"] = False
            # for new instances in the same session - modify all_config from config:
            self.config.set_regenerated_globally(self.composite_id)
        return


class AggregateScore:
    """Base class for aggregate scores.

    Implements  `load_components()`, `aggregate()` and `get_data_recency()`
    methods applicable to all aggregate scores.

    Attributes:
        components (list): List of components, required to be specified by subclasses.
        console (Console): Shared console instance for output.
        requires_processing_storage (bool): Flag indicating whether the aggregate needs processing
            storage. Set to False.
    """

    components: list
    console: Console = console
    requires_processing_storage: bool = False

    def load_components(self, load_additional_values: bool = False, **load_kwargs) -> pd.DataFrame:
        """Loads the components associated with this dimension.

        Iterates through `self.components`, loads their respective output files
        using their `storage.load()` method, and concatenates them into a single
        DataFrame.

        Args:
            load_additional_values (bool): Flag whether to load the component by ID
                only or include any additional (raw) values stored. Defaults to False.
            load_kwargs: **kwargs for storage.load()

        Returns:
            pd.DataFrame: A combined DataFrame with all (direct) components of the
                aggregate score depending on the boolean flag (i.e. Indicators
                for Dimensions, Dimension scores for Pillars by default).
        """
        dfs = []
        for component in self.components:
            df = component.storage.load(**load_kwargs)
            if load_additional_values:
                dfs.append(df)
            else:
                dfs.append(df[[component.composite_id]])
        df = pd.concat(dfs, axis=1)
        return df

    def aggregate(
        self, df: pd.DataFrame, id: str, aggregation_config: dict[str, Any]
    ) -> pd.DataFrame:
        """Aggregates component scores into aggregate score based on a provided configuration.

        This method applies a specified aggregation technique to a DataFrame of
        component scores. It uses the method, weights, and normalization settings
        defined in the `aggregation_config` dictionary. Handles potential
        '_exposure' suffixes in column names. NAs are ignored for dimensions,
        but not allowed for Pillars, Risk scores, or the CCVI score.

        Args:
            df (pd.DataFrame): DataFrame containing ONLY the component scores to 
                be aggregated. Columns should correspond to component IDs 
                ('_exposure' suffixes are also allowed).
            id (str): The composite ID of the aggregate score.
            aggregation_config (dict[str, Any]): A dictionary specifying the
                aggregation parameters, containing 'method' (str),
                'normalize' (bool), and 'weights' (a dictionary mapping
                component IDs to weights, or the string "None").

        Returns:
            pd.DataFrame: A DataFrame with a single column containing the aggregate
                scores.
        """        
        # output dataframe
        df_aggregated = pd.DataFrame(index=df.index, columns=[id])
        # rename if columns are exposure versions to make the logic below consistent
        indicators = [c.removesuffix("_exposure") for c in df.columns]
        df.columns = indicators
        # NAs are ignored in dimension aggregation, but not allowed for higher-level aggregates
        if len(id) > 4 and not id.endswith("_risk"):
            ignore_nan = True
        else:
            ignore_nan = False
        # configs
        method = aggregation_config["method"]
        normalize = aggregation_config["normalize"]
        weights = aggregation_config["weights"]
        if weights != "None":
            assert all(col in weights.keys() for col in indicators)
            # put weights are in the same order as the columns
            weights = [weights[col] for col in indicators]
        else:
            weights = None

        df_aggregated[id] = self._calculate_aggregate_score(df, method, ignore_nan, weights)
        # rescale to 0-1 if desired
        if normalize:
            df_aggregated[id] = min_max_scaling(df_aggregated[id])

        return df_aggregated

    def _calculate_aggregate_score(
        self,
        df: pd.DataFrame,
        method: Literal["mean", "pmean", "gmean"],
        ignore_nan: bool,
        weights: list[float] | None = None,
    ) -> list | np.ndarray:
        """Calculates an aggregate score from component data.

        This method implements various aggregation techniques:
        - "mean": Arithmetic mean
        - "gmean": Geometric mean
        - "pmean": Quadratic mean
        It applies the chosen method row-wise to the input DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with columns to aggregate.
            method (Literal["mean", "pmean", "gmean", "conflict_pillar"]):
                The aggregation method to use.
            ignore_nan (bool, optional): If True, NaNs are omitted from calculations.
                Defaults to False.
            weights (list[float] | None, optional): A list of weights corresponding
                to the columns in `df`. If None, equal weights are assumed.
                Defaults to None.

        Returns:
            list[float] | np.ndarray: A list or array containing the aggregate score
                for each row.
        """

        if weights is not None:
            assert len(weights) == df.shape[1]
        if ignore_nan:
            nan_policy = "omit"
        else:
            nan_policy = "propagate"
        if method == "mean":
            if ignore_nan:
                # mean does not have weights and np.average can only skip nas with masked arrays
                temp_array = np.ma.masked_array(df.values, np.isnan(df.values))
                means = np.ma.average(temp_array, axis=1, weights=weights).data
                # masked average returns 0 in all-na cases -> we dont want that so lets replace these with na again based on the mask
                out = np.where(temp_array.mask.all(axis=1), np.nan, means)
                return out
            else:
                return np.average(df.values, axis=1, weights=weights)
        elif method == "gmean":
            return gmean(df, axis=1, nan_policy=nan_policy, weights=weights)  # type: ignore
        elif method == "pmean":
            return pmean(df, 2, axis=1, nan_policy=nan_policy, weights=weights)  # type: ignore
        elif method == "conflict_pillar":
            return calculate_score_conflict(df, ignore_nan=ignore_nan)
        else:
            raise ValueError(
                f'Argument "method" needs to be one of ["mean", "pmean", "gmean"], got {method}.'
            )

    def get_data_recency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Determines the last updated quarter for each input component.

        Finds the latest non-NaN quarter for each indicator/dimension column in the
        input DataFrame and returns the results as dictionary, mapping IDs to
        'YYYY-Q' strings.

        Args:
            df (pd.DataFrame): The DataFrame containing the loaded indicator data.
        """
        ids = []
        last_updated_time = []
        # Loop through columns (indicators) to find the last non-imputed quarter
        for col in df.columns:
            non_imputed_dt = df[col].dropna().reset_index().sort_values(["year", "quarter"])
            if not non_imputed_dt.empty:
                last_record = non_imputed_dt.iloc[-1]
                last_updated = f"{int(last_record['year'])}-{int(last_record['quarter'])}"
                ids.append(col)
                last_updated_time.append(last_updated)
            else:
                raise ValueError(f"Column {col} in passed df is empty!")

        df_recency = pd.DataFrame({"id": ids, "lastUpdated": last_updated_time})
        return df_recency


class Dimension(AggregateScore):
    """Orchestrates calculation of a dimension score, aggregating multiple indicators.

    Provides a standardized implementation for combining indicator scores based
    on configuration settings. It handles loading required indicator data,
    validating inputs, filling missing time series data with the last know value,
    optionally applying exposure, performing the aggregation, and saving the results.

    Attributes:
        console (Console): Shared console instance for output.
        pillar (str): The pillar ID of the dimension.
        dim (str): The ID of the dimension.
        components (List[Indicator]): List of initialized Indicator instances
            that feed into this dimension.
        has_exposure (bool): Flag indicating if an implementation of `add_exposure()`
            exists and should be called.
        config (ConfigParser): The ConfigParser instance used for initialization.
        global_config (dict[str, Any]): Global configuration settings dictionary.
        aggregation_config (Dict[str, Any]): Dimension-specific aggregation config.
            Loaded via the ConfigParser.
        storage (StorageManager): Instance managing storage for this dimension.
        composite_id (str): Composite identifier ("pillar_dim") for this dimension.
        generated (bool): Flag indicating if the output file has been generated
            with this instance of Dimension. Always set to False during init.
        requires_processing_storage (bool): Flag indicating whether the aggregate
            needs processing storage. Set to False.
    """

    def __init__(
        self,
        pillar: str,
        dim: str,
        indicators: list[Indicator],
        config: ConfigParser,
        has_exposure: bool = False,
    ):
        """Initializes the Dimension instance.

        Sets identifiers, retrieves configurations, initializes the
        StorageManager (setting composite_id and processing subfolder), checks
        generation status, stores indicators, and sets the exposure flag.

        Args:
            pillar (str): The pillar ID for the dimension.
            dim (str): The ID for the dimension.
            indicators (list[Indicator]): List of pre-initialized Indicator objects.
            config (ConfigParser): An initialized ConfigParser instance.
            has_exposure (bool, optional): Flag indicating if this dimension
                handles exposure variants of indicators. Defaults to False.
        """
        self.pillar = pillar
        self.dim = dim
        self.components: list[Indicator] = indicators
        self.has_exposure = has_exposure
        # load config
        self.config = config
        self.global_config = config.get_global_config()
        self.aggregation_config = config.get_aggregation_config(pillar, dim)
        # setup storage and ID
        self.storage = StorageManager(
            storage_base_path=self.global_config["storage_path"],
            requires_processing_storage=self.requires_processing_storage,
        )
        self.storage.set_composite_id(pillar, dim, component_type="dimension")
        self.composite_id = self.storage.composite_id
        self.generated = False

        self.console.print(
            f"Dimension class for {self.composite_id} initialized with \nhas_exposure: {self.has_exposure} \nconfig: {self.aggregation_config}"
        )

    def add_exposure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies exposure logic if `self.has_exposure` is True.

        If exposure handling is needed for a specific dimension (indicated by
        `has_exposure=True` during initialization), this method will be called
        during `run()` and should be overridden in a subclass, otherwise raises
        NotImplementedError.

        Args:
            df (pd.DataFrame): DataFrame containing the loaded indicator data.

        Returns:
            pd.DataFrame: DataFrame containing only the indicator versions with
                exposure. Columns should be named "composite_id_exposure".
        """
        raise NotImplementedError

    def validate_indicator_input(
        self,
        indicators: list[Indicator],
    ):
        """Validate the input list of indicators.

        Ensures that each indicator belongs to the dimension based on its identfier, checks
        for duplicate indicators, and verifies that each indicator has been generated.
        Runs any indicators that have not yet been generated.

        Args:
            indicators (list[Indicator]): List of indicator subclasses which are
                part of the respective dimension.
        """
        if not isinstance(indicators, list) or not all(
            isinstance(i, Indicator) for i in indicators
        ):
            raise TypeError("'indicators' must be a list of Indicator instances.")
        indicator_ids: list[str] = [i.composite_id for i in indicators]
        assert all([id.startswith(self.composite_id) for id in indicator_ids])
        assert len(indicator_ids) == len(set(indicator_ids))
        for i in indicators:
            if i.regenerate["indicator"]:
                i.run()
            else:
                try:
                    assert i.generated
                    self.console.print(f'Indicator "{i.composite_id}" already generated.')
                except AssertionError:
                    i.run()

    def run(self, skip_run: bool = False) -> None:
        """Executes the full workflow for calculating and saving the dimension score.

        Validates input indicators (running them if needed), loads indicator data,
        checks generation status (skipping if `regenerate` is False and already
        generated during this run), imputes missing time series data using
        forward fill, optionally applies exposure logic, aggregates the indicators,
        performs sanity checks, and saves the final dimension score.

        Args:
            skip_run (bool, optional): Whether to skip the dimension run. Can be set
                to True for testing. Defaults to False, always regenerating the
                aggregation. WARNING: no checks are performed in this case, only
                works if a corresponding output file is in storage.
        """
        self.console.print(f'Generating dimension "{self.composite_id}"...')
        self.validate_indicator_input(self.components)
        if self.generated:
            self.console.print(
                f'Dimension "{self.composite_id}" already generated with this Dimension instance, no need to rerun.'
            )
        elif skip_run:
            self.console.print(
                f'Skipping dimension "{self.composite_id}" run. Only works if the output parquet already exists in storage. Only intended for testing as no checks are performed.'
            )
        else:
            self.console.print("Loading indicators...")
            df = self.load_components()
            # fill missing data with the last available observation
            imputer = PanelImputer(
                time_index=["year", "quarter"],
                location_index="pgid",
                imputation_method="ffill",
                parallelize=True,
            )
            df: pd.DataFrame = imputer.fit_transform(df)  # type: ignore
            if self.has_exposure:
                self.console.print("Adding exposure...")
                df = self.add_exposure(df)
            self.console.print("Calculating aggregate score...")
            df_aggregated = self.aggregate(df, self.composite_id, self.aggregation_config)
            # some sanity checks
            # after normlization, the indicator df should only have the indicator column
            assert list(df_aggregated.columns) == [self.composite_id], (
                "df_aggregated should only contain one column named after the composite_id"
            )
            # after normalization
            assert list(df_aggregated.index.names) == ["pgid", "year", "quarter"], (
                "df_aggregated should have a ['pgid', 'year', 'quarter'] Multiindex"
            )
            self.storage.save(df_aggregated)
            self.generated = True
            self.console.print(
                f'Dimension "{self.composite_id}" score successfull calculated and saved.'
            )
        return


class Pillar(AggregateScore):
    """Orchestrates calculation of a pillar score, aggregating multiple dimensions.

    Provides a standardized implementation for combining dimension scores based
    on configuration settings. It handles loading required dimension data,
    validating inputs, performing the aggregation, and saving the results.

    Modified version of the Dimension class.

    Attributes:
        console (Console): Shared console instance for output.
        pillar (str): The pillar ID of the Pillar.
        components (List[Dimension]): List of initialized Dimension instances
            that feed into this pillar.
        config (ConfigParser): The ConfigParser instance used for initialization.
        global_config (Dict[str, Any]): Global configuration settings dictionary.
        aggregation_config (Dict[str, Any]): Pillar-specific aggregation config.
            Loaded via the ConfigParser.
        storage (StorageManager): Instance managing storage for this pillar.
        composite_id (str): Composite identifier ("pillar") for this pillar.
        generated (bool): Flag indicating if the output file has been generated
            with this instance of Dimension. Always set to False during init.
        requires_processing_storage (bool): Flag indicating whether the aggregate
            needs processing storage. Set to False.
    """

    def __init__(
        self,
        pillar: str,
        dimensions: list[Dimension],
        config: ConfigParser,
    ):
        """Initializes the Pillar instance.

        Sets identifiers, retrieves configurations, initializes the
        StorageManager (setting composite_id and processing subfolder), checks
        generation status, stores indicators, and sets the exposure flag.

        Args:
            pillar (str): The pillar ID for the dimension.
            dimensions (list[Indicator]): List of pre-initialized Indicator objects.
            config (ConfigParser): An initialized ConfigParser instance.
        """
        self.pillar = pillar
        self.components = dimensions
        # load config
        self.config = config
        self.global_config = config.get_global_config()
        self.aggregation_config = config.get_aggregation_config(pillar)
        # setup storage and ID
        self.storage = StorageManager(
            storage_base_path=self.global_config["storage_path"],
            requires_processing_storage=self.requires_processing_storage,
        )
        self.storage.set_composite_id(pillar, component_type="pillar")
        # same as self.pillar, kept for consistency
        self.composite_id = self.storage.composite_id
        self.generated = False

        self.console.print(
            f"Pillar class for {self.composite_id} initialized with \nconfig: {self.aggregation_config}"
        )

    def validate_dimension_input(self, dimensions: list[Dimension], skip_run: bool):
        """Validate the input list of dimensions.

        Ensures that each dimension belongs to the pillar based on its identfier,
        and checks for duplicate dimensions. Calls run on all dimensions that
        have not yet been generated with the respective instance.

        Args:
            dimensions (list[Dimension]): List of Dimension instances which are
                part of the respective dimension.
            skip_run (bool): Flag whether to skip aggregate score runs passed
                from `run()` along to underlying dimensions.
                WARNING: may break stuff!
        """
        if not isinstance(dimensions, list) or not all(
            isinstance(d, Dimension) for d in dimensions
        ):
            raise TypeError("'dimensions' must be a list of Dimension instances.")
        dimension_ids: list[str] = [d.composite_id for d in dimensions]
        assert all([id.startswith(self.composite_id) for id in dimension_ids])
        assert len(dimension_ids) == len(set(dimension_ids))
        for d in dimensions:
            try:
                assert d.generated
                self.console.print(
                    f'Dimension "{d.composite_id}" already generated with this instance.'
                )
            except AssertionError:
                d.run(skip_run)

    def run(self, skip_run: bool = False) -> None:
        """Executes the full workflow for calculating and saving the dimension score.

        Validates input indicators (running them if needed), loads indicator data,
        checks generation status (skipping if `regenerate` is False and already
        generated), aggregates the indicators, performs sanity checks, and saves
        the final pillar score.

        Args:
            skip_run (bool, optional): Whether to skip the pillar run. Can be set
                to True for testing. Defaults to False, always regenerating the
                aggregation. Passed along to underlying dimensions.
                WARNING: no checks are performed in this case, only works if a
                corresponding output file is in storage. May break stuff!
        """

        self.console.print(f'Generating pillar "{self.composite_id}"...')
        self.validate_dimension_input(self.components, skip_run)
        if self.generated:
            self.console.print(
                f'Pillar "{self.composite_id}" already generated with this Pillar instance, no need to rerun.'
            )
        elif skip_run:
            self.console.print(
                f'Skipping pillar "{self.composite_id}" run. Only works if the output parquet already exists in storage. Only intended for testing as no checks are performed.'
            )
        else:
            self.console.print("Loading dimension scores...")
            df = self.load_components()
            self.console.print("Calculating aggregate score...")
            df_aggregated = self.aggregate(df, self.composite_id, self.aggregation_config)
            # some sanity checks
            # after normlization, the indicator df should only have the indicator column
            assert list(df_aggregated.columns) == [self.composite_id], (
                "df_aggregated should only contain one column named after the composite_id"
            )
            # after normalization
            assert list(df_aggregated.index.names) == ["pgid", "year", "quarter"], (
                "df_aggregated should have a ['pgid', 'year', 'quarter'] Multiindex"
            )
            self.storage.save(df_aggregated)
            self.generated = True
            self.console.print(
                f'Pillar "{self.composite_id}" score successfull calculated and saved.'
            )
        return


class Dataset(ABC):
    """Abstract base class for accessing data sources stored locally.

    Provides a common logic for initializing data sources based on
    configuration, setting up storage access, and defining an interface
    for loading data. Subclasses must define `data_key` and implement
    the `load_data` method.

    Attributes:
        console (Console): Shared console instance for output.
        data_key (str): Class attribute identifying the specific key for this
            data source within the 'data' section of the configuration file.
            Must be defined by subclasses.
        local (bool): Flag indicating if the data source is expected to be loaded
            from the local input folder. Defaults to True.
        needs_storage (bool): Flag indicating whether the dataset needs processing
            storage. If False, no processing folder will be created. Defaults to
            True.
        config (ConfigParser): The ConfigParser instance used for initialization.
        global_config (dict[str, Any]): Dictionary containing global settings.
        storage (StorageManager): Storage manager instance configured with
            the global storage path.
        regenerate (dict[str, bool]): Dictionary with regenerate settings for
            "data" and "preprocessing" based on the global config.
        data_config (dict[str, str]): Dictionary mapping the `data_key` to its
            configured file path. Only set if local=True.
    """

    console: Console = console
    data_key: str
    local: bool = True
    needs_storage: bool = True

    def __init__(self, config: ConfigParser):
        """Initializes the local data source instance.

        Retrieves global and data-specific configuration, sets up the
        StorageManager, and checks for the existence of the local data file if
        `self.local` is True. Subclasses can set a `data_keys (list[str])` attribute if
        multiple local input files are required.

        Args:
            config (ConfigParser): An initialized ConfigParser instance.
        """
        self.config = config
        self.global_config = config.get_global_config()
        # add regenerate config
        regenerate_config = config.get_regeneration_config(self.data_key)
        self.regenerate = {key: regenerate_config[key] for key in ["data", "preprocessing"]}
        if self.needs_storage:
            self.storage = StorageManager(
                storage_base_path=self.global_config["storage_path"],
                requires_processing_storage=True,
                processing_folder=self.data_key,
            )
        else:
            # avoid unnecessary warnings
            self.storage = StorageManager(
                storage_base_path=self.global_config["storage_path"],
                requires_processing_storage=False,
            )
        if self.local:
            if hasattr(self, "data_keys"):
                data_keys = getattr(self, "data_keys")
            else:
                data_keys = [self.data_key]
            self.data_config = config.get_data_config(data_keys)
            for key in self.data_config:
                self.storage.check_exists(self.data_config[key])

    @abstractmethod
    def load_data(self, *args, **kwargs) -> Any:
        """Abstract method for loading the specific data source.

        Subclasses must implement this method to read their corresponding
        data file(s) and return a pandas DataFrame.

        Returns:
            Any: The loaded data.
        """
        pass


# this is possible by adding the root folder as the PYTHONPATH var in .env
if __name__ == "__main__":
    grid = GlobalBaseGrid(ConfigParser())
    grid.run()
