# ccvi-data
Repository for data processing behind the Climate—Conflict—Vulnerability Index (CCVI). https://climate-conflict.org

## License

This project is licensed under the GNU General Public License v3.0.  
See the [LICENSE](LICENSE) file for details.

## Overview

The repository is broadly structured along the pillars of the CCVI, with corresponding `climate`, `conflict` and `vulnerability` folders containing respective indicators. Additionally, the `base` folder contains base classes providing the implementation structure, while the `utils` folder contains functionality shared across indicators and dimensions. Lastly, the `data` folder (as defined by the `config.yaml`) used to store input data required to build the indicators, save processing steps and the finished indicators and aggregated scores. The main `ccvi.py` script defines the composition of the aggrgate components (dimensions, pillars, risk scores) and provides a unified interface for users to run individual indicators, dimensions, or the whole ccvi data pipeline. Available configuration can be set in the `config.yaml`.

## Missing Functionality (25.04.2025)

*   The regeneration logic is not yet implemented for data processing.
*   The index versioning (year-quarter) and final output logic is not yet defined. The plan is that the current storage always reflects the latest versions of individual indicators/dimensions and there will be combined versions in separate folders simliar to now.
*   Implementation of aggregation so far only up to Dimension level.

## Core Concepts & Architecture

The chosen architecture breaks down the index calculation into distinct components: data loading, indicator calculation, and dimension aggregation, all managed through a central configuration system and standardized storage mechanism.

It is implemented using the following base objects and concepts:

1.  **Configuration (`ConfigParser`, `config.yaml`):**
    *   All settings are centralized in a `config.yaml` file located in the project root.
    *   The `base.objects.ConfigParser` class is responsible for loading and parsing this YAML file.
    *   It provides methods (`get_global_config()`, `get_indicator_config()`, `get_data_config()`, `get_aggregation_config()`) to retrieve specific configuration dictionaries.
    *   **Usage:** Typically, a single `ConfigParser` instance is created in the main `ccvi.py` script and **passed via dependency injection** to other components (`Indicator`, `Dimension`, `Dataset`, `GlobalBaseGrid`) and subclasses that require access to the config.

2.  **Storage (`StorageManager`):**
    *   The `base.objects.StorageManager` class provides a unified interface for persistent data.
    *   It automatically creates standard subdirectories (`input/`, `processing/`, `output/`) based on the `storage_path` defined in the global configuration. If a `processing_folder` argument is provided or `set_composite_id()` is called, uses a subfolder for processing storage.
    *   It provides methods for saving (`save()`) and loading (`load()`) pandas DataFrames as Parquet files within the `processing` or `output` directories.
    *   Can build file paths (`build_filepath()`), check for file existence (`check_exists()`), and check if components (indicators, dimensions) have already been generated for a relevant quarter (`check_component_generated()`).
    *   Each `Indicator` or `Dimension` instance creates its **own** `StorageManager` instance, initialized with the base storage path from the global config.
    *   The `StorageManager` used the `composite_id` (e.g., `pillar_dim_id`) of a given component as the default filename for saving/loading operations.

3.  **Base Grid (`GlobalBaseGrid`):**
    *   The `base.grid.GlobalBaseGrid` class is responsible for creating or loading the geospatial 0.5°x0.5° grid.
    *   It applies land masks and handles assigning grid cells to country boundaries.
    *   **Usage:** Typically, a single `GlobalBaseGrid` instance is created at the start of the process (using the `ConfigParser`) and **passed via dependency injection** to `Indicator` instances for creation of the base data structure.

4.  **Data Loading (`Dataset`, Subclasses):**
    *   The `base.data.Dataset` class is an abstract base class for accessing data sources.
    *   Concrete subclasses (e.g., `VDemData`, `ACLEDData`) implement the logic for specific datasets.
    *   Each subclass defines a unique `data_key` class attribute, which corresponds to an entry in the `config.yaml['data']` section.
    *   The `config.yaml['data']` entry for the `data_key` specifies the correspinding filename in the `input/` directory.
    *   Subclasses may utilize the `StorageManager` to cache intermediate results in the `processing/` directory.
    *   The `local` class attribute (defaulting to `True`) indicates if the source data is expected in the local `input/` folder.
    *   A subclass can optionally define a `data_keys` class attribute (list[str]) if it requires multiple input files.
    *   **Initialization:** Receives the shared `ConfigParser` instance
        *   Intializes its own `StorageManager` and loads its config (i.e. the data file path) during initialization.
        *   Datasets use a `data_key` subfolder for processing storage.
        *   Checks for existences of the required data file(s).
    *   **Workflow:** Subclasses must implement abstract methods:
        *  `load_data()`: Read the data, either from local storage or through an API call.
    *   Subclasses generally implement further preprocessing steps (e.g. `preprocess_data()`)

5.  **Indicators (`Indicator`):**
    *   The `base.objects.Indicator` class is an abstract base class defining the standard interface and workflow for all indicators.
    *   The class implements the `create_base_df()` method to create the CCVI's base data structure based on the `GlobalBaseGrid`.
    *   **Initialization:** Receives `pillar`, `dim`, `id` identifiers, the shared `ConfigParser` instance, and the shared `GlobalBaseGrid` instance. 
        *   Intializes its own `StorageManager` and loads its config during initialization.
        *   Checks if the indicator output already exists and is up-to-date using `self.storage.check_component_generated()`, setting the `self.generated` flag.
        *   Since they call `self.storage.set_composite_id()`, indicators use a `pillar_dim_id` subfolder for processing storage.
        *   Indicators typically create instances of the Dataset classes they depend on and store them as class attributes for easy access during initialization.
    *   **Workflow:**:
        *   `load_data()`: Load necessary raw data (typically using `Dataset` subclasses).  **Abstract method, MUST be implemented by subclasses.**
        *   `preprocess_data()`: Clean, filter, and prepare data. Create the grid-quarter data structure.  **Abstract method, MUST be implemented by subclasses.**
        *   `create_indicator()`: Apply the core indicator formula. **Abstract method, MUST be implemented by subclasses.**
        *   `normalize()`: Normalize the indicator values to 0-1 range including any prior transformations. **Abstract method, MUST be implemented by subclasses.**
        *   `add_raw_value()`: Add a raw value to the data. _Optional method, MAY be implemented by subclasses._
        *   `run()`: Orchestrates the workflow. Checks `self.generated` and `regenerate` flag, potentially skipping generation. Calls the steps sequentially, performs sanity checks, saves the final output using `self.storage.save()`, and sets `self.generated = True`.

6.  **Dimensions (`Dimension`):**
    *   The `base.objects.Dimension` class provides a **standardized implementation** for aggregating multiple indicators into a dimension score.
    *   **Initialization:** Receives `pillar`, `dim` identifiers, a *list* of already initialized `Indicator` instances belonging to it, and the shared `ConfigParser` instance.
        *   Intializes its own `StorageManager` and loads its config during initialization.
        *   Checks if the dimension output already exists and is up-to-date using `self.storage.check_component_generated()`, setting the `self.generated` flag.
    *   **Workflow:** 
        *   `validate_indicator_input()`: Checks if provided indicators match the dimension and runs them if they haven't been generated or if `regenerate` is True.
        *   `load_indicators()`: Loads the output data for all associated indicators.
        *   `set_data_recency()`: Determines the latest available data point for each input indicator.
        *   `add_exposure()`: Exposure logic for climate indicators (to be implmemented, currently raises `NotImplementedError`).
        *   `aggregate()`: Performs the aggregation based on configuration (method, weights, normalization).
        *   `run()`: Orchestrates the workflow. Checks `self.generated` and `regenerate` flag, potentially skipping generation but always setting data_recency. Calls the steps sequentially, forward filling missing data, performs sanity checks, saves the final output using `self.storage.save()`, and sets `self.generated = True`.

7.  **Mixins:**
    *   Used to share functionality across related indicators or dimensions within a pillar (e.g. `conflict.shared.NormalizationMixin` provides a standardized normalization procedure for conflict indicators).

8.  **Regeneration of existing steps:**
    * By default, the data processing checks for the existence previously generated data and does not repeat already finished steps for each indicator.
    * Aggregate scores are always re-generated, since they do not require much time.
    * There are three boolean regeneration flags accessible depending on the component, specified in the global config settings:
        * `regenerate_indicator`: Whether to regenerate the given indicator even if there is already a version for the latest quarter.
        * `regenerate_preprocessing`: Whether to regenerate the data preprocessing for an indicator even if there is already a version for the latest quarter.
        * `regenerate_data`: Whether to regenerate (i.e. re-download) the underlying source data of an indicator, even if there is already a version for the latest quarter.

9.  **Logging**
    * Progress messages and any other output to inform users during component generation should be handled via the shared `rich.console.Console` instance defined in `base.objects`. 
    * Logging is available as class attribute in the `Indicator`, `Dimension` and `Dataset` classes via `self.console`.


## Directory Structure
```
.
├─ base/
│ ├─ init.py
│ ├─ objects.py # Indicator, Dimension, ConfigParser, StorageManager, GlobalBaseGrid
│ ├─ data.py # LocalData, Individual Data sources...
├─ conflict/ # Root for pillar-specifc implementations
│ ├─ level/ # Dimension
│ │ ├── intensity.py # ConLevelIntensity
│ │ └── surrounding.py # ConLevelSurrounding
│ ├─ persistence/ # Dimension
│ │ └── ...
│ ├─ soctens/ # Dimension
│ │ └── ...
│ └─ shared.py
├─ climate/ # Root for pillar-specifc implementations
│ ├─ current/ # Dimension
│ │ ├── drought.py
│ │ ├── heatwave.py
│ │ ├── heavy-precipitation.py
│ │ ├── wildfires.py
│ │ ├── floods.py
│ │ └── cyclones.py
│ ├─ accumulated/ # Dimension
│ │ ├── drought.py
│ │ ├── heatwave.py
│ │ ├── heavy-precipitation.py
│ │ ├── wildfires.py
│ │ ├── floods.py
│ │ └── cyclones.py
│ ├─ longterm/ # Dimension
│ │ ├── temperature-anomaly.py
│ │ ├── relative-sea-level.py
│ │ └── precipitation-anomaly.py
├─ vulnerability/
│ │ └── ...
├─ data/ # Example base storage location (defined in config.yaml)
│ ├─ input/ # Reserved for local input data, such as manual downloads
│ ├─ processing/ # Managed by StorageManager (cached/intermediate results)
│ └─ output/ # Managed by StorageManager (final indicator/aggregate score outputs)
├─ utils/ # Various kinds of utility functions
│ ├─ conflict.py
│ ├─ climate.py
│ ├─ climate.py
│ ├─ data_processing.py
│ ├─ spatial_operations.py
│ ├─ index.py
│ └─ ...
├── config.yaml # Central configuration file
└── ccvi.py # Main script collecting all components and providing an interface to run them
```


## Configuration (`config.yaml`)

The `config.yaml` is the central configuration file for the indicator. It has the following options:

*   **`global`**: Settings applicable everywhere.
    ```yaml
    global:
      storage_path: data # The root directory for data storage 
      start_year: 2015 # The start year for index data
      regenerate: # different regenerate flags
        regenerate_indicators: True
        regenerate_preprocessing: False
        regenerate_data: False
    ```
    **`data`**: Maps data source keys (used by `Dataset` subclasses) to their filenames *relative* to the `input/` directory.
    ```yaml
    data:
      vdem: V-Dem-CY-Full+Others-v14.rds
      countries: basemap.parquet
      land_mask: ne_50m_land.zip
      # ... other data sources
    ```
*   **`indicators`**: Nested dictionary structure (`pillar > dimension > id`) containing parameters specific to each indicator (e.g., thresholds, normalization kwargs).
    ```yaml
    indicators:
      CON: # Pillar
        level: # Dimension
          intensity: # Indicator ID
            # Specific parameters for CON_level_intensity
            normalization_quantile: 0.99
            # ...
    ```
*   **`aggregation`**: Nested dictionary structure (`pillar > dimension`) containing parameters for dimension aggregation.
    ```yaml
    aggregation:
      CON: # Pillar
        level: # Dimension
          method: mean # mean, gmean or pmean
          weights: None # None, or weight for each component 
          normalize: True # whether to re-normalize to 0-1 after aggreation step
    ```


## Development of Additional Components

*   **Adding a New Data Source:**
    1.  Create a new class inheriting from `base.data.Dataset`.
    2.  Define the required `data_key` class attribute and set the `local` attribute (`True` or `False`). Optionally define the `data_keys` attribut (list) if multiple input files are needed.
    3.  Implement the `load_data()` method for reading the specific file format.
    4.  Add preprocessing methods if applicable. Storage of preprocessing steps is available through `self.storage`.
    5.  For local data sources, add an entry to `config.yaml['data']` mapping the `data_key` to the data filename and place the raw data file in the correct location within the `input/` directory.

*   **Adding a New Indicator:**
    1.  Create a new class inheriting from `base.objects.Indicator` and any required Mixin classes in the appropriate `pillar/dimension/` folder.
    2.  Ensure `__init__` calls `super().__init__` correctly and sets default `pillar`, `dim`, `id` values.
    3.  Instantiate required `Dataset` subclasses within `__init__` (e.g., `self.acled = ACLEDData(config=config)`).
    4.  Implement the abstract methods (`load_data()`, `preprocess_data()`, `create_indicator()`, `normalize()`) and optionally `add_raw_value()`. The methods should work within the universal `run()` logic implemented in the base class.
    5.  Add indicator-specific parameters to `config.yaml['indicators']` under the correct `pillar > dimension > id` structure.

*   **Adding a New Dimension:**
    1.  The **existing `base.objects.Dimension` class should be sufficient**, as it provides the standard aggregation logic.
    2.  Create a new dimension instance to the main `ccvi.py` script with the corresponding list of indicators.
    3.  Add the dimension's aggregation configuration (method, weights, normalize flag) to `config.yaml['aggregation']` under the correct `pillar > dimension` structure.
    

Any new Indicators or Dimensions must be initialized in the central `ccvi.py` script, integrated where desired, and their `run()` methods made accessible to users through the `__main__` function.

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for Python package, version and virtual environment management. To setup the local environment, use:

```bash
uv sync
```

after cloning the repository. This should install required packages and the python version.

Additional packages can be installed similar to pip with:

```bash
uv add <package-name>
```

For more information see the [official `uv` documentation](https://docs.astral.sh/uv/).

### Formatting

The repository is formated via ruff. To keep the formatting consistent after making changes, simply run

```bash
uvx ruff format .
```

The ruff linter is available via

```bash
uvx ruff check .
```
