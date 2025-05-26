# ccvi-data: Data processing for the Climate—Conflict—Vulnerability Index (CCVI)

The Climate Conflict Vulnerability Index (CCVI) is the result of a joint research project between the [Center for Crisis Early Warning (CCEW)](https://www.unibw.de/ciss-en/ccew) at [University of the Bundeswehr Munich](https://www.unibw.de/home-en), the [FutureLab "Security, Ethnic Conflicts and Migration"](https://www.pik-potsdam.de/en/institute/futurelabs-science-units/security-ethnic-conflicts-and-migration) at the [Potsdam Institute for Climate Impact Research (PIK)](https://www.pik-potsdam.de/en), and the [German Federal Foreign Office](https://www.auswaertiges-amt.de/en). 

The goal of the project is to establish a scientifically informed tool that enables policymakers and researchers to assess and map current global risks to human security arising from climate and conflict hazards, their intersections and the potential for harmful interactions. Additionally, the CCVI reveals how vulnerabilities can amplify the impacts of climate and conflict hazards, increasing risks to human security.

The data and documentation of our conceptual and technical approach are available at [https://climate-conflict.org](https://climate-conflict.org/www/index/methodology). 

The data is updated quarterly and gridded to 0.5 degrees (ca. 55km by 55km at the equator).

## Table of Contents
* [Overview](#overview)
* [Setup](#setup)
* [Configuration](#configuration)
* [Framework & Architecture](#framework--architecture)
* [Contributions](#contributions)
* [License & Disclaimer](#license)

## Overview

The repository is broadly structured along the pillars of the CCVI, with corresponding `climate`, `conflict` and `vulnerability` folders containing respective indicators. Additionally, the `base` folder contains base classes providing the implementation structure and classes for each dataset used, while the `utils` folder contains functionality shared across indicators and dimensions. The main `ccvi.py` script initializes all components and orchestrates running the CCVI data pipeline. Available configuration can be set in the `config.yaml`.

To run the CCVI data pipeline, use `uv run ccvi.py` after the [setup](#setup).

## Setup

The project can be setup and run on a single workstation.

### Technical Requirements

* Compute: CPU only. Some parallelization, though good single-core performance is a plus.
* Memory: < 128GB
* Storage: < 2TB

### Data Requirements

All data used in the CCVI is publically available. The project depends both on locally downloaded input data and data accessed via APIs, downloaded automatically as part of the data pipeline.

**Locally required input files** need to be downloaded into the input subfolder of the storage directory defined in the [config](#configuration). See the comments in the .yaml for which files are currently required and where to download them.

**Data downloaded via APIs**, but also local input data, may require registering with the data providers. API keys and other required secrets are read from a `.env` file, which must be setup according to the `.env.template`.

### Environment

**Python**

The project was developed and tested on python3.12.

**R**

The project uses R for some of its calculations via the r2py bridge. For this to work, R needs to be installed on your enviroment. The project was tested on R 4.3.2.

**Python Dependencies:**

This project uses [`uv`](https://github.com/astral-sh/uv) for Python package, version and virtual environment management. uv needs to be installed before it can be used to setup the environment.

To setup the virtual environment, use:

```bash
uv sync
```

after cloning the repository. This should install required packages and the python version.

Additional packages can be installed similar to pip with:

```bash
uv add <package-name>
```

For more information see the [official `uv` documentation](https://docs.astral.sh/uv/).

The repository is formated via [ruff](https://github.com/astral-sh/ruff).

## Configuration

The `config.yaml` is the central configuration file for the indicator. For a full description of the settings, see the file itself.

The following configurations are available

*   **`global`**: Settings applicable everywhere. Includes regeneration config, enabling the forced regeneration of processing steps despite cached versions.
    ```yaml
    global:
      # Start year for data processing and indicator generation. 
      start_year: 2015
      # Path under which the input/processing/output storage folders are contained/will be created.
      storage_path: data
      # IDs added to regenerate force regeneration of indicator calculation preprocessing or data 
      # loading even if current versions are in storage. Aggregate scores are always regenerated.
      regenerate:
        indicator:
          - pillar_dimension_id
        preprocessing:
          - data_key
        data:
          - data_key
    ```
    **`data`**: Maps data source keys to their filenames *relative* to the `input/` directory.
    ```yaml
    data:
      vdem: V-Dem-CY-Full+Others-v14.rds
      countries: geoBoundariesCGAZ_ADM0.gpkg
      land_mask: ne_50m_land.zip
      # ... other data sources
    ```
*   **`indicators`**: Nested dictionary structure (`pillar > dimension > id`) containing parameters specific to each indicator, defined by each class individially.
    ```yaml
    indicators:
      CON: # Pillar
        level: # Dimension
          intensity: # Indicator ID
            # Specific parameters for CON_level_intensity
            normalization_quantile: 0.99
            # ...
    ```
*   **`aggregation`**: Nested dictionary structure (`pillar > dimension`) containing parameters for risk scores, pillar, and dimension aggregations. 
    ```yaml
    aggregation:
      CON: # Pillar
        level: # Dimension
          method: mean # standard methods are mean, gmean or pmean
          weights: None # None (equal), or weight for all components as composite_id: weight pairs
          normalize: True # whether to re-normalize to 0-1 after aggreation step
          # ...
      RISK: # Risk scores
        CCVI:
          method: mean
          weights: None
          normalize: True
          # ...
    ```


## Framework & Architecture

The project framework was designed with modularity in mind. Individual indicator and data sources can easily be modified and replaces without affecting the whole project.

The architecture follows the composite index logic of the CCVI. Base classes were designed to handle the core functionality and provide a unique interface for `Datasets`, `Indicators`, and our two main aggregation levels `Dimensions` and Pillars. Additionally, shared `ConfigParser`, `StorageManager` and `GlobalBasegrid` classes provide the framework for the geospatial resultion, to read config, and to cache processing steps and store results.

### Data structure

All CCVI scores are stored as `.parquet` files from pandas DataFrames with a `('pgid', 'year', 'quarter')` MultiIndex, where `pgid` stands for PRIO-GRID id, an unique identifier for each grid cell.

### Datasets
*`base.objects.Dataset`*

The `Dataset` class provides the basic framework to add datasets to the CCVI. Datasets are responsible for encapsulating all logic related to accessing, downloading, and performing initial preprocessing of specific external data sources. Each Dataset subclass, is tailored to a particular source and needs to implement at minimum a `load_data()` method and set their `data_key` class attribute. The `local` attribute distinguishes between local file sources and API-based sources, with required local files defined by the corresponding data_key(s) in the config. Each dataset is initialized with the shared `ConfigParser` instance and sets up its own `StorageManager` instance. Caching preprocessing steps is handled in a subfolder within the processing/ directory, named after its data_key, which is created based on the `needs_storage` attribute. Dataset classes often include further methods for more specific data processing.

### Indicators
*`base.objects.Indicator`*

The `Indicator` class provides a framework for the processing steps each indicator needs to implement and orchestrates them. Each Indicator subclass creates one or more Dataset instances, loads and the data and applies specific calculations to transform this data into 0-1 score. Each Indicator is initialized with pillar, dim, and id identifiers, along with shared `ConfigParser` and `GlobalBaseGrid` instances and sets up its own `StorageManager`. Caching preprocessing steps is handled in a subfolder within the processing/ directory, named after their `composite_id` attribute, depending on the `requires_processing_storage` attribute. Finished indicator scores and any raw values are stored as .parquet files in the output/ directory, named after the `composite_id`. An internal generated flag, checked at initialization via `StorageManager.check_component_generated()`, determines if an up-to-date version of the indicator's output already exists in storage. 

The core workflow is defined by a series of abstract methods which subclasses implement:

* `load_data()`
* `preprocess_data()`
* `create_indicator()`
* `normalize()`

An optional add_raw_value() method can also be overridden. The workflow, (re-)generation checks and storage are orchestrated in the `run()` method.

### Dimensions and Pillars
*`base.objects.AggregatScore`, `base.objects.Dimension`, `base.objects.Pillar`, `ccvi.CCVI`*

The `Dimension` and `Pillar` classes represent the aggregation levels within the CCVI structure, with the top-level `CCVI` class performing the final risk score aggregations. Dimension classes aggregate multiple Indicator scores, while Pillar classes aggregate Dimension scores. The common logic for these aggregations is provided by the `AggregateScore` base class, which is inherited by the Dimension, Pillar, and CCVI classes. 

Aggregate score classes initialized with a list of their constituent objects (e.g., a list of Indicator instances for a Dimension) and the shared ConfigParser. They retrieve their specific aggregation parameters from the config and create their own StorageManager, using their composite_id (e.g., CON_level) as output filename. Aggregate scores are always in the data pipeline unless the same instance is run twice.

Similar to indicators, the `run()` method orchestrates the aggregation process: it 
* validates the input components and checks if they have been generated, 
* runs any missing components,
* loads the data from these components via `load_components()`,
* calculates aggreate scores via `aggregate()`.
* saves the final aggregated score as `.parquet` file to the output/ folder.

An optional add_exposure() modifies the data before aggregation depending on the `has_exposure` attribute, which is implemented for the climate pillar in the CCVI in the `climate.shared.ClimateDimension` subclass.

The `CCVI` top-level class does not store its own scores directly, but creates a DataFrame with **all** CCVI components and stores it in a versioned subfolder in the 'YYYY-Q#' (e.g. '2025-Q1') subfolder. It also creates and stores data recency metadata, denoting when the underlying datasources for each indicator were last updated.

### Utilities
*`base.objects.ConfigParser`, `base.objects.StorageManager`, `base.objects.GlobalBaseGrid`*

The framework relies on three core utility classes for its fundamental operations:

* The `ConfigParser` it loads and validates the config.yaml file and provides structured access to global (including regeneration), data source, indicator, and aggregation configurations. 
* The output structure, all indicator score I/O and some caching is handled by the `StorageManager`. This class creates the standard input/, processing/, and output/ directory structure and manages component-specific subfolders within processing/. It offers methods to save and load pandas DataFrames (as Parquet files), build file paths, and check for file existence or up-to-date generation of component outputs. It also manages the composite_id of the indicator and aggregate scores.
* The `GlobalBaseGrid` defines and manages the standard 0.5°x0.5° geospatial grid for the index. It handles the creation or loading of this grid, preprocesses country boundaries, filters water areas, and matches grid cells to countries, providing spatial resolution for all gridded indicators. The generation and caching of the base grid is orchestrated in the `run()` method.

## Contributions

We welcome bug reports through issues. While the version found on on <https://climate-conflict.org> is developed internally, with this repository we want to enable anyone to extend and adapt the CCVI to their needs and requirements, and create their own custom versions.

## License

This project is licensed under the GNU General Public License v3.0.  
See the [LICENSE](LICENSE) file for details.

## Disclaimer

The project is funded by the the German Federal Foreign Office. The views and opinions expressed in this projects, such as country assignments and boundaries, are those of the author(s) and do not necessarily reflect the official policy or position of any agency of the German government.

