import math

import pandas as pd
from base.objects import (
    Dimension,
    ConfigParser,
    GlobalBaseGrid,
    Pillar,
    AggregateScore,
    StorageManager,
)
from utils.index import get_quarter

# conflict indicators
from conflict.level import ConLevelIntensity, ConLevelSurrounding
from conflict.persistence import ConPersistenceIntensity, ConPersistenceSurrounding
from conflict.soctens import ConSoctensIntensity, ConSoctensPersistence, ConSoctensSurrounding

# climate indicators
from climate.current import (
    CliCurrentCyclones,
    CliCurrentDrought,
    CliCurrentFloods,
    CliCurrentHeatwave,
    CliCurrentHeavyPrecipitation,
    CliCurrentWildfires,
)
from climate.accumulated import (
    CliAccumulatedCyclones,
    CliAccumulatedDrought,
    CliAccumulatedFloods,
    CliAccumulatedHeatwave,
    CliAccumulatedHeavyPrecipitation,
    CliAccumulatedWildfires,
)
from climate.longterm import (
    CliLongtermPrecipitationAnomaly,
    CliLongtermRelativeSeaLevel,
    CliLongtermTemperatureAnomaly,
)
from climate.shared import ClimateDimension

# vulnerability indicators
from vulnerability.socioeconomic import (
    VulSocioeconomicAgriculture,
    VulSocioeconomicEducation,
    VulSocioeconomicDeprivation,
    VulSocioeconomicHealth,
    VulSocioeconomicInequality,
    VulSocioeconomicHunger,
)
from vulnerability.political import (
    VulPoliticalGender,
    VulPoliticalInstitutions,
    VulPoliticalSystem,
    VulPoliticalEthnic,
)
from vulnerability.demographic import (
    VulDemographicDependent,
    VulDemographicPopgrowth,
    VulDemographicUprooted,
)

### GLOBAL OBJECTS ###
config = ConfigParser()
base_grid = GlobalBaseGrid(config=config)

### CONFLICT ###
# Dim "level"
con_level_intensity = ConLevelIntensity(config=config, grid=base_grid)
con_level_surrounding = ConLevelSurrounding(config=config, grid=base_grid)
con_level = Dimension(
    "CON", "level", config=config, indicators=[con_level_intensity, con_level_surrounding]
)
# Dim "persistence"
con_persistence_intensity = ConPersistenceIntensity(
    config=config, grid=base_grid, base_indicator=con_level_intensity
)
con_persistence_surrounding = ConPersistenceSurrounding(
    config=config, grid=base_grid, base_indicator=con_level_surrounding
)
con_persistence = Dimension(
    "CON",
    "persistence",
    config=config,
    indicators=[con_persistence_intensity, con_persistence_surrounding],
)
# Dim "soctens"
con_soctens_intensity = ConSoctensIntensity(config=config, grid=base_grid)
con_soctens_persistence = ConSoctensPersistence(
    config=config, grid=base_grid, base_indicator=con_soctens_intensity
)
con_soctens_surrounding = ConSoctensSurrounding(config=config, grid=base_grid)
con_soctens = Dimension(
    "CON",
    "soctens",
    config=config,
    indicators=[con_soctens_intensity, con_soctens_persistence, con_soctens_surrounding],
)
con_pillar = Pillar("CON", config=config, dimensions=[con_level, con_persistence, con_soctens])
### VULNERABILITY ###
# Dim "socioeconomic"
vul_socioeconomic_agriculture = VulSocioeconomicAgriculture(config=config, grid=base_grid)
vul_socioeconomic_deprivation = VulSocioeconomicDeprivation(config=config, grid=base_grid)
vul_socioeconomic_education = VulSocioeconomicEducation(config=config, grid=base_grid)
vul_socioeconomic_health = VulSocioeconomicHealth(config=config, grid=base_grid)
vul_socioeconomic_inequality = VulSocioeconomicInequality(config=config, grid=base_grid)
vul_socioeconomic_insecurity = VulSocioeconomicHunger(config=config, grid=base_grid)
vul_socioeconomic = Dimension(
    "VUL",
    "socioeconomic",
    config=config,
    indicators=[
        vul_socioeconomic_agriculture,
        vul_socioeconomic_deprivation,
        vul_socioeconomic_education,
        vul_socioeconomic_health,
        vul_socioeconomic_inequality,
        vul_socioeconomic_insecurity,
    ],
)
# Dim "political"
vul_political_ethnic = VulPoliticalEthnic(config=config, grid=base_grid)
vul_political_gender = VulPoliticalGender(config=config, grid=base_grid)
vul_political_institutions = VulPoliticalInstitutions(config=config, grid=base_grid)
vul_political_system = VulPoliticalSystem(config=config, grid=base_grid)
vul_political = Dimension(
    "VUL",
    "political",
    config=config,
    indicators=[
        vul_political_ethnic,
        vul_political_gender,
        vul_political_institutions,
        vul_political_system,
    ],
)
# Dim "demographic"
vul_demographic_dependent = VulDemographicDependent(config=config, grid=base_grid)
vul_demographic_popgrowth = VulDemographicPopgrowth(config=config, grid=base_grid)
vul_demographic_uprooted = VulDemographicUprooted(config=config, grid=base_grid)
vul_demographic = Dimension(
    "VUL",
    "demographic",
    config=config,
    indicators=[vul_demographic_dependent, vul_demographic_popgrowth, vul_demographic_uprooted],
)
# Pillar
vul_pillar = Pillar(
    "VUL", config=config, dimensions=[vul_socioeconomic, vul_political, vul_demographic]
)
### CLIMATE ###
# Dim "current"
cli_current_floods = CliCurrentFloods(config=config, grid=base_grid)
cli_current_cyclones = CliCurrentCyclones(config=config, grid=base_grid)
cli_current_heavy_precipitation = CliCurrentHeavyPrecipitation(config=config, grid=base_grid)
cli_current_heatwave = CliCurrentHeatwave(config=config, grid=base_grid)
cli_current_wildfires = CliCurrentWildfires(config=config, grid=base_grid)
cli_current_drought = CliCurrentDrought(config=config, grid=base_grid)

cli_current = ClimateDimension(
    base_grid,
    "CLI",
    "current",
    config=config,
    indicators=[
        cli_current_floods,
        cli_current_cyclones,
        cli_current_heavy_precipitation,
        cli_current_heatwave,
        cli_current_wildfires,
        cli_current_drought,
    ],
)
# Dim "accumulated"
cli_accumulated_floods = CliAccumulatedFloods(config=config, grid=base_grid)
cli_accumulated_cyclones = CliAccumulatedCyclones(config=config, grid=base_grid)
cli_accumulated_heavy_precipitation = CliAccumulatedHeavyPrecipitation(
    config=config, grid=base_grid
)
cli_accumulated_heatwave = CliAccumulatedHeatwave(config=config, grid=base_grid)
cli_accumulated_wildfires = CliAccumulatedWildfires(config=config, grid=base_grid)
cli_accumulated_drought = CliAccumulatedDrought(config=config, grid=base_grid)

cli_accumulated = ClimateDimension(
    base_grid,
    "CLI",
    "accumulated",
    config=config,
    indicators=[
        cli_accumulated_floods,
        cli_accumulated_cyclones,
        cli_accumulated_heavy_precipitation,
        cli_accumulated_heatwave,
        cli_accumulated_wildfires,
        cli_accumulated_drought,
    ],
)

# Dim "longterm"
cli_longterm_relative_sea_level = CliLongtermRelativeSeaLevel(config=config, grid=base_grid)
cli_longterm_temperature_anomaly = CliLongtermTemperatureAnomaly(config=config, grid=base_grid)
cli_longterm_precipitation_anomaly = CliLongtermPrecipitationAnomaly(config=config, grid=base_grid)

cli_longterm = ClimateDimension(
    base_grid,
    "CLI",
    "longterm",
    config=config,
    indicators=[
        cli_longterm_relative_sea_level,
        cli_longterm_temperature_anomaly,
        cli_longterm_precipitation_anomaly,
    ],
)
# Pillar
cli_pillar = Pillar("CLI", config=config, dimensions=[cli_current, cli_accumulated, cli_longterm])


class CCVI(AggregateScore):
    """Orchestrates calculation the CCVI from indicator to risk scores.

    This class represents the top level composition of the
    Climate-Conflict-Vulnerability Index (CCVI). It takes initialized Pillar
    instances (Climate, Conflict, Vulnerability) and manages their execution. It
    loads all scores, calculates risk scores and the final CCVI score, and generates
    data recency metadata. All scores and the data recency information are saved
    to a versioned output folder named after the respective year and quarter
    ("YYYY_Q#").

    Attributes:
        console (Console): Shared console instance for output.
        composite_scores (dict[str, list[str]]): Defines the components for each
            composite score.
        aggregation_config (dict[str, dict[str, Any]]): Aggregation configurations
            for the composite scores, loaded from the config.yaml.
        cli (Pillar): Instance of the Climate pillar.
        con (Pillar): Instance of the Conflict pillar.
        vul (Pillar): Instance of the Vulnerability pillar.
        quarter_id (str): String representing the last completed quarter, used for
            versioning output (format "YYYY_Q#").
        global_config (dict[str, Any]): Global configuration settings.
        storage (StorageManager): Instance managing storage for this class.
        requires_processing_storage (bool): Flag indicating whether the aggregate
            needs processing storage. Set to False.
    """

    def __init__(self, config: ConfigParser, cli: Pillar, con: Pillar, vul: Pillar):
        """Initializes the CCVI calculation orchestrator.

        Loads global and aggregation config for the top-level composite scores,
        sets up output versioning, and initializes a StorageManager.

        Args:
            config (ConfigParser): An initialized ConfigParser instance.
            cli (Pillar): An initialized Climate Pillar instance.
            con (Pillar): An initialized Conflict Pillar instance.
            vul (Pillar): An initialized Vulnerability Pillar instance.
        """
        # the order is important here since CCVI is the combination of the other two
        self.composite_scores = {
            "CLI_risk": ["CLI", "VUL"],
            "CON_risk": ["CON", "VUL"],
            "CCVI": ["CLI_risk", "CON_risk"],
        }
        self.aggregation_config = {
            score: config.all_config["aggregation"]["RISK"][score]
            for score in self.composite_scores
        }
        self.cli = cli
        self.con = con
        self.vul = vul
        self.quarter_id = self._get_quarter_string()
        self.global_config = config.get_global_config()
        self.storage = StorageManager(
            storage_base_path=self.global_config["storage_path"],
            requires_processing_storage=self.requires_processing_storage,
        )

    def _get_quarter_string(self) -> str:
        """Determines the string representation of the last completed quarter.

        Calculates the year and quarter number of the most recently completed
        quarter based on the current date.

        Returns:
            str: A string formatted as "YYYY_Q#" (e.g., "2023_Q4").
        """
        last_quarter = get_quarter("last")
        year = last_quarter.year
        quarter = math.ceil(last_quarter.month / 3)
        quarter_string = f"{year}_Q{quarter}"
        return quarter_string

    def load_data(self) -> pd.DataFrame:
        """Loads all indicator and aggregate scores from storage.

        Iterates through the pillars and loads its own score and component
        dimensions. Then, for each dimension, loads its component indicators. All
        loaded scores are concatenated into a single DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing all CCVI scores, indexed by
                ('pgid', 'year', 'quarter').
        """
        dfs = []
        for pillar in [self.cli, self.con, self.vul]:
            dfs.append(pillar.storage.load())
            dfs.append(pillar.load_components())
            for dimension in pillar.components:
                dfs.append(dimension.load_components())
        df = pd.concat(dfs, axis=1)
        return df

    def run(self):
        """Executes the full CCVI data pipeline.

        This method orchestrates the entire index calculation process:
        1. Runs the `GlobalBaseGrid` generation.
        2. Triggers the `run()` method for each of the climate, conflict, and
           vulnerability pillars, which in turn triggers `run()` for all
           underlying dimensions and indicators as required.
        3. Loads all generated scores into a unified DataFrame, using `load_data()`.
        4. Calculates both climate and conflict risk scores and the final combined
           CCVI score using the inherited `aggregate()`.
        5. Determines the data recency for all scores using `get_data_recency()`.
        6. Saves the comprehensive DataFrame of all scores and the data recency
           information into a versioned subfolder (named after `self.quarter_id`)
           within the main output directory.
        """
        self.console.print("Running the CCVI data pipeline...")
        self.console.print(
            "A run can take anywhere from a few minutes to perform the aggregations based on "
            "generated indicators to several days/weeks if all data needs to be downloaded from "
            "scratch."
        )
        base_grid.run()
        self.console.print("Run pillars...")
        self.cli.run()
        self.con.run()
        self.vul.run()
        self.console.print("Load scores...")
        # load ALL scores for unified data recency metadata and versioned storage
        df = self.load_data()
        # calculate top-level aggregates
        self.console.print("Calculate risk scores and data recency...")
        for score in self.composite_scores:
            df[score] = self.aggregate(
                df[self.composite_scores[score]], score, self.aggregation_config[score]
            )
        data_recency = self.get_data_recency(df)
        self.console.print("Store results...")
        self.storage.save(df, filename=f"ccvi_scores_{self.quarter_id}", subfolder=self.quarter_id)
        self.storage.save(
            data_recency, filename=f"data_recency_{self.quarter_id}", subfolder=self.quarter_id
        )
        self.console.print("Running the CCVI data pipeline... DONE!")
        return


# Full CCVI
ccvi = CCVI(config=config, cli=cli_pillar, con=con_pillar, vul=vul_pillar)

if __name__ == "__main__":
    ############################################################################
    # each component is designed to be run individually as well if desired     #
    ############################################################################
    # base_grid.run()

    ## conflict dim "level"
    # con_level_intensity.run()
    # con_level_surrounding.run()
    # con_level.run()

    ## conflict dim "persistence"
    # con_persistence_intensity.run()
    # con_persistence_surrounding.run()
    # con_persistence.run()

    ## conflict dim "soctens"
    # con_soctens_intensity.run()
    # con_soctens_persistence.run()
    # con_soctens_surrounding.run()
    # con_soctens.run()

    ## conflict pillar
    # con_pillar.run()

    ## climate dim 1
    # cli_current_floods.run()
    # cli_current_cyclones.run()
    # cli_current_heavy_precipitation.run()
    # cli_current_heatwave.run()
    # cli_current_wildfires.run()
    # cli_current_drought.run()
    # cli_current.run()

    ## climate dim 2
    # cli_accumulated_floods.run()
    # cli_accumulated_cyclones.run()
    # cli_accumulated_heavy_precipitation.run()
    # cli_accumulated_heatwave.run()
    # cli_accumulated_wildfires.run()
    # cli_accumulated_drought.run()
    # cli_accumulated.run()

    ## climate dim 3
    # cli_longterm_relative_sea_level.run()
    # cli_longterm_temperature_anomaly.run()
    # cli_longterm_precipitation_anomaly.run()
    # cli_longterm.run()

    ## climate pillar
    # cli_pillar.run()

    ## Vulnerability dim "socioeconomic"
    # vul_socioeconomic_agriculture.run()
    # vul_socioeconomic_deprivation.run()
    # vul_socioeconomic_education.run()
    # vul_socioeconomic_health.run()
    # vul_socioeconomic_inequality.run()
    # vul_socioeconomic_insecurity.run()
    # vul_socioeconomic.run()

    ## Vulnerability dim "political"
    # vul_political_ethnic.run()
    # vul_political_gender.run()
    # vul_political_institutions.run()
    # vul_political_system.run()
    # vul_political.run()

    ## Vulnerability dim "demographic"
    # vul_demographic_dependent.run()
    # vul_demographic_popgrowth.run()
    # vul_demographic_uprooted.run()
    # vul_demographic.run()

    ## vulnerability pillar
    # vul_pillar.run()
    ############################################################################

    ############################################################################
    # full ccvi pipeline                                                       #
    ############################################################################
    ccvi.run()
