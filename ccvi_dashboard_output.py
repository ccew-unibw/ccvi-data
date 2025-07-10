"""
Script to be run manually AFTER the ccvi pipeline has run to combine all data
required for the dashboard and push it to the S3 bucket. Stores the files
versioned by quarter in the "tool" subfolder of the output folder.
"""
import os

import boto3
from dotenv import load_dotenv
import pandas as pd
from panel_imputer import PanelImputer

from base.datasets import WPPData, UNHCRData, CPIData, WBData, VDemData, FHData, SWIIDData, SDGData, HDIData, IMFData, ILOData
from base.objects import (
    Dimension,
    Pillar,
)
import ccvi
from utils.index import get_quarter  # load initialized components

def get_vul_country_data():
    config = ccvi.config
    
    wpp  = WPPData(config)
    unhcr = UNHCRData(config)
    cpi = CPIData(config)
    wb = WBData(config)
    vdem = VDemData(config)
    fh = FHData(config)
    swiid = SWIIDData(config)
    sdg = SDGData(config)
    hdi = HDIData(config)
    imf = IMFData(config)
    ilo = ILOData(config)

    df_wpp = wpp.preprocess_wpp(wpp.load_data())
    df_unhcr = unhcr.preprocess_data(unhcr.load_data())[["forcibly_displaced"]]
    df_cpi = cpi.preprocess_data(cpi.load_data())
    wb_dict = {
        "RL.EST": "rl",
        "NV.AGR.TOTL.ZS": "agr_value_added",
        "NY.GDP.MKTP.PP.CD": "gdp_ppp"
    }
    df_wb = wb.load_data(wb_dict)
    df_imf = imf.preprocess_data(imf.load_data({"PPPGDP": "gdp_ppp",}), scaling_factor=1000000000)
    df_gdp = ccvi.vul_socioeconomic_deprivation._merge_gdp_data(df_wb, df_imf)
    df_gdp = ccvi.vul_socioeconomic_deprivation._add_pc_values(df_gdp, df_wpp)
    df_wb = pd.concat([df_wb[["rl", "agr_value_added"]], df_gdp[["gdp_ppp", "gdp_ppp_pc"]]], axis=1)
    ilo_indicators = {
        "EMP_TEMP_SEX_ECO_NB_A": "agr_sector_share",  # Employment by sex and economic activity (thousands) -- Annual
        "EMP_2EMP_SEX_ECO_NB_A": "agr_sector_share_model",  # Employment by sex and economic activity -- ILO modelled estimates, Nov. 2023 (thousands) -- Annual
        "EAP_DWAP_SEX_AGE_RT_A": "labor_force_participation",  # Labour force participation rate by sex and age (%) -- Annual
        "EAP_2WAP_SEX_AGE_RT_A": "labor_force_participation_model",  # Labour force participation rate by sex and age (%) -- Annual
    }
    df_ilo = ilo.preproces_data_agrdep(ilo.load_data(ilo_indicators))
    df_swiid = swiid.preprocess_data(swiid.load_data())
    df_sdg = sdg.preprocess_data(sdg.load_data(["SN_ITK_DEFC"]))
    df_hdi = hdi.preprocess_data(hdi.load_data(["gii", "eys", "mys", "le"]))
    df_vdem = vdem.preprocess_data(vdem.load_data(["v2x_libdem", "v2xpe_exlsocgr", "v2xpe_exlgender", "v2x_rule", "v2x_civlib", "v2x_polyarchy"]))
    df_fh = fh.preprocess_data(fh.load_data())[["political_rights_percent", "civil_liberties_percent"]]
    dfs = [df_wpp, df_unhcr, df_cpi, df_wb, df_ilo, df_sdg, df_hdi, df_vdem, df_fh, df_swiid]
    dfs = [df.sort_index().loc[(slice(None), slice(2015, get_quarter("last").year)), slice(None)] for df in dfs]
    df = pd.concat(dfs, axis=1).sort_index()
    return df

### WRAPPER CLASSES FOR DASHBOARD ###
class DimToolOutputWrapper:
    """
    Wraps Dimension class, implementing `run()` storing all data for dashboard use.

    Skips some of the checks performed during the original dimension calculation.
    Runs the original validation to make sure output is up to date.
    """

    def __init__(self, dimension: Dimension):
        """Initialize Wrapped Dimension class

        Args:
            dimension (Dimension): An initialized Dimension instance.
        """
        assert isinstance(dimension, Dimension), (
            f"'dimension' arg needs to be an instance of Dimension, got {type(dimension)} instead."
        )
        self.dimension = dimension
        self.console = dimension.console
        self.composite_id = dimension.composite_id

    def run(self) -> None:
        """Loads all scores from indicators and adds dimension score.

        Checks whether all indicators are generated, runs the standard dimension
        validation, loads indicators, fills missing using forward fill, checks
        and loads dimension scores (setting generated to True), combines
        everything and stores it in the "tool" output subfolder.
        """
        self.console.print(f'Processing dimension "{self.composite_id}"...')
        for i in self.dimension.components:
            assert i.storage.check_component_generated(), (
                f"Indicator {i.composite_id} not yet generated for last quarter, check data output!"
            )
        self.dimension.validate_indicator_input(self.dimension.components)

        self.console.print("Loading indicators...")
        df = self.dimension.load_components(load_additional_values=True)
        # fill missing data with the last available observation
        imputer = PanelImputer(
            time_index=["year", "quarter"], location_index="pgid", imputation_method="ffill"
        )
        df: pd.DataFrame = imputer.fit_transform(df)  # type: ignore
        if self.dimension.has_exposure:
            self.console.print("Adding exposure...")
            df_exp = self.dimension.add_exposure(df)
            df = pd.concat([df, df_exp], axis=1)

        self.console.print("Loading aggregate score...")
        assert self.dimension.storage.check_component_generated(), (
            f"Dimension {self.dimension.composite_id} not yet generated for last quarter, check data output!"
        )
        df_aggregated = self.dimension.storage.load()

        df = pd.concat([df, df_aggregated], axis=1)
        self.dimension.storage.save(df, subfolder="tool")
        self.console.print(
            f'All Dimension "{self.composite_id}" scores successfully processed and saved.'
        )
        return


class PillarToolOutputWrapper:
    """
    Wraps Pillar class implementing `run()`, storing all data for dashboard use.

    Skips some of the checks performed during the original pillar calculation.
    Runs the original validation to make sure output is up to date.
    """

    def __init__(self, pillar: Pillar):
        """Initialize Wrapped Pillar class

        Args:
            dimension (Pillar): An initialized Dimension instance.
        """
        assert isinstance(pillar, Pillar), (
            f"'pillar' arg needs to be an instance of Pillar, got {type(pillar)} instead."
        )
        self.pillar = pillar
        self.console = pillar.console
        self.composite_id = pillar.composite_id

    def run(self) -> pd.DataFrame:
        """Loads all scores from Dimension Wrapper and adds pillar score.

        Check whether pillar has been generated for the last quarter and raises
        AssertionError otherwise.

        Returns:
            pd.DataFrame: Dataframe with all component scores and pillar score.
        """
        self.console.print(f'Processing pillar "{self.composite_id}"...')
        self.console.print("Loading components...")
        df = self.pillar.load_components(load_additional_values=True, subfolder="tool")

        self.console.print("Adding pillar score...")
        assert self.pillar.storage.check_component_generated(), (
            f"Pillar {self.composite_id} not yet generated for last quarter, check data output!"
        )
        df_aggregated = self.pillar.storage.load()
        df = pd.concat([df, df_aggregated], axis=1)
        return df


class CCVIWrapper:
    """
    Wraps CCVI class implementing a run combining all data for dashboard use.
    """

    def __init__(self, ccvi_instance: ccvi.CCVI):
        """Initialize Wrapped CCVI class.

        Args:
            ccvi_instance (CCVI): An initialized CCVI instance.
        """
        assert isinstance(ccvi_instance, ccvi.CCVI), (
            f"'ccvi_instance' arg needs to be an instance of CCVI, got {type(ccvi_instance)} instead."
        )
        self.ccvi = ccvi_instance
        self.console = ccvi_instance.console
        self.cli = PillarToolOutputWrapper(self.ccvi.cli)
        self.con = PillarToolOutputWrapper(self.ccvi.con)
        self.vul = PillarToolOutputWrapper(self.ccvi.vul)

    @staticmethod
    def copy_to_s3(directory: str) -> None:
        load_dotenv()
        bucket_name: str = os.getenv("S3_BUCKET_NAME")  # type: ignore
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("S3_ACCESS_ID"),
            aws_secret_access_key=os.getenv("S3_ACCESS_KEY"),
            endpoint_url=os.getenv("S3_ENDPOINT"),
        )
        for path, _, files in os.walk(directory):
            for f in files:
                fp = os.path.join(path, f)
                key = os.path.relpath(fp, directory)
                s3_client.upload_file(Filename=fp, Bucket=bucket_name, Key=key)
        return

    def run(self):
        """Loads all scores and data recency and stores in tool output format.

        Loads all preprocessed components from "tool" output subfolder, loads
        risk scores, combines both. Also loads data recency and grid and stores
        everything
        """
        self.console.print(f"Processing full data for dashboard output...")
        self.console.print("Load components...")
        dfs = [p.run() for p in [self.cli, self.con, self.vul]]
        df = pd.concat(dfs, axis=1)
        self.console.print("Load recency and risk scores...")
        # the quarter id makes sure this is up to date
        data_recency = self.ccvi.storage.load(
            filename=f"ccvi_scores_{self.ccvi.quarter_id}", subfolder=self.ccvi.quarter_id
        )
        df_aggregated = self.ccvi.storage.load(
            filename=f"data_recency_{self.ccvi.quarter_id}", subfolder=self.ccvi.quarter_id
        )
        # limit to scores
        df = df[[c for c in df.columns if c not in df_aggregated.columns]]
        df = pd.concat([df_aggregated, df], axis=1)
        # loading exposure is easiest via one of the climate dims
        df_exp = ccvi.cli_current._create_exposure_layers()
        # get vulnerability country data - separate loading function
        df_vul_country = get_vul_country_data()
        # store
        subfolder = os.path.join("tool", self.ccvi.quarter_id)
        self.ccvi.storage.save(df, filename="ccvi_scores", subfolder=subfolder)
        self.ccvi.storage.save(data_recency, filename="data_recency", subfolder=subfolder)
        self.ccvi.storage.save(ccvi.base_grid.load(), filename="base_grid", subfolder=subfolder)
        self.ccvi.storage.save(df_exp, filename="exposure_layers", subfolder=subfolder)
        self.ccvi.storage.save(df_exp, filename="vul_country_raw", subfolder=subfolder)
        # send to S3
        filepath = self.ccvi.storage.build_filepath("output", subfolder=subfolder)
        self.copy_to_s3(filepath)
        self.console.print("CCVI data successfully processed for dashboard and sent to S3!")
        return


### APPLY WRAPPERS ###

# CONFLICT #
con_level = DimToolOutputWrapper(ccvi.con_level)
con_persistence = DimToolOutputWrapper(ccvi.con_persistence)
con_soctens = DimToolOutputWrapper(ccvi.con_soctens)

# VULNERABILITY #
vul_socioeconomic = DimToolOutputWrapper(ccvi.vul_socioeconomic)
vul_political = DimToolOutputWrapper(ccvi.vul_political)
vul_demographic = DimToolOutputWrapper(ccvi.vul_demographic)

# CLIMATE #
cli_current = DimToolOutputWrapper(ccvi.cli_current)
cli_accumulated = DimToolOutputWrapper(ccvi.cli_accumulated)
cli_longterm = DimToolOutputWrapper(ccvi.cli_longterm)

# COMBINED AGGREGATE SCORES #
ccvi_tool = CCVIWrapper(ccvi.ccvi)

if __name__ == "__main__":
    # run everything sequentially - no checks or automatic runs for this
    cli_current.run()
    cli_accumulated.run()
    cli_longterm.run()
    con_level.run()
    con_persistence.run()
    con_soctens.run()
    vul_socioeconomic.run()
    vul_political.run()
    vul_demographic.run()
    ccvi_tool.run()
