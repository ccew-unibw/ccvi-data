import os
import pandas as pd
import requests
import geopandas as gpd
from shapely.geometry import Polygon

from base.objects import Dataset, console
from utils.index import get_quarter


def download_data_publication(output_path):
    url = "https://github.com/daniellincke/DIVA_paper_subsidence/archive/refs/tags/v1.0.tar.gz"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = os.path.join(output_path, "v1.0.tar.gz")
    r = requests.get(url, stream=True)
    with open(output_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print("Downloaded file to: ", output_file)
    os.system("tar -xvf " + output_file + " -C " + output_path)
    print("Unzipped file to: ", output_path)

    # read the coastline shapefile from the publication
    coastline = gpd.read_file(output_path + "DIVA_paper_subsidence-1.0/input/gis/cls_p32.shp")
    tabular_data = pd.read_csv(
        output_path + "DIVA_paper_subsidence-1.0/results/cls_slrrates_2015.csv"
    )

    # join tabular data with the coastline shapefile
    coastline = coastline.merge(tabular_data, on="locationid")
    coastline = coastline[["geometry", "rslr_high", "locationid"]]
    return coastline


# Create a function to generate a square polygon around a point
def create_square(lat, lon, size=0.5):
    # Create a square polygon of the specified size centered at the lat/lon
    half_size = size / 2
    return Polygon(
        [
            (lon - half_size, lat - half_size),
            (lon + half_size, lat - half_size),
            (lon + half_size, lat + half_size),
            (lon - half_size, lat + half_size),
        ]
    )


def intersect_coastline_priogrid(coastline, priogrid):
    # make a poligon grid from priogrid lat lon centroid to polygon
    # Create a GeoDataFrame with geometry as square polygons
    priogrid["geometry"] = [
        create_square(row.lat, row.lon, size=0.5) for row in priogrid.itertuples()
    ]
    priogrid = gpd.GeoDataFrame(priogrid, crs="EPSG:4326")

    # Reset index if necessary
    priogrid = priogrid.reset_index()

    # intersect the coastline with the priogrid
    priogrid = priogrid[["pgid", "lat", "lon", "geometry"]]

    priogrid = gpd.sjoin(priogrid, coastline, how="left", predicate="intersects")
    # group by pgid and max rslr_high

    priogrid = priogrid.groupby(["pgid", "lat", "lon"])["rslr_high"].max().reset_index()

    return priogrid


class DIVASubsidenceData(Dataset):
    """Handles loading and processing of  event data.

    Supports loading from local preprocessed files or via API.
    Includes methods for aggregating event data to the
    grid-quarter level.

    Attributes:
        data_key (str): Set to "diva_subsidence".
        local (bool): Indicates whether to use local dumps (True) or
            download data via the API (False).
        dataset_available (bool): Flag set to True after data is
            successfully loaded or checked by `load_data`.
    """

    data_key = "diva_subsidence"

    def __init__(self, local: bool = True, *args, **kwargs):
        """Initializes the data source.

        Sets the operation mode (local file vs API) and calls the Dataset
        initializer to setup config and storage.

        Args:
            config (ConfigParser): An initialized ConfigParser instance.
            local (bool, optional): Indicates whether to use local
                dumps (True) or download data via the API (False).
        """
        self.local = local
        self.dataset_available = False
        self.data_keys = [self.data_key, "countries"]
        super().__init__(*args, **kwargs)

    def download_data(self):
        """Downloads  data from the API.

        Downloads the  data from the API and saves it to the processing
        storage. The filename is based on the current date and time.

        Returns:
            str: The filename of the downloaded  data.
        """

        sources_path = self.storage.storage_paths["processing"]
        destination = f"{sources_path}/"

        if not os.path.exists(destination):
            os.makedirs(destination)

        # step1
        print("downloading data")
        coastline = download_data_publication(destination)
        coastline.to_parquet(f"{self.storage.storage_paths['processing']}/{self.filename}.parquet")

    def load_data(self):
        """Loads  data, checking for cached processing files first.

        Attempts to load a local  copy from the 'processing' storage
        including the last completed quarter. If not found:
        - If `self.local` is True, loads the raw dump specified in the config.
          Raises an error if the provided  dump does not fully cover the
          latest quarter.
        - If `self.local` is False, currently raises NotImplementedError (API access TBD).
        Saves the loaded raw/dump data to the processing storage.

        Returns:
            pd.DataFrame: The loaded  event data.
        """
        self.last_quarter_date = get_quarter("last", bounds="end")
        self.filename = f"relative-sea-level_{self.last_quarter_date.year}_Q{int(self.last_quarter_date.month / 3)}"
        self.columns = ["YEAR", "EVENT_TYPE", "LATITUDE", "LONGITUDE", "COUNT", "EVENT_DATE"]
        try:
            df_event_level = gpd.read_parquet(
                f"{self.storage.storage_paths['processing']}/{self.filename}.parquet"
            )
        except FileNotFoundError:
            if self.local:
                df_event_level = pd.read_parquet(self.data_config[self.data_key])
                if df_event_level["EVENT_DATE"].max() < self.last_quarter_date:
                    raise Exception(
                        "preprocessed  data out of date, please provide a version up to "
                        f"{self.last_quarter_date}."
                    )
            else:
                self.download_data()
                self.dataset_available = True
                # df_event_level = self.storage.load("processing", filename=self.filename)
                df_event_level = gpd.read_parquet(
                    f"{self.storage.storage_paths['processing']}/{self.filename}.parquet"
                )
                return df_event_level

        # Set an instance attribute for easy checking
        self.dataset_available = True
        return df_event_level

    def create_grid_quarter_aggregates(
        self,
        df_base: pd.DataFrame,
        df_event_level: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculates grid-quarter aggregates from the event level  data.

        Assigns events to grid cells (pgid), calculates quarterly aggregates
        (event counts, fatalities) for armed violence and unrest, merges with the
        base grid structure, and fills missing values based on data coverage
        information.

        Args:
            df_base (pd.DataFrame): Base data structure for indicator data.
            df_event_level (pd.DataFrame):  event-level data from `self.load_data`.

        Returns:
            pd.DataFrame: Dataframe aligned to index grid with quarterly  aggregates.
        """
        fp_preprocessed = self.storage.build_filepath("processing", filename="preprocessed")
        try:
            df = pd.read_parquet(fp_preprocessed)
            last_quarter_date = get_quarter("last")

            if df["time"].max().date() < last_quarter_date:
                raise FileNotFoundError
            return df
        except FileNotFoundError:
            console.print(
                "No preprocessed  data in storage or out of date," + " processing event data..."
            )

            # don't automatically start  download since those are separate step in the
            # indicator logic that should each be performed deliberately
            assert self.dataset_available, " download/data check has not run, check indicator logic"

            # quarter as number 1,2,3,4
            priogrid = df_base
            coastline = df_event_level

            # dorp duplicate pgid
            priogrid_dedup = priogrid.reset_index().drop_duplicates(subset=["pgid"])

            priogrid_dedup = intersect_coastline_priogrid(coastline, priogrid_dedup)
            priogrid_dedup = priogrid_dedup.loc[priogrid_dedup["rslr_high"] > 0]
            # priogrid_dedup.to_csv("priogrid_sea_level_rise.csv",index=False)
            # file ist static from Q4 2015 - assign this date
            # priogrid_dedup["year"] = 2015
            # priogrid_dedup["quarter"] = 4

            # join the priogrid_dedup with the deduped priogrid using pgid, year, quarter

            df = priogrid.reset_index().merge(
                priogrid_dedup[["pgid", "rslr_high"]], on=["pgid"], how="left"
            )

            df.rename(columns={"rslr_high": "count"}, inplace=True)

            df = df[["pgid", "year", "quarter", "lat", "lon", "count", "time"]]

            df["time"] = pd.to_datetime(df["time"])

            df = df.fillna(0)

            df.to_parquet(fp_preprocessed)
        return df


# test class
if __name__ == "__main__":
    from base.objects import ConfigParser

    config = ConfigParser()

    # Example usage
    data = DIVASubsidenceData(local=False, config=config)
    # just load the current data

    df_gee_heatwave = data.load_data()
    print(df_gee_heatwave.head())
