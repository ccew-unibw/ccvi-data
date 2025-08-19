from datetime import date, datetime, timedelta
from functools import cache
import math
import time
from typing import Any
import numpy as np
import requests

import country_converter as coco
import geopandas as gpd
import pandas as pd

from base.objects import Dataset, GlobalBaseGrid


class UCDPData(Dataset):
    """Handles downloading and preprocessing of UCDP data.

    Implements `load_data()` to download ...
    Implements `preprocess_data()` to ...

    Attributes:
        data_key (str): Set to "ucdp".
        local (bool): Set to False, as data is sourced from the UCDP API.
    """

    data_key = "ucdp"
    local = False
    url_ged: str = "https://ucdpapi.pcr.uu.se/api/gedevents/{}?pagesize=1000"

    def load_data(self) -> Any:
        try:
            if self.regenerate["data"]:
                raise FileNotFoundError
            ucdp = self.storage.load("processing", "ucdp_raw")
            # TODO: uncomment below before commit
            # ucdp = self._update_ucdp(ucdp)
        except FileNotFoundError:
            ucdp = self._download_ucpd()
            self.storage.save(ucdp, "processing", "ucdp_raw")
        return ucdp

    def preprocess_data(self, df: pd.DataFrame, grid: GlobalBaseGrid) -> pd.DataFrame:
        try:
            if self.regenerate["preprocessing"]:
                raise FileNotFoundError
            max_date_data = df["date_end"].apply(self._convert_timestr).max()
            df = self.storage.load("processing", "ucdp_preprocessed")
            # rerun if max date does not match current version from load_data()
            if df.time.max() < max_date_data:
                raise FileNotFoundError
            self.console.print("Existing preprocessed UCDP version loaded from storage.")

        except FileNotFoundError:
            df = df.set_index("id")
            df = df.rename(columns={"priogrid_gid": "pgid"})

            # just in case
            df["event_count"] = 1

            self.console.print("Matching locations with countries...")
            df = self._match_countries(df, grid.basemap)

            self.console.print("Assigning events to quarters...")
            df = self._process_time(df)
            self.storage.save(df, "processing", "ucdp_preprocessed")
        return df

    def _update_ucdp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Updates existing UCDP dataframe.

        Checks if and what needs to be updated (GED/Candidate), drops data where
        new GED data is available from input df based on versions, downloads new
        data and returns tuple of old and new dataframes.

        Args:
            df (pd.DataFrame): Existing dataframe with UCDP GED and Candidate data.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame | None]: tuple of subset of the input
                dataframe and the dataframe with new data or None if no new data
                was available.
        """
        current_year = date.today().year
        latest_ged, latest_candidate = self._get_latest_versions()
        # check input data
        ucdp_columns = self._get_ucdp_columns(latest_ged) + ["version"]
        assert all(col in df.columns for col in ucdp_columns)
        # updated official version if required
        keep_versions = []
        new_dfs = []
        if latest_ged not in df["version"].unique():
            df_ged = self._download_version(latest_ged)
            new_dfs.append(df_ged)
        else:
            keep_versions.append(latest_ged)
            # no update at all necessary
            if latest_candidate in df["version"].unique():
                return df

        for year in range(int(latest_ged[:2]), int(str(current_year + 1)[2:])):
            for i in range(1, 13):
                candidate_version = f"{year}.0.{i}"
                if candidate_version not in df["version"].unique():
                    df_month = self._download_version(candidate_version)
                    new_dfs.append(df_month)
                else:
                    keep_versions.append(candidate_version)
                if candidate_version == latest_candidate:
                    break

        df_update = pd.concat(new_dfs)
        df_keep = df.loc[df["version"].isin(keep_versions)].copy()
        try:
            assert not any(i in df_keep["id"].unique() for i in df_update["id"])
        except AssertionError:
            df_keep = self._handle_duplicate_ids(df_keep, df_update)
            assert not any(i in df_keep["id"].unique() for i in df_update["id"])
        df = pd.concat([df_keep, df_update])
        return df

    def _download_ucpd(self) -> pd.DataFrame:
        current_year = date.today().year
        latest_ged, latest_candidate = self._get_latest_versions()
        df_ged = self._download_version(latest_ged)

        if latest_candidate is not None:
            for year in range(int(latest_ged[:2]), int(str(current_year + 1)[2:])):
                for i in range(1, 13):
                    candidate_version = f"{year}.0.{i}"
                    df_month = self._download_version(candidate_version)
                    # remove duplicates from candidate in comparison to GED, otherwise remove
                    # duplicates from earlier version
                    if i == 1:
                        df_month = self._handle_duplicate_ids(df_month, df_ged)
                    else:
                        df_ged = self._handle_duplicate_ids(df_ged, df_month)
                    df_ged = pd.concat([df_ged, df_month])
                    if candidate_version == latest_candidate:
                        break

        assert df_ged["id"].is_unique, "event id is not unique, check for errors in data processing"
        return df_ged

    def _get_latest_versions(self) -> tuple[str, str | None]:
        """Check for the latest available UCDP data.

        Tries to download the first event returned by the API, iterating through
        possible versions based on the current (2025) UCDP naming conventions.

        Returns:
            tuple[str, str | None]: tuple with the latest ged and ged candidate
                version strings.
        """

        current_year = date.today().year
        # there should always be a version with last year's version number, since every year there is a new release
        safe_version = str(current_year - 1)[2:] + ".1"
        possible_version = str(current_year)[2:] + ".1"
        try:
            response = requests.get(
                self.url_ged.format(possible_version).replace("pagesize=1000", "pagesize=1")
                + "&page=0"
            )
            response.raise_for_status()
            ged_version = possible_version
        except requests.exceptions.HTTPError:
            ged_version = safe_version

        candidate_version = None
        for year in range(int(ged_version[:2]), int(str(current_year + 1)[2:])):
            for i in range(1, 13):
                if i == date.today().month and year == int(str(current_year)[2:]):
                    break
                test_version = f"{year}.0.{i}"
                try:
                    response = requests.get(
                        self.url_ged.format(test_version).replace("pagesize=1000", "pagesize=1")
                        + "&page=0"
                    )
                    response.raise_for_status()
                    candidate_version = test_version
                except requests.exceptions.HTTPError:
                    continue

        return ged_version, candidate_version

    def _handle_duplicate_ids(self, df_prior, df_update) -> pd.DataFrame:
        """Removes duplicate IDs in among df_prior and df_update from df_prior.

        Returns:
            df_prior without rows with duplicate IDs.
        """
        duplicate_ids = [i for i in df_update["id"] if i in df_prior["id"].unique()]
        df_prior = df_prior[~df_prior["id"].isin(duplicate_ids)].copy()
        return df_prior

    def _download_version(self, version: str, max_retries: int = 5) -> pd.DataFrame:
        """Helper method to download data for a single UCDP version.

        Makes get request to UCDP API and iterates through all pages, reading
        data as json, concatenating the results and converting it to a pd.DataFrame.
        Includes retries after brief sleep periods in case of errors for individual
        pages.

        Args:
            version (str): UCDP version to download.
            max_retries (int): Maximum number of retries if individual pages fail
                to download.

        Returns:
            pd.DataFrame: Full data of UCDP version.
        """
        self.console.print(f"Downloading UCDP GED version {version}...")
        url = self.url_ged.format(version)
        response = requests.get(url)
        response.raise_for_status()
        content = response.json()
        ged_entries = content["Result"]
        while content["NextPageUrl"]:
            retries = 0
            success = False
            while retries <= max_retries and not success:
                retries += 1
                try:
                    response = requests.get(content["NextPageUrl"])
                    response.raise_for_status()
                    content = response.json()
                    ged_entries.extend(content["Result"])
                    success = True
                except Exception as e:
                    time.sleep(10)
                    self.console.print(f"Ran into an exception, retrying... Exception: {e}")
                    if retries >= max_retries:
                        raise
            content = response.json()
            ged_entries.extend(content["Result"])

        df_ged = pd.DataFrame(ged_entries)
        df_ged = df_ged.drop_duplicates()
        df_ged["version"] = version
        return df_ged

    def _get_ucdp_columns(self, version: str) -> list[str]:
        """Simple helper function getting all UCDP columns by downloading a
        single event from the specified version.
        """
        response = requests.get(
            self.url_ged.format(version).replace("pagesize=1000", "pagesize=1") + "&page=0"
        )
        response.raise_for_status()
        columns = response.json()["Result"][0].keys()
        return list(columns)

    def _process_time(self, df):
        # dealing with event duration in UCDP:
        # ------------------------------------
        # splitting events with recorded duration of multiple days
        # creating 1 row for each day during the event duration
        # fatalities and event counts are divided by number of rows, so sum stays the same
        # will result in some fractions of fatalities and event counts after aggregation!

        @cache
        def split_time(time_start: date, time_end: date) -> date | list[date]:
            """
            If start and end date are not the same, returns list of days for whole period, otherwise the day.
            """
            if time_start != time_end:
                delta = time_end - time_start
                days = []
                for i in range(delta.days + 1):
                    day = time_start + timedelta(days=i)
                    days.append(day)
                return days
            else:
                return time_start

        # assign dates
        df["time_start"] = df["date_start"].apply(self._convert_timestr)
        df["time_end"] = df["date_end"].apply(self._convert_timestr)
        df = df.loc[df.time_end >= date(self.global_config["start_year"], 1, 1)]
        # get event duration in days based on start and end dates
        df["duration"] = (df["time_end"] - df["time_start"]).apply(lambda x: x.days + 1)
        df["time"] = df.apply(lambda x: split_time(x.time_start, x.time_end), axis=1)

        # columns of counts to divide over multiple rows where duration > 1 day
        adjust_columns = [
            "event_count",
            "deaths_a",
            "deaths_b",
            "deaths_civilians",
            "deaths_unknown",
            "best",
            "high",
            "low",
        ]
        for col in adjust_columns:
            df[col] = df[col] / df.duration
        # create rows for each of the days where event duration is longer than 1 day
        df = df.explode("time")
        #############################################

        # assign year and quarter
        df["year"] = df.time.apply(lambda x: x.year)
        df["quarter"] = df.time.apply(lambda x: math.ceil(x.month / 3) * 3 - 2)
        return df

    def _match_countries(self, df: pd.DataFrame, countries: gpd.GeoDataFrame, column: str = "iso3"):
        """
        Matching priorities:
        1. match based on the country geometries
        2. assign the country provided by dataset, if this is contained in our iso3s

        Args:
            df (pd.DataFrame): Df with (conflict) event data.
            countries (GeoDataFrame): GeoDataFrame with country geometries.
            column (str, optional): String of the iso3 column name in countries.


        Returns:
            Dataframe with iso3 matches added as "iso3" column.
        """
        df["iso3"] = np.nan
        # prio 1: geometry-based matching
        df_geo = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
        for row in countries.itertuples():
            df_country = df_geo.clip(gpd.GeoSeries(row.geometry)) # type: ignore
            df.loc[df_country.index, "iso3"] = getattr(row, column)
        events_dropped = 0

        # prio 2: country information-based matching
        @cache
        def convert_country_iso3(country: str) -> str:
            try:
                assert type(country) is str
                return coco.convert(country, to="ISO3") # type: ignore
            except AssertionError:
                return np.nan # type: ignore

        for idx, row in df.loc[df.iso3.isna()].iterrows():
            iso3 = convert_country_iso3(row.country)
            if iso3 in countries[column]:
                df.loc[idx, "iso3"] = iso3 # type: ignore
            else:
                events_dropped += 1
        self.console.print(events_dropped, "events could not be assigned.")
        return df.copy()

    @staticmethod
    @cache
    def _convert_timestr(timestr: str) -> date:
        time = datetime.strptime(timestr[:10], "%Y-%m-%d").date()
        return time
