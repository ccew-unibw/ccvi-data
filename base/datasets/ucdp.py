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

    Implements `load_data()` to download or incrementally update UCDP event data,
    managing both official GED releases and monthly candidate versions from the UCDP API.
    Implements `preprocess_data()` to standardize the raw event data, which includes
    matching events to country ISO3 codes and exploding multi-day events into
    daily records with corresponding quarterly assignments.

    Attributes:
        data_key (str): Set to "ucdp".
        local (bool): Set to False, as data is sourced from the UCDP API.
        url_ged (str): The base API endpoint URL for the UCDP GED datasets.
    """

    data_key = "ucdp"
    local = False
    url_ged: str = "https://ucdpapi.pcr.uu.se/api/gedevents/{}?pagesize=1000"

    def load_data(self) -> Any:
        """Loads UCDP event data, downloading or updating as necessary.

        This method first attempts to load a cached raw UCDP DataFrame from the
        processing directory. If found, it calls `_update_ucdp` to check for and
        fetch only newer data versions. If no cached file exists or if
        regeneration is forced, it performs a full download of all available
        versions via `_download_ucpd()`. The final, complete DataFrame is then
        saved to the processing cache.

        Returns:
            pd.DataFrame: A DataFrame containing the complete and up-to-date
                UCDP GED event data.
        """
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
        """Preprocesses raw UCDP event data for future processing.

        This method checks for a cached version of the preprocessed data and
        re-runs only if the input data is newer. The preprocessing includes
        matching events to standard ISO3 country codes and transforming event
        date ranges into daily records with year and quarter assignments to avoid
        be able to cleanly split the data each quarter. The final preprocessed
        DataFrame is cached to the processing directory.

        Args:
            df (pd.DataFrame): The raw UCDP event data from `load_data()`.
            grid (GlobalBaseGrid): An initialized GlobalBaseGrid instance, used
                for accessing the country basemap for spatial matching.

        Returns:
            pd.DataFrame: The preprocessed UCDP data with standardized time
                and country identifiers added.
        """
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

        This method determines the latest available official (GED) and candidate
        data versions online. It then compares these to the versions present in
        the input `df`, downloads only the missing versions, resolves any potential
        duplicate event IDs between old and new data, and returns a new,
        consolidated DataFrame.

        Args:
            df (pd.DataFrame): Existing dataframe with UCDP GED and Candidate data
                including a 'version' column.

        Returns:
            pd.DataFrame: The consolidated DataFrame containing both the retained
                old data and the newly downloaded data.
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
        """Performs a full download of all available UCDP GED and candidate data.

        Determines the latest official GED and candidate versions available
        from the API, downloads the latest GED release and iteratively downloads
        all subsequent monthly candidate releases, handling de-duplication of
        event IDs between versions to ensure a clean final dataset.

        Returns:
            pd.DataFrame: A comprehensive DataFrame of currently available UCDP
                event data.
        """
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

    def _handle_duplicate_ids(self, df_base: pd.DataFrame, df_update: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicate IDs in among df_base and df_update from df_base.

        Args:
            df_base (pd.DataFrame): The base DataFrame.
            df_update (pd.DataFrame): The DataFrame containing updated records.

        Returns:
            pd.DataFrame: The `df_base` DataFrame with duplicate ID rows removed.
        """
        duplicate_ids = [i for i in df_update["id"] if i in df_base["id"].unique()]
        df_base = df_base[~df_base["id"].isin(duplicate_ids)].copy()
        return df_base

    def _download_version(self, version: str, max_retries: int = 5) -> pd.DataFrame:
        """Downloads all pages of data for a specified UCDP version from the API.

        Makes get request to UCDP API and iterates through all pages, reading
        data as json, concatenating the results and converting it to a pd.DataFrame.
        Includes retries after brief sleep periods in case of errors for individual
        pages.

        Args:
            version (str): UCDP GED or candidate version to download.
            max_retries (int, optional): Maximum number of retries if individual
                pages fail to download. Defaults to 5.

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
        """Simple helper function to read all available UCDP data columns.

        Downloading a single event from the specified version and reads its columns.

        Args:
            version (str): The UCDP version to inspect.

        Returns:
            list[str]: A list of all column names.
        """
        response = requests.get(
            self.url_ged.format(version).replace("pagesize=1000", "pagesize=1") + "&page=0"
        )
        response.raise_for_status()
        columns = response.json()["Result"][0].keys()
        return list(columns)

    def _process_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes event time by exploding multi-day events into daily records.

        This method processes UCDP event start and end dates. For any event
        spanning multiple days, it "explodes" the single row into multiple daily
        rows, one for each day of the event's duration. Fatality counts and an
        'event_count' are distributed evenly across these new daily records,
        allowing for future aggregation to arbitrary time spans. Also assigns a
        'year' and 'quarter' to each daily record.

        Note: results in fractions of fatalities and event counts after aggregation!

        Args:
            df (pd.DataFrame): The UCDP DataFrame with 'date_start' and 'date_end'
                columns.

        Returns:
            pd.DataFrame: A DataFrame with daily event records and corresponding
                'year' and 'quarter' assignments.
        """

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
        df["time"] = df.apply(lambda x: split_time(x.time_start, x.time_end), axis=1)  # type: ignore

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

        # assign year and quarter
        df["year"] = df.time.apply(lambda x: x.year)
        df["quarter"] = df.time.apply(lambda x: math.ceil(x.month / 3) * 3 - 2)
        return df

    def _match_countries(self, df: pd.DataFrame, countries: gpd.GeoDataFrame, column: str = "iso3"):
        """Assigns a standard ISO3 country code to each UCDP event.

        Matching process:
        1. match based on location and a spatial join the country geometries
        2. assign the country provided by dataset, if this is contained in our iso3s

        Prints number of event-days it is unable to assign this way.

         Args:
            df (pd.DataFrame): The DataFrame containing UCDP events.
            countries (gpd.GeoDataFrame): A GeoDataFrame with country shapes.
            column (str, optional): The name of the ISO3 column in the `countries`
                GeoDataFrame. Defaults to "iso3".

        Returns:
            pd.DataFrame: The input DataFrame with an updated 'iso3' column.
        """
        df["iso3"] = np.nan
        # prio 1: geometry-based matching
        df_geo = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
        for row in countries.itertuples():
            df_country = df_geo.clip(gpd.GeoSeries(row.geometry))  # type: ignore
            df.loc[df_country.index, "iso3"] = getattr(row, column)
        events_dropped = 0

        # prio 2: country information-based matching
        @cache
        def convert_country_iso3(country: str) -> str:
            try:
                assert type(country) is str
                return coco.convert(country, to="ISO3")  # type: ignore
            except AssertionError:
                return np.nan  # type: ignore

        for idx, row in df.loc[df.iso3.isna()].iterrows():
            iso3 = convert_country_iso3(row.country)
            if iso3 in countries[column]:
                df.loc[idx, "iso3"] = iso3  # type: ignore
            else:
                events_dropped += 1
        self.console.print(events_dropped, "events could not be assigned.")
        return df.copy()

    @staticmethod
    @cache
    def _convert_timestr(timestr: str) -> date:
        """A cached static helper method to convert a UCDP date string to a date object.

        Args:
            timestr (str): The date string to convert.

        Returns:
            date: The corresponding `datetime.date` object.
        """
        time = datetime.strptime(timestr[:10], "%Y-%m-%d").date()
        return time
