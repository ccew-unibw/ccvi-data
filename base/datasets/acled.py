from datetime import date
import math
import pandas as pd
from tqdm import tqdm

from base.objects import ConfigParser, Dataset, GlobalBaseGrid
from utils.index import get_quarter
from utils.spatial_operations import coords_to_pgid, round_grid


class ACLEDData(Dataset):
    """Handles loading and processing of ACLED conflict event data.

    Implements `load_data()` to ingest an ACLED dataset dump or download the data
    via API (not yet implemented), storing a copy with the relevant columns in the
    processing folder.
    Implements `create_grid_quarter_aggregates()` to transform raw events into
    quarterly aggregates at the grid cell level, including specific violence type
    event and fatality counts.

    Attributes:
        data_key (str): Set to "acled".
        local (bool): Indicates whether to use local ACLED dumps (True) or
            download data via the ACLED API (False).
        acled_available (bool): Flag set to True after data is
            successfully loaded or checked by `load_data`.
        grid (GlobalBaseGrid): GlobalBaseGrid instance, used to load the country
            basemap for coverage matching.
    """

    data_key = "acled"

    def __init__(self, config: ConfigParser, grid: GlobalBaseGrid, local: bool = True):
        """Initializes the ACLED data source.

        Sets the operation mode (local file vs API), defines required data
        keys, and calls the Dataset initializer to setup config and storage.

        Args:
            config (ConfigParser): An initialized ConfigParser instance.
            grid (GlobalBaseGrid): An initialized GlobalBaseGrid instance.
            local (bool, optional): Indicates whether to use local ACLED
                dumps (True) or download data via the ACLED API (False).
        """
        self.grid = grid
        self.local = local
        self.acled_available = False
        super().__init__(config=config)

    def load_data(self):
        """Loads ACLED data, checking for cached processing files first.

        Attempts to load a local ACLED copy from the 'processing' storage
        including the last completed quarter. If not found:
        - If `self.local` is True, loads the raw dump specified in the config.
          Raises an error if the provided ACLED dump does not fully cover the
          latest quarter.
        - If `self.local` is False, currently raises NotImplementedError (API access TBD).
        Since ACLED retroactivly updated data in their database, saves the loaded
        raw/dump data for to the processing storage to keep a record for each quarter.

        Returns:
            pd.DataFrame: The loaded ACLED event data.
        """
        last_quarter_date = get_quarter("last", bounds="end")
        filename = f"acled_{last_quarter_date.year}_Q{int(last_quarter_date.month / 3)}"
        columns = ["YEAR", "EVENT_TYPE", "LATITUDE", "LONGITUDE", "FATALITIES", "EVENT_DATE"]
        try:
            df_acled = self.storage.load("processing", filename=filename)
        except FileNotFoundError:
            if self.local:
                df_acled = pd.read_parquet(self.data_config[self.data_key])
                if df_acled["EVENT_DATE"].max() < last_quarter_date:
                    raise Exception(
                        "preprocessed ACLED data out of date, please provide a version up to "
                        f"{last_quarter_date}."
                    )
            else:
                raise NotImplementedError("ACLED download not yet implemented")
                # TODO: implement the download, which also stores the file
                # TODO: @implement ACLED API query via AA acled key
            self.storage.save(df_acled[columns], "processing", filename=filename)

        # Set an instance attribute for easy checking
        self.acled_available = True
        return df_acled

    def create_grid_quarter_aggregates(
        self,
        df_base: pd.DataFrame,
        df_acled: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculates grid-quarter aggregates from the event level ACLED data.

        Assigns events to grid cells (pgid), calculates quarterly aggregates
        (event counts, fatalities) for armed violence and unrest, merges with the
        base grid structure, and fills missing values based on ACLED coverage
        information.

        Event categorization:
        - Armed violence: "Battles", "Explosions/Remote violence", "Violence against civilians"
        - Unrest: "Protests", "Riots"

        Args:
            df_base (pd.DataFrame): Base data structure for indicator data.
            df_acled (pd.DataFrame): ACLED event-level data from `self.load_data`.

        Returns:
            pd.DataFrame: Dataframe aligned to index grid with quarterly ACLED aggregates.
        """
        fp_preprocessed = self.storage.build_filepath("processing", filename="acled_preprocessed")
        try:
            df = pd.read_parquet(fp_preprocessed)
            last_quarter_date = get_quarter("last")
            if df["time"].max() < last_quarter_date:
                raise FileNotFoundError
            return df
        except FileNotFoundError:
            self.console.print(
                "No preprocessed ACLED data in storage or out of date,"
                + " processing event data..."
            )

            # don't automatically start ACLED download since those are separate step in the
            # indicator logic that should each be performed deliberately
            assert self.acled_available, (
                "ACLED download/data check has not run, check indicator logic"
            )

            # create grid and crop to countries - takes a few minutes
            df_base = self._add_acled_coverage_flag(df_base)

            df_acled.columns = [col.lower() for col in df_acled.columns]
            df_acled = df_acled.loc[
                df_acled["year"] >= df_base.index.get_level_values("year").min()
            ].copy()
            # assign quarter
            df_acled["quarter"] = list(
                df_acled["event_date"].apply(lambda x: math.ceil(x.month / 3))
            )
            # useful for grouping data with sum()
            df_acled["event_count"] = 1

            # add violence type dummies
            armed_violence = [
                "Battles",
                "Explosions/Remote violence",
                "Violence against civilians",
            ]
            unrest = ["Protests", "Riots"]

            df_acled["armed_violence"] = [
                1 if event_type in armed_violence else 0 for event_type in df_acled["event_type"]
            ]
            df_acled["unrest"] = [
                1 if event_type in unrest else 0 for event_type in df_acled["event_type"]
            ]

            # get grid cell for each event
            df_acled["pgid"] = df_acled.apply(
                lambda x: coords_to_pgid(round_grid(x["latitude"]), round_grid(x["longitude"])),
                axis=1,
            )
            # drop assignments outside the index grid
            df_acled = df_acled.loc[
                df_acled["pgid"].isin(df_base.index.get_level_values("pgid").unique())
            ]

            df = self._acled_counts_to_grid(df_base, df_acled)

            # fill NAs with 0 where coverage exists
            df.loc[df["acled_coverage"]] = df.loc[df["acled_coverage"]].fillna(0)
            self.storage.save(df, "processing", "acled_preprocessed")
        return df

    def _acled_counts_to_grid(self, df_data: pd.DataFrame, df_acled: pd.DataFrame) -> pd.DataFrame:
        """Helper method to group ACLED events and merge them onto a base DataFrame.

        Groups the processed `df_acled` by ('pgid', 'year', 'quarter') and sums
        'event_count' and 'fatalities' for:
        1. All events (prefixed with 'acled_').
        2. Unrest events (prefixed with 'unrest_').
        3. Armed violence events (prefixed with 'armed_violence_').
        These aggregated series are then merged onto the `df_data` base data
        structure.

        Args:
            df_data (pd.DataFrame): The base DataFrame, indexed by ('pgid', 'year',
                'quarter'), onto which aggregated ACLED data will be merged.
            df_acled (pd.DataFrame): Preprocessed ACLED event data.

        Returns:
            pd.DataFrame: The `df_data` DataFrame with new columns containing
                the aggregated ACLED statistics.
        """
        acled_grouped = df_acled.groupby(["pgid", "year", "quarter"])[
            ["event_count", "fatalities"]
        ].sum()
        acled_grouped_unrest = (
            df_acled.loc[df_acled.unrest == 1]
            .groupby(["pgid", "year", "quarter"])[["event_count", "fatalities"]]
            .sum()
        )
        acled_grouped_armed_violence = (
            df_acled.loc[df_acled.armed_violence == 1]
            .groupby(["pgid", "year", "quarter"])[["event_count", "fatalities"]]
            .sum()
        )

        # merge to data structure
        df_data = pd.merge(
            df_data,
            acled_grouped.rename(lambda x: "acled_" + x, axis=1),
            left_index=True,
            right_index=True,
            how="left",
        )
        df_data = pd.merge(
            df_data,
            acled_grouped_unrest.rename(lambda x: "unrest_" + x, axis=1),
            left_index=True,
            right_index=True,
            how="left",
        )
        df_data = pd.merge(
            df_data,
            acled_grouped_armed_violence.rename(lambda x: "armed_violence_" + x, axis=1),
            left_index=True,
            right_index=True,
            how="left",
        )
        return df_data

    def _add_acled_coverage_flag(self, df_data: pd.DataFrame) -> pd.DataFrame:
        """Adds a boolean 'acled_coverage' flag to the base data structure.

        This flag indicates whether ACLED data is expected to be available for a
        given grid cell (identified by its matched 'iso3') in a given year.
        It uses a hardcoded dictionary of ACLED country coverage start dates based
        on https://acleddata.com/knowledge-base/country-time-period-coverage/.
        Country names from this dictionary are matched against names in the
        GeoDataFrame (loaded from the files specified in the
        data_config["countries"] entry) to link ISO3 codes with coverage start dates.

        Args:
            df_data (pd.DataFrame): The base DataFrame, indexed by ('pgid', 'year',
                'quarter'), containing 'iso3' and 'time' columns/index levels.

        Returns:
            pd.DataFrame: The input `df_data` with an added boolean
                'acled_coverage' column.
        """
        ### ACLED coverage start dates ###

        # 1/1997
        countries1997 = [
            "Algeria",
            "Angola",
            "Benin",
            "Botswana",
            "Burkina Faso",
            "Burundi",
            "Cameroon",
            "Central African Republic",
            "Chad",
            "Democratic Republic of Congo",
            "Republic of Congo",
            "Cote d'Ivoire",
            "Ivory Coast",
            "Djibouti",
            "Egypt",
            "Equatorial Guinea",
            "Eritrea",
            "eSwatini",
            "Ethiopia",
            "Gabon",
            "Gambia",
            "Ghana",
            "Guinea",
            "Guinea-Bissau",
            "Kenya",
            "Lesotho",
            "Liberia",
            "Libya",
            "Madagascar",
            "Malawi",
            "Mali",
            "Mauritania",
            "Morocco",
            "Mozambique",
            "Namibia",
            "Niger",
            "Nigeria",
            "Rwanda",
            "Senegal",
            "Sierra Leone",
            "Somalia",
            "South Africa",
            "Sudan",
            "Tanzania",
            "Togo",
            "Tunisia",
            "Uganda",
            "Zambia",
            "Zimbabwe",
        ]

        # 1/2010
        countries2010 = [
            "Bangladesh",
            "Cambodia",
            "Laos",
            "Myanmar",
            "Nepal",
            "Pakistan",
            "Sri Lanka",
            "Thailand",
            "Vietnam",
        ]

        # 1/2011
        countries2011 = ["South Sudan"]

        # 1/2015
        countries2015 = ["Saudi Arabia", "Yemen", "Indonesia"]

        # 1/2016
        countries2016 = [
            "United Arab Emirates",
            "Israel",
            "Jordan",
            "Palestine",
            "Lebanon",
            "Oman",
            "Kuwait",
            "Qatar",
            "Bahrain",
            "India",
            "Philippines",
            "Iraq",
            "Turkey",
            "Iran",
        ]

        # 1/2017:
        countries2017 = ["Syria", "Afghanistan"]

        # 1/2018:
        countries2018 = [
            "China",
            "Japan",
            "Mongolia",
            "North Korea",
            "South Korea",
            "Taiwan",
            "Armenia",
            "Azerbaijan",
            "Georgia",
            "Kazakhstan",
            "Kyrgyzstan",
            "Tajikistan",
            "Turkmenistan",
            "Uzbekistan",
            "Albania",
            "Bosnia and Herzegovina",
            "Bulgaria",
            "Croatia",
            "Cyprus",
            "Belarus",
            "Ukraine",
            "Romania",
            "Russia",
            "North Macedonia",
            "Montenegro",
            "Serbia",
            "Moldova",
            "Kosovo",
            "Greece",
            "Malaysia",
            "Puerto Rico",
            "Guadeloupe",
            "Martinique",
            "French Guiana",
            "Curaçao",
            "Aruba",
            "US Virgin Islands",
            "Cayman Islands",
            "Sint Maarten",
            "Turks and Caicos",
            "British Virgin Islands",
            "Caribbean Netherlands (Bonaire, Sint Eustatius, and Saba)",
            "Anguilla",
            "Montserrat",
            "Falkland Islands",
            "Brazil",
            "Mexico",
            "Colombia",
            "Argentina",
            "Peru",
            "Venezuela",
            "Chile",
            "Guatemala",
            "Ecuador",
            "Cuba",
            "Bolivia",
            "Haiti",
            "Dominican Republic",
            "Honduras",
            "Paraguay",
            "El Salvador",
            "Nicaragua",
            "Costa Rica",
            "Panama",
            "Uruguay",
            "Jamaica",
            "Trinidad and Tobago",
            "Guyana",
            "Suriname",
            "Bahamas",
            "Belize",
            "Barbados",
            "Saint Lucia",
            "St. Vincent & Grenadines",
            "Grenada",
            "Antigua & Barbuda",
            "Dominica",
            "Saint Kitts & Nevis",
        ]

        # 1/2020
        countries2020 = [
            "United States",
            "Andorra",
            "Austria",
            "Bailiwick of Guernsey",
            "Bailiwick of Jersey",
            "Belgium",
            "Czech Republic",
            "Denmark",
            "Estonia",
            "Faroe Islands",
            "Finland",
            "France",
            "Germany",
            "Gibraltar",
            "Greenland",
            "Hungary",
            "Iceland",
            "Ireland",
            "Isle of Man",
            "Italy",
            "Latvia",
            "Liechtenstein",
            "Lithuania",
            "Luxembourg",
            "Malta",
            "Monaco",
            "Netherlands",
            "Norway",
            "Poland",
            "Portugal",
            "San Marino",
            "Slovakia",
            "Slovenia",
            "Spain",
            "Sweden",
            "Switzerland",
            "United Kingdom",
            "Vatican City",
            "Bhutan",
            "Brunei",
            "East Timor",
            "Cape Verde",
            "Maldives",
            "Singapore",
            "Comoros",
            "Reunion",
            "Mauritius",
            "Mayotte",
            "Saint Helena, Ascension, and Tristan da Cunha",
            "Sao Tome and Principe",
            "Seychelles",
        ]

        # 1/2021
        countries2021 = [
            "Canada",
            "Bermuda",
            "Saint Pierre and Miquelon",
            "Australia",
            "New Zealand",
            "Fiji",
            "New Caledonia",
            "Vanuatu",
            "Solomon Islands",
            "Papua New Guinea",
            "Micronesia",
            "Guam",
            "Nauru",
            "Marshall Islands",
            "Kiribati",
            "Palau",
            "Northern Mariana Islands",
            "American Samoa",
            "Cook Islands",
            "French Polynesia",
            "Norfolk Island",
            "Niue",
            "Samoa",
            "Tonga",
            "Tokelau",
            "Tuvalu",
            "Pitcairn",
            "Wallis and Futuna",
            "Heard Island and McDonald Islands",
            "Cocos (Keeling) Islands",
            "Christmas Island",
            "US Outlying Minor Islands",
            "Antarctica",
        ]

        acled_coverage_dict = {
            date(1997, 1, 1): countries1997,
            date(2010, 1, 1): countries2010,
            date(2011, 1, 1): countries2011,
            date(2015, 1, 1): countries2015,
            date(2016, 1, 1): countries2016,
            date(2017, 1, 1): countries2017,
            date(2018, 1, 1): countries2018,
            date(2020, 1, 1): countries2020,
            date(2021, 1, 1): countries2021,
        }

        start_date_dict = {}
        for start_date in acled_coverage_dict:
            for country in acled_coverage_dict[start_date]:
                start_date_dict[country] = start_date

        # create df with start dates for later merging
        df_start_dates = pd.DataFrame.from_dict(start_date_dict, orient="index").reset_index()
        df_start_dates.columns = ["country", "coverage_start"]
        # save this (did this once)
        # df_start_dates.sort_values('country').to_csv('data/acled_coverage.csv', index=False)

        # read countries based on GeoBoundaries for matching with start dates and grid
        df_countries = self.grid.basemap[['iso3', 'name']].copy()
        df_countries = df_countries.loc[df_countries["iso3"].isin(df_data["iso3"].unique())]
        df_countries = df_countries.merge(
            df_start_dates, how="left", left_on="name", right_on="country"
        )

        # some country name differences...
        matching_dict = {
            "Bahamas, The": "Bahamas",
            "Bosnia & Herzegovina": "Bosnia and Herzegovina",
            "Burma": "Myanmar",
            "Central African Rep": "Central African Republic",
            "Congo, Dem Rep of the": "Democratic Republic of Congo",
            "Congo, Rep of the": "Republic of Congo",
            "Czechia": "Czech Republic",
            "Gambia, The": "Gambia",
            "Korea, North": "North Korea",
            "Korea, South": "South Korea",
            "Macedonia": "North Macedonia",
            "Solomon Is": "Solomon Islands",
            "Sao Tome & Principe": "Sao Tome and Principe",
            "Swaziland": "eSwatini",
            "Timor-Leste": "East Timor",
            "Trinidad & Tobago": "Trinidad and Tobago",
            "Western Sahara": "Morocco",
        }
        # add coverage start date to df_countries
        for i, r in df_countries.iterrows():
            if r["name"] in matching_dict:
                matched_name = matching_dict[r["name"]]
                df_countries.at[i, "country"] = matched_name
                df_countries.at[i, "coverage_start"] = start_date_dict[matched_name]
        df_countries = df_countries.set_index("iso3").sort_index()

        # add coverage flag to full data structure based on start dates
        # if this breaks, there is probably something misspelled in the country list...
        tqdm.pandas()
        df_data["acled_coverage"] = df_data.progress_apply(
            lambda x: x["time"] >= df_countries.loc[x.iso3].coverage_start, axis=1
        )  # type: ignore - error due to progress_apply
        return df_data
