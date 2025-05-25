from datetime import date
import time

import pandas as pd
import wbgapi as wb

from base.objects import Dataset


class WBData(Dataset):
    """Handles downloading and preprocessing of WorldBank (WB) data.

    Implements `load_data()` to fetch time series for specified indicators from
    the World Bank API using the `wbgapi` package. It includes retry logic for
    API calls and standardizes the resulting DataFrame.

    Attributes:
        data_key (str): Set to "worldbank".
        local (bool): Set to False, as data is sourced from the World Bank API.
        needs_storage (bool): Set to False.
    """

    data_key: str = "worldbank"
    local: bool = False
    needs_storage: bool = False

    def load_data(self, wb_series: dict[str, str], max_retries: int = 30) -> pd.DataFrame:
        """Downloads specified indicators from the World Bank API.

        Fetches data for the given World Bank series codes for all world economies,
        covering a time range from 5 years before the configured `start_year` up
        to the current year. It implements a retry mechanism in case of API errors.
        The downloaded data is reshaped into a pandas DataFrame indexed by
        ('iso3', 'year'), with columns renamed according to `wb_series` values.
        Validates that all requested series were successfully downloaded.

        Args:
            wb_series (dict[str, str]): A dictionary mapping World Bank series codes
                to desired output column names.
            max_retries (int, optional): The maximum number of times to retry
                the API call in case of failure. Defaults to 30.

        Returns:
            pd.DataFrame: A DataFrame indexed by ('iso3', 'year') containing the
                requested World Bank indicators.
        """
        success = False
        i = 0
        df_wb = None
        while i < max_retries and not success:
            i += 1
            try:
                # load a few years earlier as well for interpolation purposes in case of missing data
                df_wb = wb.data.DataFrame(
                    series=list(wb_series.keys()),
                    economy=wb.region.members("WLD"),  # type: ignore
                    time=range(self.global_config["start_year"] - 5, date.today().year + 1),  # type: ignore
                    columns="series",
                    index=["economy", "time"],
                )
                success = True
            except wb.APIError:
                self.console.print(
                    "Error calling World Bank API, retrying in 30 secs.",
                    f"{i}/{max_retries} retries...",
                )
                time.sleep(30)

        if df_wb is None:
            raise wb.APIError(
                "", msg="Error accessing the World Bank API, could not download data."
            )

        df_wb.index = df_wb.index.set_levels(
            df_wb.index.levels[1].str.removeprefix("YR").astype(int), level=1
        )
        df_wb.index.names = ["iso3", "year"]
        df_wb = df_wb.rename(columns=wb_series)

        # check if all selected indicators are actually present - wbgapi doesn't raise an error
        try:
            assert all([v in df_wb.columns for v in wb_series.values()])
        except AssertionError:
            missing = [v for v in wb_series if wb_series[v] not in df_wb.columns]
            raise AssertionError(
                f"Indicators {missing} not present in downloaded data, check selection."
            )

        return df_wb.sort_index()
