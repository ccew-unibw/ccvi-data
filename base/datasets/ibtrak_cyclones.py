# add python path to the base directory
import os
import re
import tropycal.tracks as tracks
import ssl
import requests
import pandas as pd
from bs4 import BeautifulSoup

import numpy as np
import warnings

warnings.filterwarnings("ignore")

ssl._create_default_https_context = ssl._create_unverified_context


from base.objects import Dataset, console
from utils.index import get_quarter


def find_var(request, thresh):
    r"""
    Given a request and threshold, returns the variable for plotting. Referenced from ``TrackDataset.gridded_stats()`` and ``TrackPlot.plot_gridded()``. Internal function.

    Parameters
    ----------
    request : str
        Descriptor of the requested plot. Detailed more in the ``TrackDataset.gridded_dataset()`` function.
    thresh : dict
        Dictionary containing thresholds for the plot. Detailed more in the ``TrackDataset.gridded_dataset()`` function.

    Returns
    -------
    thresh : dict
        Returns the thresh dictionary, modified depending on the request.
    varname : str
        String denoting the variable for plotting.
    """

    # Convert command to lowercase
    request = request.lower()

    # Count of number of storms
    if request.find("count") >= 0 or request.find("num") >= 0:
        return thresh, "type"

    if request.find("time") >= 0 or request.find("day") >= 0:
        return thresh, "time"

    # Sustained wind, or change in wind speed
    if request.find("wind") >= 0 or request.find("vmax") >= 0:
        # If change in wind, determine time interval
        if request.find("change") >= 0:
            try:
                thresh["dt_window"] = int(
                    "".join(
                        [
                            c
                            for i, c in enumerate(request)
                            if c.isdigit() and i > request.find("hour") - 4
                        ]
                    )
                )
            except:
                raise RuntimeError("Error: specify time interval (hours)")
            return thresh, "dvmax_dt"
        # Otherwise, sustained wind
        else:
            return thresh, "vmax"

    elif request.find("ace") >= 0:
        return thresh, "ace"
    elif request.find("acie") >= 0:
        return thresh, "acie"

    # Minimum MSLP, or change in MSLP
    elif request.find("pressure") >= 0 or request.find("slp") >= 0:
        # If change in MSLP, determine time interval
        if request.find("change") >= 0:
            try:
                thresh["dt_window"] = int(
                    "".join(
                        [
                            c
                            for i, c in enumerate(request)
                            if c.isdigit() and i > request.find("hour") - 4
                        ]
                    )
                )
            except:
                raise RuntimeError("Error: specify time interval (hours)")
            return thresh, "dmslp_dt"
        # Otherwise, minimum MSLP
        else:
            return thresh, "mslp"

    # Storm motion or heading (vector)
    elif request.find("heading") >= 0 or request.find("motion") >= 0:
        return thresh, ("dx_dt", "dy_dt")

    elif request.find("movement") >= 0 or request.find("speed") >= 0:
        return thresh, "speed"

    # Otherwise, error
    else:
        msg = 'Error: Could not decipher variable. Please refer to documentation for examples on how to phrase the "request" string.'
        raise RuntimeError(msg)


def find_func(request, thresh):
    r"""
    Given a request and threshold, returns the requested function. Referenced from ``TrackDataset.gridded_stats()``. Internal function.

    Parameters
    ----------
    request : str
        Descriptor of the requested plot. Detailed more in the ``TrackDataset.gridded_dataset()`` function.
    thresh : dict
        Dictionary containing thresholds for the plot. Detailed more in the ``TrackDataset.gridded_dataset()`` function.

    Returns
    -------
    thresh : dict
        Returns the thresh dictionary, modified depending on the request.
    func : lambda
        Returns a function to apply to the data.
    """

    print(request)

    # Convert command to lowercase
    request = request.lower()

    # Numpy maximum function
    if request.find("max") == 0 or request.find("latest") == 0:
        return thresh, lambda x: np.nanmax(x)

    # Numpy minimum function
    if request.find("min") == 0 or request.find("earliest") == 0:
        return thresh, lambda x: np.nanmin(x)

    # Numpy average function
    elif request.find("mean") >= 0 or request.find("average") >= 0 or request.find("avg") >= 0:
        # Ensure sample minimum is at least 5 per gridpoint
        thresh["sample_min"] = max([5, thresh["sample_min"]])
        return thresh, lambda x: np.nanmean(x)

    # Numpy percentile function
    elif request.find("percentile") >= 0:
        ptile = int(
            "".join(
                [c for i, c in enumerate(request) if c.isdigit() and i < request.find("percentile")]
            )
        )
        # Ensure sample minimum is at least 5 per gridpoint
        thresh["sample_min"] = max([5, thresh["sample_min"]])
        return thresh, lambda x: np.nanpercentile(x, ptile)

    # Count function
    elif request.find("count") >= 0 or request.find("num") >= 0:
        return thresh, lambda x: len(x)

    # ACE - cumulative function
    elif request.find("ace") >= 0:
        return thresh, lambda x: np.nansum(x)
    elif request.find("acie") >= 0:
        return thresh, lambda x: np.nansum(x)

    # Otherwise, function cannot be identified
    else:
        msg = 'Cannot decipher the function. Please refer to documentation for examples on how to phrase the "request" string.'
        raise RuntimeError(msg)


def construct_title(thresh):
    r"""
    Construct a plot title for ``TrackDataset.gridded_stats()``. Internal function.

    Parameters
    ----------
    thresh : dict
        Dictionary containing thresholds for the plot. Detailed more in the ``TrackDataset.gridded_dataset()`` function.

    Returns
    -------
    thresh : dict
        Returns the thresh dictionary, modified depending on the threshold(s) specified.
    plot_subtitle : str
        String denoting the title for the plot.
    """

    # List containing entry for plot title, later merged into a string
    plot_subtitle = []

    # Symbols for greater/less than or equal to signs
    gteq = "\u2265"
    lteq = "\u2264"

    # Add sample minimum
    if not np.isnan(thresh["sample_min"]):
        plot_subtitle.append(f"{gteq} {thresh['sample_min']} storms/bin")
    else:
        thresh["sample_min"] = 0

    # Add minimum wind speed
    if not np.isnan(thresh["v_min"]):
        plot_subtitle.append(f"{gteq} {thresh['v_min']}kt")
    else:
        thresh["v_min"] = 0

    # Add maximum MSLP
    if not np.isnan(thresh["p_max"]):
        plot_subtitle.append(f"{lteq} {thresh['p_max']}hPa")
    else:
        thresh["p_max"] = 9999

    # Add minimum change in wind speed
    if not np.isnan(thresh["dv_min"]):
        plot_subtitle.append(f"{gteq} {thresh['dv_min']}kt / {thresh['dt_window']}hr")
    else:
        thresh["dv_min"] = -9999

    # Add maximum change in MSLP
    if not np.isnan(thresh["dp_max"]):
        plot_subtitle.append(f"{lteq} {thresh['dp_max']}hPa / {thresh['dt_window']}hr")
    else:
        thresh["dp_max"] = 9999

    # Add maximum change in wind speed
    if not np.isnan(thresh["dv_max"]):
        plot_subtitle.append(f"{lteq} {thresh['dv_max']}kt / {thresh['dt_window']}hr")
    else:
        thresh["dv_max"] = 9999

    # Add minimum change in MSLP
    if not np.isnan(thresh["dp_min"]):
        plot_subtitle.append(f"{gteq} {thresh['dp_min']}hPa / {thresh['dt_window']}hr")
    else:
        thresh["dp_min"] = -9999

    # Combine plot_subtitle into string
    if len(plot_subtitle) > 0:
        plot_subtitle = "\n" + ", ".join(plot_subtitle)
    else:
        plot_subtitle = ""

    # Return modified thresh and plot title
    return thresh, plot_subtitle


def interp_storm(storm_dict, hours=1, dt_window=24, dt_align="middle", method="linear"):
    r"""
    Interpolate a storm dictionary temporally to a specified time resolution. Referenced from ``TrackDataset.filter_storms()``. Internal function.

    Parameters
    ----------
    storm_dict : dict
        Dictionary containing a storm entry.
    hours : int
        Temporal resolution in hours to interpolate storm data to. Default is 1 hour.
    dt_window : int
        Time window in hours over which to calculate temporal change data. Default is 24 hours.
    dt_align : str
        Whether to align the temporal change window as "start", "middle" or "end" of the dt_window time period.
    method : str
        Method by which to interpolate lat & lon coordinates. Options are "linear" (default) or "quadratic".

    Returns
    -------
    dict
        Dictionary containing the updated storm entry.
    """

    # Create an empty dict for the new storm entry
    new_storm = {}

    # Copy over non-list attributes
    for key in storm_dict.keys():
        if not isinstance(storm_dict[key], list):
            new_storm[key] = storm_dict[key]

    # Create an empty list for entries
    for name in ["time", "vmax", "mslp", "lat", "lon", "type"]:
        new_storm[name] = []

    # Convert times to numbers for ease of calculation
    times = mdates.date2num(storm_dict["time"])

    # Convert lat & lons to arrays, and ensure lons are out of 360 degrees
    storm_dict["type"] = np.asarray(storm_dict["type"])
    storm_dict["lon"] = np.array(storm_dict["lon"]) % 360

    def round_datetime(tm, nearest_minute=10):
        discard = timedelta(
            minutes=tm.minute % nearest_minute, seconds=tm.second, microseconds=tm.microsecond
        )
        tm -= discard
        if discard >= timedelta(minutes=int(nearest_minute / 2)):
            tm += timedelta(minutes=nearest_minute)
        return tm

    # Attempt temporal interpolation
    try:
        # Create a list of target times given the requested temporal resolution
        targettimes = np.arange(times[0], times[-1] + hours / 24.0, hours / 24.0)
        targettimes = targettimes[targettimes <= times[-1] + 0.001]

        # Update times
        use_minutes = 10 if hours > (1.0 / 6.0) else hours * 60.0
        new_storm["time"] = [
            round_datetime(t.replace(tzinfo=None), use_minutes)
            for t in mdates.num2date(targettimes)
        ]
        targettimes = mdates.date2num(np.array(new_storm["time"]))

        # Create same-length lists for other things
        new_storm["special"] = [""] * len(new_storm["time"])
        new_storm["extra_obs"] = [0] * len(new_storm["time"])

        # WMO basin. Simple linear interpolation.
        basinnum = np.cumsum(
            [0]
            + [
                1 if storm_dict["wmo_basin"][i + 1] != j else 0
                for i, j in enumerate(storm_dict["wmo_basin"][:-1])
            ]
        )
        basindict = {k: v for k, v in zip(basinnum, storm_dict["wmo_basin"])}
        basininterp = np.round(np.interp(targettimes, times, basinnum)).astype(int)
        new_storm["wmo_basin"] = [basindict[k] for k in basininterp]

        # Interpolate and fill in storm type
        stormtype = [1 if i in constants.TROPICAL_STORM_TYPES else -1 for i in storm_dict["type"]]
        isTROP = np.interp(targettimes, times, stormtype)
        stormtype = [
            1 if i in constants.SUBTROPICAL_ONLY_STORM_TYPES else -1 for i in storm_dict["type"]
        ]
        isSUB = np.interp(targettimes, times, stormtype)
        stormtype = [1 if i == "LO" else -1 for i in storm_dict["type"]]
        isLO = np.interp(targettimes, times, stormtype)
        stormtype = [1 if i == "DB" else -1 for i in storm_dict["type"]]
        isDB = np.interp(targettimes, times, stormtype)
        newtype = np.where(isTROP > 0, "TROP", "EX")
        newtype[newtype == "TROP"] = np.where(isSUB[newtype == "TROP"] > 0, "SUB", "TROP")
        newtype[newtype == "EX"] = np.where(isLO[newtype == "EX"] > 0, "LO", "EX")
        newtype[newtype == "EX"] = np.where(isDB[newtype == "EX"] > 0, "DB", "EX")

        # Interpolate and fill in other variables
        for name in ["vmax", "mslp"]:
            new_storm[name] = np.interp(targettimes, times, storm_dict[name])
            new_storm[name] = np.array(
                [int(round(i)) if not np.isnan(i) else np.nan for i in new_storm[name]]
            )
        for name in ["lat", "lon"]:
            filtered_array = np.array(storm_dict[name])
            new_times = np.array(storm_dict["time"])
            if "linear" not in method:
                converted_hours = np.array(
                    [
                        1 if i.strftime("%H%M") in constants.STANDARD_HOURS else 0
                        for i in storm_dict["time"]
                    ]
                )
                filtered_array = filtered_array[converted_hours == 1]
                new_times = new_times[converted_hours == 1]
            new_times = mdates.date2num(new_times)
            if len(filtered_array) >= 3:
                func = interp.interp1d(new_times, filtered_array, kind=method)
                new_storm[name] = func(targettimes)
                new_storm[name] = np.array(
                    [round(i, 2) if not np.isnan(i) else np.nan for i in new_storm[name]]
                )
            else:
                new_storm[name] = np.interp(targettimes, times, storm_dict[name])
                new_storm[name] = np.array(
                    [int(round(i)) if not np.isnan(i) else np.nan for i in new_storm[name]]
                )

        # Correct storm type by intensity
        newtype[newtype == "TROP"] = [
            ["TD", "TS", "HU", "TY", "ST"][int(i > 34) + int(i > 63)]
            for i in new_storm["vmax"][newtype == "TROP"]
        ]
        newtype[newtype == "SUB"] = [
            ["SD", "SS"][int(i > 34)] for i in new_storm["vmax"][newtype == "SUB"]
        ]
        new_storm["type"] = newtype

        # Calculate change in wind & MSLP over temporal resolution
        new_storm["dvmax_dt"] = [np.nan] + list(
            (new_storm["vmax"][1:] - new_storm["vmax"][:-1]) / hours
        )
        new_storm["dmslp_dt"] = [np.nan] + list(
            (new_storm["mslp"][1:] - new_storm["mslp"][:-1]) / hours
        )

        # Calculate x and y position change over temporal window
        rE = 6.371e3 * 0.539957  # nautical miles
        d2r = np.pi / 180.0
        new_storm["dx_dt"] = [np.nan] + list(
            d2r
            * (new_storm["lon"][1:] - new_storm["lon"][:-1])
            * rE
            * np.cos(d2r * np.mean([new_storm["lat"][1:], new_storm["lat"][:-1]], axis=0))
            / hours
        )
        new_storm["dy_dt"] = [np.nan] + list(
            d2r * (new_storm["lat"][1:] - new_storm["lat"][:-1]) * rE / hours
        )
        new_storm["speed"] = [
            (x**2 + y**2) ** 0.5 for x, y in zip(new_storm["dx_dt"], new_storm["dy_dt"])
        ]

        # Convert change in wind & MSLP to change over specified window
        for name in ["dvmax_dt", "dmslp_dt"]:
            tmp = np.round(
                np.convolve(new_storm[name], [1] * int(dt_window / hours), mode="valid"), 1
            )
            if dt_align == "end":
                new_storm[name] = [np.nan] * (len(new_storm[name]) - len(tmp)) + list(tmp)
            if dt_align == "middle":
                tmp2 = [np.nan] * int((len(new_storm[name]) - len(tmp)) // 2) + list(tmp)
                new_storm[name] = tmp2 + [np.nan] * (len(new_storm[name]) - len(tmp2))
            if dt_align == "start":
                new_storm[name] = list(tmp) + [np.nan] * (len(new_storm[name]) - len(tmp))
            new_storm[name] = list(np.array(new_storm[name]) * (hours))

        # Convert change in position to change over specified window
        for name in ["dx_dt", "dy_dt", "speed"]:
            tmp = np.convolve(
                new_storm[name], [hours / dt_window] * int(dt_window / hours), mode="valid"
            )
            if dt_align == "end":
                new_storm[name] = [np.nan] * (len(new_storm[name]) - len(tmp)) + list(tmp)
            if dt_align == "middle":
                tmp2 = [np.nan] * int((len(new_storm[name]) - len(tmp)) // 2) + list(tmp)
                new_storm[name] = tmp2 + [np.nan] * (len(new_storm[name]) - len(tmp2))
            if dt_align == "start":
                new_storm[name] = list(tmp) + [np.nan] * (len(new_storm[name]) - len(tmp))

        new_storm["dt_window"] = dt_window
        new_storm["dt_align"] = dt_align

        # Return new dict
        return new_storm

    # Otherwise, simply return NaNs
    except:
        for name in new_storm.keys():
            try:
                storm_dict[name]
            except:
                storm_dict[name] = np.ones(len(new_storm[name])) * np.nan
        return storm_dict


def filter_storms(
    self,
    storm=None,
    year_range=None,
    date_range=None,
    thresh={},
    domain=None,
    interpolate_data=False,
    return_keys=True,
):
    r"""
    Filters all storms by various thresholds.

    Parameters
    ----------
    storm : list or str
        Single storm ID or list of storm IDs (e.g., ``'AL012022'``, ``['AL012022','AL022022']``) to search through. If None, defaults to searching through the entire dataset.
    year_range : list or tuple
        List or tuple representing the start and end years (e.g., ``(1950,2018)``). Default is start and end years of dataset.
    date_range : list or tuple
        List or tuple representing the start and end dates as a string in 'month/day' format (e.g., ``('6/1','8/15')``). Default is ``('1/1','12/31')`` or full year.
    thresh : dict
        Keywords include:

        * **sample_min** - minimum number of storms in a grid box for "request" to be applied. For the functions 'percentile' and 'average', 'sample_min' defaults to 5 and will override any value less than 5.
        * **v_min** - minimum wind for a given point to be included in "request".
        * **p_max** - maximum pressure for a given point to be included in "request".
        * **dv_min** - minimum change in wind over dt_window for a given point to be included in "request".
        * **dp_max** - maximum change in pressure over dt_window for a given point to be included in "request".
        * **dt_window** - time window over which change variables are calculated (hours). Default is 24.
        * **dt_align** - alignment of dt_window for change variables -- 'start','middle','end' -- e.g. 'end' for dt_window=24 associates a TC point with change over the past 24 hours. Default is middle.

        Units of all wind variables = kt, and pressure variables = hPa. These are added to the subtitle.
    domain : str
        Geographic domain. Default is entire basin. Please refer to :ref:`options-domain` for available domain options.
    interpolate_data : bool
        Whether to interpolate track data to hourly. Default is False.
    return_keys : bool
        If True, returns a list of storm IDs that match the specified criteria. Otherwise returns a pandas.DataFrame object with all matching data points. Default is True.

    Returns
    -------
    list or pandas.DataFrame
        Check return_keys for more information.
    """

    # Add default year aned date ranges
    if year_range is None:
        year_range = (0, 9999)
    if date_range is None:
        date_range = ("1/1", "12/31")

    # Add interpolation automatically if requested threshold necessitates it
    check_keys = [
        True if i in thresh else False
        for i in ["dv_min", "dv_max", "dp_min", "dp_max", "speed_min", "speed_max"]
    ]
    if True in check_keys:
        interpolate_data = True

    # Update thresh based on input
    default_thresh = {
        "sample_min": 1,
        "p_max": 9999,
        "p_min": 0,
        "v_min": 0,
        "v_max": 9999,
        "dv_min": -9999,
        "dp_max": 9999,
        "dv_max": 9999,
        "dp_min": -9999,
        "speed_max": 9999,
        "speed_min": -9999,
        "dt_window": 24,
        "dt_align": "middle",
    }
    for key in thresh:
        default_thresh[key] = thresh[key]
    thresh = default_thresh

    # Determine domain over which to filter data
    if domain is None:
        lon_min = 0
        lon_max = 360
        lat_min = -90
        lat_max = 90
    else:
        keys = domain.keys()
        check = [False, False, False, False]
        for key in keys:
            if key[0].lower() == "n":
                check[0] = True
                lat_max = domain[key]
            if key[0].lower() == "s":
                check[1] = True
                lat_min = domain[key]
            if key[0].lower() == "e":
                check[2] = True
                lon_max = domain[key]
            if key[0].lower() == "w":
                check[3] = True
                lon_min = domain[key]
        if False in check:
            msg = "Custom domains must be of type dict with arguments for 'n', 's', 'e' and 'w'."
            raise ValueError(msg)
        if lon_max < 0:
            lon_max += 360.0
        if lon_min < 0:
            lon_min += 360.0

    # Determine year and date range
    year_min, year_max = year_range
    date_min, date_max = [dt.strptime(i, "%m/%d") for i in date_range]
    date_max += timedelta(days=1, seconds=-1)

    # Determine if a date falls within the date range
    def date_range_test(t, t_min, t_max):
        if date_min < date_max:
            test1 = t >= t_min.replace(year=t.year)
            test2 = t <= t_max.replace(year=t.year)
            return test1 & test2
        else:
            test1 = t_min.replace(year=t.year) <= t < dt(t.year + 1, 1, 1)
            test2 = dt(t.year, 1, 1) <= t <= t_max.replace(year=t.year)
            return test1 | test2

    # Create empty dictionary to store output in
    points = {}
    for name in ["vmax", "mslp", "type", "lat", "lon", "time", "season", "stormid", "ace"] + [
        "dmslp_dt",
        "dvmax_dt",
        "acie",
        "dx_dt",
        "dy_dt",
        "speed",
    ] * int(interpolate_data):
        points[name] = []

    # Iterate over every storm in TrackDataset
    if storm is not None:
        if isinstance(storm, list):
            if isinstance(storm[0], tuple):
                stormkeys = [self.get_storm_id(s) for s in storm]
            else:
                stormkeys = storm
        elif isinstance(storm, tuple):
            stormkeys = [self.get_storm_id(storm)]
        else:
            stormkeys = [storm]
    else:
        stormkeys = self.keys

    for key in stormkeys:
        # Only interpolate storms within the provided temporal range
        if self.data[key]["year"] <= (year_range[0] - 1) or self.data[key]["year"] >= (
            year_range[-1] + 1
        ):
            continue
        subset_dates = np.array(self.data[key]["time"])[
            np.array([i in constants.TROPICAL_STORM_TYPES for i in self.data[key]["type"]])
        ]
        if len(subset_dates) == 0:
            continue
        verify_dates = [date_range_test(i, date_min, date_max) for i in subset_dates]
        if True not in verify_dates:
            continue

        # Interpolate temporally if requested
        if interpolate_data:
            istorm = interp_storm(
                self.data[key].copy(),
                hours=1,
                dt_window=thresh["dt_window"],
                dt_align=thresh["dt_align"],
            )
            self.data_interp[key] = istorm.copy()
            timeres = 1
        else:
            istorm = self.data[key]
            timeres = 6

        # Iterate over every timestep of the storm
        for i in range(len(istorm["time"])):
            # Filter to only tropical cyclones, and filter by dates & coordiates
            if (
                istorm["type"][i] in constants.TROPICAL_STORM_TYPES
                and lat_min <= istorm["lat"][i] <= lat_max
                and lon_min <= istorm["lon"][i] % 360 <= lon_max
                and year_min <= istorm["time"][i].year <= year_max
                and date_range_test(istorm["time"][i], date_min, date_max)
            ):
                # Append data points
                points["vmax"].append(istorm["vmax"][i])
                points["mslp"].append(istorm["mslp"][i])
                points["type"].append(istorm["type"][i])
                points["lat"].append(istorm["lat"][i])
                points["lon"].append(istorm["lon"][i])
                points["time"].append(istorm["time"][i])
                points["season"].append(istorm["season"])
                points["stormid"].append(key)
                if istorm["vmax"][i] > 34:
                    points["ace"].append(istorm["vmax"][i] ** 2 * 1e-4 * timeres / 6)
                else:
                    points["ace"].append(0)

                # Append separately for interpolated data
                if interpolate_data:
                    points["dvmax_dt"].append(istorm["dvmax_dt"][i])
                    points["acie"].append(
                        [0, istorm["dvmax_dt"][i] ** 2 * 1e-4 * timeres / 6][
                            istorm["dvmax_dt"][i] > 0
                        ]
                    )
                    points["dmslp_dt"].append(istorm["dmslp_dt"][i])
                    points["dx_dt"].append(istorm["dx_dt"][i])
                    points["dy_dt"].append(istorm["dy_dt"][i])
                    points["speed"].append(istorm["speed"][i])

    # Create a DataFrame from the dictionary
    p = pd.DataFrame.from_dict(points)

    # Filter by thresholds
    if thresh["v_min"] > 0:
        p = p.loc[(p["vmax"] >= thresh["v_min"])]
    if thresh["v_max"] < 9999:
        p = p.loc[(p["vmax"] <= thresh["v_max"])]
    if thresh["p_max"] < 9999:
        p = p.loc[(p["mslp"] <= thresh["p_max"])]
    if thresh["p_min"] > 0:
        p = p.loc[(p["mslp"] >= thresh["p_min"])]
    if interpolate_data:
        if thresh["dv_min"] > -9999:
            p = p.loc[(p["dvmax_dt"] >= thresh["dv_min"])]
        if thresh["dp_max"] < 9999:
            p = p.loc[(p["dmslp_dt"] <= thresh["dp_max"])]
        if thresh["dv_max"] < 9999:
            p = p.loc[(p["dvmax_dt"] <= thresh["dv_max"])]
        if thresh["dp_min"] > -9999:
            p = p.loc[(p["dmslp_dt"] >= thresh["dp_min"])]
        if thresh["speed_max"] < 9999:
            p = p.loc[(p["speed"] >= thresh["speed_max"])]
        if thresh["speed_min"] > -9999:
            p = p.loc[(p["speed"] >= thresh["speed_min"])]

    # Determine how to return data
    if return_keys:
        return [g[0] for g in p.groupby("stormid")]
    else:
        return p


def gridded_stats_(
    self,
    request,
    thresh={},
    storm=None,
    year_range=None,
    year_range_subtract=None,
    year_average=False,
    date_range=("1/1", "12/31"),
    binsize=1,
    domain=None,
    ax=None,
    return_array=False,
    cartopy_proj=None,
    **kwargs,
):
    r"""
    Creates a plot of gridded statistics.

    Parameters
    ----------
    request : str
        This string is a descriptor for what you want to plot.
        It will be used to define the variable (e.g. 'wind' --> 'vmax') and the function (e.g. 'maximum' --> np.max()).
        This string is also used as the plot title.

        Variable words to use in request:

        * **wind** - (kt). Sustained wind.
        * **pressure** - (hPa). Minimum pressure.
        * **wind change** - (kt/time). Must be followed by an integer value denoting the length of the time window '__ hours' (e.g., "wind change in 24 hours").
        * **pressure change** - (hPa/time). Must be followed by an integer value denoting the length of the time window '__ hours' (e.g., "pressure change in 24 hours").
        * **storm motion** - (km/hour). Can be followed a length of time window. Otherwise defaults to 24 hours.

        Units of all wind variables are knots and pressure variables are hPa. These are added into the title.

        Function words to use in request:

        * **maximum**
        * **minimum**
        * **average**
        * **percentile** - Percentile must be preceded by an integer [0,100].
        * **number** - Number of storms in grid box satisfying filter thresholds.

        Example usage: "maximum wind change in 24 hours", "50th percentile wind", "number of storms"

    thresh : dict, optional
        Keywords include:

        * **sample_min** - minimum number of storms in a grid box for the request to be applied. For the functions 'percentile' and 'average', 'sample_min' defaults to 5 and will override any value less than 5.
        * **v_min** - minimum wind for a given point to be included in the request.
        * **p_max** - maximum pressure for a given point to be included in the request.
        * **dv_min** - minimum change in wind over dt_window for a given point to be included in the request.
        * **dp_max** - maximum change in pressure over dt_window for a given point to be included in the request.
        * **dt_window** - time window over which change variables are calculated (hours). Default is 24.
        * **dt_align** - alignment of dt_window for change variables -- 'start','middle','end' -- e.g. 'end' for dt_window=24 associates a TC point with change over the past 24 hours. Default is middle.

        Units of all wind variables = kt, and pressure variables = hPa. These are added to the subtitle.

    year_range : list or tuple, optional
        List or tuple representing the start and end years (e.g., (1950,2018)). Default is start and end years of dataset.
    year_range_subtract : list or tuple, optional
        A year range to subtract from the previously specified "year_range". If specified, will create a difference plot.
    year_average : bool, optional
        If True, both year ranges will be computed and plotted as an annual average.
    date_range : list or tuple, optional
        List or tuple representing the start and end dates as a string in 'month/day' format (e.g., ``('6/1','8/15')``). Default is ``('1/1','12/31')`` i.e., the full year.
    binsize : float, optional
        Grid resolution in degrees. Default is 1 degree.
    domain : str, optional
        Domain for the plot. Default is "dynamic". Please refer to :ref:`options-domain` for available domain options.
    ax : axes, optional
        Instance of axes to plot on. If none, one will be generated. Default is none.
    return_array : bool, optional
        If True, returns the gridded 2D array used to generate the plot. Default is False.
    cartopy_proj : ccrs, optional
        Instance of a cartopy projection to use. If none, one will be generated. Default is none.

    Other Parameters
    ----------------
    prop : dict, optional
        Customization properties of plot. Please refer to :ref:`options-prop-gridded` for available options.
    map_prop : dict, optional
        Customization properties of Cartopy map. Please refer to :ref:`options-map-prop` for available options.

    Returns
    -------
    By default, the plot axes is returned. If "return_array" are set to True, a dictionary is returned containing both the axes and data array.

    Notes
    -----
    The following properties are available for customizing the plot, via ``prop``:

    .. list-table::
        :widths: 25 75
        :header-rows: 1

        * - Property
          - Description
        * - plot_values
          - Boolean for whether to plot label values for each gridpoint. Default is False.
        * - smooth
          - Number (in units of sigma) to smooth the data using scipy's gaussian filter. Default is 0 (no smoothing).
        * - cmap
          - Colormap to use for the plot. If string 'category' is passed (default), uses a pre-defined color scale corresponding to the Saffir-Simpson Hurricane Wind Scale.
        * - clevs
          - Contour levels for the plot. Default is minimum and maximum values in the grid.
        * - left_title
          - Title string for the left side of the plot. Default is the string passed via the 'request' keyword argument.
        * - right_title
          - Title string for the right side of the plot. Default is 'All storms'.
    """

    # Retrieve kwargs

    prop = kwargs.pop("prop", {})
    map_prop = kwargs.pop("map_prop", {})

    default_prop = {"smooth": None}
    for key in prop.keys():
        default_prop[key] = prop[key]
    prop = default_prop

    # Update thresh based on input
    default_thresh = {
        "sample_min": np.nan,
        "p_max": np.nan,
        "v_min": np.nan,
        "dv_min": np.nan,
        "dp_max": np.nan,
        "dv_max": np.nan,
        "dp_min": np.nan,
        "dt_window": 24,
        "dt_align": "middle",
    }
    for key in thresh:
        default_thresh[key] = thresh[key]
    thresh = default_thresh

    # Retrieve the requested function, variable for computing stats, and plot title. These modify thresh if necessary.
    thresh, func = find_func(request, thresh)
    thresh, varname = find_var(request, thresh)
    thresh, plot_subtitle = construct_title(thresh)
    if storm is not None:
        thresh["sample_min"] = 1
        plot_subtitle = ""

    # Determine whether request includes a vector (i.e., TC motion vector)
    VEC_FLAG = isinstance(varname, tuple)

    # Determine year range of plot
    def get_year_range(y_r):
        start_year = self.data[self.keys[0]]["year"]
        end_year = self.data[self.keys[-1]]["year"]
        if y_r is None:
            new_y_r = (start_year, end_year)
        else:
            if not isinstance(y_r, (list, tuple)):
                msg = '"year_range" and "year_range_subtract" must be of type list or tuple.'
                raise ValueError(msg)
            if year_range_subtract is not None and len(year_range_subtract) != 2:
                msg = '"year_range" and "year_range_subtract" must contain 2 elements.'
                raise ValueError(msg)
            new_y_r = (max((start_year, min(y_r))), min((end_year, max(y_r))))
        return new_y_r

    year_range = get_year_range(year_range)

    # Start date in numpy datetime64
    startdate = np.datetime64(
        "2000-" + "-".join([f"{int(d):02}" for d in date_range[0].split("/")])
    )

    # Determine year range to subtract, if making a difference plot
    if year_range_subtract is not None:
        year_range_subtract = get_year_range(year_range_subtract)

    # ---------------------------------------------------------------------------------------------------

    # Perform analysis either once or twice depending on year_range_subtract
    if year_range_subtract is None:
        years_analysis = [year_range]
    else:
        years_analysis = [year_range, year_range_subtract]
    grid_x_years = []
    grid_y_years = []
    grid_z_years = []

    for year_range_temp in years_analysis:
        # Obtain all data points for the requested threshold and year/date ranges. Interpolate data to hourly.
        print("--> Getting filtered storm tracks")
        points = self.filter_storms(
            storm,
            year_range_temp,
            date_range,
            thresh=thresh,
            interpolate_data=True,
            return_keys=False,
        )

        return points

        # Round lat/lon points down to nearest bin
        def to_bin(x):
            return np.floor(x / binsize) * binsize

        points["latbin"] = points.lat.map(to_bin)
        points["lonbin"] = points.lon.map(to_bin)

        # ---------------------------------------------------------------------------------------------------

        # Group by latbin,lonbin,stormid
        print("--> Grouping by lat/lon/storm")
        groups = points.groupby(["latbin", "lonbin", "stormid", "season"])

        # Loops through groups, and apply stat func to storms
        # Constructs a new dataframe containing the lat/lon bins, storm ID and the plotting variable
        new_df = {"latbin": [], "lonbin": [], "stormid": [], "season": [], varname: []}
        for g in groups:
            # Apply function to all time steps in which a storm tracks within a gridbox
            if VEC_FLAG:
                new_df[varname].append([func(g[1][v].values) for v in varname])
            elif varname == "date":
                new_df[varname].append(
                    func(
                        [
                            date_diff(dt(2000, t.month, t.day), startdate)
                            for t in pd.DatetimeIndex(g[1][varname].values)
                        ]
                    )
                )
            else:
                new_df[varname].append(func(g[1][varname].values))
            new_df["latbin"].append(g[0][0])
            new_df["lonbin"].append(g[0][1])
            new_df["stormid"].append(g[0][2])
            new_df["season"].append(g[0][3])
        new_df = pd.DataFrame.from_dict(new_df)

        # ---------------------------------------------------------------------------------------------------

        # Group again by latbin,lonbin
        # Construct two 1D lists: zi (grid values) and coords, that correspond to the 2D grid
        groups = new_df.groupby(["latbin", "lonbin"])

        # Apply the function to all storms that pass through a gridpoint
        if VEC_FLAG:
            zi = [
                [func(v) for v in zip(*g[1][varname])]
                if len(g[1]) >= thresh["sample_min"]
                else [np.nan] * 2
                for g in groups
            ]
        elif varname == "date":
            zi = [
                func(g[1][varname]) if len(g[1]) >= thresh["sample_min"] else np.nan for g in groups
            ]
            zi = [mdates.date2num(startdate + z) for z in zi]
        else:
            zi = [
                func(g[1][varname]) if len(g[1]) >= thresh["sample_min"] else np.nan for g in groups
            ]

        # Construct a 1D array of coordinates
        coords = [g[0] for g in groups]

        # Construct a 2D longitude and latitude grid, using the specified binsize resolution
        if prop["smooth"] is not None:
            all_lats = [
                (round(l / binsize) * binsize)
                for key in self.data.keys()
                for l in self.data[key]["lat"]
            ]
            all_lons = [
                (round(l / binsize) * binsize) % 360
                for key in self.data.keys()
                for l in self.data[key]["lon"]
            ]
            xi = np.arange(min(all_lons) - binsize, max(all_lons) + 2 * binsize, binsize)
            yi = np.arange(min(all_lats) - binsize, max(all_lats) + 2 * binsize, binsize)
            if self.basin == "all":
                xi = np.arange(0, 360 + binsize, binsize)
                yi = np.arange(-90, 90 + binsize, binsize)
        else:
            xi = np.arange(
                np.nanmin(points["lonbin"]) - binsize,
                np.nanmax(points["lonbin"]) + 2 * binsize,
                binsize,
            )
            yi = np.arange(
                np.nanmin(points["latbin"]) - binsize,
                np.nanmax(points["latbin"]) + 2 * binsize,
                binsize,
            )
        grid_x, grid_y = np.meshgrid(xi, yi)
        grid_x_years.append(grid_x)
        grid_y_years.append(grid_y)

        # Construct a 2D grid for the z value, depending on whether vector or scalar quantity
        if VEC_FLAG:
            grid_z_u = np.ones(grid_x.shape) * np.nan
            grid_z_v = np.ones(grid_x.shape) * np.nan
            for c, z in zip(coords, zi):
                grid_z_u[np.where((grid_y == c[0]) & (grid_x == c[1]))] = z[0]
                grid_z_v[np.where((grid_y == c[0]) & (grid_x == c[1]))] = z[1]
            grid_z = [grid_z_u, grid_z_v]
        else:
            grid_z = np.ones(grid_x.shape) * np.nan
            for c, z in zip(coords, zi):
                grid_z[np.where((grid_y == c[0]) & (grid_x == c[1]))] = z

        # Set zero values to nan's if necessary
        if varname == "type":
            grid_z[np.where(grid_z == 0)] = np.nan

        # Add to list of grid_z's
        grid_z_years.append(grid_z)

    # ---------------------------------------------------------------------------------------------------

    # Calculate difference between plots, if specified
    if len(grid_z_years) == 2:
        # Determine whether to use averages
        if year_average:
            years_listed = len(range(year_range[0], year_range[1] + 1))
            grid_z_years[0] = grid_z_years[0] / years_listed
            years_listed = len(range(year_range_subtract[0], year_range_subtract[1] + 1))
            grid_z_years[1] = grid_z_years[1] / years_listed

        # Construct DataArrays
        grid_z_1 = xr.DataArray(
            np.nan_to_num(grid_z_years[0]),
            coords=[grid_y_years[0].T[0], grid_x_years[0][0]],
            dims=["lat", "lon"],
        )
        grid_z_2 = xr.DataArray(
            np.nan_to_num(grid_z_years[1]),
            coords=[grid_y_years[1].T[0], grid_x_years[1][0]],
            dims=["lat", "lon"],
        )

        # Compute difference grid
        grid_z = grid_z_1 - grid_z_2

        # Reconstruct lat & lon grids
        xi = grid_z.lon.values
        yi = grid_z.lat.values
        grid_z = grid_z.values
        grid_x, grid_y = np.meshgrid(xi, yi)

        # Determine NaNs
        grid_z_years[0][np.isnan(grid_z_years[0])] = -9999
        grid_z_years[1][np.isnan(grid_z_years[1])] = -8999
        grid_z_years[0][grid_z_years[0] != -9999] = 0
        grid_z_years[1][grid_z_years[1] != -8999] = 0
        grid_z_1 = xr.DataArray(
            np.nan_to_num(grid_z_years[0]),
            coords=[grid_y_years[0].T[0], grid_x_years[0][0]],
            dims=["lat", "lon"],
        )
        grid_z_2 = xr.DataArray(
            np.nan_to_num(grid_z_years[1]),
            coords=[grid_y_years[1].T[0], grid_x_years[1][0]],
            dims=["lat", "lon"],
        )
        grid_z_check = (grid_z_1 - grid_z_2).values
        grid_z[grid_z_check == -1000] = np.nan
        print(np.nanmin(grid_z))

    else:
        # Determine whether to use averages
        if year_average:
            years_listed = len(range(year_range[0], year_range[1] + 1))
            grid_z = grid_z / years_listed

    # Create instance of plot object
    try:
        self.plot_obj
    except:
        self.plot_obj = TrackPlot()

    # Create cartopy projection using basin
    if domain is None:
        domain = self.basin
    if cartopy_proj is None:
        if max(points["lon"]) > 150 or min(points["lon"]) < -150:
            self.plot_obj.create_cartopy(proj="PlateCarree", central_longitude=180.0)
        else:
            self.plot_obj.create_cartopy(proj="PlateCarree", central_longitude=0.0)

    # Format left title for plot
    endash = "\u2013"
    dot = "\u2022"
    title_L = request.lower()
    for name in ["wind", "vmax"]:
        title_L = title_L.replace(name, "wind (kt)")
    for name in ["pressure", "mslp"]:
        title_L = title_L.replace(name, "pressure (hPa)")
    for name in ["heading", "motion"]:
        title_L = title_L.replace(name, f"heading (kt) over {thresh['dt_window']} hours")
    for name in ["speed", "movement"]:
        title_L = title_L.replace(name, f"forward speed (kt) over {thresh['dt_window']} hours")
    if request.find("change") >= 0:
        title_L = title_L + f", {thresh['dt_align']}"
    title_L = title_L[0].upper() + title_L[1:] + plot_subtitle

    # Format right title for plot
    if storm is not None:
        if isinstance(storm, list):
            title_R = "Storm Composite"
        else:
            if isinstance(storm, str):
                storm = basin.get_storm_tuple(storm)
            title_R = f"{storm[0]} {storm[1]}"
    else:
        date_range = [dt.strptime(d, "%m/%d").strftime("%b/%d") for d in date_range]
        if np.subtract(*year_range) == 0:
            y_r_title = f"{year_range[0]}"
        else:
            y_r_title = f"{year_range[0]} {endash} {year_range[1]}"
        add_avg = " year-avg" if year_average else ""
        if year_range_subtract is None:
            title_R = f"{date_range[0].replace('/', ' ')} {endash} {date_range[1].replace('/', ' ')} {dot} {y_r_title}{add_avg}"
        else:
            if np.subtract(*year_range_subtract) == 0:
                y_r_s_title = f"{year_range_subtract[0]}"
            else:
                y_r_s_title = f"{year_range_subtract[0]} {endash} {year_range_subtract[1]}"
            title_R = f"{date_range[0].replace('/', ' ')} {endash} {date_range[1].replace('/', ' ')}\n{y_r_title}{add_avg} minus {y_r_s_title}{add_avg}"
    prop["title_L"], prop["title_R"] = title_L, title_R

    # Change the masking for variables that go out to zero near the edge of the data
    if prop["smooth"] is not None:
        # Replace NaNs with zeros to apply Gaussian filter
        grid_z_zeros = grid_z.copy()
        grid_z_zeros[np.isnan(grid_z)] = 0
        initial_mask = grid_z.copy()  # Save initial mask
        initial_mask[np.isnan(grid_z)] = -9999
        grid_z_zeros = gfilt(grid_z_zeros, sigma=prop["smooth"])

        if len(grid_z_years) == 2:
            # grid_z_1_zeros = np.asarray(grid_z_1)
            # grid_z_1_zeros[grid_z_1==-9999]=0
            # grid_z_1_zeros = gfilt(grid_z_1_zeros,sigma=prop['smooth'])

            # grid_z_2_zeros = np.asarray(grid_z_2)
            # grid_z_2_zeros[grid_z_2==-8999]=0
            # grid_z_2_zeros = gfilt(grid_z_2_zeros,sigma=prop['smooth'])
            # grid_z_zeros = grid_z_1_zeros - grid_z_2_zeros
            # test_zeros = (grid_z_1_zeros<.02*np.nanmax(grid_z_1_zeros)) & (grid_z_2_zeros<.02*np.nanmax(grid_z_2_zeros))
            pass

        elif varname not in [("dx_dt", "dy_dt"), "speed", "mslp"]:
            # Apply cutoff at 2% of maximum
            test_zeros = grid_z_zeros < 0.02 * np.amax(grid_z_zeros)
            grid_z_zeros[test_zeros] = -9999
            initial_mask = grid_z_zeros.copy()

        grid_z_zeros[initial_mask == -9999] = np.nan
        grid_z = grid_z_zeros.copy()

    # Plot gridded field
    plot_ax = self.plot_obj.plot_gridded_wrapper(
        grid_x, grid_y, grid_z, varname, VEC_FLAG, domain, ax=ax, prop=prop, map_prop=map_prop
    )

    # Format grid into xarray if specified
    if return_array:
        arr = xr.DataArray(
            np.nan_to_num(grid_z), coords=[grid_y.T[0], grid_x[0]], dims=["lat", "lon"]
        )
        return arr

    # Return axis
    if return_array:
        return {"ax": plot_ax, "array": arr}
    else:
        return plot_ax


def to_bin(x):
    import numpy as np

    binsize = 0.5
    if x > 180:
        x = -180 + x - 180
    return (binsize / 2) + np.floor(x / binsize) * binsize


def cyclons_to_grid(binsize=0.5, datefrom=None, dateto=None, ibtrak_url="", temp_cyclons_csv=""):
    print("downloading ibtrak", flush=True)
    import requests

    # if not os.path.exists(temp_csv):
    url = ibtrak_url
    if not os.path.exists(temp_cyclons_csv):
        response = requests.get(url)
        open(temp_cyclons_csv, "wb").write(response.content)

    ibtracs = tracks.TrackDataset(
        basin="all",
        source="ibtracs",
        ibtracs_mode="jtwc_neumann",
        catarina=True,
        ibtracs_url=temp_cyclons_csv,
    )

    points = gridded_stats_(
        ibtracs,
        request="maximum wind",
        year_range=(datefrom.year, dateto.year),
        return_array=True,
        binsize=binsize,
    )

    return points


def get_latest_ibtracs_url():
    ibtracs_url = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
    # Fetch the HTML content of the base URL
    response = requests.get(ibtracs_url)

    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all the anchor tags that contain links to directories
    links = soup.find_all("a")

    # Extract directories and filter out the ones that aren't directories
    directories = [link.get("href") for link in links if re.match(r"v\d{2}r\d{2}", link.get("href"))]
    
    # Assuming the latest directory is the last one in alphabetical order
    if directories:
        latest_directory = directories[-1]
        latest_directory = latest_directory.replace("/", "")
        return f"https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/{latest_directory}/access/csv/ibtracs.ALL.list.{latest_directory}.csv"

    else:
        return None


class IBTRAKData(Dataset):
    """Handles loading and processing of IBTRAK event data.

    Supports loading from local preprocessed files or via API.
    Includes methods for aggregating event data to the
    grid-quarter level.

    Attributes:
        data_key (str): Set to "IBTRAK".
        local (bool): Indicates whether to use local ibtrak dumps (True) or
            download data via the ibtrak API (False).
        dataset_available (bool): Flag set to True after data is
            successfully loaded or checked by `load_data`.
    """

    data_key = "ibtrak_cyclones"

    def __init__(self, local: bool = True, *args, **kwargs):
        """Initializes the ibtrak data source.

        Sets the operation mode (local file vs API) and calls the Dataset
        initializer to setup config and storage.

        Args:
            config (ConfigParser): An initialized ConfigParser instance.
            local (bool, optional): Indicates whether to use local ibtrak
                dumps (True) or download data via the ibtrak API (False).
        """
        self.local = local
        self.dataset_available = False
        self.data_keys = [self.data_key, "countries"]
        super().__init__(*args, **kwargs)

    def download_data(self):
        """Downloads data from the API.

        Downloads the data from the API and saves it to the processing
        storage. The filename is based on the current date and time.

        Returns:
            str: The filename of the downloaded data.
        """
        ibtrak_url = get_latest_ibtracs_url()
        fromdate = "2000Q1"

        todate = self.last_quarter_date
        if not os.path.exists(self.storage.storage_paths["processing"]):
            os.makedirs(self.storage.storage_paths["processing"])

        temp_cyclons_csv = f"{self.storage.storage_paths['processing']}/temporary_cyclons_tracks_{pd.to_datetime(fromdate).to_period('Q')}_{pd.to_datetime(todate).to_period('Q')}.csv"

        df_events = cyclons_to_grid(
            binsize=1,
            datefrom=pd.to_datetime(fromdate),
            dateto=pd.to_datetime(todate),
            ibtrak_url=ibtrak_url,
            temp_cyclons_csv=temp_cyclons_csv,
        )

        print("--> mapping to bins")
        full_historical_events_path = (
            f"{self.storage.storage_paths['processing']}/full_historical_events_path.parquet"
        )
        df_events.to_parquet(full_historical_events_path)

        df_events["quarter"] = pd.to_datetime(df_events["time"]).dt.to_period("Q")

        df_events["latbin"] = df_events.lat.map(to_bin)
        df_events["lonbin"] = df_events.lon.map(to_bin)

        df_events.rename(columns={"latbin": "LATITUDE", "lonbin": "LONGITUDE"}, inplace=True)

        df_events["count"] = 1

        # columns to uppercase
        df_events.columns = [col.upper() for col in df_events.columns]

        # reanme LAT with LATITUDE and LON with LONGITUDE

        df_events["EVENT_DATE"] = pd.to_datetime(df_events["TIME"])
        df_events["YEAR"] = df_events["EVENT_DATE"].dt.year

        df_events["EVENT_TYPE"] = "cyclones"

        # "YEAR", "EVENT_TYPE", "LATITUDE", "LONGITUDE", "COUNT", "EVENT_DATE"
        # self.storage.save(df_events[self.columns], "processing", filename=self.filename)
        # beacuse those are the events I keep all the columns
        self.storage.save(df_events, "processing", filename=self.filename)

    def load_data(self):
        """Loads ibtrak data, checking for cached processing files first.

        Attempts to load a local ibtrak copy from the 'processing' storage
        including the last completed quarter. If not found:
        - If `self.local` is True, loads the raw dump specified in the config.
          Raises an error if the provided ibtrak dump does not fully cover the
          latest quarter.
        - If `self.local` is False, currently raises NotImplementedError (API access TBD).
        Saves the loaded raw/dump data to the processing storage.

        Returns:
            pd.DataFrame: The loaded ibtrak event data.
        """
        self.last_quarter_date = get_quarter("last", bounds="end")
        self.filename = (
            f"ibtrak_{self.last_quarter_date.year}_Q{int(self.last_quarter_date.month / 3)}"
        )
        self.columns = ["YEAR", "EVENT_TYPE", "LATITUDE", "LONGITUDE", "COUNT", "EVENT_DATE"]
        try:
            df_event_level = self.storage.load("processing", filename=self.filename)
        except FileNotFoundError:
            if self.local:
                df_event_level = pd.read_parquet(self.data_config[self.data_key])
                if df_event_level["EVENT_DATE"].max() < self.last_quarter_date:
                    raise Exception(
                        "preprocessed ibtrak data out of date, please provide a version up to "
                        f"{self.last_quarter_date}."
                    )
            else:
                self.download_data()
                self.dataset_available = True
                df_event_level = self.storage.load("processing", filename=self.filename)
                return df_event_level

        # Set an instance attribute for easy checking
        self.dataset_available = True
        return df_event_level

    def create_grid_quarter_aggregates(
        self,
        df_base: pd.DataFrame,
        df_event_level: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculates grid-quarter aggregates from the event level ibtrak data.

        Assigns events to grid cells (pgid), calculates quarterly aggregates
        (event counts, fatalities) for armed violence and unrest, merges with the
        base grid structure, and fills missing values based on ibtrak coverage
        information.

        Args:
            df_base (pd.DataFrame): Base data structure for indicator data.
            df_event_level (pd.DataFrame): ibtrak event-level data from `self.load_data`.

        Returns:
            pd.DataFrame: Dataframe aligned to index grid with quarterly ibtrak aggregates.
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
                "No preprocessed ibtrak data in storage or out of date,"
                + " processing event data..."
            )

            # don't automatically start ibtrak download since those are separate step in the
            # indicator logic that should each be performed deliberately
            assert self.dataset_available, (
                "ibtrak download/data check has not run, check indicator logic"
            )

            # quarter as number 1,2,3,4
            df_event_level["QUARTER"] = df_event_level["EVENT_DATE"].dt.quarter

            # drop duplicates
            df_event_level = df_event_level.drop_duplicates(
                subset=["LATITUDE", "LONGITUDE", "YEAR", "QUARTER", "STORMID"]
            )

            # aggregate
            df_event_level_aggregated = (
                df_event_level[["LATITUDE", "LONGITUDE", "YEAR", "QUARTER", "COUNT"]]
                .groupby(["LATITUDE", "LONGITUDE", "YEAR", "QUARTER"])
                .sum()
            )
            df_event_level_aggregated = df_event_level_aggregated.reset_index()
            df_base = df_base.reset_index()
            df = df_base.merge(
                df_event_level_aggregated,
                left_on=["year", "quarter", "lat", "lon"],
                right_on=["YEAR", "QUARTER", "LATITUDE", "LONGITUDE"],
                how="left",
            )

            df = df[["pgid", "year", "quarter", "lat", "lon", "COUNT"]]

            df["time"] = df["year"].astype(str) + "Q" + df["quarter"].astype(str)
            df["time"] = pd.PeriodIndex(df["time"], freq="Q")
            # convert to datetime
            df["time"] = df["time"].dt.to_timestamp()

            "lowercase all columns"
            df.columns = [col.lower() for col in df.columns]
            df = df.fillna(0)

            df.to_parquet(fp_preprocessed)
        return df


# test class
if __name__ == "__main__":
    from base.objects import ConfigParser

    config = ConfigParser()

    # Example usage
    ibtrak_data = IBTRAKData(local=False, config=config)

    df_event_level = ibtrak_data.load_data()
    print(df_event_level.head())
