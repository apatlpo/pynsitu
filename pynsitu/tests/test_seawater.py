import numpy as np
import pandas as pd

import pytest
import numpy.testing as npt

# import xarray.testing as xrt

import pynsitu as pyn


@pytest.fixture()
def sample_sw_data():
    """Sample dataframe containing seawater properties"""
    return generate_sw_data("col")


def generate_sw_data(lonlat, kind="pd_dataframe"):
    """Create data containing seawater properties"""

    time = pd.date_range(start="2018-01-01", end="2018-01-15", freq="1h")
    time_scale = pd.Timedelta("10d")
    unit_oscillation = np.cos(2 * np.pi * ((time - time[0]) / time_scale))
    unit_trend = (time - time[0]) / (time[-1] - time[0])
    t = 10 + 5 * unit_oscillation
    s = 37 + 1 * unit_oscillation
    d = 1000 * unit_trend
    df = pd.DataFrame(dict(time=time, temperature=t, salinity=s, depth=d))
    if lonlat == "attr":
        df.lon = -30.0
        df.lat = 30.0
    elif lonlat == "col":
        df.loc[:, "lon"] = -30 + 5 * unit_oscillation
        df.loc[:, "lat"] = 30 + 5 * unit_oscillation
    if kind == "pd_dataframe":
        return df
    elif kind == "xr_dataset":
        ds = df.to_xarray().expand_dims(x=range(10))
        return ds
    elif kind == "xr_dask":
        ds = df.to_xarray().expand_dims(x=range(10))
        ds = ds.chunk(dict(x=2))
        return ds


# @pytest.mark.parametrize("s", ["salinity", "conductivity"])
@pytest.mark.parametrize("lonlat", ["attr", "col"])
def test_sw_update_eos(lonlat):
    """test seawater dataframe update_eos method"""

    df = generate_sw_data(lonlat)
    # inplace modification
    df.sw.update_eos()
    assert "SA" in df.columns

    df = generate_sw_data(lonlat)
    # inplace modification
    df.sw.update_eos()
    assert "SA" in df.columns

    # not inplace modification
    df_out = df.sw.update_eos(False)
    assert "SA" in df_out.columns


def test_sw_resample(sample_sw_data):
    """test seawater dataframe update_eos method, just run the code for now"""
    #
    df = sample_sw_data.copy().set_index("time")
    df.sw.resample("1d", interpolate=False, op="mean")
    #
    df = sample_sw_data.copy().set_index("time")
    df.sw.resample("1d", interpolate=True, op="median")
    #
    df = sample_sw_data.copy().set_index("time")
    df.sw.resample("1min", interpolate=True, op="mean")
