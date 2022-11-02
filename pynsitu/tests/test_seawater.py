import numpy as np
import pandas as pd

import pytest
import numpy.testing as npt
# import xarray.testing as xrt

import pynsitu as pyn

@pytest.fixture()
def sample_sw_dataframe():
    """Sample dataframe containing seawater properties"""
    return generate_sw_dataframe("col")

def generate_sw_dataframe(lonlat):
    """Create a dataframe containing seawater properties"""
    time = pd.date_range(start="2018-01-01", end="2018-01-15", freq="1H")
    time_scale = pd.Timedelta("10D")
    unit_oscillation = np.cos(2 * np.pi * ((time - time[0]) / time_scale))
    unit_trend = ((time - time[0]) / (time[-1] - time[0]))
    t = 10 + 5 * unit_oscillation
    s = 37 + 1 * unit_oscillation
    d =  1000 * unit_trend
    df = pd.DataFrame(dict(time=time, temperature=t, salinity=s, depth=d))
    if lonlat=="attr":
        df.lon = -30.
        df.lat =  30.
    elif lonlat=="col":
        df.loc[:, "lon"] = -30 + 5*unit_oscillation
        df.loc[:, "lat"] = 30 + 5*unit_oscillation
    #df = df.set_index("time")
    return df

#@pytest.mark.parametrize("s", ["salinity", "conductivity"])
@pytest.mark.parametrize("lonlat", ["attr", "col"])
def test_sw_update_eos(lonlat):
    """test seawater dataframe update_eos method"""

    df = generate_sw_dataframe(lonlat)
    # inplace modification
    df.sw.update_eos()
    assert "SA" in df.columns

    df = generate_sw_dataframe(lonlat)
    # inplace modification
    df.sw.update_eos()
    assert "SA" in df.columns

    # not inplace modification
    df_out = df.sw.update_eos(False)
    assert "SA" in df_out.columns

def test_sw_resample(sample_sw_dataframe):
    """test seawater dataframe update_eos method, just run the code for now"""
    #
    df = sample_sw_dataframe.copy().set_index("time")
    df.sw.resample("1D", interpolate=False, op="mean")
    #
    df = sample_sw_dataframe.copy().set_index("time")
    df.sw.resample("1D", interpolate=True, op="median")
    #
    df = sample_sw_dataframe.copy().set_index("time")
    df.sw.resample("1T", interpolate=True, op="mean")
