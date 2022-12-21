import numpy as np
import pandas as pd

import pytest
import numpy.testing as npt

import pynsitu as pyn


@pytest.fixture()
def sample_drifter_data():
    """Create a drifter time series."""
    time = pd.date_range(start="2018-01-01", end="2018-01-15", freq="1H")
    v = 0.1  # m/s approx
    scale = 111e3
    time_scale = pd.Timedelta("10D")
    lon = v * np.cos(2 * np.pi * ((time - time[0]) / time_scale)) / scale
    lat = v * np.sin(2 * np.pi * ((time - time[0]) / time_scale)) / scale
    df = pd.DataFrame(dict(lon=lon, lat=lat, time=time))
    df["id"] = "myid"
    df = df.set_index("time")
    return df


def test_despike_isolated(sample_drifter_data):
    """test the despiking procedure"""

    # add a spike
    Nt = sample_drifter_data.index.size
    df = sample_drifter_data
    df["lon"][10] = df["lon"][10] + 1e-1
    df["lat"][10] = df["lat"][10] + 1e-1
    df0 = df.geo.compute_velocities()
    df0.geo.compute_accelerations(inplace=True)

    # test with very high threshold, not spikes should be detected
    df = pyn.drifters.despike_isolated(df0, 1, verbose=False)
    assert df.index.size == Nt, f"{df.index.size, Nt}"
    assert df.columns.equals(df0.columns)

    # test with reasonable threshold, several spikes detected
    df = pyn.drifters.despike_isolated(df0, 1e-4, verbose=False)
    assert df.index.size < Nt, f"{df.index.size, Nt}"
    # output length is 334 agains 337 in input, i.e. 3 data points where deleted
    # this is more than expected, why !?!?


def test_smooth_resample(sample_drifter_data):
    """test smooth_resample, just run the code for now"""
    df = sample_drifter_data.geo.compute_velocities()  # to compute x/y
    t_target = pd.date_range(df.index[0], df.index[-1], freq="30T")
    df_smooth = pyn.drifters.smooth_resample(df, t_target, 100, 1e-4, 86400.0)
