import numpy as np
import pandas as pd

import pytest
import numpy.testing as npt

import pynsitu as pyn


@pytest.fixture()
def sample_drifter_data():
    """Create a drifter time series."""
    return generate_drifter_data()


def generate_drifter_data(end="2018-01-15", freq="1H", velocities=False):
    """Create a drifter time series."""
    time = pd.date_range(start="2018-01-01", end=end, freq=freq)
    v = 0.1  # m/s approx
    scale = 111e3
    time_scale = pd.Timedelta("10D")
    lon = v * np.cos(2 * np.pi * ((time - time[0]) / time_scale)) / scale
    lat = v * np.sin(2 * np.pi * ((time - time[0]) / time_scale)) / scale
    df = pd.DataFrame(dict(lon=lon, lat=lat, time=time))
    df["id"] = "myid"
    df = df.set_index("time")
    if velocities:
        df.geo.compute_velocities(inplace=True)
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


def test_resample_smooth(sample_drifter_data):
    """test smooth_resample, just run the code for now"""
    df = sample_drifter_data.geo.compute_velocities()  # to compute x/y
    t_target = pd.date_range(df.index[0], df.index[-1], freq="30T")
    df_smooth = pyn.drifters.resample_smooth(df, t_target, 100, 1e-4, 86400.0)
    assert df_smooth is not None, df_smooth


def test_time_window_processing():
    """test the despiking procedure"""

    # common parameters
    T = pd.Timedelta("10D")
    dummy_value = 1.0
    gkwargs = dict(end="2018-03-01", velocities=True, freq="1H")

    def _processing(df, dummy=None):
        return pd.Series(dict(u=df["velocity_east"].mean(skipna=True) * 0.0 + dummy))

    # generate a longer time series
    df = generate_drifter_data(**gkwargs)
    # add gaps
    df.loc["2018-02-01":"2018-02-15", "velocity_east"] = np.NaN

    # base case
    out = pyn.drifters.time_window_processing(
        df,
        _processing,
        T,
        geo=True,
        dummy=dummy_value,
    )
    assert out["u"].mean(skipna=True) == dummy_value, out

    # x, y - non-geo case
    out = pyn.drifters.time_window_processing(
        df,
        _processing,
        T,
        xy=("x", "y"),
        dummy=dummy_value,
    )
    assert out["u"].mean(skipna=True) == dummy_value, out

    # time is float
    time_unit = pd.Timedelta("1D")
    df.index = (df.index - df.index[0]) / time_unit
    Tf = T / time_unit
    out = pyn.drifters.time_window_processing(
        df,
        _processing,
        Tf,
        geo=True,
        dummy=dummy_value,
    )
    assert out["u"].mean(skipna=True) == dummy_value, out

    ## temporal resampling

    # with time as float
    df = df.loc[(df.index < 10) | (df.index > 20)]
    dtf = pd.Timedelta(gkwargs["freq"]) / time_unit
    out = pyn.drifters.time_window_processing(
        df,
        _processing,
        Tf,
        dt=dtf,
        geo=True,
        dummy=dummy_value,
    )

    # with time as datetime
    df = generate_drifter_data(**gkwargs)
    df.loc["2018-02-01":"2018-02-15", "velocity_east"] = np.NaN
    df = df.loc[
        (df.index < pd.Timestamp("2018-02-01"))
        | (df.index > pd.Timestamp("2018-02-10"))
    ]
    out = pyn.drifters.time_window_processing(
        df,
        _processing,
        T,
        dt=gkwargs["freq"],
        geo=True,
        dummy=dummy_value,
    )
    assert out["u"].mean(skipna=True) == dummy_value, out
