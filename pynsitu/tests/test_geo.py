import numpy as np
import pandas as pd


import pytest
import numpy.testing as npt

import pynsitu as pyn


@pytest.fixture()
def sample_trajectory_data_periodic():
    """Create a trajectory time series with looping."""
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


@pytest.fixture()
def sample_trajectory_data_steady():
    """Create a trajectory time series with a steady flow."""
    freq = "1H"
    time = pd.date_range(start="2018-01-01", end="2018-01-15", freq=freq)

    ve, vn = 0.1, 0.1

    df = pd.DataFrame(dict(time=time))
    df = df.set_index("time")

    df["ve"] = ve
    df["vn"] = vn

    dt = pd.Timedelta(freq) / pd.Timedelta("1s")
    df["x"] = df["ve"].cumsum() * dt
    df["y"] = df["vn"].cumsum() * dt

    lon0, lat0 = 0, 45
    proj = pyn.geo.projection(lon0, lat0)
    df["lon"], df["lat"] = proj.xy2lonlat(df["x"], df["y"])

    return df


def test_compute_velocities(sample_trajectory_data_steady):

    df = sample_trajectory_data_steady
    df.geo.compute_velocities(centered=True, names=("vex", "vny", "vxy"), inplace=True)

    npt.assert_allclose(df["ve"], df["vex"], rtol=5e-2)
    npt.assert_allclose(df["vn"], df["vny"], rtol=5e-2)


def test_compute_acceleration(sample_trajectory_data_steady):

    # from velocities
    # ... to do

    # from xy
    df = sample_trajectory_data_steady
    df.geo.compute_accelerations(
        from_=("xy", "x", "y"), names=("aex", "any", "axy"), inplace=True
    )

    npt.assert_allclose(df["aex"], df["aex"] * 0, atol=1e-10)
    npt.assert_allclose(df["any"], df["aex"] * 0, atol=1e-10)
    npt.assert_allclose(df["axy"], df["aex"] * 0, atol=1e-10)
