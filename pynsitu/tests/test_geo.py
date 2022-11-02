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
