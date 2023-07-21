import numpy as np
import pandas as pd

# import xarray as xr
# import cftime
# import dask.array as dsar

import pytest

import numpy.testing as npt

# import xarray.testing as xrt

import pynsitu as pyn

# default timeseries
tdefault = dict(start="01-01-2018", end="15-01-2018", freq="1H")


@pytest.fixture()
def sample_tseries_data():
    return generate_time_series("time")


def generate_time_series(label="time", uniform=True):
    """Create a drifter time series."""
    time = pd.date_range(**tdefault)
    if not uniform:
        nt = time.size
        import random

        time = time[random.sample(range(nt), 2 * nt // 3)].sort_values()
    time_scale = pd.Timedelta("1D")
    v = (
        np.cos(2 * np.pi * ((time - time[0]) / time_scale))
        + np.random.randn(time.size) / 2
    )
    df = pd.DataFrame({"v": v, label: time})
    df = df.set_index(label)
    return df


@pytest.mark.parametrize("label", ["time", "date"])
def test_accessor_instantiation(label):

    df = generate_time_series(label)
    nt = df.index.size
    assert df.ts.time.size == nt


def test_trim(sample_tseries_data):

    _start = "2018/01/02 12:12:00 10 00.000  45 00.000"
    _end = "2018/01/10 12:12:00 10 00.000  45 00.000"
    _meta = dict(color="k", info="toto")
    d = pyn.events.Deployment(
        "label",
        start=_start,
        end=_end,
        meta=_meta,
    )

    sample_tseries_data.ts.trim(d)


def test_resample_uniform():
    # def resample_uniform(self, rule, inplace=False, **kwargs):
    df = generate_time_series(uniform=False)
    df.ts.resample_uniform("30s")


def test_resample_centered():
    # resample_centered(self, freq):
    df = generate_time_series(uniform=False)
    df.ts.resample_centered("2H")


def test_spectrum(sample_tseries_data):
    # spectrum(self, unit="1T", limit=None, **kwargs)

    df = sample_tseries_data
    E = df.ts.spectrum(nperseg=24 * 2)
