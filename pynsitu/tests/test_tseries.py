import numpy as np
import pandas as pd
import random

# import xarray as xr
# import cftime
# import dask.array as dsar

import pytest

import numpy.testing as npt

# import xarray.testing as xrt

import pynsitu as pyn

## -------------- method building used for testing

# default timeseries
tdefault = dict(start="01-01-2018", end="15-01-2018", freq="1h")


@pytest.fixture()
def sample_tseries_data():
    """ """
    return generate_time_series("time")


def generate_time_series(
    label="time", uniform=True, kind="pd_dataframe", time_units="datetime"
):
    """Create a drifter time series."""
    time = pd.date_range(**tdefault)
    time_scale = pd.Timedelta("1d")
    if time_units == "timedelta":
        time = time - time[0]
    elif time_units == "numeric":
        time = (time - time[0]) / pd.Timedelta("1h")
        time_scale = 1.0
    if not uniform:
        nt = time.size
        time = time[random.sample(range(nt), 2 * nt // 3)].sort_values()
    #
    v0 = (
        np.cos(2 * np.pi * ((time - time[0]) / time_scale))
        + np.random.randn(time.size) / 2
    )
    v1 = (
        np.sin(2 * np.pi * ((time - time[0]) / time_scale))
        + np.random.randn(time.size) / 2
    )
    df = pd.DataFrame({"v0": v0, "v1": v1, label: time})
    df = df.set_index(label)
    if kind == "pd_dataframe":
        return df
    elif kind == "xr_dataset":
        ds = df.to_xarray().expand_dims(x=range(10))
        return ds
    elif kind == "xr_dask":
        ds = df.to_xarray().expand_dims(x=range(10))
        ds = ds.chunk(dict(x=2))
        return ds


## -------------- basic time series editing


@pytest.mark.parametrize("label", ["time", "date"])
def test_accessor_instantiation(label):

    df = generate_time_series(label)
    nt = df.index.size
    assert df.ts.time.size == nt


@pytest.mark.parametrize(
    "kind", ["pd_dataframe", "xr_dataset", "xr_dask"]
)  # should add: pd_series, xr_dataarray
@pytest.mark.parametrize("time_units", ["datetime", "timedelta", "numeric"])
def test_trim(kind, time_units):

    _start = "2018/01/02 12:12:00 10 00.000  45 00.000"
    _end = "2018/01/10 12:12:00 10 00.000  45 00.000"
    _meta = dict(color="k", info="toto")
    d = pyn.events.Deployment(
        "label",
        start=_start,
        end=_end,
        meta=_meta,
    )

    s = generate_time_series(kind=kind, time_units=time_units)
    s.ts.trim(d)


def test_resample_uniform():
    # def resample_uniform(self, rule, inplace=False, **kwargs):
    df = generate_time_series(uniform=False)
    df.ts.resample_uniform("30s")


def test_resample_centered():
    # resample_centered(self, freq):
    df = generate_time_series(uniform=False)
    df.ts.resample_centered("2h")


## -------------- spectral calculations


# welch calculation
@pytest.mark.parametrize("kind", ["pd_dataframe", "xr_dataset", "xr_dask"])
@pytest.mark.parametrize("time_units", ["datetime", "timedelta", "numeric"])
def test_spectrum_welch(kind, time_units):
    s = generate_time_series(kind=kind, time_units=time_units)
    E = s.ts.spectrum(nperseg=24 * 2)


# periodogram calculation
@pytest.mark.parametrize("kind", ["pd_dataframe", "xr_dataset", "xr_dask"])
@pytest.mark.parametrize("time_units", ["datetime", "timedelta", "numeric"])
def test_spectrum_periodogram(kind, time_units):
    s = generate_time_series(kind=kind, time_units=time_units)
    E = s.ts.spectrum(method="periodogram")


# rotary spectrum
@pytest.mark.parametrize("kind", ["pd_dataframe", "xr_dataset", "xr_dask"])
def test_rotary_spectrum(kind):
    s = generate_time_series(kind=kind)
    E = s.ts.spectrum(nperseg=24 * 2, complex=("v0", "v1"))
