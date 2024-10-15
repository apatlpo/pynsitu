import numpy as np
import pandas as pd

import pytest
import numpy.testing as npt

import pynsitu as pyn


# def generate_drifter_data(id="myid"):
#    """Create a drifter time series."""
#    return generate_drifter_data()


def generate_drifter_data(id="myid", end="2018-01-15", freq="1h", velocities=False):
    """Create a drifter time series."""
    time = pd.date_range(start="2018-01-01", end=end, freq=freq)
    v = 0.1  # m/s approx
    scale = 111e3
    time_scale = pd.Timedelta("10d")
    lon = v * np.cos(2 * np.pi * ((time - time[0]) / time_scale)) / scale
    lat = v * np.sin(2 * np.pi * ((time - time[0]) / time_scale)) / scale
    df = pd.DataFrame(dict(lon=lon, lat=lat, time=time))
    df["id"] = id
    df = df.set_index("time")
    if velocities:
        df.geo.compute_velocities(inplace=True)
    return df


@pytest.fixture()
def sample_drifter_data():
    return generate_drifter_data()


@pytest.fixture()
def sample_drifter_dataset():
    return pd.concat(
        [generate_drifter_data(id=f"id{i}") for i in range(3)],
        axis=0,
    )


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
    acc_key = "acceleration_east", "acceleration_north", "acceleration"
    df = pyn.drifters.despike_isolated(df0, 1, acc_key=acc_key, verbose=False)
    assert df.index.size == Nt, f"{df.index.size, Nt}"
    assert df.columns.equals(df0.columns)

    # test with reasonable threshold, several spikes detected
    df = pyn.drifters.despike_isolated(df0, 1e-4, acc_key=acc_key, verbose=False)
    assert df.index.size < Nt, f"{df.index.size, Nt}"
    # output length is 334 agains 337 in input, i.e. 3 data points where deleted
    # this is more than expected, why !?!?


def test_variational_smooth(sample_drifter_data):
    """test smooth_resample, just run the code for now"""
    df = sample_drifter_data.geo.compute_velocities(distance="xy")  # to compute x/y
    df.geo.compute_accelerations(inplace=True)
    t_target = pd.date_range(df.index[0], df.index[-1], freq="30T")
    df_smooth = pyn.drifters.variational_smooth(
        df,
        t_target,
        1e-3,
        70,
        7.5e-6,
        0.15 * 86400,
        geo=False,
    )


# @pytest.mark.parametrize("method", ["lowess", "variational", "spydell"])
def _test_smooth_all(sample_drifter_dataset, method):
    """test smooth_resample, just run the code for now"""
    df = sample_drifter_dataset
    df = df.loc[
        (df.index < pd.Timestamp("2018-01-02"))
        | (df.index > pd.Timestamp("2018-01-02 12:00:00"))
    ]

    # t_target = pd.date_range(df.index[0, df.index[-1], freq="30T")
    t_target = "30min"
    if method == "lowess":
        param = pyn.drifters.optimized_parameters_lowess
    elif method == "variational":
        param = pyn.drifters.optimized_parameters_var
    elif method == "spydell":
        param = pyn.drifters.optimized_parameters_spydell

    assert "id" in df.columns, "id not in columns"
    df = (
        df.groupby("id", as_index=True)
        .apply(_add_xyuvdt_to_L1)
        # .reset_index(allow_duplicates=True)
        .drop(columns=["id"], errors="ignore")
        .reset_index()
    )
    # assert False, (df.index, df.columns)
    if df.index.name == "id":
        df = (
            df
            # .drop(columns=["id"], errors="ignore")
            .reset_index()
        )
    # assert False, (df.index.name, df.columns)

    # assert False, df.reset_index().columns
    # (df    .rename(columns={"id": "id1"})
    #    .reset_index()
    #    .drop(columns=["id1"])
    #    .set_index("id")
    # )

    # assert False, df.columns
    dfs = pyn.drifters.smooth_all(
        df,
        method,
        t_target,
        parameters=param,
        # import_columns=import_columns,
        maxgap=3 * 3600,
        spectral_diff=False,
        geo=True,
    )


def _add_xyuvdt_to_L1(df):
    if df.index.name != "time":
        if df.index.name == None:
            df = df.set_index("time")
        else:
            df = df.reset_index().set_index("time")

    # check time index is sorted
    if not df.index.is_monotonic_increasing:
        df.sort_index(inplace=True)
    df.geo.compute_velocities(
        keep_dt=True,
        inplace=True,
        fill_startend=False,
    )
    df.geo.compute_accelerations(
        inplace=True,
        fill_startend=False,
    )
    proj = df.geo.projection_reference
    df["lonc"] = proj[0]
    df["latc"] = proj[1]
    return df


def test_time_window_processing():
    """test the despiking procedure"""

    # common parameters
    T = pd.Timedelta("10d")
    dummy_value = 1.0
    gkwargs = dict(end="2018-03-01", velocities=True, freq="1h")

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
    time_unit = pd.Timedelta("1d")
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
