import warnings
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from numpy.linalg import inv
from scipy.linalg import solve
from scipy.sparse import diags
from scipy.special import erf

from pynsitu.geo import GeoAccessor, compute_velocities, compute_accelerations

from numba import njit, guvectorize, int32, float64, prange

#################################################################################
# ------------------------ DRIFTER DATA CLEANING --------------------------------
#################################################################################


def despike_isolated(df, acceleration_threshold, acc_key=None, verbose=False):
    """Drops isolated anomalous positions (spikes) in a position time series.
    Anomalous positions are first detected if acceleration exceed the provided
    threshold.
    Detected values are masked if they are combined with an adequate pattern
    of acceleration sign reversals, e.g. +-+ or -+-
    Speed acceleration should have been computed with the pynsitu.geo.GeoAccessor,
    e.g.: df.geo.compute_velocities(centered=False, acceleration=True)

    Parameters
    ----------
    df: `pandas.DataFrame`
        Input dataframe, must contain an `acceleration` column
    acceleration_threshold: float
        Threshold used to detect anomalous values
    acc_key: tuple, optional
        Keys/labels/column identifiers for x/y/absolute value of acceleration
    verbose: boolean
        Outputs number of anomalous values detected
        Default is True

    Returns
    -------
    df: `pandas.DataFrame`
        Output dataframe with spikes removed.

    """

    if acc_key is None:
        acc_key = "acceleration_east", "acceleration_north", "acceleration"

    assert acc_key[2] in df.columns, (
        "'acceleration' should be a column. You may need to leverage the "
        + "geo accessor first (pynsitu.geo.GeoAccessor) with "
        + "`df.geo.compute_velocities(acceleration=True)``"
    )

    # first pass: anomalous large acceleration values
    spikes = df[df[acc_key[2]] > acceleration_threshold]

    # second pass: seach for adequate sign reversals
    validated_single_spikes = []
    for t in spikes.index:
        C = []
        # check for a double sign reversal of acceleration
        for acc in acc_key[:1]:
            if t > df.index[0] and t < df.index[-1]:
                am = df.loc[:t, acc].iloc[-2]
                a = spikes.loc[t, acc]
                ap = df.loc[t:, acc].iloc[1]
                # check if am and ap have opposite sign to a
                C.append(am * a < 0 and ap * a < 0)
        if len(C) > 0 and any(C):
            validated_single_spikes.append(t)
    if verbose:
        print(
            f"{len(validated_single_spikes)} single spikes dropped out of {spikes.index.size}"
            + f" potential ones (acceleration threshold)"
        )
    # drops single spikes
    df = df.drop(validated_single_spikes)
    return df


def despike_all(df, acceleration_threshold, acc_key=None, verbose=False):
    """Drops isolated anomalous positions (spikes) in a position time series.
    Anomalous positions are first detected if acceleration exceed the provided
    threshold.
    Speed acceleration should have been computed with the pynsitu.geo.GeoAccessor,
    e.g.: df.geo.compute_velocities(centered=False, acceleration=True)

    Parameters
    ----------
    df: `pandas.DataFrame`
        Input dataframe, must contain an `acceleration` column
    acceleration_threshold: float
        Threshold used to detect anomalous values
    acc_key: tuple, optional
        Keys/labels/column identifiers for x/y/absolute value of acceleration
    verbose: boolean
        Outputs number of anomalous values detected
        Default is True

    Returns
    -------
    df: `pandas.DataFrame`
        Output dataframe with spikes removed.

    """

    if acc_key is None:
        acc_key = "acceleration_east", "acceleration_north", "acceleration"

    assert acc_key[2] in df.columns, (
        "'acceleration' should be a column. You may need to leverage the "
        + "geo accessor first (pynsitu.geo.GeoAccessor) with "
        + "`df.geo.compute_velocities(acceleration=True)``"
    )
    return df[
        (abs(df[acc_key[0]]) < acceleration_threshold)
        | (abs(df[acc_key[1]]) < acceleration_threshold)
    ]


def despike_pm(df, acceleration_threshold, pm=1, acc_key=None, verbose=False):
    """Drops anomalous positions (spikes) in a position time series.
    Anomalous positions are first detected if acceleration exceed the provided
    threshold.
    The pm points before and after are also removed
    Speed acceleration should have been computed with the pynsitu.geo.GeoAccessor,
    e.g.: df.geo.compute_velocities(centered=False, acceleration=True)

    Parameters
    ----------
    df: `pandas.DataFrame`
        Input dataframe, must contain an `acceleration` column
    acceleration_threshold: float
        Threshold used to detect anomalous values
    pm : int
        number of point before and after to remove
    acc_key: tuple, optional
        Keys/labels/column identifiers for x/y/absolute value of acceleration
    verbose: boolean
        Outputs number of anomalous values detected
        Default is True

    Returns
    -------
    df: `pandas.DataFrame`
        Output dataframe with spikes removed.

    """

    if acc_key is None:
        acc_key = "acceleration_east", "acceleration_north", "acceleration"

    assert acc_key[2] in df.columns, (
        "'acceleration' should be a column. You may need to leverage the "
        + "geo accessor first (pynsitu.geo.GeoAccessor) with "
        + "`df.geo.compute_velocities(acceleration=True)``"
    )
    df_ = df.copy()
    df_ = df_.reset_index()
    spikes = df_[
        (abs(df_[acc_key[1]]) > acceleration_threshold)
        | (abs(df_[acc_key[1]]) > acceleration_threshold)
    ].index.values
    spikes = np.unique(np.concatenate([spikes + pm, spikes - pm, spikes]))

    spikes = df.iloc[spikes].index

    return df.drop(spikes)


########################################################
# -----------FIND AND FILL WITH NAN BIG GAPS------------#
def nan_in_gap(df, df_gap, dtmax, inplace=False):
    """Fill gaps bigger than dtmax with nan

    Parameters
    ----------
    df : dataframe on which we want to put the gap
    df_gap : original dataframe
    dtmax: float
        max gap length in seconds
    inplace : boolean

    """
    df = df.reset_index()
    if not inplace:
        df = df.copy()
    time_start, time_end = find_gap(df_gap, dtmax)
    for i in range(len(time_start)):
        test1 = df.time > time_start[i]
        test2 = df.time < time_end[i]
        test = np.logical_not(test1 & test2)
        df = df.where(test)
    if not inplace:
        return df.set_index("time")


########################################################
# -----------COMPUTATION OF ACC (SAME FOR ALL)------------#
def compute_acc(df_out, geo, spectral_diff, method, velocities_key, accelerations_key):
    if spectral_diff:
        if geo:
            if method == "lowess" or method == "variational":
                df_out.geo.compute_accelerations(
                    from_=("xy_spectral", "x", "y"),
                    names=accelerations_key,
                    centered_velocity=True,
                    time="index",
                    fill_startend=True,
                    inplace=True,
                )
            if method == "spydell":
                df_out.geo.compute_accelerations(
                    from_=("velocities_spectral", velocities_key[0], velocities_key[1]),
                    names=accelerations_key,
                    centered_velocity=True,
                    time="index",
                    fill_startend=True,
                    inplace=True,
                )
            # should still recompute for non-geo datasets
        else:
            if method == "lowess" or method == "variational":
                compute_accelerations(
                    df_out,
                    from_=("xy_spectral", "x", "y"),
                    names=accelerations_key,
                    centered_velocity=True,
                    time="index",
                    fill_startend=True,
                    inplace=True,
                    keep_dt=False,
                )
            if method == "spydell":
                compute_accelerations(
                    df_out,
                    from_=("velocities_spectral", velocities_key[0], velocities_key[1]),
                    names=accelerations_key,
                    centered_velocity=True,
                    time="index",
                    fill_startend=True,
                    inplace=True,
                    keep_dt=True,
                )
    else:
        if geo:
            if method == "lowess" or method == "variational":
                df_out.geo.compute_accelerations(
                    from_=("xy", "x", "y"),
                    names=accelerations_key,
                    centered_velocity=True,
                    time="index",
                    fill_startend=True,
                    inplace=True,
                )
            if method == "spydell":
                df_out.geo.compute_accelerations(
                    from_=("velocities", velocities_key[0], velocities_key[1]),
                    names=accelerations_key,
                    centered_velocity=True,
                    time="index",
                    fill_startend=True,
                    inplace=True,
                )
            # should still recompute for non-geo datasets
        else:
            if method == "lowess" or method == "variational":
                compute_accelerations(
                    df_out,
                    from_=("xy", "x", "y"),
                    names=accelerations_key,
                    centered_velocity=True,
                    time="index",
                    fill_startend=True,
                    inplace=True,
                    keep_dt=False,
                )
            if method == "spydell":
                compute_accelerations(
                    df_out,
                    from_=("velocities", velocities_key[0], velocities_key[1]),
                    names=accelerations_key,
                    centered_velocity=True,
                    time="index",
                    fill_startend=True,
                    inplace=True,
                    keep_dt=True,
                )


###########################################
# -----------VARIATIONNAL METHOD------------#
def variational_smooth(
    df,
    t_target,
    acc_cut,
    position_error,
    acceleration_amplitude,
    acceleration_T,
    time_chunk=2,
    velocities_key=("velocity_east", "velocity_north", "velocity"),
    accelerations_key=("acceleration_east", "acceleration_north", "acceleration"),
    import_columns=["id"],
    spectral_diff=True,
    geo=True,
):
    """Smooth and resample a drifter position time series
    The smoothing balances positions information according to the specified
    position error and the smoothness of the output time series by specifying
    a typical acceleration amplitude and decorrelation timescale (assuming
    exponential decorrelation).
    The output trajectory `x` minimizes:
        || I(x) - x_obs ||^2 / e_x^2 + (D2 x)^T R^{-1} (D2 x)
    where e_x is the position error, `I` the time interpolation operator,
    `R` the acceleration autocorrelation, `D2` the second order derivative.

    Closest reference (but no temporal autocorrelation of acceleration considered):
    Yaremchuk and Coelho 2015. Filtering Drifter Trajectories Sampled at Submesoscale, Resolution. IEEE Journal of Oceanic Engineering

    Parameters
    ----------
                df: `pandas.DataFrame`
                    Input drifter time series, must contain projected positions (`x` and `y`)
                t_target: `pandas.core.indexes.datetimes.DatetimeIndex`
                    Output time series, as typically given by pd.date_range
                    Note that the problem seems ill-posed in the downsampling case ... need
                    to be fixed
                acc_cut : float,
                    acceleration spike cut
                position_error: float
                    Position error in meters
                acceleration_amplitude: float
                    Acceleration typical amplitude
                acceleration_T: float
                    Acceleration decorrelation timescale in seconds
                time_chunk: int/float, optional
                    Maximum time chunk (in days) to process at once.
                    Data is processed by chunks and patched together.
                velocities_key : (,,) of str,
                    ex : ('velocity_east','velocity_north', 'velocity') or ('u','v', 'U') etc
                accelerations_key : (,,) of str,
                    ex : ('acceleration_east','acceleration_north', 'acceleration') or ('ax','ay', 'Axy') or ('au','av', 'Auv') etc
                spectral_diff : boolean
                    computing velocities and accelaration with spectral diff or not
                import_columns : list of str
                    list of df constant columns we want to import (ex: id, platform)
                geo: boolean,
                    optional if geo obj with projection

    Return : interpolated dataframe with x, y, u, v, ax-ay computed from xy, au-av computed from u-v, +norms, +import_columns with index time

    """
    df = df.copy()

    # store projection to align with dataframes produced
    if geo:
        if "lonc" in df:
            proj_ref = (df.lonc.values[0], df.latc.values[0])
            if "lonc" not in import_columns:
                import_columns += ["lonc", "latc"]
        else:
            proj_ref = df.geo.projection_reference

    # index = time
    if df.index.name != "time":
        if df.index.name == None:
            df = df.set_index("time")
        else:
            df = df.reset_index().set_index("time")

    # assert x, y in dataframe
    if "x" not in df or "y" not in df:
        assert False, "positions must be labelled as 'x' and 'y'"
    if geo:
        if "lon" not in df or "lat" not in df:
            assert False, "longitude, latitude must be labelled as 'lon' and 'lat'"
    if accelerations_key[2] not in df:
        assert False, "'acceleration' should be provided"

    # despike acceleration
    try:
        # df = despike_isolated(df, acc_cut, accelerations_key)# spike are made of not only one points
        # df = despike_all(df, acc_cut, accelerations_key)# spike before and after often are also not ok
        df = despike_pm(df, acc_cut, pm=1, acc_key=accelerations_key)
    except:
        assert False, "pb despike"

    # select only x, y
    if "id" not in import_columns:
        import_columns += ["id"]
    var = ["x", "y"] + import_columns
    # if geo:
    #    var += ["lon", "lat"]
    df = df[var]

    # t_target
    if isinstance(t_target, str):
        t_target = pd.date_range(
            df.index.min().ceil(t_target), df.index.max(), freq=t_target
        )
    else:
        # enforce t_target type
        t_target = pd.DatetimeIndex(t_target)

    # Time series length in days
    T = (t_target[-1] - t_target[0]) / pd.Timedelta("1d")

    if not time_chunk or T < time_chunk * 1.1:
        df_out = _variational_smooth_one(
            df,
            t_target,
            position_error,
            acceleration_amplitude,
            acceleration_T,
        )
    else:
        print(f"Chunking dataframe into {time_chunk} days chunks")
        # divide target timeline into chunks
        D = _divide_into_time_chunks(t_target, time_chunk, overlap=0.3)
        # split computation
        delta = pd.Timedelta("3h")
        R = []
        for time in D:
            df_chunk = df.loc[
                (df.index > time[0] - delta) & (df.index < time[-1] + delta)
            ]
            df_chunk_smooth = _variational_smooth_one(
                df_chunk,
                time,
                position_error,
                acceleration_amplitude,
                acceleration_T,
            )
            R.append(df_chunk_smooth)

        # brute concatenation: introduce strong discontinuities in velocity/acceleration
        # df_out = pd.concat(R)
        # removes duplicated times
        # df_out = df_out.loc[~df_out.index.duplicated(keep="first")]

        ## patch timeseries together, not that simple ...

        col_float = [c for c in df.columns if np.issubdtype(df[c].dtype, float)]
        # note: is_numeric_dtype(df[c].dtype) lets int pass through
        i = 0
        while i < len(R) - 1:
            if i == 0:
                df_left = R[i]
            else:
                df_left = df_right
            df_right = R[i + 1]
            delta = df_left.index[-1] - df_right.index[0]
            t_mid = df_right.index[0] + delta * 0.5

            # bring time series on a common timeline
            index = df_left.index.union(df_right.index)

            #
            df_left = df_left.reindex(index, method=None)
            df_right = df_right.reindex(index, method=None)

            # build weights
            w = (1 - erf((df_left.index.to_series() - t_mid) * 5 / delta)) * 0.5
            # note: the width in the error function needs to be much smaller than delta.
            # Discontinuities visible on acceleration are visible otherwise
            # A factor 5 is chosen here

            df_left = df_left.fillna(df_right)
            df_right = df_right.fillna(df_left)

            # patch values with float dtypes
            for c in col_float:
                df_right.loc[:, c] = df_left.loc[:, c] * w + df_right.loc[:, c] * (
                    1 - w
                )

            i += 1

        df_out = df_right
        # fix non float dtypes
        # for c in df_out.columns:
        #    df_out[c] = df_out[c].astype(df[c].dtype)

    # fill na
    df_out = df_out.bfill().ffill()

    # compute velocity
    if spectral_diff:
        dist = "spectral"
    else:
        dist = "xy"

    if geo:
        # initiate lon, lat (needed to compute_acc, even if it is computed from x, y)
        df_out["lon"] = df.lonc.mean()
        df_out["lat"] = df.latc.mean()
        df_out.geo.compute_velocities(
            names=velocities_key,
            distance=dist,
            inplace=True,
            fill_startend=True,
        )
    else:
        compute_velocities(
            df_out,
            "index",
            names=velocities_key,
            distance=dist,
            inplace=True,
            centered=True,
            fill_startend=True,
        )

    # compute acceleration
    compute_acc(
        df_out, geo, spectral_diff, "variational", velocities_key, accelerations_key
    )

    # update lon/lat
    if geo:
        # first reset reference from df
        df_out.geo.set_projection_reference(proj_ref)  # inplace
        df_out.geo.compute_lonlat()  # inplace

    df_out["X"] = np.sqrt(df_out["x"] ** 2 + df_out["y"] ** 2)
    # df_out[accelerations_key[2]] = np.sqrt(
    #    df_out[accelerations_key[0]] ** 2 + df_out[accelerations_key[1]] ** 2
    # )

    # import columns/info ex: id or time
    if import_columns:
        for column in import_columns:
            try:
                df_out[column] = df[column][0]
            except:
                assert False, df.columns

    return df_out


def _variational_smooth_one(
    df,
    t_target,
    position_error,
    acceleration_amplitude,
    acceleration_T,
):
    """core processing for variational_smooth, process one time window"""

    # init final structure
    df_out = df.reindex(df.index.union(t_target), method="nearest").reindex(t_target)
    # providing "nearest" above is essential to preserve type (on int64 data typically)
    # override with interpolation for float data

    col_float = [c for c in df.columns if np.issubdtype(df[c], float)]

    for c in col_float:
        df_out[c] = (
            df[c]
            .reindex(df.index.union(t_target))
            .interpolate("time")
            .reindex(t_target)
        )
    df_out.index.name = "time"

    # exponential acceleration autocorrelation
    R = lambda dt: acceleration_amplitude**2 * np.exp(-np.abs(dt / acceleration_T))
    # get operators
    L, I = _get_smoothing_operators(t_target, df.index, position_error, R)

    # x
    df_out["x"] = solve(L, I.T.dot(df["x"].values))
    # y
    df_out["y"] = solve(L, I.T.dot(df["y"].values))

    return df_out


def _divide_into_time_chunks(time, T, overlap=0.1):
    """Divide a dataframe into chunks of duration T (in days)

    Parameters
    ----------
    time: pd.DatetimeIndex
        Timeseries
    T: float
        Size of time chunks in days

    """
    Td = pd.Timedelta("1d") * T

    # assumes time is the index
    t_first = time[0]
    t_last = time[-1]

    t = t_first
    D = []
    while t < t_last:
        # try to keep a chunk of size T even last one
        if t + Td > t_last:
            tb = max(t_last - Td, t_first)
            start, end = tb, tb + Td
        else:
            start, end = t, t + Td
        D.append(time[(time >= start) & (time <= end)])
        t = t + Td * (1 - overlap)
    return D


def _get_smoothing_operators(t_target, t, position_error, acceleration_R):
    """Core operators in order to minimize:
        (Ix - x_obs)^2 / e_x^2 + (D2 x)^T R^{-1} (D2 x)
    where R is the acceleration autocorrelation function, assumed to follow

    """

    # assumes t_target is uniform
    dt = t_target[1] - t_target[0]

    # build linear interpolator
    Nt = t_target.size
    I = np.zeros((t.size, Nt))
    i_t = np.searchsorted(
        t_target, t
    )  # Find the indices into a sorted array `t_target` such that, if the corresponding elements in `t` were inserted before the indices, the order of `t_target` would be preserved

    i = np.where((i_t > 0) & (i_t < Nt))[
        0
    ]  # remove times before and after the target times
    j = i_t[i]
    # t[0] is between t_target[i_t[0]-1=j[0]-1] and  t_target[i_t[0]=j[0]]
    w = (t[i] - t_target[j - 1]) / dt  # weight =distance to neightboors
    I[i, j - 1] = w
    I[i, j] = 1 - w

    # second order derivative
    one_second = pd.Timedelta("1s")
    dt2 = (dt / one_second) ** 2
    D2 = diags(
        [1 / dt2, -2 / dt2, 1 / dt2], [-1, 0, 1], shape=(Nt, Nt)
    ).toarray()  # Nt*Nt with -2/dt2 at the diagonale, 1/dt2 just below and above
    # fix boundaries
    # D2[0, :] = 0
    # D2[-1, :] = 0
    # need to impose boundary conditions or else pulls acceleration towards 0 as it is
    # D2[0, [0, 1]] = [-1/dt2, 1/dt2] # not good: pull velocity towards 0 at edges
    # D2[-1, [-2, -1]] = [-1/dt2, 1/dt2]  # not good: pull velocity towards 0 at edges
    # constant acceleration at boundaries (does not work ... weird):
    # D2[0, [0, 1, 2, 3]] = [-1 / dt2, 3 / dt2, -3 / dt2, 1 / dt2]
    # D2[-1, [-4, -3, -2, -1]] = [1 / dt2, -3 / dt2, 3 / dt2, -1 / dt2]

    # acceleration autocorrelation
    _t = t_target.values
    R = acceleration_R((_t[:, None] - _t[None, :]) / one_second)
    # apply constraint on laplacian only on inner points (should try to impose above boundary treatment instead)
    D2 = D2[1:-1, :]
    R = R[1:-1, 1:-1]
    # boundaries
    # R[0,:] = 0
    # R[0,0] = R[1,1]*1000
    # R[-1,:] = 0
    # R[-1,-1] = R[-2,-2]*1000
    #
    iR = inv(R)

    # assemble final operator
    L = I.T.dot(I) + D2.T.dot(iR.dot(D2)) * position_error**2

    return L, I


###########################################
# -----------EMPIRICAL METHOD------------#

import warnings


def spydell_smooth(
    df,
    t_target,
    acc_cut=1e-3,
    nb_pt_mean=5,
    velocities_key=("velocity_east", "velocity_north", "velocity"),
    accelerations_key=("acceleration_east", "acceleration_north", "acceleration"),
    import_columns=["id"],
    spectral_diff=True,
    geo=True,
):
    """
    Smooth and interpolated a trajectory with the method described in Spydell et al. 2021.

    Parameters:
    -----------
            df :  dataframe with raw trajectory, must contain 'time', 'x', 'y','u' and 'v'
            t_target: `pandas.core.indexes.datetimes.DatetimeIndex` or str
                Output time series, as typically given by pd.date_range or the delta time of the output time series as str
                In this case, t_target is then recomputed taking start-end the start end of the input trajectory and the given delta time
            acc_cut : float,
                acceleration spike cut value
            nb_pt_mean : odd int,
                number of points of wich is applied the box mean
            velocities_key : (,,) of str,
                ex : ('velocity_east','velocity_north', 'velocity') or ('u','v', 'U') etc
            accelerations_key : (,,) of str,
                ex : ('acceleration_east','acceleration_north', 'acceleration') or ('ax','ay', 'Axy') or ('au','av', 'Auv') etc
            import_columns : list of str,
                list of df constant columns we want to import (ex: id, platform)
            geo: boolean,
                optional if geo obj with projection
    Return : interpolated dataframe with x, y, u, v, ax-ay computed from xy, au-av computed from u-v, +norms, id, platform with index time
    """

    # index = time
    if df.index.name != "time":
        if df.index.name == None:
            df = df.set_index("time")
        else:
            df = df.reset_index().set_index("time")

    # assert x, y in dataframe
    if "x" not in df or "y" not in df:
        assert False, "positions must be labelled as 'x' and 'y'"
    if velocities_key[0] not in df or velocities_key[1] not in df:
        assert False, f"velocities must be labelled as {velocities_key}"

    # store projection to align with dataframes produced
    if geo:
        if "lonc" in df:
            proj_ref = (df.lonc.values[0], df.latc.values[0])
            if "lonc" not in import_columns:
                import_columns += ["lonc", "latc"]
        else:
            proj_ref = df.geo.projection_reference

    # t_target
    if isinstance(t_target, str):
        t_target = pd.date_range(
            df.index.min().ceil(t_target), df.index.max(), freq=t_target
        )

    # xarray for easy interpolation
    ds = df.to_xarray()[["x", "y", velocities_key[0], velocities_key[1]]]

    # fill little gap

    # 3) linearly interpolate velocities
    ds = ds.interp(time=t_target, method="linear")

    reg_dt = t_target[1] - t_target[0]

    # 4) integrate velocities and find constant
    # ms_x, ms_y = (df.x**2).mean(), (df.y**2).mean()
    x_cum = ds[velocities_key[0]].cumsum("time") * reg_dt / pd.Timedelta("1s")
    y_cum = ds[velocities_key[1]].cumsum("time") * reg_dt / pd.Timedelta("1s")

    # def msx_difference(x_0) :
    # return abs(ms_x-((x_0+x_cum)**2).mean())
    # def msy_difference(y_0) :
    # return abs(ms_y-((y_0+y_cum)**2).mean())
    from scipy.optimize import minimize

    def msx_difference(x_0):
        return ((ds.x - x_0 - x_cum) ** 2).mean()

    def msy_difference(y_0):
        return ((ds.y - y_0 - y_cum) ** 2).mean()

    from scipy.optimize import minimize

    x_0 = minimize(msx_difference, ds.x[0]).x
    y_0 = minimize(msy_difference, ds.y[0]).x

    ds["x"] = x_0 + x_cum
    ds["y"] = y_0 + y_cum

    # 5) remove spike and interpolate
    ds[accelerations_key[0]] = ds[velocities_key[0]].differentiate(
        "time", datetime_unit="s"
    )
    ds[accelerations_key[1]] = ds[velocities_key[1]].differentiate(
        "time", datetime_unit="s"
    )
    x = ds.where(ds[accelerations_key[0]] < acc_cut).x
    y = ds.where(ds[accelerations_key[1]] < acc_cut).y
    print(
        f"nb of spike removed { np.isnan(x).sum('time').values} over {ds.dims['time']}"
    )
    ds["x"] = x.interpolate_na("time")
    ds["y"] = y.interpolate_na("time")

    # 6) Box mean on nb_pt_mean
    if nb_pt_mean % 2 == 0:
        warnings.warn("nb_pt_mean should be odd, set to np_pt_window+1")
        nb_pt_mean += 1
    if nb_pt_mean == 0:
        assert False, "np_pt_window=0"

    n = nb_pt_mean // 2
    ds0 = 0
    ds1 = ds
    for i in np.arange(-n, n + 1):
        ds0 += ds1.shift({"time": i})
        ds1 = ds
    ds0 = ds0 / nb_pt_mean

    # test box mean
    assert (
        ds0.isel(time=n) == ds.isel(time=slice(0, nb_pt_mean)).mean()
    ), "pb with mean over n points"

    ds0 = ds0.drop([accelerations_key[0], accelerations_key[1]])

    # Build full dataframe
    df_out = ds0.to_dataframe()

    # import columns/info ex: id or time
    if "id" not in import_columns:
        import_columns += ["id"]
    if import_columns:
        for column in import_columns:
            df_out[column] = df[column][0]

    # fill na
    df_out = df_out.bfill().ffill()

    # compute acceleration
    if geo:
        # initiate lon, lat (needed to compute_acc, even if it is computed from x, y)
        df_out["lon"] = df.lonc.mean()
        df_out["lat"] = df.latc.mean()

    # update lon/lat
    if geo:
        # first reset reference from df
        df_out.geo.set_projection_reference(proj_ref)  # inplace
        df_out.geo.compute_lonlat()  # inplace

    df_out["X"] = np.sqrt(df_out["x"] ** 2 + df_out["y"] ** 2)
    compute_acc(
        df_out, geo, spectral_diff, "spydell", velocities_key, accelerations_key
    )
    # df_out[accelerations_key[2]] = np.sqrt(
    #    df_out[accelerations_key[0]] ** 2 + df_out[accelerations_key[1]] ** 2
    # )

    return df_out


###########################################
# -----------w METHOD------------#


@njit
def advance_search(nt, time, t, i, delta_plus, delta_minus):
    """find next closest neighbourgh by searching in positive and negative
    directions with respect to index i and update delta_minus, delta_plus
    """
    # delta=0 means search as stopped in that direction
    #  compute distances with i+delta_plus and i+delta_minus points
    if delta_plus > 0 and i + delta_plus < nt:
        d_plus = abs(time[i + delta_plus] - t)
    else:
        d_plus = -1.0
    if delta_minus < 0 and i + delta_minus >= 0:
        d_minus = abs(time[i + delta_minus] - t)
    else:
        d_minus = -1.0
    # update delta_plus or delta_minus
    if d_minus != -1 and (
        d_plus == -1 or d_minus <= d_plus
    ):  # correction cas d_plus=d_minus
        i_next = i + delta_minus  # next nearest point
        if i + delta_minus > 0:
            delta_minus += -1
        else:
            # stop search in that direction
            delta_minus = 0
    elif d_plus != -1 and (d_minus == -1 or d_minus > d_plus):
        i_next = i + delta_plus
        if i + delta_plus < nt - 1:
            delta_plus += 1
        else:
            # stop search in that direction
            delta_plus = 0
    else:  # nope (numba '0.56.3')
        #    # should never reach this point
        #    #assert False, (i, delta_minus, delta_plus, d_minus, d_plus)
        #    # AssertionError: (998, -1, 0, 0.9963861521472381, -1.0)
        print(
            ("WARNING : pb advance search", i, delta_minus, delta_plus, d_minus, d_plus)
        )  # ok numba 0.56.3
    return i_next, delta_minus, delta_plus


@njit
def find_nearest_neighboors(time, t, i):
    """Find 3 remaining neighbouring points
    i is a starting value (closest point)
    """
    nt = len(time)
    nb = 5
    ib = [0 for _ in range(nb)]
    ib[0] = i
    # initiate direction for advance search (with problem at boundaries solved)
    if i == nt:  # end-boundary case
        delta_plus = 0
    else:
        delta_plus = 1
    if i == 0:
        delta_minus = 0  # starting boundary case
    else:
        delta_minus = -1  # if not at boundaries delta_minus=-1 and delta_plus=1
    counter = 1
    while counter < nb:
        ib[counter], delta_minus, delta_plus = advance_search(
            nt, time, t, i, delta_plus, delta_minus
        )
        counter += 1
    return np.sort(np.array(ib))


# @guvectorize([(float64[:], float64[:])], '(n)->(n)')
# @guvectorize(["void(float64[:], float64[:])"], '(n)->(n)')
# guvectorize cannot be called from a jit method at the moment, see: https://github.com/numba/numba/issues/5720
# def I_func(v, res):
@njit
def I_func(v):
    I = np.zeros_like(v)
    for i in range(v.shape[0]):
        if v[i] > -1 or v[i] < 1:
            I[i] = v[i]
        else:
            I[i] = 0.0
    return I


@njit
def solve_position_velocity(t_nb, x_nb, time_target):
    # solve for x and u :  x + u*(t_nb-date_target) = x_nb
    t = t_nb - time_target
    t_nbs = np.sort(t_nb)
    dt = t_nbs[-1] - t_nbs[0]
    weights = 70 / 81 * (1 - np.abs(t / dt) ** 3) ** 3 * I_func(t / dt)
    w = np.sum(weights)
    wt = np.sum(weights * t)
    wt2 = np.sum(weights * t**2)
    A = np.array([[w, wt], [wt, wt2]])  # coef gradients
    b = np.array([np.sum(weights * x_nb), np.sum(weights * x_nb * t)])
    out = np.linalg.solve(A, b)
    return out[0], out[1]


@njit
def solve_position_velocity_acceleration(t_nb, x_nb, time_target):
    # solve for x and u :  x + u*(t_nb-date_target) = x_nb
    t_nbs = np.sort(t_nb)
    dt = t_nbs[-1] - t_nbs[0]
    # t = t_nb - time_target
    t = (t_nb - time_target) / dt
    weights = 70 / 81 * (1 - np.abs(t / dt) ** 3) ** 3 * I_func(t / dt)
    w = np.sum(weights)
    wt = np.sum(weights * t)
    wt2 = np.sum(weights * t**2)
    wt3 = np.sum(weights * t**3)
    wt4 = np.sum(weights * t**4)
    A = np.array(
        [[w, wt, wt2 / 2], [wt, wt2, wt3 / 2], [wt2, wt3, wt4 / 2]]
    )  # coef gradients
    b = np.array(
        [
            # np.sum(weights * x_nb),
            # np.sum(weights * x_nb * t),
            # np.sum(weights * x_nb * t**2),
            np.sum(weights * (x_nb - x_nb[0])),
            np.sum(weights * (x_nb - x_nb[0]) * t),
            np.sum(weights * (x_nb - x_nb[0]) * t**2),
        ]
    )
    # out = np.linalg.solve(A, b)# pb of multiple solution or no solution https://stackoverflow.com/questions/13795682/numpy-error-singular-matrix
    out = np.linalg.lstsq(A, b)
    # return out[0], out[1], out[2]
    # return out[0][0], out[0][1], out[0][2]
    # return out[0][0] + x_nb[0], out[0][1], out[0][2]
    return out[0][0] + x_nb[0], out[0][1] / dt, out[0][2] / dt**2


# @njit("UniTuple(float64[:], 2)(float64[:], float64[:], float64[:])")
@njit
def lowess(time, x, time_target, nb=4, degree=2):
    """perform a lowess interpolation

    Parameters
    ----------
    time: np.array
        time array, assumed to be sorted in time, should be floats
    x: np.array
        positions
    time_target: np.array
        target timeline
    nb : number of closest neighboors to consider
    degree : 2 or 3, of the polynomial
    """
    nt = len(time_target)

    assert time_target[0] >= time[0], "time_target[0] is not within time span"
    assert time_target[-1] <= time[-1], "time_target[-1] is not within time span"

    # find closest values
    # d = np.abs(time[:,None] - time_target[None, :]) # nope (numba '0.56.3')
    d = np.abs(time.reshape(len(time), 1) - time_target.reshape(1, nt))

    i_closest = np.argmin(
        d, axis=0
    )  # the indice of the nearest time in the raw time series (time) for each time of the regular time series (time_target)

    x_out = np.full(nt, np.nan)
    u_out = np.full(nt, np.nan)
    a_out = np.full(nt, np.nan)

    for i in prange(nt):
        if degree == 2:
            nb = nb
        if degree == 3:
            nb = nb
        i_nb = np.arange(i_closest[i] - nb // 2, i_closest[i] + nb // 2 + 1)
        a = [i < 0 or i > len(time) - 1 for i in i_nb]  # np.any not ok with numba
        if True in a:
            continue  # start-end edge stay nan values
        # i_nb = find_nearest_neighbours(time, time_target[i], i_closest[i])
        t_nb = time[i_nb]
        x_nb = x[i_nb]
        if degree == 2:
            try:
                x_out[i], u_out[i] = solve_position_velocity(t_nb, x_nb, time_target[i])
            except:
                print(
                    "WARNING :  pb with solve_position_velocity, set to nan",
                    "t_target =",
                    time_target[i],
                )
                x_out[i], u_out[i] = np.nan, np.nan
        elif degree == 3:
            try:
                x_out[i], u_out[i], a_out[i] = solve_position_velocity_acceleration(
                    t_nb, x_nb, time_target[i]
                )
            except:
                print(
                    "WARNING :  pb with solve_position_velocity, set to nan",
                    "t_target =",
                    time_target[i],
                )
                x_out[i], u_out[i], a_out[i] = np.nan, np.nan, np.nan

    return x_out, u_out, a_out


def lowess_smooth(
    df,
    t_target,
    degree=2,
    iteration=3,
    nb=4,
    T_low_pass=None,
    cutoff_low_pass=None,
    velocities_key=("velocity_east", "velocity_north", "velocity"),
    accelerations_key=("acceleration_east", "acceleration_north", "acceleration"),
    import_columns=None,
    spectral_diff=False,
    geo=False,
):
    """perform a lowess interpolation with optional posteriori low pass filter

    Parameters
    ----------
    df: dataframe, must contain x, y
    t_target: `pandas.core.indexes.datetimes.DatetimeIndex` or str
                Output time series, as typically given by pd.date_range or the delta time of the output time series as str
                In this case, t_target is then recomputed taking start-end the start end of the input trajectory and the given delta time
    degree : 2 or 3,
        degree of the polynomial for the lowess method
    iteration : number of time to apply LOWESS (interpolation on t_target at the last iteration)
    nb : number of closest neighboors to consider
    T_low_pass : float
        Filter length in days, if None (default), does not apply filter
    cutoff_low_pass : float
        low pass filter cutoff frequency in cpp
    velocities_key : (,,) of str,
        ex : ('velocity_east','velocity_north', 'velocity') or ('u','v', 'U') etc
    accelerations_key : (,,) of str,
        ex : ('acceleration_east','acceleration_north', 'acceleration') or ('ax','ay', 'Axy') or ('au','av', 'Auv') etc
    import_columns : list of str
        list of df constant columns we want to import (ex: id, platform)
    spectral_diff : boolean,
         if True use spectral differentiation instead of central differentiation
    geo: boolean,
        optional if geo obj with projection
    Return : dataframe with x, y, u, v, ax-ay computed from xy, au-av computed from u-v, and ae-an computed via lowess if degree = 3,+norms, id, platform
    """
    # index = time
    if df.index.name != "time":
        if df.index.name == None:
            df = df.set_index("time")
        else:
            df = df.reset_index().set_index("time")

    # assert x, y in dataframe
    if "x" not in df or "y" not in df:
        assert False, "positions must be labelled as 'x' and 'y'"

    # store projection to align with dataframes produced
    if geo:
        if "lonc" in df:
            if "lonc" not in import_columns:
                import_columns += ["lonc", "latc"]
            proj_ref = (df.lonc.values[0], df.latc.values[0])
        else:
            proj_ref = df.geo.projection_reference

    # time in seconds from first time
    df["date"] = (df.index - df.index.min()) / pd.Timedelta("1s")

    # t_target
    if isinstance(t_target, str):
        t_target = pd.date_range(
            df.index.min().ceil(t_target), df.index.max(), freq=t_target
        )
    # t_ target in seconds from first time
    date_target = (t_target - t_target[0]) / pd.Timedelta("1s")

    x_out = df.x.values
    y_out = df.y.values
    time = df.date.values
    time_target = time

    # apply lowess
    for i in range(iteration):
        if i == iteration - 1:
            time_target = (
                date_target.values
            )  # final interpolation on regular t_target time
        x_out, u_out, ax_out = lowess(time, x_out, time_target, nb=nb, degree=degree)
        y_out, v_out, ay_out = lowess(time, y_out, time_target, nb=nb, degree=degree)

    # dataframe
    if degree == 2:
        df_out = pd.DataFrame(
            {
                "x": x_out,
                "y": y_out,
                velocities_key[0]: u_out,
                velocities_key[1]: v_out,
                "time": t_target,
            }
        )
    elif degree == 3:
        df_out = pd.DataFrame(
            {
                "x": x_out,
                "y": y_out,
                velocities_key[0]: u_out,
                velocities_key[1]: v_out,
                accelerations_key[0]: ax_out,
                accelerations_key[1]: ay_out,
                "time": t_target,
            }
        )
        # df_out[accelerations_key[2]] = np.sqrt(
        #    df_out[accelerations_key[0]] ** 2 + df_out[accelerations_key[1]] ** 2
        # )

    df_out = df_out.set_index("time")

    # fill na (misssing at least 2 values at the border, because lowess need 2 nearest neightbours at each sides
    df_out = df_out.bfill().ffill()

    # APPLY LOW PASS -> only on velocities + reintegrate positions
    if T_low_pass:
        df_out = low_pass_(df_out, T_low_pass, cutoff_low_pass, velocities_key)
        print(f"LOW-PASS : {cutoff_low_pass}cpd with {T_low_pass}days length")

    # import columns/info ex: id or time
    if "id" not in import_columns:
        import_columns += ["id"]
    if import_columns:
        for column in import_columns:
            df_out[column] = df[column][0]

    # compute acceleration
    if geo:
        # initiate lon, lat (needed to compute_acc, even if it is computed from x, y)
        df_out["lon"] = df.lonc.mean()
        df_out["lat"] = df.latc.mean()
        if degree != 3:
            compute_acc(
                df_out, geo, spectral_diff, "lowess", velocities_key, accelerations_key
            )

    # update lon/lat
    if geo:
        # first reset reference from df
        df_out.geo.set_projection_reference(proj_ref)  # inplace
        df_out.geo.compute_lonlat()  # inplace
    df_out["X"] = np.sqrt(df_out["x"] ** 2 + df_out["y"] ** 2)
    df_out[velocities_key[2]] = np.sqrt(
        df_out[velocities_key[0]] ** 2 + df_out[velocities_key[1]] ** 2
    )

    # df_out[accelerations_key[2]] = np.sqrt(
    #    df_out[accelerations_key[0]] ** 2 + df_out[accelerations_key[1]] ** 2
    # )

    return df_out


###########################################
# -----------LOWPASS------------#
def posteriori_low_pass_xy(
    df,
    T=1,
    cutoff=13,
    velocities_key=("velocity_east", "velocity_north", "velocity"),
    accelerations_key=("acceleration_east", "acceleration_north", "acceleration"),
    import_columns=["id"],
):
    """Apply low pass filter to a smoothed trajectory a posteriori

    Parameters
    ----------
    df: dataframe, must contain x, y
    T : float
        Filter length in days
    cutoff : float
        low pass filter cutoff frequency in cpp
    import_columns : list of str
        list of df constant columns we want to import (ex: id, platform)
    velocities_key : (,,) of str,
        ex : ('velocity_east','velocity_north', 'velocity') or ('u','v', 'U') etc
    accelerations_key : (,,) of str,
        ex : ('acceleration_east','acceleration_north', 'acceleration') or ('ax','ay', 'Axy') or ('au','av', 'Auv') etc
    Return : dataframe with x, y, u, v, ax-ay computed from xy, au-av computed from u-v, accelerations, +norms, id, platform
    """

    from scipy.signal import filtfilt
    from scipy.integrate import cumulative_trapezoid
    from scipy.optimize import minimize

    # coefficients
    dt = df.dt.mean() / 3600 / 24  # in days
    from pynsitu.tseries import generate_filter

    taps = generate_filter(band="low", dt=dt, T=T, bandwidth=cutoff)
    dff = df[["x", "y"]]
    # apply filter
    dff["x"] = filtfilt(taps, 1, df.x.values)
    dff["y"] = filtfilt(taps, 1, df.y.values)

    # recompute velocities
    compute_velocities(
        dff,
        "index",
        names=velocities_key,
        distance="xy",
        inplace=True,
        centered=True,
        fill_startend=True,
    )

    # recompute acceleration
    compute_accelerations(
        dff,
        from_=("xy", "x", "y"),
        names=accelerations_key,
        centered_velocity=True,
        time="index",
        fill_startend=True,
        inplace=True,
        keep_dt=False,
    )

    # compute_accelerations(
    #    dff,
    #    from_=("velocities", velocities_key[0], velocities_key[1]),
    #    names=("au", "av", "Auv"),
    #    centered_velocity=True,
    #    time="index",
    #    fill_startend=True,
    #    inplace=True,
    #    keep_dt=True,
    # )

    # import columns/info ex: id or time
    if "id" not in import_columns:
        import_columns += ["id"]
    if import_columns:
        for column in import_columns:
            dff[column] = df[column][0]

    dff["X"] = np.sqrt(dff["x"] ** 2 + dff["y"] ** 2)
    dff[velocities_key[2]] = np.sqrt(
        dff[velocities_key[0]] ** 2 + dff[velocities_key[1]] ** 2
    )
    return dff


def posteriori_low_pass_uv(
    df,
    T=20,
    cutoff=4,
    velocities_key=("velocity_east", "velocity_north", "velocity"),
    accelerations_key=("acceleration_east", "acceleration_north", "acceleration"),
    import_columns=["id"],
):
    """Apply low pass filter to a smoothed trajectory a posteriori

    Parameters
    ----------
    df: dataframe, must contain x, y
    T : float
        Filter length in days
    cutoff : float
        low pass filter cutoff frequency in cpp
    import_columns : list of str
        list of df constant columns we want to import (ex: id, platform)

    Return : dataframe with x, y, u, v, ax-ay computed from xy, au-av computed from u-v, accelerations, +norms, id, platform
    """

    from scipy.signal import filtfilt
    from scipy.integrate import cumulative_trapezoid
    from scipy.optimize import minimize

    # coefficients
    dt = df.dt.mean() / 3600 / 24  # in days
    from pynsitu.tseries import generate_filter

    taps = generate_filter(band="low", dt=dt, T=T, bandwidth=cutoff)
    dff = df[[velocities_key[0], velocities_key[1]]]
    # apply filter
    dff[velocities_key[0]] = filtfilt(taps, 1, df[velocities_key[0]].values)
    dff[velocities_key[1]] = filtfilt(taps, 1, df[velocities_key[1]].values)

    # recompute position
    # ms_x, ms_y = (df.x**2).mean(), (df.y**2).mean()
    x_cum = cumulative_trapezoid(dff.u, dx=df.dt.mean(), initial=0)
    y_cum = cumulative_trapezoid(dff.v, dx=df.dt.mean(), initial=0)

    def msx_difference(x_0):
        return ((df.x - x_0 - x_cum) ** 2).mean()

    def msy_difference(y_0):
        return ((df.y - y_0 - y_cum) ** 2).mean()

    x_0 = minimize(msx_difference, df.x[0]).x
    y_0 = minimize(msy_difference, df.y[0]).x

    dff["x"] = x_0 + x_cum
    dff["y"] = y_0 + y_cum

    # recompute acceleration
    # compute_accelerations(
    #    dff,
    #    from_=("xy", "x", "y"),
    #    names=accelerations_key,
    #    centered_velocity=True,
    #    time="index",
    #    fill_startend=True,
    #    inplace=True,
    #    keep_dt=False,
    # )

    compute_accelerations(
        dff,
        from_=("velocities", velocities_key[0], velocities_key[1]),
        names=accelerations_key,
        centered_velocity=True,
        time="index",
        fill_startend=True,
        inplace=True,
        keep_dt=True,
    )

    # import columns/info ex: id or time
    if "id" not in import_columns:
        import_columns += ["id"]
    if import_columns:
        for column in import_columns:
            dff[column] = df[column][0]

    dff["X"] = np.sqrt(dff["x"] ** 2 + dff["y"] ** 2)
    dff[velocities_key[2]] = np.sqrt(
        dff[velocities_key[0]] ** 2 + dff[velocities_key[1]] ** 2
    )
    return dff


# low_pass to integrate in a smmothing function
def low_pass_(
    df, T=1, cutoff=11.5, velocities_key=("velocity_east", "velocity_north", "velocity")
):
    """apply low pass filter on velocity and reintegrate x, y
    Parameters
    ----------
    df: dataframe, must contain u, v
    cutoff : float,
        low pass filter cutoff frequency in cpp
    T : float
        Filter length in days
    velocities_key : (,,) of str,
        ex : ('velocity_east','velocity_north', 'velocity') or ('u','v', 'U') etc
    Return : dataframe with x, y, u, v
    """

    from scipy.signal import filtfilt
    from scipy.integrate import cumulative_trapezoid
    from scipy.optimize import minimize

    # index = time
    if df.index.name != "time":
        if df.index.name == None:
            df = df.set_index("time")
        else:
            df = df.reset_index().set_index("time")
    # test regularly sampled
    dt = df.index[1] - df.index[0]
    assert (df.index[1:] - df.index[:-1]).mean() == dt, "Not regularly sampled"

    # coefficients
    from pynsitu.tseries import generate_filter

    taps = generate_filter(
        band="low", dt=dt / pd.Timedelta("1d"), T=T, bandwidth=cutoff
    )

    dff = df[[velocities_key[0], velocities_key[1]]]  # create dff

    # apply filter
    dff[velocities_key[0]] = filtfilt(
        taps,
        1,
        df[velocities_key[0]].values,
        padlen=len(df[velocities_key[0]].values) - 1,
    )  # CHECK WITH AURELIEN
    dff[velocities_key[1]] = filtfilt(
        taps,
        1,
        df[velocities_key[1]].values,
        padlen=len(df[velocities_key[1]].values) - 1,
    )

    # recompute position
    # ms_x, ms_y = (df.x**2).mean(), (df.y**2).mean()
    x_cum = cumulative_trapezoid(
        dff[velocities_key[0]], dx=dt / pd.Timedelta("1s"), initial=0
    )
    y_cum = cumulative_trapezoid(
        dff[velocities_key[1]], dx=dt / pd.Timedelta("1s"), initial=0
    )

    def msx_difference(x_0):
        return ((df.x - x_0 - x_cum) ** 2).mean()

    def msy_difference(y_0):
        return ((df.y - y_0 - y_cum) ** 2).mean()

    x_0 = minimize(msx_difference, df.x[0]).x
    y_0 = minimize(msy_difference, df.y[0]).x

    dff["x"] = x_0 + x_cum
    dff["y"] = y_0 + y_cum

    return dff


###########################################
# -----------APPLY INTERPOLATIONS METHODS------------#


def find_gap(df_gap, maxgap):
    """Find gaps (return time start, time end) bigger than dtmax in the dataset

    Parameters
    ----------
    df_gap : original dataframe

    maxgap: float
        max gap length in seconds

    """
    df_gap = df_gap.reset_index()
    time_end = df_gap[df_gap.dt > maxgap].time
    index_start = time_end.index.values - 1
    time_start = df_gap.iloc[index_start].time
    return time_start.values, time_end.values


def divide_blocs(df, t_target, maxgap):
    """Cut out gaps bigger than maxgap and return blocs

    Parameters
    ----------
    df : original dataframe
    t_target : pd.datetime index,
        interpolation times
    maxgap: float
        max gap length in seconds
    Returns
    ----------
    DF : list of dataframe,
        blocs without gaps bigger than maxgap
    DF_target :list of pd.date_time_index,
        list of times out gaps bigger than maxgap
    DF_target_gap : list of pd.date_time_index,
        list of times in gaps bigger than maxgap
    """
    dti = t_target[1] - t_target[0]
    # check that maxgap > t_target delta
    assert maxgap > dti / pd.Timedelta(
        "1s"
    ), "maxgap should be bigger than the t_target delta"

    # divide into blocs if gap bigger than maxgap
    DF = []
    DF_target = []
    DF_target_gap = []

    df = df.reset_index()
    ts, te = find_gap(df, maxgap)
    tcut = np.sort([df.time.min()] + list(ts) + list(te) + [df.time.max()])
    tcut0 = tcut
    tcut_to_remove = []
    for i in range(0, len(tcut) - 1, 2):
        test1 = df.time >= tcut[i]
        test2 = df.time <= tcut[i + 1]
        test = test1 & test2
        df_ = df[test]
        # if less than two points in the
        if (
            len(df_) < 5 or (df_.time.max() - df_.time.min()) < dti * 6
        ):  # minimum of 5 values for matrice n in variational method not to be singular
            print(
                "WARNING: not enougth points between two gaps, include them in the gap"
            )
            tcut_to_remove.append(tcut[i])
            tcut_to_remove.append(tcut[i + 1])
            continue
        else:
            DF.append(df_)

    tcut = [t for t in tcut if t not in tcut_to_remove]
    for i in range(0, len(tcut) - 1, 2):
        test = (t_target >= tcut[i]) & (t_target <= tcut[i + 1])
        DF_target.append(t_target[test])
        DF_target_gap.append(t_target[np.logical_not(test)])
    return DF, DF_target, DF_target_gap


def gap_array(time, t_target):
    """Returns time distance between time in t_target and their nearest neightbor in time

    Parameters
    ----------
    time : ndarray of datetime,
    t_target : ndarray of datetime,
    """
    nt = len(t_target)
    assert t_target[0] >= time[0], "time_target[0] is not within time span"
    assert t_target[-1] <= time[-1], "time_target[-1] is not within time span"

    d = np.abs(time.reshape(len(time), 1) - t_target.reshape(1, nt))
    i_closest = np.argmin(d, axis=0)
    t_closest = np.min(d, axis=0)
    return (t_target - time[i_closest]) / pd.Timedelta("1s")


def smooth(
    df,
    method,
    t_target,
    maxgap=4 * 3600,
    parameters=dict(),
    velocities_key=("velocity_east", "velocity_north", "velocity"),
    accelerations_key=("acceleration_east", "acceleration_north", "acceleration"),
    import_columns=["id"],
    spectral_diff=False,
    geo=True,
):
    """
    Smooth and interpolated a trajectory
    Parameters:
    -----------
            df :  dataframe with raw trajectory,
                must contain 'time','x','y', 'u', 'v', 'dt'
            method : str
                smoothing method among : 'spydell', 'variational' or 'lowess'
            t_target: `pandas.core.indexes.datetimes.DatetimeIndex` or str
                Output time series, as typically given by pd.date_range or the delta time of the output time series as str
                In this case, t_target is then recomputed taking start-end the start end of the input trajectory and the given delta time
            maxgap : float,
                max gap tolerated in SECONDS
            parameters : dict,
                contains all parameters to give to method :
                - variational : dict(acc_cut =, position_error=, acceleration_amplitude=, acceleration_T=,time_chunk=)
                - lowess : dict(degree=)
                - spydell : dict(acc_cut =, nb_pt_mean=)
            velocities_key : (,,) of str,
                ex : ('velocity_east','velocity_north', 'velocity') or ('u','v', 'U') etc
            accelerations_key : (,,) of str,
                ex : ('acceleration_east','acceleration_north', 'acceleration') or ('ax','ay', 'Axy') or ('au','av', 'Auv') etc
            import_columns : list of str,
                list of df constant columns we want to import (ex: id, platform)
            geo: boolean,
                optional if geo obj with projection
            acc: boolean,
                optional compute acceleration
    Return : interpolated dataframe with x, y, u, v, ax-ay computed from xy, au-av computed from u-v, +norms, id, platform with index time
    """
    # check maxgap
    assert maxgap > df.dt.min(), "maxgap smaller than the smallest dt"

    # index = time
    if df.index.name != "time":
        if df.index.name == None:
            df = df.set_index("time")
        else:
            df = df.reset_index().set_index("time")

    # check time index is sorted
    if not df.index.is_monotonic_increasing:
        df.sort_index(inplace=True)

    # t_target
    if isinstance(t_target, str):
        t_target = pd.date_range(
            df.index.min().ceil(t_target), df.index.max(), freq=t_target
        )  # add ceil so all drifters in smooth_all are on the same time grid
    else:
        # enforce t_target type
        t_target = pd.DatetimeIndex(t_target)

    # divide into blocs if gap bigger than maxgap
    DF, DF_target, DF_target_gap = divide_blocs(df, t_target, maxgap)
    print(f"Divided into {len(DF)} segments")

    # APPLY ON SEGMENTS
    if method == "variational" or method == "spydell":
        DF_out = []
        for i in range(len(DF)):
            try:
                df_, t_target_ = DF[i], DF_target[i]
            except:
                continue  # if DF_target is empty
            if method == "variational":
                param = [
                    "acc_cut",
                    "position_error",
                    "acceleration_amplitude",
                    "acceleration_T",
                    "time_chunk",
                ]
                assert np.all(
                    [p in param for p in parameters]
                ), f"parameters keys must be in {param}"
                # try :
                df_out = variational_smooth(
                    df_,
                    t_target_,
                    **parameters,
                    velocities_key=velocities_key,
                    accelerations_key=accelerations_key,
                    import_columns=import_columns,
                    spectral_diff=spectral_diff,
                    geo=geo,
                )
                # except :
                # assert False, (df_, t_target_, len(df_), len(t_target_))

            elif method == "spydell":
                param = ["acc_cut", "nb_pt_mean"]
                assert np.all(
                    [p in param for p in parameters]
                ), f"parameters keys must be in {param}"
                try:
                    df_out = spydell_smooth(
                        df_,
                        t_target_,
                        **parameters,
                        velocities_key=velocities_key,
                        accelerations_key=accelerations_key,
                        import_columns=import_columns,
                        spectral_diff=spectral_diff,
                        geo=geo,
                    )
                except:
                    assert False, (df_, t_target_, len(df_), len(t_target_))
            DF_out.append(df_out)
        dfo = pd.concat(DF_out)
        dfo = dfo.reindex(t_target).interpolate(
            method="slinear", limit_area="inside"
        )  # LINEAR INTERPOLATION IN GAPS

    # APPLY ON the whole trajectory (linear interpolation in gaps already done by LOWESS)
    elif method == "lowess":
        param = ["degree", "iteration", "nb", "T_low_pass", "cutoff_low_pass"]
        assert np.all(
            [p in param for p in parameters]
        ), f"parameters keys must be in {param}"
        try:
            dfo = lowess_smooth(
                df,
                t_target,
                **parameters,
                velocities_key=velocities_key,
                accelerations_key=accelerations_key,
                import_columns=import_columns,
                spectral_diff=spectral_diff,
                geo=geo,
            )
        except:
            assert False, (df, t_target)
    else:

        assert False, 'method must be "variational", "lowess", or "spydell"'
    # gap_mask :
    t_target_values = pd.DatetimeIndex(
        pd.concat([pd.Series(dti) for dti in DF_target])
    ).sort_values()
    gapmask = pd.DataFrame(index=t_target_values)
    gapmask["gap_mask"] = 1  # 1 where a value is computed
    gapmask = gapmask.reindex(
        t_target, fill_value=0
    )  # 0 in gaps = linearly interpolated data
    dfo = pd.concat([dfo, gapmask], axis=1)

    # gap value : time distance to the nearest neightbors in seconds
    dfo["gaps"] = gap_array(df.index.values, t_target.values)

    # import columns/info ex: id or time
    if "id" not in import_columns:
        import_columns += ["id"]
    if import_columns:
        for column in import_columns:
            dfo[column] = df[column][0]

    return dfo


def smooth_all(
    df,
    method,
    t_target,
    maxgap=4 * 3600,
    parameters=dict(),
    velocities_key=("velocity_east", "velocity_north", "velocity"),
    accelerations_key=("acceleration_east", "acceleration_north", "acceleration"),
    import_columns=["id"],
    spectral_diff=True,
    geo=True,
):
    """
    Smooth and interpolated all trajectories
    Parameters:
    -----------
            df :  dataframe with raw trajectory,
                must contain 'time', 'velocity_east', 'velocity_north'
            method : str
                smoothing method among : 'spydell', 'variational' or 'lowess'
            t_target: `pandas.core.indexes.datetimes.DatetimeIndex` or str
                Output time series, as typically given by pd.date_range or the delta time of the output time series as str
                In this case, t_target is then recomputed taking start-end the start end of the input trajectory and the given delta time
            maxgap : float,
                max gap tolerated in SECONDS
            parameters : dict,
                contains all parameters to give to method :
                - variational : dict(acc_cut =, position_error=, acceleration_amplitude=, acceleration_T=,time_chunk=)
                - lowess : dict(degree=)
                - spydell : dict(acc_cut =, nb_pt_mean=)
            velocities_key : (,,) of str,
                ex : ('velocity_east','velocity_north', 'velocity') or ('u','v', 'U') etc
            accelerations_key : (,,) of str,
                ex : ('acceleration_east','acceleration_north', 'acceleration') or ('ax','ay', 'Axy') or ('au','av', 'Auv') etc
            import_columns : list of str,
                list of df constant columns we want to import (ex: id, platform)
            geo: boolean,
                optional if geo obj with projection
            acc: boolean,
                optional compute acceleration
    Return : interpolated dataframe with x, y, u, v, ax-ay computed from xy, au-av computed from u-v, +norms, id, platform with index time
    """
    dfa = df.groupby("id").apply(
        smooth,
        method,
        t_target,
        maxgap,
        parameters,
        velocities_key,
        accelerations_key,
        import_columns,
        spectral_diff,
        geo,
    )
    dfa = (
        dfa.reset_index(level="id", drop=True)
        .reset_index()
        .rename(columns={"index": "time"})
        .set_index("time")
    )
    return dfa


#################################################################################
# ------------------------ time window processing -------------------------------


def time_window_processing(
    df,
    myfun,
    T,
    overlap=0.5,
    id_label="id",
    dt=None,
    limit=None,
    geo=None,
    xy=None,
    **myfun_kwargs,
):
    """Break each drifter time series into time windows and process each windows

    myfun signature must be myfun(df, **kwargs) and it must return a pandas Series
    Drop duplicates if a `date` column is present

    Parameters
    ----------
        df: Dataframe
            This dataframe represents a drifter time series
        myfun
            Method that will be applied to each window
        T: float, pd.Timedelta
            Length of the time windows, must be in the same dtype and units than column "time"
        overlap: float
            Amount of overlap between temporal windows.
            Should be between 0 and 1.
            Default is 0.5
        id_label: str, optional
            Label used to identify drifters
        dt: float, str
            Conform time series to some time step, if string must conform to rule option of
            pandas resample method
        geo: boolean
            Turns on geographic processing of spatial coordinates
        xy: tuple
            specify x, y spatial coordinates if not geographic
        **myfun_kwargs
            Keyword arguments for myfun

    """
    from pandas.api.types import is_datetime64_any_dtype as is_datetime

    if hasattr(df, id_label):
        dr_id = df[id_label].unique()[0]
    elif df.index.name == id_label:
        dr_id = df.index.unique()[0]
    elif hasattr(df, "name"):
        # when mapped after groupby
        dr_id = df.name
    else:
        assert False, "Cannot find float id"
    #
    # dim_x, dim_y, geo = guess_spatial_dims(df)
    if geo != None:
        # old, used to go through 3 vectors
        # df = compute_vector(df, lon_key=dim_x, lat_key=dim_y)
        # new, leverage GeoAccessor
        df.geo.project()
        proj = df.geo.projection

    if df.index.name == "time":
        df = df.reset_index()

    # drop duplicated values - requires a date column
    if "date" in df.columns:
        df = df.drop_duplicates(subset="date")
    # p = p.where(p.time.diff() != 0).dropna() # duplicates - old

    df = df.sort_values("time")
    t_is_date = is_datetime(df["time"])
    if isinstance(T, str):
        T = pd.Timedelta(T)

    # temporal resampling to fill gaps
    if dt != None:
        if isinstance(dt, float):
            # enforce regular sampling
            tmin, tmax = df.index[0], df.index[-1]
            tmax = tmin + int((tmax - tmin) / dt) * dt
            regular_time = np.arange(tmin, tmax, dt)
            df = df.reindex(regular_time).interpolate(limit=limit)
        elif isinstance(dt, str):
            # df = df.set_index("date").resample(dt).pad().reset_index()
            c = None
            if t_is_date:
                c = "time"
            elif "date" in df.columns and is_datetime(df["date"]):
                c = "date"
            assert (
                c is not None
            ), "dt is str but no `time` nor `date` columns are datetime-like"
            df = df.set_index(c).resample(dt).interpolate(limit=limit)
            # fill some NaNs
            df[id_label] = df[id_label].interpolate()
            if c == "date":
                df["time"] = df["time"].interpolate()
            df = df.reset_index()
            # by default converts to days then
            dt = pd.Timedelta(dt) / pd.Timedelta("1d")
        if geo is not None:
            df.geo.compute_lonlat()

    #
    df = df.set_index("time")
    tmin, tmax = df.index[0], df.index[-1]
    #
    if geo is not None:
        xy = ["lon", "lat"]
    elif xy is not None:
        xy = list(xy)
    else:
        xy = []
    out = None
    t = tmin
    while t + T < tmax:
        #
        _df = df.reset_index()
        _df = _df.loc[(_df.time >= t) & (_df.time < t + T)].set_index("time")
        # _df = df.loc[t : t + T] # not robust for floats
        if t_is_date:
            # iloc because pandas include the last date
            _df = _df.iloc[:-1, :]
        # compute average position
        if geo:
            # x, y = mean_position(_df, Lx=Lx)
            x, y = proj.xy2lonlat(_df["x"].mean(), _df["y"].mean())
        else:
            x, y = _df[xy[0]].mean(), _df[xy[1]].mean()
        # apply myfun
        myfun_out = myfun(df, **myfun_kwargs)
        if out is None:
            size_out = myfun_out.index.size
            columns_out = xy + ["id"] + list(myfun_out.index)
            out = pd.DataFrame({c: [] for c in columns_out})
        # combine with mean position and time
        if myfun_out.index.size == size_out:
            out.loc[t + T / 2.0] = [x, y] + [dr_id] + list(myfun_out)
        t += T * (1 - overlap)
    out.index = out.index.rename("time")
    return out


# kept for archives - not used anymore
def _mean_position(df, Lx=None):
    """Compute the mean position of a dataframe
    !!! to be overhauled !!!

    Parameters:
    -----------
        df: dafaframe
            dataframe containing position data
        Lx: float, optional
            Domain width for periodical domains
    """
    # guess grid type
    dim_x, dim_y, geo = guess_spatial_dims(df)
    # lon = next((c for c in df.columns if "lon" in c.lower()), None)
    # lat = next((c for c in df.columns if "lat" in c.lower()), None)
    if geo:
        lon, lat = dim_x, dim_y
        if "v0" not in df:
            df = compute_vector(df, lon_key=lon, lat_key=lat)
        mean = compute_lonlat(
            df.mean(),
            dropv=True,
            lon_key=lon,
            lat_key=lat,
        )
        return mean[lon], mean[lat]
    else:
        if Lx is not None:
            x = (
                (
                    np.angle(np.exp(1j * (df[dim_x] * 2.0 * np.pi / L - np.pi)).mean())
                    + np.pi
                )
                * Lx
                / 2.0
                / np.pi
            )
        else:
            x = df[dim_x].mean()
        y = df[dim_y].mean()
        return x, y


#################################################################################
# ------------------------ OPTIMIZE METHODS -------------------------------

optimized_parameters_lowess = dict(
    degree=2, iteration=2, T_low_pass=0.45, nb=4, cutoff_low_pass=8
)
optimized_parameters_var = dict(
    acc_cut=3e-4,
    position_error=70,
    acceleration_amplitude=7.5e-6,
    acceleration_T=0.15 * 86400,
    time_chunk=2,
)
optimized_parameters_spydell = dict(nb_pt_mean=7, acc_cut=3e-4)
maxgap = 3 * 3600
