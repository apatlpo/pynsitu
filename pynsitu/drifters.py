import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from numpy.linalg import inv
from scipy.linalg import solve
from scipy.sparse import diags
from scipy.special import erf

from .geo import GeoAccessor


# ------------------------ drifter data cleaning --------------------------------


def despike_isolated(df, acceleration_threshold, verbose=True):
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
    verbose: boolean
        Outputs number of anomalous values detected
        Default is True

    Returns
    -------
    df: `pandas.DataFrame`
        Output dataframe with spikes removed.

    """

    assert "acceleration" in df.columns, (
        "'acceleration' should be a column. You may need to leverage the "
        + "geo accessor first (pynsitu.geo.GeoAccessor) with "
        + "`df.geo.compute_velocities(acceleration=True)``"
    )

    # first pass: anomalous large acceleration values
    spikes = df[df["acceleration"] > acceleration_threshold]

    # second pass: seach for adequate sign reversals
    validated_single_spikes = []
    for t in spikes.index:
        C = []
        # check for a double sign reversal of acceleration
        for _dir in ["east", "north"]:
            if t > df.index[0] and t < df.index[-1]:
                am = df.loc[:t, "acceleration_" + _dir].iloc[-2]
                a = spikes.loc[t, "acceleration_" + _dir]
                ap = df.loc[t:, "acceleration_" + _dir].iloc[1]
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


def resample_smooth(
    df,
    t_target,
    position_error,
    acceleration_amplitude,
    acceleration_T,
    velocity_acceleration=True,
    time_chunk=2,
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

    Parameters
    ----------
    df: `pandas.DataFrame`
        Input drifter time series, must contain projected positions (`x` and `y`)
    t_target: `pandas.core.indexes.datetimes.DatetimeIndex`
        Output time series, as typically given by pd.date_range
        Note that the problem seems ill-posed in the downsampling case ... need
        to be fixed
    position_error: float
        Position error in meters
    acceleration_amplitude: float
        Acceleration typical amplitude
    acceleration_T: float
        Acceleration decorrelation timescale in seconds
    velocity_acceleration: boolean, optional
        Updates velocity and acceleration
    time_chunk: int/float
        Maximum time chunk (in days) to process at once.
        Data is processed by chunks and patched together.
    """

    T = (t_target[-1] - t_target[0]) / pd.Timedelta("1D")

    # store projection to align with dataframes produced
    proj = df.geo.projection_reference

    if T < time_chunk * 1.1:

        dfi = _resample_smooth_one(
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
        delta = pd.Timedelta("3H")
        R = []
        for time in D:
            df_chunk = df.loc[
                (df.index > time[0] - delta) & (df.index < time[-1] + delta)
            ]
            df_chunk.geo.set_projection_reference(proj)
            df_chunk_smooth = _resample_smooth_one(
                df_chunk, time, position_error, acceleration_amplitude, acceleration_T
            )
            R.append(df_chunk_smooth)

        # brute concatenation: introduce strong discontinuities in velocity/acceleration
        # dfi = pd.concat(R)
        # removes duplicated times
        # dfi = dfi.loc[~dfi.index.duplicated(keep="first")]

        ## patch timeseries together, not that simple ...

        col_nums = [c for c in df.columns if is_numeric_dtype(df[c].dtype)]
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
            df_left = df_left.reindex(index, method=None)
            df_right = df_right.reindex(index, method=None)

            # build weights
            w = (1 - erf((df_left.index.to_series() - t_mid) * 5 / delta)) * 0.5
            # note: the width in the error function needs to be much smaller than delta.
            # Discontinuities visible on acceleration are visible otherwise
            # A factor 5 is chosen here

            df_left = df_left.fillna(df_right)
            df_right = df_right.fillna(df_left)

            # patch
            for c in col_nums:
                df_right.loc[:, c] = df_left.loc[:, c] * w + df_right.loc[:, c] * (
                    1 - w
                )

            i += 1

        dfi = df_right
        dfi.geo.set_projection_reference(proj)
        # lon/lat are updated in _resample_smooth_one but x/y have been modified
        # with the patching and hence need to be recomputed
        dfi.geo.compute_lonlat()  # inplace

    # recompute velocity, should be an option?
    if velocity_acceleration:
        dfi = dfi.geo.compute_velocities()
        dfi = dfi.geo.compute_accelerations()

    return dfi


def _resample_smooth_one(
    df,
    t_target,
    position_error,
    acceleration_amplitude,
    acceleration_T,
):
    """core processing for resample_smooth, process one time window"""

    # init final structure
    dfi = (
        df.reindex(df.index.union(t_target), method=None)
        .interpolate(method="time")
        .bfill()
        .ffill()
        .reindex(t_target)
    )
    dfi.index.name = "time"

    # exponential acceleration autocorrelation
    R = lambda dt: acceleration_amplitude**2 * np.exp(-np.abs(dt / acceleration_T))
    # get operators
    L, I = _get_smoothing_operators(t_target, df.index, position_error, R)

    # x
    dfi["x"] = solve(L, I.T.dot(df["x"].values))

    # y
    dfi["y"] = solve(L, I.T.dot(df["y"].values))

    # update lon/lat
    # first reset reference from df
    dfi.geo.set_projection_reference(df.geo._geo_proj_ref)  # inplace
    dfi.geo.compute_lonlat()  # inplace

    return dfi


def _get_smoothing_operators(t_target, t, position_error, acceleration_R):
    """Core operators in order to minimize:
        (Ix - x_obs)^2 / e_x^2 + (D2 x)^T R^{-1} (D2 x)
    where R is the acceleration autocorrelation, assumed to follow

    """

    # assumes t_target is uniform
    dt = t_target[1] - t_target[0]

    # build linear interpolator
    Nt = t_target.size
    I = np.zeros((t.size, Nt))
    i_t = np.searchsorted(t_target, t)
    i = np.where((i_t > 0) & (i_t < Nt))[0]
    j = i_t[i]
    w = (t[i] - t_target[j - 1]) / dt
    I[i, j - 1] = w
    I[i, j] = 1 - w

    # second order derivative
    one_second = pd.Timedelta("1S")
    dt2 = (dt / one_second) ** 2
    D2 = diags([1 / dt2, -2 / dt2, 1 / dt2], [-1, 0, 1], shape=(Nt, Nt)).toarray()
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


def _divide_into_time_chunks(time, T, overlap=0.1):
    """Divide a dataframe into chunks of duration T (in days)

    Parameters
    ----------
    time: pd.DatetimeIndex
        Timeseries
    T: float
        Size of time chunks in days

    """
    Td = pd.Timedelta("1D") * T

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


# ------------------------ time window processing -------------------------------


def time_window_processing(
    df,
    myfun,
    columns,
    T,
    N,
    spatial_dims=None,
    Lx=None,
    overlap=0.5,
    id_label="id",
    dt=None,
    geo=None,
    **myfun_kwargs,
):
    """Break each drifter time series into time windows and process each windows

    Parameters
    ----------
        df: Dataframe
            This dataframe represents a drifter time series
        T: float
            Length of the time windows, must be in the same units that column "time"
        myfun
            Method that will be applied to each window
        columns: list of str
            List of columns of df that will become inputs of myfun
        N: int
            Length of myfun outputs
        spatial_dims: tuple, optional
            Tuple indicating column labels for spatial coordinates.
            Guess otherwise
        Lx: float
            Domain width for periodical domains in x direction
        overlap: float
            Amount of overlap between temporal windows.
            Should be between 0 and 1.
            Default is 0.5
        id_label: str, optional
            Label used to identify drifters
        dt: float, str
            Conform time series to some time step, if string must conform to rule option of
            pandas resample method
        geo:
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
    if geo is not None:
        # old, used to go through 3 vectors
        # df = compute_vector(df, lon_key=dim_x, lat_key=dim_y)
        # new, leverage GeoAccessor
        df.geo.project()
        proj = df.geo.projection
    #
    # drop duplicated values
    df = df.drop_duplicates(subset="date")
    # p = p.where(p.time.diff() != 0).dropna() # duplicates - old
    #
    df = df.sort_values("time")
    # temporal resampling to fill gaps
    if dt is not None:
        if isinstance(dt, float):
            # enforce regular sampling
            tmin, tmax = df.index[0], df.index[-1]
            tmax = tmin + int((tmax - tmin) / dt) * dt
            regular_time = np.arange(tmin, tmax, dt)
            df = df.reindex(regular_time).interpolate()
        elif isinstance(dt, str):
            # df = df.set_index("date").resample(dt).pad().reset_index()
            df = df.set_index("date").resample(dt).interpolate().reset_index()
            # by default converts to days then
            dt = pd.Timedelta(dt) / pd.Timedelta("1D")
        if geo is not None:
            # old
            # df = compute_lonlat(
            #    df,
            #    lon_key=dim_x,
            #    lat_key=dim_y,
            # )
            # new
            df.geo.compute_lonlat()
    #
    df = df.set_index("time")
    tmin, tmax = df.index[0], df.index[-1]
    t_is_date = is_datetime(df.index)
    #
    # need to create an empty dataframe, in case the loop below is empty
    # get column names from fake output:
    myfun_out = myfun(*[None for c in columns], N, dt, **myfun_kwargs)
    size_out = myfun_out.index.size
    #
    columns_out = ["x", "y"] + ["id"] + list(myfun_out.index)
    out = pd.DataFrame({c: [] for c in columns_out})
    t = tmin
    while t + T < tmax:
        #
        _df = df.loc[t : t + T]
        if t_is_date:
            # iloc because pandas include the last date
            _df = _df.iloc[:-1, :]
        # compute average position
        # x, y = mean_position(_df, Lx=Lx)
        x, y = proj.xy2lonlat(_df["x"].mean(), _df["y"].mean())
        # apply myfun
        myfun_out = myfun(*[_df[c] for c in columns], N, dt, **myfun_kwargs)
        # combine with mean position and time
        if myfun_out.index.size == size_out:
            out.loc[t + T / 2.0] = [x, y] + [dr_id] + list(myfun_out)
        t += T * (1 - overlap)
    return out


# should be updated
def mean_position(df, Lx=None):
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
