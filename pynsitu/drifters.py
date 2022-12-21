import numpy as np
import pandas as pd

from numpy.linalg import inv
from scipy.linalg import solve
from scipy.sparse import diags

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
            _dir = "east"
            if t > df.index[0] and t < df.index[-1]:
                am = df.loc[:t, "acceleration_" + _dir].iloc[-2]
                a = spikes.loc[t, "acceleration_" + _dir]
                ap = df.loc[t:, "acceleration_" + _dir].iloc[1]
            # check if am and ap have opposite sign to a
            C.append(am * a < 0 and ap * a < 0)
        if all(C):
            validated_single_spikes.append(t)
    if verbose:
        print(
            f"{len(validated_single_spikes)} single spikes found out {spikes.index.size}"
            + f" potential ones (acceleration threshold)"
        )
    # drops single spikes
    df = df.drop(validated_single_spikes)
    return df


def smooth_resample(
    df,
    t_target,
    position_error,
    acceleration_amplitude,
    acceleration_T,
    velocity_acceleration=True,
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
    """

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
    # x0 = interp1d(t, df["x"], kind="cubic", fill_value="extrapolate")(t_target)
    dfi["x"] = solve(L, I.T.dot(df["x"].values))

    # y
    # y0 = interp1d(t, df["y"], kind="cubic", fill_value="extrapolate")(t_target)
    dfi["y"] = solve(L, I.T.dot(df["y"].values))

    # update lon/lat
    # first reset reference from df
    dfi.geo.set_projection_reference(df.geo.projection_reference)  # inplace
    dfi.geo.compute_lonlat()  # inplace

    # recompute velocity, should be an option?
    if velocity_acceleration:
        dfi = dfi.geo.compute_velocities()
        dfi = dfi.geo.compute_accelerations()

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
    w = (t - t_target[i_t - 1]) / dt
    I[:, i_t - 1] = np.diagflat(w, k=0)
    I[:, i_t] = np.diagflat(1 - w, k=0)

    # second order derivative
    one_second = pd.Timedelta("1S")
    dt2 = (dt / one_second) ** 2
    D2 = diags([1 / dt2, -2 / dt2, 1 / dt2], [-1, 0, 1], shape=(Nt, Nt)).toarray()
    # fix edges
    # D2[0, [0, 1]] = [-1/dt2, 1/dt2] # not good: pull velocity towards 0 at edges
    # D2[-1, [-2, -1]] = [-1/dt2, 1/dt2]  # not good: pull velocity towards 0 at edges
    D2[0, :] = 0
    D2[-1, :] = 0
    # acceleration autocorrelation
    _t = t_target.values
    R = acceleration_R((_t[:, None] - _t[None, :]) / one_second)
    iR = inv(R)
    # assemble final operator
    L = I.T.dot(I) + D2.T.dot(iR.dot(D2)) * position_error**2

    return L, I


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
