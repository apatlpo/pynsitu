



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
    dim_x, dim_y, geo = guess_spatial_dims(df)
    if geo:
        df = compute_vector(df, lon_key=dim_x, lat_key=dim_y)
    #
    # drop duplicated values
    df = df.drop_duplicates(subset="date")
    # p = p.where(p.time.diff() != 0).dropna() # duplicates - old
    #
    df = df.sort_values("time")
    #
    if dt is not None:
        if isinstance(dt, float):
            # enforce regular sampling
            tmin, tmax = df.index[0], df.index[-1]
            tmax = tmin + int((tmax - tmin) / dt) * dt
            regular_time = np.arange(tmin, tmax, dt)
            df = df.reindex(regular_time).interpolate()
        elif isinstance(dt, str):
            df = df.set_index("date").resample(dt).pad().reset_index()
            # by default converts to days then
            dt = pd.Timedelta(dt) / pd.Timedelta("1D")
        if geo:
            df = compute_lonlat(
                df,
                lon_key=dim_x,
                lat_key=dim_y,
            )
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
    columns_out = [dim_x, dim_y] + ["id"] + list(myfun_out.index)
    out = pd.DataFrame({c: [] for c in columns_out})
    t = tmin
    while t + T < tmax:
        #
        _df = df.loc[t : t + T]
        if t_is_date:
            # iloc because pandas include the last date
            _df = _df.iloc[:-1, :]
        # compute average position
        x, y = mean_position(_df, Lx=Lx)
        # apply myfun
        myfun_out = myfun(*[_df[c] for c in columns], N, dt, **myfun_kwargs)
        # combine with mean position and time
        if myfun_out.index.size == size_out:
            out.loc[t + T / 2.0] = [x, y] + [dr_id] + list(myfun_out)
        t += T * (1 - overlap)
    return out

def mean_position(df, Lx=None):
    """Compute the mean position of a dataframe

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
