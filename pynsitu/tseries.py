import warnings

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
)
import xarray as xr

from scipy import signal

import matplotlib.pyplot as plt

# from matplotlib.dates import date2num, datetime
# from matplotlib.colors import cnames

try:
    import pytide
except:
    print("Warning: could not import pytide")

try:
    import pyTMD

    # generates tons of warnings, turn off till we actually need pyTMD
    # pass
except:
    print("Warning: could not import pyTMD")

# ------------------------------ parameters / general utils ------------------------------------

deg2rad = np.pi / 180.0
cpd = 86400 / 2 / np.pi


def timedelta2rule(dt, base_unit="1s"):
    """compute a rule based a pd.Timedelta"""
    return str(round(dt / pd.Timedelta(base_unit))) + base_unit[-1]


# ----------------------------- pandas tseries extension -----------------------


class TimeSeries:
    def __init__(self, obj):
        self._time = self._validate(obj)
        self._obj = obj
        self._reset_tseries()
        self._update_time_dtype()

    def _validate(self, obj):
        pass

    def _reset_tseries(self):
        """reset all variables related to accessor"""
        self._time_origin = None
        self._dt = None
        self._delta_time_unit = None
        self._tidal_harmonics = None

    def _update_time_dtype(self):
        self.is_numeric = False
        self.is_datetime = False
        self.is_timedelta = False
        if is_numeric_dtype(self.time.dtype):
            self.dtype = "numeric"
            self.is_numeric = True
        elif is_datetime64_any_dtype(self.time.dtype):
            self.dtype = "datetime"
            self.is_datetime = True
        elif is_timedelta64_dtype(self.time.dtype):
            self.dtype = "timedelta"
            self.is_timedelta = True
        assert any(
            [self.is_numeric, self.is_datetime, self.is_timedelta]
        ), "time dtype not implemented: {self.time.dtype}"

    @property
    def time(self):
        """return time as a series"""
        if self._time_index:
            return self._obj.index.to_series().rename(self._time)
        elif self._time:
            return self._obj[self._time]

    @property
    def dt(self):
        """most likely time increment"""
        if self._dt is None:
            self._dt = np.median(self.get_dt())
        if not self._is_timeline_uniform:
            warnings.warn("dt set but time series is non-uniform")
        return self._dt

    @property
    def _is_timeline_uniform(self):
        dt = np.unique(self.get_dt())
        dt_median = np.median(dt)
        return all((dt - dt_median) / dt_median < 1e-6)

    @property
    def time_origin(self):
        """define a reference time if none is available"""
        if self._time_origin is None:
            # default value
            if self.is_numeric:
                self._time_origin = 0.0
            elif self.is_datetime:
                self._time_origin = pd.Timestamp("2010-01-01")
            elif self.is_timedelta:
                self._time_origin = pd.Timedelta("0")
        return self._time_origin

    @property
    def delta_time_unit(self):
        """define a reference time interval if none is available"""
        if self._delta_time_unit is None:
            # default value
            if self.is_numeric:
                self._delta_time_unit = 1
            else:
                self._delta_time_unit = pd.Timedelta("1s")
        return self._delta_time_unit

    def set_time_origin_delta(self, time_origin=None, delta_time_unit=None):
        """set time reference variables

        Parameters
        ----------
        time_origin: pd.Timestamp
            Time origin used for the computation of physical time intervals
        delta_time_unit: pd.Timedelta
            Time unit interval used for the computation of physical time intervals
        """
        assert (
            time_origin is not None or delta_time_unit is not None
        ), "one of time_origin and delta_time_unit must be provided"
        if time_origin is not None:
            self._time_origin = time_origin
        if delta_time_unit is not None:
            self._delta_time_unit = delta_time_unit

    ## signal processing
    def _spectrum_common(
        self,
        method,
        unit,
        include,
        ignore,
        complex,
        **kwargs,
    ):
        """common treatment of spectral inputs"""
        # time line must be uniform
        assert (
            self._is_timeline_uniform
        ), "spectrum for non-uniform time series is not implemented"
        # time units used for frequency
        if unit is not None:
            assert not self.is_numeric, "unit cannot be specified if time is numeric"
            if isinstance(unit, str):
                unit = pd.Timedelta(unit)
        else:
            unit = self.delta_time_unit
        dt = self.dt / unit
        if isinstance(include, str):
            include = [include]
        if include is None:
            include = self.variables
        if ignore is not None:
            include = [v for v in include if v not in ignore]
        if complex is not None:
            for v in complex:
                assert (
                    v in include
                ), f"variable {v} is not in object and thus cannot be used for the rotary spectral calculation, please adjust complex input variable"
            include = list(complex)
        _kwargs = dict(**kwargs)
        # compute the number of frequency points
        if "nperseg" in kwargs:
            N = kwargs["nperseg"]
            if isinstance(N, str):
                N = round(pd.Timedelta(N) / self.dt)
            if "return_onesided" in kwargs and kwargs["return_onesided"]:
                N = N // 2 + 1  # was int(N/2)+1
            _kwargs["nperseg"] = N
        return method, dt, unit, include, _kwargs


@pd.api.extensions.register_dataframe_accessor("ts")
class TimeSeriesAccessor(TimeSeries):
    """Pandas DataFrame accessor in order to edit and process timeseries-like data"""

    # @staticmethod
    def _validate(self, obj):
        """verify there is a column time"""
        time = None
        time_potential = ["time", "date"]
        # time is a column
        for c in list(obj.columns):
            if c.lower() in time_potential:
                time = c
                assert obj[
                    time
                ].is_monotonic_increasing, (
                    "time should is not monotonically increasing, please sort first"
                )
        # time is the index
        if obj.index.name in time_potential:
            time = obj.index.name
            self._time_index = True
            assert (
                obj.index.is_monotonic_increasing
            ), "time should is not monotonically increasing, please sort first"
        else:
            self._time_index = False
        if not time:
            raise AttributeError(
                "Did not find time column."
                + " You need to rename the relevant column. \n"
                + "Case insentive options are: "
                + "/".join(time_potential)
            )
        else:
            return time

    @property
    def time(self):
        """return time as a series"""
        if self._time_index:
            return self._obj.index.to_series().rename(self._time)
        elif self._time:
            return self._obj[self._time]

    def get_dt(self):
        """get time intervals as an array"""
        # bfill required to fill first item
        return self.time.diff().bfill()

    def set_time_physical(self, inplace=True, overwrite=False):
        """add physical time to object

        Parameters
        ----------
        inplace: boolean, optional
            Add physical time as an additional column, returns the variable otherwise
        overwrite: boolean, optional
            Enable overwriting an existing physical time
        """
        d = self._obj
        time = (self.time - self.time_origin) / self.delta_time_unit
        if inplace:
            if overwrite or "timep" not in d.columns:
                d["timep"] = time
        else:
            return time

    @property
    def variables(self):
        return self._obj.columns

    # time series and/or campaign related material
    def trim(self, d):
        """given a deployment item, trim data temporally

        Parameters
        ----------
        d: pynsitu.events.Deployment
        """
        df = self._obj
        if self.is_datetime:
            if self._time_index:
                time = df.index
            else:
                time = df[self._time]
            df = df.loc[(time >= d.start.time) & (time <= d.end.time)]
            # copying is necessary to avoid warning:
            # SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame.
            df = df.copy()
        return df

    # resampling
    def resample_uniform(self, rule, inplace=False, **kwargs):
        """resample on a uniform time line via interpolation"""
        df = self._obj
        if not self._time_index:
            df = df.set_index(self._time)
        # leverage standard pandas resampling to get new time line
        new_time = df.resample(rule).count().index
        df_interp = (
            df.reindex(new_time.union(df.index)).interpolate(**kwargs).reindex(new_time)
        )
        # does not perform the same operation
        # df = df.resample(rule).interpolate(**kwargs)
        if inplace:
            self._obj = df_interp
        else:
            return df_interp

    def resample_centered(self, freq):
        """centered resampling, i.e. data at t is representative
        of data within [t-dt/2, t+dt/2]

        Parameters
        ----------
        freq: str
            Frequency of the resampling, e.g. "1H"
        Returns
        -------
        df_rs: pandas.core.resample.DatetimeIndexResampler
            This means the reduction step needs to be performed, e.g.
                df_rs.mean() or df_rs.median()
        """
        df = self._obj
        if not self._time_index:
            df = df.set_index(self._time)
        df = df.shift(0.5, freq=freq).resample(freq)
        return df

    ### signal processing

    def spectrum(
        self,
        method="welch",
        unit=None,
        include=None,
        ignore=None,
        complex=None,
        fill_limit=None,
        **kwargs,
    ):
        """compute spectra of timeseries

        Parameters
        ----------
        method: str, optional
            Spectral method, e.g. welch, ...
        unit: str, pd.Timedelta, optional
            time unit to use for frequencies (e.g. "1T", "1D")
        include: str, list, optional
            variables to compute the spectrum on
        ignore: str, list, optional
            list of variables to exclude from the spectral calculation
        complex: tuple, optional
            Specify varibles for the calculation of rotary spectral calculation, e.g. complex= (`v0`, `v1`) computes the spectrum of `v0 + 1j*v1`
        fill_limit: int, optional
            maximum number of points that can be interpolated
        **kwargs: passed to the spectral method
        """
        method, dt, unit, include, kwargs = self._spectrum_common(
            method,
            unit,
            include,
            ignore,
            complex,
            **kwargs,
        )
        df = self._obj
        # massage timeseries (deal with NaNs)
        D = {}
        for c in include:
            D[c] = _pd_interpolate_NaN_or_do_nothing(df[c], self.dt, limit=fill_limit)
        if complex is not None:
            s = D[include[0]] + 1j * D[include[1]]
            c = "_".join(include)
            D = {c: s}
        # actually compute spectra
        E = {}
        for c, s in D.items():
            E[c] = compute_spectrum_pd(s, method, dt, **kwargs)
        return pd.DataFrame(E)

    # tidal analysis
    def tidal_analysis(
        self,
        col,
        constituents=[],
        library="pytide",
        plot=True,
    ):
        """compute a tidal analysis on one column

        Parameters
        ----------
        col: str
            Column to consider
        constituents: list, optional
            List of consistuents
        library: str, optional
            Tidal library to use
        """
        # select and drop nan
        df = self._obj.reset_index()[[self._time, col]].dropna()

        if library == "pytide":
            dfh = pytide_harmonic_analysis(
                df.time,
                df[col],
                constituents=constituents,
            )
        dh = self._tidal_harmonics
        if dh is None:
            dh = {col: dfh}
        else:
            dh[col] = dfh
        self._tidal_harmonics = dh

        # plot amplitudes
        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.stem(dfh.frequency, np.abs(dfh.amplitude))
            for c, r in dfh.iterrows():
                ax.text(r["frequency"] + 0.05, abs(r["amplitude"]), c)
            ax.grid()

    def tidal_predict(
        self,
        col=None,
        time=None,
        amplitudes=None,
        constituents=None,
        library="pytide",
        **kwargs,
    ):
        if self._tidal_harmonics is None or (
            col is not None and col not in self._tidal_harmonics
        ):
            raise AttributeError(
                f"not amplitudes found for {col} or provided, "
                + "you need to run an harmonic analysis first"
            )
        if time is None:
            time = self.time
        if amplitudes is None:
            amplitudes = self._tidal_harmonics[col]["amplitude"]
        if constituents is not None:
            if isinstance(constituents, str):
                constituents = [constituents]
            amplitudes = amplitudes.loc[constituents]

        if library == "pytide":
            s = pytide_predict_tides(time, amplitudes, **kwargs)

        return pd.Series(s, index=time, name=f"{col}_tidal")

    def package_harmonics(self):
        """package harmonics for storage"""
        H = []
        for c, h in self._tidal_harmonics.items():
            H.append(h.to_xarray().expand_dims(vars=pd.Index([c])))
        ds = xr.concat(H, "vars")
        # convert to real/imag for netcdf storage
        ds["amplitude_real"] = np.real(ds["amplitude"])
        ds["amplitude_imag"] = np.imag(ds["amplitude"])
        ds = ds.drop("amplitude")
        return ds

    def load_harmonics(self, file):
        """load harmonics from a file"""
        ds = xr.open_dataset(file)
        dfh = {}
        for c in ds.vars:
            _c = str(c.values)
            _df = ds.sel(vars=_c).to_dataframe()
            _df["amplitude"] = _df["amplitude_real"] + 1j * _df["amplitude_imag"]
            _df = _df.drop(columns=["amplitude_real", "amplitude_imag"])
            dfh[_c] = _df
        self._tidal_harmonics = dfh


def _pd_interpolate_NaN_or_do_nothing(s, dt, **kwargs):
    """try interpolating a pandas.Series or do nothing

    Parameters
    ----------
    s: pd.Series, pd.DataFrame
        input time series
    dt: str, pd.Timedelta
        sampling rate of output timeseries
    **kwargs: passed to pandas interpolate method
    """
    if isinstance(dt, pd.Timedelta):
        dt = timedelta2rule(dt)
    # deal with NaNs if any
    if s.isnull().any():
        try:
            s = s.resample(dt).interpolate(**kwargs)
        except:
            # too many NaNs, let the spectral method return NaNs
            pass
    return s


# ----------------------------- xarray accessor --------------------------------


@xr.register_dataset_accessor("ts")
class XrTimeSeriesAccessor(TimeSeries):
    """Xarray Dataset accessor in order to edit and process timeseries-like data"""

    # @staticmethod
    def _validate(self, obj):
        """verify there are latitude and longitude variables"""
        time = None
        time_potential = ["time", "date"]
        for c in list(obj.variables):
            if c.lower() in time_potential:
                time = c
        if not time:
            raise AttributeError(
                "Did not find time variables. Case insentive options are: "
                + "/".join(time_potential)
            )
        else:
            assert len(obj[time].dims) == 1, "time variable should be one dimensional"
            assert (
                time == obj[time].dims[0]
            ), "time/date variable and dimenion are not labelled identically, this is not supported at the moment"
            return time

    @property
    def time(self):
        """return time (may have a different name)"""
        return self._obj[self._time]

    def get_dt(self):
        """get time intervals as an array"""
        return self.time.diff(self._time)

    def set_time_physical(self, overwrite=True):
        """add physical time to object"""
        ds = self._obj
        if "timep" not in ds.variables or overwrite:
            ds["timep"] = (ds[self._time] - self.time_origin) / self.delta_time_unit

    @property
    def variables(self):
        """list only time variables"""
        ds = self._obj
        return [v for v in list(ds) if self._time in ds[v].dims]

    # time series and/or campaign related material
    def trim(self, d, inplace=False):
        """given a deployment item, trim data temporally

        Parameters
        ----------
        d: pynsitu.events.Deployment
        """
        ds = self._obj
        if self.is_datetime:
            ds = ds.sel({self._time: slice(d.start.time, d.end.time)})
        if inplace:
            self._obj = ds
        else:
            return ds

    ## signal processing

    def spectrum(
        self,
        method="welch",
        unit=None,
        include=None,
        ignore=None,
        complex=None,
        fill_limit=None,
        **kwargs,
    ):
        """compute spectra of timeseries

        Parameters
        ----------
        method: str, optional
            Spectral method, e.g. welch, ...
        unit: str, pd.Timedelta, optional
            time unit to use for frequencies (e.g. "1T", "1D")
        include: str, list, optional
            variables to compute the spectrum on
        ignore: str, list, optional
            list of variables to exclude from the spectral calculation
        complex: tuple, optional
            Specify varibles for the calculation of rotary spectral calculation, e.g. complex= (`v0`, `v1`) computes the spectrum of `v0 + 1j*v1`
        fill_limit: int, optional
            maximum number of points that can be interpolated
        **kwargs: passed to the spectral method
        """
        method, dt, unit, include, kwargs = self._spectrum_common(
            method,
            unit,
            include,
            ignore,
            complex,
            **kwargs,
        )
        ds = self._obj
        # massage timeseries (deal with NaNs)
        D = {}
        for v in include:
            D[v] = _xr_interpolate_NaN_or_do_nothing(
                ds[v], self.dt, self._time, limit=fill_limit
            )
        if complex is not None:
            da = D[include[0]] + 1j * D[include[1]]
            c = "_".join(include)
            D = {c: da}
        # actually compute spectra
        E = []
        for v, da in D.items():
            E.append(
                compute_spectrum_xr(da, method, dt, self._time, **kwargs).rename(v)
            )
        return xr.merge(E)


def _xr_interpolate_NaN_or_do_nothing(da, dt, time, **kwargs):
    """try interpolating a pandas.Series or do nothing

    Parameters
    ----------
    s: pd.Series, pd.DataFrame
        input time series
    dt: str, pd.Timedelta
        sampling rate of output timeseries
    **kwargs: passed to pandas interpolate method
    """
    if isinstance(dt, pd.Timedelta):
        dt = timedelta2rule(dt)
    # deal with NaNs if any
    if da.isnull().any():
        try:
            da = da.resample(dt).interpolate_na(dim=time, **kwargs)
        except:
            # too many NaNs, let the spectral method return NaNs
            pass
    return da


# -------------------------- filtering ----------------------------------


def generate_filter(
    band,
    T=10,
    dt=1 / 24,
    lat=None,
    bandwidth=None,
    normalized_bandwidth=None,
):
    """Wrapper around scipy.signal.firwing

    Parameters
    ----------
    band: str, float
        Frequency band (e.g. "semidiurnal", ...) or filter central frequency in cpd
    T: float
        Filter length in days
    dt: float
        Filter/time series time step
    lat: float
        Latitude (for inertial band)
    bandwidth: float
        Filter bandwidth in cpd
    dt: float
        days
    """
    numtaps = int(T / dt)
    pass_zero = False
    #
    if band == "low":
        pass_zero = True
        cutoff = [bandwidth]
    elif band == "subdiurnal":
        pass_zero = True
        cutoff = [1.0 / 2.0]
    elif band == "semidiurnal":
        omega = 1.9322  #  M2 24/12.4206012 = 1.9322
    elif band == "diurnal":
        omega = 1.0  # K1 24/23.93447213 = 1.0027
    elif band == "inertial":
        assert lat is not None, "latitude needs to be provided to generate_filter"
        from .geo import coriolis

        omega = coriolis(lat) / 2.0 / np.pi
    elif isinstance(band, float):
        omega = band
    #
    if bandwidth is not None and band != "low":
        cutoff = [omega - bandwidth, omega + bandwidth]
    elif normalized_bandwidth is not None:
        cutoff = [
            omega * (1 - normalized_bandwidth),
            omega * (1.0 + normalized_bandwidth),
        ]
    #
    h = signal.firwin(
        numtaps, cutoff=cutoff, pass_zero=pass_zero, fs=1 / dt, scale=True
    )
    return h


def filter_response(h, dt=1 / 24):
    """Returns the frequency response

    Parameters
    ----------
    h: np.array
        filter kernel/weights
    dt: float, optional

    Returns
    -------
    H: np.array
        frequency response function
    w: np.array
        frequencies
    """
    w, hh = signal.freqz(h, worN=8000, fs=1 / dt)
    return hh, w


# -------------------------- spectral analysis ----------------------------------

# common kwargs for scipy welch, periodogram
scipy_spectrum_kwargs = dict(
    window="hann",
    return_onesided=False,
    detrend=False,
    scaling="density",
)


def compute_spectrum_pd(
    v,
    method,
    dt,
    **kwargs,
):
    """Compute the spectrum of a pandas time series
    Treatment of NaNs is assumed to be carried out beforehand

    Parameters
    ----------
        da: ndarray, xr.DataArray
            Time series, the index must be time (and named as it) if dt is not provided
        method: string
            Method that will be employed for spectral calculations.
            Implemented methods are 'welch', 'periodogram' (not tested)
        dt: float
            Time spacing
        **kwargs: passed to the spectral calculation method
    See:
        - https://docs.scipy.org/doc/scipy/reference/signal.html#spectral-analysis
        - https://krischer.github.io/mtspec/
        - http://nipy.org/nitime/examples/multi_taper_spectral_estimation.html
    """
    assert v is not None or "nperseg" in kwargs, "nperseg needs to be specified"
    if v is None:
        # dask distribution related constraints
        v = pd.Series(np.random.randn(kwargs["nperseg"]))

    # enables feeding np.arrays
    if not isinstance(v, pd.Series):
        v = pd.Series(v)
        v.index.rename("time", inplace=True)

    assert is_numeric_dtype(type(dt)), f"dt must be of numeric type, found {type(dt)}"

    if method in ["welch", "periodogram"]:
        mkwargs = dict(**scipy_spectrum_kwargs)
        mkwargs.update(fs=1 / dt, axis=0, method=getattr(signal, method))
        mkwargs.update(**kwargs)
        if method == "welch":
            assert "nperseg" in kwargs, "nperseg is required for method welch"
            if "alpha" not in mkwargs:
                # required because alpha cannot be passed to periodogram
                mkwargs["alpha"] = 0.5

        f, E = _scipy_spectra_wrapper(v, **mkwargs)
    # elif method == "mtspec":
    #    from mtspec import mtspec
    #    lE, f = mtspec(
    #        data=v, delta=dt, time_bandwidth=4.0, number_of_tapers=6, quadratic=True
    #    )
    # elif method == "mt":
    #    import nitime.algorithms as tsa
    #    dkwargs = {"NW": 2, "sides": "twosided", "adaptive": False, "jackknife": False}
    #    dkwargs.update(kwargs)
    #    lf, E, nu = tsa.multi_taper_psd(v, Fs=1 / dt, **dkwargs)
    #    f = fftfreq(len(lf)) * 24.0
    #    # print('Number of tapers = %d' %(nu[0]/2))

    # place back in pd.Series along with frequency
    E = pd.Series(E, index=pd.Index(f, name="frequency")).sort_index()

    return E


def compute_spectrum_xr(
    da,
    method,
    dt,
    time,
    rechunk=False,
    **kwargs,
):
    """Compute the spectrum of a pandas time series
    Treatment of NaNs is assumed to be carried out beforehand

    Parameters
    ----------
        da: ndarray, xr.DataArray
            Time series, the index must be time (and named as it) if dt is not provided
        method: string
            Method that will be employed for spectral calculations.
            Implemented methods are 'welch', 'periodogram' (not tested)
        dt: float
            Time spacing
        time: str
            Name of time dimension in xarray object
        rechunk: boolean
            Automatically rechunk along time dimension
        **kwargs: passed to the spectral calculation method
    See:
        - https://docs.scipy.org/doc/scipy/reference/signal.html#spectral-analysis
        - https://krischer.github.io/mtspec/
        - http://nipy.org/nitime/examples/multi_taper_spectral_estimation.html
    """

    # enables feeding np.arrays
    if not isinstance(da, xr.DataArray):
        da = xr.DataArray(da).rename(dim_0="time")
        warnings.warn("spectral calculation on np.array: make sure axis 0 is time")

    assert is_numeric_dtype(type(dt)), f"dt must be of numeric type, found {type(dt)}"

    # init apply_ufunc kwargs
    aukwargs = dict(
        output_dtypes=[da.dtype],
        input_core_dims=[
            [time],
        ],
        # input_core_dims=[[time], [time]], # for two input variables
        output_core_dims=[["frequency"]],
    )
    # dask backend case
    dask_backend = da.chunks is not None
    if dask_backend:
        # check number of time chunks
        time_chunks_num = len(da.chunks[da.get_axis_num(time)])
        assert (
            time_chunks_num == 1 or rechunk
        ), "number of chunks along the time dimension is not one, rechunk or set rechunk to True"
        if time_chunks_num > 1 and rechunk:
            da = da.chunk({time: -1})
        aukwargs.update(
            dask="parallelized",
        )

    # spectral calculation kwargs
    mkwargs = dict(axis=-1, fs=1 / dt)
    if method in ["welch", "periodogram"]:
        mkwargs.update(**scipy_spectrum_kwargs)
        if method == "welch":
            assert "nperseg" in kwargs, "nperseg is required for method welch"
            if "alpha" not in mkwargs:
                # necessary because alpha cannot be passed to periodogram
                mkwargs["alpha"] = 0.5
        mkwargs["method"] = getattr(signal, method)
        func = _scipy_spectra_wrapper
    # elif method == "mtspec":
    #    from mtspec import mtspec
    #    lE, f = mtspec(
    #        data=v, delta=dt, time_bandwidth=4.0, number_of_tapers=6, quadratic=True
    #    )
    # elif method == "mt":
    #    import nitime.algorithms as tsa
    #    dkwargs = {"NW": 2, "sides": "twosided", "adaptive": False, "jackknife": False}
    #    dkwargs.update(kwargs)
    #    lf, E, nu = tsa.multi_taper_psd(v, Fs=1 / dt, **dkwargs)
    #    f = fftfreq(len(lf)) * 24.0
    #    # print('Number of tapers = %d' %(nu[0]/2))
    # update kwargs
    mkwargs.update(**kwargs)
    # run once to get frequency
    _da = da.isel(**{d: 0 for d in da.dims if d != time}).values
    f, _ = func(
        _da,
        **mkwargs,
    )
    # update output size if dask backend
    if dask_backend:
        aukwargs.update(dask_gufunc_kwargs={"output_sizes": {"frequency": f.size}})

    # core calculation
    mkwargs["ufunc"] = True
    E = xr.apply_ufunc(
        func,
        da,
        kwargs=mkwargs,
        **aukwargs,
    )
    E = E.assign_coords(frequency=f).sortby("frequency")

    # if real:
    #    E = np.real(E).astype(np.float64)

    return E


def _scipy_spectra_wrapper(
    x, method=None, fs=None, axis=-1, ufunc=False, alpha=None, **kwargs
):
    """wrapper around scipy spectral methods
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html#scipy.signal.periodogram
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html#scipy.signal.csd
    """
    assert method is not None, "method must be provided"
    assert fs is not None, "fs must be provided"
    #
    dkwargs = dict(**kwargs)
    if alpha is not None:
        dkwargs["noverlap"] = round(alpha * kwargs["nperseg"])
    f, E = method(x, fs=fs, axis=axis, **dkwargs)
    #
    if ufunc:
        return E
    else:
        return f, E


# -------------------------- tidal analysis ----------------------------------

tidal_constituents = [
    "2n2",
    "eps2",
    "j1",
    "k1",
    "k2",
    "l2",
    "lambda2",
    "m2",
    "m3",
    "m4",
    "m6",
    "m8",
    "mf",
    "mks2",
    "mm",
    "mn4",
    "ms4",
    "msf",
    "msqm",
    "mtm",
    "mu2",
    "n2",
    "n4",
    "nu2",
    "o1",
    "p1",
    "q1",
    "r2",
    "s1",
    "s2",
    "s4",
    "sa",
    "ssa",
    "t2",
]


def pytide_harmonic_analysis(time, eta, constituents=[]):
    """Distributed harmonic analysis

    Parameters
    ----------
    time:

    constituents: list
        tidal consituent e.g.:
            ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "S1", "M4"]
    """
    wt = pytide.WaveTable(
        constituents
    )  # not working on months like time series, need to restrict
    if isinstance(time, pd.Series):
        time = time.values
    if isinstance(eta, pd.Series):
        eta = eta.values
    # demean:
    eta = eta - eta.mean()
    # enforce correct type
    time = time.astype("datetime64")
    # compute nodal modulations
    f, vu = wt.compute_nodal_modulations(time)
    # compute harmonic analysis
    a = wt.harmonic_analysis(eta, f, vu)
    return pd.DataFrame(
        dict(
            amplitude=a,
            constituent=wt.constituents(),
            frequency=wt.freq() * 86400 / 2 / np.pi,
            frequency_rad=wt.freq(),
        )
    ).set_index("constituent")


def pytide_predict_tides(
    time,
    har,
    cplx=False,
):
    """Predict tides based on pytide outputs

    v = Re ( conj(amplitude) * dsp.f * np.exp(1j*vu) )

    see: https://pangeo-pytide.readthedocs.io/en/latest/pytide.html#pytide.WaveTable.harmonic_analysis

    Parameters
    ----------
    time: xr.DataArray
        Target time
    har: xr.DataArray, xr.Dataset, optional
        Complex amplitudes. Load constituents from a reference station otherwise
    """

    if isinstance(time, pd.Series):
        time = time.values

    # build wave table
    wt = pytide.WaveTable(list(har.index))

    # compute nodal modulations
    time = time.astype("datetime64")
    _time = [(pd.Timestamp(t) - pd.Timestamp(1970, 1, 1)).total_seconds() for t in time]
    f, vu = wt.compute_nodal_modulations(time)
    v = (f * np.exp(1j * vu) * np.conj(har.values[:, None])).sum(axis=0)
    if cplx:
        return v
    return np.real(v)


def load_equilibrium_constituents(c=None):
    """Load equilibrium tide amplitudes

    Parameters
    ----------
    c: str, list
        constituent or list of constituent

    Returns
    -------
    amplitude: amplitude of equilibrium tide in m for tidal constituent
    phase: phase of tidal constituent
    omega: angular frequency of constituent in radians
    alpha: load love number of tidal constituent
    species: spherical harmonic dependence of quadrupole potential
    """
    if c is None:
        c = tidal_constituents
    if isinstance(c, list):
        df = pd.DataFrame({_c: load_equilibrium_constituents(_c) for _c in c}).T
        df = df.sort_values("omega")
        return df
    elif isinstance(c, str):
        p_names = ["amplitude", "phase", "omega", "alpha", "species"]
        p = pyTMD.load_constituent(c)
        return pd.Series({_n: _p for _n, _p in zip(p_names, p)})
