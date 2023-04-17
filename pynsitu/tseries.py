import warnings

import numpy as np
import pandas as pd
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

# ------------------------------ parameters ------------------------------------

deg2rad = np.pi / 180.0
cpd = 86400 / 2 / np.pi

# ----------------------------- pandas tseries extension -----------------------


@pd.api.extensions.register_dataframe_accessor("ts")
class TimeSeriesAccessor:
    def __init__(self, pandas_obj):
        self._time = self._validate(pandas_obj)
        self._obj = pandas_obj
        self._reset_tseries()

    # @staticmethod
    def _validate(self, obj):
        """verify there is a column time"""
        time = None
        time_potential = ["time", "date"]
        # time is a column
        for c in list(obj.columns):
            if c.lower() in time_potential:
                time = c
        # time is the index
        if obj.index.name in time_potential:
            time = obj.index.name
            self._time_index = True
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

    def _reset_tseries(self):
        """reset all variables related to accessor"""
        self._time_ref = None
        self._delta_time_ref = None
        self._tidal_harmonics = None

    @property
    def time(self):
        """return time as a series"""
        if self._time_index:
            return self._obj.index.to_series().rename(self._time)
        elif self._time:
            return self._obj[self._time]

    @property
    def time_reference(self):
        """define a reference time if none is available"""
        if self._time_ref is None:
            # default value
            self._time_ref = pd.Timestamp("2010-01-01")
        return self._time_ref

    @property
    def delta_time_reference(self):
        """define a reference time if none is available"""
        if self._delta_time_ref is None:
            # default value
            self._delta_time_ref = pd.Timedelta("1s")
        return self._delta_time_ref

    def set_time_reference(self, time_ref=None, dt=None, reset=True):
        """set time references"""
        if reset:
            self._reset_tseries()
        self._time_ref = time_ref
        self._delta_ref = time_ref

    def time_physical(self, overwrite=True):
        """add physical time to object"""
        d = self._obj
        if "timep" not in d.columns or overwrite:
            d["timep"] = (self.time - self.time_reference) / self.delta_time_reference

    def _check_uniform_timeline(self):
        dt = self.time.diff() / pd.Timedelta(self.delta_time_reference)
        # could use .unique instead
        dt_min = dt.min()
        dt_max = dt.max()
        # print(f" min(dt)= {dt_min} max(dt)= {dt_max} ")
        return dt_min == dt_max

    # time series and/or campaign related material
    def trim(self, d):
        """given a deployment item, trim data"""
        if self._time_index:
            time = self._obj.index
        else:
            time = self._obj[self._time]
        df = self._obj.loc[(time >= d.start.time) & (time <= d.end.time)]
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
            return df

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

    # signal processing
    def spectrum(self, method="welch"):
        pass

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


# ----------------------------- xarray accessor --------------------------------


@xr.register_dataset_accessor("ts")
class XrTimeSeriesAccessor:
    def __init__(self, xarray_obj):
        self._time = self._validate(xarray_obj)
        self._obj = xarray_obj
        self._reset_tseries()

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

    def _reset_tseries(self):
        """reset all variables related to accessor"""
        self._time_ref = None
        self._delta_time_ref = None
        self._tidal_harmonics = None

    @property
    def time(self):
        """return time (may have a different name)"""
        return self._obj[self._time]

    @property
    def time_reference(self):
        """define a reference time if none is available"""
        if self._time_ref is None:
            # default value
            self._time_ref = pd.Timestamp("2010-01-01")
        return self._time_ref

    @property
    def delta_time_reference(self):
        """define a reference time if none is available"""
        if self._delta_time_ref is None:
            # default value
            self._delta_time_ref = pd.Timedelta("1s")
        return self._delta_time_ref

    def set_time_reference(self, time_ref=None, dt=None, reset=True):
        """set time references"""
        if reset:
            self._reset_tseries()
        self._time_ref = time_ref
        self._delta_ref = time_ref

    def time_physical(self, overwrite=True):
        """add physical time to object"""
        ds = self._obj
        if "timep" not in ds.variables or overwrite:
            ds["timep"] = (
                ds[self._time] - self.time_reference
            ) / self.delta_time_reference

    # time series and/or campaign related material
    def trim(self, d, inplace=False):
        """given a deployment item, trim data"""
        ds = self._obj.sel({self._time: slice(d.start.time, d.end.time)})
        if inplace:
            self._obj = ds
        else:
            return ds

    # resample on regular time line
    def resample(self, freq, inplace=False):
        ds, t = self._obj, self._time
        new_time = pd.date_range(ds[t].values[0], ds[t].values[-1], freq=freq)
        dsi = ds.interp(**{t: new_time})
        if inplace:
            self._obj = dsi
        else:
            return dsi

    # signal processing ...


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
    """Returns the frequency response"""
    w, hh = signal.freqz(h, worN=8000, fs=1 / dt)
    return hh, w


# -------------------------- spectral analysis ----------------------------------


def get_spectrum(
    v,
    N,
    dt=None,
    N_fill_limit=24,
    method="periodogram",
    detrend=False,
    nan_ratio=0.1,
    **kwargs,
):
    """Compute a lagged correlation between two time series
    These time series are assumed to be regularly sampled in time
    and along the same time line.
    Parameters
    ----------
        v: ndarray, pd.Series
            Time series, the index must be time if dt is not provided
        N: int
            Length of the output
        dt: float, optional
            Time step
        N_fill_limit : int
            Limit length of Nan gap filled by interpolatation
        method: string
            Method that will be employed for spectral calculations.
            Default is 'periodogram'
        detrend: str or function or False, optional
            Turns detrending on or off. Default is False.
        nan_ratio : float between 0 and 1
            limit of nan value ratio tolerated in time series
            default is 0.1
    CAUTION : if v ndarray need to give dt
    See:
        - https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.periodogram.html
        - https://krischer.github.io/mtspec/
        - http://nipy.org/nitime/examples/multi_taper_spectral_estimation.html
    """
    if v is None:
        _v = pd.Series(np.random.randn(N))

    elif not isinstance(v, pd.Series):
        v = pd.Series(
            v
        )  # harmonising type (solving pb np.ndarray -> isnan(), pd.Series ->isna(), iloc pb)

    # returned spectrum is nan array if to many nan holes
    elif (
        v.iloc[:N].isna().sum() / N >= nan_ratio
    ):  # Avoids computing spectra where a large part of value come from interpolation
        warnings.warn(
            f"Nan values ratio {round(100 * v.iloc[:N].isna().sum()/N)}% exceeds the {100*nan_ratio}% tolerated limit, replace spectrum by nan array",
            UserWarning,
        )
        _v = pd.Series(np.full(N, np.nan), v.index[:N])

    # fill little nan hole by interpolation
    elif N_fill_limit is not None:
        mask = v.isna()
        x = (
            mask.groupby((mask != mask.shift()).cumsum()).transform(
                lambda x: len(x) > N_fill_limit
            )
            * mask
        )  # True in holes bigger than N_fill_limit
        if x.any():
            warnings.warn(
                f"Hole(s) bigger than the tolerated length for interpolation {N_fill_limit}, replace spectrum by nan array",
                UserWarning,
            )
            _v = pd.Series(np.full(N, np.nan), v.index[:N])
        else:
            _v = v.interpolate(
                method="quadratic"
            )  # interpolate on hole smaller than N_fill_limit
            _v = _v.iloc[:N]

    else:
        _v = v.iloc[:N]

    if dt is None:
        dt = _v.reset_index()["index"].diff().mean()

    if detrend and not method == "periodogram":
        print("!!! Not implemented yet except for periodogram")
    if method == "periodogram":
        from scipy import signal

        dkwargs = {
            "window": "hann",
            "return_onesided": False,
            "detrend": detrend,
            "scaling": "density",
        }
        dkwargs.update(kwargs)
        f, E = signal.periodogram(_v, fs=1 / dt, axis=0, **dkwargs)
    elif method == "mtspec":
        from mtspec import mtspec

        lE, f = mtspec(
            data=_v, delta=dt, time_bandwidth=4.0, number_of_tapers=6, quadratic=True
        )
    elif method == "mt":
        import nitime.algorithms as tsa

        dkwargs = {"NW": 2, "sides": "twosided", "adaptive": False, "jackknife": False}
        dkwargs.update(kwargs)
        lf, E, nu = tsa.multi_taper_psd(_v, Fs=1 / dt, **dkwargs)
        f = fftfreq(len(lf)) * 24.0
        # print('Number of tapers = %d' %(nu[0]/2))
    return pd.Series(E, index=f)


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
    v = (f * np.exp(1j * vu) * np.conj(har[:, None])).sum(axis=0)
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



