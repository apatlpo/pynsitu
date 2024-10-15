import os

import numpy as np
import xarray as xr
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime

import matplotlib.pyplot as plt
from matplotlib.dates import date2num, datetime
from matplotlib.colors import cnames

try:
    from bokeh.io import output_notebook, show
    from bokeh.layouts import gridplot
    from bokeh.models import HoverTool
    from bokeh.models import CrosshairTool
    from bokeh.plotting import figure
except:
    print("Warning: could not import bokeh")

try:
    import gsw
except:
    print("Warning: could not import gsw")

# ------------------------------ parameters ------------------------------------


# ----------------------------- seawater accessor - common --------------------------
_t_potential = [
    "temperature",
    "temp",
    "t",
]
_s_potential = [
    "salinity",
    "psal",
    "s",
]
_c_potential = [
    "conductivity",
    "cond",
    "c",
]
_p_potential = [
    "pressure",
    "p",
]
_d_potential = [
    "depth",
]


class SeawaterAccessor:
    def __init__(self, _obj):
        """Seawater accessor
        The data requires the following columns (pandas.dataframe) or variables (xarray.Dataset):
            - in situ temperature, accepted names: ["temperature", "temp", "t"]
            - practical salinity or conductivity, accepted names are:
                ["salinity", "psal", "s"]
                ["conductivity", "cond", "c"]
            - pressure or depth, accepted names:
                ["pressure", "p"]
                ["depth"]
        Accepted units ares:
            - temperature: degC
            - practical salinity: PSU
            - conductivity: mS/cm
            - pressure: dbar
            - depth: m
        Longitude and Latitude are treated differently and may be columns or
        attributes or in the `attrs` dictionnary
        """
        self._t, self._s, self._c, self._p = self._validate(_obj)
        self._obj = _obj
        self._update_SA_PT()

    def init(self):
        """simply instantiate accessor"""
        return


# ----------------------------- seawater accessor - pandas --------------------------


@pd.api.extensions.register_dataframe_accessor("sw")
class PdSeawaterAccessor(SeawaterAccessor):
    """Pandas DataFrame accessor in order to carry and process seawater properties

    The DataFrame requires the following columns:
        - in situ temperature, accepted names: ["temperature", "temp", "t"]
        - practical salinity or conductivity, accepted names are:
            ["salinity", "psal", "s"]
            ["conductivity", "cond", "c"]
        - pressure or depth, accepted names:
            ["pressure", "p"]
            ["depth"]
    Accepted units ares:
        - temperature: degC
        - practical salinity: PSU
        - conductivity: mS/cm
        - pressure: dbar
        - depth: m
    Longitude and Latitude are treated differently and may be columns or
    attributes or in the `attrs` dictionnary
    """

    # @staticmethod
    def _validate(self, obj):
        """verify all necessary information is here"""
        # search for lon/lat in columns, as standard attribute, in attrs dict
        lon_names = ["longitude", "long", "lon"]
        lat_names = ["latitude", "lat"]
        for k in obj.columns:
            if k.lower() in lon_names:
                self._lon = k
                fill_lon = False
            if k.lower() in lat_names:
                self._lat = k
                fill_lat = False
        for k in obj.attrs:
            if k.lower() in lon_names:
                _lon = obj.attrs[k]
                self._lon = k
                fill_lon = True
            if k.lower() in lon_names:
                _lat = obj.attrs[k]
                self._lat = k
                fill_lat = True
        for k in obj.__dict__:
            if k.lower() in lon_names:
                _lon = getattr(obj, k)
                self._lon = k
                fill_lon = True
            if k.lower() in lat_names:
                _lat = getattr(obj, k)
                self._lat = k
                fill_lat = True
        if fill_lon:
            obj.loc[:, self._lon] = _lon
        if fill_lat:
            obj.loc[:, self._lat] = _lat
        if not hasattr(self, "_lon"):
            raise AttributeError("Did not find an attribute longitude")
        if not hasattr(self, "_lat"):
            raise AttributeError("Did not find an attribute latitude")
        # check all values of lon/lat are not NaN
        if ~all(~pd.isna(obj[self._lon])) or ~all(~pd.isna(obj[self._lat])):
            print(
                "some values of longitude and latitudes are NaN, you may want to fill in with correct values"
            )

        # deal now with actual seawater properties
        t, s, c, p, d = None, None, None, None, None
        for col in list(obj.columns):
            if col.lower() in _t_potential:
                t = col
            elif col.lower() in _s_potential:
                s = col
            elif col.lower() in _c_potential:
                c = col
            elif col.lower() in _p_potential:
                p = col
            elif col.lower() in _d_potential:
                d = col
        if not t or (not s and not c) or (not p and not d):
            raise AttributeError(
                "Did not find temperature, salinity and pressure columns. \n"
                + "Case insentive options are: "
                + "/".join(_t_potential)
                + " , "
                + "/".join(_s_potential)
                + " , \n"
                + "/".join(_c_potential)
                + " , "
                + "/".join(_p_potential)
            )
        else:
            # compute pressure from depth and depth from pressure if need be
            if not p:
                p = "pressure"
                obj.loc[:, p] = gsw.p_from_z(
                    -obj[d],
                    obj[self._lat].median(),
                )
            if not d:
                obj.loc[:, "depth"] = -gsw.z_from_p(
                    obj[p],
                    obj[self._lat].median(),
                )
            return t, s, c, p

    def reset(self, extra=[]):  # , inplace=True):
        """delete core seawater variables for update"""
        # if inplace:
        df = self._obj.copy()
        # else:
        #    df = self._obj.copy()
        df.drop(
            columns=[
                "SA",
                "CT",
                "sigma0",
            ]
            + extra,
            errors="ignore",
            inplace=True,
        )
        # recompute SA, PT & co
        # self._update_SA_PT()
        df.sw.init()
        return df

    def _update_SA_PT(self):
        """update SA and property, do not overwrite existing values"""
        df = self._obj
        t, p = df[self._t], df[self._p]
        lon, lat = df[self._lon], df[self._lat]
        # compute practical salinity from conductivity if need be
        if self._s is None:
            self._s = "salinity"
            df.loc[:, self._s] = gsw.SP_from_C(df[self._c], t, p)
        s = df[self._s]
        #
        if "SA" not in df.columns:
            df.loc[:, "SA"] = gsw.SA_from_SP(s, p, lon, lat)
        if "CT" not in df.columns:
            df.loc[:, "CT"] = gsw.CT_from_t(df["SA"], t, p)
        if "sigma0" not in df.columns:
            df.loc[:, "sigma0"] = gsw.sigma0(df["SA"], df["CT"])

    def set_columns(self, **kwargs):
        """set accessor column names: t, s, c, p, d, lon, lat
        and update internal eos variables (SA, PT)
        """
        for k, v in kwargs.items():
            setattr(self, "_" + k, v)
        self._update_SA_PT()

    def update_eos(self, inplace=True):
        """update eos related variables (e.g. in situ temperature, practical
        salinity, sigma0) based on SA (absolute salinity) and CT (conservative temperature).

        Parameters
        ----------
        inplace: boolean, optional
            convenience argument to trigger a copy of the dataframe instead of
            an inplace modification (False by default).
        """
        if inplace:
            df = self._obj
        else:
            df = self._obj.copy()
        sa, ct, p = df.loc[:, "SA"], df.loc[:, "CT"], df.loc[:, self._p]
        lon, lat = df[self._lon], df[self._lat]
        df.loc[:, self._t] = gsw.t_from_CT(sa, ct, p)
        df.loc[:, self._s] = gsw.SP_from_SA(sa, p, lon, lat)
        df.loc[:, "sigma0"] = gsw.sigma0(sa, ct)
        if not inplace:
            return df

    def apply_with_eos_update(self, fun, *args, **kwargs):
        """Apply a function and update eos related variables
        This is an helper method
        """
        # apply function
        df = fun(self._obj, *args, **kwargs)
        # update eos related variables
        df = df.sw.update_eos(inplace=False)
        return df

    def resample(
        self,
        rule,
        op="mean",
        interpolate=False,
        **kwargs,
    ):
        """Temporal resampling
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html
        This is NOT done inplace

        Parameters
        ----------
        rule: DateOffset, Timedelta or str
            Passed to pandas.DataFrame.resample, examples:
                - '10T': 10 minutes
                - '10S': 10 seconds
        op: str, optional
            operation to perform while resampling ("mean" by default)
        interpolate: boolean, optional
            activates interpolation for upsampling (False by default)
        **kwargs:
            passed to resample
        """
        assert is_datetime(self._obj.index.dtype), (
            "The index should be a datetime object"
            + ". You may need to perform a set_index"
        )
        return self.apply_with_eos_update(_resample, rule, op, interpolate, **kwargs)

    def compute_vertical_profile(
        self,
        step,
        speed_threshold=None,
        op="mean",
        depth_min=0,
        depth_max=None,
    ):
        return self.apply_with_eos_update(
            _get_profile,
            depth_min,
            depth_max,
            step,
            speed_threshold,
            op,
        )

    def plot_bokeh(
        self,
        deployments=None,
        rule=None,
        width=400,
        cross=True,
    ):
        """Bokeh plot, useful to clean data

        Parameters
        ----------
        deployments: dict-like, pynsitu.events.Deployments for instance, optional
            Deployments
        rule: str, optional
            resampling rule
        width: int, optional
            Plot width in pixels
        cross: boolean, optional
            ...
        """

        if rule is not None:
            df = self.resample(rule)
        else:
            df = self._obj

        output_notebook()
        TOOLS = "pan,wheel_zoom,box_zoom,reset,help"
        crosshair = CrosshairTool(dimensions="both")

        # line specs
        lw, c = 3, "black"

        from .events import Deployment

        if deployments is not None:
            from .events import Deployment

            if isinstance(deployments, Deployment):
                deployments = deployments.to_deployments()

        def _add_start_end(s, y):
            # _y = y.iloc[y.index.get_loc(_d.start.time), method='nearest')]
            if deployments is not None:
                for label, d in deployments.items():
                    s.line(
                        x=[d.start.time, d.start.time],
                        y=[y.min(), y.max()],
                        color="cadetblue",
                        line_width=2,
                    )
                    s.line(
                        x=[d.end.time, d.end.time],
                        y=[y.min(), y.max()],
                        color="salmon",
                        line_width=2,
                    )

        def add_box(label, column, y_reverse=False, **kwargs):
            # create a new plot and add a renderer
            s = figure(
                tools=TOOLS,
                width=width,
                height=300,
                title=label,
                x_axis_type="datetime",
                **kwargs,
            )
            s.line("time", column, source=df, line_width=lw, color=c)
            if cross:
                s.cross("time", column, source=df, color="orange", size=10)
            s.add_tools(
                HoverTool(
                    tooltips=[
                        ("Time", "@time{%F %T}"),
                        (label, "@{" + column + "}{0.0000f}"),
                    ],
                    formatters={
                        "@time": "datetime",
                        "@" + column: "printf",
                    },
                    mode="vline",
                )
            )
            _add_start_end(s, df[column])
            s.add_tools(crosshair)
            if y_reverse:
                s.y_range.flipped = True
            return s

        s1 = add_box("temperature [degC]", self._t)
        s2 = add_box("salinity [psu]", self._s, x_range=s1.x_range)
        s3 = add_box("depth [m]", self._p, y_reverse=True, x_range=s1.x_range)
        grid = [[s1, s2, s3]]
        s4 = add_box("longitude [deg]", self._lon, x_range=s1.x_range)
        s5 = add_box("latitude [deg]", self._lat, x_range=s1.x_range)
        grid = grid + [[s4, s5]]
        p = gridplot(grid)
        show(p)


def _resample(df, rule, op, interpolate, **kwargs):
    """temporal resampling"""
    if op == "mean":
        df = df.resample(rule, **kwargs).mean()
    elif op == "median":
        df = df.resample(rule, **kwargs).median()
    if interpolate:
        df = df.interpolate(method="slinear")
    return df


def _get_profile(df, depth_min, depth_max, step, speed_threshold, op):
    """construct a vertical profile from time series"""
    assert "depth" in df, "depth must be a column to produce a vertical profile"
    # make a copy and get rid of duplicates
    df = df[~df.index.duplicated(keep="first")]
    if "time" not in df.columns:
        # assumes time is the index
        df = df.reset_index()
    if speed_threshold:
        #    dt = pd.Series(df.index).diff()/pd.Timedelta("1s")
        #    dt.index = df.index
        #    df.loc[:,"dt"] = dt
        # else:
        df.loc[:, "dt"] = df.loc[:, "time"].diff() / pd.Timedelta("1s")
        dzdt = np.abs(df.loc[:, "depth"])
        df = df.loc[dzdt < speed_threshold]
    if depth_max is None:
        depth_max = float(df.loc[:, "depth"].max())
    bins = np.arange(depth_min, depth_max, step)
    df.loc[:, "depth_cut"] = pd.cut(df.loc[:, "depth"], bins)
    if op == "mean":
        df = (
            df.groupby(df.loc[:, "depth_cut"])
            .mean(numeric_only=False)
            .drop(columns=["depth"])
        )
    df.loc[:, "depth"] = df.index.map(lambda bin: bin.mid).astype(float)
    df.loc[:, "z"] = -df.loc[:, "depth"]
    return df.set_index("z").bfill()


# ----------------------------- xarray accessor --------------------------------


@xr.register_dataset_accessor("sw")
class XrSeawaterAccessor(SeawaterAccessor):
    def _validate(self, obj):
        """verify all necessary information is here"""

        # search for lon/lat in columns, as standard attribute, in attrs dict
        lon_names = ["longitude", "long", "lon"]
        lat_names = ["latitude", "lat"]
        for k in dir(obj):
            if k.lower() in lon_names:
                self._lon = getattr(obj, k)
            if k.lower() in lat_names:
                self._lat = getattr(obj, k)
        if not hasattr(self, "_lon"):
            raise AttributeError("Did not find an attribute longitude")
        if not hasattr(self, "_lat"):
            raise AttributeError("Did not find an attribute latitude")

        # deal now with actual seawater properties
        t, s, c, p, d = None, None, None, None, None
        for v in list(obj.variables):
            if v.lower() in _t_potential:
                t = v
            elif v.lower() in _s_potential:
                s = v
            elif v.lower() in _c_potential:
                c = v
            elif v.lower() in _p_potential:
                p = v
            elif v.lower() in _d_potential:
                d = v
        if not t or (not s and not c) or (not p and not d):
            raise AttributeError(
                "Did not find temperature, salinity and pressure columns. \n"
                + "Case insentive options are: "
                + "/".join(_t_potential)
                + " , "
                + "/".join(_s_potential)
                + " , \n"
                + "/".join(_c_potential)
                + " , "
                + "/".join(_p_potential)
            )
        else:
            # compute pressure from depth and depth from pressure if need be
            if not p:
                p = "pressure"
                obj[p] = gsw.p_from_z(-obj[d], self._lat)
            if not d:
                obj["depth"] = -gsw.z_from_p(obj[p], self._lat)
            return t, s, c, p

    def set_vdim(self, vdim):
        """let the user specify which dimension is the depth dimension"""
        self._vdim = vdim
        # self._odims = [d for d in self._obj[self._t].dims if d!=vdim]

    def _update_SA_PT(self):
        ds = self._obj
        t, s, p = ds[self._t], ds[self._s], ds[self._p]
        ds["SA"] = gsw.SA_from_SP(s, p, self._lon, self._lat)
        ds["CT"] = gsw.CT_from_t(ds.SA, t, p)
        # return ds

    def update_eos(self, ds=None, overwrite=True):
        if ds is None:
            ds = self._obj
        sa, ct, p = ds["SA"], ds["CT"], ds[self._p]
        if self._t not in ds or overwrite:
            ds[self._t] = gsw.t_from_CT(sa, ct, p)
        if self._s not in ds or overwrite:
            ds[self._s] = gsw.SP_from_SA(sa, p, self._lon, self._lat)
        if "sigma0" not in ds or overwrite:
            ds["sigma0"] = gsw.sigma0(sa, ct)

    def apply_with_eos_update(self, fun, *args, **kwargs):
        """apply a function and update eos related variables"""
        # apply function
        ds = fun(self._obj, *args, **kwargs)
        # update eos related variables
        ds.sw.init()
        ds = self.update_eos(ds)
        return ds

    def resample(
        self,
        dz,
        depth_max=None,
    ):
        """resample along depth coordinate

        Parameters
        ----------
        interpolate: boolean, optional
            turn on interpolation for upsampling
        kwargs:
            passed to resample
        """
        assert hasattr(
            self, "_vdim"
        ), "you need to run set_vdim first to specify the vertical dimension"
        ds = self._obj
        if not depth_max:
            depth_max = float(self._obj[self._d].max())
        depth_bins = np.arange(0, depth_max, dz)
        depth = (depth_bins[:-1] + depth_bins[1:]) * 0.5

        def _resample(ds, *args, **kwargs):
            dsr = (
                ds.groupby_bins(self._vdim, depth_bins, labels=depth)
                .mean(dim=self._vdim)
                .rename(**{self._vdim + "_bins": self._vdim})
            )
            return dsr

        return self.apply_with_eos_update(_resample)

    @property
    def sigma0(self):
        ds = self._obj
        if "sigma0" not in ds:
            sigma0 = gsw.sigma0(ds.SA, ds.CT)
        else:
            sigma0 = ds.sigma0
        return sigma0

    @property
    def N2(self):
        """compute buoyancy frequency"""
        assert hasattr(
            self, "_vdim"
        ), "you need to run set_vdim first to specify the vertical dimension"
        #
        ds = self._obj
        ds = ds.transpose(..., self._vdim)
        t, s, p = ds[self._t], ds[self._s], ds[self._p]
        assert p.ndim == 1, "pressure must be 1D at the moment"
        N2, p_mid = gsw.Nsquared(ds.SA, ds.CT, p, lat=self._lat, axis=0)
        # dp = (ds.DEPTH.isel(**{vdim: 1}) - ds.DEPTH.isel(**{vdim: 0}))
        # sign_dp = int(np.sign(dp).median().values)
        d_mid = self._vdim + "_mid"
        ds = ds.assign_coords(
            p_mid=((d_mid,), p_mid),
            z_mid=((d_mid,), -p_mid),  # approx here
        )
        # ds = ds.assign_coords(z_mid=-ds.DEPTH_MID)
        ds["N2"] = ((d_mid,), N2)
        return ds.N2


def plot_ts(s_lim, t_lim, figsize=None):
    """plot T/S diagram

    Parameters
    ----------
    s_lim: tuple
        salinity limits
    t_lim: tuple
        temperature limits

    """

    n = 100
    ds = xr.Dataset(
        dict(
            s=("s", np.linspace(*s_lim, n), dict(long_name="salinity")),
            t=("t", np.linspace(*t_lim, n), dict(long_name="temperature")),
        )
    )
    ds["temperature"] = ds["t"] + ds["s"] * 0
    ds["salinity"] = ds["t"] * 0 + ds["s"]
    ds["pressure"] = 1.0 + ds["salinity"] * 0
    ds["longitude"] = 0.0
    ds["latitude"] = 49.0

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    cs = ds.sw.sigma0.plot.contour(x="s", ax=ax, colors="k")
    ax.clabel(cs, inline=1, fontsize=10)
    ax.grid()

    return fig, ax
