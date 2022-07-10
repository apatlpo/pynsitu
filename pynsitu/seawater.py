import os

import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
from  matplotlib.dates import date2num, datetime
from matplotlib.colors import cnames
##
from bokeh.io import output_notebook, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool, FuncTickFormatter
from bokeh.models import CrosshairTool
from bokeh.plotting import figure

import gsw

# ------------------------------ parameters ------------------------------------


# ----------------------------- pandas geo extension --------------------------

@pd.api.extensions.register_dataframe_accessor("sw")
class PdSeawaterAccessor:
    def __init__(self, pandas_obj):
        self._t, self._s, self._p = self._validate(pandas_obj)
        self._obj = pandas_obj
        self._update_SA_PT()

    #@staticmethod
    def _validate(self, obj):
        """verify there are columns for temperature, salinity and pressure
        Check longitude are
        """
        if hasattr(obj, "longitude"):
            self._lon = obj.longitude
        else:
            raise AttributeError("Did not find an attribute longitude")
        if hasattr(obj, "latitude"):
            self._lat = obj.latitude
        else:
            raise AttributeError("Did not find an attribute latitude")

        t, s, p = None, None, None
        t_potential = ["temperature", "temp", "t"]
        s_potential = ["salinity", "s"]
        p_potential = ["pressure", "p"]
        d_potential = ["depth",]
        for c in list(obj.columns):
            if c.lower() in t_potential:
                t = c
            elif c.lower() in s_potential:
                s = c
            elif c.lower() in p_potential:
                p = c
            elif c.lower() in d_potential:
                d = c
        if not t or not s or (not p and not d):
            raise AttributeError("Did not find temperature, salinity and pressure columns. "
                                 +"Case insentive options are: "
                                 + "/".join(t_potential)
                                 + " , " + "/".join(s_potential)
                                 + " , " + "/".join(p_potential)
                                )
        else:
            # compute pressure from depth and depth from pressure if need be
            if not p:
                p = "pressure"
                obj[p] = gsw.p_from_z(-obj[d], self._lat.median())
            if not d:
                obj["depth"] = -gsw.z_from_p(obj[p], self._lat.median())
            return t, s, p

    def _update_SA_PT(self):
        df = self._obj
        t, s, p = df[self._t], df[self._s], df[self._p]
        df['SA'] = gsw.SA_from_SP(s, p, self._lon, self._lat)
        df['CT'] = gsw.CT_from_t(df.SA, t, p)
        df["sigma0"] = gsw.sigma0(df.SA, df.CT)

    def trim(self, d):
        """given a deployment item, trim data"""
        time = self._obj.index
        df = self._obj.loc[(time >= d.start.time) & (time <= d.end.time)]
        return df

    def apply_PA_PT(self, fun, *args, **kwargs):
        """ apply a function that requires working with projected coordinates x/y"""
        # ensures projection exists
        #self.project()
        # apply function
        df = fun(self._obj, *args, **kwargs)
        # update lon/lat
        #df[self._lon], df[self._lat] = _xy2lonlat(df.x, df.y, self.projection)
        return df

    def resample(self,
                 rule,
                 interpolate=False,
                 #inplace=True,
                 op="mean",
                 **kwargs,
        ):
        ''' temporal resampling
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html

        Parameters
        ----------
        rule: DateOffset, Timedelta or str
            Passed to pandas.DataFrame.resample, examples:
                - '10T': 10 minutes
                - '10S': 10 seconds
        inplace: boolean, optional
            turn inplace resampling on, default is False
        interpolate: boolean, optional
            turn on interpolation for upsampling
        kwargs:
            passed to resample
        '''
        return self.apply_PA_PT(_resample, rule, op, interpolate, **kwargs)

    def compute_vertical_profile(self,
                                 step,
                                 speed_threshold=None,
                                 op="mean",
                                 depth_min=0,
                                 depth_max=None,
                                 ):
        return self.apply_PA_PT(_get_profile,
                                depth_min, depth_max, step, speed_threshold,
                                op,
                                )

    def plot_bokeh(self,
                   unit=None,
                   rule=None,
                   plot_width=400,
        ):
        """ bokeh plot

        Parameters
        ----------
        unit:
        rule:
        """

        if rule is not None:
            df = self.resample(rule)
        else:
            df = self._obj

        output_notebook()
        TOOLS = 'pan,wheel_zoom,box_zoom,reset,help'
        crosshair = CrosshairTool(dimensions="both")

        # line specs
        lw, c = 3, 'black'

        def _add_start_end(s, y):
            #_y = y.iloc[y.index.get_loc(_d.start.time), method='nearest')]
            if unit is not None:
                for _d in unit:
                    s.line(x=[_d.start.time, _d.start.time],
                           y=[y.min(), y.max()],
                           color='cadetblue', line_width=2)
                    s.line(x=[_d.end.time, _d.end.time],
                           y=[y.min(), y.max()],
                           color='salmon', line_width=2)


        def add_box(label, column, y_reverse=False, **kwargs):
            # create a new plot and add a renderer
            s = figure(tools=TOOLS,
                       plot_width=plot_width,
                       plot_height=300,
                       title=label,
                       x_axis_type='datetime',
                       **kwargs,
                       )
            s.line('time', column, source=df, line_width=lw, color=c)
            s.add_tools(HoverTool(
                tooltips=[('Time','@time{%F %T}'),
                          (label, '@{'+column+'}{0.0000f}'),
                          ],
                formatters={'@time': 'datetime','@'+column: 'printf',},
                mode='vline'
                ))
            _add_start_end(s, df[column])
            s.add_tools(crosshair)
            if y_reverse:
                s.y_range.flipped = True
            return s

        s1 = add_box("temperature [degC]", self._t)
        s2 = add_box("salinity [psu]", self._s, x_range=s1.x_range)
        s3 = add_box("depth [m]", self._p, y_reverse=True, x_range=s1.x_range)
        grid = [[s1, s2, s3]]
        if not isinstance(self._lon, float) and not isinstance(self._lat, float):
            s4 = add_box("longitude [deg]", "longitude", x_range=s1.x_range)
            s5 = add_box("latitude [deg]", "latitude", x_range=s1.x_range)
            grid = grid+[[s4, s5]]
        p = gridplot(grid)
        show(p)

def _resample(df, rule, op, interpolate, **kwargs):
    """temporal resampling"""
    if op=="mean":
        df = df.resample(rule, **kwargs).mean()
    elif op=="median":
        df = df.resample(rule, **kwargs).median()
    if interpolate:
        df = df.interpolate(method='linear')
    return df

def _get_profile(df, depth_min, depth_max, step, speed_threshold, op):
    """ construct a vertical profile from time series
    """
    assert "depth" in df, "depth must be a column to produce a vertical profile"
    # make a copy and get rid of duplicates
    df = df[~df.index.duplicated(keep='first')]
    if speed_threshold:
        if "time" not in df.colums:
            # assumes time is the index
            dt = pd.Series(df.index).diff()/pd.Timedelta("1s")
            dt.index = df.index
            df["dt"] = dt
        else:
            df["dt"] = df["time"].diff()/pd.Timedelta("1s")
        dzdt = np.abs(df["depth"])
        df = df.loc[dzdt<speed_threshold]
    if depth_max is None:
        depth_max = float(df["depth"].max())
    bins = np.arange(depth_min, depth_max, step)
    df["depth_cut"] = pd.cut(df.depth, bins)
    if op=="mean":
        df = df.groupby(df.depth_cut).mean().drop(columns=["depth"])
    print()
    df["depth"] = df.index.map(lambda bin: bin.mid).astype(float)
    df["z"] = -df["depth"]
    return df.set_index("z")

# ----------------------------- xarray accessor --------------------------------

@xr.register_dataset_accessor("sw")
class XrSeawaterAccessor:
    def __init__(self, xarray_obj):
        self._t, self._s, self._p = self._validate(xarray_obj)
        self._obj = xarray_obj
        self._update_SA_PT()

    def _validate(self, obj):
        """verify there are columns for temperature, salinity and pressure
        Check longitude are
        """
        if hasattr(obj, "longitude"):
            self._lon = obj.longitude
        else:
            raise AttributeError("Did not find an attribute longitude")
        if hasattr(obj, "latitude"):
            self._lat = obj.latitude
        else:
            raise AttributeError("Did not find an attribute latitude")

        t, s, p = None, None, None
        t_potential = ["temperature", "temp", "t"]
        s_potential = ["salinity", "s"]
        p_potential = ["pressure", "p"]
        for c in list(obj.variables):
            if c.lower() in t_potential:
                t = c
            elif c.lower() in s_potential:
                s = c
            elif c.lower() in p_potential:
                p = c
        if not t or not s or not p:
            raise AttributeError("Did not find temperature, salinity and pressure columns. "
                                 +"Case insentive options are: "
                                 + "/".join(t_potential)
                                 + " , " + "/".join(s_potential)
                                 + " , " + "/".join(p_potential)
                                )
        else:
            return t, s, p

    def _update_SA_PT(self):
        ds = self._obj
        t, s, p = ds[self._t], ds[self._s], ds[self._p]
        ds['SA'] = gsw.SA_from_SP(s, p, self._lon, self._lat)
        ds['CT'] = gsw.CT_from_t(ds.SA, t, p)
        ds["sigma0"] = gsw.sigma0(ds.SA, ds.CT)

    @property
    def sigma0(self):
        return self._obj["sigma0"]


def plot_ts(s_lim, t_lim):
    """plot T/S diagram

    Parameters
    ----------
    s_lim: tuple
        salinity limits
    t_lim: tuple
        temperature limits

    """

    n=100
    ds = xr.Dataset(dict(salinity=("salinity", np.linspace(*s_lim, n)),
                         temperature=("temperature", np.linspace(*t_lim, n)),
                   )
              )
    ds["pressure"] = 1.
    ds["longitude"] = 0.
    ds["latitude"] = 49.

    fig, ax = plt.subplots(1,1)
    cs = ds.sw.sigma0.plot.contour(x="salinity", ax=ax, colors="k")
    ax.clabel(cs, inline=1, fontsize=10)
    ax.grid()

    return fig, ax
