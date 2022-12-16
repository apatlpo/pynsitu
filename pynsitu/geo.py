import numpy as np
import xarray as xr
import pandas as pd

# import geopandas as gpd
# from shapely.geometry import Polygon, Point
# from shapely import wkt
# from shapely.ops import transform
try:
    import pyproj

    crs_wgs84 = pyproj.CRS("EPSG:4326")
except:
    print("Warning: could not import pyproj")

# import pyinterp.geohash as geohash
# import geojson

import matplotlib.pyplot as plt

# from matplotlib.dates import date2num, datetime
# from matplotlib.colors import cnames

try:
    from bokeh.io import output_notebook, show
    from bokeh.layouts import gridplot
    from bokeh.models import HoverTool, CustomJSHover
    from bokeh.models import CrosshairTool
    from bokeh.plotting import figure
except:
    print("Warning: could not import bokeh")
    CustomJSHover = lambda *args, **kargs: None

# ------------------------------ parameters ------------------------------------

g = 9.81
deg2rad = np.pi / 180.0
rad2deg = 180.0 / np.pi
deg2m = 111319
earth_radius = 6378.0

omega_earth = 2.0 * np.pi / 86164.0905

# ------------------------------ Coriolis --------------------------------------


def coriolis(lat, signed=True):
    if signed:
        return 2.0 * omega_earth * np.sin(lat * deg2rad)
    else:
        return 2.0 * omega_earth * np.sin(np.abs(lat) * deg2rad)


def dfdy(lat, units="1/s/m"):
    df = 2.0 * omega_earth * np.cos(lat * deg2rad) * deg2rad / deg2m
    if units == "cpd/100km":
        df = df * 86400 / 2.0 / np.pi * 100 * 1e3
    return df


# ----------------------------- projections  -----------------------------------


class projection(object):
    """wrapper around pyproj to easily convert to local cartesian coordinates"""

    def __init__(self, lon_ref, lat_ref):
        self.proj = pyproj.Proj(
            proj="aeqd",
            lat_0=lat_ref,
            lon_0=lon_ref,
            datum="WGS84",
            units="m",
        )

    def lonlat2xy(self, lon, lat):
        """transforms lon, lat to x,y coordinates"""
        return self.proj.transform(lon, lat)

    def xy2lonlat(self, x, y):
        """transforms x,y to lon, lat coordinates"""
        _inv_dir = pyproj.enums.TransformDirection.INVERSE
        return self.proj.transform(x, y, direction=_inv_dir)


# ----------------------------- lon/lat formatters  ----------------------------


def dec2degmin(dec):
    """decimal degrees to degrees and minutes"""
    sign = np.sign(dec)
    adeg = int(abs(dec))
    min = (abs(dec) - adeg) * 60.0
    return [int(sign * adeg), min]


def degmin2dec(deg, min):
    """converts lon or lat in deg, min to decimal"""
    return deg + np.sign(deg) * min / 60.0


def print_degmin(l):
    """Print lon/lat, deg + minutes decimales"""
    dm = dec2degmin(l)
    # return '%d deg %.5f' %(int(l), (l-int(l))*60.)
    return "{} {:.5f}".format(*dm)


## bokeh formatters
lon_hover_formatter = CustomJSHover(
    code="""
    var D = value;
    var deg = Math.abs(Math.trunc(D));
    if (D<0){
        var dir = "W";
    } else {
        var dir = "E";
    }
    var min = (Math.abs(D)-deg)*60.0;
    return deg + dir + " " + min.toFixed(3)
"""
)

lat_hover_formatter = CustomJSHover(
    code="""
    var D = value;
    var deg = Math.abs(Math.trunc(D));
    if (D<0){
        var dir = "S";
    } else {
        var dir = "N";
    }
    var min = (Math.abs(D)-deg)*60.0;
    return deg + dir + " " + min.toFixed(3)
"""
)


# ----------------------------- pandas geo extension --------------------------


def _xy2lonlat(x, y, proj=None):
    """compute longitude/latitude from projected coordinates"""
    _inv_dir = pyproj.enums.TransformDirection.INVERSE
    assert proj is not None, "proj must not be None"
    return proj.transform(x, y, direction=_inv_dir)


@pd.api.extensions.register_dataframe_accessor("geo")
class GeoAccessor:
    def __init__(self, pandas_obj):
        self._lon, self._lat = self._validate(pandas_obj)
        self._obj = pandas_obj
        self._reset_geo()

    # @staticmethod
    def _validate(self, obj):
        """verify there is a column latitude and a column longitude"""
        lon, lat = None, None
        lat_potential = ["lat", "latitude"]
        lon_potential = ["lon", "longitude"]
        for c in list(obj.columns):
            if c.lower() in lat_potential:
                lat = c
            elif c.lower() in lon_potential:
                lon = c
        if not lat or not lon:
            raise AttributeError(
                "Did not find latitude and longitude columns. Case insentive options are: "
                + "/".join(lat_potential)
                + " , "
                + "/".join(lon_potential)
            )
        else:
            return lon, lat

    def _reset_geo(self):
        """reset all variables related to geo"""
        self._geo_proj_ref = None
        self._geo_proj = None
        # self._obj.drop(columns=["x", "y"], errors="ignore", inplace=True)

    @property
    def projection_reference(self):
        """define a reference projection if none is available"""
        if self._geo_proj_ref is None:
            # return the geographic center point of this DataFrame
            lat, lon = self._obj[self._lat], self._obj[self._lon]
            self._geo_proj_ref = (float(lon.iloc[0]), float(lat.iloc[0]))
        return self._geo_proj_ref

    def set_projection_reference(self, ref, reset=True):
        """set projection reference point, (lon, lat) tuple"""
        if reset:
            self._reset_geo()
        self._geo_proj_ref = ref

    @property
    def projection(self):
        if self._geo_proj is None:
            lonc, latc = self.projection_reference
            self._geo_proj = projection(lonc, latc)
            # self._geo_proj = pyproj.Proj(proj="aeqd",
            #                             lat_0=latc, lon_0=lonc,
            #                             datum="WGS84", units="m")
        return self._geo_proj

    def project(self, overwrite=True):
        """add (x,y) projection to object"""
        d = self._obj
        if "x" not in d.columns or "y" not in d.columns or overwrite:
            d.loc[:, "x"], d.loc[:, "y"] = self.projection.lonlat2xy(
                d.loc[:, self._lon],
                d.loc[:, self._lat],
            )

    def compute_lonlat(self):
        """update longitude and latitude from projected coordinates"""
        d = self._obj
        assert ("x" in d.columns) and (
            "y" in d.columns
        ), "x/y coordinates must be available"
        d.loc[:, self._lon], d.loc[:, self._lat] = self.projection.xy2lonlat(
            d["x"], d["y"]
        )

    # time series and/or campaign related material

    def trim(self, d):
        """given a deployment item, trim data"""
        time = self._obj.index
        df = self._obj.loc[(time >= d.start.time) & (time <= d.end.time)]
        return df

    def apply_xy(self, fun, *args, **kwargs):
        """apply a function that requires working with projected coordinates x/y"""
        # ensures projection exists
        self.project()
        # apply function
        df = fun(self._obj, *args, **kwargs)
        # update lon/lat
        df.loc[:, self._lon], df.loc[:, self._lat] = self.projection.xy2lonlat(
            df["x"], df["y"]
        )
        return df

    def resample(
        self,
        rule,
        interpolate=False,
        # inplace=True,
        **kwargs,
    ):
        """temporal resampling
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
        """

        def _resample(df):
            df = df.resample(rule, **kwargs).mean()
            if interpolate:
                df = df.interpolate(method="linear")
            return df

        return self.apply_xy(_resample)

    def compute_velocities(
        self,
        time="index",
        distance="geoid",
        centered=True,
        acceleration=False,
        keep_dt=False,
        fill_startend=True,
    ):
        """compute velocity
        Parameters
        ----------
        time: str, optional
            Column name. Default is "index", i.e. considers the index
        distance: str, optional
            Method to compute distances.
            Default is geoid ("WGS84" with pyproj).
            Uses projected fields otherwise ("x", "y")
        centered: boolean
            Centers velocity calculation temporally (True by default).
        acceleration: boolean
            Computes acceleration (False by default).
        keep_dt: boolean
            Keeps time intervals (False by default).
        """
        return self.apply_xy(
            _compute_velocities,
            self._lon,
            self._lat,
            time,
            distance,
            centered,
            acceleration,
            keep_dt,
            fill_startend,
        )

    # --- transect
    def compute_transect(self, ds, vmin=None, dt_max=None):
        """Average data along a transect of step ds

        Parameters
        ----------
        ds: float
            transect spacing in meters
        vmin: float, optional
            ship minimum speed, used to compute a maximum search time for each
            transect cell
        dt_max: pd.Timedelta, optional
            maximum search time for each transect cell
        """

        # compute velocities, thereby ensures projection exists
        df = self.compute_velocities()

        # init transect time and position
        t = df.index[0]
        x = df.x[0]
        y = df.y[0]

        if vmin is not None:
            dt_max = pd.Timedelta(ds / vmin, unit="seconds")

        T, D = [], []
        while t:
            t, x, y, d = _step_trajectory(df, t, x, y, ds, dt_max)
            if t:
                T.append(t)
                D.append(d)

        df = pd.concat(D, axis=1).T
        df.loc[:, "time"] = T

        # compute and add along-transect coordinate
        dx = df.x.diff().fillna(0)
        dy = df.y.diff().fillna(0)
        s = np.sqrt(dx**2 + dy**2).cumsum()
        df.loc[:, "s"] = s

        return df.set_index("s")

    # ---- plotting

    def plot_lonlat(self):
        """simple lon/lat plot"""
        # plot this array's data on a map, e.g., using Cartopy
        df = self._obj
        fig, ax = plt.subplots(1, 1)
        ax.plot(df[self._lon], df[self._lat])

    def plot_bokeh(
        self,
        deployments=None,
        rule=None,
        mindec=True,
        velocity=False,
        acceleration=False,
    ):
        """ Bokeh plot, useful to clean data

        Parameters
        ----------
        deployments: dict-like, pynsitu.events.Deployments for instance, optional
            Deployments
        rule: str, optional
            resampling rule
        mindec: boolean
            Plot longitude and latitude as minute/decimals
        """

        if rule is not None:
            df = self.resample(rule)
        else:
            df = self._obj

        if (velocity and "velocity" not in df.columns) or (
            acceleration and "acceleration" not in df.columns
        ):
            df = df.geo.compute_velocities(acceleration=acceleration)

        if mindec:
            _lon_tooltip = "@" + self._lon + "{custom}"
            _lat_tooltip = "@" + self._lat + "{custom}"
            _lon_formatter = lon_hover_formatter
            _lat_formatter = lat_hover_formatter
            # ll_formater = FuncTickFormatter(code="""
            #    return Math.floor(tick) + " + " + (tick % 1).toFixed(2)
            # """)
        else:
            _lon_tooltip = "@{" + self._lon + "}{0.4f}"
            _lat_tooltip = "@{" + self._lat + "}{0.4f}"
            _lon_formatter = "printf"
            _lat_formatter = "printf"

        output_notebook()
        figkwargs = dict(
            tools="pan,wheel_zoom,box_zoom,reset,help",
            plot_width=350,
            plot_height=300,
            x_axis_type="datetime",
        )
        crosshair = CrosshairTool(dimensions="both")
        # line specs
        lw, c = 3, "black"

        from .events import Deployment
        if deployments is not None:
            if isinstance(deployments, Deployment):
                deployments = deployments.to_deployments()

        def _add_start_end(s, ymin, ymax=None):
            """add deployments start and end as colored vertical bars"""
            # _y = y.iloc[y.index.get_loc(_d.start.time), method='nearest')]
            if ymax is None:
                ymax = ymin.max()
            elif not isinstance(ymax, float):
                ymax = ymax.max()
            if not isinstance(ymin, float):
                ymin = ymin.min()
            print(ymin, ymax)
            if deployments is not None:
                for label, d in deployments.items():
                    s.line(
                        x=[d.start.time, d.start.time],
                        y=[ymin, ymax],
                        color="cadetblue",
                        line_width=2,
                    )
                    s.line(
                        x=[d.end.time, d.end.time],
                        y=[ymin, ymax],
                        color="salmon",
                        line_width=2,
                    )

        # create a new plot and add a renderer
        s1 = figure(title="longitude", **figkwargs)
        s1.line("time", self._lon, source=df, line_width=lw, color=c)
        s1.add_tools(
            HoverTool(
                tooltips=[
                    ("Time", "@time{%F %T}"),
                    ("longitude", _lon_tooltip),
                ],  #
                formatters={
                    "@time": "datetime",
                    "@" + self._lon: _lon_formatter,
                },  #'printf'
                mode="vline",
            )
        )
        _add_start_end(s1, df[self._lon])
        s1.add_tools(crosshair)
        S = [s1]
        #
        s2 = figure(title="latitude", x_range=s1.x_range, **figkwargs)
        s2.line("time", self._lat, source=df, line_width=lw, color=c)
        s2.add_tools(
            HoverTool(
                tooltips=[
                    ("Time", "@time{%F %T}"),
                    ("latitude", _lat_tooltip),
                ],
                formatters={
                    "@time": "datetime",
                    "@" + self._lat: _lat_formatter,
                },
                mode="vline",
            )
        )
        _add_start_end(s2, df[self._lat])
        s2.add_tools(crosshair)
        S.append(s2)
        #
        if velocity:
            s3 = figure(title="speed", x_range=s1.x_range, **figkwargs)
            s3.line(
                "time",
                "velocity",
                source=df,
                line_width=lw,
                color=c,
                legend_label="velocity",
            )
            s3.line(
                "time",
                "velocity_east",
                source=df,
                line_width=lw,
                color="orange",
                legend_label="velocity_east",
            )
            s3.line(
                "time",
                "velocity_north",
                source=df,
                line_width=lw,
                color="blue",
                legend_label="velocity_north",
            )
            s3.add_tools(
                HoverTool(
                    tooltips=[
                        ("Time", "@time{%F %T}"),
                        ("Velocity", "@{velocity}{0.3f} m/s"),
                    ],
                    formatters={
                        "@time": "datetime",
                        "@velocity": "printf",
                    },
                    mode="vline",
                )
            )
            _add_start_end(s3, -np.abs(df["velocity"]), np.abs(df["velocity"]))
            s3.add_tools(crosshair)
            S.append(s3)
        if acceleration:
            s4 = figure(title="acceleration", x_range=s1.x_range, **figkwargs)
            s4.line("time", "acceleration", source=df, line_width=lw, color=c)
            s4.line(
                "time",
                "acceleration_east",
                source=df,
                line_width=lw,
                color="orange",
                legend_label="acceleration_east",
            )
            s4.line(
                "time",
                "acceleration_north",
                source=df,
                line_width=lw,
                color="blue",
                legend_label="acceleration_north",
            )
            s4.add_tools(
                HoverTool(
                    tooltips=[
                        ("Time", "@time{%F %T}"),
                        ("Acceleration", "@{acceleration}{0.2e} m/s^2"),
                    ],
                    formatters={
                        "@time": "datetime",
                        "@acceleration": "printf",
                    },
                    mode="vline",
                )
            )
            _add_start_end(s4, -df["acceleration"], ymax = df["acceleration"])
            s4.add_tools(crosshair)
            S.append(s4)
        #
        p = gridplot(S, ncols=2)
        show(p)

    def plot_bokeh_map(self, unit=None, rule=None, mindec=True):
        """bokeh plot"""

        if rule is not None:
            df = self.resample(rule)
        else:
            df = self._obj
        # ensure we have projection
        self.project()

        if mindec:
            _lon_tooltip = "@" + self._lon + "{custom}"
            _lat_tooltip = "@" + self._lat + "{custom}"
            _lon_formatter = lon_hover_formatter
            _lat_formatter = lat_hover_formatter
            # ll_formater = FuncTickFormatter(code="""
            #    return Math.floor(tick) + " + " + (tick % 1).toFixed(2)
            # """)
        else:
            _lon_tooltip = "@{" + self._lon + "}{0.4f}"
            _lat_tooltip = "@{" + self._lat + "}{0.4f}"
            _lon_formatter = "printf"
            _lat_formatter = "printf"

        output_notebook()
        TOOLS = "pan,wheel_zoom,box_zoom,reset,help"

        # line specs
        lw = 5
        c = "black"

        # create a new plot and add a renderer
        s1 = figure(
            tools=TOOLS,
            plot_width=600,
            plot_height=600,
            title="map",
            match_aspect=True,  # if projected for equal axis
            # x_axis_type='datetime',
        )
        s1.line("x", "y", source=df, line_width=lw, color=c)
        s1.add_tools(
            HoverTool(
                tooltips=[
                    ("Time", "@time{%F %T}"),
                    ("longitude", _lon_tooltip),
                    ("latitude", _lat_tooltip),
                ],
                formatters={
                    "@time": "datetime",
                    "@" + self._lon: _lon_formatter,
                    "@" + self._lat: _lat_formatter,
                },
                # mode='vline'
            )
        )

        p = gridplot(
            [
                [
                    s1,
                ]
            ]
        )
        show(p)


def _step_trajectory(df, t, x, y, ds, dt_max):
    """compute next position along transect"""

    # select temporally
    df = df.loc[(df.index > t)].copy()
    if dt_max:
        df = df.loc[(df.index < t + dt_max)]

    # select spatially
    df.loc[:, "s"] = np.sqrt((df.loc[:, "x"] - x) ** 2 + (df.loc[:, "y"] - y) ** 2)
    df = df.loc[(df.loc[:, "s"] > ds / 2) & (df.loc[:, "s"] < 1.5 * ds)]

    if df.empty:
        return None, None, None, None

    t = df.reset_index().loc[:, "time"].mean()
    dfm = df.mean()
    x, y = dfm["x"], dfm["y"]

    return t, x, y, dfm


def _compute_velocities(
    df,
    lon_key,
    lat_key,
    time,
    distance,
    centered,
    acceleration,
    keep_dt,
    fill_startend,
):
    """core method to compute velocity from a dataframe"""
    # drop duplicates
    df = df[~df.index.duplicated(keep="first")].copy()
    # dt_i = t_i - t_{i-1}
    if time == "index":
        t = df.index.to_series()
        dt = t.diff() / pd.Timedelta("1s")
        dt.index = df.index  # necessary?
        df.loc[:, "dt"] = dt
    else:
        t = df[time]
        df.loc[:, "dt"] = t.diff() / pd.Timedelta("1s")
    is_uniform = df["dt"].dropna().unique().size == 1
    if distance == "geoid":
        from pyproj import Geod

        g = Geod(ellps="WGS84")
        # see:
        #   https://pyproj4.github.io/pyproj/stable/api/geod.html#pyproj.Geod.inv
        #   https://proj.org/usage/ellipsoids.html
        lon, lat = df[lon_key], df[lat_key]
        az12, az21, dist = g.inv(lon.shift(1), lat.shift(1), lon, lat)
        # need to convert into dx and dy
        dxdt = pd.Series(dist * np.sin(az12 * deg2rad), index=df.index) / df["dt"]
        dydt = pd.Series(dist * np.cos(az12 * deg2rad), index=df.index) / df["dt"]
    else:
        # leverage local projection, less accurate away from central point
        dxdt = df["x"].diff() / df["dt"]  # u_i = x_i - x_{i-1}
        dydt = df["y"].diff() / df["dt"]  # v_i = y_i - y_{i-1}
    if centered:
        w = dt / (dt + dt.shift(-1))
        df.loc[:, "velocity_east"] = dxdt + (dxdt.shift(-1) - dxdt) * w
        df.loc[:, "velocity_north"] = dydt + (dydt.shift(-1) - dxdt) * w
    else:
        df.loc[:, "velocity_east"] = dxdt
        df.loc[:, "velocity_north"] = dydt
    df.loc[:, "velocity"] = np.sqrt(
        df.loc[:, "velocity_east"] ** 2 + df.loc[:, "velocity_north"] ** 2
    )
    if acceleration:
        dt_acc = (dt.shift(-1) + dt) * 0.5
        df.loc[:, "acceleration_east"] = (dxdt.shift(-1) - dxdt) / dt_acc
        df.loc[:, "acceleration_north"] = (dydt.shift(-1) - dydt) / dt_acc
        df.loc[:, "acceleration"] = np.sqrt(
            df["acceleration_east"] ** 2 + df["acceleration_north"] ** 2
        )
    if not keep_dt:
        df = df.drop(columns=["dt"])
    if fill_startend:
        # fill end values
        df = df.bfill().ffill()
    return df


# ----------------------------- xarray accessor --------------------------------


@xr.register_dataset_accessor("geo")
class XrGeoAccessor:
    def __init__(self, xarray_obj):
        self._lon, self._lat = self._validate(xarray_obj)
        self._obj = xarray_obj
        self._reset_geo()

    # @staticmethod
    def _validate(self, obj):
        """verify there are latitude and longitude variables"""
        lon, lat = None, None
        lat_potential = ["lat", "latitude"]
        lon_potential = ["lon", "longitude"]
        for c in list(obj.variables):
            if c.lower() in lat_potential:
                lat = c
            elif c.lower() in lon_potential:
                lon = c
        if not lat or not lon:
            raise AttributeError(
                "Did not find latitude and longitude variables. Case insentive options are: "
                + "/".join(lat_potential)
                + " , "
                + "/".join(lon_potential)
            )
        else:
            return lon, lat

    def _reset_geo(self):
        """reset all variables related to geo"""
        self._geo_proj_ref = None
        self._geo_proj = None

    def set_projection_reference(self, ref, reset=True):
        """set projection reference point, (lon, lat) tuple"""
        if reset:
            self._reset_geo()
        self._geo_proj_ref = ref

    @property
    def projection(self):
        if self._geo_proj is None:
            lonc, latc = self._geo_proj_ref
            self._geo_proj = pyproj.Proj(
                proj="aeqd",
                lat_0=latc,
                lon_0=lonc,
                datum="WGS84",
                units="m",
            )
        return self._geo_proj

    def project(self, overwrite=True, **kwargs):
        """add (x,y) projection to object"""
        d = self._obj
        dkwargs = dict(vectorize=True)
        dkwargs.update(**kwargs)
        if "x" not in d.variables or "y" not in d.variables or overwrite:
            proj = self.projection.transform
            if True:
                _x, _y = proj(
                    d[self._lon],
                    d[self._lat],
                )
                dims = d[self._lon].dims
                d["x"], d["y"] = (dims, _x), (dims, _y)
            else:
                d["x"], d["y"] = xr.apply_ufunc(
                    self.projection.transform, d[self._lon], d[self._lat], **dkwargs
                )

    def compute_lonlat(self, x=None, y=None, **kwargs):
        """update longitude and latitude from projected coordinates"""
        d = self._obj
        assert ("x" in d.variables) and (
            "y" in d.variables
        ), "x/y coordinates must be available"
        dkwargs = dict()
        dkwargs.update(**kwargs)
        if x is not None and y is not None:
            lon, lat = _xy2lonlat(x, y, proj=self.projection)
            return (x.dims, lon), (x.dims, lat)
        else:
            d[self._lon], d[self._lat] = xr.apply_ufunc(
                _xy2lonlat,
                d["x"],
                d["y"],
                kwargs=dict(proj=self.projection),
                **dkwargs,
            )

    # time series related code

    # speed ...
