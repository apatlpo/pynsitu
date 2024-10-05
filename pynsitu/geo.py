import warnings

import numpy as np
import numpy.testing as npt
import xarray as xr
import pandas as pd

try:
    import pyproj

    crs_wgs84 = pyproj.CRS("EPSG:4326")
except:
    print("Warning: could not import pyproj")

import matplotlib.pyplot as plt

try:
    from bokeh.io import output_notebook, show
    from bokeh.layouts import gridplot
    from bokeh.models import HoverTool, CustomJSHover
    from bokeh.models import CrosshairTool
    from bokeh.plotting import figure
    import hvplot.pandas
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
    """returns Coriolis frequency in rad/s"""
    if signed:
        return 2.0 * omega_earth * np.sin(lat * deg2rad)
    else:
        return 2.0 * omega_earth * np.sin(np.abs(lat) * deg2rad)


def dfdy(lat, units="1/s/m"):
    df = 2.0 * omega_earth * np.cos(lat * deg2rad) * deg2rad / deg2m
    if units == "cpd/100km":
        df = df * 86400 / 2.0 / np.pi * 100 * 1e3
    return df


# ----------------------------- projections, distances  -----------------------


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


def azimuth_distance(lon0, lat0, lon1, lat1, ellps="WGS84"):
    """compute azimuths and distances between two points

    Returns
    -------
    az12 (deg), az21 (deg), dist (meters)

    """
    g = pyproj.Geod(ellps=ellps)
    # az12,az21,dist
    return g.inv(lon0, lat0, lon1, lat1)


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

# -----------------------------  geo extension, common part -------------------


class GeoAccessor:
    def __init__(self, _obj):
        self._lon, self._lat = self._validate(_obj)
        self._obj = _obj
        self._reset_geo()

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
    def projection_reference(self):
        """define a reference projection if none is available"""
        if self._geo_proj_ref is None:
            # return the geographic center point
            lat, lon = self._obj[self._lat], self._obj[self._lon]
            lon_ref, lat_ref = lon.median(), lat.median()
            assert not np.isnan(lat_ref) and not np.isnan(
                lon_ref
            ), "lat, lon data do not contain any valid data"
            self._geo_proj_ref = (lon_ref, lat_ref)
        return self._geo_proj_ref

    @property
    def projection(self):
        if self._geo_proj is None:
            lonc, latc = self.projection_reference
            self._geo_proj = projection(lonc, latc)
        return self._geo_proj


# ----------------------------- pandas geo extension --------------------------


@pd.api.extensions.register_dataframe_accessor("geo")
class PdGeoAccessor(GeoAccessor):
    """Pandas DataFrame accessor in order to process geographical data"""

    # @staticmethod
    def _validate(self, obj):
        """verify there is a column latitude and a column longitude"""
        lon, lat = None, None
        lat_potential = ["lat", "latitude"]
        lon_potential = ["lon", "long", "longitude"]
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
        d[self._lon], d[self._lat] = self.projection.xy2lonlat(d["x"], d["y"])

    ## --- time series and/or campaign related material

    def trim(self, d):
        """given a deployment item, trim data"""
        time = self._obj.index
        df = self._obj.loc[(time >= d.start.time) & (time <= d.end.time)]
        return df

    def apply_xy(self, fun, **kwargs):
        """apply a function that requires working with projected coordinates x/y"""
        # ensures projection exists
        self.project()
        # apply function
        df = fun(self._obj, **kwargs)
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

    ## --- velocity/acceleration

    def compute_velocities(
        self,
        time="index",
        distance="geoid",
        centered=True,
        keep_dt=False,
        fill_startend=True,
        names=None,
        inplace=False,  # need to return something to give to apply_xy
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
        centered: boolean, optional
            Centers velocity calculation temporally (True by default).
        keep_dt: boolean, optional
            Keeps time intervals (False by default).
        fill_startend : boolean, optional
            fill dataframe start and end (Nan values due to the derivation/centering method) (True by default).
        names :  tuple, optional
            Contains columns names for eastern, northen and norm velocities
            ("velocity_east", "velocity_north", "velocity" by default
        inplace : boolean, optional
            if True add velocities to dataset, if False return only a dataframe with time, id (for identification) and computed velocities
        """
        if "x" not in self._obj.columns or "y" not in self._obj.columns:
            self.project()
        df = compute_velocities(
            self._obj,
            time,
            names=names,
            centered=centered,
            fill_startend=fill_startend,
            lon_key=self._lon,
            lat_key=self._lat,
            distance=distance,
            keep_dt=keep_dt,
            inplace=inplace,
        )

        if not inplace:
            return df

    def compute_dt(
        self,
        time="index",
        fill_startend=True,
        inplace=False,  # need to return something to give to apply_xy
    ):
        """compute dt
        Parameters
        ----------
        time: str, optional
            Column name. Default is "index", i.e. considers the index
        fill_startend : boolean, optional
            fill dataframe start and end (Nan values due to the derivation/centering method) (True by default).
        inplace : boolean, optional
            if True add dt to dataset, if False return only a dataframe with time, id (for identification) and computed velocities
        """
        df = compute_dt(
            self._obj,
            time,
            fill_startend=fill_startend,
            inplace=inplace,
        )

        if not inplace:
            return df

    def compute_accelerations(
        self,
        from_=(
            "velocities",
            "velocity_east",
            "velocity_north",
        ),
        names=None,
        centered_velocity=True,
        time="index",
        keep_dt=False,
        fill_startend=True,
        inplace=False,
    ):
        """compute acceleration from velocities or position
        Parameters
        ----------
        df : dataframe,
            dataframe containing trajectories
        from_ :  tuple of str, optional
            (key, east_name, north_name)
            if key = 'velocities', compute accelaration from velocities
            if key = 'lonlat', compute acceleration from lonlat time series
            if key = 'xy', compute acceleration from xy time series
        names :  tuple, optional
            Contains columns names for eastern, northen and norm acceleration
            ("acceleration_east", "acceleration_north", "acceleration") by default
        centered_velocities : boolean, optional
            True if the velocities is centered temporally (True by default)
        time: str, optional
            Column name. Default is "index", i.e. considers the index
        keep_dt: boolean
            Keeps time intervals (False by default).
        fill_startend : boolean
            fill dataframe start and end (Nan values due to the derivation/centering method) (True by default).
        inplace : boolean
            if True add acceleration to dataset, if False return only a dataframe with time, id (for identification) and computed acceleration
        """
        df = compute_accelerations(
            self._obj,
            from_,
            names,
            centered_velocity,
            time,
            keep_dt,
            fill_startend,
            inplace,
        )
        if not inplace:
            return df

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

    def plot_bokeh(
        self,
        deployments=None,
        rule=None,
        mindec=True,
        velocity=False,
        acceleration=False,
    ):
        """Plot time series: longitude, latitude, velocities, acceleration

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
            df = df.geo.compute_velocities()
            df = df.geo.compute_accelerations()

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
            width=350,
            height=300,
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
            _add_start_end(s4, -df["acceleration"], ymax=df["acceleration"])
            s4.add_tools(crosshair)
            S.append(s4)
        #
        p = gridplot(S, ncols=2)
        show(p)

    def plot_on_map(self, rule=None, coords="geo", **kwargs):
        """Produce map with trajectory on map
        Requires geoviews

        Parameters
        ----------
        rule: str, optional
            resampling rule
        coords: str, optional
            Controls coordinates:
                - "xy": x/y space
                - "geo": geographical coordinates (lon/lat)
        **kwargs: passed to hvplot
        """

        dkwargs = dict(hover_cols=["time"], width=600, height=400)
        if coords == "geo":
            coords = dict(x=self._lon, y=self._lat, geo=True)
            dkwargs["tiles"] = "CartoLight"
        elif coords == "xy":
            self.project()
            coords = dict(x="x", y="y")
        dkwargs = dict(**coords, **dkwargs)
        dkwargs.update(**kwargs)

        if rule is not None:
            df = self.resample(rule)
        else:
            df = self._obj

        return df.hvplot.points(**dkwargs), coords


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


def compute_accelerations(
    df,
    from_=(
        "velocities",
        "velocity_east",
        "velocity_north",
    ),
    names=None,
    centered_velocity=True,
    time="index",
    keep_dt=False,
    fill_startend=True,
    inplace=False,
):
    """compute acceleration from velocities or position
    Parameters
    ----------
    df : dataframe,
        dataframe containing trajectories
    from_ :  tuple of str,
        (key, east_name, north_name)
        if key = 'velocities', compute accelaration from velocities
        if key = 'lonlat', compute acceleration from lonlat time series
        if key = 'xy', compute acceleration from xy time series
        if key = 'xy_spectral' compute from velocities via spectral method
        if key = 'velocities_spectral' compute from velocities via spectral method
    names :  tuple, optional
        Contains columns names for eastern, northen and norm acceleration
        ("acceleration_east", "acceleration_north", "acceleration") by default
    centered_velocities : boolean
        True if the velocities is centered temporally (True by default)
    time: str, optional
        Column name. Default is "index", i.e. considers the index
    keep_dt: boolean
        Keeps time intervals (False by default).
    fill_startend : boolean
        fill dataframe start and end (Nan values due to the derivation/centering method) (True by default).
    inplace : boolean
        if True add acceleration to dataset, if False return only a dataframe with time, id (for identification) and computed acceleration
    """
    assert from_[1] in df.columns and from_[2] in df.columns, (
        from_[1] + " and/or " + from_[2] + " not in the dataframe, check names"
    )

    if names is None:
        names = ("acceleration_east", "acceleration_north", "acceleration")

    # drop duplicates
    if not inplace:
        df = df[~df.index.duplicated(keep="first")]
    else:
        if df.index.duplicated(keep="first").any():
            df.reset_index(inplace=True)
            df.drop_duplicates(subset="time", keep="first", inplace=True)
            df.set_index("time", inplace=True)
        # df.drop_duplicates(keep="first", inplace=True)# drop all nan lines, does not consider the index time -> pb

    if not inplace:
        df = df.copy()

    if not centered_velocity:
        warnings.warn(
            "Acceleration computation with non centered velocity", UserWarning
        )

    # dt
    if time == "index":
        t = df.index.to_series()
        assert t.is_monotonic_increasing, "dataframe must be sorted along time"
        dt = t.diff() / pd.Timedelta("1s")
        dt.index = df.index  # necessary?
        if "dt" in df.columns:
            npt.assert_allclose(
                dt[1:],  # ignores first point, which is ultimately padded below
                df["dt"][1:],
                atol=1e-2,
                err_msg="dt already in present and different than derived one",
            )
            keep_dt = True
        df.loc[:, "dt"] = dt
    else:
        assert df[time].is_monotonic_increasing, "dataframe must be sorted along time"
        t = df[time]
        dt = t.diff() / pd.Timedelta("1s")
        df.loc[:, "dt"] = dt
    # is_uniform = df["dt"].dropna().unique().size == 1

    # compute acc from velocities
    assert from_[0] in [
        "lonlat",
        "xy",
        "velocities",
        "xy_spectral",
        "velocities_spectral",
    ], "from_ should be 'lonlat', 'xy', 'velocities', 'xy_spectral', 'velocities_spectral'"

    if from_[0] == "velocities":
        if centered_velocity:
            w = dt / (dt + dt.shift(-1))
            ae = df.loc[:, from_[1]].diff() / dt
            an = df.loc[:, from_[2]].diff() / dt
            normv = np.sqrt(df.loc[:, from_[1]] ** 2 + df.loc[:, from_[2]] ** 2)
            a = normv.diff() / dt
            df.loc[:, names[0]] = ae + (ae.shift(-1) - ae) * w
            df.loc[:, names[1]] = an + (an.shift(-1) - an) * w
            df.loc[:, names[2]] = a + (a.shift(-1) - a) * w
        else:
            dt_acc = (dt.shift(-1) + dt) * 0.5
            normv = np.sqrt(df[from_[1]] ** 2 + df[from_[2]] ** 2)
            df.loc[:, names[0]] = (df[from_[1]].shift(-1) - df[from_[1]]) / dt_acc
            df.loc[:, names[1]] = (df[from_[2]].shift(-1) - df[from_[2]]) / dt_acc
            df.loc[:, names[2]] = (normv.shift(-1) - normv) / dt_acc

    # compute acc from positions in lonlat
    elif from_[0] == "lonlat":
        df_v = compute_velocities(
            df,
            from_[1],
            from_[2],
            time,
            None,
            False,
            False,
        )

        dt_acc = (dt.shift(-1) + dt) * 0.5
        normv = np.sqrt(df_v["velocity_east"] ** 2 + df_v["velocity_north"] ** 2)
        df.loc[:, names[0]] = (
            df_v["velocity_east"].shift(-1) - df_v["velocity_east"]
        ) / dt_acc
        df.loc[:, names[1]] = (
            df_v["velocity_north"].shift(-1) - df_v["velocity_north"]
        ) / dt_acc
        df.loc[:, names[2]] = (normv.shift(-1) - normv) / dt_acc

    # compute acc from positions in xy
    elif from_[0] == "xy":
        # leverage local projection, less accurate away from central point
        normX = np.sqrt(df[from_[1]] ** 2 + df[from_[2]] ** 2)
        dxdt = df[from_[1]].diff() / df["dt"]  # u_i = x_i - x_{i-1}
        dydt = df[from_[2]].diff() / df["dt"]  # v_i = y_i - y_{i-1}
        dXdt = normX.diff() / df["dt"]
        dt_acc = (dt.shift(-1) + dt) * 0.5
        df.loc[:, names[0]] = (dxdt.shift(-1) - dxdt) / dt_acc
        df.loc[:, names[1]] = (dydt.shift(-1) - dydt) / dt_acc
        df.loc[:, names[2]] = (dXdt.shift(-1) - dXdt) / dt_acc

    elif from_[0] == "xy_spectral":
        normX = np.sqrt(df[from_[1]] ** 2 + df[from_[2]] ** 2)
        df.loc[:, names[0]] = spectral_diff(df[from_[1]], df["dt"][1:], 2)
        df.loc[:, names[1]] = spectral_diff(df[from_[2]], df["dt"][1:], 2)
        df.loc[:, names[2]] = spectral_diff(normX, df["dt"][1:], 2)

    elif from_[0] == "velocities_spectral":
        normv = np.sqrt(df[from_[1]] ** 2 + df[from_[2]] ** 2)
        df.loc[:, names[0]] = spectral_diff(df[from_[1]], df["dt"][1:], 1)
        df.loc[:, names[1]] = spectral_diff(df[from_[2]], df["dt"][1:], 1)
        df.loc[:, names[2]] = spectral_diff(normv, df["dt"][1:], 2)
    else:
        assert (
            False
        ), "from_ should be 'lonlat', 'xy', 'velocities', 'xy_spectral', 'velocities_spectral'"

    # update acceleration norm (modified to be done for each method, including sign)
    # df.loc[:, names[2]] = np.sqrt(df[names[0]] ** 2 + df[names[1]] ** 2)

    if not keep_dt:
        del df["dt"]

    # fill end values
    if fill_startend:
        if inplace:
            df.bfill(inplace=True)
            df.ffill(inplace=True)
        else:
            df = df.bfill().ffill()

    if not inplace:
        return df


def compute_velocities(
    df,
    time,
    names,
    centered,
    fill_startend,
    distance,
    lon_key="lon",
    lat_key="lat",
    keep_dt=False,
    inplace=False,
):
    """core method to compute velocity from a dataframe
    Parameters
    ----------
    df : dataframe,
        dataframe containing trajectories
    lon_key: str
           longitude column name in dataframe
    lat_key: str
           latitude column name in dataframe
    time: str
        Column name corresponding to time.
        Can be "index", in which case the index is used
    names :  tuple
        Contains columns names for eastern, northen and norm velocities
        ("velocity_east", "velocity_north", "velocity" by default
    centered: boolean
        Centers velocity calculation temporally
    fill_startend : boolean
        fill dataframe start and end (Nan values due to the derivation/centering method) (True by default)
    distance: str
        Method to compute distances:
            - "geoid" is based geodetic distance and bearing ("WGS84" with pyproj)
            - "spectral" is a spectral estimation (requires uniform time sampling)
            - "xy" is from "x" and "y" columns (projected fields)
    keep_dt: boolean, optional
        Keeps time intervals (False by default).
    inplace : boolean, optional
        if True add velocities to dataset, if False return only a dataframe with time, id (for identification) and computed velocities.
    """
    if distance == "geoid":
        assert lon_key in df.columns and lat_key in df.columns, (
            lon_key + " and/or " + lat_key + " not in the dataframe, check names"
        )

    if names is None:
        names = ("velocity_east", "velocity_north", "velocity")

    # drop duplicates
    if not inplace:
        df = df[~df.index.duplicated(keep="first")]
    else:
        if df.index.duplicated(keep="first").any():
            df.reset_index(inplace=True)
            df.drop_duplicates(subset="time", keep="first", inplace=True)
            df.set_index("time", inplace=True)

    if not centered:
        warnings.warn("Velocity computation is not centered", UserWarning)

    # dt_i = t_i - t_{i-1}
    if time == "index":
        t = df.index.to_series()
        dt = t.diff() / pd.Timedelta("1s")
        dt.index = df.index  # necessary?
        df.loc[:, "dt"] = dt
    else:
        t = df[time]
        dt = t.diff() / pd.Timedelta("1s")
        df.loc[:, "dt"] = dt
    # is_uniform = df["dt"].dropna().unique().size == 1

    assert distance in [
        "geoid",
        "spectral",
        "xy",
    ], "distance must be one of 'geoid', 'spectral', 'xy'"
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
    elif distance == "spectral":
        assert (
            df.dt[1:] == df.dt[1]
        ).all(), "time must be regularly sampled to apply spectral method"
        dxdt = spectral_diff(df["x"], df["dt"][1:], 1)
        dydt = spectral_diff(df["y"], df["dt"][1:], 1)
        # skips first dt which is in general NaN
        centered = False
    elif distance == "xy":
        # leverage local projection, less accurate away from central point
        dxdt = df["x"].diff() / df["dt"]  # u_i = x_i - x_{i-1}
        dydt = df["y"].diff() / df["dt"]  # v_i = y_i - y_{i-1}

    if centered:
        w = dt / (dt + dt.shift(-1))
        df.loc[:, names[0]] = dxdt + (dxdt.shift(-1) - dxdt) * w
        df.loc[:, names[1]] = dydt + (dydt.shift(-1) - dydt) * w

        # boundaries, impose constant acceleration
        i0, i1 = df.index[[0, -1]]
        # print( dxdt[1] - dt[1] / dt[2] * (dxdt[2] - dxdt[1]), dxdt[1], dxdt[2], dt[1] / dt[2] )
        df.loc[i0, names[0]] = dxdt[1] - dt[1] / dt[2] * (dxdt[2] - dxdt[1])
        df.loc[i0, names[1]] = dydt[1] - dt[1] / dt[2] * (dydt[2] - dydt[1])
        df.loc[i1, names[0]] = dxdt[-2] + dt[-1] / dt[-2] * (dxdt[-2] - dxdt[-3])
        df.loc[i1, names[1]] = dydt[-2] + dt[-1] / dt[-2] * (dydt[-2] - dydt[-3])
    else:
        df.loc[:, names[0]] = dxdt
        df.loc[:, names[1]] = dydt
        # boundaries ?

    df.loc[:, names[2]] = np.sqrt(df.loc[:, names[0]] ** 2 + df.loc[:, names[1]] ** 2)

    if not keep_dt:
        del df["dt"]

    # fill end values
    if fill_startend:
        if inplace:
            df.bfill(inplace=True)
            df.ffill(inplace=True)
        else:
            df = df.bfill().ffill()

    if not inplace:
        return df


def compute_dt(
    df,
    time,
    fill_startend=True,
    inplace=False,
):
    """core method to compute dt from a dataframe
    Parameters
    ----------
    df : dataframe,
        dataframe containing trajectories
    time: str
        Column name corresponding to time.
        Can be "index", in which case the index is used
    fill_startend : boolean
        fill dataframe start and end (Nan values due to the derivation/centering method) (True by default)
    inplace : boolean, optional
        if True add dt to dataset, if False return only a dataframe with time, id (for identification) and computed dt.
    """

    # drop duplicates
    if not inplace:
        df = df[~df.index.duplicated(keep="first")]
    else:
        if df.index.duplicated(keep="first").any():
            df.reset_index(inplace=True)
            df.drop_duplicates(subset="time", keep="first", inplace=True)
            df.set_index("time", inplace=True)

    # dt_i = t_i - t_{i-1}
    if time == "index":
        t = df.index.to_series()
        dt = t.diff() / pd.Timedelta("1s")
        dt.index = df.index  # necessary?
        df.loc[:, "dt"] = dt
    else:
        t = df[time]
        dt = t.diff() / pd.Timedelta("1s")
        df.loc[:, "dt"] = dt
    # is_uniform = df["dt"].dropna().unique().size == 1

    # fill end values
    if fill_startend:
        if inplace:
            df.bfill(inplace=True)
            df.ffill(inplace=True)
        else:
            df = df.bfill().ffill()

    if not inplace:
        return df


def spectral_diff(x, dt, order, dx0=0.0, time=None):
    """Differentiate (order=1, 2) or integrate (order=-1) spectrally a pd.Series presumed uniform

    Parameters
    ----------
    x: pd.Series
        time series to differentiate/integrate
    dt: array-like
        time intervals used to estimate the time step and verify timeline is uniform
    order: int
        order of differentiation: 1, 2, -1 (integration)
    dx0: float
        initial values for integrations (order=-1)
    time: np.array
        array of datetimes
    """
    if time == None:
        time = x.index.values
    from scipy.fftpack import diff

    # from scipy.signal import detrend
    _dt = np.unique(dt)
    assert len(_dt) == 1, "timeseries need to be uniform for spectral differentiation"
    assert (
        not x.isnull().any()
    ), "position data must not contain NaNs for spectral differentation"
    # make signal periodic
    if order > 0:
        _t = (time - time[0]) / pd.Timedelta("1s")
        fit = np.polyfit(_t, x, 2)
        x = x - (fit[2] + fit[1] * _t + fit[0] * _t**2)
    # detrend
    # x_detrended = detrend(x)
    # x_trend = x - x_detrended
    # x = x_detrended
    npad = x.size // 2
    xp = np.pad(x, npad, mode="reflect")
    # apply diff
    dx = diff(xp, order=order, period=xp.size) / _dt**order
    # adjust output
    if order == -1:
        # trend needs to be added back to the integrated signal
        dx += np.cumsum(dx * 0 + xp.mean()) * _dt
    dx = dx[npad : npad + x.size]
    if dx0 is not None and order == -1:
        # initial value is adjusted
        dx = dx - dx[0] + dx0
    if order == 1:
        dx = dx + fit[1] + 2 * fit[0] * _t
    elif order == 2:
        dx = dx + fit[0]
    return pd.Series(dx, index=x.index)


# ----------------------------- xarray accessor --------------------------------


@xr.register_dataset_accessor("geo")
class XrGeoAccessor(GeoAccessor):
    """Xarray Dataset accessor in order to process geographical data"""

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

    def project(self, overwrite=True, **kwargs):
        """add (x,y) projection to object"""
        d = self._obj
        if "x" not in d.variables or "y" not in d.variables or overwrite:
            _x, _y = self.projection.lonlat2xy(d[self._lon], d[self._lat])
            dims = d[self._lon].dims
            d["x"], d["y"] = (dims, _x), (dims, _y)
        else:
            print("project silently fails")

    def compute_lonlat(self, x=None, y=None, **kwargs):
        """update longitude and latitude from projected coordinates"""
        d = self._obj
        if x is None and y is None:
            x, y = d["x"], d["y"]
        lon, lat = self.projection.xy2lonlat(x, y)
        return (x.dims, lon), (x.dims, lat)

    # time series related code

    # speed ...
