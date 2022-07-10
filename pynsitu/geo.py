import os

import numpy as np
import xarray as xr
import pandas as pd

import geopandas as gpd
from shapely.geometry import Polygon, Point
from shapely import wkt
from shapely.ops import transform
import pyproj
crs_wgs84 = pyproj.CRS("EPSG:4326")
import pyinterp
#import pyinterp.geohash as geohash

import matplotlib.pyplot as plt
from  matplotlib.dates import date2num, datetime
from matplotlib.colors import cnames
#
import cartopy.crs as ccrs
from cartopy.io import shapereader
import cartopy.feature as cfeature
#
from bokeh.io import output_notebook, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool, CustomJSHover, FuncTickFormatter
from bokeh.plotting import figure


# ------------------------------ parameters ------------------------------------

g = 9.81
deg2rad = np.pi / 180.0
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


# ----------------------------- formatters  -----------------------------------


def dec2degmin(dec):
    # return coordinates with seconds
    sign = np.sign(dec)
    deg = np.trunc(abs(dec))
    sec = (abs(dec) - deg) * 60.
    return [int(sign*deg), sec]

def degmin2dec(deg, min):
    """ converts lon or lat in deg, min to decimal
    """
    return deg + np.sign(deg) * min/60.

def print_degmin(l):
    ''' Print lon/lat, deg + minutes decimales
    '''
    dm = dec2degmin(l)
    #return '%d deg %.5f' %(int(l), (l-int(l))*60.)
    return '{} {:.5f}'.format(*dm)

## bokeh formatters

lon_hover_formatter = CustomJSHover(code="""
    var D = value;
    var deg = Math.abs(Math.trunc(D));
    if (D<0){
        var dir = "W";
    } else {
        var dir = "E";
    }
    var min = (Math.abs(D)-deg)*60.0;
    return deg + dir + " " + min.toFixed(3)
""")

lat_hover_formatter = CustomJSHover(code="""
    var D = value;
    var deg = Math.abs(Math.trunc(D));
    if (D<0){
        var dir = "S";
    } else {
        var dir = "N";
    }
    var min = (Math.abs(D)-deg)*60.0;
    return deg + dir + " " + min.toFixed(3)
""")


# ----------------------------- plotting methods -------------------------------


def plot_map(fig=None,
             region=None,
             coast='110m',
             land='110m',
             rivers='110m',
             figsize=(10, 10),
             bounds=None,
             cp=None,
             grid_linewidth=1,
             **kwargs,
             ):
    """ Plot a map of the campaign area

    Parameters
    ----------
    fig: matplotlib.figure.Figure, optional
        Figure handle, create one if not passed
    coast: str, optional
        Determines which coast dataset to use, e.g. ['10m', '50m', '110m']
    bounds: list, optional
        Geographical bounds, e.g. [lon_min, lon_max, lat_min, lat_max]
    cp: cognac.utils.campaign, optional
        Campaign object
    grid_linewidth: float, optional
        geographical grid line width
    """
    crs = ccrs.PlateCarree()
    #
    if fig is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig.clf()

    if bounds is None:
        if cp is not None:
            bounds = cp.bounds
        else:
            bounds = _bounds_default

    ax = fig.add_subplot(111, projection=crs)
    ax.set_extent(bounds, crs=crs)
    gl = ax.gridlines(crs=crs,
                      draw_labels=True,
                      linewidth=grid_linewidth,
                      color='k',
                      alpha=0.5, linestyle='--',
                      )
    gl.xlabels_top = False
    gl.ylabels_right = False

    # overrides kwargs for regions
    if region:
        coast = region
        land = region
        rivers = region

    #
    _coast_root = os.getenv('HOME')+'/data/coastlines/'
    coast_kwargs = dict(edgecolor='black', facecolor=cfeature.COLORS['land'], zorder=-1)
    if coast in ['10m', '50m', '110m']:
        ax.coastlines(resolution=coast, color='k')
    elif coast in ['auto', 'coarse', 'low', 'intermediate', 'high', 'full']:
        shpfile = shapereader.gshhs('h')
        shp = shapereader.Reader(shpfile)
        ax.add_geometries(
            shp.geometries(), crs, **coast_kwargs)
    elif coast=='bseine':
        # for production, see: /Users/aponte/Data/coastlines/log
        shp = shapereader.Reader(_coast_root+'baie_de_seine/bseine.shp')
        for record, geometry in zip(shp.records(), shp.geometries()):
            ax.add_geometries([geometry], crs, **coast_kwargs)
    elif coast=='med':
        # for production, see: /Users/aponte/Data/coastlines/log
        shp = shapereader.Reader(_coast_root+'med/med_coast.shp')
        for record, geometry in zip(shp.records(), shp.geometries()):
            ax.add_geometries([geometry], crs, **coast_kwargs)
    elif coast=='med_high':
        # for production, see: /Users/aponte/Data/coastlines/log
        shp = shapereader.Reader(_coast_root+'/med/med_high_coast.shp')
        for record, geometry in zip(shp.records(), shp.geometries()):
            ax.add_geometries([geometry], crs, **coast_kwargs)

    #
    _land_kwargs = dict(edgecolor='face', facecolor=cfeature.COLORS['land'])
    if land in ['10m', '50m', '110m']:
        land = cfeature.NaturalEarthFeature('physical', 'land', land, **_land_kwargs)
        #ax.add_feature(cfeature.LAND)
        ax.add_feature(land)
    elif land=='bseine':
        shp = shapereader.Reader(_coast_root+'baie_de_seine/bseine_land.shp')
        for record, geometry in zip(shp.records(), shp.geometries()):
            ax.add_geometries([geometry], crs, **_land_kwargs)

    #
    _rivers_kwargs = dict(facecolor='none', edgecolor='blue') #, zorder=-1
    if rivers in ['10m', '50m', '110m']:
        rivers = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines',
                                              rivers, **_rivers_kwargs)
        ax.add_feature(cfeature.RIVERS)
    elif rivers=='bseine':
        shp = shapereader.Reader(_coast_root+'baie_de_seine/bseine_rivers.shp')
        for record, geometry in zip(shp.records(), shp.geometries()):
            ax.add_geometries([geometry], crs, **_rivers_kwargs)
        shp = shapereader.Reader(_coast_root+'baie_de_seine/bseine_water.shp')
        for record, geometry in zip(shp.records(), shp.geometries()):
            ax.add_geometries([geometry], crs, facecolor='blue', edgecolor='none')

    return [fig, ax, crs]

# etopo1
_bathy_etopo1 = os.path.join(os.getenv('HOME'),
                        'Data/bathy/etopo1/zarr/ETOPO1_Ice_g_gmt4.zarr',
                        )

# med: GEBCO bathymetry
_med_file = 'gebco_2020_n44.001617431640625_s41.867523193359375_w4.61151123046875_e8.206787109375.nc'
_bathy_med = os.path.join(os.getenv('HOME'), 'Data/bathy/gebco1', _med_file)

def load_bathy(bathy, bounds=None, steps=None, **kwargs):
    """ Load bathymetry
    """
    if bathy=='etopo1':
        ds = xr.open_dataset(_bathy_etopo1)
        #ds = ds.rename({'x': 'lon', 'y': 'lat', 'z': 'elevation'})
        ds = ds.rename({'z': 'elevation'})
        if bounds is None and steps is None:
            steps = (4, 4)
    elif bathy=='med':
        ds = xr.open_dataset(_bathy_med)
    if steps is not None:
        ds = ds.isel(lon=slice(0, None, steps[0]),
                     lat=slice(0, None, steps[1]),
                     )
    if bounds is not None:
        ds = ds.sel(lon=slice(bounds[0], bounds[1]),
                    lat=slice(bounds[2], bounds[3]),
                    )
    return ds

def plot_bathy(fac,
               levels=[-6000., -4000., -2000., -1000., -500., -200., -100.],
               clabel=True,
               bathy='etopo1',
               steps=None,
               bounds=None,
               **kwargs,
               ):
    fig, ax, crs = fac
    if isinstance(levels, tuple):
        levels = np.arange(*levels)
    #print(levels)
    ds = load_bathy(bathy, bounds=bounds, steps=steps)
    cs = ax.contour(ds.lon, ds.lat, ds.elevation, levels,
                    linestyles='-', colors='black',
                    linewidths=0.5,
                    )
    if clabel:
        plt.clabel(cs, cs.levels, inline=True, fmt='%.0f', fontsize=9)

# ----------------------------- pandas geo extension --------------------------

def _xy2lonlat(x, y, proj):
    """ compute longitude/latitude from projected coordinates """
    _inv_dir = pyproj.enums.TransformDirection.INVERSE
    return proj.transform(x, y, direction=_inv_dir)

@pd.api.extensions.register_dataframe_accessor("geo")
class GeoAccessor:
    def __init__(self, pandas_obj):
        self._lon, self._lat = self._validate(pandas_obj)
        self._obj = pandas_obj
        self._reset_geo()

    #@staticmethod
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
            raise AttributeError("Did not find latitude and longitude columns. Case insentive options are: "
                                 + "/".join(lat_potential) + " , " + "/".join(lon_potential)
                                )
        else:
            return lon, lat

    def _reset_geo(self):
        """reset all variables related to geo"""
        self._geo_proj_ref=None
        self._geo_proj=None
        #self._obj.drop(columns=["x", "y"], errors="ignore", inplace=True)

    @property
    def projection_reference(self):
        """ define a reference projection if none is available """
        if self._geo_proj_ref is None:
            # return the geographic center point of this DataFrame
            lat, lon = self._obj[self._lat], self._obj[self._lon]
            self._geo_proj_ref = (float(lon.mean()), float(lat.mean()))
        return self._geo_proj_ref

    def set_projection_reference(self, ref, reset=True):
        """ set projection reference point, (lon, lat) tuple"""
        if reset:
            self._reset_geo()
        self._geo_proj_ref = ref

    @property
    def projection(self):
        if self._geo_proj is None:
            lonc, latc = self.projection_reference
            self._geo_proj = pyproj.Proj(proj="aeqd", lat_0=latc, lon_0=lonc,
                                         datum="WGS84", units="m")
        return self._geo_proj

    def project(self, overwrite=False):
        """add (x,y) projection to object"""
        d = self._obj
        if "x" not in d.columns or "y" not in d.columns or overwrite:
            d["x"], d["y"] = self.projection.transform(d[self._lon], d[self._lat])

    def compute_lonlat(self):
        """update longitude and latitude from projected coordinates """
        d = self._obj
        assert ("x" in d.columns) and ("y" in d.columns), "x/y coordinates must be available"
        d[self._lon], d[self._lat] = _xy2lonlat(d.x, d.y, self.projection)

    def trim(self, d):
        """given a deployment item, trim data"""
        time = self._obj.index
        df = self._obj.loc[(time >= d.start.time) & (time <= d.end.time)]
        return df

    def apply_xy(self, fun):
        """ apply a function that requires working with projected coordinates x/y"""
        # ensures projection exists
        self.project()
        # apply function
        df = fun(self._obj)
        # update lon/lat
        df[self._lon], df[self._lat] = _xy2lonlat(df.x, df.y, self.projection)
        return df

    def resample(self,
                 rule,
                 interpolate=False,
                 #inplace=True,
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
        def _resample(df):
            df = df.resample(rule, **kwargs).mean()
            if interpolate:
                df = df.interpolate(method='linear')
            return df
        return self.apply_xy(_resample)

    def compute_velocities(self, time="index"):
        """ compute velocity """
        def _compute_velocities(df):
            df = df[~df.index.duplicated(keep='first')]
            if time=="index":
                dt = pd.Series(df.index).diff()/pd.Timedelta("1s")
                dt.index = df.index
                df["dt"] = dt
            else:
                df["dt"] = df[time].diff()/pd.Timedelta("1s")
            df["u"] = df.x.diff()/df.dt
            df["v"] = df.y.diff()/df.dt
            df["velocity"] = np.sqrt(df.u**2+df.v**2)
            df = df.drop(columns=["dt"])
            return df
        return self.apply_xy(_compute_velocities)

    def plot_lonlat(self):
        """ simple lon/lat plot """
        # plot this array's data on a map, e.g., using Cartopy
        df = self._obj
        fig, ax = plt.subplots(1,1)
        ax.plot(df[self._lon], df[self._lat])

    def plot_bokeh(self, unit=None, rule=None, mindec=True):
        """ bokeh plot

        Parameters
        ----------
        unit:
        rule:
        mindec: boolean
            Plot longitude and latitude as minute/decimals
        """

        if rule is not None:
            df = self.resample(rule)
        else:
            df = self._obj

        if 'velocity' not in df.columns:
            df = df.geo.compute_velocities()

        if mindec:
            _lon_tooltip = '@'+self._lon+'{custom}'
            _lat_tooltip = '@'+self._lat+'{custom}'
            _lon_formatter = lon_hover_formatter
            _lat_formatter = lat_hover_formatter
            #ll_formater = FuncTickFormatter(code="""
            #    return Math.floor(tick) + " + " + (tick % 1).toFixed(2)
            #""")
        else:
            _lon_tooltip = '@{'+self._lon+'}{0.4f}'
            _lat_tooltip = '@{'+self._lat+'}{0.4f}'
            _lon_formatter = 'printf'
            _lat_formatter = 'printf'

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

        # create a new plot and add a renderer
        s1 = figure(tools=TOOLS,
                    plot_width=300, plot_height=300,
                    title='longitude',
                    x_axis_type='datetime',
                    )
        s1.line('time', self._lon, source=df, line_width=lw, color=c)
        s1.add_tools(HoverTool(
            tooltips=[('Time','@time{%F %T}'),('longitude', _lon_tooltip),], #
            formatters={'@time': 'datetime','@'+self._lon: _lon_formatter,}, #'printf'
            mode='vline'
            ))
        _add_start_end(s1, df[self._lon])
        s1.add_tools(crosshair)
        #
        s2 = figure(tools=TOOLS,
                    plot_width=300, plot_height=300,
                    title='latitude',
                    x_axis_type='datetime',
                    x_range=s1.x_range
                    )
        s2.line('time', self._lat, source=df, line_width=lw, color=c)
        s2.add_tools(HoverTool(
            tooltips=[('Time','@time{%F %T}'),('latitude',_lat_tooltip),],
            formatters={'@time': 'datetime','@'+self._lat : _lat_formatter,},
            mode='vline'
            ))
        _add_start_end(s2, df[self._lat])
        s2.add_tools(crosshair)
        #
        s3 = figure(tools=TOOLS,
                    plot_width=300, plot_height=300,
                    title='speed',
                    x_axis_type='datetime',
                    x_range=s1.x_range
                    )
        s3.line('time', 'velocity', source=df, line_width=lw, color=c)
        s3.add_tools(HoverTool(
            tooltips=[('Time','@time{%F %T}'),('Velocity','@{velocity}{0.2f} m/s'),],
            formatters={'@time': 'datetime','@velocity' : 'printf',},
            mode='vline'
            ))
        _add_start_end(s3, df['velocity'])
        s3.add_tools(crosshair)

        p = gridplot([[s1, s2, s3]])
        show(p)

    def plot_bokeh_map(self, unit=None, rule=None, mindec=True):
        """ bokeh plot"""

        if rule is not None:
            df = self.resample(rule)
        else:
            df = self._obj
        # ensure we have projection
        self.project()

        if mindec:
            _lon_tooltip = '@'+self._lon+'{custom}'
            _lat_tooltip = '@'+self._lat+'{custom}'
            _lon_formatter = lon_hover_formatter
            _lat_formatter = lat_hover_formatter
            #ll_formater = FuncTickFormatter(code="""
            #    return Math.floor(tick) + " + " + (tick % 1).toFixed(2)
            #""")
        else:
            _lon_tooltip = '@{'+self._lon+'}{0.4f}'
            _lat_tooltip = '@{'+self._lat+'}{0.4f}'
            _lon_formatter = 'printf'
            _lat_formatter = 'printf'

        output_notebook()
        TOOLS = 'pan,wheel_zoom,box_zoom,reset,help'

        # line specs
        lw = 5
        c = 'black'

        # create a new plot and add a renderer
        s1 = figure(tools=TOOLS,
                    plot_width=600, plot_height=600,
                    title='map',
                    match_aspect=True, # if projected for equal axis
                    #x_axis_type='datetime',
                   )
        s1.line('x', 'y', source=df, line_width=lw, color=c)
        s1.add_tools(HoverTool(
            tooltips=[('Time','@time{%F %T}'),
                      ('longitude', _lon_tooltip),
                      ('latitude', _lat_tooltip)],
            formatters={'@time': 'datetime',
                        '@'+self._lon : _lon_formatter,
                        '@'+self._lat : _lat_formatter,
                        },
            #mode='vline'
            ))

        p = gridplot([[s1,]])
        show(p)
