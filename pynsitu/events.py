#
# ------------------------- Event/Deployment objects -----------------------------------
#
import os
from glob import glob
import yaml

import numpy as np
import pandas as pd
import xarray as xr
import math

import matplotlib.pyplot as plt
from  matplotlib.dates import date2num, datetime
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.colors import cnames

import folium
from folium.plugins import MeasureControl, MousePosition

from .geo import plot_map, plot_bathy, load_bathy_contours
#from .utils import dec2degmin, \
#                plot_map, plot_bathy, \
#                load_bathy_contours, store_bathy_contours

#_bounds_default = [6.4, 6.6, 42.92, 43.2]
#_bounds_default = [6., 6.6, 42.7, 43.2]

class event(object):

    def __init__(self, label=None, logline=None, coord_min=None):

        # split string
        l=logline.split()

        # label
        self.label = label

        # time information
        # assumes: 02/09/2016 05:35:00 7 17.124 43 19.866
        self.time = pd.to_datetime(l[0]+' '+l[1],
                                   #dayfirst=False,
                                   infer_datetime_format=True,
                                   )

        # lon, lat data
        if len(l)==6 or coord_min==True:
            # degrees + minute decimals
            lon_deg = float(l[2])
            self.lon = lon_deg + math.copysign(1,lon_deg) * float(l[3])/60.
            lat_deg = float(l[4])
            self.lat = lat_deg + math.copysign(1,lat_deg) * float(l[5])/60.
            # -0. is allowed but np.sign does not recognize it, hence the call to math.copysign
        elif len(l)==4 or coord_min==False:
            # degrees decimal
            self.lon = float(l[2])
            self.lat = float(l[3])
        else:
            self.lon = None
            self.lat = None

    def degmin(self):
        """ Returns dict containing longitude and latitude in degrees/minutes
        """
        return dict(lon=dec2degmin(lon), lat=dec2degmin(lat))

    def __str__(self):
        if self.lon and self.lat:
            return '{} {} {:.2f} {:.2f}'.format(self.label,
                                                self.time,
                                                self.lon,
                                                self.lat,
                                                )
        else:
            return '{} {}'.format(self.label, self.time)


class deployment(object):

    def __init__(self, label=None, start=None, end=None, loglines=None):
        self.label=label
        if loglines is None:
            self.start=start
            self.end=end
        else:
            self.start=event(label='start', logline=loglines[0])
            self.end = event(label='end', logline=loglines[1])
            if len(loglines)==3:
                self.info = loglines[2]["info"]
            else:
                self.info = dict()

    def __repr__(self):
        return 'cognac.insitu.events.deployment({})'.format(str(self))

    def __str__(self):
        return self.label+' / '+str(self.start)+' / '+str(self.end)

    def plot_time(self, ax, y0=0., dy=0.5, **kwargs):
        #t0 = self.start.time
        #t1 = self.end.time
        #rect = Rectangle((t0, y0-dy/2.), t1-t0, dy, **kwargs)
        #ax.add_patch(rect)
        #ax.plot([t0,t1],[y0,y0],lw=20)
        #ax.plot([self.start.time,self.end.time],[y0,y0],lw=20)
        pass

    def plot_on_map(self,
                    ax,
                    line=True,
                    label=False,
                    yshift=1,
                    s=30,
                    **kwargs,
                    ):
        if self.start.lon is None:
            # exits right for deployments that do not have lon/lat info
            return
        #
        x0, y0 = self.start.lon, self.start.lat
        x1, y1 = self.end.lon, self.end.lat
        #
        ax.scatter(x0, y0, s, marker='o', **kwargs)
        ax.scatter(x1, y1, s, marker='*', **kwargs)
        #
        if line:
            ax.plot([x0, x1], [y0, y1], '-', **kwargs)
        if label:
            xb, yb = ax.get_xbound(), ax.get_ybound()
            xoffset = 0.02 * (xb[1]-xb[0])
            yoffset = 0.02 * (yb[1]-yb[0]) * yshift
            if type(label) is not str:
                label = self.label
            ax.text(x0+xoffset, y0+yoffset, label, fontsize=10)
        return

class objdict(object):
    ''' Dict like object that treats some parameters (e.g. path and color)
    as attributes
    '''
    def __init__(self, *args,**kwargs):
        self._dict = dict(*args,**kwargs)
        self._skip = ['path', 'label']

    def __contains__(self, item):
        return item in self._dict

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __iter__(self):
        for key, value in self._dict.items():
            if key not in self._skip:
                yield value

    def __repr__(self):
        return 'cognac.insitu.events.unit({})'.format(str(self))

    def __str__(self):
        return self['label']+'\n'+'\n'.join(str(d) for d in self)


class campaign(object):
    ''' Campaign object, gathers deployments information
    '''

    def __init__(self, file):

        # open yaml information file
        if ".yaml" not in file:
            file = file+".yaml"
        with open(file, 'r') as stream:
            cp = yaml.full_load(stream)

        default_attr = {'name': 'unknown',
                        'lon': None, 'lat': None,
                        'start': None, 'end': None,
                        'path': None,
                        'bathy': None,
                        'raw': '',
                        }
        for key, value in default_attr.items():
            if key in cp:
                setattr(self, key, cp[key])
            else:
                setattr(self, key, value)

        if self.lon and self.lat:
            # ensure coords are floats
            self.lon = tuple(float(l) for l in self.lon)
            self.lat = tuple(float(l) for l in self.lat)
            #
            self.bounds = self.lon + self.lat
            self.lon_mid = (self.lon[0]+self.lon[1])*.5
            self.lat_mid = (self.lat[0]+self.lat[1])*.5

        if self.start:
            self.start = pd.Timestamp(self.start)
        if self.end:
            self.end = pd.Timestamp(self.end)

        # path to raw data
        self.pathr = os.path.join(self.path, self.raw)

        # path to processed data
        if 'pathp' in cp:
            self.pathp = cp['pathp']
        else:
            self.pathp = os.path.join(self.path, 'datap')

        self._units = {}
        for u, info in cp['units'].items():
            print(u)
            self._units[u] = objdict(path=self.path, label=u)
            for d, value in info['deployments'].items():
                self._units[u][d] = deployment(label=d, loglines=value)
            for d, value in info.items():
                if d=='path':
                    if isinstance(value, str):
                        _p = os.path.join(self.path, value)
                    else:
                        _p = [os.path.join(self.path, v) for v in value]
                    self._units[u]['path'] = _p
                elif d!='deployments':
                    self._units[u][d] = value
                    self._units[u]._skip.append(d)

    def __repr__(self):
        return 'cognac.insitu.events.campaign({})'.format(str(self))

    def __str__(self):
        #fmt = "%Y-%m-%d %H:%M:%S"
        fmt = "%Y/%m/%d"
        start = self.start.strftime(fmt)
        end = self.end.strftime(fmt)
        return self.name+' {} to {}'.format(start, end)

    def __getitem__(self, item):
        if item in self._units:
            return self._units[item]
        else:
            return None

    def __iter__(self):
        #for key, value in self._units.items():
        #    yield value
        for key in self._units:
            yield key

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name==other
        else:
            return False

    def items(self):
        for key, value in self._units.items():
            yield key, value

    def plot_map(self, **kwargs):
        """ Plot map
        Wrapper around utils.plot_map, see related doc
        """
        dkwargs = dict(bounds=self.bounds,
                       #bathy=self.bathy['label'],
                       levels=self.bathy['levels'],
                       )
        #dkwargs.update((k,v) for k,v in kwargs.items() if v is not None)
        dkwargs.update(kwargs)
        fac = plot_map(cp=self, **dkwargs)
        plot_bathy(fac, bathy=self.bathy['label'], **dkwargs)
        return fac

    def map(self,
            width='70%',
            height='70%',
            tiles='Cartodb Positron',
            ignore=[],
            overwrite_contours=False,
            zoom=10,
            **kwargs,
            ):
        ''' Plot overview map with folium

        Parameters:
        ----------

        tiles: str
            tiles used, see `folium.Map?``
                - "OpenStreetMap"
                - "Mapbox Bright" (Limited levels of zoom for free tiles)
                - "Mapbox Control Room" (Limited levels of zoom for free tiles)
                - "Stamen" (Terrain, Toner, and Watercolor)
                - "Cloudmade" (Must pass API key)
                - "Mapbox" (Must pass API key)
                - "CartoDB" (positron and dark_matter)

        '''

        if ignore=='all':
            ignore=self._units

        m = folium.Map(location=[self.lat_mid, self.lon_mid],
                       width=width,
                       height=height,
                       zoom_start=zoom,
                       tiles=tiles,
                      )

        # bathymetric contours
        contour_file = os.path.join(self.pathp,'bathy_contours.geojson')
        if (not os.path.isfile(contour_file) or
            (os.path.isfile(contour_file) and overwrite_contours)
            ):
            store_bathy_contours(self.bathy['label'],
                                 contour_file=contour_file,
                                 levels=self.bathy['levels'],
                                 bounds=self.bounds,
                                 )
        contours_geojson = load_bathy_contours(contour_file)

        tooltip = folium.GeoJsonTooltip(fields=['title'],
                                        aliases=['depth'],
                                        )
        popup = folium.GeoJsonPopup(fields=['title'],
                                    aliases=['depth'],
                                    )
        #colorscale = branca.colormap.linear.Greys_03.scale(levels[-1],levels[0])
        def style_func(feature):
            return {'color':   feature['properties']['stroke'], #colorscale(feature['properties']['level-value']),
                    'weight':  3, #x['properties']['stroke-width'],
                    #'fillColor': x['properties']['fill'],
                    'opacity': 1.,
                    #'popup': feature['properties']['title'],
                   }
        folium.GeoJson(contours_geojson,
                       name='geojson',
                       style_function=style_func,
                       tooltip=tooltip,
                       popup=popup,
                       ).add_to(m)

        # campaign details
        for uname, u in self.items():
            if uname not in ignore:
                for d in u:
                    if d.start.lat is None:
                        continue
                    folium.Polygon([(d.start.lat, d.start.lon),
                                    (d.end.lat, d.end.lon)
                                    ],
                                   tooltip=uname+' '+d.label+'<br>'
                                            +str(d.start.time)+'<br>'
                                            +str(d.end.time),
                                   color=cnames[u['color']],
                                   dash_array='10 20',
                                   opacity=.5
                                  ).add_to(m)
                    folium.Circle((d.start.lat, d.start.lon),
                                  tooltip=uname+' '+d.label+'<br>'
                                            +str(d.start.time),
                                  radius=2*1e2,
                                  color=cnames[u['color']],
                                 ).add_to(m)
                    folium.Circle((d.end.lat, d.end.lon),
                                  tooltip=uname+' '+d.label+'<br>'
                                            +str(d.end.time),
                                  radius=1e2,
                                  color=cnames[u['color']],
                                 ).add_to(m)

        # useful plugins

        MeasureControl().add_to(m)

        fmtr_lon = "function(dec) {var min= (dec-Math.round(dec))*60; " \
                    +"direction = (dec < 0) ? 'W' : 'E'; " \
                    +"return L.Util.formatNum(dec, 0) + direction + L.Util.formatNum(min, 2);};"
        fmtr_lat = "function(dec) {var min= (dec-Math.round(dec))*60; " \
                    +"direction = (dec < 0) ? 'S' : 'N'; " \
                    +"return L.Util.formatNum(dec, 0) + direction + L.Util.formatNum(min, 2);};"
        MousePosition(lat_formatter=fmtr_lon, lng_formatter=fmtr_lat).add_to(m)

        return m

    def timeline(self,
                 height=.3,
                 legend=3,
                 skip_ship=True,
                 start_scale=1,
                 ):
        """ Plot the campaign deployment timeline
        """

        fig = plt.figure(figsize=(15,5))
        ax=fig.add_subplot(111)

        y=0
        yticks, yticks_labels = [], []
        starts, ends = [], []
        for uname, u in self.items():
            if skip_ship and uname=="ship":
                continue
            for d in u:
                start = mdates.date2num(d.start.time)
                end = mdates.date2num(d.end.time)
                rect = Rectangle((start, y-height/2.), end-start, height,
                                 color=u['color'])
                ax.add_patch(rect)
                starts.append(start)
                ends.append(end)
            yticks.append(y)
            yticks_labels.append(uname)
            y+=-1

        ax.set_title(self.name)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks_labels)
        if legend:
            self.add_legend(ax, loc=legend, skip_ship=skip_ship)

        # assign date locator / formatter to the x-axis to get proper labels
        locator = mdates.AutoDateLocator(minticks=3)
        formatter = mdates.AutoDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        # set the limits
        delta_time = max(ends) - min(starts)
        plt.xlim([min(starts)-delta_time*.05*start_scale, max(ends)+delta_time*.05])
        plt.ylim([y+1-2*height,2*height])

    def add_legend(self, ax,
                   labels=None,
                   skip_ship=True,
                   colors=None,
                   **kwargs,
                   ):
        """ Add legend for units on an axis,
        Used for timelines as well as maps

        Parameters
        ----------
        ax: pyplot.axes
        labels: list, optional
            List of labels to consider amongst cp units
        skip_ship: boolean, optional
        colors: dict, optional
        **kwargs: passed to legend
        """
        from matplotlib.lines import Line2D
        if labels is None:
            labels = list(self._units)
        if skip_ship:
            labels = [l for l in labels if l!="ship"]
        custom_lines = []
        for label in labels:
            if colors and label in colors:
                c = colors[label]
            else:
                c = self[label]['color']
            custom_lines.append(Line2D([0], [0], color=c, lw=4))
        ax.legend(custom_lines, labels, **kwargs)

    def timeline_old(self):
        """ older version of timeline
        """
        fig = plt.figure(figsize=(15,5))
        ax=fig.add_subplot(111)

        y=0
        yticks, yticks_labels = [], []
        for uname, u in self.items():
            for d in u:
                ax.plot([d.start.time,d.end.time],[y,y], lw=4, color=u['color'])
            yticks.append(y)
            yticks_labels.append(uname)
            y+=1

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks_labels)
        self.add_legend(ax, loc=2)

    def load(self, item, toframe=False):
        """ load processed data files
        recall item

        Returns
        -------
        output: dict
            {'unit0': {'deployment0': data, ...}}
        """

        # straight netcdf file
        if ".nc" in item:
            file = os.path.join(self.pathp, item)
            ds = xr.open_dataset(file)
            if toframe:
                ds = ds.to_dataframe()
            return ds

        _files = self._get_unit_files(item)
        D = {}
        for d, f in _files.items():
            ds = xr.open_dataset(f)
            if toframe:
                ds = ds.to_dataframe()
            D[d] = ds
        if len(D)==1:
            return D[0]
        else:
            return D

        # !!! code below can probably be deleted

        # particular units
        if item=='ship':
            ship_file = os.path.join(self.pathp, 'ship.nc')
            if os.path.isfile(ship_file):
                return xr.open_dataset(ship_file).to_dataframe()
            else:
                return
        elif item=="argo":
            argo_file = os.path.join(self.pathp, 'argo.nc')
            if os.path.isfile(argo_file):
                return xr.open_dataset(argo_file).compute()
            else:
                return

        _files = [f.split('/')[-1] for f in data_files]

        units = set(f.split('_')[0] for f in _files)
        if unit:
            if isinstance(unit,str):
                units = [unit]
            else:
                units = [u for u in units if u in unit]

        data = {}
        for u in units:
            _files = [f.split('/')[-1]
                      for f in self._get_processed_files(unit=u,
                                               item=item,
                                               )
                      ]
            deployment_index=2 if item is not None else 1
            deployments = [f.split('_')[deployment_index].split('.')[0] for f in _files]
            data[u] = {d: _load_processed_file(
                self._get_processed_files(unit=u,
                                          item=item,
                                          deployment=d,
                                          )
                )
                       for d in deployments
                       }
        return data

    def _get_unit_files(self, unit, extension="nc"):
        """get all processed files associated with one unit"""
        files = sorted(glob(os.path.join(self.pathp, unit+'*.'+extension)))
        # find out whether there are multiple deployments
        if len(files)==1:
            return dict(d0=files[0])
        D = {}
        for f in files:
            s = f.split("/")[-1].split(".")[0].replace(unit,"").split("_")
            assert len(s)==2, f"there must be 0 or 1 underscore, but we have: {f}, {s}"
            D[s[1]] = f
        return D

    def _get_processed_files(self,
                   unit='*',
                   item='*',
                   deployment='*',
                   extension='nc',
                   ):
        """ Return processed data files

        Parameters
        ----------
        unit, item, d, extention: str, optional
            Defaults: '*', '*', '*', 'nc'
            Typical file path: self.pathp+unit+'_'+item+'_'+deployment+'.'+extension

        """
        if item is None:
            _item = ""
        else:
            _item = item+'_'
        if any([_=='*' for _ in [unit, item, deployment]]):
            return glob(os.path.join(self.pathp, unit+'_'+_item+deployment+'.'+extension))
        else:
            return os.path.join(self.pathp, unit+'_'+_item+deployment+'.'+extension)

def _load_processed_file(file, **kwargs):
    """ load preprocessed file, select object type based on filename
    """
    if "_gps_" in file or "_iridium" in file:
        from .gps import gps
        return gps(file=file)
    elif "emission" in file:
        from .source import source_rtsys
        return source_rtsys(file=file)
    elif "ctd" in file:
        from .ctd import ctd
        return ctd(file=file)
    else:
        return file+" not loaded"
