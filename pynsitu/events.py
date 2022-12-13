#
# ------------------------- Event/Deployment objects -----------------------------------
#
import os
from glob import glob
from collections import UserDict
import re

import numpy as np
import pandas as pd
import xarray as xr
import math

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.colors import cnames

from .maps import plot_map, plot_bathy, load_bathy_contours, store_bathy_contours


class Event(object):
    """An event is an atom used to describe deployments.
    It contains four elementary information:
            label, longitude, latitude, time
    """

    def __init__(self, label=None, logline=None):
        """Instantiate event object

        Parameters
        ----------
        label: str
            Event label
        logline: str
            Log line specifying relevant information. Here are accepted formats:
                - "02/09/2016 05:35:00 7 17.124 43 19.866"
                - "02/09/2016 05:35:00 7.124 43.866"
                - "02/09/2016 05:35:00"
        """

        # label
        self.label = label

        # split string
        l = logline.split()

        # time information
        self.time = pd.to_datetime(
            l[0] + " " + l[1],
            infer_datetime_format=True,
        )

        # lon, lat data
        if len(l) == 6:
            # degrees + minute decimals
            lon_deg = float(l[2])
            self.lon = lon_deg + math.copysign(1, lon_deg) * float(l[3]) / 60.0
            lat_deg = float(l[4])
            self.lat = lat_deg + math.copysign(1, lat_deg) * float(l[5]) / 60.0
            # -0. is allowed but np.sign does not recognize it, hence the call to math.copysign
        elif len(l) == 4:
            # degrees decimal
            self.lon = float(l[2])
            self.lat = float(l[3])
        else:
            self.lon = None
            self.lat = None

    def __str__(self):
        if self.lon and self.lat:
            return "{} {} {:.2f} {:.2f}".format(
                self.label,
                self.time,
                self.lon,
                self.lat,
            )
        else:
            return "{} {}".format(self.label, self.time)


class Deployment(object):
    """A deployment describes data collection during a continuous stretch of
    time and is thus described by:
        - a label
        - a start event (see class event`)
        - an end event (see class `event`)
        - a meta dictionnary containing various pieces of information
    """

    def __init__(self, label, start=None, end=None, meta=None, loglines=None):
        """Instantiate a `deployment`
        start and end or loglines must be provided

        Parameters
        ----------
        label: str
            Label of the deployment
        start: pynsitu.events.event
            Starting event
        end: pynsitu.events.event, optional
        meta: dict, optional
            meta information about the deployment
        loglines: list, optional
            List of loglines corresponding. Accepted forms:
                [log_start, log_end] or [log_start, log_end, meta]
            where log_start and log_end are str sufficient for the instantiations
            of events (see `event` doc), and where meta is a dictionnary
            containing relevant information about the deployment
        """

        self.label = label

        assert (
            start is not None or loglines is not None
        ), "start or loglines must be provided"

        if start is None:
            start = loglines[0]
        if not isinstance(start, Event):
            self.start = Event(label="start", logline=start)
        #
        if end is None and loglines is not None:
            end = loglines[1]
        if end is not None:
            end = Event(label="end", logline=end)
        self.end = end

        if meta is None:
            if len(loglines) == 3:
                meta = loglines[2]
            else:
                meta = dict()
        self.meta = dict(**meta)

    def __getitem__(self, key):
        if key in self.meta:
            return self.meta[key]
        return getattr(self, key)

    def __repr__(self):
        return "cognac.insitu.events.deployment({})".format(str(self))

    def __str__(self):
        return self.label + " / " + str(self.start) + " / " + str(self.end)

    def plot_on_map(
        self,
        ax,
        line=True,
        label=False,
        label_yshift=1,
        s=10,
        **kwargs,
    ):
        """Plot deployment on a map

        Parameters
        ----------
        ax: matplotlib.pyplot.axes
            Axis where to plot the event
        line: boolean, optional
            Plot a line between start and end
        label: boolean, optional
            Print label (False by default)
        label_yshift: float, optional
            Shifts the label in the y direction (1 by default)
        **kwargs: optional
            Passed to pyplot plotting methods, if cartopy is used, one should
            at least pass `transform=ccrs.PlateCarree()`
        """
        if self.start.lon is None:
            # exits right for deployments that do not have lon/lat info
            return
        #
        x0, y0 = self.start.lon, self.start.lat
        x1, y1 = self.end.lon, self.end.lat
        #
        ax.scatter(x0, y0, s, marker="o", **kwargs)
        ax.scatter(x1, y1, s, marker="*", **kwargs)
        #
        if line:
            ax.plot([x0, x1], [y0, y1], "-", **kwargs)
        if label:
            xb, yb = ax.get_xbound(), ax.get_ybound()
            xoffset = 0.02 * (xb[1] - xb[0])
            yoffset = 0.02 * (yb[1] - yb[0]) * label_yshift
            if type(label) is not str:
                label = self.label
            ax.text(x0 + xoffset, y0 + yoffset, label, fontsize=10)
        return


class Deployments(UserDict):
    """deployement dictionnary, provides shortcuts to access data in meta subdicts, e.g.:
    p = deployments(meta=dict(a=1))
    p["a"] # returns 1
    """

    def __init__(self, *args, **kwargs):
        self.meta = dict(label="deployments", color="0.5")
        super().__init__(*args, **kwargs)
        if "meta" in self.data:
            self.meta.update(self.data.pop("meta"))

    def __getitem__(self, key):
        if key in self.meta:
            return self.meta[key]
        return self.data[key]

    # def __iter__(self):
    #    """ yield value instead of key """
    #    for key, value in self.data.items():
    #        yield value

    def __repr__(self):
        return "cognac.insitu.events.deployments({})".format(str(self))

    def __str__(self):
        return self["label"] + "\n" + "\n".join(str(d) for d in self)


class Platform(UserDict):
    """platform dictionnary, provides shortcuts to access data in meta, sensors and deployments subdicts, e.g.:
    p = platform(sensors=dict(a=1), deployments=dict(b=2))
    p["a"] # returns 1
    """

    def __getitem__(self, key):
        for t in ["meta", "sensors", "deployments"]:
            if key in self.data[t]:
                return self.data[t][key]
        return self.data[key]


class Campaign(object):
    """Campaign object, gathers deployments information from a yaml file"""

    def __init__(self, file):

        # open yaml information file
        import yaml

        if ".yaml" not in file and ".yml" not in file:
            file = file + ".yaml"
        with open(file, "r") as stream:
            cp = yaml.full_load(stream)

        # process campaign meta data
        self.meta = _process_meta_campaign(cp)
        self.name = self.meta["name"]

        # deployments
        if "deployments" in cp:
            self.deployments = Deployments(
                {
                    d: Deployment(label=d, **v) if d != "meta" else v
                    for d, v in cp["deployments"].items()
                }
            )

        # platforms
        if "platforms" in cp:
            self.platforms = _process_platforms(cp["platforms"])

        # dev
        self.cp = cp

    def __repr__(self):
        return "cognac.insitu.events.campaign({})".format(str(self))

    def __str__(self):
        # fmt = "%Y-%m-%d %H:%M:%S"
        fmt = "%Y/%m/%d"
        start = self["start"].strftime(fmt)
        end = self["end"].strftime(fmt)
        return self["name"] + " {} to {}".format(start, end)

    def __getitem__(self, item):
        if item in self.meta:
            return self.meta[item]
        elif item in self.deployments:
            return self.deployments[item]
        elif item in self.platforms:
            return self.platforms[item]
        else:
            return None

    def __iter__(self):
        """iterates around deployments and platforms"""
        for key in list(self.deployments) + list(self.platforms):
            yield key

    def items(self):
        """loops around deployments and platforms, useful?"""
        for key, value in {**self.deployments, **self.platforms}.items():
            yield key, value

    def plot_map(self, **kwargs):
        """Plot map
        Wrapper around geo.plot_map, see related doc
        """
        dkwargs = dict(
            bounds=self["bounds"],
            # bathy=self.bathy['label'],
            levels=self["bathy"]["levels"],
        )
        # dkwargs.update((k,v) for k,v in kwargs.items() if v is not None)
        dkwargs.update(kwargs)
        fac = plot_map(cp=self, **dkwargs)
        plot_bathy(fac, bathy=self["bathy"]["label"], **dkwargs)
        return fac

    def map(
        self,
        width="70%",
        height="70%",
        tiles="Cartodb Positron",
        ignore=[],
        overwrite_contours=False,
        zoom=10,
        **kwargs,
    ):
        """Plot overview map with folium

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

        """
        import folium
        from folium.plugins import MeasureControl, MousePosition

        if ignore == "all":
            ignore = list(self.deployments) + self(self.platforms)

        m = folium.Map(
            location=[self["lat_mid"], self["lon_mid"]],
            width=width,
            height=height,
            zoom_start=zoom,
            tiles=tiles,
        )

        # bathymetric contours
        contour_file = os.path.join(self["path_processed"], "bathy_contours.geojson")
        if not os.path.isfile(contour_file) or (
            os.path.isfile(contour_file) and overwrite_contours
        ):
            store_bathy_contours(
                self.bathy["label"],
                contour_file=contour_file,
                levels=self.bathy["levels"],
                bounds=self.bounds,
            )
        contours_geojson = load_bathy_contours(contour_file)

        tooltip = folium.GeoJsonTooltip(
            fields=["title"],
            aliases=["depth"],
        )
        popup = folium.GeoJsonPopup(
            fields=["title"],
            aliases=["depth"],
        )
        # colorscale = branca.colormap.linear.Greys_03.scale(levels[-1],levels[0])
        def style_func(feature):
            return {
                "color": feature["properties"][
                    "stroke"
                ],  # colorscale(feature['properties']['level-value']),
                "weight": 3,  # x['properties']['stroke-width'],
                #'fillColor': x['properties']['fill'],
                "opacity": 1.0,
                #'popup': feature['properties']['title'],
            }

        folium.GeoJson(
            contours_geojson,
            name="geojson",
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
                    folium.Polygon(
                        [(d.start.lat, d.start.lon), (d.end.lat, d.end.lon)],
                        tooltip=uname
                        + " "
                        + d.label
                        + "<br>"
                        + str(d.start.time)
                        + "<br>"
                        + str(d.end.time),
                        color=cnames[u["color"]],
                        dash_array="10 20",
                        opacity=0.5,
                    ).add_to(m)
                    folium.Circle(
                        (d.start.lat, d.start.lon),
                        tooltip=uname + " " + d.label + "<br>" + str(d.start.time),
                        radius=2 * 1e2,
                        color=cnames[u["color"]],
                    ).add_to(m)
                    folium.Circle(
                        (d.end.lat, d.end.lon),
                        tooltip=uname + " " + d.label + "<br>" + str(d.end.time),
                        radius=1e2,
                        color=cnames[u["color"]],
                    ).add_to(m)

        # useful plugins

        MeasureControl().add_to(m)

        fmtr_lon = (
            "function(dec) {var min= (dec-Math.round(dec))*60; "
            + "direction = (dec < 0) ? 'W' : 'E'; "
            + "return L.Util.formatNum(dec, 0) + direction + L.Util.formatNum(min, 2);};"
        )
        fmtr_lat = (
            "function(dec) {var min= (dec-Math.round(dec))*60; "
            + "direction = (dec < 0) ? 'S' : 'N'; "
            + "return L.Util.formatNum(dec, 0) + direction + L.Util.formatNum(min, 2);};"
        )
        MousePosition(lat_formatter=fmtr_lon, lng_formatter=fmtr_lat).add_to(m)

        return m

    def timeline(
        self,
        platforms=True,
        sensors=True,
        deployments=True,
        height=0.3,
        labels=False,
        start_scale=1,
        ax=None,
        grid=True,
    ):
        """Plot the campaign deployment timeline

        Parameters
        ----------
        platforms: boolean, optional
        sensors: boolean, optional
        deployments: boolean, optional
        height: float, optional
            bar heights
        legend:, optional
        start_scale:
        ax: pyplot.axes, optional
        grid: boolean, optional

        """

        if ax is None:
            fig = plt.figure(figsize=(15, 5))
            ax = fig.add_subplot(111)

        y = 0
        yticks, yticks_labels = [], []
        starts, ends = [], []

        def plot_d(d, y, label=None, color=None, **kwargs):
            """plot deployment as single rectangle"""
            start = mdates.date2num(d.start.time)
            end = mdates.date2num(d.end.time)
            rect = Rectangle(
                (start, y - height / 2.0), end - start, height, color=color
            )
            ax.add_patch(rect)
            starts.append(start)
            ends.append(end)
            if label is not None:
                if color in ["black", "k"]:
                    color_txt = "w"
                else:
                    color_txt = "k"
                print(color_txt)
                ax.text(start, y, label, va="center", color=color_txt)

        # common deployments
        if deployments:
            for _, d in self.deployments.items():
                _kwargs = dict(label=d.label, **d.meta)
                # if not labels:
                #    _kwargs.pop("label")
                plot_d(d, y, **_kwargs)
                yticks.append(y)
            yticks_labels.append("deployments")
            y += -1

        # platform
        for p, pf in self.platforms.items():
            if platforms and pf["deployments"]:
                for _, d in pf["deployments"].items():
                    _kwargs = dict(label=d.label, **pf["meta"])
                    if not labels:
                        _kwargs.pop("label")
                    plot_d(d, y, **_kwargs)
                yticks.append(y)
                yticks_labels.append(p)
                y += -1
            #
            if sensors:
                for s, sv in pf["sensors"].items():
                    for _, d in sv.items():
                        _kwargs = {**sv.meta}
                        _kwargs.pop("label")
                        plot_d(d, y, **_kwargs)
                    yticks.append(y)
                    yticks_labels.append(p + " " + s)
                    y += -1

        ax.set_title(self.name)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks_labels)

        # assign date locator / formatter to the x-axis to get proper labels
        locator = mdates.AutoDateLocator(minticks=3)
        formatter = mdates.AutoDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        if grid:
            ax.set_axisbelow(True)
            ax.grid()

        # set the limits
        delta_time = max(ends) - min(starts)
        plt.xlim(
            [
                min(starts) - delta_time * 0.05 * start_scale,
                max(ends) + delta_time * 0.05,
            ]
        )
        plt.ylim([y + 1 - 2 * height, 2 * height])
        print(y, y + 1 - 2 * height)
        return ax

    def add_legend(
        self,
        ax,
        labels=None,
        skip_ship=True,
        colors=None,
        **kwargs,
    ):
        """Add legend for units on an axis,
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
            labels = [l for l in labels if l != "ship"]
        custom_lines = []
        for label in labels:
            if colors and label in colors:
                c = colors[label]
            else:
                c = self[label]["color"]
            custom_lines.append(Line2D([0], [0], color=c, lw=4))
        ax.legend(custom_lines, labels, **kwargs)

    def load(self, item, toframe=False):
        """load processed data files
        recall item

        Returns
        -------
        output: dict
            {'unit0': {'deployment0': data, ...}}
        """

        # straight netcdf file
        if ".nc" in item:
            file = os.path.join(self["path_processed"], item)
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
        if not D:
            return None
        # elif len(D)==1:
        #    # may want just to return D instead
        #    return D[list(D.keys())[0]]
        else:
            return D

    def _get_unit_files(self, unit, extension="nc"):
        """get all processed files associated with one unit"""
        files = sorted(
            glob(os.path.join(self["path_processed"], unit + "*." + extension)),
            key=_extract_last_digit,
        )
        # find out whether there are multiple deployments
        if len(files) == 1:
            return dict(d0=files[0])
        D = {}
        for f in files:
            s = f.split("/")[-1].split(".")[0].replace(unit, "").split("_")
            assert (
                len(s) == 2
            ), f"there must be 0 or 1 underscore, but we have: {f}, {s}"
            D[s[1]] = f
        return D

    def _get_processed_files(
        self,
        unit="*",
        item="*",
        deployment="*",
        extension="nc",
    ):
        """Return processed data files

        Parameters
        ----------
        unit, item, d, extention: str, optional
            Defaults: '*', '*', '*', 'nc'
            Typical file path: self["path_processed"]+unit+'_'+item+'_'+deployment+'.'+extension

        """
        if item is None:
            _item = ""
        else:
            _item = item + "_"
        if any([_ == "*" for _ in [unit, item, deployment]]):
            return glob(
                os.path.join(
                    self["path_processed"],
                    unit + "_" + _item + deployment + "." + extension,
                )
            )
        else:
            return os.path.join(
                self["path_processed"],
                unit + "_" + _item + deployment + "." + extension,
            )


_default_campaign_meta = {
    "name": "unknown",
    "lon": None,
    "lat": None,
    "start": None,
    "end": None,
    "bathy": None,
    "path": None,
    "path_raw": "",
    "path_processed": "",
}


def _process_meta_campaign(cp):
    """process meta campaign data"""

    # fill in meta information
    meta = dict(**_default_campaign_meta)
    meta.update(**cp["campaign"])

    lon, lat = meta["lon"], meta["lat"]
    if lon and lat:
        # ensure coords are floats
        meta["lon"] = tuple(float(l) for l in lon)
        meta["lat"] = tuple(float(l) for l in lat)
        #
        meta["bounds"] = lon + lat
        meta["lon_mid"] = (lon[0] + lon[1]) * 0.5
        meta["lat_mid"] = (lat[0] + lat[1]) * 0.5

    meta["start"] = pd.Timestamp(meta["start"]) if meta["start"] else None
    meta["end"] = pd.Timestamp(meta["end"]) if meta["end"] else None

    # path to raw data
    path_raw = meta["path_raw"]
    if path_raw:
        if path_raw[0] != "/":
            path_raw = os.path.join(meta["path"], meta["path_raw"])
    meta["path_raw"] = path_raw

    # path to processed data
    path_processed = meta["path_processed"]
    if path_processed:
        if path_processed[0] != "/":
            path_processed = os.path.join(meta["path"], path_processed)
    meta["path_processed"] = path_processed

    return meta


def _process_platforms(platforms):
    """process platforms data"""

    pfs = dict()

    for p, v in platforms.items():

        pf = Platform()

        pmeta = dict()
        if "meta" in v:
            pmeta.update(**v["meta"])
        pf["meta"] = pmeta

        # deployments
        D = Deployments(meta=dict(label=p, **pmeta))
        if "deployments" in v:
            D.update(
                {
                    d: Deployment(label=d, loglines=vd)
                    for d, vd in v["deployments"].items()
                    if d != "meta"
                }
            )
        pf["deployments"] = D

        # sensors
        sensors = dict()
        if "sensors" in v:
            # o["sensors"] = list(v["sensors"])
            for s, vs in v["sensors"].items():
                D = Deployments(meta=dict(label=s, **pmeta))
                if "deployments" in vs:
                    D.update(
                        {
                            d: Deployment(label=d, loglines=vd) if d != "meta" else vd
                            for d, vd in vs["deployments"].items()
                        }
                    )
                sensors[s] = D
        pf["sensors"] = sensors

        # store in platforms dict
        pfs[p] = pf

    return pfs


def _extract_last_digit(filename):
    """extract last digit prior to extension in filename"""
    last_str = filename.split("_")[-1].split(".")[0]
    return int(re.search(r"\d+$", last_str).group())
