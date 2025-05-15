#
# ------------------------- Event/Deployment objects -----------------------------------
#
import os
from glob import glob
from collections import UserDict
import re

import pandas as pd
import xarray as xr
import math

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.colors import cnames

from .maps import crs, plot_map, load_bathy_contours, store_bathy_contours


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
            if loglines is not None and len(loglines) == 3:
                meta = loglines[2]
                assert isinstance(meta, dict)
                if "meta" in meta:
                    # not pretty but seems necessary to handle all cases
                    meta = meta["meta"]
            else:
                meta = dict()
        # add length in days
        meta["duration_days"] = (self.end.time - self.start.time) / pd.Timedelta("1D")

        self.meta = dict(**meta)

    def __getitem__(self, key):
        if key in self.meta:
            return self.meta[key]
        return getattr(self, key)

    def __contains__(self, item):
        if item in self.meta:
            return True
        return False

    def __repr__(self):
        return "Deployment({})".format(str(self))

    def __str__(self):
        return self.label + " / " + str(self.start) + " / " + str(self.end)

    def to_deployments(self):
        """converts to deployments object"""
        return Deployments(**{self.label: self})

    def plot_on_map(
        self,
        ax,
        line=False,
        label=True,
        label_xyshift=(0.1, 0.1),
        s=5,
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
        label_xyshift: tuple, optional
            Shifts the label in the x and y direction, (.1,.1) by default
        **kwargs: optional
            Passed to pyplot plotting methods, if cartopy is used, one should
            at least pass `transform=ccrs.PlateCarree()`
        """
        if self.start.lon is None:
            # exits right for deployments that do not have lon/lat info
            return
        dkwargs = dict(transform=crs)
        dkwargs.update(**kwargs)
        #
        x0, y0 = self.start.lon, self.start.lat
        x1, y1 = self.end.lon, self.end.lat
        #
        ax.scatter(x0, y0, s, marker="o", **dkwargs)
        ax.scatter(x1, y1, s, marker="*", **dkwargs)
        #
        if line:
            ax.plot([x0, x1], [y0, y1], "-", **dkwargs)
        if label:
            if type(label) is not str:
                label = self.label
            ax.text(
                x0 + label_xyshift[0],
                y0 + label_xyshift[1],
                label,
                fontsize=10,
                transform=crs,
            )


class Deployments(UserDict):
    """deployement dictionnary, provides shortcuts to access data in meta subdicts, e.g.:
    p = Deployments(meta=dict(a=1))
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
        return "Deployments({})".format(str(self))

    def __str__(self):
        return self["label"] + "\n" + "\n".join(str(d) for d in self)


class Platform(UserDict):
    """Platform dictionnary, provides shortcuts to access data in meta, sensors and deployments subdicts, e.g.:
    p = platform(sensors=dict(a=1), deployments=dict(b=2))
    p["a"] # returns 1
    """

    def __getitem__(self, key):
        for t in ["meta", "sensors", "deployments"]:
            if t in self.data and key in self.data[t]:
                return self.data[t][key]
        if key in self.data:
            return self.data[key]

    def deployments(self):
        for s in self.data["deployments"]:
            yield s

    def sensors(self):
        for s in self.data["sensors"]:
            yield s

    def __repr__(self):
        return "cognac.insitu.events.Platform({})".format(str(self))

    def __str__(self):
        if "label" in self.data["meta"]:
            out = "Platform " + self["label"] + "\n"
        else:
            out = "Platform - no label\n"
        # deployments
        if self.data["deployments"]:
            out += (
                " general deployments: "
                + " / ".join([d for d in self.deployments()])
                + "\n"
            )
        if self.data["sensors"]:
            out += " sensors: " + " / ".join([d for d in self.sensors()]) + "\n"
        return out


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
        if "deployments" in cp and cp["deployments"] is not None:
            self.deployments = Deployments(
                {
                    d: Deployment(label=d, **v) if d != "meta" else v
                    for d, v in cp["deployments"].items()
                }
            )
        else:
            self.deployments = None

        # platforms
        if "platforms" in cp and cp["platforms"] is not None:
            self.platforms = _process_platforms(cp["platforms"])
        else:
            self.platforms = None

        # dev
        self.cp = cp

    def __repr__(self):
        return "Campaign({})".format(str(self))

    def __str__(self):
        # fmt = "%Y-%m-%d %H:%M:%S"
        fmt = "%Y/%m/%d"
        start = self["start"].strftime(fmt)
        end = self["end"].strftime(fmt)
        return self["name"] + " {} to {}".format(start, end)

    def __getitem__(self, item):
        if self.meta and item in self.meta:
            return self.meta[item]
        elif self.deployments and item in self.deployments:
            return self.deployments[item]
        elif self.platforms and item in self.platforms:
            return self.platforms[item]
        else:
            return None

    def __iter__(self):
        """iterates around deployments and platforms"""
        L = []
        if self.deployments:
            L += list(self.deployments)
        if self.platforms:
            L += list(self.platforms)
        for key in L:
            yield key

    def get_all_deployments(self):
        """loops over all deployments, e.g.:

        for label, deployment, platform, sensor, meta in cp.get_all_deployments():
            ...

        """
        if self.deployments:
            for label, d in self.deployments.items():
                yield label, d, None, None, d.meta
        if self.platforms:
            for p, vp in self.platforms.items():
                if vp["deployments"]:
                    for label, d in vp["deployments"].items():
                        _meta = dict(**vp["meta"])
                        _meta.update(**d.meta)
                        yield label, d, p, None, _meta
                if vp["sensors"]:
                    _meta = dict(**vp["meta"])
                    for s, vs in vp["sensors"].items():
                        _meta.update(**vs.meta)
                        for label, d in vs.items():
                            yield label, d, p, s, _meta

    def map(self, bathy=None, coastline=None, rivers=None, **kwargs):
        """Plot map
        Wrapper around geo.plot_map, see related doc
        """
        if bathy is None and "bathy" in self.meta and "path" in self["bathy"]:
            bathy = self["bathy"]["path"]
        if coastline is None and "coastline" in self.meta:
            coastline = self["coastline"]
        if rivers is None and "rivers" in self.meta:
            rivers = self["rivers"]
        dkwargs = dict(
            extent=self["bounds"],
            bathy=bathy,
            bathy_levels=self["bathy"]["levels"],
            coastline=coastline,
            rivers=rivers,
        )
        dkwargs.update(**kwargs)
        fac = plot_map(**dkwargs)
        return fac

    def map_folium(
        self,
        width="60%",
        height="60%",
        tiles="Cartodb Positron",
        ignore=None,
        bathy=True,
        overwrite_contours=False,
        zoom=10,
    ):
        """Plot overview map with folium

        Parameters:
        ----------
        width: str, optional
            width of the plot
        height: str, optional
            height of the plot
        tiles: str, optional
            tiles used, see `folium.Map?`` (default is Cartodb Positron)
                - "OpenStreetMap"
                - "Mapbox Bright" (Limited levels of zoom for free tiles)
                - "Mapbox Control Room" (Limited levels of zoom for free tiles)
                - "Stamen" (Terrain, Toner, and Watercolor)
                - "Cloudmade" (Must pass API key)
                - "Mapbox" (Must pass API key)
                - "CartoDB" (positron and dark_matter)
        ignore: list, optional
            Ignore deployment labels
        bathy: boolean, optional
            Turn on/off bathymetric contours plotting
        overwrite_contours: boolean, optional
            Overwrite contour file (default is False)
        zoom: int
            Folium zoom level, see Folium doc `zoom_start` kwarg
            https://python-visualization.github.io/folium/quickstart.html#Getting-Started
        """
        import folium
        from folium.plugins import MeasureControl, MousePosition

        if ignore == "all":
            ignore = [out[0] for out in self.get_all_deployments()]

        m = folium.Map(
            location=[self["lat_mid"], self["lon_mid"]],
            width=width,
            height=height,
            zoom_start=zoom,
            tiles=tiles,
        )

        # bathymetric contours
        if bathy:
            contour_file = os.path.join(
                self["path_processed"], "bathy_contours.geojson"
            )
            if not os.path.isfile(contour_file) or (
                os.path.isfile(contour_file) and overwrite_contours
            ):
                store_bathy_contours(
                    self["bathy"]["path"],
                    contour_file=contour_file,
                    levels=self["bathy"]["levels"],
                    bounds=self["bounds"],
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

        if bathy:
            folium.GeoJson(
                contours_geojson,
                name="geojson",
                style_function=style_func,
                tooltip=tooltip,
                popup=popup,
            ).add_to(m)

        # campaign details
        for label, d, p, s, meta in self.get_all_deployments():
            if "color" in meta:
                color = meta["color"]
            else:
                color = "black"
            if ignore is None or label not in ignore:
                if d.start.lat is None:
                    continue
                _label = " / ".join([x for x in [label, p, s] if x is not None])
                folium.Polygon(
                    [(d.start.lat, d.start.lon), (d.end.lat, d.end.lon)],
                    tooltip=_label
                    + "<br>"
                    + str(d.start.time)
                    + "<br>"
                    + str(d.end.time),
                    color=cnames[color],
                    dash_array="10 20",
                    opacity=0.5,
                ).add_to(m)
                folium.Circle(
                    (d.start.lat, d.start.lon),
                    tooltip=_label + "<br>" + str(d.start.time),
                    radius=2 * 1e2,
                    color=cnames[color],
                ).add_to(m)
                folium.Circle(
                    (d.end.lat, d.end.lon),
                    tooltip=_label + "<br>" + str(d.end.time),
                    radius=1e2,
                    color=cnames[color],
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
        align_deployments=False,
        height=0.6,
        labels=False,
        ax=None,
        grid=True,
        exclude=[],
        figsize=None,
    ):
        """Plot the campaign deployment timeline

        Parameters
        ----------
        platforms: boolean, optional
            Show platforms
        sensors: boolean, optional
            Show sensors
        deployments: boolean, optional
            Show deployments
        align_deployments: boolean, optional
            Align deployments vertically
        height: float, optional
            bar heights, 0.6 by default
        ax: pyplot.axes, optional
        grid: boolean, optional
            Turn grid one (default is True)
        exclude: list, optional
            list of platforms or deployments to exclude
        figsize: tuple, optional
            enforce the size of the output figure
        """
        n = len(self.platforms)
        if ax is None:
            if figsize is None:
                figsize = (15, n / 4)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)

        y = 0
        yticks, yticks_labels = [], []
        starts, ends = [], []

        def plot_d(d, y, label=None, color=None, **kwargs):
            """plot deployment as single rectangle"""
            start = mdates.date2num(d.start.time)
            end = mdates.date2num(d.end.time)

            # normalize rgba if need be, better to convert to hex with matplotlib.colors.to_hex or other
            if isinstance(color, tuple):
                if max(color) > 1:
                    color = tuple(c / 256 for c in color)

            rect = Rectangle(
                (start, y - height / 2.0), end - start, height, color=color
            )
            ax.add_patch(rect)
            starts.append(start)
            ends.append(end)
            if label is not None:
                if color in ["black", "k", "grey"]:
                    color_txt = "w"
                else:
                    color_txt = "k"
                ax.text(start, y, label, va="center", color=color_txt)

        # common deployments
        if deployments and self.deployments:
            if align_deployments:
                yticks.append(y)
                yticks_labels.append("deployments")
                # y += -1
            for _, d in self.deployments.items():
                if d.label in exclude:
                    continue
                _kwargs = dict(**d.meta)
                if align_deployments:
                    _kwargs["label"] = d.label
                plot_d(d, y, **_kwargs)
                if not align_deployments:
                    yticks.append(y)
                    yticks_labels.append(d.label)
                    y += -1
            if align_deployments:
                y += -1

        # platform
        if platforms and self.platforms:
            for p, pf in self.platforms.items():
                if platforms and pf["deployments"] and p not in exclude:
                    for dlabel, d in pf["deployments"].items():
                        _kwargs = dict(**pf["meta"])
                        if not labels:
                            _kwargs.pop("label")
                        else:
                            _kwargs["label"] = dlabel
                        plot_d(d, y, **_kwargs)
                    yticks.append(y)
                    yticks_labels.append(p)
                    y += -1
                #
                if sensors and pf["sensors"]:
                    for s, sv in pf["sensors"].items():
                        if s in exclude:
                            continue
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
        start_scale = 1
        plt.xlim(
            [
                min(starts) - delta_time * 0.05 * start_scale,
                max(ends) + delta_time * 0.05,
            ]
        )
        plt.ylim([y + 1 - 2 * height, 2 * height])

        return ax

    def add_legend(
        self,
        ax,
        labels=[],
        skip=None,
        colors={},
        **kwargs,
    ):
        """Add legend for deployment/platforms on an axis.
        To be used for timelines (see `Campaign.timeline`) as well as maps

        Parameters
        ----------
        ax: pyplot.axes
        labels: list, optional
            List of labels to consider amongst cp deployments/platforms
        skip: list, optional
            List of deployments and platforms to skip
        colors: dict, optional
        **kwargs: passed to legend
        """
        from matplotlib.lines import Line2D

        # fill labels to show
        labels = list(labels)
        if self.deployments:
            labels += list(self.deployments)
        if self.platforms:
            labels += list(self.platforms)

        # get rid of labels to skip
        if skip is not None:
            skip = list(skip)
            labels = [l for l in labels if l not in skip]

        # show
        custom_lines = []
        for label in labels:
            if label in colors:
                c = colors[label]
            elif "color" in self[label]["meta"]:
                c = self[label]["meta"]["color"]
            else:
                c = None
                labels.pop(label)
            if c is not None:
                custom_lines.append(Line2D([0], [0], color=c, lw=4))
        ax.legend(custom_lines, labels, **kwargs)

    def load(self, item, toframe=False, ignore=False):
        """load processed data files

        Parameters
        ----------
        item: str
            Name of netcdf file
        toframe: boolean
            Transform to pd.DataFrame
        ignore: boolean
            Ignore non-existent files

        Returns
        -------
        output: xr.Dataset, pd.DataFrame, dict
            {'file0': ds0, 'file1': ds1, ...}
            {'platform0': {'deployment0': data, ...}}
        """

        file_path = self.load_path(item)
        assert file_path is not None or ignore, "File(s) not found"

        # straight netcdf file
        if isinstance(file_path, str):
            ds = xr.open_dataset(file_path)
            if toframe:
                ds = ds.to_dataframe()
            return ds
        elif isinstance(file_path, dict):
            D = {k: xr.open_dataset(f) for k, f in file_path.items()}
            if toframe:
                D = {k: ds.to_dataframe() for k, ds in D.items()}
            return D

    def load_path(self, item):
        """load processed file path(s)

        Parameters
        ----------
        item: str
            Name of netcdf file

        Returns
        -------
        file_path: str, dict

        """
        # straight netcdf file
        if ".nc" in item:
            file = os.path.join(self["path_processed"], item)
            if not os.path.isfile(file):
                return None
            return file

        # straight zarr archive
        if ".zarr" in item:
            file = os.path.join(self["path_processed"], item)
            if not os.path.isdir(file):
                return None
            return file

        if "*" in item:
            files = sorted(
                glob(os.path.join(self["path_processed"], item)),
            )
            if len(files) == 0:
                return None
            keys = [f.split("/")[-1].replace(".nc", "") for f in files]
        else:
            files = sorted(
                glob(os.path.join(self["path_processed"], item + "_*.nc")),
            )
            if len(files) == 0:
                return None
            keys = [
                f.split("/")[-1].replace(item + "_", "").replace(".nc", "")
                for f in files
            ]

        return {k: f for f, k in zip(files, keys)}


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
        lon = tuple(float(l) for l in lon)
        lat = tuple(float(l) for l in lat)
        meta["lon"] = lon
        meta["lat"] = lat
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

        pmeta = dict(label=p)
        if "meta" in v:
            pmeta.update(**v["meta"])
        pf["meta"] = pmeta

        # deployments
        D = Deployments(meta=pmeta)
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
                smeta = dict(**pmeta)
                smeta.update(label=s)
                if "meta" in vs:
                    smeta.update(**vs["meta"])
                D = Deployments(meta=smeta)
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
