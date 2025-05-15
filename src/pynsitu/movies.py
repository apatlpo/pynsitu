import os, shutil
from pathlib import Path
from subprocess import check_output, STDOUT

import xarray as xr
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

try:
    from tqdm import tqdm
except:
    tqdm = lambda x: x

from .maps import crs

# to do:
#   ? automatic zooming to include platforms
#   shipboard ADCP


class movie(object):
    """Movie generator object

    Parameters
    ----------
    start: str or datetime-like
        Movie start time
    end: str or datetime-like
        Movie end time
    freq: str, datetime.timedelta, or DateOffset
        Movie snapshots frequency
    map_generator: method
        Function generating a snapshot
    fig_dir: str
        Path to directory where temporary figures will be generated
    extent: tuple, optional
        Geographical extent
    title: method, optional
        Method generating figures, should take time as an input
    extra: list, optional
        List of methods adding extra information on figures.
        Method signatures must be e(t, self.data, fig, ax)
    legend: str, list of str, optional
        List of labels that will legended
    **kwargs: actual data of pandas or xarray, each kwarg must be a dict with
    items `data` and `dtype`. `dtype` must be one of: "drifter", "moving", "vector_field"
    "drifter" data must have an id variable
    """

    def __init__(
        self,
        start,
        end,
        freq,
        map_generator,
        fig_dir,
        extent=None,
        title=None,
        extra=None,
        legend=None,
        **kwargs,
    ):

        # required fields
        self.time_range = pd.date_range(start, end, freq=freq)
        self.mgen = map_generator
        self.fig_dir = fig_dir
        assert len(kwargs) > 0, "No data provided"
        self.data = kwargs

        # optional varibales
        self.extent = extent
        if isinstance(extent, dict):
            if "include" not in extent:
                self.extent["include"] = list(kwargs)
            if "exclude" in extent:
                self.extent["include"] = [
                    d for d in self.extent["include"] if d not in extent["exclude"]
                ]
        if title is None:
            title = lambda t: t.strftime("%Y-%m-%d %X")
        self.title = title
        if hasattr(extra, "__call__"):
            extra = [extra]
        assert (
            isinstance(extra, list) or extra is None
        ), "extra must be a method or a list of methods"
        self.extra = extra
        self.legend = legend

        # create directory if not existent
        Path(fig_dir).mkdir(parents=True, exist_ok=True)

    def make_figures(
        self, dpi=150, figformat="png", extents=None, cleanup_dir=True, **kwargs
    ):
        """loop over all times and print associated figures

        Parameters
        ----------
        dpi: int
            Figure resolution in dpi
        figformat: str, optional
            Format of figures
        extents: tuple or dict, optional
            Geographical extent passed to plot_snaphot
        cleanup_dir: boolean, optional
            cleanup figure directory
        **kwargs: passed to plot_snapshot
        """
        # cleanup dir
        if cleanup_dir:
            _clean_dir(self.fig_dir)
        for i, t in enumerate(tqdm(self.time_range)):
            if extents is not None:
                extent = extents[i]
            else:
                extent = None
            fig = self.plot_snapshot(t, extent=extent, **kwargs)
            fig_path = os.path.join(self.fig_dir, f"fig_{i:04}.{figformat}")
            fig.savefig(fig_path, dpi=dpi)
            # other relevant options: facecolor='w', bbox_inches = 'tight'
            plt.close(fig)

    def dry_run(self, **kwargs):
        """loop over all times to collect extents, and eventually smooth their temporal evolution

        Parameters
        ----------
        **kwargs: plot_snapshot kwargs

        Returns
        -------
        extents: list
            List of extents
        """
        extents = []
        for _, t in enumerate(tqdm(self.time_range)):
            extent = self.plot_snapshot(t, dry_run=True, **kwargs)
            extents.append(extent)
        return extents

    def plot_snapshot(self, t, extent=None, legend=None, dry_run=False):
        """plot one snapshot

        Parameters
        ----------
        t: int, pd.Timestamp
        extent: tuple, optional
            Geographical extent
        legend: str or list of str
            Labels to be legended
        dry_run: boolean, optional

        Returns
        -------
        fig: plt.Figure
            Figure handle
        """

        if isinstance(t, int):
            t = self.time_range[t]

        if extent is not None or self.extent is not None:
            if extent is not None:
                pass
            elif isinstance(self.extent, dict):
                extent, centroid = _compute_extent(
                    t,
                    self.extent["buffer"],
                    self.extent["include"],
                    self.data,
                )
                if extent is None:
                    return
                if "aspect_ratio" in self.extent:
                    fig, ax, cbar = self.mgen(extent)  # slow unfortunately
                    plt.close(fig)
                    extent = adjust_extent_aspect_ratio(
                        self.extent["aspect_ratio"],
                        extent,
                        ax.projection,
                    )
            elif extent is None:
                extent = self.extent
            # dry run to get extents (and eventually smooth them)
            if dry_run:
                plt.close(fig)
                return extent
            fig, ax, cbar = self.mgen(extent)
        else:
            fig, ax, cbar = self.mgen()

        hdl, h = [], None
        for label, d in self.data.items():
            if d["dtype"] == "drifter":
                kw = {
                    k: v for k, v in d.items() if k not in ["dtype", "data", "colors"]
                }
                h = []
                for i in d["data"].reset_index()["id"].unique():
                    if "colors" in d:
                        kw["color"] = d["colors"][i]
                    _h = plot_moving_platform(
                        d["data"].loc[d["data"]["id"] == i], i, t, ax, **kw
                    )
                    if _h is not None:
                        h += _h
            elif d["dtype"] == "moving":
                kw = {k: v for k, v in d.items() if k not in ["dtype", "data"]}
                h = plot_moving_platform(d["data"], label, t, ax, **kw)
            elif d["dtype"] == "vector_field":
                kw = {k: v for k, v in d.items() if k not in ["dtype", "data"]}
                plot_vector_field(d["data"], label, t, ax, **kw)
            elif d["dtype"] == "custom":
                kw = {k: v for k, v in d.items() if k not in ["dtype", "data", "func"]}
                d["func"](d["data"], label, t, ax, **kw)
            # append legend handler
            if isinstance(h, list):
                hdl += h
            elif h is not None:
                hdl.append(h)
            h = None

        if self.legend is not None and legend is None:
            legend = self.legend
        if legend is not None:
            hdl = [h for h in hdl if h is not None]
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
            if isinstance(legend, str):
                loc = legend
            else:
                loc = None
            ax.legend(handles=hdl, loc=loc)

        ax.set_title(self.title(t))

        # add extra information on the figure
        if self.extra is not None:
            for e in self.extra:
                e(t, self.data, fig, ax)

        return fig


# ---- plot components


def plot_moving_platform(
    df,
    label,
    t,
    ax,
    dt_trail=None,
    marker_memory="10min",
    head_style={},
    **trail_style,
):
    """Plot one drifter trail

    Parameter
    ---------
    df: pandas.DataFrame
        must contain columns: "time", "longitude", "latitude"
    label: str
        drifter label
    t: pd.Timestamp
        time of snapshot
    ax: matplotlib.pyplot.axis
    dt_trail: str, optional
        temporal trail size
    head_style: dict, optional
        head scatter kwargs
    **trail_style: optional, trail line style
    """
    # recursive call if multiple deployments have been passed
    if isinstance(df, dict):
        hdl = []
        for k, _df in df.items():
            h = plot_moving_platform(
                _df,
                label,  # +" "+k
                t,
                ax,
                dt_trail=dt_trail,
                head_style=head_style,
                **trail_style,
            )
            if h is not None:
                hdl += [h[0]]  # only keep one deployment
        return hdl
    # massage dataframe
    df = df.reset_index().set_index("time").sort_index()
    if isinstance(dt_trail, str):
        dt_trail = pd.Timedelta(dt_trail)
    if dt_trail is None:
        df = df.loc[:t]
    else:
        df = df.loc[t - dt_trail : t]

    trail_kw = dict(
        transform=crs,
        color="k",
        lw=2,
        label=label,
    )
    trail_kw.update(**trail_style)

    head_kw = dict(
        transform=crs,
        color=trail_kw["color"],
        lw=2,
        marker="^",
        markeredgecolor="k",
        markeredgewidth=0.5,
        markersize=8,
        label=label,
    )
    head_kw.update(**head_style)

    if df.index.size > 0:
        hdl = ax.plot(df.longitude, df.latitude, **trail_kw)
        # if df.index[-1] == t:
        if df.index[-1] >= t - pd.Timedelta(marker_memory):
            ax.plot(df.longitude[-1], df.latitude[-1], **head_kw)
        return hdl


def plot_vector_field(
    ds,
    label,
    t,
    ax,
    di=None,
    uref=None,
    quiverkey_kwargs={},
    **quiver_kwargs,
):
    """plot a vector field on the map

    Parameters
    ----------
    ds: xr.Dataset
    label: str
    t: str, pd.Timestamp
        Time to be plotted
    ax: matplotlib.pyplot.axis
        Axis handle
    di: int, optional
        quiver step
    uref: velocity, required
        reference velocity
    quiverkey_kwargs: kwargs passed to quiverkey
    **quiver_kwargs: kwargs passed to quiver
    """
    if di is not None:
        ds = ds.isel(longitude=slice(0, None, di), latitude=slice(0, None, di))

    # interpolate at time t
    ds = ds.interp(time=t)

    _quiver_kwargs = dict(
        color="k",
        x="longitude",
        y="latitude",
        u="u",
        v="v",
        transform=crs,
        zorder=20,
        add_guide=False,
    )  # scale=1e2,
    _quiver_kwargs.update(**quiver_kwargs)
    _quiverk_kwargs = dict(color="k", labelpos="N", coordinates="axes", zorder=20)
    _quiverk_kwargs.update(**quiverkey_kwargs)

    q = ds.plot.quiver(**_quiver_kwargs)

    assert uref is not None, "uref is required"
    qk = ax.quiverkey(
        q, 0.1, 0.9, uref, f"{uref} m/s", transform=crs, **_quiverk_kwargs
    )


# ---- control geographical extent


def _compute_extent(t, buffer, include, datasets):
    """find geographical bounds based on a various datasets and a buffer size"""
    ds_flat = _select_flatten_data(datasets, t, include)
    if not ds_flat:
        return None, None
    lon_min = min([float(a.longitude.min()) for a in ds_flat])
    lon_max = max([float(a.longitude.max()) for a in ds_flat])
    lat_min = min([float(a.latitude.min()) for a in ds_flat])
    lat_max = max([float(a.latitude.max()) for a in ds_flat])
    lon_centroid = 0.5 * (lon_min + lon_max)
    lat_centroid = 0.5 * (lat_min + lat_max)
    lon_scale = 1 / np.cos(lat_centroid * np.pi / 180.0)
    extent = (
        lon_min - buffer * lon_scale,
        lon_max + buffer * lon_scale,
        lat_min - buffer,
        lat_max + buffer,
    )
    return extent, (lon_centroid, lat_centroid)


def _select_flatten_data(data, end, include, start=None):
    """from dict to list of datasets (pd.DataFrame or xr.Datasets)"""
    D0 = [_dict["data"] for label, _dict in data.items() if label in include]
    D1 = []
    for d in D0:
        if isinstance(d, dict):
            D1 += [_select(_d, start, end) for _, _d in d.items()]
        else:
            D1.append(_select(d, start, end))
    # drops None's
    return [d for d in D1 if d is not None]


def _select(d, *args):
    """dispatch depending on dataset type"""
    if isinstance(d, xr.Dataset):
        return _select_xr(d, *args)
    elif isinstance(d, pd.DataFrame):
        return _select_pd(d, *args)


def _select_pd(d, start, end):
    if "time" not in d.columns:
        d = d.reset_index()
        assert (
            "time" in d.columns
        ), "time should be in dataframe as a column or an index"
    d = d.loc[d.time <= end]
    if d.empty:
        return None
    if start is not None:
        d = d.loc[d.time >= start]
    # rename lon/lat if need be:
    if "lon" in d.columns:
        d = d.rename(columns=dict(lon="longitude"))
    if "lat" in d.columns:
        d = d.rename(columns=dict(lat="latgitude"))
    return d


def _select_xr(d, start, end):
    d = d.where(d.time <= end, drop=True)
    if d.time.size == 0:
        return None
    if start is not None:
        d = d.where(d.time >= start, drop=True)
    if "lon" in d:
        d = d.rename(lon="longitude")
    if "lat" in d:
        d = d.rename(lat="latgitude")
    return d


def adjust_extent_aspect_ratio(aspect_ratio, extent, out_crs):
    """
    Generate extent from aspect ratio, target extent, and projection
    latitude bounds are adjusted to maintain the aspect ratio

    Parameters
    ----------
    aspect_ratio: tuple:
        Aspect ratio x/y
    extent: tuple/list
        longitude and latitude bounds
    out_crs: cartopy.crs
        Out crs for extent values.

    Returns:
        tuple: (lon_min, lon_max, lat_min, lat_max) or in projected coordinates
    """

    c = dict(
        SW=(extent[0], extent[2]),
        SE=(extent[1], extent[2]),
        NE=(extent[1], extent[3]),
        NW=(extent[0], extent[3]),
    )
    # compute transformed coordinates
    cxy = {k: out_crs.transform_point(*p, src_crs=crs) for k, p in c.items()}
    cxyc = {k: p[0] + 1j * p[1] for k, p in cxy.items()}
    mxy = dict(
        S=(cxyc["SW"] + cxyc["SE"]) * 0.5,
        E=(cxyc["SE"] + cxyc["NE"]) * 0.5,
        N=(cxyc["NW"] + cxyc["NE"]) * 0.5,
        W=(cxyc["SW"] + cxyc["NW"]) * 0.5,
    )
    Ex = np.abs(mxy["E"] - mxy["W"])
    Ey = np.abs(mxy["N"] - mxy["S"])

    a = aspect_ratio[1] / aspect_ratio[0]
    if Ey / Ex >= a:
        # initial aspect ratio is taller -> make wider
        Lx = Ey / a
        Ly = Ey
    else:
        # initial aspect ratio is wider -> make taller
        Lx = Ex
        Ly = Ex * a
    # Lx = max(Ex, Ey / a)
    # Ly = Lx * a

    # central point
    x = (cxy["SW"][0] + cxy["SE"][0] + cxy["NW"][0] + cxy["NE"][0]) / 4
    y = (cxy["SW"][1] + cxy["SE"][1] + cxy["NW"][1] + cxy["NE"][1]) / 4

    if out_crs != crs:
        lon_min, _ = crs.transform_point(x - Lx / 2, y, src_crs=out_crs)
        lon_max, _ = crs.transform_point(x + Lx / 2, y, src_crs=out_crs)
        _, lat_min = crs.transform_point(x, y - Ly / 2, src_crs=out_crs)
        _, lat_max = crs.transform_point(x, y + Ly / 2, src_crs=out_crs)

    # print(extent, (lon_min, lon_max, lat_min, lat_max), Ex, Ey, Lx, Ly, cxy, mxy)

    return lon_min, lon_max, lat_min, lat_max


def pad_smooth_extents(extents, tau):
    """temporally smooth a collection of extents

    Parameters
    ----------
    extents: list of 4-tuples
        List of extents (tuples)
    tau: int
        Defines the timescale of smoothing

    Returns
    -------
    extents: list
        List of extents (tuples), smoothed
    """
    extents = extents.copy()  # makes a copy first

    # pad
    first = 0
    while extents[first] is None:
        first += 1
    for i in range(first):
        extents[i] = extents[first]
    last = len(extents) - 1
    while extents[last] is None:
        last -= 1
    for i in range(last + 1, len(extents)):
        extents[i] = extents[last]

    # convert to dataframe
    df = pd.DataFrame(extents)
    df_smooth = df.rolling(tau, center=True, min_periods=(tau // 2)).mean()
    # return df, df_smooth # dev
    return [tuple(r.values) for _, r in df_smooth.iterrows()]


def generate_mpg(fig_dir, movie_name, output_dir=None):
    """generate movies, requires ffmpeg in environment, do with image2Movies otherwise

    https://stackoverflow.com/questions/24961127/how-to-create-a-video-from-images-with-ffmpeg

    Parameters
    ----------
    fig_dir: str
        Path to directory containing all figures
    movie_name: str
        Name movies (without `.mp4` extension)
    output_dir: str, optional
        Path to output directory, default is local directory
    """
    if output_dir is None:
        output_dir = os.getcwd()
    movie_path = os.path.join(output_dir, movie_name + ".mp4")
    if os.path.isfile(movie_path):
        os.remove(movie_path)
    com = f"""ffmpeg -framerate 10 -pattern_type glob -i '{fig_dir}/*.png' -c:v libx264 -pix_fmt yuv420p {movie_path}"""
    _ = check_output(com, shell=True, stderr=STDOUT, universal_newlines=True).rstrip(
        "\n"
    )
    print(f"movies should be ready at: {movie_path}")


def _clean_dir(folder):
    """cleanup file in a directory"""
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))
