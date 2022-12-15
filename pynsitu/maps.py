import os

import xarray as xr
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

try:
    import cmocean.cm as cm
except:
    print("Warning: could not import cmocean")

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.io import shapereader

    crs = ccrs.PlateCarree()
except:
    print("Warning: could not import cartopy")

# ------------------------------ cartopy map -----------------------------


def plot_map(
    da=None,
    extent="global",
    projection=None,
    title=None,
    fig=None,
    figsize=None,
    ax=None,
    colorbar=True,
    colorbar_kwargs={},
    centered_clims=False,
    gridlines=True,
    bathy=False,
    bathy_levels=None,
    land=False,
    coastline="110m",
    rivers=False,
    **kwargs,
):
    """Plot a geographical map

    Parameters
    ----------
    da: xr.DataArray, optional
        Scalar field to plot
    extent: str, list/tuple, optional
        Geographical extent, "global" or [lon_min, lon_max, lat_min, lat_max]
    projection: cartopy.crs.??, optional
        Cartopy projection, e.g.: `projection = ccrs.Robinson()`
    title: str, optional
        Title
    fig: matplotlib.figure.Figure, optional
        Figure handle, create one if not passed
    figsize: tuple, optional
        Figure size, e.g. (10,5)
    ax: matplotlib.axes.Axes, optional
        Axis handle
    colorbar: boolean, optional
        add colorbar (default is True)
    colorbar_kwargs: dict, optional
        kwargs passed to colorbar
    centered_clims: boolean, optional
        Center color limits (default is False)
    gridlines: boolean, optional
        Add grid lines (default is True)
    bathy: boolean, optional
        Plot bathymetry (default is False)
    bathy_levels: list/tuple
        Levels of bathymetry to plot
    land: boolean, optional
        (default is True)
    coast_resolution: str, optional
    """

    #
    if figsize is None:
        figsize = (10, 5)
    if fig is None:
        fig = plt.figure(figsize=figsize)
    if extent == "global":
        proj = ccrs.Robinson()
        extent = None
    else:
        _lon_central = (extent[0] + extent[1]) * 0.5
        _lat_central = (extent[2] + extent[3]) * 0.5
        # used to be ccrs.Orthographic(...)
        proj = ccrs.LambertAzimuthalEqualArea(
            central_longitude=_lon_central,
            central_latitude=_lat_central,
        )
    if projection is not None:
        proj = projection
    if ax is None:
        ax = fig.add_subplot(111, projection=proj)

    if extent is not None:
        ax.set_extent(extent)

    # copy kwargs for update
    kwargs = kwargs.copy()

    if centered_clims and da is not None:
        vmax = float(abs(da).max())
        vmin = -vmax
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax

    if bathy:
        da = load_bathy(bathy, bounds=extent)["depth"]
        if bathy_levels is not None:
            CS = da.plot.contour(
                x="longitude",
                y="latitude",
                ax=ax,
                transform=crs,
                levels=bathy_levels,
                colors="k",
            )
            ax.clabel(CS, CS.levels, inline=False, fontsize=10)
            da = None
        else:
            kwargs.update(cmap=cm.deep, vmin=0)

    if da is not None:
        im = da.squeeze().plot.pcolormesh(
            x="longitude",
            y="latitude",
            ax=ax,
            transform=crs,
            add_colorbar=False,
            **kwargs,
        )

    # coastlines and land:
    if land:
        _plot_land(ax, land)
    if coastline:
        _plot_coastline(ax, coastline)
    if rivers:
        _plot_rivers(ax, rivers)

    if da is not None and colorbar:
        # cbar = fig.colorbar(im, extend="neither", shrink=0.7, **colorbar_kwargs)
        axins = inset_axes(
            ax,
            width="5%",  # width = 5% of parent_bbox width
            height="100%",  # height : 50%
            loc="lower left",
            bbox_to_anchor=(1.05, 0.0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        # cbar = fig.colorbar(im, extend="neither", shrink=0.9,
        cbar = fig.colorbar(im, extend="neither", cax=axins, **colorbar_kwargs)
    else:
        cbar = None

    if gridlines:
        gl = ax.gridlines(
            draw_labels=True,
            dms=False,
            x_inline=False,
            y_inline=False,
        )
        gl.right_labels = False
        gl.top_labels = False

    if title is not None:
        ax.set_title(
            title,
            fontdict={
                "fontsize": 12,
            },
        )  # "fontweight": "bold"
    #
    return {"fig": fig, "ax": ax, "cbar": cbar}


def _plot_land(ax, land, **kwargs):
    """plot land"""
    dkwargs = dict(edgecolor="face", facecolor=cfeature.COLORS["land"])
    dkwargs.update(**kwargs)
    if isinstance(land, bool) and land:
        land = "50m"
    if land in ["10m", "50m", "110m"]:
        land = cfeature.NaturalEarthFeature("physical", "land", land, **dkwargs)
        # ax.add_feature(cfeature.LAND)
        ax.add_feature(land)
    elif isinstance(land, str):
        shp = shapereader.Reader(land)
        for record, geometry in zip(shp.records(), shp.geometries()):
            ax.add_geometries([geometry], crs, zorder=0, **dkwargs)


def _plot_coastline(ax, coast, **kwargs):
    """plot coastline"""
    dkwargs = dict(edgecolor="black", facecolor=cfeature.COLORS["land"], zorder=5)
    dkwargs.update(**kwargs)
    if isinstance(coast, bool) and coast:
        coast = "50m"
    if coast in ["10m", "50m", "110m"]:
        ax.coastlines(resolution=coast, color="k")
    # elif coast in ["auto", "coarse", "low", "intermediate", "high", "full"]:
    elif coast in ["c", "l", "i", "h", "f"]:
        # ["coarse", "low", "intermediate", "high", "full"]
        shpfile = shapereader.gshhs(coast)
        shp = shapereader.Reader(shpfile)
        ax.add_geometries(shp.geometries(), crs, **dkwargs)
    elif isinstance(coast, str):
        # for production, see: /Users/aponte/Data/coastlines/log
        shp = shapereader.Reader(coast)
        for record, geometry in zip(shp.records(), shp.geometries()):
            ax.add_geometries([geometry], crs, **dkwargs)


def _plot_rivers(ax, rivers, **kwargs):
    """plot rivers"""
    dkwargs = dict(facecolor="blue", edgecolor="blue", zorder=6)
    dkwargs.update(**kwargs)
    if isinstance(rivers, bool) and rivers:
        rivers = "50m"
    if rivers in ["10m", "50m", "110m"]:
        rivers = cfeature.NaturalEarthFeature(
            "physical", "rivers_lake_centerlines", rivers, **dkwargs
        )
        ax.add_feature(cfeature.RIVERS)
    elif isinstance(rivers, str):
        shp = shapereader.Reader(rivers)
        for record, geometry in zip(shp.records(), shp.geometries()):
            ax.add_geometries([geometry], crs, **dkwargs)


# ------------------------------ bathymetry -----------------------------

# etopo1
_bathy_etopo1 = os.path.join(
    os.getenv("HOME"),
    "Data/bathy/etopo1/zarr/ETOPO1_Ice_g_gmt4.zarr",
)


def load_bathy(bathy, bounds=None, steps=None, land=False):
    """Load bathymetry

    Parameters
    ----------
    bathy: str
        "etopo1" or filepath to bathymetric file
    bounds: list, tuple, optional
        Bounds to be selected (lon_min, lon_max, lat_min, lat_max)
    steps: list, tuple, optional
        subsampling steps (di_lon, di_lat)
    """
    if bathy == "etopo1":
        ds = xr.open_dataset(_bathy_etopo1)
        # ds = ds.rename({'x': 'lon', 'y': 'lat', 'z': 'elevation'})
        ds = ds.rename({"z": "elevation"})
        if bounds is None and steps is None:
            steps = (4, 4)
    else:
        ds = xr.open_dataset(bathy)

    if "depth" not in ds and "elevation" in ds:
        ds["depth"] = -ds.elevation

    # mask land
    if not land:
        ds["depth"] = ds["depth"].where(ds["depth"] > 0)

    if "lon" in ds.dims:
        ds = ds.rename(lon="longitude")
    if "lat" in ds.dims:
        ds = ds.rename(lat="latitude")

    assert ("longitude" in ds.dims) and (
        "latitude" in ds.dims
    ), f"lon, lat must be in bathymetric dataset, this not the case in {bathy}"

    if steps is not None:
        ds = ds.isel(
            longitude=slice(0, None, steps[0]),
            latitude=slice(0, None, steps[1]),
        )

    if bounds is not None:
        ds = ds.sel(
            longitude=slice(bounds[0], bounds[1]),
            latitude=slice(bounds[2], bounds[3]),
        )

    return ds


def plot_bathy(
    fac,
    levels=[-6000.0, -4000.0, -2000.0, -1000.0, -500.0, -200.0, -100.0],
    clabel=True,
    bathy="etopo1",
    steps=None,
    bounds=None,
    **kwargs,
):
    fig, ax, crs = fac
    if isinstance(levels, tuple):
        levels = np.arange(*levels)
    # print(levels)
    ds = load_bathy(bathy, bounds=bounds, steps=steps)
    cs = ax.contour(
        ds.lon,
        ds.lat,
        ds.elevation,
        levels,
        linestyles="-",
        colors="black",
        linewidths=0.5,
    )
    if clabel:
        plt.clabel(cs, cs.levels, inline=True, fmt="%.0f", fontsize=9)


def store_bathy_contours(
    bathy,
    contour_file="contours.geojson",
    levels=[0, 100, 500, 1000, 2000, 3000],
    **kwargs,
):
    """Store bathymetric contours as a geojson
    The geojson may be used for folium plots
    """

    # Create contour data lon_range, lat_range, Z
    depth = load_bathy(bathy, **kwargs)["elevation"]
    if isinstance(levels, tuple):
        levels = np.arange(*levels)
    contours = depth.plot.contour(levels=levels, cmap="gray_r")

    # Convert matplotlib contour to geojson
    from geojsoncontour import contour_to_geojson

    contours_geojson = contour_to_geojson(
        contour=contours,
        geojson_filepath=contour_file,
        ndigits=3,
        unit="m",
    )


def load_bathy_contours(contour_file):
    """load bathymetric contours as geojson"""
    import geojson

    with open(contour_file, "r") as f:
        contours = geojson.load(f)
    return contours
