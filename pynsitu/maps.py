import os

import xarray as xr

from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cmocean.cm as cm

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.io import shapereader
    crs = ccrs.PlateCarree()
except:
    print("Warning: could not import cartopy")

default_resolution = "10m"

def plot_map_tmp(
    fig=None,
    region=None,
    coast="110m",
    land="110m",
    rivers="110m",
    figsize=(10, 10),
    bounds=None,
    cp=None,
    grid_linewidth=1,
    **kwargs,
):
    """Plot a map of the campaign area

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

    if fig is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig.clf()

    if bounds is None:
        if cp is not None:
            bounds = cp.bounds
        else:
            assert False, "bounds need to be provided somehow"

    ax = fig.add_subplot(111, projection=crs)
    ax.set_extent(bounds, crs=crs)
    gl = ax.gridlines(
        crs=crs,
        draw_labels=True,
        linewidth=grid_linewidth,
        color="k",
        alpha=0.5,
        linestyle="--",
    )
    gl.xlabels_top = False
    gl.ylabels_right = False

    # overrides kwargs for regions
    if region:
        coast = region
        land = region
        rivers = region

    #
    _coast_root = os.getenv("HOME") + "/data/coastlines/"
    coast_kwargs = dict(edgecolor="black", facecolor=cfeature.COLORS["land"], zorder=-1)
    if coast in ["10m", "50m", "110m"]:
        ax.coastlines(resolution=coast, color="k")
    elif coast in ["auto", "coarse", "low", "intermediate", "high", "full"]:
        shpfile = shapereader.gshhs("h")
        shp = shapereader.Reader(shpfile)
        ax.add_geometries(shp.geometries(), crs, **coast_kwargs)
    elif coast == "bseine":
        # for production, see: /Users/aponte/Data/coastlines/log
        shp = shapereader.Reader(_coast_root + "baie_de_seine/bseine.shp")
        for record, geometry in zip(shp.records(), shp.geometries()):
            ax.add_geometries([geometry], crs, **coast_kwargs)
    elif coast == "med":
        # for production, see: /Users/aponte/Data/coastlines/log
        shp = shapereader.Reader(_coast_root + "med/med_coast.shp")
        for record, geometry in zip(shp.records(), shp.geometries()):
            ax.add_geometries([geometry], crs, **coast_kwargs)
    elif coast == "med_high":
        # for production, see: /Users/aponte/Data/coastlines/log
        shp = shapereader.Reader(_coast_root + "/med/med_high_coast.shp")
        for record, geometry in zip(shp.records(), shp.geometries()):
            ax.add_geometries([geometry], crs, **coast_kwargs)

    #
    _land_kwargs = dict(edgecolor="face", facecolor=cfeature.COLORS["land"])
    if land in ["10m", "50m", "110m"]:
        land = cfeature.NaturalEarthFeature("physical", "land", land, **_land_kwargs)
        # ax.add_feature(cfeature.LAND)
        ax.add_feature(land)
    elif land == "bseine":
        shp = shapereader.Reader(_coast_root + "baie_de_seine/bseine_land.shp")
        for record, geometry in zip(shp.records(), shp.geometries()):
            ax.add_geometries([geometry], crs, **_land_kwargs)

    #
    _rivers_kwargs = dict(facecolor="none", edgecolor="blue")  # , zorder=-1
    if rivers in ["10m", "50m", "110m"]:
        rivers = cfeature.NaturalEarthFeature(
            "physical", "rivers_lake_centerlines", rivers, **_rivers_kwargs
        )
        ax.add_feature(cfeature.RIVERS)
    elif rivers == "bseine":
        shp = shapereader.Reader(_coast_root + "baie_de_seine/bseine_rivers.shp")
        for record, geometry in zip(shp.records(), shp.geometries()):
            ax.add_geometries([geometry], crs, **_rivers_kwargs)
        shp = shapereader.Reader(_coast_root + "baie_de_seine/bseine_water.shp")
        for record, geometry in zip(shp.records(), shp.geometries()):
            ax.add_geometries([geometry], crs, facecolor="blue", edgecolor="none")

    # need to perform once more
    ax.set_extent(bounds, crs=crs)

    return [fig, ax, crs]


def plot_map(
    da=None,
    extent="global",
    projection=None,
    title=None,
    fig=None,
    ax=None,
    colorbar=True,
    colorbar_kwargs={},
    center_colormap=False,
    gridlines=True,
    dticks=(1, 1),
    bathy=False,
    bathy_levels=None,
    land=True,
    coast_resolution=None,
    offline=False,
    figsize=0,
    **kwargs,
):

    #
    if figsize == 0:
        _figsize = (10, 5)
    elif figsize == 1:
        _figsize = (20, 10)
    else:
        _figsize = figsize
    if fig is None:
        fig = plt.figure(figsize=_figsize)
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

    # copy kwargs for update
    kwargs = kwargs.copy()

    if center_colormap and da is not None:
        vmax = float(abs(da).max())
        vmin = -vmax
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax

    if bathy:
        da = load_bathy(bathy)["depth"]
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

    set_extent = False
    if isinstance(extent, list) or isinstance(extent, tuple):
        set_extent = True

    # coastlines and land:
    if land:
        dland = dict(
            scale=default_resolution,
            edgecolor="face",
            facecolor=cfeature.COLORS["land"],
        )
        if isinstance(land, dict):
            dland.update(**land)
            # land = dict(args=['physical', 'land', '10m'],
            #            kwargs= dict(edgecolor='face', facecolor=cfeature.COLORS['land']),
            #           )
        land_feature = cfeature.NaturalEarthFeature("physical", "land", **dland)
        # else:
        #    land_feature = cfeature.LAND
        ax.add_feature(land_feature, zorder=0)
    if coast_resolution is not None:
        ax.coastlines(resolution=coast_resolution, color="k")

    if set_extent:
        ax.set_extent(extent)

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
    with open(contour_file, "r") as f:
        contours = geojson.load(f)
    return contours
