import xarray as xr

from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cmocean.cm as cm

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.geodesic as cgeo
crs = ccrs.PlateCarree()

default_resolution = "10m"

def plot_map(da=None,
            extent=0,
            bathy=False,
            bathy_levels=None,
            title=None,
            fig=None,
            ax=None,
            colorbar=True,
            colorbar_kwargs={},
            center_colormap=False,
            gridlines=True,
            dticks=(1, 1),
            land=True,
            coast_resolution=None,
            offline=False,
            figsize=0,
            **kwargs,
           ):

    #
    if figsize==0:
        _figsize = (10, 5)
    elif figsize==1:
        _figsize = (20, 10)
    else:
        _figsize = figsize
    if fig is None:
        fig = plt.figure(figsize=_figsize)
    if ax is None:
        ax = fig.add_subplot(111, projection=ccrs.Orthographic(5., 43.))

    # copy kwargs for update
    kwargs = kwargs.copy()

    if center_colormap and da is not None:
        vmax = float(abs(da).max())
        vmin = -vmax
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax

    if bathy:
        da = load_bathy(bathy)
        if bathy_levels is not None:
            CS = (da
                 .plot
                 .contour(x="longitude", y="latitude",
                          ax=ax,
                          transform=crs,
                          levels=bathy_levels,
                          colors="k",
                          )
                )
            ax.clabel(CS, CS.levels, inline=True, fontsize=10)
            da=None
        else:
            kwargs.update(cmap=cm.deep, vmin=0)

    if da is not None:
        im = (da
             .squeeze()
             .plot
             .pcolormesh(x="longitude", y="latitude",
                         ax=ax,
                         transform=crs,
                         add_colorbar=False,
                         **kwargs,
                        )
            )

    if isinstance(extent, list) or isinstance(extent, tuple):
        set_extent = True

    # coastlines and land:
    if land:
        dland = dict(scale=default_resolution,
                     edgecolor='face',
                     facecolor=cfeature.COLORS['land'])
        if isinstance(land, dict):
            dland.update(**land)
            #land = dict(args=['physical', 'land', '10m'],
            #            kwargs= dict(edgecolor='face', facecolor=cfeature.COLORS['land']),
            #           )
        land_feature = cfeature.NaturalEarthFeature('physical', 'land', **dland)
        #else:
        #    land_feature = cfeature.LAND
        ax.add_feature(land_feature,  zorder=0)
    if coast_resolution is not None:
        ax.coastlines(resolution=coast_resolution, color='k')

    if set_extent:
        ax.set_extent(extent)

    if da is not None and colorbar:
        #cbar = fig.colorbar(im, extend="neither", shrink=0.7, **colorbar_kwargs)
        axins = inset_axes(ax,
               width="5%",  # width = 5% of parent_bbox width
               height="100%",  # height : 50%
               loc='lower left',
               bbox_to_anchor=(1.05, 0., 1, 1),
               bbox_transform=ax.transAxes,
               borderpad=0,
               )
        #cbar = fig.colorbar(im, extend="neither", shrink=0.9,
        cbar = fig.colorbar(im,
                            extend="neither",
                            cax=axins,
                            **colorbar_kwargs)
    else:
        cbar = None

    if gridlines:
        gl = ax.gridlines(draw_labels=True, dms=False,
                     x_inline=False, y_inline=False,
                    )
        gl.right_labels=False
        gl.top_labels=False

    if title is not None:
        ax.set_title(title, fontdict={"fontsize": 12, }) #"fontweight": "bold"
    #
    return {"fig": fig, "ax": ax, "cbar": cbar}


def load_bathy(bathy, land=False):
    """ outputs bathymetry as a dataarray with longitude, latitude coordinates
    """

    if isinstance(bathy, str):
        # must be a netcdf file
        ds = xr.open_dataset(bathy)
        if "lon" in ds:
            ds = ds.rename(lon="longitude")
        if "lat" in ds:
            ds = ds.rename(lat="latitude")
        if "elevation" in ds:
            ds["depth"] = -ds.elevation
            ds = ds.drop_vars("elevation")
        da = ds.depth
    # mask land
    if not land:
        da = da.where(da>0)
    return da
