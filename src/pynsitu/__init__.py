#
__all__ = [
    "events",
    "campaign",
    "drifters",
    "geo",
    "maps",
    "seawater",
    "tseries",
    "tides",
    "movies",
    "get_cmap_colors",
]

# deal with config file
try:
    import os

    _config_dir = os.path.expanduser("~/.config/pynsitu/")
    _config_file = os.path.join(_config_dir, "pynsitu.yaml")
    # create a config directory if necessary
    if not os.path.isdir(_config_dir):
        os.mkdir(_config_dir)
    # copy config file if need be
    if not os.path.isfile(_config_file):
        import importlib.resources as importlib_resources

        pyn_path = importlib_resources.files("pynsitu")
        import shutil

        shutil.copy(os.path.join(pyn_path, "pynsitu.yaml"), _config_file)
    # load config
    import yaml

    with open(_config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
except:
    config = dict()

# useful parameters
from .geo import deg2rad, rad2deg, g, deg2m

#
from pandas import Timedelta

day = Timedelta("1d")
hour = Timedelta("1h")
minute = Timedelta("1min")
second = Timedelta("1s")
#
knot = 0.514
nautical_mile = 1852  # meters


from . import events
from .events import Campaign
from . import drifters
from . import geo
from . import maps
from . import seawater
from . import tseries
from . import tides
from . import movies

# ------------------------ general utilities -----------------------------

import numpy as np

# misc plotting
import matplotlib.colors as colors
import matplotlib.cm as cmx

# colormaps shortcuts
if maps.cm is not None:
    thermal = maps.cm.thermal
    haline = maps.cm.haline


def get_cmap_colors(Nc, cmap="plasma"):
    """load colors from a colormap to plot lines

    Parameters
    ----------
    Nc: int
        Number of colors to select
    cmap: str, optional
        Colormap to pick color from (default: 'plasma')
    """
    scalarMap = cmx.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=Nc), cmap=cmap)
    return [scalarMap.to_rgba(i) for i in range(Nc)]


# utils for vector manipulation and conversions


def rotate(theta, u, v):
    """rotate by angle theta (rad)"""
    cos, sin = np.cos(theta), np.sin(theta)
    return u * cos + v * sin, -u * sin + v * cos


def uv2speedheading(u, v):
    """converts eastward and northward velocities into speed and heading
    Atmospheric conventions

    Parameters
    ----------
    u, v: velocity components

    Returns
    -------
    speed
    heading: in degrees
    """
    return np.sqrt(u**2 + v**2), ((np.arctan2(-u, -v)) % (2 * np.pi)) * rad2deg


def speedheading2uv(speed, heading):
    """converts speed and heading to eastward and northward velocities
    Atmospheric conventions

    Parameters
    ----------
    speed
    heading: in degrees
    """
    return speed * np.sin(heading * deg2rad - np.pi), speed * np.cos(
        heading * deg2rad - np.pi
    )
