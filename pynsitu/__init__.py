#
__all__ = ["events",
           "campaign",
           "geo",
           "maps",
           "seawater",
           "tseries",
           "get_cmap_colors",
           ]

# various parameters
from .geo import deg2rad, rad2deg, g, deg2m

from . import events
from .events import campaign
from . import geo
from . import maps
from . import seawater
from . import tseries

import numpy as np

# misc plotting
import matplotlib.colors as colors
import matplotlib.cm as cmx

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

# utils for vector conversions

def uv2speedheading(u, v):
    """ converts eastward and northward velocities into speed and heading
    Atmospheric conventions

    Parameters
    ----------
    u, v: velocity components

    Returns
    -------
    speed
    heading: in degrees
    """
    return np.sqrt(u**2+v**2), ((np.arctan2(-u,-v))%(2*np.pi))*rad2deg

def speedheading2uv(speed, heading):
    """ converts speed and heading to eastward and northward velocities
    Atmospheric conventions

    Parameters
    ----------
    speed
    heading: in degrees
    """
    return speed*np.sin(heading*deg2rad-np.pi), speed*np.cos(heading*deg2rad-np.pi)

#from .events import *
#from .geo import *
#from .seawater import *
#import .events.py
#import geo
#import seawater
#import maps
