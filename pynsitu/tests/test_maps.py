from matplotlib.gridspec import GridSpec

# import numpy as np
# import pandas as pd

# import xarray as xr
# import cftime
# import dask.array as dsar

import pytest

import numpy.testing as npt

# import xarray.testing as xrt

import pynsitu as pyn


def test_plot_map():

    # most basic map plot
    fig, ax, _ = pyn.maps.plot_map()

    # specify extent
    fig, ax, _ = pyn.maps.plot_map(extent=[0, 10, 40, 50])

    # Gridspec
