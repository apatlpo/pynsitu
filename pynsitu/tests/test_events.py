# import numpy as np
import pandas as pd

import pytest
import numpy.testing as npt
# import xarray.testing as xrt

import pynsitu as pyn

# It may not be critical to test low level code ...
# one should test campaign objects for example


def test_events_instantiation():
    """test creation of events objects"""

    e = pyn.events.event(
        label="label", logline="02/09/2016 05:35:00 7 17.124 43 19.866"
    )
    assert e.label == "label"
    assert e.time == pd.to_datetime("02/09/2016 05:35:00")
    assert e.lon == 7 + 17.124 / 60
    assert e.lat == 43 + 19.866 / 60

    e = pyn.events.event(label="label", logline="02/09/2016 05:35:00 7.124 43.866")
    assert e.lon == 7.124
    assert e.lat == 43.866

    e = pyn.events.event(label="label", logline="02/09/2016 05:35:00")
    assert e.lon is None
    assert e.lat is None


def test_events_str():
    """test conversion to string"""
    e = pyn.events.event(
        label="label", logline="02/09/2016 05:35:00 7 17.124 43 19.866"
    )
    assert str(e) == "label 2016-02-09 05:35:00 7.29 43.33"
