# import numpy as np
import pandas as pd

import pytest
import numpy.testing as npt

# import xarray.testing as xrt

import pynsitu as pyn

# It may not be critical to test low level code ...
# one should test campaign objects for example

## events class


def test_events_instantiation():
    """test creation of events objects"""

    e = pyn.events.Event(
        label="label", logline="02/09/2016 05:35:00 7 17.124 43 19.866"
    )
    assert e.label == "label"
    assert e.time == pd.to_datetime("02/09/2016 05:35:00")
    assert e.lon == 7 + 17.124 / 60
    assert e.lat == 43 + 19.866 / 60

    e = pyn.events.Event(label="label", logline="02/09/2016 05:35:00 7.124 43.866")
    assert e.lon == 7.124
    assert e.lat == 43.866

    e = pyn.events.Event(label="label", logline="02/09/2016 05:35:00")
    assert e.lon is None
    assert e.lat is None


def test_events_str():
    """test conversion to string"""
    e = pyn.events.Event(
        label="label", logline="02/09/2016 05:35:00 7 17.124 43 19.866"
    )
    assert str(e) == "label 2016-02-09 05:35:00 7.29 43.33"


## deployment


def test_deployment_instantiation():
    """test creation of deployment objects"""

    _start = "02/09/2016 05:35:00 7 17.124 43 19.866"
    _end = "02/09/2016 05:35:00 7 17.124 43 19.866"
    _meta = dict(color="k", info="toto")

    d = pyn.events.Deployment(
        "label",
        start=_start,
        end=_end,
        meta=_meta,
    )
    assert d.label == "label"
    assert d["label"] == "label"
    assert d.start.time == pd.to_datetime("02/09/2016 05:35:00")
    assert d.start.lon == 7 + 17.124 / 60
    assert d.start.lat == 43 + 19.866 / 60

    _loglines = [
        _start,
        _end,
    ]
    d = pyn.events.Deployment(
        "label",
        loglines=_loglines,
    )

    _loglines = [_start, _end, _meta]
    d = pyn.events.Deployment(
        "label",
        loglines=_loglines,
    )


## campaign class


def test_campaign_instantiation():
    """test creation of campaign object"""

    yaml = "pynsitu/tests/campaign.yaml"
    cp = pyn.Campaign(yaml)

    # str
    print(cp)

    # common deployments
    print(cp["underway"])

    # platforms
    print(cp["tide_gauge"])

    for uname, u in cp.items():
        print(uname, type(u), u)

    # plot time line
    cp.timeline()

    # test various features of campaign class
    # units = [u for u in cp]
    # assert units[0]=="underway"


if __name__ == "__main__":
    test_campaign_instantiation()
