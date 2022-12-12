[![CI](https://github.com/apatlpo/pynsitu/actions/workflows/ci.yaml/badge.svg)](https://github.com/apatlpo/pynsitu/actions/workflows/ci.yaml)
[![code-style](https://github.com/apatlpo/pynsitu/actions/workflows/linting.yaml/badge.svg)](https://github.com/apatlpo/pynsitu/actions/workflows/linting.yaml)
[![Check and Maybe Release Python Package](https://github.com/apatlpo/pynsitu/actions/workflows/release.yaml/badge.svg)](https://github.com/apatlpo/pynsitu/actions/workflows/release.yaml)

# pynsitu: oceanographic insitu data toolbox

## install

At the moment, the unique method to install pynsitu, along with its dependencies, is to clone and manually install the adequate environment:

```
git clone https://github.com/apatlpo/pynsitu.git
cd pynsitu
conda create -n pynsitu python=3.10
conda env update -n pynsitu -f ci/environment.yml
#conda env create -n pynsitu -f ci/environment.yml
#mamba env create -n pynsitu -f ci/environment.yml
pip install -e .
```



## data model

pandas and/or xarray based with accessors

- lon/lat/depth, geodetic properties
- water properties

Requires:

- cartopy, bokeh
- pytide, pyTMD
