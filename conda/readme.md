## new notes for building and pushing package to pypi

References:

- Create pypi package: [Pypi doc](https://packaging.python.org/en/latest/tutorials/packaging-projects/)

- [conda-forge doc](https://conda-forge.org/docs/maintainer/adding_pkgs/)

- [pyOpenSci doc](https://github.com/pyOpenSci/pyosPackage/blob/main/pyproject.toml)


0. Make sure modifications are final: version has been updated and code is formated with black:
   -  version number updated across all files: `doc/conf.py`, `pyproject.toml`
   -  info appended to `doc/whats-new.rst`
   -  make sure all tests pass by running in adequate environment: `pytest tests` see contributor guide

1. Create a conda environment to build

```
conda create -n pypi pip
conda activate pypi
pip install "black[jupyter]"
python3 -m pip install --upgrade build
python3 -m pip install --upgrade twine
```

2. Package files

```
python3 -m build
```

3. Push to TestPyPI for testing:

```
# push to test
python3 -m twine upload --repository testpypi dist/*
```

4. Upload to pyPI (note that this is definitive):
   
```
python3 -m twine upload dist/*
```

To create a simple python environment for testing:
```
conda create -n test python=3.11
conda activate test
# inside pynsitu dir:
pip install -e .
```
Such test may be useful to check dependencies are correctly installed.


```
conda create --name grayskull
conda activate grayskull
conda install -c conda-forge grayskull
cd tmp
grayskull pypi --strict-conda-forge pynsitu
cd ..
git clone https://github.com/apatlpo/staged-recipes.git
cd staged-recipes/recipes
mkdir pynsitu
cp ????meta.yaml pynsitu

curl -sL https://github.com/apatlpo/pynsitu/archive/v0.0.2.tar.gz | openssl sha256
```

<!---
---

## old notes


Sylvie's gists in order to create pypi and conda packages are found [here](https://gist.github.com/slgentil)

In order to release a new version of the library:

- update tag in `conda/meta.yaml`, `conda/convert_upload.sh`, `doc/conf.py`
- if need be, update python versions in `setup.cfg`, `conda/conda_build_config.yaml`, `conda/convert_upload.sh`, `github/ci.yaml`
- install libraries required to compile and export packages in `base` environment:

```
conda activate base
conda install conda-build conda-verify anaconda-client
```

- run in library root dir (`pynsitu/`):

``` 
conda build -c pyviz -c conda-forge -c apatlpo --output-folder ${HOME}/Code/wheels/  ./conda
```

- run `convert_upload.sh` to produce and upload packages

- create release on github
-->
