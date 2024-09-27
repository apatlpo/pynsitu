
Sylvie's gists in order to create pypi and conda packages are found [here](https://gist.github.com/slgentil)

In order to release a new version of the library:

- update tag in `conda/meta.yaml`, `conda/convert_upload.sh`, `doc/conf.py`
- if need be, update python versions in `setup.cfg`, `conda/conda_build_config.yaml`, `conda/convert_upload.sh`, `github/ci.yaml`
- install `conda-build` in `base` environment:

```
conda activate base
conda install conda-build anaconda-client
```

- run in library root dir (`pynsitu/`):

``` 
conda build -c pyviz -c conda-forge -c apatlpo --output-folder ${HOME}/Code/wheels/  ./conda
```

- run `convert_upload.sh` to produce and upload packages

- create release on github