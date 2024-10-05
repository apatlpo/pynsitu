#!/bin/bash

# install first some packages in base:
#   conda activate base
#   conda install conda-build anaconda-client

# the package needs to be built once first:
# conda build -c pyviz -c conda-forge -c apatlpo --output-folder ${HOME}/Code/wheels/  ./conda

# launch the script after anaconda login
library="pynsitu"
tag="0.0.2"
#in_directory="${HOME}/.miniconda3/envs/condabuild/conda-bld/osx-64/"
in_directory="${HOME}/Code/wheels/osx-64/"
out_directory="${HOME}/Code/wheels/"

### all python and platforms

for python in 38 39 310
do
    # starting platform
    platform="osx-64"
    echo $platform $python
    anaconda upload --force ${in_directory}/${library}-${tag}-py${python}_0.tar.bz2
    # other platforms
    for platform in  win-64 linux-64
    do 
        echo $platform $python
        #ls ${in_directory}pynsitu*
        conda convert -f -p $platform -o $out_directory ${in_directory}${library}-${tag}-py${python}_0.tar.bz2
        anaconda upload --force ${out_directory}${platform}/${library}-${tag}-py${python}_0.tar.bz2
    done
done

### single python and platform
#python="39"
#platform="osx-64"
#conda convert -f -p $platform -o $out_directory ${in_directory}${library}-${tag}-py${python}_0.tar.bz2
#anaconda upload ${out_directory}${platform}/${library}-${tag}-py${python}_0.tar.bz2
