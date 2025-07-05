.. _installation-label:

Installation
============

The quickest way
----------------

pynsitu is compatible both with Python 3.

Install with pip
----------------
To install latest release via pip, run in an adequate python environment::

    $ python -m pip install pynsitu

Create conda environment and install latest code
------------------------------------------------
First clone the pynsitu repository and create a conda environment::

    $ git clone https://github.com/apatlpo/pynsitu.git
    $ cd pynsitu
    $ conda create -n insitu python=3.10
    $ conda env update -n insitu -f ci/environment.yml
    $ conda activate insitu

Then install pynsitu library in the environment::
    
    $ python -m pip install pynsitu
