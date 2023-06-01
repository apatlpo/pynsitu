.. _installation-label:

Installation
============

The quickest way
----------------

pynsitu is compatible both with Python 3. The major dependencies are xarray_ and pandas_.

Install dependencies via conda
-----------------------------

Directly from Anaconda_ (along with pynsitu, recommended)::

    $ conda install pynsitu -c apatlpo

Or directly in a dedicated environment:

    $ conda create -n myenv -c apatlpo python=3.10 pynsitu

Or more manually (without pynsitu, see next section))::

    $ git clone https://github.com/apatlpo/pynsitu.git
    $ cd pynsitu
    $ conda create -n pynsitu python=3.10
    $ conda env update -n pynsitu -f ci/environment.yml

Install pynsitu from GitHub repo
-----------------------------
To get the latest version::

    $ git clone https://github.com/apatlpo/pynsitu.git
    $ cd pynsitu
    $ python setup.py install .

Developers can track source code changes by::

    $ git clone https://github.com/apatlpo/pynsitu.git
    $ cd pynsitu
    $ python setup.py develop .

.. _xarray: http://xarray.pydata.org
.. _pandas: https://pandas.pydata.org
.. _Anaconda: https://www.continuum.io/downloads
