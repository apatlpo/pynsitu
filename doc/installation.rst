.. _installation-label:

Installation
============

The quickest way
----------------

pynsitu is compatible both with Python 3. The major dependencies are xarray_ and pandas_.

Install dependencies via conda
-----------------------------

Advised way to install the required environment is via Anaconda_::

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
