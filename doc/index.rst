.. pynsitu documentation master file, created by
   from xrft doc on 29/10/2022


pynsitu: Oceanographic In Situ Data analysis
==============================================

**pynsitu** is a Python package for the analysis of in situ oceanographic
data based on pandas_ and xarray_ arrays.
It is:

- **Powerful**: It keeps the metadata and coordinates of the original xarray dataset and provides a clean work flow of DFT.
- **Easy-to-use**: It uses the native arguments of numpy FFT and provides a simple, high-level API.
- **Fast**: It uses the dask API of FFT and map_blocks to allow parallelization of DFT.

.. note::

    some note about current status:
    ... is at early stage of development and will keep improving in the future.
    The discrete Fourier transform API should be quite stable,
    but minor utilities could change in the next version.
    If you find any bugs or would like to request any enhancements,
    please `raise an issue on GitHub <https://github.com/apatlpo/pynsitu/issues>`_.

Documentation
-------------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   why-pynsitu
   limitations
   installation
   contributor_guide

.. toctree::
   :maxdepth: 1
   :caption: Examples

   DFT-iDFT_example
   Parseval_example
   chunk_example
   MITgcm_example

.. toctree::
   :maxdepth: 1
   :caption: Help & reference

   whats-new
   api


.. _pandas: https://pandas.pydata.org
.. _xarray: http://xarray.pydata.org
.. _dask: http://dask.pydata.org/en/latest/array-api.html
