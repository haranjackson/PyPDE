.. PyPDE documentation master file, created by
   sphinx-quickstart on Tue Nov 26 14:42:02 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyPDE
=====

A Python library for solving any system of hyperbolic or parabolic Partial
Differential Equations. The PDEs can have stiff source terms and
non-conservative components.

Key Features:

* Any first or second order system of PDEs
* Your fluxes and sources are written in Python for ease
* Any number of spatial dimensions
* Arbitrary order of accuracy
* C++ under the hood for speed
* Based on the ADER-WENO method

Please feel free to message me with questions/suggestions:
jackson.haran@gmail.com

**Quickstart**: check out the :doc:`core functionality<pages/core_functionality>`
and :doc:`example code<pages/example_code>`.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   pages/installation
   pages/background
   pages/core_functionality
   pages/example_pdes
   pages/example_code
