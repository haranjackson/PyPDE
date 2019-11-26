=====
PyPDE
=====

A Python library for solving any system of hyperbolic or parabolic Partial Differential Equations.
The PDEs can have stiff source terms and non-conservative components.

Key Features:

* Any first or second order system of PDEs
* Your fluxes and sources are written in Python for ease
* Any number of spatial dimensions
* Arbitrary order of accuracy
* C++ under the hood for speed
* Based on the ADER-WENO method

Please feel free to message me with questions/suggestions:
jackson.haran@gmail.com

Installation
------------

``pip install pypde``

Background
----------

We can solve any system of PDEs of the form:

.. image:: https://github.com/haranjackson/PyPDE/raw/master/images/SystemExpanded.png
   :width: 480px
   :alt: Godunov-Romenski equations
   :align: center

or, more succinctly:

.. image:: https://github.com/haranjackson/PyPDE/raw/master/images/System.png
   :height: 40px
   :alt: Godunov-Romenski equations
   :align: center

Note that this includes the Navier-Stokes equations:

.. image:: https://github.com/haranjackson/PyPDE/raw/master/images/NavierStokes.png
   :width: 480px
   :alt: Godunov-Romenski equations
   :align: center

where:

.. image:: https://github.com/haranjackson/PyPDE/raw/master/images/TotalStressNS.png
   :height: 40px
   :alt: Godunov-Romenski equations
   :align: center

and the Reactive Euler equations:

.. image:: https://github.com/haranjackson/PyPDE/raw/master/images/ReactiveEuler.png
   :width: 480px
   :alt: Godunov-Romenski equations
   :align: center

where ``K`` is a (potentially large) function depending on temperature ``T``.

It also includes the the Godunov-Romenski equations:

.. image:: https://github.com/haranjackson/PyPDE/raw/master/images/GodunovRomenski.png
   :width: 480px
   :alt: Godunov-Romenski equations
   :align: center

where now θ is a (potentially very small) function of A, and:

.. image:: https://github.com/haranjackson/PyPDE/raw/master/images/TotalStressGR.png
   :height: 40px
   :alt: Godunov-Romenski equations
   :align: center

Usage
-----

// TODO