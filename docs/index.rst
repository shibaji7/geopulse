GeoPulse
========

.. image:: logo.png
   :alt: GeoPulse
   :width: 320px
   :align: center

|

**Unified engine for geomagnetically induced currents (GIC), induced
voltages, and total harmonic distortion across grounded infrastructure —
submarine cables, power grids, oil/gas pipelines, and electrified
railways — from a single, modular, extensible core.**

GeoPulse is a scientific-Python engine intended to accompany
peer-reviewed publications. Every stage carries uncertainty via the
:class:`~geopulse.uq.uncertain.Uncertain` type, and the surface
impedance ``Z`` is a single polymorphic abstraction spanning 1-D scalar,
2-D tensor, and (planned) 3-D kernel representations.

Highlights
----------

* **100 passing tests**, reproduces Horton (2012) Table VII and
  Boteler (1997) DSTL to sub-1 % relative error.
* Six-dependency core (``numpy``, ``scipy``, ``matplotlib``, ``h5py``,
  ``pyyaml``, ``loguru``). Everything else is behind optional extras.
* Modular ABC design (:mod:`~geopulse.sources`,
  :mod:`~geopulse.earth`, :mod:`~geopulse.network`,
  :mod:`~geopulse.solver`, :mod:`~geopulse.devices`) so third-party
  plugins can slot in via Python entry points.
* Fully typed (``py.typed`` marker); ``mypy`` clean.
* Schema-versioned HDF5 I/O for reproducible archives.

.. toctree::
   :maxdepth: 2
   :caption: Start here

   getting_started

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

.. toctree::
   :maxdepth: 1
   :caption: Project

   contributing
   changelog

Citation
--------

If you use GeoPulse in your research, please cite:

    Chakraborty, S. & Boteler, D. (in preparation). *GeoPulse: A unified
    engine for geomagnetically induced currents across grounded
    infrastructure*.

The engine is Apache-2.0 licensed. The explicit patent grant is
deliberate — GeoPulse is intended for use by industry partners as well as
academics.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
