.. multimodars documentation master file, created by
   sphinx-quickstart on Wed Jul 23 22:38:42 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to multimodars documentation!
=====================================

*multimodars: A Rust-powered cardiac multi-image modality fusion package.*

.. warning::
   
   Not intended for clinical use.

This package aims to combine different cardiac imaging modalities to combined 3D models. A particular focus is on the fusion of intravascular 
imaging (either intravascular ultrasound [IVUS] or optical coherence tomography [OCT]) with a 3D geometry created from computed coronary tomography angiography (CCTA).

The main functionalities are organized into two major categories:

1. **Intravascular Alignment**: This category addresses the co-registration of intravascular frames through translation and rotation to minimize the distance between adjacent frames.
   This approach is particularly useful for correcting misalignments introduced by cardiac motion during intravascular image acquisition. It further encompasses the alignment of
   separate intravascular pullbacks (e.g., acquired under rest and stress conditions) as well as the co-registration of systolic and diastolic frames within the same pullback.

2. **CCTA-Intravascular Fusion**: This category covers the fusion of intravascular frames with a three-dimensional geometry reconstructed from coronary computed tomography angiography (CCTA).
   This is particularly valuable for generating high-resolution three-dimensional models of coronary arteries by combining the high spatial resolution of intravascular imaging with the
   three-dimensional geometry provided by CCTA. Functionalities in this category include the anatomical labeling of CCTA geometries based on centerlines, the registration of intravascular
   frames onto a centerline, and the subsequent fusion of the resulting geometry with the CCTA reconstruction.

Intravascular image data are accepted either as CSV files following the output format of
`AIVUS <https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA>`_, an open-source tool for automated intravascular ultrasound analysis,
or directly as NumPy arrays, affording greater flexibility for integration with custom pipelines.

Table of Contents
-----------------

.. toctree::
   :hidden:

   Home <self>

.. toctree::
   :maxdepth: 2

   installation
   Intravascular Alignment Tutorial <tutorial_intravascular>
   CCTA Intravascular Fusion Tutorial <tutorial_ccta>
   Intravascular Notebook <notebooks/intravascular_notebook>
   CCTA Notebook <notebooks/ccta_notebook>
   Benchmarks <benchmark>
   api/index
   changelog

License
-------

This package is covered by the open source `MIT License <https://github.com/yungselm/multimoda-rs/blob/main/LICENSE>`_.

Developers
----------

 - `Anselm Stark <https://github.com/yugnselm>`_:sup:`1,2`
 - `Marc Ilic <https://github.com/cicram>`_:sup:`1,2`
 - `Ali Mokthari <https://github.com/alimokh91>`_:sup:`1,2`
 - `Pooya Mohammadi Kazaj <https://github.com/pooya-mohammadi>`_:sup:`1,2`
 - `Isaac Shiri <https://github.com/Isaacshiri>`_:sup:`1`

:sup:`1`\ Department of Cardiology, Inselspital, Bern University Hospital, University of Bern, Switzerland
:sup:`2`\ Graduate School for Cellular and Biomedical Sciences, University of Bern, Bern, Switzerland

Contributing
------------
We'd welcome your contributions to multimodars. Please read the
`contributing guidelines <https://github.com/yungselm/multimoda-rs/blob/main/CONTRIBUTING.md>`_ on how to contribute to ``multimodars``.

Changelog
---------

* :doc:`changelog`

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`