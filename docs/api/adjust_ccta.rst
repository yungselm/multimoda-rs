Adjust CCTA
===========

Labeling
--------

.. autofunction:: multimodars.ccta.labeling.label_geometry
.. autofunction:: multimodars.ccta.labeling.label_anomalous_region

Scaling and morphing
--------------------

.. autofunction:: multimodars.ccta.scaling.scale_region_centerline_morphing
.. autofunction:: multimodars.ccta.scaling.find_distal_and_proximal_scaling
.. autofunction:: multimodars.ccta.scaling.find_aorta_scaling
.. autofunction:: multimodars.ccta.scaling.find_aortic_wall_scaling
.. autofunction:: multimodars.ccta.scaling.sync_results_to_mesh

Trimming and stitching
----------------------

.. autofunction:: multimodars.ccta.mesh_regions.remove_labeled_points_from_mesh
.. autofunction:: multimodars.ccta.mesh_regions.keep_labeled_points_from_mesh
.. autofunction:: multimodars.ccta.stitching.stitch_ccta_to_intravascular
.. autofunction:: multimodars.ccta.stitching.order_points_list

Boundary rings
--------------

Open-boundary extraction is shared by the trimming and debug-plot paths.  The two entry points
differ in whether they may modify the mesh: :func:`~multimodars.ccta.stitching.clean_open_boundary`
is used while trimming and reports rim vertices to delete, whereas
:func:`~multimodars.ccta.stitching.order_boundary_rings` is read-only and reports the rings a mesh
actually has.

.. autofunction:: multimodars.ccta.stitching.clean_open_boundary
.. autofunction:: multimodars.ccta.stitching.order_boundary_rings
.. autofunction:: multimodars.ccta.stitching.open_boundary_edges

Debug plots
-----------

.. autofunction:: multimodars.ccta.debug_plots.plot_boundary_edges
.. autofunction:: multimodars.ccta.debug_plots.compare_centerline_scaling
