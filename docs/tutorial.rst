Tutorial
========

This step-by-step tutorial demonstrates how to:

- Run the workflow from csv files
- Run the workflow by building geometries from numpy arrays
- Finetuning of alignment algorithms
- Reordering algorithm
- Alignment with a centerline
- Saving everything as .obj files
- Utility functions to link to numpy

1. Workflow csv files
^^^^^^^^^^^^^^^^^^^^^

After pip installing or locally building the package install it in the familiar way.

.. code-block:: python

    import multimodars as mm

To run the whole workflow from .csv files the following requirements have to be met.
Files should be named ``diastolic_contours.csv``, ``systolic_contours.csv``, 
``diastolic_reference_points.csv`` and ``systolic_reference_points.csv`` depending on the required analysis.
Every file should be structured in the following way (no headers):
+------------+----------+----------+----------+
| frame      | x        | y        | z        |
+------------+----------+----------+----------+
| 771        | 2.4862   | 6.7096   | 24.5370  |
| 771        | 2.5118   | 6.7017   | 24.5370  |
| 771        | 2.5370   | 6.6936   | 24.5370  |
+------------+----------+----------+----------+

To acquire meaningful measurement data, the coordinates should be provided in mm or another SI unit instead of pixel values.
Optionally a record file can be provided `combined_sorted_manual.csv`, which should have the following structure, here the first column should contain the desired frame order and measurement_1 
represent the thickness of the wall between aorta and coronary and measurement_2 for the thickness between pulmonary artery and coronary. This is based on the output of
the AIVUS-CAA software _a link: https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/:
+-----------------+---------------+---------------+---------------+---------------+
| frame           | position      |   phase       | measurement_1 | measurement_2 |
+-----------------+---------------+---------------+---------------+---------------+
| 18              |  23.99        |       D       |               |               |
+-----------------+---------------+---------------+---------------+---------------+
| 37              |  22.79        |       D       |               |     2.35      |
+-----------------+---------------+---------------+---------------+---------------+
| 212             |  21.59        |       D       |     1.38      |     2.34      |
+-----------------+---------------+---------------+---------------+---------------+
| 94              |  20.39        |       D       |     1.38      |     2.11      |
+-----------------+---------------+---------------+---------------+---------------+
|  ...            |    ...        |     ...       |     ...       |     ...       |
+-----------------+---------------+---------------+---------------+---------------+
| 47              |  18.78        |       S       |               |               |
+-----------------+---------------+---------------+---------------+---------------+

This simplifies the workflow, by just providing a directory to automatically process:

.. code-block:: python

    rest, stress, dia, sys, (rest_logs, stress_logs, dia_logs, sys_logs) = mm.from_file(
        mode="full",
        rest_input_path="rest_csv_files",
        stress_input_path="stress_csv_files",
        rest_output_path="output/rest",
        stress_output_path="output/stress",
    )

However the preferred more flexible array is from numpy arrays.

2. Workflow numpy arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here the geometry can be directly build from arrays with the same structure as before: 
+------------+----------+----------+----------+
| ...        |   ...    |   ...    |   ...    |
+------------+----------+----------+----------+
| 771        | 2.4862   |  6.7096  |  24.5370 |
+------------+----------+----------+----------+
| 771        | 2.5118   |  6.7017  |  24.5370 |
+------------+----------+----------+----------+
| 771        | 2.5370   |  6.6936  |  24.5370 |
+------------+----------+----------+----------+
| ...        |   ...    |   ...    |   ...    |
+------------+----------+----------+----------+
catheter and walls are optional. However it is not recommended to provide the catheter points directly, but rather the image center (in mm), radius of the catheter (e.g. 0.5mm for IVUS)
and number of points to represent the catheter. If no walls are provided a default wall with 1mm offset is created.

.. code-block:: python

    prestent = mm.numpy_to_geometry(
        contours_arr=contours,
        catheters_arr=np.array([]),
        walls_arr=np.array([]),
        reference_arr=references,
    )

    poststent = mm.numpy_to_geometry(
        contours_arr=contours,
        catheters_arr=np.array([]),
        walls_arr=np.array([]),
        reference_arr=references,
    )

    pair, logs = mm.from_array(
        mode="singlepair",
        geometry_dia=prestent,
        geometry_sys=poststent,
        step_rotation_deg=0.1,
        range_rotation_deg=30,
        image_center=(4.5, 4.5),
        radius=0.5,
        n_points=20,
        write_obj=True,
        output_path="output/stent_comparison",
        interpolation_steps=28,
        bruteforce=False,
        sample_size=200,
    )

This ``from_array`` function automatically aligns the frames within a pullback and then between pullbacks. The algorithm translates contours to the same centroid as the most proximal contour,
and then finds the best rotation based on contour **AND** contour points.

.. image:: ../paper/figures/Figure2.jpg
   :alt: Example figure
   :align: center
   :width: 400px

The number of catheter points (``n_points``) therefore influences how much weight is given to the original image center. For mostly round contours, where Hausdorff distances are similar in different angles,
this image center can increase accuarcy of the right rotation. For stenotic section or coronary artery anomalies, where the vessel has distinct shape difference, this number can be kept
rather small (default 20 points compared to 500 for the contour).

``range_rotation_deg`` and ``step_rotation_deg`` define the +/- degree range where the rotation is tested (default 90° so full range) and step_rotation_deg in what step sizes (default 0.5°).
This algorithm is optimized and where it downsamples the original contour to 200 points, and performs coars steps (full provided range in 1° steps, then in +/- 5° degrees around the optimal angle
in 0.1° steps and so on until the desired acccuracy). If bruteforce is set to 'True' the complete range is sweeped with the provided acccuracy (not recommended O(n^3)).

If ``write_obj`` is set to True, geometries will be saved as .obj files. if interpolation steps are not 0, additionally interpolated geometries will be created. This is useful if the dynamic
behaviour will be rendered later on.
