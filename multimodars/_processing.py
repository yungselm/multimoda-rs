from __future__ import annotations

from .multimodars import (
    PyContour,
    PyContourType,
    PyGeometry,
    PyGeometryPair,
    PyInputData,
    PyCenterline,
    from_file_full as _from_file_full,
    from_file_doublepair as _from_file_doublepair,
    from_file_singlepair as _from_file_singlepair,
    from_file_single as _from_file_single,
    from_array_full as _from_array_full,
    from_array_doublepair as _from_array_doublepair,
    from_array_singlepair as _from_array_singlepair,
    from_array_single as _from_array_single,
    align_three_point as _align_three_point,
    align_manual as _align_manual,
    align_combined as _align_combined,
    to_obj as _to_obj,
    find_centerline_bounded_points_simple as _find_centerline_bounded_points_simple,
    find_proximal_distal_scaling as _find_proximal_distal_scaling,
    build_adjacency_map as _build_adjacency_map,
    discretize_vessel as _discretize_vessel,
)

_AlignLog = list[tuple[int, int, float, float, float, float, float]]


def _default_contour_types() -> list[PyContourType]:
    return [PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall]


# ---------------------------------------------------------------------------
# Processing functions — from CSV files
# ---------------------------------------------------------------------------


def from_file_full(
    input_path_ab: str,
    input_path_cd: str,
    labels: list[str] | None = None,
    step_rotation_deg: float = 0.5,
    range_rotation_deg: float = 90.0,
    sample_size: int = 500,
    image_center: tuple[float, float] = (4.5, 4.5),
    radius: float = 0.5,
    n_points: int = 20,
    write_obj: bool = True,
    watertight: bool = True,
    contour_types: list[PyContourType] | None = None,
    output_path_ab: str = "output/rest",
    output_path_cd: str = "output/stress",
    output_path_ac: str = "output/diastole",
    output_path_bd: str = "output/systole",
    interpolation_steps: int = 0,
    bruteforce: bool = False,
    smooth: bool = True,
    postprocessing: bool = True,
) -> tuple[
    PyGeometryPair,
    PyGeometryPair,
    PyGeometryPair,
    PyGeometryPair,
    tuple[_AlignLog, _AlignLog, _AlignLog, _AlignLog],
]:
    """Process four intravascular imaging geometries in parallel from CSV folders.

    Reads REST and STRESS acquisitions from two input folders, aligns frames
    within and between each cardiac phase in parallel, and writes interpolated
    OBJ meshes.

    .. parsed-literal::

                            ``output_path_ac`` (Diastole: a vs. c)
                    ┌──────────────────────────────────────────┐
                    ▼                                          ▼
        **a** REST diastole                      **c** STRESS diastole
                │  ``output_path_ab`` (Rest: a+b)        │  ``output_path_cd`` (Stress: c+d)
                ▼                                        ▼
        **b** REST systole                       **d** STRESS systole
                    └──────────────────────────────────────────┘
                            ``output_path_bd`` (Systole: b vs. d)

    .. warning::

       The CSV must have **no header**. Each row is
       ``(frame index, x-coord (mm), y-coord (mm), z-coord (mm))``:

    .. code-block:: text

       185, 5.32, 2.37, 0.0
       ...

    Parameters
    ----------
    input_path_ab : str
        Path to the REST input folder (contains diastolic ``a`` and systolic ``b`` CSVs).
    input_path_cd : str
        Path to the STRESS input folder (contains diastolic ``c`` and systolic ``d`` CSVs).
    labels : list of str, optional
        Labels for the four geometries ``[rest_dia, rest_sys, stress_dia,
        stress_sys]``.  Must be exactly 4 strings; if a different number is
        supplied the last component of each input path is used instead.
        Default is ``[]``.
    step_rotation_deg : float, optional
        Rotation step in degrees.  Default is ``0.5``.
    range_rotation_deg : float, optional
        Rotation search range (±) in degrees; a range of 90° gives 180°
        total.  Default is ``90.0``.
    sample_size : int, optional
        Number of points to downsample to during alignment.  Default is
        ``500``.
    image_center : tuple of float, optional
        Image center ``(x, y)`` in mm.  Default is ``(4.5, 4.5)``.
    radius : float, optional
        Catheter radius in mm.  Default is ``0.5``.
    n_points : int, optional
        Number of catheter points; more points increases the influence of
        the image center.  Default is ``20``.
    write_obj : bool, optional
        Whether to write OBJ files to disk.  Default is ``True``.
    watertight : bool, optional
        Whether to write a watertight or shell mesh.  Default is ``True``.
    contour_types : list of PyContourType, optional
        Contour types to export.  Default is
        ``[PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall]``.
    output_path_ab : str, optional
        Output directory for REST results (pair a+b).  Default is ``"output/rest"``.
    output_path_cd : str, optional
        Output directory for STRESS results (pair c+d).  Default is ``"output/stress"``.
    output_path_ac : str, optional
        Output directory for DIASTOLE results (pair a+c).  Default is
        ``"output/diastole"``.
    output_path_bd : str, optional
        Output directory for SYSTOLE results (pair b+d).  Default is
        ``"output/systole"``.
    interpolation_steps : int, optional
        Number of interpolated meshes between phases.  Default is ``28``.
    bruteforce : bool, optional
        Whether to use brute-force alignment (one comparison per step).
        Default is ``False``.
    smooth : bool, optional
        Whether to smooth frames after alignment using a 3-point moving
        average.  Default is ``True``.
    postprocessing : bool, optional
        Whether to adjust spacing within/between geometries to equal
        offsets.  Default is ``True``.

    Returns
    -------
    rest : PyGeometryPair
        Aligned geometry pair for the REST condition.
    stress : PyGeometryPair
        Aligned geometry pair for the STRESS condition.
    diastole : PyGeometryPair
        Aligned geometry pair for the diastolic phase.
    systole : PyGeometryPair
        Aligned geometry pair for the systolic phase.
    logs : tuple of list
        4-tuple of alignment logs ``(logs_a, logs_b, logs_c, logs_d)``;
        each entry is a list of
        ``(id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y)``.

    Examples
    --------
    >>> import multimodars as mm
    >>> rest, stress, dia, sys, _ = mm.from_file_full(
    ...     "data/ivus_rest", "data/ivus_stress"
    ... )
    """
    if contour_types is None:
        contour_types = _default_contour_types()
    return _from_file_full(
        input_path_ab,
        input_path_cd,
        labels or [],
        step_rotation_deg,
        range_rotation_deg,
        sample_size,
        image_center,
        radius,
        n_points,
        write_obj,
        watertight,
        contour_types,
        output_path_ab,
        output_path_cd,
        output_path_ac,
        output_path_bd,
        interpolation_steps,
        bruteforce,
        smooth,
        postprocessing,
    )


def from_file_doublepair(
    input_path_ab: str,
    input_path_cd: str,
    labels: list[str] | None = None,
    step_rotation_deg: float = 0.5,
    range_rotation_deg: float = 90.0,
    sample_size: int = 500,
    image_center: tuple[float, float] = (4.5, 4.5),
    radius: float = 0.5,
    n_points: int = 20,
    write_obj: bool = True,
    watertight: bool = True,
    contour_types: list[PyContourType] | None = None,
    output_path_ab: str = "output/rest",
    output_path_cd: str = "output/stress",
    interpolation_steps: int = 0,
    bruteforce: bool = False,
    smooth: bool = True,
    postprocessing: bool = True,
) -> tuple[
    PyGeometryPair,
    PyGeometryPair,
    tuple[_AlignLog, _AlignLog, _AlignLog, _AlignLog],
]:
    """Process two diastole/systole pairs in parallel from CSV folders.

    Reads REST and STRESS acquisitions independently, aligns frames within
    each pair, and writes interpolated OBJ meshes.

    .. parsed-literal::

        **a** REST diastole                      **c** STRESS diastole
                │  ``output_path_ab`` (Rest: a+b)        │  ``output_path_cd`` (Stress: c+d)
                ▼                                        ▼
        **b** REST systole                       **d** STRESS systole

    .. warning::

       The CSV must have **no header**. Each row is
       ``(frame index, x-coord (mm), y-coord (mm), z-coord (mm))``:

    .. code-block:: text

       185, 5.32, 2.37, 0.0
       ...

    Parameters
    ----------
    input_path_ab : str
        Path to the REST input folder (contains diastolic ``a`` and systolic ``b`` CSVs).
    input_path_cd : str
        Path to the STRESS input folder (contains diastolic ``c`` and systolic ``d`` CSVs).
    labels : list of str, optional
        Labels for the four geometries ``[rest_dia, rest_sys, stress_dia,
        stress_sys]``.  Must be exactly 4 strings; if a different number is
        supplied the last component of each input path is used instead.
        Default is ``[]``.
    step_rotation_deg : float, optional
        Rotation step in degrees.  Default is ``0.5``.
    range_rotation_deg : float, optional
        Rotation search range (±) in degrees.  Default is ``90.0``.
    sample_size : int, optional
        Number of points to downsample to.  Default is ``500``.
    image_center : tuple of float, optional
        Image center ``(x, y)`` in mm.  Default is ``(4.5, 4.5)``.
    radius : float, optional
        Catheter radius in mm.  Default is ``0.5``.
    n_points : int, optional
        Number of catheter points.  Default is ``20``.
    write_obj : bool, optional
        Whether to write OBJ files.  Default is ``True``.
    watertight : bool, optional
        Whether to write a watertight or shell mesh.  Default is ``True``.
    contour_types : list of PyContourType, optional
        Contour types to export.  Default is
        ``[PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall]``.
    output_path_ab : str, optional
        Output directory for REST results (pair a+b).  Default is ``"output/rest"``.
    output_path_cd : str, optional
        Output directory for STRESS results (pair c+d).  Default is ``"output/stress"``.
    interpolation_steps : int, optional
        Number of interpolated meshes between phases.  Default is ``28``.
    bruteforce : bool, optional
        Whether to use brute-force alignment.  Default is ``False``.
    smooth : bool, optional
        Whether to smooth frames after alignment.  Default is ``True``.
    postprocessing : bool, optional
        Whether to equalise spacing within/between geometries.  Default is
        ``True``.

    Returns
    -------
    rest : PyGeometryPair
        Aligned geometry pair for the REST condition.
    stress : PyGeometryPair
        Aligned geometry pair for the STRESS condition.
    logs : tuple of list
        4-tuple of alignment logs ``(logs_a, logs_b, logs_c, logs_d)``;
        each entry is a list of
        ``(id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y)``.

    Examples
    --------
    >>> import multimodars as mm
    >>> rest, stress, _ = mm.from_file_doublepair(
    ...     "data/ivus_rest", "data/ivus_stress"
    ... )
    """
    if contour_types is None:
        contour_types = _default_contour_types()
    return _from_file_doublepair(
        input_path_ab,
        input_path_cd,
        labels or [],
        step_rotation_deg,
        range_rotation_deg,
        sample_size,
        image_center,
        radius,
        n_points,
        write_obj,
        watertight,
        contour_types,
        output_path_ab,
        output_path_cd,
        interpolation_steps,
        bruteforce,
        smooth,
        postprocessing,
    )


def from_file_singlepair(
    input_path: str,
    labels: list[str] | None = None,
    step_rotation_deg: float = 0.5,
    range_rotation_deg: float = 90.0,
    sample_size: int = 500,
    image_center: tuple[float, float] = (4.5, 4.5),
    radius: float = 0.5,
    n_points: int = 20,
    write_obj: bool = True,
    watertight: bool = True,
    contour_types: list[PyContourType] | None = None,
    output_path: str = "output/singlepair",
    interpolation_steps: int = 0,
    bruteforce: bool = False,
    smooth: bool = True,
    postprocessing: bool = True,
) -> tuple[PyGeometryPair, tuple[_AlignLog, _AlignLog]]:
    """Process a single diastole/systole pair from an input CSV folder.

    Reads one acquisition folder, aligns the diastolic and systolic frames,
    and returns a single ``PyGeometryPair``.

    .. code-block:: text

       Pipeline:
          diastole
            │
            ▼
          systole

    .. warning::

       The CSV must have **no header**. Each row is
       ``(frame index, x-coord (mm), y-coord (mm), z-coord (mm))``:

    .. code-block:: text

       185, 5.32, 2.37, 0.0
       ...

    Parameters
    ----------
    input_path : str
        Path to the input CSV folder.
    labels : list of str, optional
        Labels for the two geometries ``[diastole, systole]``.  Must be
        exactly 2 strings; if a different number is supplied the last
        component of the input path is used instead.  Default is ``[]``.
    step_rotation_deg : float, optional
        Rotation step in degrees.  Default is ``0.5``.
    range_rotation_deg : float, optional
        Rotation search range (±) in degrees.  Default is ``90.0``.
    sample_size : int, optional
        Number of points to downsample to.  Default is ``500``.
    image_center : tuple of float, optional
        Image center ``(x, y)`` in mm.  Default is ``(4.5, 4.5)``.
    radius : float, optional
        Catheter radius in mm.  Default is ``0.5``.
    n_points : int, optional
        Number of catheter points.  Default is ``20``.
    write_obj : bool, optional
        Whether to write OBJ files.  Default is ``True``.
    watertight : bool, optional
        Whether to write a watertight or shell mesh.  Default is ``True``.
    contour_types : list of PyContourType, optional
        Contour types to export.  Default is
        ``[PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall]``.
    output_path : str, optional
        Directory path to write the processed geometry.  Default is
        ``"output/singlepair"``.
    interpolation_steps : int, optional
        Number of interpolated meshes.  Default is ``28``.
    bruteforce : bool, optional
        Whether to use brute-force alignment.  Default is ``False``.
    smooth : bool, optional
        Whether to smooth frames after alignment.  Default is ``True``.
    postprocessing : bool, optional
        Whether to equalise spacing within/between geometries.  Default is
        ``True``.

    Returns
    -------
    pair : PyGeometryPair
        Aligned diastole/systole geometry pair.
    logs : tuple of list
        2-tuple of alignment logs ``(logs_a, logs_b)``; each entry is a list
        of ``(id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y)``.

    Examples
    --------
    >>> import multimodars as mm
    >>> pair, _ = mm.from_file_singlepair("data/ivus_rest")
    """
    if contour_types is None:
        contour_types = _default_contour_types()
    return _from_file_singlepair(
        input_path,
        labels or [],
        step_rotation_deg,
        range_rotation_deg,
        sample_size,
        image_center,
        radius,
        n_points,
        write_obj,
        watertight,
        contour_types,
        output_path,
        interpolation_steps,
        bruteforce,
        smooth,
        postprocessing,
    )


def from_file_single(
    input_path: str,
    labels: list[str] | None = None,
    diastole: bool = True,
    step_rotation_deg: float = 0.5,
    range_rotation_deg: float = 90.0,
    sample_size: int = 500,
    image_center: tuple[float, float] = (4.5, 4.5),
    radius: float = 0.5,
    n_points: int = 20,
    write_obj: bool = True,
    watertight: bool = True,
    contour_types: list[PyContourType] | None = None,
    output_path: str = "output/single",
    bruteforce: bool = False,
    smooth: bool = True,
) -> tuple[PyGeometry, _AlignLog]:
    """Process a single intravascular imaging geometry from a CSV file.

    Reads one phase (diastole or systole) from an IVUS CSV file, aligns
    frames within the geometry, and optionally writes OBJ output.

    .. warning::

       The CSV must have **no header**. Each row is
       ``(frame index, x-coord (mm), y-coord (mm), z-coord (mm))``.

    Parameters
    ----------
    input_path : str
        Path to the input CSV folder.
    labels : list of str, optional
        Label for the geometry (1 string).  If a different number is
        supplied the last component of the input path is used instead.
        Default is ``[]``.
    diastole : bool, optional
        When ``True`` process the diastolic phase; otherwise systole.
        Default is ``True``.
    step_rotation_deg : float, optional
        Rotation step in degrees.  Default is ``0.5``.
    range_rotation_deg : float, optional
        Rotation search range (±) in degrees.  Default is ``90.0``.
    sample_size : int, optional
        Number of points to downsample to.  Default is ``500``.
    image_center : tuple of float, optional
        Image center ``(x, y)`` in mm.  Default is ``(4.5, 4.5)``.
    radius : float, optional
        Catheter radius in mm.  Default is ``0.5``.
    n_points : int, optional
        Number of catheter points.  Default is ``20``.
    write_obj : bool, optional
        Whether to write OBJ files.  Default is ``True``.
    watertight : bool, optional
        Whether to write a watertight or shell mesh.  Default is ``True``.
    contour_types : list of PyContourType, optional
        Contour types to export.  Default is
        ``[PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall]``.
    output_path : str, optional
        Directory path to write the processed geometry.  Default is
        ``"output/single"``.
    bruteforce : bool, optional
        Whether to use brute-force alignment.  Default is ``False``.
    smooth : bool, optional
        Whether to smooth frames after alignment.  Default is ``True``.

    Returns
    -------
    geom : PyGeometry
        Processed geometry for the chosen phase.
    logs : list
        Alignment log entries; each entry is
        ``(id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y)``.

    Examples
    --------
    >>> import multimodars as mm
    >>> geom, _ = mm.from_file_single("data/ivus.csv", diastole=False)
    """
    if contour_types is None:
        contour_types = _default_contour_types()
    return _from_file_single(
        input_path,
        labels or [],
        diastole,
        step_rotation_deg,
        range_rotation_deg,
        sample_size,
        image_center,
        radius,
        n_points,
        write_obj,
        watertight,
        contour_types,
        output_path,
        bruteforce,
        smooth,
    )


# ---------------------------------------------------------------------------
# Processing functions — from PyInputData arrays
# ---------------------------------------------------------------------------


def from_array_full(
    input_data_a: PyInputData,
    input_data_b: PyInputData,
    input_data_c: PyInputData,
    input_data_d: PyInputData,
    step_rotation_deg: float = 0.5,
    range_rotation_deg: float = 90.0,
    sample_size: int = 500,
    image_center: tuple[float, float] = (4.5, 4.5),
    radius: float = 0.5,
    n_points: int = 20,
    write_obj: bool = True,
    watertight: bool = True,
    contour_types: list[PyContourType] | None = None,
    output_path_ab: str = "output/rest",
    output_path_cd: str = "output/stress",
    output_path_ac: str = "output/diastole",
    output_path_bd: str = "output/systole",
    interpolation_steps: int = 0,
    bruteforce: bool = False,
    smooth: bool = True,
    postprocessing: bool = True,
) -> tuple[
    PyGeometryPair,
    PyGeometryPair,
    PyGeometryPair,
    PyGeometryPair,
    tuple[_AlignLog, _AlignLog, _AlignLog, _AlignLog],
]:
    """Process four ``PyInputData`` objects in parallel, aligning and interpolating between phases.

    Accepts pre-loaded input data for REST diastole, REST systole, STRESS
    diastole, and STRESS systole, then aligns frames within and between
    each cardiac phase.

    .. parsed-literal::

                            ``output_path_ac`` (Diastole: a vs. c)
                    ┌──────────────────────────────────────────┐
                    ▼                                          ▼
        **a** REST diastole                      **c** STRESS diastole
                │  ``output_path_ab`` (Rest: a+b)        │  ``output_path_cd`` (Stress: c+d)
                ▼                                        ▼
        **b** REST systole                       **d** STRESS systole
                    └──────────────────────────────────────────┘
                            ``output_path_bd`` (Systole: b vs. d)

    Parameters
    ----------
    input_data_a : PyInputData
        Diastolic REST input data.
    input_data_b : PyInputData
        Systolic REST input data.
    input_data_c : PyInputData
        Diastolic STRESS input data.
    input_data_d : PyInputData
        Systolic STRESS input data.
    step_rotation_deg : float, optional
        Rotation step in degrees.  Default is ``0.5``.
    range_rotation_deg : float, optional
        Rotation search range (±) in degrees.  Default is ``90.0``.
    sample_size : int, optional
        Number of points to downsample to.  Default is ``500``.
    image_center : tuple of float, optional
        Image center ``(x, y)`` in mm.  Default is ``(4.5, 4.5)``.
    radius : float, optional
        Catheter radius in mm.  Default is ``0.5``.
    n_points : int, optional
        Number of catheter points.  Default is ``20``.
    write_obj : bool, optional
        Whether to write OBJ files.  Default is ``True``.
    watertight : bool, optional
        Whether to write a watertight or shell mesh.  Default is ``True``.
    contour_types : list of PyContourType, optional
        Contour types to export.  Default is
        ``[PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall]``.
    output_path_ab : str, optional
        Output directory for REST results (pair a+b).  Default is ``"output/rest"``.
    output_path_cd : str, optional
        Output directory for STRESS results (pair c+d).  Default is ``"output/stress"``.
    output_path_ac : str, optional
        Output directory for DIASTOLE results (pair a+c).  Default is
        ``"output/diastole"``.
    output_path_bd : str, optional
        Output directory for SYSTOLE results (pair b+d).  Default is
        ``"output/systole"``.
    interpolation_steps : int, optional
        Number of interpolation steps between phases.  Default is ``28``.
    bruteforce : bool, optional
        Whether to use brute-force alignment.  Default is ``False``.
    smooth : bool, optional
        Whether to smooth frames after alignment.  Default is ``True``.
    postprocessing : bool, optional
        Whether to equalise spacing within/between geometries.  Default is
        ``True``.

    Returns
    -------
    rest : PyGeometryPair
        Aligned geometry pair for the REST condition.
    stress : PyGeometryPair
        Aligned geometry pair for the STRESS condition.
    diastole : PyGeometryPair
        Aligned geometry pair for the diastolic phase.
    systole : PyGeometryPair
        Aligned geometry pair for the systolic phase.
    logs : tuple of list
        4-tuple of alignment logs ``(logs_a, logs_b, logs_c, logs_d)``;
        each entry is a list of
        ``(id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y)``.

    Examples
    --------
    >>> import multimodars as mm
    >>> rest, stress, dia, sys, _ = mm.from_array_full(
    ...     rest_dia, rest_sys, stress_dia, stress_sys
    ... )
    """
    if contour_types is None:
        contour_types = _default_contour_types()
    return _from_array_full(
        input_data_a,
        input_data_b,
        input_data_c,
        input_data_d,
        step_rotation_deg,
        range_rotation_deg,
        sample_size,
        image_center,
        radius,
        n_points,
        write_obj,
        watertight,
        contour_types,
        output_path_ab,
        output_path_cd,
        output_path_ac,
        output_path_bd,
        interpolation_steps,
        bruteforce,
        smooth,
        postprocessing,
    )


def from_array_doublepair(
    input_data_a: PyInputData,
    input_data_b: PyInputData,
    input_data_c: PyInputData,
    input_data_d: PyInputData,
    step_rotation_deg: float = 0.5,
    range_rotation_deg: float = 90.0,
    sample_size: int = 500,
    image_center: tuple[float, float] = (4.5, 4.5),
    radius: float = 0.5,
    n_points: int = 20,
    write_obj: bool = True,
    watertight: bool = True,
    contour_types: list[PyContourType] | None = None,
    output_path_ab: str = "output/rest",
    output_path_cd: str = "output/stress",
    interpolation_steps: int = 0,
    bruteforce: bool = False,
    smooth: bool = True,
    postprocessing: bool = True,
) -> tuple[
    PyGeometryPair,
    PyGeometryPair,
    tuple[_AlignLog, _AlignLog, _AlignLog, _AlignLog],
]:
    """Process two ``PyInputData`` pairs in parallel, aligning frames within each pair independently.

    Accepts pre-loaded data for REST (diastole + systole) and STRESS
    (diastole + systole), aligns each pair independently, and writes
    interpolated OBJ meshes.

    .. parsed-literal::

        **a** REST diastole                      **c** STRESS diastole
                │  ``output_path_ab`` (Rest: a+b)        │  ``output_path_cd`` (Stress: c+d)
                ▼                                        ▼
        **b** REST systole                       **d** STRESS systole

    Parameters
    ----------
    input_data_a : PyInputData
        Diastolic REST input data.
    input_data_b : PyInputData
        Systolic REST input data.
    input_data_c : PyInputData
        Diastolic STRESS input data.
    input_data_d : PyInputData
        Systolic STRESS input data.
    step_rotation_deg : float, optional
        Rotation step in degrees.  Default is ``0.5``.
    range_rotation_deg : float, optional
        Rotation search range (±) in degrees.  Default is ``90.0``.
    sample_size : int, optional
        Number of points to downsample to.  Default is ``500``.
    image_center : tuple of float, optional
        Image center ``(x, y)`` in mm.  Default is ``(4.5, 4.5)``.
    radius : float, optional
        Catheter radius in mm.  Default is ``0.5``.
    n_points : int, optional
        Number of catheter points.  Default is ``20``.
    write_obj : bool, optional
        Whether to write OBJ files.  Default is ``True``.
    watertight : bool, optional
        Whether to write a watertight or shell mesh.  Default is ``True``.
    contour_types : list of PyContourType, optional
        Contour types to export.  Default is
        ``[PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall]``.
    output_path_ab : str, optional
        Output directory for REST results (pair a+b).  Default is ``"output/rest"``.
    output_path_cd : str, optional
        Output directory for STRESS results (pair c+d).  Default is ``"output/stress"``.
    interpolation_steps : int, optional
        Number of interpolation steps between phases.  Default is ``28``.
    bruteforce : bool, optional
        Whether to use brute-force alignment.  Default is ``False``.
    smooth : bool, optional
        Whether to smooth frames after alignment.  Default is ``True``.
    postprocessing : bool, optional
        Whether to equalise spacing within/between geometries.  Default is
        ``True``.

    Returns
    -------
    rest : PyGeometryPair
        Aligned geometry pair for the REST condition.
    stress : PyGeometryPair
        Aligned geometry pair for the STRESS condition.
    logs : tuple of list
        4-tuple of alignment logs ``(logs_a, logs_b, logs_c, logs_d)``;
        each entry is a list of
        ``(id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y)``.

    Examples
    --------
    >>> import multimodars as mm
    >>> rest, stress, _ = mm.from_array_doublepair(
    ...     rest_dia, rest_sys, stress_dia, stress_sys
    ... )
    """
    if contour_types is None:
        contour_types = _default_contour_types()
    return _from_array_doublepair(
        input_data_a,
        input_data_b,
        input_data_c,
        input_data_d,
        step_rotation_deg,
        range_rotation_deg,
        sample_size,
        image_center,
        radius,
        n_points,
        write_obj,
        watertight,
        contour_types,
        output_path_ab,
        output_path_cd,
        interpolation_steps,
        bruteforce,
        smooth,
        postprocessing,
    )


def from_array_singlepair(
    input_data_a: PyInputData,
    input_data_b: PyInputData,
    step_rotation_deg: float = 0.5,
    range_rotation_deg: float = 90.0,
    sample_size: int = 500,
    image_center: tuple[float, float] = (4.5, 4.5),
    radius: float = 0.5,
    n_points: int = 20,
    write_obj: bool = True,
    watertight: bool = True,
    contour_types: list[PyContourType] | None = None,
    output_path: str = "output/singlepair",
    interpolation_steps: int = 0,
    bruteforce: bool = False,
    smooth: bool = True,
    postprocessing: bool = True,
) -> tuple[PyGeometryPair, tuple[_AlignLog, _AlignLog]]:
    """Align and interpolate between two ``PyInputData`` objects (diastole and systole).

    Accepts pre-loaded diastolic and systolic input data, aligns frames
    between the two phases, and returns a single ``PyGeometryPair``.

    .. code-block:: text

       diastole ──▶ systole

    Parameters
    ----------
    input_data_a : PyInputData
        Diastolic input data.
    input_data_b : PyInputData
        Systolic input data.
    step_rotation_deg : float, optional
        Rotation step in degrees.  Default is ``0.5``.
    range_rotation_deg : float, optional
        Rotation search range (±) in degrees.  Default is ``90.0``.
    sample_size : int, optional
        Number of points to downsample to.  Default is ``500``.
    image_center : tuple of float, optional
        Image center ``(x, y)`` in mm.  Default is ``(4.5, 4.5)``.
    radius : float, optional
        Catheter radius in mm.  Default is ``0.5``.
    n_points : int, optional
        Number of catheter points.  Default is ``20``.
    write_obj : bool, optional
        Whether to write OBJ files.  Default is ``True``.
    watertight : bool, optional
        Whether to write a watertight or shell mesh.  Default is ``True``.
    contour_types : list of PyContourType, optional
        Contour types to export.  Default is
        ``[PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall]``.
    output_path : str, optional
        Directory path to write interpolated output files.  Default is
        ``"output/singlepair"``.
    interpolation_steps : int, optional
        Number of interpolation steps between phases.  Default is ``28``.
    bruteforce : bool, optional
        Whether to use brute-force alignment.  Default is ``False``.
    smooth : bool, optional
        Whether to smooth frames after alignment.  Default is ``True``.
    postprocessing : bool, optional
        Whether to equalise spacing within/between geometries.  Default is
        ``True``.

    Returns
    -------
    pair : PyGeometryPair
        Aligned diastole/systole geometry pair.
    logs : tuple of list
        2-tuple of alignment logs ``(logs_a, logs_b)``; each entry is a list
        of ``(id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y)``.

    Examples
    --------
    >>> import multimodars as mm
    >>> pair, _ = mm.from_array_singlepair(rest_dia, rest_sys)
    """
    if contour_types is None:
        contour_types = _default_contour_types()
    return _from_array_singlepair(
        input_data_a,
        input_data_b,
        step_rotation_deg,
        range_rotation_deg,
        sample_size,
        image_center,
        radius,
        n_points,
        write_obj,
        watertight,
        contour_types,
        output_path,
        interpolation_steps,
        bruteforce,
        smooth,
        postprocessing,
    )


def from_array_single(
    input_data: PyInputData,
    step_rotation_deg: float = 0.5,
    range_rotation_deg: float = 90.0,
    sample_size: int = 500,
    image_center: tuple[float, float] = (4.5, 4.5),
    radius: float = 0.5,
    n_points: int = 20,
    write_obj: bool = False,
    watertight: bool = True,
    contour_types: list[PyContourType] | None = None,
    output_path: str = "output/single",
    bruteforce: bool = False,
    smooth: bool = True,
) -> tuple[PyGeometry, _AlignLog]:
    """Process a single geometry phase from a ``PyInputData`` object.

    Accepts pre-loaded input data for one cardiac phase, aligns frames
    within the geometry, and optionally writes OBJ output.

    Parameters
    ----------
    input_data : PyInputData
        Input data for a single cardiac phase (e.g. diastolic REST).
    step_rotation_deg : float, optional
        Rotation step in degrees.  Default is ``0.5``.
    range_rotation_deg : float, optional
        Rotation search range (±) in degrees.  Default is ``90.0``.
    sample_size : int, optional
        Number of points to downsample to.  Default is ``500``.
    image_center : tuple of float, optional
        Image center ``(x, y)`` in mm.  Default is ``(4.5, 4.5)``.
    radius : float, optional
        Catheter radius in mm.  Default is ``0.5``.
    n_points : int, optional
        Number of catheter points.  Default is ``20``.
    write_obj : bool, optional
        Whether to write OBJ files.  Default is ``False``.
    watertight : bool, optional
        Whether to write a watertight or shell mesh.  Default is ``True``.
    contour_types : list of PyContourType, optional
        Contour types to export.  Default is
        ``[PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall]``.
    output_path : str, optional
        Directory path to write the processed geometry.  Default is
        ``"output/single"``.
    bruteforce : bool, optional
        Whether to use brute-force alignment.  Default is ``False``.
    smooth : bool, optional
        Whether to smooth frames after alignment.  Default is ``True``.

    Returns
    -------
    geom : PyGeometry
        Processed geometry for the chosen phase.
    logs : list
        Alignment log entries; each entry is
        ``(id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y)``.

    Examples
    --------
    >>> import multimodars as mm
    >>> geom, _ = mm.from_array_single(input_data)
    """
    if contour_types is None:
        contour_types = _default_contour_types()
    return _from_array_single(
        input_data,
        step_rotation_deg,
        range_rotation_deg,
        sample_size,
        image_center,
        radius,
        n_points,
        write_obj,
        watertight,
        contour_types,
        output_path,
        bruteforce,
        smooth,
    )


# ---------------------------------------------------------------------------
# Alignment functions
# ---------------------------------------------------------------------------


def align_three_point(
    centerline: PyCenterline,
    geometry: PyGeometryPair | PyGeometry,
    aortic_ref_pt: tuple[float, float, float],
    upper_ref_pt: tuple[float, float, float],
    lower_ref_pt: tuple[float, float, float],
    angle_step_deg: float = 1.0,
    write: bool = False,
    watertight: bool = True,
    interpolation_steps: int = 0,
    output_dir: str = "output/aligned",
    contour_types: list[PyContourType] | None = None,
    case_name: str = "None",
) -> tuple[PyGeometryPair | PyGeometry, PyCenterline]:
    """Align a geometry (or geometry pair) to the centerline using three reference points.

    Creates centerline-aligned meshes based on three anatomical reference points
    (aorta, upper section, lower section).  Only works for elliptic vessels such
    as coronary artery anomalies.

    Parameters
    ----------
    centerline : PyCenterline
        Centerline of the vessel.
    geometry : PyGeometryPair or PyGeometry
        Single geometry or diastolic/systolic geometry pair to align.
    aortic_ref_pt : tuple of float
        ``(x, y, z)`` reference point at the aortic ostium.
    upper_ref_pt : tuple of float
        ``(x, y, z)`` upper section reference point.
    lower_ref_pt : tuple of float
        ``(x, y, z)`` lower section reference point.
    angle_step_deg : float, optional
        Step size in degrees for the rotation search.  Default is ``1.0``.
    write : bool, optional
        Whether to write the aligned meshes to OBJ files.  Default is ``False``.
    watertight : bool, optional
        Whether to write a watertight or shell mesh.  Default is ``True``.
    interpolation_steps : int, optional
        Number of interpolation steps between phases.  Only used when *geometry*
        is a ``PyGeometryPair``.  Default is ``0``.
    output_dir : str, optional
        Output directory for aligned meshes.  Default is ``"output/aligned"``.
    contour_types : list of PyContourType, optional
        Contour types to export.  Default is
        ``[PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall]``.
    case_name : str, optional
        Case name used as a filename prefix.  Default is ``"None"``.

    Returns
    -------
    geometry : PyGeometryPair or PyGeometry
        Aligned geometry, matching the type of the input.
    centerline : PyCenterline
        Resampled centerline.

    Examples
    --------
    >>> import multimodars as mm
    >>> result, cl = mm.align_three_point(
    ...     centerline,
    ...     geometry_pair,
    ...     (12.2605, -201.3643, 1751.0554),
    ...     (11.7567, -202.1920, 1754.7975),
    ...     (15.6605, -202.1920, 1749.9655),
    ... )
    """
    if contour_types is None:
        contour_types = _default_contour_types()
    return _align_three_point(
        centerline,
        geometry,
        aortic_ref_pt,
        upper_ref_pt,
        lower_ref_pt,
        angle_step_deg,
        write,
        watertight,
        interpolation_steps,
        output_dir,
        contour_types,
        case_name,
    )


def align_manual(
    centerline: PyCenterline,
    geometry: PyGeometryPair | PyGeometry,
    rotation_angle: float,
    ref_point: tuple[float, float, float],
    write: bool = False,
    watertight: bool = True,
    interpolation_steps: int = 0,
    output_dir: str = "output/aligned",
    contour_types: list[PyContourType] | None = None,
    case_name: str = "None",
) -> tuple[PyGeometryPair | PyGeometry, PyCenterline]:
    """Align a geometry (or geometry pair) to the centerline using a manual rotation angle.

    Creates centerline-aligned meshes using an explicit rotation angle and a
    single reference point on the centerline.  Only works for elliptic vessels
    such as coronary artery anomalies.

    Parameters
    ----------
    centerline : PyCenterline
        Centerline of the vessel.
    geometry : PyGeometryPair or PyGeometry
        Single geometry or diastolic/systolic geometry pair to align.
    rotation_angle : float
        Rotation angle in radians to apply.
    ref_point : tuple of float
        ``(x, y, z)`` reference point on the centerline.
    write : bool, optional
        Whether to write the aligned meshes to OBJ files.  Default is ``False``.
    watertight : bool, optional
        Whether to write a watertight or shell mesh.  Default is ``True``.
    interpolation_steps : int, optional
        Number of interpolation steps between phases.  Only used when *geometry*
        is a ``PyGeometryPair``.  Default is ``0``.
    output_dir : str, optional
        Output directory for aligned meshes.  Default is ``"output/aligned"``.
    contour_types : list of PyContourType, optional
        Contour types to export.  Default is
        ``[PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall]``.
    case_name : str, optional
        Case name used as a filename prefix.  Default is ``"None"``.

    Returns
    -------
    geometry : PyGeometryPair or PyGeometry
        Aligned geometry, matching the type of the input.
    centerline : PyCenterline
        Resampled centerline.

    Examples
    --------
    >>> import multimodars as mm
    >>> result, cl = mm.align_manual(
    ...     centerline, geometry_pair, rotation_angle=1.57, ref_point=(1.0, 2.0, 3.0)
    ... )
    """
    if contour_types is None:
        contour_types = _default_contour_types()
    return _align_manual(
        centerline,
        geometry,
        rotation_angle,
        ref_point,
        write,
        watertight,
        interpolation_steps,
        output_dir,
        contour_types,
        case_name,
    )


def align_combined(
    centerline: PyCenterline,
    geometry: PyGeometryPair | PyGeometry,
    aortic_ref_pt: tuple[float, float, float],
    upper_ref_pt: tuple[float, float, float],
    lower_ref_pt: tuple[float, float, float],
    points: list[tuple[float, float, float]],
    angle_step_deg: float = 1.0,
    angle_range_deg: float = 15.0,
    index_range: int = 2,
    write: bool = False,
    watertight: bool = True,
    interpolation_steps: int = 0,
    output_dir: str = "output/aligned",
    contour_types: list[PyContourType] | None = None,
    case_name: str = "None",
) -> tuple[PyGeometryPair | PyGeometry, PyCenterline]:
    """Align a geometry (or geometry pair) using three reference points and Hausdorff refinement.

    Creates centerline-aligned meshes using three anatomical reference points
    for an initial orientation and a set of additional points for
    Hausdorff distance-based fine-tuning of the rotation.

    Parameters
    ----------
    centerline : PyCenterline
        Centerline of the vessel.
    geometry : PyGeometryPair or PyGeometry
        Single geometry or diastolic/systolic geometry pair to align.
    aortic_ref_pt : tuple of float
        ``(x, y, z)`` reference point at the aortic ostium.
    upper_ref_pt : tuple of float
        ``(x, y, z)`` upper section reference point.
    lower_ref_pt : tuple of float
        ``(x, y, z)`` lower section reference point.
    points : list of tuple of float
        Point cloud used for Hausdorff distance calculation during rotation
        refinement.
    angle_step_deg : float, optional
        Step size in degrees for the rotation search.  Default is ``1.0``.
    angle_range_deg : float, optional
        Total rotation search range in degrees.  Default is ``15.0``.
    index_range : int, optional
        Number of centerline indices considered around the reference.
        Default is ``2``.
    write : bool, optional
        Whether to write the aligned meshes to OBJ files.  Default is ``False``.
    watertight : bool, optional
        Whether to write a watertight or shell mesh.  Default is ``True``.
    interpolation_steps : int, optional
        Number of interpolation steps between phases.  Only used when *geometry*
        is a ``PyGeometryPair``.  Default is ``0``.
    output_dir : str, optional
        Output directory for aligned meshes.  Default is ``"output/aligned"``.
    contour_types : list of PyContourType, optional
        Contour types to export.  Default is
        ``[PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall]``.
    case_name : str, optional
        Case name used as a filename prefix.  Default is ``"None"``.

    Returns
    -------
    geometry : PyGeometryPair or PyGeometry
        Aligned geometry, matching the type of the input.
    centerline : PyCenterline
        Resampled centerline.

    Examples
    --------
    >>> import multimodars as mm
    >>> result, cl = mm.align_combined(
    ...     centerline,
    ...     geometry_pair,
    ...     (12.2605, -201.3643, 1751.0554),
    ...     (11.7567, -202.1920, 1754.7975),
    ...     (15.6605, -202.1920, 1749.9655),
    ...     point_cloud,
    ... )
    """
    if contour_types is None:
        contour_types = _default_contour_types()
    return _align_combined(
        centerline,
        geometry,
        aortic_ref_pt,
        upper_ref_pt,
        lower_ref_pt,
        points,
        angle_step_deg,
        angle_range_deg,
        index_range,
        write,
        watertight,
        interpolation_steps,
        output_dir,
        contour_types,
        case_name,
    )


# ---------------------------------------------------------------------------
# OBJ export
# ---------------------------------------------------------------------------


def to_obj(
    geometry: PyGeometry,
    output_path: str,
    watertight: bool = True,
    contour_types: list[PyContourType] | None = None,
    filename_prefix: str = "",
) -> None:
    """Convert a ``PyGeometry`` object into OBJ files and write them to disk.

    Writes the specified contour types as OBJ meshes without UV coordinates.
    Each contour type is written to its own file together with a corresponding
    MTL material file.

    Parameters
    ----------
    geometry : PyGeometry
        Input geometry instance containing the mesh data.
    output_path : str
        Directory path where the OBJ and MTL files will be written.
    watertight : bool, optional
        Whether to write a watertight or shell mesh.  Default is ``True``.
    contour_types : list of PyContourType, optional
        Contour types to export.  Default is
        ``[PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall]``.
    filename_prefix : str, optional
        Optional prefix prepended to all output filenames.  Default is ``""``.

    Raises
    ------
    RuntimeError
        If any of the underlying file writes fail.

    Examples
    --------
    >>> import multimodars as mm
    >>> mm.to_obj(geometry, "output/meshes", watertight=True)
    """
    if contour_types is None:
        contour_types = _default_contour_types()
    return _to_obj(geometry, output_path, watertight, contour_types, filename_prefix)


# ---------------------------------------------------------------------------
# IVUS/CCTA Fusion
# ---------------------------------------------------------------------------


def find_centerline_bounded_points_simple(
    centerline: PyCenterline,
    points: list[tuple[float, float, float]],
    radius: float,
) -> list[tuple[float, float, float]]:
    """Find points bounded by spheres along a coronary vessel centerline.

    This version accepts and returns simple Python lists of tuples.

    Parameters
    ----------
    centerline : PyCenterline
        Centerline of the vessel.
    points : list of tuple of float
        List of ``(x, y, z)`` point coordinates.
    radius : float
        Radius of the bounding spheres around each centerline point.

    Returns
    -------
    bounded_points : list of tuple of float
        Filtered points that are inside the bounding spheres.

    Examples
    --------
    >>> import multimodars as mm
    >>>
    >>> # Load centerline and point cloud
    >>> centerline = mm.load_centerline("path/to/centerline.json")
    >>> points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), ...]  # or mesh.vertices.tolist()
    >>>
    >>> # Find points bounded by centerline spheres
    >>> bounded_points = mm.find_centerline_bounded_points(centerline, points, 2.0)
    >>> print(f"Found {len(bounded_points)} points inside vessel bounds")"""
    return _find_centerline_bounded_points_simple(centerline, points, radius)


def find_proximal_distal_scaling(
    anomalous_points: list[tuple[float, float, float]],
    n_proximal: int,
    n_distal: int,
    centerline: PyCenterline,
    proximal_reference: list[tuple[float, float, float]],
    distal_reference: list[tuple[float, float, float]],
):
    """Find the optimal diameter scaling for the proximal and distal regions.

    Parameters
    ----------
    anomalous_points : list of tuple of float
        ``(x, y, z)`` coordinates of the anomalous vessel region.
    n_proximal : int
        Number of proximal points used for comparison.
    n_distal : int
        Number of distal points used for comparison.
    centerline : PyCenterline
        Centerline of the vessel region.
    proximal_reference : list of tuple of float
        Reference ``(x, y, z)`` points from the CCTA mesh for the proximal region.
    distal_reference : list of tuple of float
        Reference ``(x, y, z)`` points for the distal region.

    Returns
    -------
    proximal_scaling : float
        Optimal scaling distance for the proximal region.
    distal_scaling : float
        Optimal scaling distance for the distal region.

    Examples
    --------
    >>> import multimodars as mm"""
    return _find_proximal_distal_scaling(
        anomalous_points,
        n_proximal,
        n_distal,
        centerline,
        proximal_reference,
        distal_reference,
    )


def build_adjacency_map(
    faces: list[list[int]],
) -> dict[int, set[int]]:
    """Build a vertex adjacency map from a triangle mesh face list.
    For each triangle face, all three undirected edges are recorded so that
    every vertex maps to the set of vertices it shares an edge with.

    Parameters
    ----------
    faces : list of list of int
        Triangle faces, each represented as a three-element array of vertex
        indices ``[v0, v1, v2]``.

    Returns
    -------
    adjacency_map : dict of int to set of int
        Mapping from each vertex index to the set of its directly connected
        neighbour vertex indices.

    Examples
    --------
    >>> import multimodars as mm
    >>>
    >>> faces = [[0, 1, 2], [1, 2, 3]]
    >>> adj = mm.build_adjacency_map(faces)
    >>> print(adj[1])  # {0, 2, 3}"""
    return _build_adjacency_map(
        faces,
    )


def discretize_vessel(
    centerline: "PyCenterline",
    points: list[tuple[float, float, float]],
    branch_id: int = 0,
    step_size: float = 0.5,
    n_points: int = 200,
) -> list[PyContour]:
    """Discretize a vessel surface mesh along a centerline branch into uniform cross-sections.

    Walks the specified centerline branch at uniform arc-length intervals of ``step_size``,
    projects the supplied mesh points onto each perpendicular cross-sectional plane, discards
    incomplete slices (empty or not covering all four angular quadrants), and resamples the
    remaining contours to exactly ``n_points`` evenly-spaced points via a closed Catmull-Rom
    spline.

    Parameters
    ----------
    centerline : PyCenterline
        Centerline object containing one or more branches.
    points : list of tuple of (float, float, float)
        3-D surface mesh points ``(x, y, z)`` to project onto each cross-section.
    branch_id : int, optional
        Index of the centerline branch to walk. Default is ``0``.
    step_size : float, optional
        Arc-length distance between consecutive cross-sections in the same units as
        ``centerline`` and ``points``. Default is ``0.5``.
    n_points : int, optional
        Number of evenly-spaced points per output contour. Default is ``200``.

    Returns
    -------
    contours : list of PyContour
        One contour per surviving cross-section, each containing exactly ``n_points``
        uniformly distributed points lying on a Catmull-Rom spline fit to the projected
        surface points.

    Examples
    --------
    >>> import multimodars as mm
    >>>
    >>> contours = mm.discretize_vessel(centerline, mesh_points, branch_id=0, step_size=0.5)
    >>> print(len(contours))"""
    return _discretize_vessel(
        centerline,
        points,
        branch_id,
        step_size,
        n_points,
    )
