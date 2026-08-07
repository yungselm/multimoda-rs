"""
Type stubs for the multimodars Rust extension module.

These stubs describe the public interface of the compiled
``multimodars.multimodars`` extension so that type checkers (mypy, pyright)
and IDEs can offer autocompletion and type checking for downstream code.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

# Alignment log entry: (id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y)
# Rust: Vec<(u32, u32, f64, f64, f64, f64, f64)>
_AlignLog = list[tuple[int, int, float, float, float, float, float]]

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class PyContourType:
    """Enumeration of supported intravascular contour types.

    Members:
        Lumen: The vessel lumen.
        Eem: External elastic membrane.
        Calcification: Calcifications.
        Sidebranch: Side branch lumen.
        Catheter: Imaging catheter artifact.
        Wall: Vessel wall (not visible on images, but created).
    """

    Lumen: PyContourType
    Eem: PyContourType
    Calcification: PyContourType
    Sidebranch: PyContourType
    Catheter: PyContourType
    Wall: PyContourType

    @property
    def name(self) -> str: ...
    @staticmethod
    def from_string(name: str) -> PyContourType: ...
    @staticmethod
    def all_types() -> list[PyContourType]: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

# ---------------------------------------------------------------------------
# Core data classes
# ---------------------------------------------------------------------------

class PyContourPoint:
    """A single 3-D point on a contour or centerline.

    Attributes
    ----------
    frame_index : int
        Frame number in the acquisition sequence.
    point_index : int
        Index of this point within its contour.
    x : float
        X-coordinate in mm.
    y : float
        Y-coordinate in mm.
    z : float
        Z-coordinate in mm.
    aortic : bool
        ``True`` when the point is at an aortic position.
    """

    frame_index: int
    point_index: int
    x: float
    y: float
    z: float
    aortic: bool

    def __init__(
        self,
        frame_index: int,
        point_index: int,
        x: float,
        y: float,
        z: float,
        aortic: bool,
    ) -> None: ...
    def distance(self, other: PyContourPoint) -> float: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class PyContour:
    """A closed 3-D contour consisting of ordered contour points.

    Attributes
    ----------
    id : int
        Contour identifier.
    original_frame : int
        Frame index from which this contour originates.
    points : list of PyContourPoint
        Ordered list of contour points.
    centroid : tuple of float
        ``(x, y, z)`` centroid coordinates.
    aortic_thickness : float or None
        Aortic wall thickness at this contour, if available.
    pulmonary_thickness : float or None
        Pulmonary wall thickness at this contour, if available.
    kind : str
        String representation of the contour type (e.g. ``"Lumen"``).
    """

    id: int
    original_frame: int
    points: list[PyContourPoint]
    centroid: tuple[float, float, float]
    aortic_thickness: float | None
    pulmonary_thickness: float | None
    kind: str

    def __init__(
        self,
        id: int,
        original_frame: int,
        points: list[PyContourPoint],
        centroid: tuple[float, float, float],
        aortic_thickness: float | None,
        pulmonary_thickness: float | None,
        kind: str,
    ) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def compute_centroid(self) -> None: ...
    def points_as_tuples(self) -> list[tuple[float, float, float]]: ...
    def find_farthest_points(
        self,
    ) -> tuple[tuple[PyContourPoint, PyContourPoint], float]: ...
    def find_closest_opposite(
        self,
    ) -> tuple[tuple[PyContourPoint, PyContourPoint], float]: ...
    def get_elliptic_ratio(self) -> float: ...
    def get_area(self) -> float: ...
    def rotate(self, angle_deg: float) -> PyContour: ...
    def translate(self, dx: float, dy: float, dz: float) -> PyContour: ...
    def sort_contour_points(self) -> PyContour: ...

class PyRecord:
    """Per-frame measurement record.

    Attributes
    ----------
    frame : int
        Frame number within the acquisition sequence.
    phase : str
        Cardiac phase: ``"D"`` for diastole or ``"S"`` for systole.
    measurement_1 : float or None
        Primary measurement value (e.g. aortic wall thickness).
    measurement_2 : float or None
        Secondary measurement value (e.g. pulmonary wall thickness).
    """

    frame: int
    phase: str
    measurement_1: float | None
    measurement_2: float | None

    def __init__(
        self,
        frame: int,
        phase: str,
        measurement_1: float | None,
        measurement_2: float | None,
    ) -> None: ...
    def __repr__(self) -> str: ...

class PyFrame:
    """A single intravascular imaging frame.

    Attributes
    ----------
    id : int
        Frame identifier.
    centroid : tuple of float
        ``(x, y, z)`` centroid of the frame.
    lumen : PyContour
        Lumen contour for this frame.
    extras : dict of str to PyContour
        Additional contour types keyed by name (e.g. ``"Eem"``, ``"Wall"``).
    reference_point : PyContourPoint or None
        Reference position used for alignment, if available.
    """

    id: int
    centroid: tuple[float, float, float]
    lumen: PyContour
    extras: dict[str, PyContour]
    reference_point: PyContourPoint | None

    def __init__(
        self,
        id: int,
        centroid: tuple[float, float, float],
        lumen: PyContour,
        extras: dict[str, PyContour],
        reference_point: PyContourPoint | None,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def rotate(self, angle_deg: float) -> PyFrame: ...
    def translate(self, dx: float, dy: float, dz: float) -> PyFrame: ...
    def sort_frame_points(self) -> PyFrame: ...

class PyGeometry:
    """A full intravascular imaging geometry (sequence of frames).

    Attributes
    ----------
    frames : list of PyFrame
        Ordered list of imaging frames.
    label : str
        Human-readable label for this geometry.
    """

    frames: list[PyFrame]
    label: str

    def __init__(self, frames: list[PyFrame], label: str) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def get_contours_by_type(self, contour_type: str) -> list[PyContour]: ...
    def get_lumen_contours(self) -> list[PyContour]: ...
    def get_contours(self, contour_type: str) -> list[PyContour]: ...
    def rotate(self, angle_deg: float) -> PyGeometry: ...
    def translate(self, dx: float, dy: float, dz: float) -> PyGeometry: ...
    def smooth_frames(self) -> PyGeometry: ...
    def get_summary(self) -> tuple[float, float, float]: ...
    def center_to_contour(self, contour_type: PyContourType) -> PyGeometry: ...
    def get_frame_at_z(self, z: float) -> PyFrame: ...
    def get_frame_at_index(self, index: int) -> PyFrame: ...
    def replace_frame(self, index: int, frame: PyFrame) -> PyGeometry: ...
    def downsample(self, n_points: int) -> PyGeometry: ...
    def sort_frame_points(self) -> PyGeometry: ...

class PyGeometryPair:
    """A diastolic/systolic geometry pair.

    Attributes
    ----------
    geom_a : PyGeometry
        First geometry (typically diastolic).
    geom_b : PyGeometry
        Second geometry (typically systolic).
    label : str
        Human-readable label.
    """

    geom_a: PyGeometry
    geom_b: PyGeometry
    label: str

    def __init__(self, geom_a: PyGeometry, geom_b: PyGeometry, label: str) -> None: ...
    def __repr__(self) -> str: ...
    def get_summary(
        self,
    ) -> tuple[
        tuple[tuple[float, float, float], tuple[float, float, float]], list[list[float]]
    ]: ...

class PyCenterlinePoint:
    """A point on a vessel centerline with its local normal vector.

    Attributes
    ----------
    contour_point : PyContourPoint
        Position of the centerline point in 3-D space.
    tangent : tuple of float
        Tangent vector ``(tx, ty, tz)`` at this centerline position.
    branch_id : int
        Branch identifier (0 for main vessel, 1+ for side branches).
    radius : float
        Local vessel radius at this centerline point, if available.
    """

    contour_point: PyContourPoint
    tangent: tuple[float, float, float]
    branch_id: int
    radius: float

    def __init__(
        self,
        contour_point: PyContourPoint,
        tangent: tuple[float, float, float],
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class PyCenterline:
    """A vessel centerline consisting of ordered centerline points.

    Attributes
    ----------
    points : list of PyCenterlinePoint
        Ordered list of centerline points.
    branch_start_indices : list of int
        Index into ``points`` where each branch begins.
        ``branch_start_indices[0]`` is always 0 (main vessel).
    """

    points: list[PyCenterlinePoint]
    branch_start_indices: list[int]

    def __init__(self, points: list[PyCenterlinePoint]) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    @staticmethod
    def from_contour_points(contour_points: list[PyContourPoint]) -> PyCenterline: ...
    def points_as_tuples(self) -> list[tuple[float, float, float]]: ...
    def calculate_branches(self, spacing_tolerance: float = 1.0) -> PyCenterline: ...
    def find_sharp_angles(self, branch_id: int, cos_threshold: float) -> list[int]:
        """Return global point_index values where the opening angle is sharp.

        Parameters
        ----------
        branch_id : int
            Branch to inspect (0 = main vessel).
        cos_threshold : float
            Cosine above which an angle is considered sharp.
            Use ``0.0`` for < 90°, ``0.5`` for < 60°, ``0.866`` for < 30°.

        Returns
        -------
        list[int]
            ``point_index`` values (indices into ``points``) where sharp
            angles were found, suitable for ``split_branch``.
        """
        ...

    def split_branch(self, branch_id: int, point_index: int) -> PyCenterline:
        """Split a branch at a point and return the updated centerline.

        Both resulting segments share the split point. Branches are re-sorted
        by descending length afterwards, so branch 0 is always the longest
        overall - the same invariant ``calculate_branches`` establishes.

        Parameters
        ----------
        branch_id : int
        point_index : int
            Global index into ``points`` (as returned by ``find_sharp_angles``)
            where the split occurs. Must fall within `branch_id`'s own range.
        """
        ...

    def merge_branches(self, branch_id_a: int, branch_id_b: int) -> PyCenterline:
        """Merge two branches into one and return the updated centerline.

        Segments are joined at the closest endpoint pair. Branches are
        re-sorted by descending length afterwards, so branch 0 is always the
        longest overall - the same invariant ``calculate_branches`` establishes.

        Parameters
        ----------
        branch_id_a : int
        branch_id_b : int
        """
        ...

    def get_branch(self, branch_id: int) -> PyCenterline:
        """Return a new centerline containing only the points of one branch.

        All retained points are reassigned to branch_id=0 and
        branch_start_indices is reset to [0].

        Parameters
        ----------
        branch_id : int
            Branch to extract.

        Raises
        ------
        ValueError
            If branch_id does not exist in this centerline.
        """
        ...

    def remove_branch_overlap(self) -> PyCenterline:
        """Remove the run-alongside-main-branch prefix duplicated by every side branch.

        Some centerline export formats (e.g. VTP) write every branch starting
        from the vessel origin, so side branches share a common prefix with
        branch 0. This trims that prefix from each side branch, keeping the
        bifurcation junction point and the diverged portion. Branches that
        overlap branch 0 entirely are dropped. The trim threshold is one mean
        inter-point spacing of branch 0.

        Returns
        -------
        PyCenterline
            New centerline with overlapping prefixes removed from all side
            branches.
        """
        ...

    def trim_start(self, mm: float) -> PyCenterline:
        """Trim `mm` of arc length off the start of branch 0.

        Useful when the main branch starts at the aortic inlet and the
        proximal region is outside the region of interest.

        Parameters
        ----------
        mm : float
            Arc-length in mm to remove from the start of branch 0.

        Returns
        -------
        PyCenterline
            New centerline with the inlet trimmed from branch 0.
        """
        ...

    def resample(self, spacing_mm: float) -> PyCenterline:
        """Resample every branch independently to even arc-length spacing.

        Interior points are linearly interpolated (position and radius)
        between the two nearest original points; tangents are recomputed
        afterwards. No interpolation occurs across a bifurcation.

        Parameters
        ----------
        spacing_mm : float
            Target arc-length spacing in mm between consecutive points.

        Returns
        -------
        PyCenterline
            New centerline resampled to even spacing per branch.
        """
        ...

    def smooth(self, sigma: float) -> PyCenterline:
        """Smooth centerline positions with a Gaussian kernel (per branch).

        `sigma` is the half-width in number of centerline points. A value of
        ``1.0`` is a gentle neighbourhood average; ``2-5`` removes noise while
        keeping the overall vessel path; larger values heavily round corners.
        Branches are processed independently so no smoothing bleeds across a
        bifurcation.

        Parameters
        ----------
        sigma : float
            Half-width of the Gaussian kernel in number of centerline points.

        Returns
        -------
        PyCenterline
            New centerline with smoothed positions and recomputed tangents.
        """
        ...

    def orient_by_max_z(self) -> PyCenterline:
        """Reverse branch 0 if its highest-z point isn't already at its start,
        then apply the same "closer end goes first" rule to every side branch,
        using branch 0 (post-reversal) as the reference.

        For centerlines with no anatomical reference to orient against, e.g.
        the aorta — use ``orient_to_reference`` instead whenever one is
        available. Only correct under the standard CT/DICOM convention where
        z increases toward the head, so the aortic root/valve is the
        highest-z point.

        Returns
        -------
        PyCenterline
            New centerline with all branches in canonical order.
        """
        ...

    def orient_to_reference(self, reference: PyCenterline) -> PyCenterline:
        """Reverse branch 0 so the end nearer `reference`'s branch 0 is its
        start, then apply the same rule to every side branch.

        Any side branches `reference` has are ignored so a stray one can't
        skew the distance check — only `reference`'s branch 0 is ever
        measured against. Distance to `reference` is the minimum distance to
        any point of its branch 0, not a single fixed point — e.g. for a
        coronary centerline, `reference` would be the aorta centerline, not
        one ostium point.

        Parameters
        ----------
        reference : PyCenterline
            Centerline to orient towards (e.g. the aorta, for a coronary
            centerline).

        Returns
        -------
        PyCenterline
            New centerline with all branches in canonical order.
        """
        ...

class PyInputData:
    """Intravascular imaging input data for one cardiac phase.

    Attributes
    ----------
    lumen : list of PyContour
        Vessel lumen contours.
    eem : list of PyContour or None
        EEM (external elastic membrane) contours.
    calcification : list of PyContour or None
        Calcification contours.
    sidebranch : list of PyContour or None
        Sidebranch contours.
    record : list of PyRecord or None
        Metadata records.
    ref_point : PyContourPoint
        Reference point used for alignment.
    diastole : bool
        ``True`` when the data corresponds to the diastolic phase.
    label : str
        Human-readable label for this dataset.
    """

    lumen: list[PyContour]
    eem: list[PyContour] | None
    calcification: list[PyContour] | None
    sidebranch: list[PyContour] | None
    record: list[PyRecord] | None
    ref_point: PyContourPoint
    diastole: bool
    label: str

    def __init__(
        self,
        lumen: list[PyContour],
        eem: list[PyContour] | None,
        calcification: list[PyContour] | None,
        sidebranch: list[PyContour] | None,
        record: list[PyRecord] | None,
        ref_point: PyContourPoint,
        diastole: bool,
        label: str,
    ) -> None: ...
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# CCTA mesh labelling and scaling functions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Processing functions — from CSV files
# ---------------------------------------------------------------------------

def from_file_full(
    input_path_ab: str,
    input_path_cd: str,
    labels: list[str] = ...,
    step_rotation_deg: float = ...,
    range_rotation_deg: float = ...,
    sample_size: int = ...,
    image_center: tuple[float, float] = ...,
    radius: float = ...,
    n_points: int = ...,
    write_obj: bool = ...,
    watertight: bool = ...,
    contour_types: list[PyContourType] | None = ...,
    output_path_ab: str = ...,
    output_path_cd: str = ...,
    output_path_ac: str = ...,
    output_path_bd: str = ...,
    interpolation_steps: int = ...,
    bruteforce: bool = ...,
    smooth: bool = ...,
    postprocessing: bool = ...,
) -> tuple[
    PyGeometryPair,
    PyGeometryPair,
    PyGeometryPair,
    PyGeometryPair,
    tuple[_AlignLog, _AlignLog, _AlignLog, _AlignLog],
]: ...
def from_file_doublepair(
    input_path_ab: str,
    input_path_cd: str,
    labels: list[str] = ...,
    step_rotation_deg: float = ...,
    range_rotation_deg: float = ...,
    sample_size: int = ...,
    image_center: tuple[float, float] = ...,
    radius: float = ...,
    n_points: int = ...,
    write_obj: bool = ...,
    watertight: bool = ...,
    contour_types: list[PyContourType] | None = ...,
    output_path_ab: str = ...,
    output_path_cd: str = ...,
    interpolation_steps: int = ...,
    bruteforce: bool = ...,
    smooth: bool = ...,
    postprocessing: bool = ...,
) -> tuple[
    PyGeometryPair,
    PyGeometryPair,
    tuple[_AlignLog, _AlignLog, _AlignLog, _AlignLog],
]: ...
def from_file_singlepair(
    input_path: str,
    labels: list[str] = ...,
    step_rotation_deg: float = ...,
    range_rotation_deg: float = ...,
    sample_size: int = ...,
    image_center: tuple[float, float] = ...,
    radius: float = ...,
    n_points: int = ...,
    write_obj: bool = ...,
    watertight: bool = ...,
    contour_types: list[PyContourType] | None = ...,
    output_path: str = ...,
    interpolation_steps: int = ...,
    bruteforce: bool = ...,
    smooth: bool = ...,
    postprocessing: bool = ...,
) -> tuple[PyGeometryPair, tuple[_AlignLog, _AlignLog]]: ...
def from_file_single(
    input_path: str,
    labels: list[str] = ...,
    diastole: bool = ...,
    step_rotation_deg: float = ...,
    range_rotation_deg: float = ...,
    sample_size: int = ...,
    image_center: tuple[float, float] = ...,
    radius: float = ...,
    n_points: int = ...,
    write_obj: bool = ...,
    watertight: bool = ...,
    contour_types: list[PyContourType] | None = ...,
    output_path: str = ...,
    bruteforce: bool = ...,
    smooth: bool = ...,
) -> tuple[PyGeometry, _AlignLog]: ...

# ---------------------------------------------------------------------------
# Processing functions — from PyInputData arrays
# ---------------------------------------------------------------------------

def from_array_full(
    input_data_a: PyInputData,
    input_data_b: PyInputData,
    input_data_c: PyInputData,
    input_data_d: PyInputData,
    step_rotation_deg: float = ...,
    range_rotation_deg: float = ...,
    sample_size: int = ...,
    image_center: tuple[float, float] = ...,
    radius: float = ...,
    n_points: int = ...,
    write_obj: bool = ...,
    watertight: bool = ...,
    contour_types: list[PyContourType] | None = ...,
    output_path_ab: str = ...,
    output_path_cd: str = ...,
    output_path_ac: str = ...,
    output_path_bd: str = ...,
    interpolation_steps: int = ...,
    bruteforce: bool = ...,
    smooth: bool = ...,
    postprocessing: bool = ...,
) -> tuple[
    PyGeometryPair,
    PyGeometryPair,
    PyGeometryPair,
    PyGeometryPair,
    tuple[_AlignLog, _AlignLog, _AlignLog, _AlignLog],
]: ...
def from_array_doublepair(
    input_data_a: PyInputData,
    input_data_b: PyInputData,
    input_data_c: PyInputData,
    input_data_d: PyInputData,
    step_rotation_deg: float = ...,
    range_rotation_deg: float = ...,
    sample_size: int = ...,
    image_center: tuple[float, float] = ...,
    radius: float = ...,
    n_points: int = ...,
    write_obj: bool = ...,
    watertight: bool = ...,
    contour_types: list[PyContourType] | None = ...,
    output_path_ab: str = ...,
    output_path_cd: str = ...,
    interpolation_steps: int = ...,
    bruteforce: bool = ...,
    smooth: bool = ...,
    postprocessing: bool = ...,
) -> tuple[
    PyGeometryPair,
    PyGeometryPair,
    tuple[_AlignLog, _AlignLog, _AlignLog, _AlignLog],
]: ...
def from_array_singlepair(
    input_data_a: PyInputData,
    input_data_b: PyInputData,
    step_rotation_deg: float = ...,
    range_rotation_deg: float = ...,
    sample_size: int = ...,
    image_center: tuple[float, float] = ...,
    radius: float = ...,
    n_points: int = ...,
    write_obj: bool = ...,
    watertight: bool = ...,
    contour_types: list[PyContourType] | None = ...,
    output_path: str = ...,
    interpolation_steps: int = ...,
    bruteforce: bool = ...,
    smooth: bool = ...,
    postprocessing: bool = ...,
) -> tuple[PyGeometryPair, tuple[_AlignLog, _AlignLog]]: ...
def from_array_single(
    input_data: PyInputData,
    step_rotation_deg: float = ...,
    range_rotation_deg: float = ...,
    sample_size: int = ...,
    image_center: tuple[float, float] = ...,
    radius: float = ...,
    n_points: int = ...,
    write_obj: bool = ...,
    watertight: bool = ...,
    contour_types: list[PyContourType] | None = ...,
    output_path: str = ...,
    bruteforce: bool = ...,
    smooth: bool = ...,
) -> tuple[PyGeometry, _AlignLog]: ...

# ---------------------------------------------------------------------------
# Alignment functions
# ---------------------------------------------------------------------------

def align_three_point(
    centerline: PyCenterline,
    geometry: PyGeometryPair | PyGeometry,
    main_ref_pt: tuple[float, float, float],
    counterclockwise_ref_pt: tuple[float, float, float],
    clockwise_ref_pt: tuple[float, float, float],
    angle_step_deg: float = ...,
    write: bool = ...,
    watertight: bool = ...,
    interpolation_steps: int = ...,
    output_dir: str = ...,
    contour_types: list[PyContourType] | None = ...,
    case_name: str = ...,
    align_wall_anomalous: bool = ...,
) -> tuple[PyGeometryPair | PyGeometry, float, float]: ...
def align_manual(
    centerline: PyCenterline,
    geometry: PyGeometryPair | PyGeometry,
    rotation_angle_deg: float,
    ref_point: tuple[float, float, float],
    write: bool = ...,
    watertight: bool = ...,
    interpolation_steps: int = ...,
    output_dir: str = ...,
    contour_types: list[PyContourType] | None = ...,
    case_name: str = ...,
    align_wall_anomalous: bool = ...,
) -> tuple[PyGeometryPair | PyGeometry, float, float]: ...
def align_combined(
    centerline: PyCenterline,
    geometry: PyGeometryPair | PyGeometry,
    main_ref_pt: tuple[float, float, float],
    counterclockwise_ref_pt: tuple[float, float, float],
    clockwise_ref_pt: tuple[float, float, float],
    points: list[tuple[float, float, float]],
    angle_step_deg: float = ...,
    angle_range_deg: float = ...,
    index_range: int = ...,
    write: bool = ...,
    watertight: bool = ...,
    interpolation_steps: int = ...,
    output_dir: str = ...,
    contour_types: list[PyContourType] | None = ...,
    case_name: str = ...,
    align_wall_anomalous: bool = ...,
) -> tuple[PyGeometryPair | PyGeometry, float, float]: ...

# ---------------------------------------------------------------------------
# OBJ export
# ---------------------------------------------------------------------------

def to_obj(
    geometry: PyGeometry,
    output_path: str,
    watertight: bool = ...,
    contour_types: list[PyContourType] | None = ...,
    filename_prefix: str = ...,
) -> None: ...

# ---------------------------------------------------------------------------
# VTP centerline reader
# ---------------------------------------------------------------------------

def read_centerline_vtp(path: str) -> PyCenterline: ...

# Implemented in Python (multimodars.ccta.centerline_prep), re-declared here for
# discoverability alongside the other centerline loading/prep functions.
def load_centerline(
    source: PyCenterline | Path | str | np.ndarray,
    name: str,
) -> PyCenterline:
    """Load a centerline from any supported source.

    Parameters
    ----------
    source : PyCenterline, Path, str, or numpy.ndarray
        A ``.vtp`` file path, a CSV file path (comma-delimited, columns
        x, y, z, ...), an existing ``PyCenterline`` (returned as-is), or an
        ``(N, 3+)`` array of points.
    name : str
        Label used in log output (e.g. ``"Aorta"``, ``"RCA"``, ``"LCA"``).

    Returns
    -------
    PyCenterline
        The loaded centerline, unprepared - pass it to
        :func:`prepare_centerline` next.
    """
    ...

def prepare_centerline(
    centerline: PyCenterline,
    ref_centerline: PyCenterline | None = ...,
    spacing_mm: float | None = ...,
    branch_spacing_tolerance: float = ...,
    rm_start_mm: float = ...,
    smooth_sigma: float = ...,
) -> PyCenterline:
    """Run the standard branch/order/smooth prep pipeline on a centerline.

    *ref_centerline* doubles as the "is this a coronary?" signal: the aorta has
    no upstream reference to orient against, while a coronary (RCA/LCA) orients
    towards the aorta's branch 0. Applies, in order:

    1. ``calculate_branches(branch_spacing_tolerance)`` - only when
       *ref_centerline* is given (i.e. *centerline* is a coronary) and
       *centerline* does not already carry branch structure (e.g. it came
       from a ``.vtp`` file, which reports its branches directly). The aorta
       (*ref_centerline* is ``None``) never needs branch detection.
    2. ``remove_branch_overlap()`` - trims the run-alongside-main-branch prefix
       some centerline export formats (e.g. VTP) attach to every side branch.
       A no-op for a single-branch centerline.
    3. ``trim_start(rm_start_mm)`` - only if ``rm_start_mm > 0``.
    4. ``resample(spacing_mm)`` - only if ``spacing_mm`` is given.
    5. ``orient_to_reference(ref_centerline)`` if *ref_centerline* is given,
       otherwise ``orient_by_max_z()`` - normalise branch ordering.
    6. ``smooth(smooth_sigma)`` - only if ``smooth_sigma > 0``.

    Parameters
    ----------
    centerline : PyCenterline
        Centerline to prepare, e.g. the output of :func:`load_centerline`.
    ref_centerline : PyCenterline, optional
        Reference centerline to orient towards (e.g. the aorta, for a
        coronary centerline). ``None`` (default) means *centerline* has no
        reference (e.g. it is the aorta itself).
    spacing_mm : float, optional
        Target arc-length spacing in mm passed to ``resample``. ``None``
        (default) skips resampling.
    branch_spacing_tolerance : float, optional
        Passed to ``calculate_branches`` when branch extraction is needed.
        Default ``1.0``.
    rm_start_mm : float, optional
        Arc-length in mm to trim from the start of branch 0 (e.g. the aortic
        inlet region). Default ``0.0`` (no trim).
    smooth_sigma : float, optional
        Half-width of the Gaussian smoothing kernel in number of centerline
        points. Default ``2.5``. Set to ``0.0`` to skip smoothing.

    Returns
    -------
    PyCenterline
        The prepared centerline.
    """
    ...

# ---------------------------------------------------------------------------
# CCTA mesh labelling and scaling functions
# ---------------------------------------------------------------------------

def remove_occluded_points_ray_triangle(
    centerline_coronary: PyCenterline,
    centerline_aorta: PyCenterline,
    range_mm: float,
    points: list[tuple[float, float, float]],
    faces: list[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ],
    step_size_mm: float,
) -> list[tuple[float, float, float]]: ...
def adjust_diameter_centerline_morphing_simple(
    centerline: PyCenterline,
    points: list[tuple[float, float, float]],
    diameter_adjustment_mm: float,
) -> list[tuple[float, float, float]]: ...
def find_points_by_cl_region(
    centerline: PyCenterline,
    frames: list[PyFrame],
    points: list[tuple[float, float, float]],
) -> tuple[
    list[tuple[float, float, float]],
    list[tuple[float, float, float]],
    list[tuple[float, float, float]],
]: ...
def clean_outlier_points(
    points_to_cleanup: list[tuple[float, float, float]],
    reference_points: list[tuple[float, float, float]],
    neighborhood_radius: float,
    min_neigbor_ratio: float,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]: ...
def find_proximal_distal_scaling(
    anomalous_points: list[tuple[float, float, float]],
    n_proximal: int,
    n_distal: int,
    centerline: PyCenterline,
    proximal_reference: list[tuple[float, float, float]],
    distal_reference: list[tuple[float, float, float]],
) -> tuple[float, float]: ...
def find_aortic_scaling(
    intramural_points: list[tuple[float, float, float]],
    reference_points: list[tuple[float, float, float]],
    centerline: PyCenterline,
) -> float: ...
def find_aortic_wall_scaling(
    centerline: PyCenterline,
    ref_pt_coronary: tuple[float, float, float],
    aortic_pts: list[tuple[float, float, float]],
) -> float: ...
def find_centerline_bounded_points_simple(
    centerline: PyCenterline,
    points: list[tuple[float, float, float]],
    radius: float,
) -> list[tuple[float, float, float]]: ...
def find_faces_near_points(
    vertices: list[tuple[float, float, float]],
    faces: list[list[int]],
    points: list[tuple[float, float, float]],
    tol: float = ...,
) -> list[
    tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
]: ...
def find_aortic_points(
    vertices: list[tuple[float, float, float]],
    points_a: list[tuple[float, float, float]],
    points_b: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]: ...
def final_reclassification(
    vertices: list[tuple[float, float, float]],
    faces: list[list[int]],
    rca_points: list[tuple[float, float, float]],
    lca_points: list[tuple[float, float, float]],
    rca_removed_points: list[tuple[float, float, float]],
    lca_removed_points: list[tuple[float, float, float]],
) -> tuple[
    list[tuple[float, float, float]],
    list[tuple[float, float, float]],
    list[tuple[float, float, float]],
    list[tuple[float, float, float]],
    list[tuple[float, float, float]],
]: ...
def build_adjacency_map(
    faces: list[list[int]],
) -> dict[int, set[int]]: ...
def keep_largest_connected_component(
    vertices: list[tuple[float, float, float]],
    faces: list[list[int]],
    points: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]: ...
def fix_mesh_winding(
    faces: list[list[int]],
) -> list[list[int]]: ...
def smooth_mesh_labels(
    labels: list[int],
    adjacency_map: dict[int, set[int]],
    iterations: int,
) -> list[int]: ...
def discretize_vessel(
    centerline: PyCenterline,
    points: list[tuple[float, float, float]],
    branch_id: int = ...,
    step_size: float = ...,
    n_points: int = ...,
) -> list[PyContour]: ...

# (main_ref, counter_clock_ref, clock_ref) — each is an (x, y, z) tuple.
_RefTriplet = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]

class PyDiscretizedVesselTree:
    """Fully discretized coronary vessel tree.

    Attributes
    ----------
    discretized_aorta : list of PyContour
        Cross-sectional contours along the aorta.
    discretized_rca_main : list of PyContour
        Cross-sectional contours along the RCA main vessel.
    discretized_lca_main : list of PyContour
        Cross-sectional contours along the LCA main vessel.
    rca_branches : list of list of PyContour
        Per-side-branch contours for the RCA.  ``rca_branches[i]`` →
        branch_id ``i + 1``.
    lca_branches : list of list of PyContour
        Per-side-branch contours for the LCA.
    rca_references : list of (main_ref, counter_clock_ref, clock_ref)
        Orientation triplets along the RCA, sorted proximal → distal.
        Each element is ``((x,y,z), (x,y,z), (x,y,z))``.
        Index 0 is always the ostium reference.
    lca_references : list of (main_ref, counter_clock_ref, clock_ref)
        Same structure for the LCA.
    ao_rca : tuple[float, float, float]
        Centroid ``(x, y, z)`` of the aorta slice closest to the RCA ostium.
    ao_lca : tuple[float, float, float]
        Centroid ``(x, y, z)`` of the aorta slice closest to the LCA ostium.
    """

    discretized_aorta: list[PyContour]
    discretized_rca_main: list[PyContour]
    discretized_lca_main: list[PyContour]
    rca_branches: list[list[PyContour]]
    lca_branches: list[list[PyContour]]
    rca_references: list[_RefTriplet]
    lca_references: list[_RefTriplet]
    ao_rca: tuple[float, float, float]
    ao_lca: tuple[float, float, float]

    def __repr__(self) -> str: ...
    def calculate_ref_pts(self) -> None: ...

def discretize_vessel_tree(
    ao_cl: PyCenterline,
    rca_cl: PyCenterline,
    lca_cl: PyCenterline,
    points_ao: list[tuple[float, float, float]],
    points_rca_main: list[tuple[float, float, float]],
    points_lca_main: list[tuple[float, float, float]],
    side_branches_rca: list[list[tuple[float, float, float]]],
    side_branches_lca: list[list[tuple[float, float, float]]],
    branch_id_rca: int = ...,
    branch_id_lca: int = ...,
    step_size: float = ...,
    n_points: int = ...,
    calculate_ref_pts: bool = ...,
) -> PyDiscretizedVesselTree: ...
