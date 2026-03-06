"""
Type stubs for the multimodars Rust extension module.

These stubs describe the public interface of the compiled
``multimodars.multimodars`` extension so that type checkers (mypy, pyright)
and IDEs can offer autocompletion and type checking for downstream code.
"""

from __future__ import annotations

# Alignment log entry: (id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y)
# Rust: Vec<(u32, u32, f64, f64, f64, f64, f64)>
_AlignLog = list[tuple[int, int, float, float, float, float, float]]

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class PyContourType:
    """Enumeration of supported intravascular contour types."""

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

    def __init__(
        self, geom_a: PyGeometry, geom_b: PyGeometry, label: str
    ) -> None: ...
    def __repr__(self) -> str: ...
    def get_summary(
        self,
    ) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], list[list[float]]]: ...


class PyCenterlinePoint:
    """A point on a vessel centerline with its local normal vector.

    Attributes
    ----------
    contour_point : PyContourPoint
        Position of the centerline point in 3-D space.
    normal : tuple of float
        Normal vector ``(nx, ny, nz)`` at this centerline position.
    """

    contour_point: PyContourPoint
    normal: tuple[float, float, float]

    def __init__(
        self,
        contour_point: PyContourPoint,
        normal: tuple[float, float, float],
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...


class PyCenterline:
    """A vessel centerline consisting of ordered centerline points.

    Attributes
    ----------
    points : list of PyCenterlinePoint
        Ordered list of centerline points.
    """

    points: list[PyCenterlinePoint]

    def __init__(self, points: list[PyCenterlinePoint]) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    @staticmethod
    def from_contour_points(contour_points: list[PyContourPoint]) -> PyCenterline: ...
    def points_as_tuples(self) -> list[tuple[float, float, float]]: ...


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
# Processing functions — from CSV files
# ---------------------------------------------------------------------------

def from_file_full(
    input_path_a: str,
    input_path_b: str,
    label: str = "full",
    diastole: bool = True,
    step_rotation_deg: float = 0.5,
    range_rotation_deg: float = 90.0,
    image_center: tuple[float, float] = (4.5, 4.5),
    radius: float = 0.5,
    n_points: int = 20,
    write_obj: bool = True,
    watertight: bool = True,
    output_path_a: str = "output/rest",
    output_path_b: str = "output/stress",
    output_path_c: str = "output/diastole",
    output_path_d: str = "output/systole",
    interpolation_steps: int = 28,
    bruteforce: bool = False,
    sample_size: int = 500,
    contour_types: list[PyContourType] = ...,
    smooth: bool = True,
    postprocessing: bool = True,
) -> tuple[
    PyGeometryPair,
    PyGeometryPair,
    PyGeometryPair,
    PyGeometryPair,
    tuple[_AlignLog, _AlignLog, _AlignLog, _AlignLog],
]: ...


def from_file_doublepair(
    input_path_a: str,
    input_path_b: str,
    label: str = "double_pair",
    diastole: bool = True,
    step_rotation_deg: float = 0.5,
    range_rotation_deg: float = 90.0,
    image_center: tuple[float, float] = (4.5, 4.5),
    radius: float = 0.5,
    n_points: int = 20,
    write_obj: bool = True,
    watertight: bool = True,
    output_path_a: str = "output/rest",
    output_path_b: str = "output/stress",
    interpolation_steps: int = 28,
    bruteforce: bool = False,
    sample_size: int = 500,
    contour_types: list[PyContourType] = ...,
    smooth: bool = True,
    postprocessing: bool = True,
) -> tuple[
    PyGeometryPair,
    PyGeometryPair,
    tuple[_AlignLog, _AlignLog, _AlignLog, _AlignLog],
]: ...


def from_file_singlepair(
    input_path: str,
    label: str = "single_pair",
    diastole: bool = True,
    step_rotation_deg: float = 0.5,
    range_rotation_deg: float = 90.0,
    image_center: tuple[float, float] = (4.5, 4.5),
    radius: float = 0.5,
    n_points: int = 20,
    write_obj: bool = True,
    watertight: bool = True,
    output_path: str = "output/singlepair",
    interpolation_steps: int = 28,
    bruteforce: bool = False,
    sample_size: int = 500,
    contour_types: list[PyContourType] = ...,
    smooth: bool = True,
    postprocessing: bool = True,
) -> tuple[PyGeometryPair, tuple[_AlignLog, _AlignLog]]: ...


def from_file_single(
    input_path: str,
    label: str = "single",
    diastole: bool = True,
    step_rotation_deg: float = 0.5,
    range_rotation_deg: float = 90.0,
    image_center: tuple[float, float] = (4.5, 4.5),
    radius: float = 0.5,
    n_points: int = 20,
    write_obj: bool = True,
    watertight: bool = True,
    output_path: str = "output/single",
    bruteforce: bool = False,
    sample_size: int = 500,
    contour_types: list[PyContourType] = ...,
    smooth: bool = True,
) -> tuple[PyGeometry, _AlignLog]: ...

# ---------------------------------------------------------------------------
# Processing functions — from PyInputData arrays
# ---------------------------------------------------------------------------

def from_array_full(
    input_data_a: PyInputData,
    input_data_b: PyInputData,
    input_data_c: PyInputData,
    input_data_d: PyInputData,
    label: str = "full",
    diastole: bool = True,
    step_rotation_deg: float = 0.5,
    range_rotation_deg: float = 90.0,
    image_center: tuple[float, float] = (4.5, 4.5),
    radius: float = 0.5,
    n_points: int = 20,
    write_obj: bool = True,
    watertight: bool = True,
    output_path_a: str = "output/rest",
    output_path_b: str = "output/stress",
    output_path_c: str = "output/diastole",
    output_path_d: str = "output/systole",
    interpolation_steps: int = 28,
    bruteforce: bool = False,
    sample_size: int = 500,
    contour_types: list[PyContourType] = ...,
    smooth: bool = True,
    postprocessing: bool = True,
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
    label: str = "double_pair",
    diastole: bool = True,
    step_rotation_deg: float = 0.5,
    range_rotation_deg: float = 90.0,
    image_center: tuple[float, float] = (4.5, 4.5),
    radius: float = 0.5,
    n_points: int = 20,
    write_obj: bool = True,
    watertight: bool = True,
    output_path_a: str = "output/rest",
    output_path_b: str = "output/stress",
    interpolation_steps: int = 28,
    bruteforce: bool = False,
    sample_size: int = 500,
    contour_types: list[PyContourType] = ...,
    smooth: bool = True,
    postprocessing: bool = True,
) -> tuple[
    PyGeometryPair,
    PyGeometryPair,
    tuple[_AlignLog, _AlignLog, _AlignLog, _AlignLog],
]: ...


def from_array_singlepair(
    input_data_a: PyInputData,
    input_data_b: PyInputData,
    label: str = "single_pair",
    diastole: bool = True,
    step_rotation_deg: float = 0.5,
    range_rotation_deg: float = 90.0,
    image_center: tuple[float, float] = (4.5, 4.5),
    radius: float = 0.5,
    n_points: int = 20,
    write_obj: bool = True,
    watertight: bool = True,
    output_path: str = "output/singlepair",
    interpolation_steps: int = 28,
    bruteforce: bool = False,
    sample_size: int = 500,
    contour_types: list[PyContourType] = ...,
    smooth: bool = True,
    postprocessing: bool = True,
) -> tuple[PyGeometryPair, tuple[_AlignLog, _AlignLog]]: ...


def from_array_single(
    input_data: PyInputData,
    label: str = "single",
    diastole: bool = True,
    step_rotation_deg: float = 0.5,
    range_rotation_deg: float = 90.0,
    image_center: tuple[float, float] = (4.5, 4.5),
    radius: float = 0.5,
    n_points: int = 20,
    write_obj: bool = True,
    watertight: bool = True,
    output_path: str = "output/single",
    bruteforce: bool = False,
    sample_size: int = 500,
    contour_types: list[PyContourType] = ...,
    smooth: bool = True,
) -> tuple[PyGeometry, _AlignLog]: ...

# ---------------------------------------------------------------------------
# Alignment functions
# ---------------------------------------------------------------------------

def align_three_point(
    centerline: PyCenterline,
    geometry_pair: PyGeometryPair,
    aortic_ref_pt: tuple[float, float, float],
    upper_ref_pt: tuple[float, float, float],
    lower_ref_pt: tuple[float, float, float],
    angle_step_deg: float = 1.0,
    write: bool = False,
    watertight: bool = True,
    interpolation_steps: int = 28,
    output_dir: str = "output/aligned",
    contour_types: list[PyContourType] = ...,
    case_name: str = "None",
) -> tuple[PyGeometryPair, PyCenterline]: ...


def align_manual(
    centerline: PyCenterline,
    geometry_pair: PyGeometryPair,
    rotation_angle: float,
    ref_point: tuple[float, float, float],
    write: bool = False,
    watertight: bool = True,
    interpolation_steps: int = 28,
    output_dir: str = "output/aligned",
    contour_types: list[PyContourType] = ...,
    case_name: str = "None",
) -> tuple[PyGeometryPair, PyCenterline]: ...


def align_combined(
    centerline: PyCenterline,
    geom_pair: PyGeometryPair,
    aortic_ref_pt: tuple[float, float, float],
    upper_ref_pt: tuple[float, float, float],
    lower_ref_pt: tuple[float, float, float],
    points: list[tuple[float, float, float]],
    angle_step_deg: float = 1.0,
    angle_range_deg: float = 15.0,
    index_range: int = 2,
    write: bool = False,
    watertight: bool = True,
    interpolation_steps: int = 28,
    output_dir: str = "output/aligned",
    contour_types: list[PyContourType] = ...,
    case_name: str = "None",
) -> tuple[PyGeometryPair, PyCenterline]: ...

# ---------------------------------------------------------------------------
# OBJ export
# ---------------------------------------------------------------------------

def to_obj(
    geometry: PyGeometry,
    output_path: str,
    watertight: bool,
    contour_types: list[PyContourType],
    filename_prefix: str,
) -> None: ...

# ---------------------------------------------------------------------------
# CCTA mesh labelling and scaling functions
# ---------------------------------------------------------------------------

def find_centerline_bounded_points_simple(
    centerline: PyCenterline,
    points: list[tuple[float, float, float]],
    radius: float,
) -> list[tuple[float, float, float]]: ...


def remove_occluded_points_ray_triangle(
    centerline_coronary: PyCenterline,
    centerline_aorta: PyCenterline,
    range_coronary: int,
    points: list[tuple[float, float, float]],
    faces: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]],
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


def build_adjacency_map(
    faces: list[list[int]],
) -> dict[int, set[int]]: ...


def smooth_mesh_labels(
    labels: list[int],
    adjacency_map: dict[int, set[int]],
    iterations: int,
) -> list[int]: ...
