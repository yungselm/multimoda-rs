# tests/conftest.py
import pytest
import numpy as np
from multimodars import PyContour, PyContourPoint, PyGeometry, PyFrame
from multimodars._converters import numpy_to_geometry


def _compute_centroid(points):
    """Compute centroid from a list of PyContourPoints"""
    if not points:
        return (0.0, 0.0, 0.0)

    sum_x = sum(p.x for p in points)
    sum_y = sum(p.y for p in points)
    sum_z = sum(p.z for p in points)
    n = len(points)
    return (sum_x / n, sum_y / n, sum_z / n)


def _create_round_contour(
    contour_id: int,
    center: tuple = (0.0, 0.0, 3.0),
    radius: float = 4.0,
    num_points: int = 20,
    aortic: bool = True,
) -> PyContour:
    """
    Generate a circular contour in the XY plane with fixed Z.
    """
    cx, cy, cz = center
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = []
    for idx, theta in enumerate(angles):
        x = cx + radius * np.cos(theta)
        y = cy + radius * np.sin(theta)
        pt = PyContourPoint(
            frame_index=contour_id, point_index=idx, x=x, y=y, z=cz, aortic=aortic
        )
        points.append(pt)

    centroid = _compute_centroid(points)
    contour = PyContour(
        id=contour_id,
        original_frame=contour_id,
        points=points,
        centroid=centroid,
        aortic_thickness=None,
        pulmonary_thickness=None,
        kind="Lumen",
    )
    return contour


def _create_elliptic_contour(
    contour_id: int,
    center: tuple = (2.0, 2.0, 3.0),
    major_radius: float = 5.0,
    minor_radius: float = 3.0,
    num_points: int = 20,
    aortic: bool = False,
) -> PyContour:
    """
    Generate an elliptical contour in the XY plane with fixed Z.
    """
    cx, cy, cz = center
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = []
    for idx, theta in enumerate(angles):
        x = cx + major_radius * np.cos(theta)
        y = cy + minor_radius * np.sin(theta)
        pt = PyContourPoint(
            frame_index=contour_id, point_index=idx, x=x, y=y, z=cz, aortic=aortic
        )
        points.append(pt)

    centroid = _compute_centroid(points)
    contour = PyContour(
        id=contour_id,
        original_frame=contour_id,
        points=points,
        centroid=centroid,
        aortic_thickness=None,
        pulmonary_thickness=None,
        kind="Lumen",
    )
    return contour


@pytest.fixture
def sample_contour_points():
    return [
        PyContourPoint(frame_index=0, point_index=0, x=0.0, y=4.0, z=3.0, aortic=False),
        PyContourPoint(frame_index=0, point_index=1, x=0.0, y=0.0, z=3.0, aortic=False),
        PyContourPoint(frame_index=0, point_index=2, x=4.0, y=0.0, z=3.0, aortic=True),
        PyContourPoint(frame_index=0, point_index=3, x=4.0, y=4.0, z=3.0, aortic=True),
    ]


@pytest.fixture
def sample_contour(sample_contour_points):
    centroid = _compute_centroid(sample_contour_points)
    contour = PyContour(
        id=0,
        original_frame=0,
        points=sample_contour_points,
        centroid=centroid,
        aortic_thickness=None,
        pulmonary_thickness=None,
        kind="Lumen",
    )
    return contour


@pytest.fixture
def sample_contour_round():
    """Synthetic circular contour with 20 points"""
    contour = _create_round_contour(contour_id=1)
    return contour


@pytest.fixture
def sample_contour_elliptic():
    """Synthetic elliptical contour with 20 points"""
    contour = _create_elliptic_contour(contour_id=2)
    return contour


@pytest.fixture
def sample_geometry(sample_contour):
    # Create a simple frame with the contour as lumen
    frame = PyFrame(
        id=0,
        centroid=sample_contour.centroid,
        lumen=sample_contour,
        extras={},  # No extras for simple test
        reference_point=PyContourPoint(
            frame_index=0, point_index=0, x=0.0, y=0.0, z=0.0, aortic=False
        ),
    )

    return PyGeometry(frames=[frame], label="test_geometry")


@pytest.fixture
def sample_array_data():
    return np.array(
        [
            [
                [0, 1.0, 2.0, 3.0],
                [0, 4.0, 5.0, 6.0],
                [0, 0.0, 0.0, 0.0],
                [0, 0.0, 0.0, 0.0],
            ],
            [
                [1, 7.0, 8.0, 9.0],
                [1, 10.0, 11.0, 12.0],
                [0, 0.0, 0.0, 0.0],
                [0, 0.0, 0.0, 0.0],
            ],
        ],
        dtype=float,
    )


@pytest.fixture
def sample_rest_dia_arr():
    try:
        raw = np.genfromtxt(
            "data/fixtures/idealized_geometry/diastolic_contours.csv", delimiter=","
        )
        ref = np.genfromtxt(
            "data/fixtures/idealized_geometry/diastolic_reference_points.csv",
            delimiter=",",
        )
        return numpy_to_geometry(
            lumen_arr=raw,
            catheter_arr=np.zeros((0, 4)),
            wall_arr=np.zeros((0, 4)),
            reference_arr=ref,
        )
    except FileNotFoundError:
        pytest.skip("Test data files not found")


@pytest.fixture
def sample_rest_sys_arr():
    try:
        raw = np.genfromtxt(
            "data/fixtures/idealized_geometry/systolic_contours.csv", delimiter=","
        )
        ref = np.genfromtxt(
            "data/fixtures/idealized_geometry/systolic_reference_points.csv",
            delimiter=",",
        )
        return numpy_to_geometry(
            lumen_arr=raw,
            catheter_arr=np.zeros((0, 4)),
            wall_arr=np.zeros((0, 4)),
            reference_arr=ref,
        )
    except FileNotFoundError:
        pytest.skip("Test data files not found")


@pytest.fixture
def sample_stress_dia_arr():
    try:
        raw = np.genfromtxt(
            "data/fixtures/idealized_geometry/diastolic_contours.csv", delimiter=","
        )
        ref = np.genfromtxt(
            "data/fixtures/idealized_geometry/diastolic_reference_points.csv",
            delimiter=",",
        )
        return numpy_to_geometry(
            lumen_arr=raw,
            catheter_arr=np.zeros((0, 4)),
            wall_arr=np.zeros((0, 4)),
            reference_arr=ref,
        )
    except FileNotFoundError:
        pytest.skip("Test data files not found")


@pytest.fixture
def sample_stress_sys_arr():
    try:
        raw = np.genfromtxt(
            "data/fixtures/idealized_geometry/systolic_contours.csv", delimiter=","
        )
        ref = np.genfromtxt(
            "data/fixtures/idealized_geometry/systolic_reference_points.csv",
            delimiter=",",
        )
        return numpy_to_geometry(
            lumen_arr=raw,
            catheter_arr=np.zeros((0, 4)),
            wall_arr=np.zeros((0, 4)),
            reference_arr=ref,
        )
    except FileNotFoundError:
        pytest.skip("Test data files not found")
