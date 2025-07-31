# tests/conftest.py
import pytest
import numpy as np
from multimodars import PyContour, PyContourPoint, PyGeometry, numpy_to_geometry


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
        pt = PyContourPoint(0, idx, x, y, cz, aortic)
        points.append(pt)
    contour = PyContour(contour_id, points)
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
        pt = PyContourPoint(0, idx, x, y, cz, aortic)
        points.append(pt)
    contour = PyContour(contour_id, points)
    return contour


@pytest.fixture
def sample_contour_points():
    return [
        PyContourPoint(0, 0, 0.0, 4.0, 3.0, False),
        PyContourPoint(0, 1, 0.0, 0.0, 3.0, False),
        PyContourPoint(0, 2, 4.0, 0.0, 3.0, True),
        PyContourPoint(0, 3, 4.0, 4.0, 3.0, True),
    ]


@pytest.fixture
def sample_contour(sample_contour_points):
    contour = PyContour(0, sample_contour_points)
    # Check centroid is as expected
    assert contour.centroid == (2.0, 2.0, 3.0)
    return contour


@pytest.fixture
def sample_contour_round():
    """Synthetic circular contour with 20 points"""
    contour = _create_round_contour(contour_id=1)
    assert pytest.approx(contour.centroid, rel=1e-6) == (0.0, 0.0, 3.0)
    return contour


@pytest.fixture
def sample_contour_elliptic():
    """Synthetic elliptical contour with 20 points"""
    contour = _create_elliptic_contour(contour_id=2)
    assert pytest.approx(contour.centroid, rel=1e-6) == (2.0, 2.0, 3.0)
    return contour


@pytest.fixture
def sample_geometry(sample_contour):
    return PyGeometry(
        contours=[sample_contour],
        catheters=[sample_contour],
        walls=[sample_contour],
        reference_point=PyContourPoint(0, 0, 0.0, 0.0, 0.0, False),
    )


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
    raw = np.genfromtxt(
        "data/fixtures/idealized_geometry/diastolic_contours.csv", delimiter=","
    )
    ref = np.genfromtxt(
        "data/fixtures/idealized_geometry/diastolic_reference_points.csv", delimiter=","
    )
    return numpy_to_geometry(
        contours_arr=raw,
        catheters_arr=np.array([]),
        walls_arr=np.array([]),
        reference_arr=ref,
    )


@pytest.fixture
def sample_rest_sys_arr():
    raw = np.genfromtxt(
        "data/fixtures/idealized_geometry/systolic_contours.csv", delimiter=","
    )
    ref = np.genfromtxt(
        "data/fixtures/idealized_geometry/systolic_reference_points.csv", delimiter=","
    )
    return numpy_to_geometry(
        contours_arr=raw,
        catheters_arr=np.array([]),
        walls_arr=np.array([]),
        reference_arr=ref,
    )


@pytest.fixture
def sample_stress_dia_arr():
    raw = np.genfromtxt(
        "data/fixtures/idealized_geometry/diastolic_contours.csv", delimiter=","
    )
    ref = np.genfromtxt(
        "data/fixtures/idealized_geometry/diastolic_reference_points.csv", delimiter=","
    )
    return numpy_to_geometry(
        contours_arr=raw,
        catheters_arr=np.array([]),
        walls_arr=np.array([]),
        reference_arr=ref,
    )


@pytest.fixture
def sample_stress_sys_arr():
    raw = np.genfromtxt(
        "data/fixtures/idealized_geometry/systolic_contours.csv", delimiter=","
    )
    ref = np.genfromtxt(
        "data/fixtures/idealized_geometry/systolic_reference_points.csv", delimiter=","
    )
    return numpy_to_geometry(
        contours_arr=raw,
        catheters_arr=np.array([]),
        walls_arr=np.array([]),
        reference_arr=ref,
    )
