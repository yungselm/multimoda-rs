# tests/conftest.py
import pytest
import numpy as np
from multimodars import PyContour, PyContourPoint, PyGeometry

@pytest.fixture
def sample_contour_points():
    return [
        PyContourPoint(0, 0, 1.0, 2.0, 3.0, False),
        PyContourPoint(0, 1, 4.0, 5.0, 6.0, True)
    ]

@pytest.fixture
def sample_contour(sample_contour_points):
    return PyContour(0, sample_contour_points, (2.5, 3.5, 4.5))

@pytest.fixture
def sample_geometry(sample_contour):
    return PyGeometry(
        contours=[sample_contour],
        catheter=[sample_contour],
        walls=[sample_contour],
        reference_point=PyContourPoint(0, 0, 0.0, 0.0, 0.0, False)
    )

@pytest.fixture
def sample_array_data():
    return np.array([
        [[0, 1.0, 2.0, 3.0], [0, 4.0, 5.0, 6.0], [0, 0.0, 0.0, 0.0], [0, 0.0, 0.0, 0.0]],
        [[1, 7.0, 8.0, 9.0], [1, 10.0, 11.0, 12.0], [0, 0.0, 0.0, 0.0], [0, 0.0, 0.0, 0.0]]
    ], dtype=float)