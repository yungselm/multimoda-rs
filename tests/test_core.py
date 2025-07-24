# tests/test_core.py
import pytest
from multimodars import PyContourPoint, PyContour, PyGeometry

def test_contour_point_distance():
    p1 = PyContourPoint(0, 0, 0.0, 0.0, 0.0, False)
    p2 = PyContourPoint(0, 0, 3.0, 4.0, 0.0, False)
    assert p1.distance(p2) == 5.0

def test_contour_operations(sample_contour):
    assert sample_contour.get_area() > 0
    rotated = sample_contour.rotate(45)
    assert rotated.centroid == sample_contour.centroid

def test_geometry_smoothing(sample_geometry):
    original_z = sample_geometry.contours[0].points[0].z
    sample_geometry.smooth_contours(window_size=1)
    assert sample_geometry.contours[0].points[0].z == original_z