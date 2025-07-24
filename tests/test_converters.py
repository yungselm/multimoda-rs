# tests/test_converters.py
import numpy as np
from multimodars._converters import *
from multimodars import PyGeometry

def test_geometry_to_numpy(sample_geometry):
    arr = geometry_to_numpy(sample_geometry)
    assert arr.shape == (2, 4, 4)
    assert arr[0, 0, 1] == 1.0  # First contour point x
    assert arr[1, 1, 3] == 12.0  # Catheter point z

def test_numpy_to_geometry_layers(sample_array_data):
    geom = numpy_to_geometry_layers(
        sample_array_data[:, 0, :],
        sample_array_data[:, 1, :],
        sample_array_data[:, 2, :],
        sample_array_data[:, 3, :]
    )
    assert len(geom.contours) == 1
    assert len(geom.contours[0].points) == 2
    assert geom.contours[0].points[1].y == 8.0

def test_numpy_to_geometry(sample_array_data):
    geom = numpy_to_geometry(sample_array_data)
    assert geom.reference_point.z == 0.0
    assert geom.catheter[0].points[0].x == 4.0

def test_numpy_to_centerline():
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    centerline = numpy_to_centerline(arr, True)
    assert len(centerline.points) == 2
    assert centerline.points[0].aortic
    assert centerline.points[1].x == 4.0