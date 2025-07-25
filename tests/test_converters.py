# tests/test_converters.py
import numpy as np
import pytest

from multimodars import (
    PyContourPoint,
    PyContour,
    PyCenterline,
    PyGeometry,
    PyGeometryPair,
)
from multimodars import (
    to_array,
    numpy_to_geometry,
    numpy_to_geometry_layers,
    numpy_to_centerline,
)


def _make_simple_contour(contour_id: int, n: int = 4, offset: float = 0.0):
    pts = [
        PyContourPoint(
            frame_index=contour_id,
            point_index=i,
            x=float(i) + offset,
            y=2.0 * float(i) + offset,
            z=3.0 * float(i) + offset,
            aortic=(i % 2 == 0),
        )
        for i in range(n)
    ]
    return PyContour(contour_id, pts)


def _make_simple_centerline(n: int = 5):
    # centerline wraps a list of contour_points
    pts = [
        PyContourPoint(
            frame_index=i,
            point_index=0,
            x=float(i),
            y=float(i) + 0.5,
            z=float(i) + 1.0,
            aortic=False,
        )
        for i in range(n)
    ]
    return PyCenterline.from_contour_points(pts)


def test_to_array_and_back_contour():
    c = _make_simple_contour(7, n=3, offset=1.0)
    arr = to_array(c)
    # Expect shape (3,4) columns: frame,x,y,z
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3, 4)
    # Round‑trip: build a geometry layer from this array
    # (we can piggy‑back numpy_to_geometry_layers)
    rebuilt = numpy_to_geometry_layers(
        contours_arr=arr,
        catheter_arr=np.empty((0, 4)),
        walls_arr=np.empty((0, 4)),
        reference_arr=np.array([0.0, 0.0, 0.0, 0.0])
    )
    # Should get one contour with 3 points
    assert len(rebuilt.contours) == 1
    pts = rebuilt.contours[0].points
    assert len(pts) == 3
    for orig, new in zip(c.points, pts):
        assert pytest.approx(orig.x) == new.x
        assert pytest.approx(orig.y) == new.y
        assert pytest.approx(orig.z) == new.z

def test_to_array_centerline_and_back():
    cl = _make_simple_centerline(n=4)
    arr = to_array(cl)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (4, 4)
    # Convert back
    new_cl = numpy_to_centerline(arr[:, 1:4])  # expects (N,3)
    assert isinstance(new_cl, PyCenterline)
    # Round‐trip via to_array: new_cl → array2
    arr2 = to_array(new_cl)
    assert isinstance(arr2, np.ndarray)
    assert arr2.shape == arr.shape
    # Compare spatial columns (x,y,z)
    np.testing.assert_allclose(arr2[:, 1:4], arr[:, 1:4], rtol=1e-6, atol=0)

def test_to_array_and_back_geometry_roundtrip():
    # Build a geometry with two contours, one catheter, one wall, one reference
    c0 = _make_simple_contour(0, n=2, offset=0.0)
    c1 = _make_simple_contour(1, n=3, offset=10.0)
    cat = _make_simple_contour(0, n=2, offset=-5.0)
    wall = _make_simple_contour(0, n=4, offset=2.0)
    ref = PyContourPoint(0, 0, 100.0, 101.0, 102.0, False)
    geom = PyGeometry(
        contours=[c0, c1],
        catheter=[cat],
        walls=[wall],
        reference_point=ref
    )

    arr = to_array(geom)
    # arr shape: (max_len, 4 layers, 4 coords)
    total_pts = len(c0.points) + len(c1.points)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (total_pts, 4, 4)

    # Round-trip via numpy_to_geometry
    geom2 = numpy_to_geometry(arr)
    # same number of contours/catheters/walls
    assert len(geom2.contours) == 2
    assert len(geom2.catheter) == 1
    assert len(geom2.walls) == 1
    # reference matches
    assert pytest.approx(geom2.reference_point.x) == ref.x
    assert pytest.approx(geom2.reference_point.y) == ref.y
    assert pytest.approx(geom2.reference_point.z) == ref.z

    # Each contour round-trips
    for orig, roundp in zip(geom.contours, geom2.contours):
        assert len(orig.points) == len(roundp.points)
        for o_pt, r_pt in zip(orig.points, roundp.points):
            assert pytest.approx(o_pt.x) == r_pt.x
            assert pytest.approx(o_pt.y) == r_pt.y
            assert pytest.approx(o_pt.z) == r_pt.z

def test_to_array_geometry_pair():
    g1 = PyGeometry(contours=[_make_simple_contour(0,n=1)], catheter=[], walls=[],
                    reference_point=PyContourPoint(0,0,0,0,0,False))
    g2 = PyGeometry(contours=[_make_simple_contour(1,n=2)], catheter=[], walls=[],
                    reference_point=PyContourPoint(0,0,1,1,1,False))
    pair = PyGeometryPair(dia_geom=g1, sys_geom=g2)
    arr1, arr2 = to_array(pair)
    # arr1 shape (1,4,4), arr2 shape (2,4,4)
    assert isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray)
    assert arr1.shape[0] == 1
    assert arr2.shape[0] == 2
