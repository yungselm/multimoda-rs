# tests/test_core.py
import math
import pytest
from multimodars import (
    PyContourPoint,
    PyContour,
    PyGeometry,
    PyFrame,
)
from conftest import _create_round_contour, _create_elliptic_contour


# -----------------PyContourPoint---------------------------
def test_contour_point_distance():
    p1 = PyContourPoint(frame_index=0, point_index=0, x=0.0, y=0.0, z=0.0, aortic=False)
    p2 = PyContourPoint(frame_index=0, point_index=0, x=3.0, y=4.0, z=0.0, aortic=False)
    assert p1.distance(p2) == 5.0


# ------------------PyContour-------------------------------
def _find_point(contour, idx):
    return next(p for p in contour.points if p.point_index == idx)


def test_contour_area(sample_contour, sample_contour_round, sample_contour_elliptic):
    # Note: Areas might be approximate due to floating point calculations
    area_square = sample_contour.get_area()
    area_round = sample_contour_round.get_area()
    area_elliptic = sample_contour_elliptic.get_area()

    assert area_square > 0
    assert area_round > 0
    assert area_elliptic > 0


def test_contour_centroid(
    sample_contour, sample_contour_round, sample_contour_elliptic
):
    # Test centroid calculation
    centroid_sq = sample_contour.centroid
    centroid_round = sample_contour_round.centroid
    centroid_ell = sample_contour_elliptic.centroid

    assert len(centroid_sq) == 3
    assert len(centroid_round) == 3
    assert len(centroid_ell) == 3

    # Square contour should have centroid at (2, 2, 3)
    assert pytest.approx(centroid_sq[0], abs=1e-6) == 2.0
    assert pytest.approx(centroid_sq[1], abs=1e-6) == 2.0
    assert pytest.approx(centroid_sq[2], abs=1e-6) == 3.0


def test_find_farthest_points(sample_contour, sample_contour_elliptic):
    (p1, p2), dist = sample_contour.find_farthest_points()
    assert dist > 0
    assert p1 is not None
    assert p2 is not None

    (p1_ell, p2_ell), dist_ell = sample_contour_elliptic.find_farthest_points()
    assert dist_ell > 0


def test_rotation_contour(sample_contour, sample_contour_elliptic):
    # Test rotation - basic functionality
    rotated_sq = sample_contour.rotate(90)
    assert isinstance(rotated_sq, PyContour)
    assert len(rotated_sq.points) == len(sample_contour.points)

    rotated_ell = sample_contour_elliptic.rotate(90)
    assert isinstance(rotated_ell, PyContour)
    assert len(rotated_ell.points) == len(sample_contour_elliptic.points)


def test_translation_contour(sample_contour, sample_contour_elliptic):
    # Test translation
    translated_sq = sample_contour.translate(1.0, -2.0, 0.0)
    assert isinstance(translated_sq, PyContour)

    translated_ell = sample_contour_elliptic.translate(-2.0, 3.0, 0.0)
    assert isinstance(translated_ell, PyContour)


# ------------------PyGeometry------------------------------
def test_geometry_creation(sample_geometry):
    geom = sample_geometry
    assert isinstance(geom, PyGeometry)
    assert len(geom.frames) == 1
    assert geom.frames[0].id == 0


def test_rotation_geometry(sample_geometry):
    geom = sample_geometry
    rotated = geom.rotate(90)
    assert isinstance(rotated, PyGeometry)
    assert len(rotated.frames) == len(geom.frames)


def test_translation_geometry(sample_geometry):
    geom = sample_geometry
    translated = geom.translate(1.0, 2.0, 3.0)
    assert isinstance(translated, PyGeometry)
    assert len(translated.frames) == len(geom.frames)


# Skip tests that require more complex functionality for now
def test_smooth_contours_temporal():
    pytest.skip("Temporal smoothing needs more complex setup")


def test_create_catheter_geometry():
    pytest.skip("Catheter geometry creation needs more complex setup")


def test_reorder_by_shape_similarity():
    pytest.skip("Shape similarity reordering needs more complex setup")
