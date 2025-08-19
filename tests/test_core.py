# tests/test_core.py
import math

import pytest
from multimodars import (
    PyContourPoint,
    PyContour,
    PyGeometry,
    PyGeometryPair,
    create_catheter_geometry,
)
from conftest import _create_round_contour, _create_elliptic_contour

# -----------------PyContourPoint---------------------------
def test_contour_point_distance():
    p1 = PyContourPoint(0, 0, 0.0, 0.0, 0.0, False)
    p2 = PyContourPoint(0, 0, 3.0, 4.0, 0.0, False)
    assert p1.distance(p2) == 5.0


# ------------------PyContour-------------------------------
def _find_point(contour, idx):
    return next(p for p in contour.points if p.point_index == idx)


def test_contour_area(sample_contour, sample_contour_round, sample_contour_elliptic):
    assert sample_contour.get_area() == 16
    assert pytest.approx(sample_contour_round.get_area(), rel=1.0) == 50.27
    assert pytest.approx(sample_contour_elliptic.get_area(), rel=1.0) == 37.7


def test_contour_elliptic_ratio(
    sample_contour, sample_contour_round, sample_contour_elliptic
):
    assert sample_contour.get_elliptic_ratio() == 1.0
    assert pytest.approx(sample_contour_round.get_elliptic_ratio(), rel=1.0) == 1.0
    assert (
        pytest.approx(sample_contour_elliptic.get_elliptic_ratio(), rel=1.0)
        == 1.666666667
    )


def test_find_farthest_points(sample_contour, sample_contour_elliptic):
    (_, _), dist = sample_contour.find_farthest_points()
    assert pytest.approx(dist, rel=10e-6) == 5.656854249
    (_, _), dist = sample_contour_elliptic.find_farthest_points()
    assert dist == 10.0


def test_find_closest_opposite(sample_contour, sample_contour_elliptic):
    (_, _), dist = sample_contour.find_closest_opposite()
    assert pytest.approx(dist, rel=10e-6) == 5.656854249
    (_, _), dist = sample_contour_elliptic.find_closest_opposite()
    assert dist == 6.0


def test_rotation_contour(sample_contour, sample_contour_elliptic):
    # - Square contour -
    # Before rotation, point_index 0 is at (0, 4)
    p0_before = _find_point(sample_contour, 0)
    assert (p0_before.x, p0_before.y) == pytest.approx((0.0, 4.0))

    # Rotate 90° about its centroid (2,2)
    rotated_sq = sample_contour.rotate(90)
    p0_after = _find_point(rotated_sq, 0)
    # (0,4) around (2,2) → should go to (0,0)
    assert (p0_after.x, p0_after.y) == pytest.approx((0.0, 0.0))

    # - Elliptic contour -
    # centered at (2,2) with point 0 at angle=0 → (7,2)
    p0e_before = _find_point(sample_contour_elliptic, 0)
    assert (p0e_before.x, p0e_before.y) == pytest.approx((7.0, 2.0))

    # Rotate 90° about the same centroid (2,2)
    rotated_ell = sample_contour_elliptic.rotate(90)
    p0e_after = _find_point(rotated_ell, 0)
    # (7,2) around (2,2) → (2,7)
    assert (p0e_after.x, p0e_after.y) == pytest.approx((2.0, 7.0))


def test_translation_contour(sample_contour, sample_contour_elliptic):
    # - Square contour -
    p0_sq_before = _find_point(sample_contour, 0)
    assert (p0_sq_before.x, p0_sq_before.y) == pytest.approx((0.0, 4.0))

    translated_sq = sample_contour.translate(1.0, -2.0, 0.0)
    p0_sq_after = _find_point(translated_sq, 0)
    # (0+1, 4-2) → (1, 2)
    assert (p0_sq_after.x, p0_sq_after.y) == pytest.approx((1.0, 2.0))

    # - Elliptic contour -
    p0_el_before = _find_point(sample_contour_elliptic, 0)
    assert (p0_el_before.x, p0_el_before.y) == pytest.approx((7.0, 2.0))

    translated_el = sample_contour_elliptic.translate(-2.0, 3.0, 0.0)
    p0_el_after = _find_point(translated_el, 0)
    # (7-2, 2+3) → (5, 5)
    assert (p0_el_after.x, p0_el_after.y) == pytest.approx((5.0, 5.0))


# ------------------PyGeometry------------------------------
def test_smooth_contours_temporal():
    # build three circles with radii 10, 9, 10
    c1 = _create_round_contour(0, radius=10.0)
    c2 = _create_round_contour(1, radius=9.0)
    c3 = _create_round_contour(2, radius=10.0)

    # simplest Geometry has only contours and random reference
    ref = PyContourPoint(0, 0, 0.0, 0.0, 0.0, False)
    geom = PyGeometry(
        contours=[c1, c2, c3], catheters=[], walls=[], reference_point=ref
    )

    smoothed = geom.smooth_contours()
    p_mid = _find_point(smoothed.contours[1], 0)

    # the middle contour was radius 9; after smoothing: (10 + 9 + 10) / 3 = 9.666...
    expected = (10.0 + 9.0 + 10.0) / 3.0
    assert pytest.approx(p_mid.x, rel=1e-6) == expected
    # y should remain ~0
    assert pytest.approx(p_mid.y, abs=1e-6) == 0.0

    # check that the first and last contours used mirror logic:
    # first contour smoothed = (c1 + c1 + c2) / 3 = (10+10+9)/3 = 9.666...
    p0_first = _find_point(smoothed.contours[0], 0)
    assert pytest.approx(p0_first.x, rel=1e-6) == expected

    # last contour smoothed = (c2 + c3 + c3) / 3 = (9+10+10)/3 = 9.666...
    p0_last = _find_point(smoothed.contours[2], 0)
    assert pytest.approx(p0_last.x, rel=1e-6) == expected


def _radius_sequence(geom: PyGeometry):
    """Return a list of radii computed from centroid→point_index 0 for each contour."""
    seq = []
    for c in geom.contours:
        # find the point with index 0
        p0 = next(p for p in c.points if p.point_index == 0)
        # compute dist from centroid
        cx, cy, _ = c.centroid
        r = math.hypot(p0.x - cx, p0.y - cy)
        seq.append(r)
    return seq


def test_create_catheter_geometry():
    contours = [
        _create_round_contour(0, center=(1, 1, 0), radius=5.0),
        _create_round_contour(1, center=(2, 2, 1), radius=4.0),
        _create_round_contour(2, center=(3, 3, 2), radius=3.0),
    ]

    ref_point = PyContourPoint(0, 0, 0.0, 0.0, 0.0, False)
    geom = PyGeometry(
        contours=contours,
        catheters=[],
        walls=[],
        reference_point=ref_point,
    )

    new_geom = create_catheter_geometry(
        geom, image_center=(0.0, 0.0), radius=0.5, n_points=10
    )

    assert len(new_geom.catheters) > 0
    for catheter in new_geom.catheters:
        assert isinstance(catheter, PyContour)
        assert len(catheter.points) == 10
        for pt in catheter.points:
            assert isinstance(pt, PyContourPoint)

    assert len(new_geom.contours) == len(geom.contours)


def test_reorder_by_shape_similarity():
    # Build three contours:
    # 0 = very elliptic (major=10, minor=2)
    # 1 = round         (major=5,  minor=5)
    # 2 = slight elliptic (major=6, minor=4)
    very = _create_elliptic_contour(
        0, center=(0, 0, 0), major_radius=10.0, minor_radius=2.0
    )
    rnd = _create_round_contour(1, center=(0, 0, 1), radius=5.0)
    slight = _create_elliptic_contour(
        2, center=(0, 0, 2), major_radius=6.0, minor_radius=4.0
    )

    # Give each a same‑length catheter so reorder() won’t panic
    cat_very = _create_round_contour(0, center=(0, 0, 0), radius=0.1)
    cat_rnd = _create_round_contour(1, center=(0, 0, 1), radius=0.1)
    cat_slight = _create_round_contour(2, center=(0, 0, 2), radius=0.1)

    ref = PyContourPoint(0, 0, 0.0, 0.0, 0.0, False)
    geom = PyGeometry(
        contours=[very, rnd, slight],
        catheters=[cat_very, cat_rnd, cat_slight],
        walls=[],
        reference_point=ref,
    )

    # Sanity: starting major‐radius sequence [10, 5, 6]
    assert _radius_sequence(geom) == pytest.approx([10.0, 5.0, 6.0])

    # Reorder with delta=0 so only Hausdorff similarity matters
    reordered = geom.reorder(delta=0.0, max_rounds=5)

    # After reorder, we should go 0→2→1 (very→slight→round) i.e. [10, 6, 5]
    result = _radius_sequence(reordered)
    assert result == pytest.approx([10.0, 6.0, 5.0])


def test_rotation_geometry(sample_geometry):
    # Grab the original geometry, which has one square contour centered at (2,2,3)
    geom = sample_geometry
    orig_centroid = geom.contours[0].centroid
    # point_index 0 before rotation is at (0,4,3)
    p0_before = next(p for p in geom.contours[0].points if p.point_index == 0)
    assert (p0_before.x, p0_before.y, p0_before.z) == pytest.approx((0.0, 4.0, 3.0))

    rotated = geom.rotate(90)

    # Centroid should remain unchanged
    new_centroid = rotated.contours[0].centroid
    assert new_centroid == pytest.approx(orig_centroid)

    # After rotation, that same point should move (0,4)→(0,0) in XY, z stays 3
    p0_after = next(p for p in rotated.contours[0].points if p.point_index == 0)
    assert (p0_after.x, p0_after.y, p0_after.z) == pytest.approx((0.0, 0.0, 3.0))


def test_translation_geometry(sample_geometry):
    # Grab the original geometry
    geom = sample_geometry
    orig_centroid = geom.contours[0].centroid
    p0_before = next(p for p in geom.contours[0].points if p.point_index == 0)

    translated = geom.translate(1.0, 2.0, 3.0)
    new_centroid = translated.contours[0].centroid
    expected_centroid = (
        orig_centroid[0] + 1.0,
        orig_centroid[1] + 2.0,
        orig_centroid[2] + 3.0,
    )
    assert new_centroid == pytest.approx(expected_centroid)

    p0_after = next(p for p in translated.contours[0].points if p.point_index == 0)
    expected_point = (p0_before.x + 1.0, p0_before.y + 2.0, p0_before.z + 3.0)
    assert (p0_after.x, p0_after.y, p0_after.z) == pytest.approx(expected_point)


# -----------------PyCenterlinePoint------------------------


# -------------------PyCenterline---------------------------
