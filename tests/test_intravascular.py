"""Tests for the multimodars python implementation.

Covers:
  - alignment:   align_three_point, align_manual, align_combined.
"""

from __future__ import annotations

import numpy as np
import pytest

from multimodars import PyGeometry, PyGeometryPair
from multimodars._converters import numpy_to_centerline, numpy_to_geometry
from multimodars._processing import align_manual, align_three_point

# Reference points from the align_three_point docstring example
AORTIC_REF_PT = (12.2605, -201.3643, 1751.0554)
UPPER_REF_PT = (11.7567, -202.1920, 1754.7975)
LOWER_REF_PT = (15.6605, -202.1920, 1749.9655)


def _geom_to_points_array(geom: PyGeometry) -> np.ndarray:
    """Flatten all lumen contour points across all frames into a (N, 3) array."""
    rows = []
    for frame in geom.frames:
        for pt in frame.lumen.points:
            rows.append((pt.x, pt.y, pt.z))
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def geom_a() -> PyGeometry:
    try:
        raw = np.genfromtxt(
            "data/fixtures/idealized_geometry/diastolic_contours.csv", delimiter=","
        )
        ref = np.genfromtxt(
            "data/fixtures/idealized_geometry/diastolic_reference_points.csv",
            delimiter=",",
        )
    except OSError:
        pytest.skip("Idealized geometry test data not found")
    return numpy_to_geometry(
        lumen_arr=raw,
        catheter_arr=np.zeros((0, 4)),
        wall_arr=np.zeros((0, 4)),
        reference_arr=ref,
    )


@pytest.fixture(scope="module")
def geom_b() -> PyGeometry:
    try:
        raw = np.genfromtxt(
            "data/fixtures/idealized_geometry/systolic_contours.csv", delimiter=","
        )
        ref = np.genfromtxt(
            "data/fixtures/idealized_geometry/systolic_reference_points.csv",
            delimiter=",",
        )
    except OSError:
        pytest.skip("Idealized geometry test data not found")
    return numpy_to_geometry(
        lumen_arr=raw,
        catheter_arr=np.zeros((0, 4)),
        wall_arr=np.zeros((0, 4)),
        reference_arr=ref,
    )


@pytest.fixture(scope="module")
def geometry_pair(geom_a: PyGeometry, geom_b: PyGeometry) -> PyGeometryPair:
    return PyGeometryPair(geom_a=geom_a, geom_b=geom_b, label="test")


@pytest.fixture(scope="module")
def centerline():
    try:
        arr = np.genfromtxt("examples/data/centerline_raw.csv", delimiter=",")
    except OSError:
        pytest.skip("Centerline test data not found")
    return numpy_to_centerline(arr)


# ---------------------------------------------------------------------------
# align_three_point
# ---------------------------------------------------------------------------


class TestAlignThreePoint:
    def test_pair_returns_geometry_pair(self, centerline, geometry_pair):
        result, _ = align_three_point(
            centerline,
            geometry_pair,
            AORTIC_REF_PT,
            UPPER_REF_PT,
            LOWER_REF_PT,
            write=False,
        )
        assert isinstance(result, PyGeometryPair)

    def test_single_returns_geometry(self, centerline, geom_a):
        result, _ = align_three_point(
            centerline,
            geom_a,
            AORTIC_REF_PT,
            UPPER_REF_PT,
            LOWER_REF_PT,
            write=False,
        )
        assert isinstance(result, PyGeometry)

    def test_pair_geom_a_matches_single(self, centerline, geom_a, geometry_pair):
        """Aligning geom_a alone must produce the same points as geom_a inside a pair."""
        result_pair, _ = align_three_point(
            centerline,
            geometry_pair,
            AORTIC_REF_PT,
            UPPER_REF_PT,
            LOWER_REF_PT,
            write=False,
        )
        result_geom, _ = align_three_point(
            centerline,
            geom_a,
            AORTIC_REF_PT,
            UPPER_REF_PT,
            LOWER_REF_PT,
            write=False,
        )
        pts_pair = _geom_to_points_array(result_pair.geom_a)
        pts_geom = _geom_to_points_array(result_geom)
        np.testing.assert_allclose(pts_pair, pts_geom, atol=1e-10)

    def test_resampled_centerlines_match(self, centerline, geom_a, geometry_pair):
        _, cl_pair = align_three_point(
            centerline,
            geometry_pair,
            AORTIC_REF_PT,
            UPPER_REF_PT,
            LOWER_REF_PT,
            write=False,
        )
        _, cl_geom = align_three_point(
            centerline,
            geom_a,
            AORTIC_REF_PT,
            UPPER_REF_PT,
            LOWER_REF_PT,
            write=False,
        )
        pts_pair = np.array(
            [
                (p.contour_point.x, p.contour_point.y, p.contour_point.z)
                for p in cl_pair.points
            ]
        )
        pts_geom = np.array(
            [
                (p.contour_point.x, p.contour_point.y, p.contour_point.z)
                for p in cl_geom.points
            ]
        )
        np.testing.assert_allclose(pts_pair, pts_geom, atol=1e-10)

    def test_frame_count_preserved(self, centerline, geom_a, geometry_pair):
        n_frames = len(geom_a.frames)
        result_pair, _ = align_three_point(
            centerline,
            geometry_pair,
            AORTIC_REF_PT,
            UPPER_REF_PT,
            LOWER_REF_PT,
            write=False,
        )
        result_geom, _ = align_three_point(
            centerline,
            geom_a,
            AORTIC_REF_PT,
            UPPER_REF_PT,
            LOWER_REF_PT,
            write=False,
        )
        assert len(result_pair.geom_a.frames) == n_frames
        assert len(result_geom.frames) == n_frames


# ---------------------------------------------------------------------------
# align_manual
# ---------------------------------------------------------------------------


class TestAlignManual:
    def test_pair_returns_geometry_pair(self, centerline, geometry_pair):
        result, _ = align_manual(
            centerline,
            geometry_pair,
            rotation_angle=286.0,
            ref_point=AORTIC_REF_PT,
            write=False,
        )
        assert isinstance(result, PyGeometryPair)

    def test_single_returns_geometry(self, centerline, geom_a):
        result, _ = align_manual(
            centerline,
            geom_a,
            rotation_angle=286.0,
            ref_point=AORTIC_REF_PT,
            write=False,
        )
        assert isinstance(result, PyGeometry)

    def test_pair_geom_a_matches_single(self, centerline, geom_a, geometry_pair):
        """Aligning geom_a alone must produce the same points as geom_a inside a pair."""
        result_pair, _ = align_manual(
            centerline,
            geometry_pair,
            rotation_angle=286.0,
            ref_point=AORTIC_REF_PT,
            write=False,
        )
        result_geom, _ = align_manual(
            centerline,
            geom_a,
            rotation_angle=286.0,
            ref_point=AORTIC_REF_PT,
            write=False,
        )
        pts_pair = _geom_to_points_array(result_pair.geom_a)
        pts_geom = _geom_to_points_array(result_geom)
        np.testing.assert_allclose(pts_pair, pts_geom, atol=1e-10)

    def test_frame_count_preserved(self, centerline, geom_a, geometry_pair):
        n_frames = len(geom_a.frames)
        result_pair, _ = align_manual(
            centerline,
            geometry_pair,
            rotation_angle=286.0,
            ref_point=AORTIC_REF_PT,
            write=False,
        )
        result_geom, _ = align_manual(
            centerline,
            geom_a,
            rotation_angle=286.0,
            ref_point=AORTIC_REF_PT,
            write=False,
        )
        assert len(result_pair.geom_a.frames) == n_frames
        assert len(result_geom.frames) == n_frames
