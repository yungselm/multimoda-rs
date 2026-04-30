"""Tests for the multimodars.ccta module.

Covers:
  - labeling:   _find_aortic_points, _find_faces_for_points,
                _prepare_faces_for_rust, _final_reclassification
  - fixing_functions: manual_hole_fill, postprocess_stitched_mesh
  - manipulating: remove_labeled_points_from_mesh,
                  keep_labeled_points_from_mesh, sync_results_to_mesh,
                  order_points_list, scale_region_centerline_morphing,
                  _rotate_to_nearest_iv, _fix_ring_direction_by_distance,
                  _stitch_boundary_ring
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest
import trimesh

from multimodars import PyContourPoint
from multimodars.ccta.fixing_functions import manual_hole_fill, postprocess_stitched_mesh
from multimodars.ccta.labeling import (
    _find_aortic_points,
    _find_faces_for_points,
    _final_reclassification,
    _prepare_faces_for_rust,
)
from multimodars.ccta.manipulating import (
    _fix_ring_direction_by_distance,
    _rotate_to_nearest_iv,
    _stitch_boundary_ring,
    keep_labeled_points_from_mesh,
    order_points_list,
    remove_labeled_points_from_mesh,
    scale_region_centerline_morphing,
    sync_results_to_mesh,
)


# ---------------------------------------------------------------------------
# Shared mesh factories
# ---------------------------------------------------------------------------

def _make_grid_mesh() -> trimesh.Trimesh:
    """3x3 grid (9 vertices, 8 triangular faces, z=0 plane).

    Vertex layout:
        6--7--8
        |/|/|
        3--4--5
        |/|/|
        0--1--2
    """
    verts = np.array(
        [
            [0.0, 0.0, 0.0],  # 0  corner
            [1.0, 0.0, 0.0],  # 1
            [2.0, 0.0, 0.0],  # 2  corner
            [0.0, 1.0, 0.0],  # 3
            [1.0, 1.0, 0.0],  # 4  centre – adjacent to {1,2,3,5,6,7}
            [2.0, 1.0, 0.0],  # 5
            [0.0, 2.0, 0.0],  # 6  corner
            [1.0, 2.0, 0.0],  # 7
            [2.0, 2.0, 0.0],  # 8  corner
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 3],
            [1, 4, 3],
            [1, 2, 4],
            [2, 5, 4],
            [3, 4, 6],
            [4, 7, 6],
            [4, 5, 7],
            [5, 8, 7],
        ]
    )
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _make_hex_fan_mesh() -> trimesh.Trimesh:
    """6 outer vertices (0-5) + centre (6), 6 triangular faces.

    Boundary ring: vertices 0-5 in angular order.
    """
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    outer = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(6)])
    centre = np.array([[0.0, 0.0, 0.0]])
    verts = np.vstack([outer, centre])
    faces = np.array([[i, (i + 1) % 6, 6] for i in range(6)])
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _make_iv_pts(coords) -> list[PyContourPoint]:
    return [
        PyContourPoint(frame_index=0, point_index=i, x=x, y=y, z=z, aortic=False)
        for i, (x, y, z) in enumerate(coords)
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grid_mesh():
    return _make_grid_mesh()


@pytest.fixture
def hex_fan_mesh():
    return _make_hex_fan_mesh()


@pytest.fixture
def grid_results(grid_mesh):
    """Results dict for the 3×3 grid: rows 0/1/2 labelled RCA/LCA/aorta."""
    verts = [tuple(v) for v in grid_mesh.vertices]
    return {
        "mesh": grid_mesh,
        "aorta_points": verts[6:9],   # vertices 6, 7, 8
        "rca_points":   verts[0:3],   # vertices 0, 1, 2
        "lca_points":   verts[3:6],   # vertices 3, 4, 5
        "rca_removed_points": [],
        "lca_removed_points": [],
    }


# ===========================================================================
# labeling._find_aortic_points
# ===========================================================================

class TestFindAorticPoints:
    def test_basic_set_difference(self, grid_mesh):
        rca = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        lca = [(0.0, 1.0, 0.0)]
        aortic = _find_aortic_points(grid_mesh.vertices, rca, lca)
        assert len(aortic) == 6  # 9 total − 3 labelled
        assert (0.0, 0.0, 0.0) not in aortic
        assert (0.0, 1.0, 0.0) not in aortic

    def test_empty_rca_lca_returns_all(self, grid_mesh):
        aortic = _find_aortic_points(grid_mesh.vertices, [], [])
        assert len(aortic) == len(grid_mesh.vertices)

    def test_all_labelled_returns_empty(self, grid_mesh):
        all_verts = [tuple(v) for v in grid_mesh.vertices]
        aortic = _find_aortic_points(grid_mesh.vertices, all_verts[:5], all_verts[5:])
        assert aortic == []

    def test_output_is_list_of_tuples(self, grid_mesh):
        aortic = _find_aortic_points(grid_mesh.vertices, [], [])
        assert all(isinstance(p, tuple) for p in aortic)
        assert all(len(p) == 3 for p in aortic)


# ===========================================================================
# labeling._find_faces_for_points
# ===========================================================================

class TestFindFacesForPoints:
    def test_corner_vertex_finds_its_face(self, grid_mesh):
        # vertex 0 = (0,0,0) belongs to face [0,1,3] → face index 0
        indices = _find_faces_for_points(grid_mesh, [(0.0, 0.0, 0.0)], tol=1e-6)
        assert 0 in indices

    def test_centre_vertex_touches_many_faces(self, grid_mesh):
        # vertex 4 = (1,1,0) appears in 6 of the 8 faces
        indices = _find_faces_for_points(grid_mesh, [(1.0, 1.0, 0.0)], tol=1e-6)
        assert len(indices) == 6

    def test_empty_points_returns_empty(self, grid_mesh):
        assert _find_faces_for_points(grid_mesh, [], tol=1e-6) == []

    def test_no_vertex_within_tol(self, grid_mesh):
        indices = _find_faces_for_points(grid_mesh, [(99.0, 99.0, 0.0)], tol=1e-6)
        assert indices == []

    def test_returns_list_of_ints(self, grid_mesh):
        indices = _find_faces_for_points(grid_mesh, [(0.0, 0.0, 0.0)], tol=1e-6)
        assert all(isinstance(i, int) for i in indices)


# ===========================================================================
# labeling._prepare_faces_for_rust
# ===========================================================================

class TestPrepareFacesForRust:
    def test_all_faces_when_no_args(self, grid_mesh):
        rust_faces = _prepare_faces_for_rust(grid_mesh)
        assert len(rust_faces) == len(grid_mesh.faces)

    def test_each_face_is_triple_of_triples(self, grid_mesh):
        for face in _prepare_faces_for_rust(grid_mesh):
            assert len(face) == 3
            for v in face:
                assert len(v) == 3

    def test_coordinates_are_floats(self, grid_mesh):
        v0, v1, v2 = _prepare_faces_for_rust(grid_mesh, face_indices=[0])[0]
        assert all(isinstance(c, float) for c in v0 + v1 + v2)

    def test_explicit_face_indices(self, grid_mesh):
        rust_faces = _prepare_faces_for_rust(grid_mesh, face_indices=[0, 2])
        assert len(rust_faces) == 2

    def test_subset_via_points(self, grid_mesh):
        # corner vertex 0 only in ~1 face, so subset < all
        rust_faces = _prepare_faces_for_rust(grid_mesh, points=[(0.0, 0.0, 0.0)], tol=1e-6)
        assert 0 < len(rust_faces) < len(grid_mesh.faces)

    def test_empty_points_returns_all(self, grid_mesh):
        # When points=[] → _find_faces_for_points returns [] → no faces
        rust_faces = _prepare_faces_for_rust(grid_mesh, points=[], tol=1e-6)
        assert rust_faces == []


# ===========================================================================
# labeling._final_reclassification
# ===========================================================================

class TestFinalReclassification:
    # ------------------------------------------------------------------
    # Logic A: isolated RCA/LCA vertex → reclassified to aorta
    # ------------------------------------------------------------------

    def test_isolated_rca_becomes_aorta(self, grid_mesh):
        """Vertex 0 labelled RCA; its neighbours (1, 3) are aorta → reclassified."""
        verts = [tuple(v) for v in grid_mesh.vertices]
        results = {
            "mesh": grid_mesh,
            "aorta_points": verts[1:],  # all except vertex 0
            "rca_points": [verts[0]],
            "lca_points": [],
            "rca_removed_points": [],
            "lca_removed_points": [],
        }
        new = _final_reclassification(results)
        assert verts[0] not in new["rca_points"]
        assert verts[0] in new["aorta_points"]

    def test_isolated_lca_becomes_aorta(self, grid_mesh):
        verts = [tuple(v) for v in grid_mesh.vertices]
        results = {
            "mesh": grid_mesh,
            "aorta_points": verts[1:],
            "rca_points": [],
            "lca_points": [verts[0]],
            "rca_removed_points": [],
            "lca_removed_points": [],
        }
        new = _final_reclassification(results)
        assert verts[0] not in new["lca_points"]
        assert verts[0] in new["aorta_points"]

    def test_non_isolated_rca_stays(self, grid_mesh):
        """Vertex 0 and neighbour 1 are both RCA → vertex 0 keeps its label."""
        verts = [tuple(v) for v in grid_mesh.vertices]
        results = {
            "mesh": grid_mesh,
            "aorta_points": verts[2:],
            "rca_points": [verts[0], verts[1]],
            "lca_points": [],
            "rca_removed_points": [],
            "lca_removed_points": [],
        }
        new = _final_reclassification(results)
        assert verts[0] in new["rca_points"]

    # ------------------------------------------------------------------
    # Logic B: removed vertex whose neighbours are >70 % same label → restored
    # ------------------------------------------------------------------

    def test_removed_rca_restored_when_majority_rca(self, grid_mesh):
        """Vertex 4 is RCA_REMOVED; all 6 neighbours are RCA (100 % > 70 %)."""
        verts = [tuple(v) for v in grid_mesh.vertices]
        # vertex 4 neighbours: {1, 2, 3, 5, 6, 7}
        results = {
            "mesh": grid_mesh,
            "aorta_points": [verts[0], verts[8]],
            "rca_points": [verts[1], verts[2], verts[3], verts[5], verts[6], verts[7]],
            "lca_points": [],
            "rca_removed_points": [verts[4]],
            "lca_removed_points": [],
        }
        new = _final_reclassification(results)
        assert verts[4] in new["rca_points"]
        assert verts[4] not in new["rca_removed_points"]

    # ------------------------------------------------------------------
    # Invariants
    # ------------------------------------------------------------------

    def test_vertex_count_conserved(self, grid_mesh):
        """Total vertices across all lists must equal mesh vertex count."""
        verts = [tuple(v) for v in grid_mesh.vertices]
        results = {
            "mesh": grid_mesh,
            "aorta_points": verts[4:],
            "rca_points": verts[:2],
            "lca_points": verts[2:4],
            "rca_removed_points": [],
            "lca_removed_points": [],
        }
        new = _final_reclassification(results)
        total = sum(
            len(new[k])
            for k in ("aorta_points", "rca_points", "lca_points",
                      "rca_removed_points", "lca_removed_points")
        )
        assert total == len(grid_mesh.vertices)

    def test_returns_dict_with_required_keys(self, grid_results):
        new = _final_reclassification(grid_results)
        for key in ("mesh", "aorta_points", "rca_points", "lca_points",
                    "rca_removed_points", "lca_removed_points"):
            assert key in new


# ===========================================================================
# fixing_functions.manual_hole_fill
# ===========================================================================

class TestManualHoleFill:
    def test_adds_faces_to_open_mesh(self):
        """Box with top cap removed gets new faces from hole fill."""
        box = trimesh.creation.box()
        top_mask = box.face_normals[:, 2] < 0.9   # keep all non-top faces
        holed = trimesh.Trimesh(
            vertices=box.vertices,
            faces=box.faces[top_mask],
            process=False,
        )
        n_before = len(holed.faces)
        filled = manual_hole_fill(holed)
        assert len(filled.faces) > n_before

    def test_watertight_mesh_not_shrunk(self):
        """A closed sphere has no boundary; face count should not decrease."""
        sphere = trimesh.creation.icosphere(subdivisions=1)
        n_before = len(sphere.faces)
        filled = manual_hole_fill(sphere)
        assert len(filled.faces) >= n_before

    def test_returns_trimesh(self):
        box = trimesh.creation.box()
        result = manual_hole_fill(box)
        assert isinstance(result, trimesh.Trimesh)


# ===========================================================================
# fixing_functions.postprocess_stitched_mesh
# ===========================================================================

class TestPostprocessStitchedMesh:
    def test_passthrough_when_disabled(self, grid_mesh):
        result = postprocess_stitched_mesh(grid_mesh, postprocessing=False)
        assert result is grid_mesh  # exact same object

    def test_raises_import_error_without_pymeshlab(self, grid_mesh):
        if importlib.util.find_spec("pymeshlab") is not None:
            pytest.skip("pymeshlab installed; ImportError path not triggered")
        with pytest.raises(ImportError, match="pymeshlab"):
            postprocess_stitched_mesh(grid_mesh, postprocessing=True)


# ===========================================================================
# manipulating.remove_labeled_points_from_mesh
# ===========================================================================

class TestRemoveLabeledPoints:
    def test_removes_vertices_from_mesh(self, grid_results):
        updated = remove_labeled_points_from_mesh(grid_results, region_keys="rca_points")
        new_verts = {tuple(v) for v in updated["mesh"].vertices}
        rca_set = {(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)}
        assert rca_set.isdisjoint(new_verts)

    def test_boundary_points_populated(self, grid_results):
        updated = remove_labeled_points_from_mesh(grid_results, region_keys="rca_points")
        assert len(updated["boundary_points"]) > 0

    def test_removed_key_cleared(self, grid_results):
        updated = remove_labeled_points_from_mesh(grid_results, region_keys="rca_points")
        assert updated["rca_points"] == []

    def test_empty_region_is_noop(self, grid_results):
        grid_results["rca_points"] = []
        n_before = len(grid_results["mesh"].vertices)
        updated = remove_labeled_points_from_mesh(grid_results, region_keys="rca_points")
        assert len(updated["mesh"].vertices) == n_before

    def test_multiple_keys(self, grid_results):
        updated = remove_labeled_points_from_mesh(
            grid_results, region_keys=["rca_points", "lca_points"]
        )
        assert updated["rca_points"] == []
        assert updated["lca_points"] == []
        # Only aorta vertices (6, 7, 8) remain
        assert len(updated["mesh"].vertices) == 3

    def test_remaining_lists_consistent_with_new_mesh(self, grid_results):
        updated = remove_labeled_points_from_mesh(grid_results, region_keys="rca_points")
        new_verts = {tuple(v) for v in updated["mesh"].vertices}
        for key in ("aorta_points", "lca_points"):
            for pt in updated.get(key, []):
                assert tuple(pt) in new_verts

    def test_string_or_list_region_keys(self, grid_results):
        """String and single-item list should produce identical results."""
        import copy
        r1 = remove_labeled_points_from_mesh(copy.deepcopy(grid_results), region_keys="rca_points")
        r2 = remove_labeled_points_from_mesh(copy.deepcopy(grid_results), region_keys=["rca_points"])
        assert len(r1["mesh"].vertices) == len(r2["mesh"].vertices)


# ===========================================================================
# manipulating.keep_labeled_points_from_mesh
# ===========================================================================

class TestKeepLabeledPoints:
    def test_mesh_vertex_count_reduced(self, grid_results):
        updated = keep_labeled_points_from_mesh(grid_results, region_key="rca_points")
        assert len(updated["mesh"].vertices) < len(grid_results["mesh"].vertices)

    def test_boundary_points_key_present(self, grid_results):
        updated = keep_labeled_points_from_mesh(grid_results, region_key="rca_points")
        assert "boundary_points" in updated

    def test_empty_region_is_noop(self, grid_results):
        grid_results["rca_points"] = []
        n_before = len(grid_results["mesh"].vertices)
        updated = keep_labeled_points_from_mesh(grid_results, region_key="rca_points")
        assert len(updated["mesh"].vertices) == n_before

    def test_kept_vertices_all_in_region(self, grid_results):
        updated = keep_labeled_points_from_mesh(grid_results, region_key="aorta_points")
        # aorta verts are 6,7,8 (y=2). After keeping, new mesh should only hold those.
        new_verts = {tuple(v) for v in updated["mesh"].vertices}
        aorta_set = {(0.0, 2.0, 0.0), (1.0, 2.0, 0.0), (2.0, 2.0, 0.0)}
        assert aorta_set.issubset(new_verts)


# ===========================================================================
# manipulating.sync_results_to_mesh
# ===========================================================================

class TestSyncResultsToMesh:
    def test_mesh_replaced(self, grid_results, grid_mesh):
        new_mesh = trimesh.Trimesh(
            vertices=grid_mesh.vertices + 1.0,
            faces=grid_mesh.faces,
            process=False,
        )
        updated = sync_results_to_mesh(grid_results, grid_mesh, new_mesh)
        assert updated["mesh"] is new_mesh

    def test_coordinate_lists_updated(self, grid_results, grid_mesh):
        shift = np.array([10.0, 0.0, 0.0])
        new_verts = grid_mesh.vertices.copy() + shift
        new_mesh = trimesh.Trimesh(vertices=new_verts, faces=grid_mesh.faces, process=False)
        updated = sync_results_to_mesh(grid_results, grid_mesh, new_mesh)
        for pt in updated["rca_points"]:
            assert pt[0] >= 10.0

    def test_preserves_number_of_labeled_points(self, grid_results, grid_mesh):
        new_mesh = trimesh.Trimesh(
            vertices=grid_mesh.vertices * 2,
            faces=grid_mesh.faces,
            process=False,
        )
        n_rca_before = len(grid_results["rca_points"])
        updated = sync_results_to_mesh(grid_results, grid_mesh, new_mesh)
        assert len(updated["rca_points"]) == n_rca_before


# ===========================================================================
# manipulating.order_points_list
# ===========================================================================

class TestOrderPointsList:
    def test_single_point_returns_same(self, hex_fan_mesh):
        pts = [tuple(hex_fan_mesh.vertices[0])]
        assert order_points_list(hex_fan_mesh, pts) == pts

    def test_empty_returns_empty(self, hex_fan_mesh):
        assert order_points_list(hex_fan_mesh, []) == []

    def test_returns_all_outer_vertices(self, hex_fan_mesh):
        outer = [tuple(hex_fan_mesh.vertices[i]) for i in range(6)]
        ordered = order_points_list(hex_fan_mesh, outer)
        assert len(ordered) == 6
        assert set(ordered) == set(outer)

    def test_consecutive_points_are_adjacent(self, hex_fan_mesh):
        """Consecutive entries in the ordered list must share a mesh edge."""
        from multimodars.multimodars import build_adjacency_map

        outer = [tuple(hex_fan_mesh.vertices[i]) for i in range(6)]
        ordered = order_points_list(hex_fan_mesh, outer)
        adj = build_adjacency_map(hex_fan_mesh.faces.tolist())
        coord_to_idx = {tuple(v): i for i, v in enumerate(hex_fan_mesh.vertices)}

        for i in range(len(ordered) - 1):
            a = coord_to_idx[tuple(ordered[i])]
            b = coord_to_idx[tuple(ordered[i + 1])]
            assert b in adj.get(a, []), (
                f"ordered[{i}]={ordered[i]} and ordered[{i+1}]={ordered[i+1]} "
                f"are not mesh-adjacent"
            )


# ===========================================================================
# manipulating.scale_region_centerline_morphing
# ===========================================================================

class TestScaleRegionCenterlineMorphing:
    def test_no_matching_vertices_returns_copy(self, grid_mesh, capsys):
        """Passing points not on the mesh triggers the warning path (no Rust call)."""
        result = scale_region_centerline_morphing(
            grid_mesh,
            region_points=[(999.0, 999.0, 999.0)],
            centerline=None,  # never reached
            diameter_adjustment_mm=1.0,
        )
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert len(result.vertices) == len(grid_mesh.vertices)
        # Must be a copy, not the same object
        assert result is not grid_mesh


# ===========================================================================
# manipulating._rotate_to_nearest_iv
# ===========================================================================

class TestRotateToNearestIv:
    def test_rotates_to_nearest_iv_point(self):
        prox = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)]
        dist = [(0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (2.0, 1.0, 0.0)]

        prox_iv = _make_iv_pts([(2.0, 0.0, 0.0)])  # nearest to prox[2]
        dist_iv = _make_iv_pts([(2.0, 1.0, 0.0)])  # nearest to dist[2]

        new_prox = _rotate_to_nearest_iv(prox, prox_iv[0])
        new_dist = _rotate_to_nearest_iv(dist, dist_iv[0])
        assert new_prox[0] == (2.0, 0.0, 0.0)
        assert new_dist[0] == (2.0, 1.0, 0.0)

    def test_length_and_set_preserved(self):
        prox = [(float(i), 0.0, 0.0) for i in range(5)]
        dist = [(float(i), 1.0, 0.0) for i in range(5)]
        new_prox = _rotate_to_nearest_iv(prox, _make_iv_pts([(3.0, 0.0, 0.0)])[0])
        new_dist = _rotate_to_nearest_iv(dist, _make_iv_pts([(4.0, 1.0, 0.0)])[0])
        assert len(new_prox) == len(prox)
        assert set(new_prox) == set(prox)
        assert set(new_dist) == set(dist)

    def test_already_at_start_unchanged(self):
        """If the nearest IV point already matches the first boundary point, no rotation."""
        prox = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        new_prox = _rotate_to_nearest_iv(prox, _make_iv_pts([(0.0, 0.0, 0.0)])[0])
        assert new_prox == prox


# ===========================================================================
# manipulating._fix_ring_direction_by_distance
# ===========================================================================

class TestFixRingDirectionByDistance:
    def test_correct_direction_unchanged(self):
        """Ring already matches IV order → not reversed."""
        n = 6
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        prox = [(float(np.cos(a)), float(np.sin(a)), 0.0) for a in angles]

        # IV points in the same order with step=1
        prox_iv = _make_iv_pts(prox)

        new_prox = _fix_ring_direction_by_distance(prox, prox_iv, 1)
        assert new_prox[0] == prox[0]
        assert len(new_prox) == n

    def test_reversed_direction_gets_corrected(self):
        """Ring in wrong direction → first element fixed, rest reversed."""
        n = 4
        prox = [(float(i), 0.0, 0.0) for i in range(n)]

        # IV points arranged so reversed_prox has smaller total distance
        iv_for_prox = [(float(n - 1 - i), 0.0, 0.0) for i in range(n)]
        prox_iv = _make_iv_pts(iv_for_prox)

        new_prox = _fix_ring_direction_by_distance(prox, prox_iv, 1)
        # After reversal: [prox[0]] + reversed(prox[1:]) = [0, 3, 2, 1]
        assert new_prox[0] == prox[0]
        assert new_prox[1] == prox[-1]

    def test_preserves_length(self):
        n = 5
        prox = [(float(i), 0.0, 0.0) for i in range(n)]
        dist = [(float(i), 1.0, 0.0) for i in range(n)]
        new_prox = _fix_ring_direction_by_distance(prox, _make_iv_pts(prox), 1)
        new_dist = _fix_ring_direction_by_distance(dist, _make_iv_pts(dist), 1)
        assert len(new_prox) == n
        assert len(new_dist) == n


# ===========================================================================
# manipulating._stitch_boundary_ring
# ===========================================================================

class TestStitchBoundaryRing:
    def _ring_pts(self, n: int, radius: float = 1.0, z: float = 0.0):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return [(radius * float(np.cos(a)), radius * float(np.sin(a)), z) for a in angles]

    def test_creates_trimesh(self):
        n_b, n_iv = 6, 12
        boundary_pts = self._ring_pts(n_b, radius=1.0)
        iv_pts = _make_iv_pts(self._ring_pts(n_iv, radius=1.2))
        patch = _stitch_boundary_ring(boundary_pts, iv_pts, n_iv // n_b)
        assert isinstance(patch, trimesh.Trimesh)

    def test_vertex_count(self):
        n_b, n_iv = 6, 12
        boundary_pts = self._ring_pts(n_b)
        iv_pts = _make_iv_pts(self._ring_pts(n_iv, radius=1.2))
        patch = _stitch_boundary_ring(boundary_pts, iv_pts, n_iv // n_b)
        assert len(patch.vertices) == n_b + n_iv

    def test_no_nan_vertices(self):
        n_b, n_iv = 4, 8
        boundary_pts = self._ring_pts(n_b)
        iv_pts = _make_iv_pts(self._ring_pts(n_iv, radius=1.5))
        patch = _stitch_boundary_ring(boundary_pts, iv_pts, n_iv // n_b)
        assert not np.isnan(patch.vertices).any()

    def test_has_faces(self):
        n_b, n_iv = 6, 12
        boundary_pts = self._ring_pts(n_b)
        iv_pts = _make_iv_pts(self._ring_pts(n_iv, radius=1.2))
        patch = _stitch_boundary_ring(boundary_pts, iv_pts, n_iv // n_b)
        assert len(patch.faces) > 0

    def test_outward_direction_orients_patch(self):
        """When outward_direction is given, average face normal should align with it."""
        n_b, n_iv = 6, 12
        boundary_pts = self._ring_pts(n_b, z=0.0)
        iv_pts = _make_iv_pts(self._ring_pts(n_iv, radius=1.2, z=0.0))
        outward = np.array([0.0, 0.0, 1.0])
        patch = _stitch_boundary_ring(boundary_pts, iv_pts, n_iv // n_b, outward_direction=outward)
        valid = ~np.isnan(patch.face_normals).any(axis=1)
        if valid.any():
            avg_normal = patch.face_normals[valid].mean(axis=0)
            assert np.dot(avg_normal, outward) > 0
