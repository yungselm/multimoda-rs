"""Tests for the multimodars.ccta module.

Covers:
  - multimodars.multimodars: find_faces_near_points, find_aortic_points,
                final_reclassification (Rust bindings backing
                labeling.label_geometry's occlusion-removal and
                adjacency-based label-smoothing steps)
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
from multimodars.ccta.fixing_functions import (
    manual_hole_fill,
    postprocess_stitched_mesh,
)
from multimodars.multimodars import (
    find_faces_near_points,
    find_aortic_points,
    final_reclassification,
)
from multimodars.ccta.manipulating import (
    _clamp_to_plane,
    _enforce_layer_gap_from_plane,
    _fast_fix_normals,
    _fix_ring_direction_by_distance,
    _prepare_prox_dist_boundary_pts,
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
        "aorta_points": verts[6:9],  # vertices 6, 7, 8
        "rca_points": verts[0:3],  # vertices 0, 1, 2
        "lca_points": verts[3:6],  # vertices 3, 4, 5
        "rca_removed_points": [],
        "lca_removed_points": [],
    }


# ===========================================================================
# multimodars.multimodars.find_aortic_points
# (Rust binding that replaced labeling._find_aortic_points)
# ===========================================================================


class TestFindAorticPoints:
    def test_basic_set_difference(self, grid_mesh):
        verts = [tuple(v) for v in grid_mesh.vertices]
        rca = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        lca = [(0.0, 1.0, 0.0)]
        aortic = find_aortic_points(verts, rca, lca)
        assert len(aortic) == 6  # 9 total − 3 labelled
        assert (0.0, 0.0, 0.0) not in aortic
        assert (0.0, 1.0, 0.0) not in aortic

    def test_empty_rca_lca_returns_all(self, grid_mesh):
        verts = [tuple(v) for v in grid_mesh.vertices]
        aortic = find_aortic_points(verts, [], [])
        assert len(aortic) == len(verts)

    def test_all_labelled_returns_empty(self, grid_mesh):
        verts = [tuple(v) for v in grid_mesh.vertices]
        aortic = find_aortic_points(verts, verts[:5], verts[5:])
        assert aortic == []

    def test_output_is_list_of_tuples(self, grid_mesh):
        verts = [tuple(v) for v in grid_mesh.vertices]
        aortic = find_aortic_points(verts, [], [])
        assert all(isinstance(p, tuple) for p in aortic)
        assert all(len(p) == 3 for p in aortic)


# ===========================================================================
# multimodars.multimodars.find_faces_near_points
# (Rust binding that replaced labeling._find_faces_for_points +
#  labeling._prepare_faces_for_rust)
# ===========================================================================


class TestFindFacesNearPoints:
    @staticmethod
    def _call(grid_mesh, points, tol=1e-6):
        vertices = [tuple(v) for v in grid_mesh.vertices]
        faces = grid_mesh.faces.tolist()
        return find_faces_near_points(vertices, faces, points, tol)

    def test_corner_vertex_finds_its_face(self, grid_mesh):
        # vertex 0 = (0,0,0) belongs to exactly face [0,1,3]
        result = self._call(grid_mesh, [(0.0, 0.0, 0.0)])
        verts = grid_mesh.vertices
        expected = (tuple(verts[0]), tuple(verts[1]), tuple(verts[3]))
        assert len(result) == 1
        assert result[0] == expected

    def test_centre_vertex_touches_many_faces(self, grid_mesh):
        # vertex 4 = (1,1,0) appears in 6 of the 8 faces
        result = self._call(grid_mesh, [(1.0, 1.0, 0.0)])
        assert len(result) == 6

    def test_empty_points_returns_empty(self, grid_mesh):
        assert self._call(grid_mesh, []) == []

    def test_no_vertex_within_tol(self, grid_mesh):
        result = self._call(grid_mesh, [(99.0, 99.0, 0.0)])
        assert result == []

    def test_each_face_is_triple_of_triples(self, grid_mesh):
        result = self._call(grid_mesh, [(1.0, 1.0, 0.0)])
        for face in result:
            assert len(face) == 3
            for v in face:
                assert len(v) == 3
                assert all(isinstance(c, float) for c in v)

    def test_subset_via_points(self, grid_mesh):
        # corner vertex 0 only in 1 face, so subset < all faces
        result = self._call(grid_mesh, [(0.0, 0.0, 0.0)])
        assert 0 < len(result) < len(grid_mesh.faces)


# ===========================================================================
# multimodars.multimodars.final_reclassification
# (Rust binding that replaced labeling._final_reclassification)
# ===========================================================================


class TestFinalReclassification:
    @staticmethod
    def _call(mesh, rca=(), lca=(), rca_removed=(), lca_removed=()) -> dict:
        verts = [tuple(v) for v in mesh.vertices]
        faces = mesh.faces.tolist()
        aorta_pts, rca_pts, lca_pts, rca_removed_pts, lca_removed_pts = (
            final_reclassification(
                verts, faces, list(rca), list(lca), list(rca_removed), list(lca_removed)
            )
        )
        return {
            "aorta_points": aorta_pts,
            "rca_points": rca_pts,
            "lca_points": lca_pts,
            "rca_removed_points": rca_removed_pts,
            "lca_removed_points": lca_removed_pts,
        }

    # ------------------------------------------------------------------
    # Logic A: isolated RCA/LCA vertex → reclassified to aorta
    # ------------------------------------------------------------------

    def test_isolated_rca_becomes_aorta(self, grid_mesh):
        """Vertex 0 labelled RCA; its neighbours (1, 3) are aorta → reclassified."""
        verts = [tuple(v) for v in grid_mesh.vertices]
        new = self._call(grid_mesh, rca=[verts[0]])
        assert verts[0] not in new["rca_points"]
        assert verts[0] in new["aorta_points"]

    def test_isolated_lca_becomes_aorta(self, grid_mesh):
        verts = [tuple(v) for v in grid_mesh.vertices]
        new = self._call(grid_mesh, lca=[verts[0]])
        assert verts[0] not in new["lca_points"]
        assert verts[0] in new["aorta_points"]

    def test_non_isolated_rca_stays(self, grid_mesh):
        """Vertex 0 and neighbour 1 are both RCA → vertex 0 keeps its label."""
        verts = [tuple(v) for v in grid_mesh.vertices]
        new = self._call(grid_mesh, rca=[verts[0], verts[1]])
        assert verts[0] in new["rca_points"]

    # ------------------------------------------------------------------
    # Logic B: removed vertex whose neighbours are >70 % same label → restored
    # ------------------------------------------------------------------

    def test_removed_rca_restored_when_majority_rca(self, grid_mesh):
        """Vertex 4 is RCA_REMOVED; all 6 neighbours are RCA (100 % > 70 %)."""
        verts = [tuple(v) for v in grid_mesh.vertices]
        # vertex 4 neighbours: {1, 2, 3, 5, 6, 7}
        new = self._call(
            grid_mesh,
            rca=[verts[1], verts[2], verts[3], verts[5], verts[6], verts[7]],
            rca_removed=[verts[4]],
        )
        assert verts[4] in new["rca_points"]
        assert verts[4] not in new["rca_removed_points"]

    # ------------------------------------------------------------------
    # Invariants
    # ------------------------------------------------------------------

    def test_vertex_count_conserved(self, grid_mesh):
        """Total vertices across all lists must equal mesh vertex count."""
        verts = [tuple(v) for v in grid_mesh.vertices]
        new = self._call(grid_mesh, rca=verts[:2], lca=verts[2:4])
        total = sum(
            len(new[k])
            for k in (
                "aorta_points",
                "rca_points",
                "lca_points",
                "rca_removed_points",
                "lca_removed_points",
            )
        )
        assert total == len(grid_mesh.vertices)

    def test_returns_dict_with_required_keys(self, grid_mesh, grid_results):
        new = self._call(
            grid_mesh,
            rca=grid_results["rca_points"],
            lca=grid_results["lca_points"],
        )
        for key in (
            "aorta_points",
            "rca_points",
            "lca_points",
            "rca_removed_points",
            "lca_removed_points",
        ):
            assert key in new


# ===========================================================================
# manipulating._fast_fix_normals
# (Rust-backed drop-in replacement for trimesh.Trimesh.fix_normals(), via
# multimodars.multimodars.fix_mesh_winding)
# ===========================================================================


class TestFastFixNormals:
    def test_matches_trimesh_on_inconsistent_quad(self):
        """Same quad, split with one face deliberately wound the wrong way."""
        verts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        )
        faces = np.array([[0, 1, 2], [2, 0, 3]])

        ours = trimesh.Trimesh(vertices=verts, faces=faces.copy(), process=False)
        _fast_fix_normals(ours)

        reference = trimesh.Trimesh(vertices=verts, faces=faces.copy(), process=False)
        reference.fix_normals()

        assert ours.faces.tolist() == reference.faces.tolist()

    def test_already_consistent_mesh_unchanged(self):
        """A closed icosphere is already winding-consistent; faces shouldn't move."""
        mesh = trimesh.creation.icosphere(subdivisions=1)
        before = mesh.faces.copy()
        _fast_fix_normals(mesh)
        assert mesh.faces.tolist() == before.tolist()

    def test_flips_inverted_sphere_outward(self):
        """A sphere with all faces flipped inward should end up outward-facing."""
        mesh = trimesh.creation.icosphere(subdivisions=1)
        mesh.invert()
        assert mesh.volume < 0.0
        _fast_fix_normals(mesh)
        assert mesh.volume > 0.0


# ===========================================================================
# fixing_functions.manual_hole_fill
# ===========================================================================


class TestManualHoleFill:
    def test_adds_faces_to_open_mesh(self):
        """Box with top cap removed gets new faces from hole fill."""
        box = trimesh.creation.box()
        top_mask = box.face_normals[:, 2] < 0.9  # keep all non-top faces
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
        updated = remove_labeled_points_from_mesh(
            grid_results, region_keys="rca_points"
        )
        new_verts = {tuple(v) for v in updated["mesh"].vertices}
        rca_set = {(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)}
        assert rca_set.isdisjoint(new_verts)

    def test_boundary_points_populated(self, grid_results):
        updated = remove_labeled_points_from_mesh(
            grid_results, region_keys="rca_points"
        )
        assert len(updated["boundary_points"]) > 0

    def test_removed_key_cleared(self, grid_results):
        updated = remove_labeled_points_from_mesh(
            grid_results, region_keys="rca_points"
        )
        assert updated["rca_points"] == []

    def test_empty_region_is_noop(self, grid_results):
        grid_results["rca_points"] = []
        n_before = len(grid_results["mesh"].vertices)
        updated = remove_labeled_points_from_mesh(
            grid_results, region_keys="rca_points"
        )
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
        updated = remove_labeled_points_from_mesh(
            grid_results, region_keys="rca_points"
        )
        new_verts = {tuple(v) for v in updated["mesh"].vertices}
        for key in ("aorta_points", "lca_points"):
            for pt in updated.get(key, []):
                assert tuple(pt) in new_verts

    def test_string_or_list_region_keys(self, grid_results):
        """String and single-item list should produce identical results."""
        import copy

        r1 = remove_labeled_points_from_mesh(
            copy.deepcopy(grid_results), region_keys="rca_points"
        )
        r2 = remove_labeled_points_from_mesh(
            copy.deepcopy(grid_results), region_keys=["rca_points"]
        )
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
        new_mesh = trimesh.Trimesh(
            vertices=new_verts, faces=grid_mesh.faces, process=False
        )
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
        return [
            (radius * float(np.cos(a)), radius * float(np.sin(a)), z) for a in angles
        ]

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
        patch = _stitch_boundary_ring(
            boundary_pts, iv_pts, n_iv // n_b, outward_direction=outward
        )
        valid = ~np.isnan(patch.face_normals).any(axis=1)
        if valid.any():
            avg_normal = patch.face_normals[valid].mean(axis=0)
            assert np.dot(avg_normal, outward) > 0


# ---------------------------------------------------------------------------
# Additional mesh factories for ostium tests
# ---------------------------------------------------------------------------


def _make_concentric_ring_mesh() -> trimesh.Trimesh:
    """Three concentric rings of 4 vertices each in the z=0 plane.

    Ring A (inner, radius 1): indices 0-3
    Ring B (middle, radius 2): indices 4-7
    Ring C (outer, radius 3): indices 8-11

    Faces connect adjacent rings so adjacency map gives
    A-neighbours = B and B-neighbours = C.
    """
    angles = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    ring_a = [(np.cos(a), np.sin(a), 0.0) for a in angles]
    ring_b = [(2.0 * np.cos(a), 2.0 * np.sin(a), 0.0) for a in angles]
    ring_c = [(3.0 * np.cos(a), 3.0 * np.sin(a), 0.0) for a in angles]
    verts = np.array(ring_a + ring_b + ring_c, dtype=float)
    faces = []
    for i in range(4):
        j = (i + 1) % 4
        faces.extend(
            [
                [i, j, i + 4],
                [j, j + 4, i + 4],
                [i + 4, j + 4, i + 8],
                [j + 4, j + 8, i + 8],
            ]
        )
    return trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)


def _make_annular_xz_mesh() -> trimesh.Trimesh:
    """Annular mesh lying in the XZ plane (y=0).

    Inner ring (radius 1, 8 verts): indices 0-7  — these become boundary_points.
    Outer ring (radius 2, 8 verts): indices 8-15 — second aortic layer.

    The ring plane has normal [0,1,0]; the IV ring (XY plane) has normal [0,0,1].
    The angle between them is 90°, so the anomalous clamping path is triggered.
    """
    n = 8
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    inner = np.column_stack([np.cos(angles), np.zeros(n), np.sin(angles)])
    outer = np.column_stack([2.0 * np.cos(angles), np.zeros(n), 2.0 * np.sin(angles)])
    verts = np.vstack([inner, outer])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.extend([[i, j, i + n], [j, j + n, i + n]])
    return trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)


# ===========================================================================
# manipulating._clamp_to_plane
# ===========================================================================


class TestClampToPlane:
    """Plane: z=0, normal=[0,0,1]. Correct side is z>0."""

    _origin = np.array([0.0, 0.0, 0.0])
    _normal = np.array([0.0, 0.0, 1.0])

    def test_wrong_side_point_projected_onto_plane(self):
        pts = [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.5, 0.0, -0.5)]
        result = _clamp_to_plane(pts, self._origin, self._normal, overshoot=0.0)
        assert result[2][2] == pytest.approx(0.0, abs=1e-10)
        assert result[0][2] == pytest.approx(1.0)
        assert result[1][2] == pytest.approx(1.0)

    def test_correct_side_points_unchanged_without_overshoot(self):
        pts = [(0.0, 0.0, 0.5), (1.0, 0.0, 1.5), (0.0, 1.0, 2.0)]
        result = _clamp_to_plane(pts, self._origin, self._normal, overshoot=0.0)
        for orig, res in zip(pts, result):
            assert res == pytest.approx(orig)

    def test_overshoot_pushes_wrong_side_past_plane(self):
        """Wrong-side point clamped to plane then pushed 1 mm past it."""
        pts = [(0.0, 0.0, 2.0), (0.0, 0.0, -0.5)]
        result = _clamp_to_plane(pts, self._origin, self._normal, overshoot=1.0)
        assert result[0][2] == pytest.approx(2.0)  # already beyond overshoot
        assert result[1][2] == pytest.approx(1.0)  # clamped to 0, then pushed to 1

    def test_overshoot_pushes_near_plane_correct_side_point(self):
        """Correct-side point within overshoot distance is pushed to that distance."""
        pts = [(0.0, 0.0, 3.0), (0.0, 0.0, 0.3)]
        result = _clamp_to_plane(pts, self._origin, self._normal, overshoot=1.0)
        assert result[0][2] == pytest.approx(3.0)  # far enough, unchanged
        assert result[1][2] == pytest.approx(1.0)  # 0.3 < 1.0, pushed

    def test_all_points_satisfy_minimum_gap(self):
        pts = [(float(i), 0.0, float(i % 5) * 0.2 - 0.2) for i in range(10)]
        result = _clamp_to_plane(pts, self._origin, self._normal, overshoot=1.0)
        assert all(p[2] >= 1.0 - 1e-9 for p in result)

    def test_returns_list_of_3_tuples(self):
        pts = [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0)]
        result = _clamp_to_plane(pts, self._origin, self._normal)
        assert isinstance(result, list)
        assert all(isinstance(p, tuple) and len(p) == 3 for p in result)


# ===========================================================================
# manipulating._enforce_layer_gap_from_plane
# ===========================================================================


class TestEnforceLayerGapFromPlane:
    """Uses the concentric-ring mesh (z=0 plane, IV normal=[0,0,1]).

    Seed = ring A (radius 1, indices 0-3).
    IV centre = origin.  Radial push per ring = 0.1 mm.

    Ring B (radius 2) → pushed 0.1 mm outward → expected radius 2.1.
    Ring C (radius 3) → pushed 0.2 mm outward → expected radius 3.2.
    """

    _origin = np.array([0.0, 0.0, 0.0])
    _normal = np.array([0.0, 0.0, 1.0])
    _seeds = {0, 1, 2, 3}

    def test_ring1_pushed_radially_outward(self):
        mesh = _make_concentric_ring_mesh()
        result = _enforce_layer_gap_from_plane(
            mesh, self._seeds, self._origin, self._normal, layer_step_mm=0.1
        )
        for i in range(4, 8):
            r = np.linalg.norm(result.vertices[i, :2])
            assert r == pytest.approx(2.1, abs=1e-6)

    def test_ring2_pushed_twice_the_step(self):
        mesh = _make_concentric_ring_mesh()
        result = _enforce_layer_gap_from_plane(
            mesh, self._seeds, self._origin, self._normal, layer_step_mm=0.1
        )
        for i in range(8, 12):
            r = np.linalg.norm(result.vertices[i, :2])
            assert r == pytest.approx(3.2, abs=1e-6)

    def test_seed_vertices_untouched(self):
        mesh = _make_concentric_ring_mesh()
        result = _enforce_layer_gap_from_plane(
            mesh, self._seeds, self._origin, self._normal, layer_step_mm=0.1
        )
        for i in self._seeds:
            np.testing.assert_allclose(result.vertices[i], mesh.vertices[i])

    def test_z_coordinates_unchanged(self):
        """Push is within the IV plane; z must not change."""
        mesh = _make_concentric_ring_mesh()
        result = _enforce_layer_gap_from_plane(
            mesh, self._seeds, self._origin, self._normal, layer_step_mm=0.1
        )
        np.testing.assert_allclose(
            result.vertices[:, 2], mesh.vertices[:, 2], atol=1e-10
        )

    def test_returns_trimesh(self):
        mesh = _make_concentric_ring_mesh()
        result = _enforce_layer_gap_from_plane(
            mesh, self._seeds, self._origin, self._normal
        )
        assert isinstance(result, trimesh.Trimesh)


# ===========================================================================
# manipulating._prepare_prox_dist_boundary_pts
# ===========================================================================


class TestPrepareProxDistBoundaryPts:
    """Two sub-cases: non-anomalous (proximal_is_ostium=False) and anomalous."""

    # ------------------------------------------------------------------
    # Non-anomalous: proximal path must be identical to the distal path
    # (both use order_points_list — no projection, no clamping).
    # ------------------------------------------------------------------

    def test_non_anomalous_prox_same_as_distal_algo(self):
        """proximal_is_ostium=False: prox result is just ordered by adjacency, same as dist."""
        mesh = _make_hex_fan_mesh()
        outer = [tuple(mesh.vertices[i]) for i in range(6)]

        # Put three outer vertices near prox_centroid, three near dist_centroid
        prox_centroid = (1.0, 0.0, 0.0)  # near vertex 0
        dist_centroid = (-1.0, 0.0, 0.0)  # near vertex 3
        results = {"boundary_points": outer}

        prox_pts, dist_pts, _ = _prepare_prox_dist_boundary_pts(
            mesh, results, prox_centroid, dist_centroid, proximal_is_ostium=False
        )

        # Sets must cover all outer vertices between them
        assert set(prox_pts) | set(dist_pts) == set(outer)
        # Both halves are non-empty
        assert len(prox_pts) > 0 and len(dist_pts) > 0

    def test_non_anomalous_prox_points_are_mesh_adjacent(self):
        """Proximal points returned by non-anomalous path are edge-connected in order."""
        from multimodars.multimodars import build_adjacency_map

        mesh = _make_hex_fan_mesh()
        outer = [tuple(mesh.vertices[i]) for i in range(6)]
        prox_centroid = (1.0, 0.0, 0.0)
        dist_centroid = (-100.0, 0.0, 0.0)  # all vertices are proximal
        results = {"boundary_points": outer}

        prox_pts, _, _ = _prepare_prox_dist_boundary_pts(
            mesh, results, prox_centroid, dist_centroid, proximal_is_ostium=False
        )

        adj = build_adjacency_map(mesh.faces.tolist())
        coord_to_idx = {tuple(v): i for i, v in enumerate(mesh.vertices)}
        for k in range(len(prox_pts) - 1):
            a = coord_to_idx[tuple(prox_pts[k])]
            b = coord_to_idx[tuple(prox_pts[k + 1])]
            assert b in adj.get(a, [])

    # ------------------------------------------------------------------
    # Anomalous: boundary ring in XZ plane (normal [0,1,0]),
    # IV frame in XY plane (normal [0,0,1]).  Angle = 90° > 45° threshold
    # → clamping + overshoot are applied.
    # ------------------------------------------------------------------

    def _make_anomalous_results(self):
        mesh = _make_annular_xz_mesh()
        inner = [tuple(mesh.vertices[i]) for i in range(8)]
        return mesh, {"boundary_points": inner}

    def _make_iv_frame_xy(self, n: int = 8, radius: float = 0.5):
        """IV lumen ring in the XY plane (z=0), normal ≈ [0,0,1]."""
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        coords = [
            (radius * float(np.cos(a)), radius * float(np.sin(a)), 0.0) for a in angles
        ]
        return _make_iv_pts(coords)

    def test_anomalous_prox_boundary_pts_respect_overshoot(self):
        """After clamping, all proximal boundary points must be >= overshoot from IV plane."""
        mesh, results = self._make_anomalous_results()
        iv_pts = self._make_iv_frame_xy()
        overshoot = 1.0
        # IV plane: z=0 (normal [0,0,1]), prox_centroid at origin
        prox_centroid = (0.0, 0.0, 0.0)
        dist_centroid = (100.0, 0.0, 0.0)  # all boundary pts go to prox

        prox_pts, _, _ = _prepare_prox_dist_boundary_pts(
            mesh,
            results,
            prox_centroid,
            dist_centroid,
            proximal_is_ostium=True,
            proximal_iv_frame_pts=iv_pts,
            clamp_overshoot=overshoot,
        )

        # IV plane normal is [0,0,1]; correct side is z>0 after clamping.
        # All returned proximal points must be >= overshoot from IV plane.
        assert all(p[2] >= overshoot - 1e-6 for p in prox_pts), (
            f"Some prox boundary points are closer than {overshoot} mm to IV plane: "
            f"{[p[2] for p in prox_pts]}"
        )

    def test_anomalous_no_prox_point_on_wrong_side(self):
        """No proximal boundary point must end up on the intravascular (z<0) side."""
        mesh, results = self._make_anomalous_results()
        iv_pts = self._make_iv_frame_xy()
        prox_centroid = (0.0, 0.0, 0.0)
        dist_centroid = (100.0, 0.0, 0.0)

        prox_pts, _, _ = _prepare_prox_dist_boundary_pts(
            mesh,
            results,
            prox_centroid,
            dist_centroid,
            proximal_is_ostium=True,
            proximal_iv_frame_pts=iv_pts,
        )

        assert all(p[2] >= -1e-6 for p in prox_pts)

    def test_anomalous_outer_ring_pushed_radially_outward(self):
        """After clamping, outer-ring mesh vertices are shifted radially away from IV centre."""
        mesh, results = self._make_anomalous_results()
        iv_pts = self._make_iv_frame_xy()
        prox_centroid = (0.0, 0.0, 0.0)
        dist_centroid = (100.0, 0.0, 0.0)

        _, _, updated_mesh = _prepare_prox_dist_boundary_pts(
            mesh,
            results,
            prox_centroid,
            dist_centroid,
            proximal_is_ostium=True,
            proximal_iv_frame_pts=iv_pts,
        )

        # Outer ring (indices 8-15) should be pushed radially outward.
        # Vertices with x != 0 in the IV-plane projection are the ones that move.
        moved = False
        for i in range(8, 16):
            old_r = np.linalg.norm(mesh.vertices[i, [0, 1]])  # XY radius
            new_r = np.linalg.norm(updated_mesh.vertices[i, [0, 1]])
            if old_r > 1e-6:  # skip vertices projecting to IV centre
                assert (
                    new_r >= old_r - 1e-6
                ), f"Vertex {i} moved inward: {old_r:.4f} → {new_r:.4f}"
                if new_r > old_r + 1e-6:
                    moved = True
        assert moved, "Expected at least some outer-ring vertices to move outward"

    def test_non_anomalous_mesh_unchanged(self):
        """proximal_is_ostium=False must not modify mesh vertex positions."""
        mesh = _make_hex_fan_mesh()
        outer = [tuple(mesh.vertices[i]) for i in range(6)]
        results = {"boundary_points": outer}
        prox_centroid = (1.0, 0.0, 0.0)
        dist_centroid = (-100.0, 0.0, 0.0)

        _, _, updated_mesh = _prepare_prox_dist_boundary_pts(
            mesh, results, prox_centroid, dist_centroid, proximal_is_ostium=False
        )

        np.testing.assert_allclose(updated_mesh.vertices, mesh.vertices)
