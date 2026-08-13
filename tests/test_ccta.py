"""Tests for the multimodars.ccta module.

Covers:
  - multimodars.multimodars: find_faces_near_points, find_aortic_points,
                final_reclassification (Rust bindings backing
                labeling.label_geometry's occlusion-removal and
                adjacency-based label-smoothing steps)
  - labeling: _keep_largest_connected_component (island-point filter for
                find_points_by_cl_region's proximal/distal/anomalous output)
  - fixing_functions: manual_hole_fill, postprocess_stitched_mesh
  - stitching: remove_labeled_points_from_mesh,
               keep_labeled_points_from_mesh, order_points_list,
               _rotate_to_nearest_iv, _fix_ring_direction_by_distance,
               _stitch_rings, _prepare_prox_dist_boundary_pts,
               _condition_ostium_ring, _clamp_to_plane,
               _enforce_layer_gap_from_plane, and the ring conditioning
               helpers (_redistribute_ring_evenly,
               _smooth_ring_preserving_size, _densify_boundary,
               _assign_rings_to_ends, _shift_plane_clear_of, _toward_aorta)
  - boundary: open_boundary_edges, order_boundary_rings, clean_open_boundary
  - scaling: scale_region_centerline_morphing, sync_results_to_mesh
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
from multimodars.ccta.labeling import _keep_largest_connected_component
from multimodars.multimodars import (
    find_faces_near_points,
    find_aortic_points,
    final_reclassification,
)
from multimodars.ccta.boundary import (
    clean_open_boundary,
    open_boundary_edges,
    order_boundary_rings,
)
from multimodars.ccta.stitching import (
    _assign_rings_to_ends,
    _clamp_to_plane,
    _condition_ostium_ring,
    _densify_boundary,
    _enforce_layer_gap_from_plane,
    _fast_fix_normals,
    _fix_ring_direction_by_distance,
    _prepare_prox_dist_boundary_pts,
    _redistribute_ring_evenly,
    _ring_calibre,
    _rotate_to_nearest_iv,
    _shift_plane_clear_of,
    _smooth_ring_preserving_size,
    _stitch_rings,
    _toward_aorta,
    keep_labeled_points_from_mesh,
    order_points_list,
    remove_labeled_points_from_mesh,
)

from multimodars.ccta.scaling import (
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


def _make_restore_blob_mesh() -> trimesh.Trimesh:
    """9 vertices, 3 triangular faces (plus one degenerate face to link an
    isolated pair) - purpose-built to demonstrate Logic B's majority-vote
    label propagation.

    Vertex 0 (aorta) bridges removed vertices 1 and 2, each of which also
    touches two distinct "outer" vertices (3,4 for vertex 1; 5,6 for vertex
    2). Vertex 1's own real neighbours are {0,3,4} and vertex 2's are
    {0,5,6} - each resolved independently in round 0 by its own local
    majority, plus a fully isolated removed pair (7, 8) with no external
    connectivity at all.
    """
    verts = np.array([[i, 0.0, 0.0] for i in range(9)], dtype=float)
    faces = np.array([[1, 0, 2], [1, 3, 4], [2, 5, 6]])
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _make_island_mesh() -> trimesh.Trimesh:
    """12 vertices - purpose-built to demonstrate component-level Logic A.

    A 2-vertex candidate island {0,1} borders a 6-vertex cluster {2..7} on
    one side, and a fully separate, larger 4-vertex cluster {8..11} has zero
    connectivity to the rest (so it's always the "largest" component when
    {0,1} is the subject label, and never itself when {2..7} is).

    Vertex 0's neighbours: {1,2,3,4}; vertex 1's neighbours: {0,2,4,5}.
    Combined external boundary of component {0,1} is exactly {2,3,4,5}.
    """
    verts = np.array([[i, 0.0, 0.0] for i in range(12)], dtype=float)
    faces = np.array(
        [
            [0, 1, 2],
            [1, 4, 5],
            [0, 3, 4],
            [2, 3, 6],
            [4, 5, 7],
            [6, 7, 3],
            [8, 9, 10],
            [8, 10, 11],
        ]
    )
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _make_chain_mesh() -> trimesh.Trimesh:
    """6-vertex chain 0(aorta)-1(removed)-2(removed)-3(removed)-4(removed)-
    5(RCA), purpose-built to demonstrate Logic B's multi-round propagation.

    Vertices 2 and 3 have zero direct real neighbours - they can only be
    resolved by inheriting evidence from 1 and 4 respectively. This is a
    minimal abstraction of a real intramural "box" patch: one connected
    removed component with a large aorta-facing face and a smaller
    RCA-facing face, joined through the same fold.

    Degenerate (repeated-vertex) faces, one per desired edge, so the graph
    is exactly the path 0-1-2-3-4-5 with no incidental extra edges.
    """
    verts = np.array([[i, 0.0, 0.0] for i in range(6)], dtype=float)
    faces = np.array([[0, 1, 1], [1, 2, 2], [2, 3, 3], [3, 4, 4], [4, 5, 5]])
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
def restore_blob_mesh():
    return _make_restore_blob_mesh()


@pytest.fixture
def island_mesh():
    return _make_island_mesh()


@pytest.fixture
def chain_mesh():
    return _make_chain_mesh()


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
        """Vertex 0 labelled RCA; its neighbours (1, 3) are aorta → reclassified.

        {6,7,8} form a separate, larger (size-3) RCA component elsewhere in
        the grid, disconnected from vertex 0's {1,3} neighbourhood - needed
        so vertex 0 is correctly the minority island rather than the sole
        (and therefore protected) component.
        """
        verts = [tuple(v) for v in grid_mesh.vertices]
        new = self._call(grid_mesh, rca=[verts[0], verts[6], verts[7], verts[8]])
        assert verts[0] not in new["rca_points"]
        assert verts[0] in new["aorta_points"]
        assert verts[6] in new["rca_points"]
        assert verts[7] in new["rca_points"]
        assert verts[8] in new["rca_points"]

    def test_isolated_lca_becomes_aorta(self, grid_mesh):
        verts = [tuple(v) for v in grid_mesh.vertices]
        new = self._call(grid_mesh, lca=[verts[0], verts[6], verts[7], verts[8]])
        assert verts[0] not in new["lca_points"]
        assert verts[0] in new["aorta_points"]
        assert verts[6] in new["lca_points"]
        assert verts[7] in new["lca_points"]
        assert verts[8] in new["lca_points"]

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
    # Logic B (majority-vote propagation): each removed vertex is judged
    # against its own directly-adjacent real neighbours first; undecided
    # vertices inherit evidence from already-resolved neighbours in
    # synchronized rounds, so a genuinely mixed removed patch splits
    # sub-region by sub-region instead of being judged as one atomic blob.
    # ------------------------------------------------------------------

    def test_restores_via_local_majority(self, restore_blob_mesh):
        """Vertex 1's real neighbours are {0(aorta), 3(RCA), 4(RCA)} - 2/3
        RCA, a local majority that decides it directly in round 0, with no
        need to consult the rest of the component. Same for vertex 2 via
        {0(aorta), 5(RCA), 6(RCA)}.
        """
        verts = [tuple(v) for v in restore_blob_mesh.vertices]
        new = self._call(
            restore_blob_mesh,
            rca=[verts[3], verts[4], verts[5], verts[6]],
            rca_removed=[verts[1], verts[2]],
        )
        assert verts[1] in new["rca_points"]
        assert verts[2] in new["rca_points"]
        assert not new["rca_removed_points"]

    def test_keeps_removed_when_local_majority_aorta(self, restore_blob_mesh):
        """Only vertices 4 and 6 are RCA (one per removed vertex); 0, 3 and 5
        default to aorta, so vertex 1's real neighbours {0(aorta), 3(aorta),
        4(RCA)} and vertex 2's {0(aorta), 5(aorta), 6(RCA)} are each
        individually 2/3 aorta -> both stay removed (a genuine occlusion,
        not a false positive).
        """
        verts = [tuple(v) for v in restore_blob_mesh.vertices]
        new = self._call(
            restore_blob_mesh,
            rca=[verts[4], verts[6]],
            rca_removed=[verts[1], verts[2]],
        )
        assert verts[1] in new["rca_removed_points"]
        assert verts[2] in new["rca_removed_points"]
        assert verts[1] not in new["rca_points"]
        assert verts[2] not in new["rca_points"]

    def test_splits_chain_by_propagation(self, chain_mesh):
        """A single connected removed component (a chain
        0(aorta)-1-2-3-4-5(RCA)) splits along its length: vertices 2 and 3
        have zero direct real neighbours and only resolve by inheriting
        evidence from 1 and 4 respectively in round 1. The old
        whole-component vote saw one aorta neighbour and one RCA neighbour
        (50/50) and would have left all four removed.
        """
        verts = [tuple(v) for v in chain_mesh.vertices]
        new = self._call(
            chain_mesh,
            rca=[verts[5]],
            rca_removed=[verts[1], verts[2], verts[3], verts[4]],
        )
        assert verts[1] in new["rca_removed_points"]
        assert verts[2] in new["rca_removed_points"]
        assert verts[3] in new["rca_points"]
        assert verts[4] in new["rca_points"]

    # ------------------------------------------------------------------
    # Logic A (component-level): a small aorta component mesh-surrounded by
    # RCA/LCA is reclassified based on its boundary majority, but the single
    # largest component of a label is always excluded, even when it's the
    # only one - never reclassified wholesale.
    # ------------------------------------------------------------------

    def test_aorta_island_promoted_to_rca(self, island_mesh):
        """{2..7} labelled RCA; {0,1} and {8..11} default to aorta. Aorta
        splits into {0,1} (size 2) and {8..11} (size 4) - the larger is
        excluded as the presumed main body, leaving {0,1}'s 100%-RCA
        boundary {2,3,4,5} to promote it.
        """
        verts = [tuple(v) for v in island_mesh.vertices]
        new = self._call(island_mesh, rca=verts[2:8])
        assert verts[0] in new["rca_points"]
        assert verts[1] in new["rca_points"]

    def test_aorta_island_promoted_to_lca(self, island_mesh):
        verts = [tuple(v) for v in island_mesh.vertices]
        new = self._call(island_mesh, lca=verts[2:8])
        assert verts[0] in new["lca_points"]
        assert verts[1] in new["lca_points"]

    def test_aorta_island_stays_when_boundary_mixed(self, island_mesh):
        """{0,1}'s boundary {2,3,4,5} is split 2 RCA / 2 LCA - neither clears
        70%, so the island must stay aorta.
        """
        verts = [tuple(v) for v in island_mesh.vertices]
        new = self._call(island_mesh, rca=verts[2:4], lca=verts[4:6])
        assert verts[0] in new["aorta_points"]
        assert verts[1] in new["aorta_points"]
        assert verts[0] not in new["rca_points"]
        assert verts[0] not in new["lca_points"]

    def test_largest_component_never_reclassified(self, island_mesh):
        """{0,1,8..11} labelled RCA, leaving {2..7} as the sole aorta
        component (no islands at all) - even though it borders RCA
        extensively, it must never be reclassified: it's the (only, and
        therefore always "largest") component, which is exactly the
        catastrophic-mass-reclassification risk the "always exclude
        largest" rule guards against.
        """
        verts = [tuple(v) for v in island_mesh.vertices]
        new = self._call(island_mesh, rca=verts[0:2] + verts[8:12])
        for v in verts[2:8]:
            assert v in new["aorta_points"]
            assert v not in new["rca_points"]

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
# labeling._keep_largest_connected_component
# (island-point filter for find_points_by_cl_region's proximal/distal/
# anomalous output - see grid_mesh layout in the module docstring above:
# vertices {0,1,3,4} are mutually mesh-adjacent; vertex 8's neighbours are
# {5,7}, neither of which is in that cluster, so it's a true singleton
# within the induced subgraph over {0,1,3,4,8}.)
# ===========================================================================


class TestKeepLargestConnectedComponent:
    def test_drops_isolated_point(self, grid_mesh):
        verts = [tuple(v) for v in grid_mesh.vertices]
        points = [verts[0], verts[1], verts[3], verts[4], verts[8]]
        result = _keep_largest_connected_component(grid_mesh, points)
        assert set(result) == {verts[0], verts[1], verts[3], verts[4]}
        assert verts[8] not in result

    def test_fully_connected_set_unchanged(self, grid_mesh):
        verts = [tuple(v) for v in grid_mesh.vertices]
        points = [verts[0], verts[1], verts[3]]
        result = _keep_largest_connected_component(grid_mesh, points)
        assert set(result) == set(points)

    def test_empty_input_returns_empty(self, grid_mesh):
        assert _keep_largest_connected_component(grid_mesh, []) == []

    def test_single_point_returns_unchanged(self, grid_mesh):
        verts = [tuple(v) for v in grid_mesh.vertices]
        points = [verts[0]]
        assert _keep_largest_connected_component(grid_mesh, points) == points

    def test_points_not_on_mesh_returned_unchanged(self, grid_mesh):
        points = [(99.0, 99.0, 99.0), (100.0, 100.0, 100.0)]
        assert _keep_largest_connected_component(grid_mesh, points) == points


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
# stitching._stitch_rings
# ===========================================================================


def _ring_coords(n: int, radius: float = 1.0, z: float = 0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(radius * float(np.cos(a)), radius * float(np.sin(a)), z) for a in angles]


class TestStitchRings:
    def _ring_pts(self, n: int, radius: float = 1.0, z: float = 0.0):
        return _ring_coords(n, radius, z)

    def test_creates_trimesh(self):
        patch = _stitch_rings(
            self._ring_pts(6), _make_iv_pts(self._ring_pts(12, radius=1.2))
        )
        assert isinstance(patch, trimesh.Trimesh)

    def test_vertex_count(self):
        n_b, n_iv = 6, 12
        patch = _stitch_rings(
            self._ring_pts(n_b), _make_iv_pts(self._ring_pts(n_iv, radius=1.2))
        )
        assert len(patch.vertices) == n_b + n_iv

    def test_no_nan_vertices(self):
        patch = _stitch_rings(
            self._ring_pts(4), _make_iv_pts(self._ring_pts(8, radius=1.5))
        )
        assert not np.isnan(patch.vertices).any()

    @pytest.mark.parametrize(
        "n_b, n_iv", [(6, 6), (4, 6), (8, 24), (10, 100), (67, 100), (33, 50)]
    )
    def test_strip_is_closed_annulus(self, n_b, n_iv):
        """A closed annulus needs exactly n_b + n_iv triangles and no extra holes.

        The predecessor emitted only n_iv triangles, leaving one hole per boundary
        segment; the surface only closed because fill_holes ran afterwards.
        """
        patch = _stitch_rings(
            self._ring_pts(n_b), _make_iv_pts(self._ring_pts(n_iv, radius=1.6, z=1.0))
        )
        assert len(patch.faces) == n_b + n_iv
        # The only open edges are the two rings the strip is bounded by.
        assert len(open_boundary_edges(patch.faces)) == n_b + n_iv

    def test_no_degenerate_faces(self):
        patch = _stitch_rings(
            self._ring_pts(10), _make_iv_pts(self._ring_pts(100, radius=1.6, z=1.0))
        )
        assert patch.area_faces.min() > 0.0

    def test_rejects_tiny_rings(self):
        with pytest.raises(ValueError, match="at least 3 points"):
            _stitch_rings(self._ring_pts(2), _make_iv_pts(self._ring_pts(8)))

    def test_outward_direction_orients_patch(self):
        """When outward_direction is given, average face normal should align with it."""
        boundary_pts = self._ring_pts(6, z=0.0)
        iv_pts = _make_iv_pts(self._ring_pts(12, radius=1.2, z=0.0))
        outward = np.array([0.0, 0.0, 1.0])
        patch = _stitch_rings(boundary_pts, iv_pts, outward_direction=outward)
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


def _make_two_rim_tube(n_theta: int = 12, jitter: float = 0.0) -> trimesh.Trimesh:
    """Open tube split into two halves, giving four rims at z=0,1,2,3.

    The band between z=1 and z=2 is omitted, so the two halves are separate and
    the outermost rims (z=0 and z=3) are unambiguously nearest to a centroid
    below and above the tube.  *jitter* roughens those two outer rims radially
    and along z, so ring conditioning (projection + respacing) has visible work
    to do - a perfect circle would be conditioned into itself.
    """
    angles = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    rings = [0.0, 1.0, 2.0, 3.0]
    rng = np.random.default_rng(0)
    verts = []
    for j, z in enumerate(rings):
        outer_rim = j in (0, 3)
        for a in angles:
            r = 1.0
            dz = 0.0
            if jitter and outer_rim:
                r += jitter * float(rng.standard_normal())
                dz = jitter * float(rng.standard_normal())
            verts.append([r * np.cos(a), r * np.sin(a), z + dz])
    verts = np.array(verts, dtype=float)

    def vid(i, j):
        return j * n_theta + (i % n_theta)

    faces = []
    for j in (0, 2):  # skip the band between z=1 and z=2 -> separate halves
        for i in range(n_theta):
            faces.extend(
                [
                    [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)],
                    [vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)],
                ]
            )
    return trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)


class TestPrepareProxDistBoundaryPts:
    """Stitching now consumes two whole rings rather than one flat point list."""

    N_THETA = 12

    def _two_rim_results(self, jitter: float = 0.0):
        mesh = _make_two_rim_tube(self.N_THETA, jitter=jitter)
        rings = order_boundary_rings(mesh.faces, mesh.vertices)
        assert len(rings) == 4, "tube should expose 4 rims (2 per half)"
        results = {
            f"boundary_points_{k + 1}": [tuple(mesh.vertices[i]) for i in ring]
            for k, ring in enumerate(rings)
        }
        return mesh, results

    def test_requires_two_rings(self):
        """A single stored ring is a usage error, not a ZeroDivisionError deeper in."""
        mesh = _make_hex_fan_mesh()
        results = {"boundary_points_1": [tuple(mesh.vertices[i]) for i in range(6)]}
        with pytest.raises(ValueError, match="target_boundaries=2"):
            _prepare_prox_dist_boundary_pts(
                mesh, results, (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)
            )

    def test_rings_assigned_whole_not_split(self):
        """Each returned ring must come from exactly one stored ring, never a mix."""
        mesh, results = self._two_rim_results()
        stored = [set(v) for k, v in results.items()]

        prox_pts, dist_pts, _ = _prepare_prox_dist_boundary_pts(
            mesh,
            results,
            (0.0, 0.0, -5.0),
            (0.0, 0.0, 8.0),
            proximal_is_ostium=False,
        )

        assert len(prox_pts) > 0 and len(dist_pts) > 0
        # Points move (projection/respacing), so compare counts against a stored
        # ring rather than coordinates: a torn ring would not match any of them.
        assert any(len(prox_pts) == len(s) for s in stored)
        assert any(len(dist_pts) == len(s) for s in stored)

    def test_prox_ring_is_nearer_prox_centroid(self):
        mesh, results = self._two_rim_results()
        prox_pts, dist_pts, _ = _prepare_prox_dist_boundary_pts(
            mesh,
            results,
            (0.0, 0.0, -5.0),
            (0.0, 0.0, 8.0),
            proximal_is_ostium=False,
        )
        assert np.mean([p[2] for p in prox_pts]) < np.mean([p[2] for p in dist_pts])

    def test_densifies_both_rings_to_target(self):
        mesh, results = self._two_rim_results()
        prox_pts, dist_pts, _ = _prepare_prox_dist_boundary_pts(
            mesh,
            results,
            (0.0, 0.0, -5.0),
            (0.0, 0.0, 8.0),
            proximal_is_ostium=False,
            target_n=40,
        )
        assert len(prox_pts) == 40
        assert len(dist_pts) == 40

    def test_conditioning_applies_to_both_rims(self):
        """Both rims are conditioned now, not just the proximal ostium one."""
        mesh, results = self._two_rim_results(jitter=0.15)
        _, _, updated = _prepare_prox_dist_boundary_pts(
            mesh,
            results,
            (0.0, 0.0, -5.0),
            (0.0, 0.0, 8.0),
            proximal_is_ostium=False,
        )
        # Rings z=0 and z=3 are the two chosen rims; both must have been moved.
        for j in (0, 3):
            idx = list(range(j * self.N_THETA, (j + 1) * self.N_THETA))
            assert not np.allclose(
                updated.vertices[idx], mesh.vertices[idx]
            ), f"rim at ring {j} was not conditioned"

    def test_conditioned_rim_is_planar_and_evenly_spaced(self):
        mesh, results = self._two_rim_results(jitter=0.15)
        prox_pts, _, _ = _prepare_prox_dist_boundary_pts(
            mesh,
            results,
            (0.0, 0.0, -5.0),
            (0.0, 0.0, 8.0),
            proximal_is_ostium=False,
        )
        pts = np.asarray(prox_pts, dtype=np.float64)
        # Planar: spread along the best-fit normal collapses.
        centred = pts - pts.mean(axis=0)
        normal = np.linalg.svd(centred, full_matrices=False)[2][-1]
        assert np.abs(centred @ normal).max() < 1e-9
        # Evenly spaced: segment lengths are near-uniform.
        loop = np.vstack([pts, pts[:1]])
        seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
        assert seg.max() / seg.min() < 1.05


# ===========================================================================
# stitching._condition_ostium_ring
# ===========================================================================


class TestConditionOstiumRing:
    """Boundary ring in the XZ plane (normal [0,1,0]) vs IV frame in XY
    (normal [0,0,1]).  The 90 deg angle triggers the clamping path."""

    def _make_anomalous(self):
        mesh = _make_annular_xz_mesh()
        ring = [tuple(mesh.vertices[i]) for i in range(8)]
        return mesh, ring

    def _make_iv_frame_xy(self, n: int = 8, radius: float = 0.5):
        """IV lumen ring in the XY plane (z=0), normal ~ [0,0,1]."""
        return _make_iv_pts(_ring_coords(n, radius, 0.0))

    def test_boundary_pts_respect_overshoot(self):
        mesh, ring = self._make_anomalous()
        overshoot = 1.0
        out, _ = _condition_ostium_ring(
            mesh,
            ring,
            (0.0, 0.0, 0.0),
            self._make_iv_frame_xy(),
            None,
            45.0,
            overshoot,
        )
        assert all(abs(p[2]) >= overshoot - 1e-6 for p in out), (
            f"Some points are closer than {overshoot} mm to the IV plane: "
            f"{[p[2] for p in out]}"
        )

    def test_no_point_left_on_wrong_side(self):
        mesh, ring = self._make_anomalous()
        out, _ = _condition_ostium_ring(
            mesh, ring, (0.0, 0.0, 0.0), self._make_iv_frame_xy(), None, 45.0, 0.5
        )
        signs = {np.sign(round(p[2], 9)) for p in out}
        assert len(signs - {0.0}) == 1, f"points straddle the IV plane: {signs}"

    def test_outer_ring_pushed_radially_outward(self):
        mesh, ring = self._make_anomalous()
        _, updated = _condition_ostium_ring(
            mesh, ring, (0.0, 0.0, 0.0), self._make_iv_frame_xy(), None, 45.0, 0.5
        )
        moved = False
        for i in range(8, 16):
            old_r = float(np.linalg.norm(mesh.vertices[i, [0, 1]]))
            new_r = float(np.linalg.norm(updated.vertices[i, [0, 1]]))
            if old_r > 1e-6:
                assert (
                    new_r >= old_r - 1e-6
                ), f"Vertex {i} moved inward: {old_r:.4f} -> {new_r:.4f}"
                if new_r > old_r + 1e-6:
                    moved = True
        assert moved, "Expected at least some outer-ring vertices to move outward"

    def test_noop_without_iv_frame(self):
        mesh, ring = self._make_anomalous()
        out, updated = _condition_ostium_ring(
            mesh, ring, (0.0, 0.0, 0.0), None, None, 45.0, 0.5
        )
        assert out == ring
        np.testing.assert_allclose(updated.vertices, mesh.vertices)


# ---------------------------------------------------------------------------
# Grid-with-hole factory for boundary-ring tests
# ---------------------------------------------------------------------------


def _make_grid_with_hole(n: int = 9, remove=None):
    """Triangulated n x n grid with *remove* vertices' faces dropped.

    Defaults to the single centre vertex, leaving two open boundaries: the outer
    perimeter and a small inner rim around the hole.  Returns
    ``(mesh, removed_indices, rim_seed_indices)``.
    """

    def vid(i, j):
        return j * n + i

    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    verts = np.column_stack(
        [xs.ravel().astype(float), ys.ravel().astype(float), np.zeros(n * n)]
    )
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            faces.append([vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)])
            faces.append([vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)])
    faces = np.array(faces)

    removed = [vid(n // 2, n // 2)] if remove is None else [vid(*p) for p in remove]
    keep = ~np.any(np.isin(faces, removed), axis=1)
    kept_faces = faces[keep]

    # Rim seeds: kept vertices that shared a face with a removed vertex.
    touching = set(faces[~keep].ravel().tolist()) - set(removed)
    mesh = trimesh.Trimesh(vertices=verts, faces=kept_faces, process=False)
    return mesh, removed, touching


def _open_edge_degrees(faces):
    from collections import Counter

    counts = Counter()
    for a, b in open_boundary_edges(faces):
        counts[int(a)] += 1
        counts[int(b)] += 1
    return set(counts.values())


def _traces_real_edges(faces, ring_indices) -> bool:
    """True when every consecutive pair in *ring_indices* is a real open edge."""
    edges = {frozenset((int(a), int(b))) for a, b in open_boundary_edges(faces)}
    n = len(ring_indices)
    return all(
        frozenset((ring_indices[k], ring_indices[(k + 1) % n])) in edges
        for k in range(n)
    )


# ===========================================================================
# boundary.open_boundary_edges
# ===========================================================================


class TestOpenBoundaryEdges:
    def test_empty_faces(self):
        assert len(open_boundary_edges(np.empty((0, 3), dtype=np.int64))) == 0

    def test_closed_mesh_has_none(self):
        sphere = trimesh.creation.icosphere(subdivisions=2)
        assert len(open_boundary_edges(sphere.faces)) == 0

    def test_hole_and_perimeter_are_open(self):
        mesh, _, _ = _make_grid_with_hole()
        edges = open_boundary_edges(mesh.faces)
        # 9x9 grid perimeter = 32 edges, plus a 6-edge rim around the removed vertex
        assert len(edges) == 32 + 6

    def test_every_returned_edge_used_once(self):
        mesh, _, _ = _make_grid_with_hole()
        from collections import Counter

        tri = mesh.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        counts = Counter(frozenset((int(a), int(b))) for a, b in tri)
        for a, b in open_boundary_edges(mesh.faces):
            assert counts[frozenset((int(a), int(b)))] == 1


# ===========================================================================
# boundary.order_boundary_rings
# ===========================================================================


class TestOrderBoundaryRings:
    def test_reports_every_rim_without_seeds(self):
        mesh, _, _ = _make_grid_with_hole()
        rings = order_boundary_rings(mesh.faces, mesh.vertices)
        assert sorted(len(r) for r in rings) == [6, 32]

    def test_seeds_select_only_the_touching_rim(self):
        mesh, _, seeds = _make_grid_with_hole()
        rings = order_boundary_rings(mesh.faces, mesh.vertices, seeds)
        assert [len(r) for r in rings] == [6]

    def test_ring_traces_real_mesh_edges(self):
        mesh, _, seeds = _make_grid_with_hole()
        rings = order_boundary_rings(mesh.faces, mesh.vertices, seeds)
        assert _traces_real_edges(mesh.faces, rings[0])

    def test_target_n_reduces_ring_count(self):
        mesh, _, _ = _make_grid_with_hole()
        rings = order_boundary_rings(mesh.faces, mesh.vertices, target_n=1)
        assert len(rings) == 1

    def test_no_open_boundary_returns_empty(self):
        sphere = trimesh.creation.icosphere(subdivisions=2)
        assert order_boundary_rings(sphere.faces, sphere.vertices) == []


# ===========================================================================
# boundary.clean_open_boundary
# ===========================================================================


class TestCleanOpenBoundary:
    def test_clean_hole_needs_no_culling(self):
        mesh, _, seeds = _make_grid_with_hole()
        drop, rings = clean_open_boundary(mesh.faces, mesh.vertices, seeds)
        assert drop == set()
        assert [len(r) for r in rings] == [6]

    def test_pinch_junction_is_culled_from_the_mesh(self):
        """Two diagonally-touching holes share one rim vertex of degree 4.

        That vertex cannot sit on a clean ring, so it is removed from the mesh
        rather than merely skipped in the ring list.
        """
        mesh, _, seeds = _make_grid_with_hole(remove=[(4, 4), (6, 6)])
        raw_degrees = _open_edge_degrees(mesh.faces)
        assert 4 in raw_degrees, "fixture should expose a pinch junction"

        drop, rings = clean_open_boundary(mesh.faces, mesh.vertices, seeds)
        assert len(drop) == 1

        surviving = mesh.faces[~np.any(np.isin(mesh.faces, list(drop)), axis=1)]
        assert not np.isin(surviving, list(drop)).any()
        assert _open_edge_degrees(surviving) == {2}
        assert _traces_real_edges(surviving, rings[0])

    def test_returns_empty_when_seeds_match_nothing(self):
        mesh, _, _ = _make_grid_with_hole()
        drop, rings = clean_open_boundary(mesh.faces, mesh.vertices, {10_000})
        assert rings == []


# ===========================================================================
# stitching._redistribute_ring_evenly
# ===========================================================================


def _seg_lengths(pts):
    arr = np.asarray(pts, dtype=np.float64)
    loop = np.vstack([arr, arr[:1]])
    return np.linalg.norm(np.diff(loop, axis=0), axis=1)


class TestRedistributeRingEvenly:
    def _clustered_ring(self, n: int = 24):
        """Half the points crammed into a short arc."""
        half = n // 2
        angles = np.concatenate(
            [
                np.linspace(0, 0.4, half),
                np.linspace(0.5, 2 * np.pi, n - half, endpoint=False),
            ]
        )
        return [(float(np.cos(a)), float(np.sin(a)), 0.0) for a in angles]

    def test_spacing_becomes_uniform(self):
        ring = self._clustered_ring()
        before = _seg_lengths(ring)
        after = _seg_lengths(_redistribute_ring_evenly(ring))
        assert before.max() / before.min() > 5.0  # fixture really is clustered
        assert after.max() / after.min() < 1.05

    def test_count_preserved_and_start_fixed(self):
        ring = self._clustered_ring()
        out = _redistribute_ring_evenly(ring)
        assert len(out) == len(ring)
        np.testing.assert_allclose(out[0], ring[0])

    def test_does_not_shrink_the_ring(self):
        ring = self._clustered_ring()
        before = _seg_lengths(ring).sum()
        after = _seg_lengths(_redistribute_ring_evenly(ring)).sum()
        assert after == pytest.approx(before, rel=0.02)

    def test_can_change_count(self):
        assert len(_redistribute_ring_evenly(self._clustered_ring(), n_out=40)) == 40

    def test_degenerate_input_passes_through(self):
        pts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        assert _redistribute_ring_evenly(pts) == pts


# ===========================================================================
# stitching._smooth_ring_preserving_size
# ===========================================================================


class TestSmoothRingPreservingSize:
    def _noisy_ring(self, n: int, radius: float = 2.0, noise: float = 0.08):
        rng = np.random.default_rng(n)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radii = radius + noise * rng.standard_normal(n)
        return [
            (
                float(radii[i] * np.cos(a)),
                float(radii[i] * np.sin(a)),
                float(0.15 * rng.standard_normal()),
            )
            for i, a in enumerate(angles)
        ]

    @pytest.mark.parametrize("n", [17, 29, 50, 100])
    def test_calibre_is_preserved(self, n):
        """Plain Laplacian smoothing shrinks a ring, worst on coarse ones
        (~16 % at n=17), which showed up as a pinched distal seam."""
        ring = self._noisy_ring(n)
        before = _ring_calibre(np.asarray(ring, dtype=np.float64))
        after = _ring_calibre(
            np.asarray(_smooth_ring_preserving_size(ring), dtype=np.float64)
        )
        assert after == pytest.approx(before, rel=0.01)

    def test_still_reduces_roughness(self):
        ring = self._noisy_ring(17)

        def roughness(pts):
            arr = np.asarray(pts, dtype=np.float64)
            return float(np.linalg.norm(arr - arr.mean(axis=0), axis=1).std())

        assert roughness(_smooth_ring_preserving_size(ring)) < 0.5 * roughness(ring)

    def test_does_not_circularise_an_ellipse(self):
        """Scaling is uniform about the centroid - ostia are elliptical."""
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        ellipse = [
            (3.0 * float(np.cos(a)), 1.5 * float(np.sin(a)), 0.0) for a in angles
        ]
        out = np.asarray(_smooth_ring_preserving_size(ellipse), dtype=np.float64)
        assert np.ptp(out[:, 0]) / 2 == pytest.approx(3.0, rel=0.02)
        assert np.ptp(out[:, 1]) / 2 == pytest.approx(1.5, rel=0.02)


# ===========================================================================
# stitching._densify_boundary
# ===========================================================================


class TestDensifyBoundary:
    def _hole_rim(self):
        mesh, _, seeds = _make_grid_with_hole()
        ring_idx = order_boundary_rings(mesh.faces, mesh.vertices, seeds)[0]
        return mesh, [tuple(mesh.vertices[i]) for i in ring_idx]

    @pytest.mark.parametrize("target", [6, 12, 24, 100])
    def test_hits_exact_target_count(self, target):
        mesh, ring = self._hole_rim()
        _, dense = _densify_boundary(mesh, ring, target)
        assert len(dense) == max(target, len(ring))

    def test_adds_one_vertex_and_one_face_per_inserted_point(self):
        mesh, ring = self._hole_rim()
        target = 30
        new_mesh, dense = _densify_boundary(mesh, ring, target)
        extra = target - len(ring)
        assert len(new_mesh.vertices) == len(mesh.vertices) + extra
        assert len(new_mesh.faces) == len(mesh.faces) + extra

    def test_leaves_no_t_junctions(self):
        """Each subdivided edge's face is refanned onto its opposite vertex, so
        every rim vertex keeps open-edge degree 2."""
        mesh, ring = self._hole_rim()
        new_mesh, _ = _densify_boundary(mesh, ring, 40)
        assert _open_edge_degrees(new_mesh.faces) == {2}

    def test_dense_ring_traces_real_mesh_edges(self):
        mesh, ring = self._hole_rim()
        new_mesh, dense = _densify_boundary(mesh, ring, 40)
        coord_to_idx = {tuple(v): i for i, v in enumerate(new_mesh.vertices)}
        idx = [coord_to_idx[tuple(p)] for p in dense]
        assert _traces_real_edges(new_mesh.faces, idx)

    def test_target_below_current_count_is_ignored(self, capsys):
        mesh, ring = self._hole_rim()
        new_mesh, dense = _densify_boundary(mesh, ring, len(ring) - 1)
        assert dense == list(ring)
        assert len(new_mesh.faces) == len(mesh.faces)
        assert "more than the target" in capsys.readouterr().out


# ===========================================================================
# stitching._assign_rings_to_ends
# ===========================================================================


class TestAssignRingsToEnds:
    def _rings(self):
        return [_ring_coords(8, 1.0, 0.0), _ring_coords(8, 1.0, 10.0)]

    def test_picks_nearest_pairing(self):
        prox, dist, leftover = _assign_rings_to_ends(
            self._rings(), (0.0, 0.0, -2.0), (0.0, 0.0, 12.0)
        )
        assert np.mean([p[2] for p in prox]) == pytest.approx(0.0)
        assert np.mean([p[2] for p in dist]) == pytest.approx(10.0)
        assert leftover == []

    def test_swaps_when_centroids_swap(self):
        prox, _, _ = _assign_rings_to_ends(
            self._rings(), (0.0, 0.0, 12.0), (0.0, 0.0, -2.0)
        )
        assert np.mean([p[2] for p in prox]) == pytest.approx(10.0)

    def test_reports_leftover_rings(self):
        rings = self._rings() + [_ring_coords(8, 1.0, 5.0)]
        _, _, leftover = _assign_rings_to_ends(
            rings, (0.0, 0.0, -2.0), (0.0, 0.0, 12.0)
        )
        assert leftover == [2]

    def test_assigns_whole_rings(self):
        """A ring is handed over intact, never split between the two ends."""
        rings = self._rings()
        prox, dist, _ = _assign_rings_to_ends(rings, (0.0, 0.0, -2.0), (0.0, 0.0, 12.0))
        assert set(prox) in (set(rings[0]), set(rings[1]))
        assert set(dist) in (set(rings[0]), set(rings[1]))
        assert set(prox).isdisjoint(set(dist))


# ===========================================================================
# stitching._toward_aorta / _shift_plane_clear_of
# ===========================================================================


class TestTowardAorta:
    def test_uses_aorta_centroid_not_vessel_axis(self):
        """The vessel axis is unusable for an intramural course: the coronary runs
        inside the aortic wall, so the lumen sits roughly perpendicular to it."""
        ring_centroid = np.zeros(3)
        aorta = [(-8.0, a, b) for a in (-3, 0, 3) for b in (-3, 0, 3)]  # aorta at -x
        vessel_axis = np.array([0.0, 1.0, 0.0])  # perpendicular to that

        direction, source = _toward_aorta(ring_centroid, aorta, vessel_axis)
        assert direction[0] < 0, "must point at the aorta, not along the axis"
        assert "aorta_points" in source

    def test_falls_back_to_vessel_axis(self):
        direction, source = _toward_aorta(np.zeros(3), [], np.array([0.0, 1.0, 0.0]))
        np.testing.assert_allclose(direction, [0.0, 1.0, 0.0])
        assert "vessel axis" in source

    def test_none_when_nothing_available(self):
        direction, _ = _toward_aorta(np.zeros(3), [], None)
        assert direction is None


class TestShiftPlaneClearOf:
    def _straddling_points(self):
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        return np.array(
            [[0.3 * np.cos(a), 0.3 * np.sin(a), z] for a in angles for z in (-0.5, 0.5)]
        )

    def test_shifts_until_all_points_are_behind(self):
        pts = self._straddling_points()
        origin = np.zeros(3)
        normal = np.array([0.0, 0.0, 1.0])
        overshoot = 0.5

        new_origin, new_normal, moved = _shift_plane_clear_of(
            origin, normal, pts, normal, overshoot
        )
        assert moved == pytest.approx(1.0)
        assert ((pts - new_origin) @ new_normal).max() <= -overshoot + 1e-9

    def test_orients_normal_along_outward(self):
        pts = self._straddling_points()
        _, new_normal, _ = _shift_plane_clear_of(
            np.zeros(3),
            np.array([0.0, 0.0, -1.0]),  # points the wrong way
            pts,
            np.array([0.0, 0.0, 1.0]),
            0.5,
        )
        assert new_normal[2] > 0

    def test_no_move_when_already_clear(self):
        pts = self._straddling_points()
        origin = np.zeros(3)
        normal = np.array([0.0, 0.0, 1.0])
        new_origin, new_normal, _ = _shift_plane_clear_of(
            origin, normal, pts, normal, 0.5
        )
        _, _, moved_again = _shift_plane_clear_of(
            new_origin, new_normal, pts, normal, 0.5
        )
        assert moved_again == 0.0


# ===========================================================================
# Per-ring boundary storage
# ===========================================================================


class TestBoundaryRingStorage:
    def _results(self):
        mesh, removed, _ = _make_grid_with_hole()
        return {
            "mesh": mesh,
            "anomalous_points": [tuple(mesh.vertices[i]) for i in removed],
            "aorta_points": [],
        }

    def test_stores_per_ring_and_flat_keys(self):
        updated = remove_labeled_points_from_mesh(
            self._results(), "anomalous_points", target_boundaries=1
        )
        assert "boundary_points_1" in updated
        assert updated["boundary_points"] == updated["boundary_points_1"]

    def test_flat_key_is_the_concatenation(self):
        mesh, removed, _ = _make_grid_with_hole(remove=[(2, 2), (6, 6)])
        results = {
            "mesh": mesh,
            "anomalous_points": [tuple(mesh.vertices[i]) for i in removed],
        }
        updated = remove_labeled_points_from_mesh(
            results, "anomalous_points", target_boundaries=2
        )
        rings = [updated["boundary_points_1"], updated["boundary_points_2"]]
        assert updated["boundary_points"] == rings[0] + rings[1]

    def test_stale_per_ring_keys_are_cleared(self):
        """A later trim producing fewer rings must not leave a phantom ring."""
        results = self._results()
        results["boundary_points_2"] = [(99.0, 99.0, 99.0)]
        results["boundary_points_3"] = [(98.0, 98.0, 98.0)]
        updated = remove_labeled_points_from_mesh(
            results, "anomalous_points", target_boundaries=1
        )
        assert "boundary_points_2" not in updated
        assert "boundary_points_3" not in updated

    def test_keep_labeled_points_also_stores_rings(self):
        # Keep a contiguous block, so surviving faces (and therefore a rim) exist.
        n = 9
        mesh, _, _ = _make_grid_with_hole(n)
        keep = [
            tuple(mesh.vertices[j * n + i]) for j in range(2, 7) for i in range(2, 7)
        ]
        updated = keep_labeled_points_from_mesh({"mesh": mesh, "k": keep}, "k")
        assert len(updated["mesh"].faces) > 0
        assert "boundary_points_1" in updated
        assert updated["boundary_points"] == updated["boundary_points_1"]

    def test_sync_results_to_mesh_remaps_per_ring_keys(self):
        updated = remove_labeled_points_from_mesh(
            self._results(), "anomalous_points", target_boundaries=1
        )
        old_mesh = updated["mesh"]
        moved = old_mesh.copy()
        moved.vertices = moved.vertices + np.array([0.0, 0.0, 5.0])

        synced = sync_results_to_mesh(updated, old_mesh, moved)
        assert len(synced["boundary_points_1"]) == len(updated["boundary_points_1"])
        assert all(
            p[2] == pytest.approx(q[2] + 5.0)
            for p, q in zip(synced["boundary_points_1"], updated["boundary_points_1"])
        )
