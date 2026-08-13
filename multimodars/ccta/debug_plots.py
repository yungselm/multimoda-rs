from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, cast

import numpy as np
import trimesh
from trimesh.points import PointCloud
from trimesh.visual import ColorVisuals

from .boundary import open_boundary_edges, order_boundary_rings

if TYPE_CHECKING:
    from ..multimodars import PyCenterline, PyDiscretizedVesselTree


def _make_point_cloud(arr: np.ndarray, color: list[int]) -> PointCloud:
    c = np.array(color, dtype=np.uint8)
    return PointCloud(arr, colors=np.tile(c, (len(arr), 1)))


def _get_cl_arry(cl: PyCenterline) -> np.ndarray:
    return np.array(
        [(p.contour_point.x, p.contour_point.y, p.contour_point.z) for p in cl.points],
        dtype=np.float64,
    )


def _group_by_branch(cl: PyCenterline) -> dict[int, list]:
    by_branch: dict[int, list] = defaultdict(list)
    for p in cl.points:
        by_branch[p.branch_id].append(p.contour_point)
    return by_branch


def plot_results_key(
    results: dict,
    aorta_points: bool = True,
    rca_points: bool = False,
    lca_points: bool = False,
    rca_removed_points: bool = False,
    proximal_points: bool = False,
    distal_points: bool = False,
    anomalous_points: bool = False,
    cl_rca: PyCenterline | None = None,
    cl_lca: PyCenterline | None = None,
    cl_aorta: PyCenterline | None = None,
):
    """Open an interactive 3-D scene visualising selected regions from a results dict.

    Toggle which regions are shown by passing ``True``/``False`` for each flag.

    Colour coding:

    * **Yellow** - aortic points.
    * **Blue** - RCA coronary points.
    * **Green** - LCA coronary points.
    * **Red** - removed / reassigned points (RCA + LCA combined).
    * **Cyan** - proximal points.
    * **Magenta** - distal points.
    * **Orange** - anomalous (intramural) points.

    Parameters
    ----------
    results : dict
        Dictionary with keys ``"aorta_points"``, ``"rca_points"``,
        ``"lca_points"``, ``"rca_removed_points"``, ``"proximal_points"``,
        ``"distal_points"``, ``"anomalous_points"`` - each a list of
        ``(x, y, z)`` tuples.
    aorta_points : bool
        Show aortic points (yellow).
    rca_points : bool
        Show RCA coronary points (blue).
    lca_points : bool
        Show LCA coronary points (green).
    rca_removed_points : bool
        Show removed RCA points (red).
    proximal_points : bool
        Show proximal points (cyan).
    distal_points : bool
        Show distal points (magenta).
    anomalous_points : bool
        Show anomalous points (orange).
    """
    print("\n=== RESULTS KEY PLOT ===")

    region_config = [
        ("aorta_points", aorta_points, [255, 255, 0, 255], "Yellow  = Aorta"),
        ("rca_points", rca_points, [0, 0, 255, 255], "Blue    = RCA"),
        ("lca_points", lca_points, [0, 255, 0, 255], "Green   = LCA"),
        (
            "rca_removed_points",
            rca_removed_points,
            [255, 0, 0, 255],
            "Red     = Removed",
        ),
        ("proximal_points", proximal_points, [0, 255, 255, 255], "Cyan    = Proximal"),
        ("distal_points", distal_points, [255, 0, 255, 255], "Magenta = Distal"),
        (
            "anomalous_points",
            anomalous_points,
            [255, 165, 0, 255],
            "Orange  = Anomalous",
        ),
    ]

    scene_geoms = []
    for key, enabled, color, label in region_config:
        pts = results.get(key, [])
        print(
            f"  {label:30s}  n={len(pts):6d}  {'[shown]' if enabled and pts else '[hidden]'}"
        )
        if enabled and pts:
            arr = np.array(pts, dtype=np.float64)
            scene_geoms.append(_make_point_cloud(arr, color))

    if not scene_geoms:
        print("Nothing to show - all regions are disabled or empty.")
        return

    mesh_visual = results["mesh"]
    mesh_visual.visual.face_colors = [200, 200, 200, 100]
    scene_geoms.append(mesh_visual)

    if cl_rca:
        scene_geoms.append(_make_point_cloud(_get_cl_arry(cl_rca), [0, 100, 200, 255]))
    if cl_lca:
        scene_geoms.append(_make_point_cloud(_get_cl_arry(cl_lca), [0, 150, 0, 255]))
    if cl_aorta:
        scene_geoms.append(
            _make_point_cloud(_get_cl_arry(cl_aorta), [200, 200, 0, 255])
        )

    trimesh.Scene(scene_geoms).show()


def compare_centerline_scaling(
    original_mesh: trimesh.Trimesh,
    scaled_mesh: trimesh.Trimesh,
    region_points: list,
    centerline=None,
):
    """Open three interactive scenes comparing the original and scaled meshes.

    * **Scene 1** - original mesh with the scaled region highlighted in red and
      the centerline in cyan.
    * **Scene 2** - scaled mesh with the same overlays.
    * **Scene 3** - side-by-side view: original (blue-ish) shifted 150 mm along
      the x-axis next to the scaled mesh (orange-ish).

    Parameters
    ----------
    original_mesh : trimesh.Trimesh
        Mesh before scaling.
    scaled_mesh : trimesh.Trimesh
        Mesh after scaling (e.g. from
        :func:`~multimodars.ccta.adjust_ccta.scale_region_centerline_morphing`).
    region_points : list of tuple
        ``(x, y, z)`` coordinates of the scaled region, highlighted in red.
    centerline : PyCenterline, optional
        Centerline overlaid in cyan; skipped when ``None``.
    """
    print("\n=== CENTERLINE SCALING COMPARISON ===")

    region_cloud = _make_point_cloud(
        np.array(region_points, dtype=np.float64), [255, 0, 0, 255]
    )

    cast(ColorVisuals, original_mesh.visual).face_colors = [200, 200, 200, 100]
    scene1 = trimesh.Scene([original_mesh, region_cloud])

    cl_arr: np.ndarray | None = None
    if centerline is not None:
        cl_arr = _get_cl_arry(centerline)
        scene1.add_geometry(_make_point_cloud(cl_arr, [0, 255, 255, 255]))

    print("Showing Scene 1: Original mesh with RCA region (red) and centerline (cyan)")
    scene1.show()

    cast(ColorVisuals, scaled_mesh.visual).face_colors = [200, 200, 200, 100]
    scene2 = trimesh.Scene([scaled_mesh, region_cloud])
    if cl_arr is not None:
        scene2.add_geometry(_make_point_cloud(cl_arr, [0, 255, 255, 255]))
    print("Showing Scene 2: Scaled mesh with RCA region (red) and centerline (cyan)")
    scene2.show()

    scaled_mesh_shifted = scaled_mesh.copy()
    shift_amount = np.array([150, 0, 0])
    scaled_mesh_shifted.apply_translation(shift_amount)

    cast(ColorVisuals, original_mesh.visual).face_colors = [0, 100, 200, 100]
    cast(ColorVisuals, scaled_mesh_shifted.visual).face_colors = [200, 100, 0, 100]

    scene3 = trimesh.Scene([original_mesh, scaled_mesh_shifted])
    if cl_arr is not None:
        scene3.add_geometry(
            _make_point_cloud(cl_arr + shift_amount, [0, 255, 255, 255])
        )

    print("Showing Scene 3: Side-by-side comparison (Blue=Original, Orange=Scaled)")
    scene3.show()


def plot_vessel_tree(
    tree: PyDiscretizedVesselTree,
    pts_per_contour: int = 24,
) -> None:
    """Open an interactive trimesh scene of a discretized vessel tree.

    Colour coding
    -------------
    * Silver         — aorta contour points
    * Steel-blue     — RCA main
    * Shades of blue — RCA side branches
    * Coral          — LCA main
    * Shades of orange — LCA side branches
    * Yellow         — contour centroids
    * Red            — main reference point
    * Orange         — counter-clockwise reference
    * Magenta        — clockwise reference

    Parameters
    ----------
    tree:
        Fully populated vessel tree (output of
        :func:`~multimodars.ccta.discretization_map.discretize_vessel_tree`).
    pts_per_contour:
        Number of evenly-spaced points sampled from each contour ring.
    """
    _RCA_BRANCH_COLORS = [
        [79, 163, 224, 255],
        [126, 200, 227, 255],
        [168, 216, 234, 255],
        [184, 223, 237, 255],
    ]
    _LCA_BRANCH_COLORS = [
        [224, 127, 79, 255],
        [227, 168, 126, 255],
        [234, 192, 168, 255],
        [237, 208, 184, 255],
    ]

    scene_geoms = []

    def _add_contours(contours, color: list[int]) -> None:
        rx: list[float] = []
        ry: list[float] = []
        rz: list[float] = []
        cx: list[float] = []
        cy: list[float] = []
        cz: list[float] = []
        for c in contours:
            pts = c.points
            m = len(pts)
            if m == 0:
                continue
            step = max(1, m // pts_per_contour)
            s = pts[::step]
            rx.extend(p.x for p in s)
            ry.extend(p.y for p in s)
            rz.extend(p.z for p in s)
            if c.centroid:
                cx.append(c.centroid[0])
                cy.append(c.centroid[1])
                cz.append(c.centroid[2])
        if rx:
            arr = np.array([rx, ry, rz], dtype=np.float64).T
            scene_geoms.append(_make_point_cloud(arr, color))
        if cx:
            arr = np.array([cx, cy, cz], dtype=np.float64).T
            scene_geoms.append(_make_point_cloud(arr, [255, 255, 0, 255]))

    def _add_refs(refs) -> None:
        mx, my, mz = [], [], []
        ccx, ccy, ccz = [], [], []
        clx, cly, clz = [], [], []
        for main_ref, cc_ref, clock_ref in refs:
            mx.append(main_ref[0])
            my.append(main_ref[1])
            mz.append(main_ref[2])
            ccx.append(cc_ref[0])
            ccy.append(cc_ref[1])
            ccz.append(cc_ref[2])
            clx.append(clock_ref[0])
            cly.append(clock_ref[1])
            clz.append(clock_ref[2])
        for xs, ys, zs, color in [
            (mx, my, mz, [255, 0, 0, 255]),
            (ccx, ccy, ccz, [255, 165, 0, 255]),
            (clx, cly, clz, [255, 0, 255, 255]),
        ]:
            if xs:
                arr = np.array([xs, ys, zs], dtype=np.float64).T
                scene_geoms.append(_make_point_cloud(arr, color))

    _add_contours(tree.discretized_aorta, [192, 192, 192, 100])
    _add_contours(tree.discretized_rca_main, [70, 130, 180, 255])
    for i, branch in enumerate(tree.rca_branches):
        _add_contours(branch, _RCA_BRANCH_COLORS[i % len(_RCA_BRANCH_COLORS)])
    _add_contours(tree.discretized_lca_main, [255, 127, 80, 255])
    for i, branch in enumerate(tree.lca_branches):
        _add_contours(branch, _LCA_BRANCH_COLORS[i % len(_LCA_BRANCH_COLORS)])
    _add_refs(tree.rca_references)
    _add_refs(tree.lca_references)

    trimesh.Scene(scene_geoms).show()


def plot_centerline_branches(
    rca_cl: PyCenterline,
    lca_cl: PyCenterline,
    results_dict: dict | None = None,
) -> None:
    """Open an interactive trimesh scene of centerline branch assignments.

    Colour coding
    -------------
    * RCA main — steel-blue; side branches in lighter blues.
    * LCA main — red; side branches in oranges/pinks.
    * Surface mesh points — same palette, semi-transparent.

    Parameters
    ----------
    rca_cl, lca_cl:
        Centerlines after ``calculate_branches`` and ``orient_by_max_z``
        (output of :func:`~multimodars.ccta.discretization_map.label_branches`).
    results_dict:
        Optional.  When provided, labelled surface points
        (``rca_points_main``, ``rca_points_side_N``, ``lca_points_main``,
        ``lca_points_side_N``) are overlaid semi-transparently.
    """
    _RCA_COLORS = [
        [31, 119, 180, 255],
        [23, 190, 207, 255],
        [148, 103, 189, 255],
        [44, 160, 44, 255],
        [127, 127, 127, 255],
    ]
    _LCA_COLORS = [
        [214, 39, 40, 255],
        [255, 127, 14, 255],
        [227, 119, 194, 255],
        [188, 189, 34, 255],
        [140, 86, 75, 255],
    ]

    scene_geoms = []

    def _add_cl_branches(cl: PyCenterline, colors: list[list[int]]) -> None:
        by_branch = _group_by_branch(cl)
        for bid in sorted(by_branch):
            pts = by_branch[bid]
            arr = np.array([(p.x, p.y, p.z) for p in pts], dtype=np.float64)
            scene_geoms.append(_make_point_cloud(arr, colors[bid % len(colors)]))

    def _add_surface_points(key: str, color: list[int]) -> None:
        pts = results_dict.get(key, [])  # type: ignore[union-attr]
        if not pts:
            return
        arr = np.array(pts, dtype=np.float64)
        scene_geoms.append(_make_point_cloud(arr, color[:3] + [80]))

    _add_cl_branches(rca_cl, _RCA_COLORS)
    _add_cl_branches(lca_cl, _LCA_COLORS)

    if results_dict is not None:
        _add_surface_points("rca_points_main", _RCA_COLORS[0])
        i = 1
        while f"rca_points_side_{i}" in results_dict:
            _add_surface_points(
                f"rca_points_side_{i}", _RCA_COLORS[i % len(_RCA_COLORS)]
            )
            i += 1
        _add_surface_points("lca_points_main", _LCA_COLORS[0])
        i = 1
        while f"lca_points_side_{i}" in results_dict:
            _add_surface_points(
                f"lca_points_side_{i}", _LCA_COLORS[i % len(_LCA_COLORS)]
            )
            i += 1

    trimesh.Scene(scene_geoms).show()


def plot_centerline_edges(
    cl: PyCenterline,
    cos_threshold: float = 0.0,
) -> None:
    """Open an interactive trimesh scene of a centerline with sharp angles highlighted.

    Each branch is shown as a coloured point cloud.  Positions flagged by
    ``find_sharp_angles`` are overlaid as red points so they are easy to spot
    and decide whether a ``split_branch`` call is needed.

    Parameters
    ----------
    cl:
        Centerline after ``calculate_branches`` (and optionally
        ``orient_by_max_z``).
    cos_threshold:
        Cosine threshold forwarded to ``cl.find_sharp_angles``.  ``0.0`` flags
        all angles ≥ 90 °; negative values flag increasingly obtuse bends.
    """
    _PALETTE = [
        [31, 119, 180, 255],
        [255, 127, 14, 255],
        [44, 160, 44, 255],
        [214, 39, 40, 255],
        [148, 103, 189, 255],
        [140, 86, 75, 255],
        [227, 119, 194, 255],
        [127, 127, 127, 255],
        [188, 189, 34, 255],
        [23, 190, 207, 255],
    ]

    scene_geoms = []
    by_branch = _group_by_branch(cl)

    for bid in sorted(by_branch):
        pts = by_branch[bid]
        arr = np.array([(p.x, p.y, p.z) for p in pts], dtype=np.float64)
        scene_geoms.append(_make_point_cloud(arr, _PALETTE[bid % len(_PALETTE)]))

        # find_sharp_angles returns global point_index values, not positions
        # local to `pts` - offset by the branch's own start to index into it.
        branch_start = (
            cl.branch_start_indices[bid] if bid < len(cl.branch_start_indices) else 0
        )
        sharp_pos = cl.find_sharp_angles(bid, cos_threshold)
        if sharp_pos:
            sharp_pts = [
                pts[i - branch_start]
                for i in sharp_pos
                if 0 <= i - branch_start < len(pts)
            ]
            if sharp_pts:
                sharp_arr = np.array(
                    [(p.x, p.y, p.z) for p in sharp_pts], dtype=np.float64
                )
                scene_geoms.append(_make_point_cloud(sharp_arr, [255, 0, 0, 255]))

    trimesh.Scene(scene_geoms).show()


def plot_sharp_angles(
    cl: PyCenterline,
    branch_id: int,
    sharp_positions: list[int],
    context_pts: int = 3,
) -> None:
    """Open an interactive trimesh scene highlighting each sharp-angle position.

    The full centerline is shown dimmed in gray. Each flagged position is
    overlaid in a distinct colour together with *context_pts* neighbours on
    each side so individual angles are easy to count and locate.

    Colour coding
    -------------
    * Gray (dim) — all centerline points (background).
    * Distinct colours per index — each sharp-angle position and its local context.

    Parameters
    ----------
    cl:
        Centerline after ``calculate_branches``.
    branch_id:
        Branch that was inspected (0 = main vessel).
    sharp_positions:
        Global point_index values as returned by :func:`find_sharp_angles` or
        ``cl.find_sharp_angles``.
    context_pts:
        Number of neighbouring points shown on each side of each position.
    """
    _PALETTE = [
        [255, 127, 14, 255],
        [44, 160, 44, 255],
        [148, 103, 189, 255],
        [23, 190, 207, 255],
        [188, 189, 34, 255],
        [227, 119, 194, 255],
        [140, 86, 75, 255],
        [214, 39, 40, 255],
        [31, 119, 180, 255],
        [255, 215, 0, 255],
    ]

    by_branch = _group_by_branch(cl)
    scene_geoms = []

    for pts in by_branch.values():
        arr = np.array([(p.x, p.y, p.z) for p in pts], dtype=np.float64)
        scene_geoms.append(_make_point_cloud(arr, [150, 150, 150, 80]))

    branch_pts = by_branch.get(branch_id, [])
    n = len(branch_pts)
    branch_start = (
        cl.branch_start_indices[branch_id]
        if branch_id < len(cl.branch_start_indices)
        else 0
    )
    for i, point_index in enumerate(sharp_positions):
        pos = point_index - branch_start
        lo = max(0, pos - context_pts)
        hi = min(n, pos + context_pts + 1)
        segment = branch_pts[lo:hi]
        if not segment:
            continue
        arr = np.array([(p.x, p.y, p.z) for p in segment], dtype=np.float64)
        scene_geoms.append(_make_point_cloud(arr, _PALETTE[i % len(_PALETTE)]))

    trimesh.Scene(scene_geoms).show()


def plot_boundary_edges(
    results: dict,
    key: str = "boundary_points",
    show_mesh: bool = True,
    target_boundaries: int | None = None,
) -> None:
    """Open an interactive trimesh scene of a stored boundary ring set and its edges.

    Re-derives the rings from the mesh's open edges with
    :func:`~multimodars.ccta.boundary.order_boundary_rings`, draws each ring's
    edges as connected line segments, and colours the ring vertices red -> blue
    by their walk order.  This makes both the seam direction and any split into
    multiple disconnected rings immediately visible.

    Colour coding
    -------------
    * Mesh - light gray, semi-transparent.
    * Ring vertices - red (ring start) -> blue (ring end) gradient.
    * Ring edges - one distinct colour per connected ring.

    Parameters
    ----------
    results : dict
        Must contain ``"mesh"`` and *key*.
    key : str
        Which stored point list to treat as the boundary.  Defaults to
        ``"boundary_points"``; pass ``"prox_boundary_points"`` or
        ``"dist_boundary_points"`` to inspect the stitching seams instead.
    show_mesh : bool
        Include the mesh in the scene for context.
    target_boundaries : int, optional
        Leave as ``None`` (the default) to draw every ring the mesh actually
        has - a debug view should not merge rings behind your back.  Set an
        integer only to preview what reducing to that many rings would look
        like.
    """
    print("\n=== BOUNDARY EDGES PLOT ===")

    mesh: trimesh.Trimesh = results["mesh"]
    pts = results.get(key, [])
    if not pts:
        print(f"Nothing to show - results['{key}'] is empty or missing.")
        return

    # Map the stored boundary coordinates back to their mesh vertex indices.
    coord_to_idx = {tuple(v): i for i, v in enumerate(mesh.vertices)}
    boundary_indices = {
        idx for pt in pts if (idx := coord_to_idx.get(tuple(pt))) is not None
    }
    missing = len(pts) - len(boundary_indices)
    if missing:
        print(f"  {missing} boundary point(s) not found on the mesh - skipped.")
    if not boundary_indices:
        print("Nothing to show - no boundary point matched a mesh vertex.")
        return

    rings = order_boundary_rings(
        mesh.faces,
        mesh.vertices,
        boundary_indices,
        target_n=target_boundaries,
    )
    print(
        f"  {len(boundary_indices)} points in {len(rings)} ring(s): "
        f"{[len(r) for r in rings]}"
    )

    # Every open boundary edge, so a ring is only drawn closed when its two ends
    # really are joined on the mesh.
    edge_set = {frozenset((int(a), int(b))) for a, b in open_boundary_edges(mesh.faces)}

    scene_geoms: list = []
    if show_mesh:
        mesh_visual = mesh.copy()
        cast(ColorVisuals, mesh_visual.visual).face_colors = [200, 200, 200, 100]
        scene_geoms.append(mesh_visual)

    _RING_COLORS = [
        [255, 127, 14, 255],
        [44, 160, 44, 255],
        [148, 103, 189, 255],
        [140, 86, 75, 255],
    ]
    for r, ring in enumerate(rings):
        coords = mesh.vertices[ring]
        n = len(ring)

        # Red (ring start) -> blue (ring end) gradient so walk order is visible.
        t = (np.arange(n) / max(1, n - 1))[:, None]
        colors = np.hstack(
            [
                (255 * (1 - t)).astype(np.uint8),  # R
                np.zeros((n, 1), dtype=np.uint8),  # G
                (255 * t).astype(np.uint8),  # B
                np.full((n, 1), 255, dtype=np.uint8),  # A
            ]
        )
        scene_geoms.append(PointCloud(coords, colors=colors))

        # Edges: connect consecutive ring vertices.  Close the loop only when the
        # ring's ends are a real boundary edge, so an open arc gets no false edge.
        segs = np.stack([coords[:-1], coords[1:]], axis=1)
        closed = n > 2 and frozenset((ring[0], ring[-1])) in edge_set
        if closed:
            segs = np.concatenate([segs, coords[[n - 1, 0]][None]], axis=0)
        if len(segs):
            path = trimesh.load_path(segs)
            path.colors = np.tile(
                _RING_COLORS[r % len(_RING_COLORS)], (len(path.entities), 1)
            )
            scene_geoms.append(path)

    trimesh.Scene(scene_geoms).show()
