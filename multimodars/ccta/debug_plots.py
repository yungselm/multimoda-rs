from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import trimesh
from trimesh.points import PointCloud
from trimesh.visual import ColorVisuals

if TYPE_CHECKING:
    from ..multimodars import PyCenterline, PyDiscretizedVesselTree


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
            colors = np.tile(color, (len(pts), 1))
            scene_geoms.append(PointCloud(arr, colors=colors))

    if not scene_geoms:
        print("Nothing to show - all regions are disabled or empty.")
        return

    mesh_visual = results["mesh"]
    mesh_visual.visual.face_colors = [200, 200, 200, 100]
    scene_geoms.append(mesh_visual)

    if cl_rca:
        rca = _get_cl_arry(cl_rca)
        scene_geoms.append(PointCloud(rca, colors=[0, 100, 200, 255]))
    if cl_lca:
        lca = _get_cl_arry(cl_lca)
        scene_geoms.append(PointCloud(lca, colors=[0, 150, 0, 255]))
    if cl_aorta:
        ao = _get_cl_arry(cl_aorta)
        scene_geoms.append(PointCloud(ao, colors=[200, 200, 0, 255]))

    scene = trimesh.Scene(scene_geoms)
    scene.show()


def _get_cl_arry(cl: PyCenterline) -> np.ndarray:
    arr = np.array(
        [(p.contour_point.x, p.contour_point.y, p.contour_point.z) for p in cl.points],
        dtype=np.float64,
    )
    return arr


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

    region_array = np.array(region_points, dtype=np.float64)
    region_colors = np.tile([255, 0, 0, 255], (len(region_points), 1))  # Red

    # Scene 1: Original mesh with region highlighted
    scene1 = trimesh.Scene(
        [original_mesh, PointCloud(region_array, colors=region_colors)]
    )
    cast(ColorVisuals, original_mesh.visual).face_colors = [
        200,
        200,
        200,
        100,
    ]  # Semi-transparent

    centerline_points = None
    centerline_colors = None
    if centerline is not None:
        centerline_points = np.array(
            [
                (p.contour_point.x, p.contour_point.y, p.contour_point.z)
                for p in centerline.points
            ],
            dtype=np.float64,
        )
        centerline_colors = np.tile(
            [0, 255, 255, 255], (len(centerline_points), 1)
        )  # Cyan
        scene1.add_geometry(PointCloud(centerline_points, colors=centerline_colors))

    print("Showing Scene 1: Original mesh with RCA region (red) and centerline (cyan)")
    scene1.show()

    # Scene 2: Scaled mesh with region highlighted
    scene2 = trimesh.Scene(
        [scaled_mesh, PointCloud(region_array, colors=region_colors)]
    )
    cast(ColorVisuals, scaled_mesh.visual).face_colors = [
        200,
        200,
        200,
        100,
    ]  # Semi-transparent

    if centerline_points is not None:
        scene2.add_geometry(PointCloud(centerline_points, colors=centerline_colors))

    print("Showing Scene 2: Scaled mesh with RCA region (red) and centerline (cyan)")
    scene2.show()

    # Scene 3: Side-by-side comparison
    scaled_mesh_shifted = scaled_mesh.copy()
    shift_amount = np.array([150, 0, 0])  # Adjust based on your mesh size
    scaled_mesh_shifted.apply_translation(shift_amount)

    scene3 = trimesh.Scene([original_mesh, scaled_mesh_shifted])

    cast(ColorVisuals, original_mesh.visual).face_colors = [
        0,
        100,
        200,
        100,
    ]  # Blue-ish
    cast(ColorVisuals, scaled_mesh_shifted.visual).face_colors = [
        200,
        100,
        0,
        100,
    ]  # Orange-ish

    if centerline_points is not None:
        centerline_shifted = centerline_points + shift_amount
        scene3.add_geometry(PointCloud(centerline_shifted, colors=centerline_colors))

    print("Showing Scene 3: Side-by-side comparison (Blue=Original, Orange=Scaled)")
    scene3.show()


def plot_vessel_tree(
    tree: PyDiscretizedVesselTree,
    title: str = "Discretized Vessel Tree",
    pts_per_contour: int = 24,
) -> None:
    """Show an interactive 3-D Plotly visualisation of a discretized vessel tree.

    Each vessel segment is rendered as a stack of contour rings (subsampled to
    ``pts_per_contour`` points for performance) plus centroid dots that trace
    the vessel centre-line.  Orientation reference triplets are shown as
    distinct marker symbols.

    Colour coding
    -------------
    * Silver         — aorta
    * Steel-blue     — RCA main
    * Shades of blue — RCA side branches
    * Coral          — LCA main
    * Shades of orange — LCA side branches
    * Yellow dots    — contour centroids
    * Red ×          — main reference point
    * Orange ▲       — counter-clockwise reference
    * Magenta ▼      — clockwise reference

    Parameters
    ----------
    tree:
        Fully populated vessel tree (output of
        :func:`~multimodars.ccta.discretization_map.discretize_vessel_tree`).
    title:
        Figure title.
    pts_per_contour:
        Number of evenly-spaced points sampled from each contour ring.
        Lower values improve rendering speed; higher values show more detail.
    """
    import plotly.graph_objects as go  # lazy import — plotly is optional

    _RCA_BRANCH_COLORS = ["#4fa3e0", "#7ec8e3", "#a8d8ea", "#b8dfed"]
    _LCA_BRANCH_COLORS = ["#e07f4f", "#e3a87e", "#eac0a8", "#edd0b8"]

    traces: list[go.BaseTraceType] = []

    # ── helpers ──────────────────────────────────────────────────────────────

    def _ring_xyz(contour, n: int):
        """Subsample a contour ring and close it; return (xs, ys, zs) lists."""
        pts = contour.points
        m = len(pts)
        if m == 0:
            return [], [], []
        step = max(1, m // n)
        sampled = pts[::step]
        xs = [p.x for p in sampled] + [sampled[0].x]
        ys = [p.y for p in sampled] + [sampled[0].y]
        zs = [p.z for p in sampled] + [sampled[0].z]
        return xs, ys, zs

    def _add_contours(contours, color: str, name: str, opacity: float = 0.7) -> None:
        """One NaN-separated line trace for all rings + one marker trace for centroids."""
        rx, ry, rz = [], [], []
        cx, cy, cz = [], [], []
        for c in contours:
            xs, ys, zs = _ring_xyz(c, pts_per_contour)
            if not xs:
                continue
            rx += xs + [None]
            ry += ys + [None]
            rz += zs + [None]
            if c.centroid:
                cx.append(c.centroid[0])
                cy.append(c.centroid[1])
                cz.append(c.centroid[2])

        if rx:
            traces.append(
                go.Scatter3d(
                    x=rx,
                    y=ry,
                    z=rz,
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=1),
                    opacity=opacity,
                )
            )
        if cx:
            traces.append(
                go.Scatter3d(
                    x=cx,
                    y=cy,
                    z=cz,
                    mode="markers",
                    name=f"{name} centroids",
                    marker=dict(color="yellow", size=2),
                    showlegend=False,
                )
            )

    def _add_refs(refs, label: str) -> None:
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
        if not mx:
            return
        traces.append(
            go.Scatter3d(
                x=mx,
                y=my,
                z=mz,
                mode="markers",
                name=f"{label} main ref",
                marker=dict(color="red", symbol="x", size=5),
            )
        )
        traces.append(
            go.Scatter3d(
                x=ccx,
                y=ccy,
                z=ccz,
                mode="markers",
                name=f"{label} CCW ref",
                marker=dict(color="orange", symbol="diamond", size=5),
            )
        )
        traces.append(
            go.Scatter3d(
                x=clx,
                y=cly,
                z=clz,
                mode="markers",
                name=f"{label} CW ref",
                marker=dict(color="magenta", symbol="square", size=5),
            )
        )

    # ── build traces ─────────────────────────────────────────────────────────

    _add_contours(tree.discretized_aorta, "silver", "Aorta", opacity=0.4)
    _add_contours(tree.discretized_rca_main, "steelblue", "RCA main")
    for i, branch in enumerate(tree.rca_branches):
        _add_contours(
            branch,
            _RCA_BRANCH_COLORS[i % len(_RCA_BRANCH_COLORS)],
            f"RCA branch {i + 1}",
        )
    _add_contours(tree.discretized_lca_main, "coral", "LCA main")
    for i, branch in enumerate(tree.lca_branches):
        _add_contours(
            branch,
            _LCA_BRANCH_COLORS[i % len(_LCA_BRANCH_COLORS)],
            f"LCA branch {i + 1}",
        )
    _add_refs(tree.rca_references, "RCA")
    _add_refs(tree.lca_references, "LCA")

    # ── layout ───────────────────────────────────────────────────────────────

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
    )
    fig.show()


def plot_centerline_branches(
    rca_cl: PyCenterline,
    lca_cl: PyCenterline,
    results_dict: dict | None = None,
    title: str = "Centerline Branch Assignment",
) -> None:
    """Show an interactive 3-D Plotly visualisation of centerline branch assignments.

    Each branch of the RCA and LCA is drawn as a separate trace so it gets its
    own legend entry and can be toggled on/off.  Optionally overlays the
    labelled surface-mesh points from *results_dict* so you can verify that
    branch labels transferred correctly to the geometry.

    Colour coding
    -------------
    * RCA branch 0 (main) — steel-blue; side branches in lighter blues.
    * LCA branch 0 (main) — coral; side branches in lighter oranges.
    * Surface mesh points — same palette, smaller markers.

    Parameters
    ----------
    rca_cl, lca_cl:
        Centerlines after ``calculate_branches`` and ``check_centerline``
        (output of :func:`~multimodars.ccta.discretization_map.prepare_centerlines`).
    results_dict:
        Optional.  When provided, labelled surface points
        (``rca_points_main``, ``rca_points_side_N``, ``lca_points_main``,
        ``lca_points_side_N``) are overlaid as small scatter markers.
    title:
        Figure title.
    """
    import plotly.graph_objects as go  # lazy import — plotly is optional

    # distinct hues per branch so each is visually separable
    _RCA_COLORS = ["#1f77b4", "#17becf", "#9467bd", "#2ca02c", "#7f7f7f"]
    _LCA_COLORS = ["#d62728", "#ff7f0e", "#e377c2", "#bcbd22", "#8c564b"]

    traces: list[go.BaseTraceType] = []

    def _add_cl_branches(cl: PyCenterline, colors: list[str], vessel: str) -> None:
        from collections import defaultdict

        by_branch: dict[int, list] = defaultdict(list)
        for p in cl.points:
            by_branch[p.branch_id].append(p.contour_point)
        for bid in sorted(by_branch):
            pts = by_branch[bid]
            label = f"{vessel} branch {bid}" if bid > 0 else f"{vessel} main"
            color = colors[bid % len(colors)]
            traces.append(
                go.Scatter3d(
                    x=[p.x for p in pts],
                    y=[p.y for p in pts],
                    z=[p.z for p in pts],
                    mode="markers",
                    name=label,
                    marker=dict(color=color, size=3),
                )
            )

    def _add_surface_points(
        results_dict: dict, key: str, color: str, label: str
    ) -> None:
        pts = results_dict.get(key, [])
        if not pts:
            return
        xs, ys, zs = zip(*pts)
        traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                name=label,
                marker=dict(color=color, size=1.5, opacity=0.4),
            )
        )

    _add_cl_branches(rca_cl, _RCA_COLORS, "RCA")
    _add_cl_branches(lca_cl, _LCA_COLORS, "LCA")

    if results_dict is not None:
        _add_surface_points(
            results_dict, "rca_points_main", _RCA_COLORS[0], "RCA main pts"
        )
        i = 1
        while f"rca_points_side_{i}" in results_dict:
            _add_surface_points(
                results_dict,
                f"rca_points_side_{i}",
                _RCA_COLORS[i % len(_RCA_COLORS)],
                f"RCA side {i} pts",
            )
            i += 1
        _add_surface_points(
            results_dict, "lca_points_main", _LCA_COLORS[0], "LCA main pts"
        )
        i = 1
        while f"lca_points_side_{i}" in results_dict:
            _add_surface_points(
                results_dict,
                f"lca_points_side_{i}",
                _LCA_COLORS[i % len(_LCA_COLORS)],
                f"LCA side {i} pts",
            )
            i += 1

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
    )
    fig.show()


def plot_centerline_edges(
    cl: PyCenterline,
    cos_threshold: float = 0.0,
    title: str = "Centerline Edges",
) -> None:
    """Show a centerline coloured by branch with sharp-angle positions highlighted.

    Each branch is drawn as a connected line + point cloud using a distinct colour
    from a high-contrast qualitative palette.  Positions returned by
    ``find_sharp_angles`` for every branch are overlaid as large red × markers so
    they are easy to spot and decide whether a ``split_branch`` call is needed.

    Parameters
    ----------
    cl:
        Centerline after ``calculate_branches`` (and optionally
        ``check_centerline``).
    cos_threshold:
        Cosine threshold forwarded to ``cl.find_sharp_angles``.  Angles whose
        cosine exceeds this value are *not* flagged.  ``0.0`` flags all angles
        ≥ 90 °; negative values flag increasingly obtuse bends.
    title:
        Figure title.
    """
    import plotly.graph_objects as go
    from collections import defaultdict

    # Plotly default qualitative palette — 10 maximally distinct hues
    _PALETTE = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    traces: list[go.BaseTraceType] = []

    by_branch: dict[int, list] = defaultdict(list)
    for p in cl.points:
        by_branch[p.branch_id].append(p.contour_point)

    for bid in sorted(by_branch):
        pts = by_branch[bid]
        color = _PALETTE[bid % len(_PALETTE)]
        label = "main" if bid == 0 else f"branch {bid}"

        traces.append(
            go.Scatter3d(
                x=[p.x for p in pts],
                y=[p.y for p in pts],
                z=[p.z for p in pts],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(color=color, size=3),
            )
        )

        sharp_pos = cl.find_sharp_angles(bid, cos_threshold)
        if sharp_pos:
            sharp_pts = [pts[i] for i in sharp_pos if i < len(pts)]
            traces.append(
                go.Scatter3d(
                    x=[p.x for p in sharp_pts],
                    y=[p.y for p in sharp_pts],
                    z=[p.z for p in sharp_pts],
                    mode="markers",
                    name=f"{label} sharp angles",
                    marker=dict(color="red", size=8, symbol="x"),
                )
            )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
    )
    fig.show()
