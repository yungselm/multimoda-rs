"""Replay a HolOrama fusion case up to the stitch and show the stitch part by part.

Reads a case folder holding the CCTA mesh, its HolOrama sidecar
(``*_fusion_*.json``: labels, deleted branches, alignment, scaling) and the
pullback CSVs, reruns the pipeline headless, stitches, and opens a pyvista
window with one toggle per layer:

* CCTA mesh after removal (and after the side branches were cut)
* intravascular wall (with the side-branch holes)
* proximal / distal patches, side-branch bridges
* the final stitched mesh, and its open edges in red
* contour 0 of every side branch in the tree, the IV centerline

Every layer is also written to ``<case>/stitch_debug/`` (``.ply``, colours
kept), together with ``stitched.stl`` and ``summary.txt``.

The replay uses HolOrama's default GUI values (prepare centerlines, labelling,
tree, pullback) and ``align_manual`` with the saved rotation and centerline
point, so it is close to, not exactly, the GUI's ``align_combined`` result.

Usage (WSL, from the repo root)::

    .venv/bin/python examples/visualize_stitching.py
    .venv/bin/python examples/visualize_stitching.py --remove anomalous_points proximal_points
    .venv/bin/python examples/visualize_stitching.py --no-tree --no-show
    .venv/bin/python examples/visualize_stitching.py --branch-cut rca:3:2 --distal-cut-slice 40
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import time

import numpy as np
import pyvista as pv
import trimesh

import multimodars as mm
from multimodars.ccta.stitching import core as stitching_core
from multimodars.ccta.stitching import open_boundary_edges

DEFAULT_CASE = "/mnt/c/Users/ansel/OneDrive/arcomo_test"

# HolOrama GUI defaults.
PREPARE = {"spacing_mm": 0.5, "branch_spacing_tolerance": 2.0, "smooth_sigma": 1.5}
RM_START_MM = 5.0
LABEL = {
    "acute_takeoff_rca": False,
    "acute_takeoff_lca": False,
    "range_mm_takeoff_rca": 45,
    "range_mm_takeoff_lca": 45,
    "step_size_mm": 0.5,
    "bounding_sphere_radius_mm_rca": 3.5,
    "bounding_sphere_radius_mm_lca": 3.5,
}
STITCH = {
    "prox_start_mode": "highest_z",
    "dist_start_mode": "nearest_iv",
    "clamp_overshoot": 0.0,
    "fillet_bulge": 0.5,
    "fillet_layers": 5,
    "seam_points_a": 2,
    "seam_points_b": 4,
}

# name: (colour, opacity, shown at start)
LAYERS = {
    "ccta": ("lightgray", 1.0, True),
    "iv_wall": ("steelblue", 1.0, True),
    "prox_patch": ("limegreen", 1.0, True),
    "dist_patch": ("orange", 1.0, True),
    "bridges": ("magenta", 1.0, True),
    "stitched": ("wheat", 1.0, False),
    "open_edges": ("red", 1.0, True),
    "tree_contour0": ("yellow", 1.0, True),
    "iv_centerline": ("cyan", 1.0, False),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def find_case_files(case: str) -> tuple[str, dict]:
    sidecars = sorted(glob.glob(os.path.join(case, "*_fusion_*.json")))
    if not sidecars:
        raise FileNotFoundError(f"no *_fusion_*.json in {case}")
    with open(sidecars[-1], encoding="utf-8") as f:
        state = json.load(f)
    mesh = sidecars[-1].split("_fusion_")[0] + ".stl"
    if not os.path.exists(mesh):
        raise FileNotFoundError(mesh)
    return mesh, state


def replay(case: str, sync_tree: bool) -> dict:
    """Everything the stitch needs, scaled, before any removal."""
    mesh_path, state = find_case_files(case)
    log(f"case: {mesh_path}")

    def load_cl(name: str):
        return mm.load_centerline(os.path.join(case, f"{name}_cl.vtp"), name)

    ao = mm.prepare_centerline(load_cl("ao"), rm_start_mm=0.0, **PREPARE)
    cls = {
        v: mm.prepare_centerline(
            load_cl(v), ref_centerline=ao, rm_start_mm=RM_START_MM, **PREPARE
        )
        for v in ("rca", "lca")
    }
    log("label_geometry")
    results = mm.label_geometry(
        path_ccta_geometry=trimesh.load_mesh(mesh_path),
        centerline_aorta=ao,
        centerline_rca=cls["rca"],
        centerline_lca=cls["lca"],
        control_plot=False,
        **LABEL,
    )
    vertices = np.asarray(results["mesh"].vertices)
    if len(vertices) != state["n_vertices"]:
        raise ValueError(
            f"mesh has {len(vertices)} vertices, the sidecar {state['n_vertices']}"
        )

    def points(indices) -> list[tuple]:
        return [
            tuple(v) for v in vertices[np.asarray(indices, dtype=np.int64)].tolist()
        ]

    # Saved labels are indices into the labelled mesh.
    for step in ("label_geometry", "label_branches"):
        for key, value in state["labels"][step].items():
            results[key] = (
                {int(b): points(i) for b, i in value.items() if i}
                if isinstance(value, dict)
                else points(value)
            )

    log("discretize_vessel_tree")
    tree = mm.discretize_vessel_tree(
        ao,
        cls["rca"],
        cls["lca"],
        results,
        step_size=1.0,
        n_points=100,
        control_plot=False,
    )
    # HolOrama deletes after discretizing and prunes the tree.
    for d in state.get("deleted_branches", []):
        results, cls[d["vessel"]] = mm.remove_branch(
            results, cls[d["vessel"]], d["branch"], fill_hole=d["fill_hole"]
        )
        branches = getattr(tree, d["vessel"])
        for b in d.get("removed", [d["branch"]]):
            branches.pop(b, None)
        setattr(tree, d["vessel"], branches)
    tree.calculate_ref_pts()

    present = {tuple(v) for v in results["mesh"].vertices.tolist()}
    for key, value in state["labels"]["label_anomalous"].items():
        results[key] = [p for p in points(value) if p in present]

    log("pullback")
    plan = state["pullback_plan"]
    csv_dir = os.path.join(case, os.path.basename(plan["input_path"].rstrip("/\\")))
    inp = mm.numpy_to_inputdata(
        lumen_arr=np.genfromtxt(
            os.path.join(csv_dir, "tagged_contours.csv"), delimiter="\t"
        ),
        reference_points=np.genfromtxt(
            os.path.join(csv_dir, "tagged_reference_points.csv"), delimiter="\t"
        ),
        record=None,
        diastole=True,
        label=plan["label"],
    )
    geometry, _ = mm.from_array_single(
        input_data=inp,
        step_rotation_deg=plan.get("step_rotation_deg", 0.1),
        sample_size=plan.get("sample_size", 200),
        image_center=(5.0, 5.0),
        n_points=plan.get("n_points", 20),
        output_path=os.path.join(case, "stitch_debug", "pullback"),
        watertight=False,
        write_obj=False,
        smooth=True,
    )
    alignment = state["alignment"]
    vessel = alignment["vessel"]
    branch_cl = cls[vessel].get_branch(
        alignment["branch"], with_parents=alignment.get("with_parents", False)
    )
    aligned, _, _ = mm.align_manual(
        branch_cl,
        geometry,
        alignment["rotation_deg"],
        tuple(alignment["centerline_point"]),
        write=False,
        watertight=False,
        align_wall_anomalous=False,
    )
    aligned = getattr(aligned, "geom_a", aligned)

    log("scaling")
    scaling = state["scaling"]["applied"]
    opposite = "rca" if vessel == "lca" else "lca"

    def scale(results: dict, region: list, cl, adjustment: float) -> dict:
        if not region or adjustment == 0.0:
            return results
        old = results["mesh"]
        new = mm.scale_region_centerline_morphing(
            mesh=old,
            region_points=region,
            centerline=cl,
            radius_adjustment_mm=adjustment,
        )
        if sync_tree:
            mm.sync_tree_to_mesh(tree, old, new)
        return mm.sync_results_to_mesh(results, old, new)

    results = scale(
        results, results["distal_points"], cls[vessel], scaling["distal_scaling"]
    )
    results = scale(
        results,
        results["aorta_points"] + results[f"{vessel}_removed_points"],
        ao,
        scaling["aortic_scaling"],
    )
    results = scale(
        results, results["proximal_points"], cls[vessel], scaling["proximal_scaling"]
    )
    remaining = scaling.get("remaining_vessels_scaling", 0.0)
    for key, cl in (
        (f"{opposite}_points", cls[opposite]),
        ("off_path_points", cls[vessel]),
    ):
        results = scale(results, results.get(key, []), cl, remaining)

    return {
        "results": results,
        "aligned": aligned,
        "tree": tree,
        "centerlines": cls,
        "vessel": vessel,
    }


# ---------------------------------------------------------------------------
# Stitch, capturing the parts
# ---------------------------------------------------------------------------


def stitch_with_parts(
    case_data: dict, remove: list[str], use_tree: bool, cuts: dict
) -> dict:
    """Run the removal and the stitch, capturing the parts the stitch merges.

    Wraps three of the stitch module's internals for the duration of the call:
    the ring stitcher (proximal and distal patches), the side-branch
    reconnection (cut CCTA mesh, IV wall with holes, bridges) and the IV mesh
    builder (IV wall when there is no side-branch step).
    """
    removed = mm.remove_labeled_points_from_mesh(
        case_data["results"], remove, target_boundaries=2
    )
    parts: dict = {"bridges": [], "ring_patches": []}
    real_stitch_rings = stitching_core._stitch_rings
    real_reconnect = stitching_core.reconnect_side_branches
    real_to_trimesh = stitching_core.geometry_to_trimesh

    def spy_to_trimesh(*args, **kwargs):
        parts["iv_wall"] = real_to_trimesh(*args, **kwargs)
        return parts["iv_wall"]

    def spy_stitch_rings(*args, **kwargs):
        patch = real_stitch_rings(*args, **kwargs)
        parts["ring_patches"].append(patch)
        return patch

    def spy_reconnect(ccta, iv, *args, **kwargs):
        parts["ring_patches"] = parts["ring_patches"][:2]  # prox, dist so far
        out = real_reconnect(ccta, iv, *args, **kwargs)
        parts["ccta"], parts["iv_wall"], parts["bridges"] = out
        return out

    stitching_core._stitch_rings = spy_stitch_rings
    stitching_core.reconnect_side_branches = spy_reconnect
    stitching_core.geometry_to_trimesh = spy_to_trimesh
    try:
        stitched = mm.stitch_ccta_to_intravascular(
            case_data["aligned"],
            removed["mesh"],
            copy.deepcopy(removed),
            tree=case_data["tree"] if use_tree else None,
            centerlines=case_data["centerlines"],
            **cuts,
            **STITCH,
        )
    finally:
        stitching_core._stitch_rings = real_stitch_rings
        stitching_core.reconnect_side_branches = real_reconnect
        stitching_core.geometry_to_trimesh = real_to_trimesh

    prox, dist = parts["ring_patches"][:2]
    parts.setdefault("ccta", removed["mesh"])  # no tree: no branch was cut
    parts.update(prox_patch=prox, dist_patch=dist, stitched=stitched["mesh"])
    return parts


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


def to_pv(mesh: trimesh.Trimesh) -> pv.PolyData:
    return pv.PolyData.from_regular_faces(
        np.asarray(mesh.vertices), np.asarray(mesh.faces)
    )


def lines(polylines: list[np.ndarray], closed: bool = True) -> pv.PolyData:
    points, cells, offset = [], [], 0
    for line in polylines:
        n = len(line)
        idx = list(range(offset, offset + n)) + ([offset] if closed else [])
        cells += [len(idx), *idx]
        points.append(line)
        offset += n
    if not points:
        return pv.PolyData()
    poly = pv.PolyData(np.vstack(points))
    poly.lines = np.asarray(cells)
    return poly


def open_edge_lines(mesh: trimesh.Trimesh) -> pv.PolyData:
    edges = open_boundary_edges(np.asarray(mesh.faces))
    if len(edges) == 0:
        return pv.PolyData()
    used, local = np.unique(edges, return_inverse=True)
    poly = pv.PolyData(np.asarray(mesh.vertices)[used])
    poly.lines = np.column_stack([np.full(len(edges), 2), local.reshape(-1, 2)]).ravel()
    return poly


def build_layers(parts: dict, case_data: dict) -> dict[str, pv.PolyData]:
    layers: dict[str, pv.PolyData] = {}
    for name in ("ccta", "iv_wall", "prox_patch", "dist_patch", "stitched"):
        if parts.get(name) is not None:
            layers[name] = to_pv(parts[name])
    if parts["bridges"]:
        layers["bridges"] = to_pv(trimesh.util.concatenate(parts["bridges"]))

    stitched = parts["stitched"].copy()
    stitched.merge_vertices()
    layers["open_edges"] = open_edge_lines(stitched)

    tree, vessel = case_data["tree"], case_data["vessel"]
    layers["tree_contour0"] = lines(
        [
            np.asarray(b.contours[0].points_as_tuples())
            for k, b in getattr(tree, vessel).items()
            if k != 0 and b.contours
        ]
    )
    centroids = np.array([f.centroid for f in case_data["aligned"].frames])
    layers["iv_centerline"] = lines([centroids], closed=False)
    return layers


def summary(parts: dict) -> str:
    stitched = parts["stitched"].copy()
    stitched.merge_vertices()
    bodies = stitched.split(only_watertight=False)
    edges = open_boundary_edges(np.asarray(stitched.faces))
    out = [
        f"watertight: {stitched.is_watertight}",
        f"bodies: {[len(b.vertices) for b in bodies]}",
        f"open edges: {len(edges)}",
        f"side-branch bridges: {len(parts['bridges'])}",
    ]
    if len(edges):
        rings = mm.ccta.stitching.order_boundary_rings(
            stitched.faces, stitched.vertices
        )
        for ring in rings:
            p = stitched.vertices[ring]
            out.append(
                f"  open ring of {len(ring)} edges at {np.round(p.mean(axis=0), 1)}"
                f" (span {np.round(np.ptp(p, axis=0), 1)})"
            )
    return "\n".join(out)


def save(layers: dict[str, pv.PolyData], parts: dict, out_dir: str, text: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, poly in layers.items():
        if poly.n_points:
            colour = np.array(pv.Color(LAYERS[name][0]).int_rgb, dtype=np.uint8)
            poly = poly.copy()
            poly.point_data["RGB"] = np.tile(colour, (poly.n_points, 1))
            poly.save(os.path.join(out_dir, f"{name}.ply"), texture="RGB")
    parts["stitched"].export(os.path.join(out_dir, "stitched.stl"))
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


def show(layers: dict[str, pv.PolyData], title: str) -> None:
    plotter = pv.Plotter(title=title)
    actors = {}
    for name, poly in layers.items():
        if not poly.n_points:
            continue
        colour, opacity, visible = LAYERS[name]
        if name in ("open_edges", "tree_contour0", "iv_centerline"):
            actor = plotter.add_mesh(
                poly,
                color=colour,
                line_width=5 if name == "open_edges" else 3,
                render_lines_as_tubes=True,
            )
        else:
            actor = plotter.add_mesh(
                poly,
                color=colour,
                opacity=opacity,
                show_edges=name != "ccta",
                edge_color="dimgray",
                line_width=0.5,
                smooth_shading=False,
            )
        actor.SetVisibility(visible)
        actors[name] = actor

    size, gap = 26, 6
    for i, (name, actor) in enumerate(actors.items()):
        y = 10 + i * (size + gap)

        def toggle(state: bool, actor=actor) -> None:
            actor.SetVisibility(state)

        plotter.add_checkbox_button_widget(
            toggle,
            value=bool(actor.GetVisibility()),
            position=(10, y),
            size=size,
            color_on=LAYERS[name][0],
            color_off="gray",
            border_size=2,
        )
        plotter.add_text(name, position=(10 + size + 8, y + 4), font_size=10)

    # Start at the IV segment, where the stitching happens.
    focus = layers.get("iv_wall") or layers.get("stitched")
    plotter.show_axes()
    plotter.camera.focal_point = focus.center
    plotter.reset_camera(bounds=focus.bounds)
    plotter.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("case", nargs="?", default=DEFAULT_CASE)
    parser.add_argument(
        "--remove",
        nargs="+",
        default=["anomalous_points"],
        help="results keys to remove before stitching (HolOrama's checkboxes)",
    )
    parser.add_argument(
        "--no-tree", action="store_true", help="stitch without side branches"
    )
    parser.add_argument(
        "--no-sync-tree",
        action="store_true",
        help="do not move the tree with the scaling (the old behaviour)",
    )
    parser.add_argument(
        "--branch-cut",
        action="append",
        default=[],
        metavar="VESSEL:BRANCH:SLICE",
        help="contour a side branch is cut at, e.g. rca:3:2 (repeatable; default 0)",
    )
    parser.add_argument(
        "--distal-cut-slice",
        type=int,
        help="contour the distal rim is cut flat at (default: nearest the opening)",
    )
    parser.add_argument(
        "--proximal-cut-slice",
        type=int,
        help="same for a proximal rim that is not the ostium",
    )
    parser.add_argument("--out", help="output folder (default <case>/stitch_debug)")
    parser.add_argument("--no-show", action="store_true", help="only write the files")
    args = parser.parse_args()

    case_data = replay(args.case, sync_tree=not args.no_sync_tree)
    log(f"removing {args.remove} and stitching")
    branch_cut_slices = {}
    for spec in args.branch_cut:
        vessel, branch, slice_idx = spec.split(":")
        branch_cut_slices[(vessel.lower(), int(branch))] = int(slice_idx)
    cuts = {
        "branch_cut_slices": branch_cut_slices,
        "distal_cut_slice": args.distal_cut_slice,
        "proximal_cut_slice": args.proximal_cut_slice,
    }
    parts = stitch_with_parts(
        case_data, args.remove, use_tree=not args.no_tree, cuts=cuts
    )
    layers = build_layers(parts, case_data)
    text = summary(parts)
    print(text)

    out_dir = args.out or os.path.join(args.case, "stitch_debug")
    save(layers, parts, out_dir, text)
    log(f"written to {out_dir}")
    if not args.no_show:
        show(layers, f"stitch: removed {', '.join(args.remove)}")


if __name__ == "__main__":
    main()
