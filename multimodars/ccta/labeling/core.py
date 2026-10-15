from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from ...io.read_geometrical import read_mesh
from ...multimodars import (
    PyCenterline,
    clean_outlier_points,
    final_reclassification,
    find_aortic_points,
    find_centerline_bounded_points,
    find_points_by_cl_region,
    keep_largest_connected_component,
    remove_occluded_points,
)
from .._results import branches_key, vessel_branches, vessel_of_centerline
from ..debug_plots import plot_results_key


def label_geometry(
    path_ccta_geometry: Path | str | trimesh.Trimesh,
    centerline_aorta: PyCenterline,
    centerline_rca: PyCenterline,
    centerline_lca: PyCenterline,
    acute_takeoff_rca: bool = False,
    acute_takeoff_lca: bool = False,
    range_mm_takeoff_rca: float = 60.0,
    range_mm_takeoff_lca: float = 60.0,
    step_size_mm: float = 1.0,
    bounding_sphere_radius_mm_rca: float = 3.0,
    bounding_sphere_radius_mm_lca: float = 3.0,
    tolerance_float: float = 1e-6,
    control_plot: bool = True,
) -> dict:
    """Label CCTA mesh vertices as aorta, RCA, or LCA using centerline-based region detection.

    Loads a 3-D surface mesh and three centerlines (aorta, RCA, LCA), then assigns
    each mesh vertex to one of the anatomical regions. For vessels with an acute
    takeoff angle from the aorta (e.g. an anomalous intramural course, or simply
    an acute ostium where the coronary and aortic walls overlap) an additional
    occlusion-removal step uses ray-triangle intersection to strip the overlapping
    points, followed by adjacency-map reclassification to clean up isolated
    mis-labelled vertices. Herfore, a ray is cast from every aorta point to the
    centerline points of the coronary near the ostium and if 3 faces are
    intersected by the ray the points from the first face must correspond to the
    overlapping/intramural section.

    Parameters
    ----------
    path_ccta_geometry : Path or str
        Path to the CCTA surface mesh file (any format supported by
        :func:`multimodars.io.read_geometrical.read_mesh`).
    centerline_aorta : PyCenterline
        Aortic centerline. Used as-is - load and prepare it beforehand (branch
        extraction, trimming, resampling, orientation, smoothing), e.g. with
        :func:`~multimodars.ccta.centerline_prep.load_centerline` followed by
        :func:`~multimodars.ccta.centerline_prep.prepare_centerline`.
    centerline_rca : PyCenterline
        RCA centerline, prepared the same way as *centerline_aorta*.
    centerline_lca : PyCenterline
        LCA centerline, prepared the same way as *centerline_aorta*.
    acute_takeoff_rca : bool, optional
        When ``True`` applies ray-triangle occlusion removal to the RCA region
        to strip points overlapping the aortic wall near an acute-angle takeoff
        (e.g. anomalous/intramural courses).  Default is ``False``.
    acute_takeoff_lca : bool, optional
        When ``True`` applies ray-triangle occlusion removal to the LCA region,
        same as *acute_takeoff_rca* but for the LCA.  Default is ``False``.
    range_mm_takeoff_rca : float, optional
        Arc-length in mm along the RCA centerline, from its proximal end,
        examined during occlusion removal (the overlapping/intramural segment
        length). Expressed as a physical length rather than a point count so
        it stays correct regardless of centerline resampling density.
        Default is ``60.0``.
    range_mm_takeoff_lca : float, optional
        Same as *range_mm_takeoff_rca* but for the LCA. Default is ``60.0``.
    step_size_mm : float, optional
        Step size in mm for iterating over coronary centerline points during
        occlusion removal.  Default is ``1.0`` mm.
    bounding_sphere_radius_mm_rca : float, optional
        Radius in millimetres of the rolling sphere used to collect candidate
        mesh vertices around each centerline point of the rca.  Default is ``3.0``.
    bounding_sphere_radius_mm_lca : float, optional
        Radius in millimetres of the rolling sphere used to collect candidate
        mesh vertices around each centerline point of the lca.  Default is ``3.0``.
    tolerance_float : float, optional
        Distance tolerance used when matching mesh vertices to points during
        face lookup.  Default is ``1e-6``.
    control_plot : bool, optional
        When ``True`` opens an interactive 3-D scene showing the labelled mesh
        after processing.  Default is ``True``.

    Returns
    -------
    results : dict
        Dictionary with keys:

        * ``"mesh"`` - the original :class:`trimesh.Trimesh` object.
        * ``"aorta_points"`` - list of ``(x, y, z)`` tuples for aortic vertices.
        * ``"rca_points"`` - list of ``(x, y, z)`` tuples for RCA vertices.
        * ``"lca_points"`` - list of ``(x, y, z)`` tuples for LCA vertices.
        * ``"rca_removed_points"`` - RCA vertices removed by occlusion detection.
        * ``"lca_removed_points"`` - LCA vertices removed by occlusion detection.

    Raises
    ------
    Exception
        Re-raises any error that occurs while reading the mesh or centerline
        files, after printing a descriptive message.
    """
    if isinstance(path_ccta_geometry, trimesh.Trimesh):
        mesh = path_ccta_geometry
        print(
            f"Using provided mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
        )
    else:
        try:
            mesh = read_mesh(path_ccta_geometry)
            print(
                f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
            )
        except Exception as e:
            print(f"Error reading CCTA mesh from {path_ccta_geometry}: {e}")
            raise

    points_list = [tuple(vertex) for vertex in mesh.vertices.tolist()]
    mesh_faces_list = mesh.faces.tolist()

    # Rust implementation using a rolling sphere with fixed radius
    rca_points_found = find_centerline_bounded_points(
        centerline=centerline_rca,
        points=points_list,
        radius=bounding_sphere_radius_mm_rca,
    )
    lca_points_found = find_centerline_bounded_points(
        centerline=centerline_lca,
        points=points_list,
        radius=bounding_sphere_radius_mm_lca,
    )

    print(f"\nRCA points found: {len(rca_points_found)}")
    print(f"LCA points found: {len(lca_points_found)}")

    # With an acute take-off the coronary runs along the aortic wall, so its
    # bounding spheres also pick up wall vertices: remove those hidden from the
    # centerlines.
    rca_removed_points: list[tuple[float, float, float]] = []
    lca_removed_points: list[tuple[float, float, float]] = []
    final_rca_points_found, final_lca_points_found = rca_points_found, lca_points_found
    if acute_takeoff_rca:
        final_rca_points_found, rca_removed_points = remove_occluded_points(
            points_list,
            mesh_faces_list,
            rca_points_found,
            centerline_rca,
            centerline_aorta,
            range_mm_takeoff_rca,
            step_size_mm=step_size_mm,
            tol=tolerance_float,
        )
        print(f"RCA: relabeled {len(rca_removed_points)} points in intramural course")
    if acute_takeoff_lca:
        final_lca_points_found, lca_removed_points = remove_occluded_points(
            points_list,
            mesh_faces_list,
            lca_points_found,
            centerline_lca,
            centerline_aorta,
            range_mm_takeoff_lca,
            step_size_mm=step_size_mm,
            tol=tolerance_float,
        )
        print(f"LCA: relabeled {len(lca_removed_points)} points in intramural course")

    print("\nRemoving LCA and RCA island points...")
    aortic_points = find_aortic_points(
        points_list, final_rca_points_found, final_lca_points_found
    )
    print(f"length before: {len(final_lca_points_found + final_rca_points_found)}")
    final_lca_points, final_aortic_points = clean_outlier_points(
        final_lca_points_found, aortic_points, 2.0, 0.4
    )  # based on patient data, only precleaning anyways, rest done by final_reclassification
    final_rca_points, _ = clean_outlier_points(
        final_rca_points_found, final_aortic_points, 2.0, 0.4
    )
    final_aortic_points = find_aortic_points(
        points_list, final_rca_points, final_lca_points
    )
    # add also the rca_removed points and lca_removed points to aortic points
    final_aortic_points = list(
        set(final_aortic_points) | set(rca_removed_points) | set(lca_removed_points)
    )
    print(f"length after: {len(final_lca_points + final_rca_points)}")

    results: dict[str, Any] = {
        "mesh": mesh,
        "aorta_points": final_aortic_points,
        "rca_points": final_rca_points,
        "lca_points": final_lca_points,
        "rca_removed_points": rca_removed_points,
        "lca_removed_points": lca_removed_points,
    }

    # final reclassification based on adjacency map
    print("\nApplying final reclassification based on adjacency map...")
    aorta_pts, rca_pts, lca_pts, rca_removed_pts, lca_removed_pts = (
        final_reclassification(
            points_list,
            mesh_faces_list,
            results["rca_points"],
            results["lca_points"],
            results["rca_removed_points"],
            results["lca_removed_points"],
        )
    )
    new_results: dict[str, Any] = {
        "mesh": mesh,
        "aorta_points": aorta_pts,
        "rca_points": rca_pts,
        "lca_points": lca_pts,
        "rca_removed_points": rca_removed_pts,
        "lca_removed_points": lca_removed_pts,
    }
    print(f"aorta_points:{len(new_results['aorta_points'])}")
    print(f"rca_points:{len(new_results['rca_points'])}")
    print(f"lca_points:{len(new_results['lca_points'])}")
    print(f"rca_removed_points:{len(new_results['rca_removed_points'])}")
    print(f"lca_removed_points:{len(new_results['lca_removed_points'])}")

    if control_plot:
        plot_results_key(
            new_results,
            aorta_points=True,
            rca_points=True,
            lca_points=True,
            rca_removed_points=True,
            proximal_points=True,
            distal_points=False,
            anomalous_points=False,
            cl_rca=centerline_rca,
            cl_lca=centerline_lca,
            cl_aorta=centerline_aorta,
        )

    return new_results


def label_anomalous_region(
    centerline,
    frames,
    results: dict,
    branch_id: int = 0,
    debug_plot: bool = False,
) -> dict:
    """Split the pullback branch into proximal, anomalous and distal regions.

    Uses the intravascular imaging frames, aligned to branch *branch_id*, to find
    where along that branch the anomalous (intramural) segment begins and ends.
    Only the pullback path is split: branch *branch_id* and its parent branches,
    each up to the junction the path leaves it at (the path
    ``centerline.get_branch(branch_id, with_parents=True)`` follows).  Every
    other vertex of the vessel (its other branches, and parent vessels beyond
    the junction) goes to ``"off_path_points"`` and keeps its branch label.

    The vessel (RCA or LCA) is the one *centerline* belongs to, and its vertices
    per branch come from :func:`label_branches`.

    Parameters
    ----------
    centerline : PyCenterline
        Centerline of the whole coronary vessel, with all its branches, the one
        the branches were labelled with.
    frames : list of PyFrame
        Intravascular imaging frames, aligned onto branch *branch_id* (possibly
        continuing into its parent vessels).
    results : dict
        Labelled results with ``"mesh"`` and the per-branch labels
        (``"rca_branches"`` / ``"lca_branches"``).
    branch_id : int, optional
        Branch the frames are aligned to. Default ``0``, the main vessel.
    debug_plot : bool, optional
        When ``True`` opens an interactive visualisation of the three
        sub-regions.  Default is ``False``.

    Returns
    -------
    dict
        The input *results* dictionary extended with:

        * ``"proximal_points"`` - path vertices proximal to the anomalous segment.
        * ``"distal_points"`` - path vertices distal to the anomalous segment.
        * ``"anomalous_points"`` - path vertices within the anomalous segment.
        * ``"off_path_points"`` - the vessel's vertices off the pullback path.

        Island points dropped from the three path regions are moved from the
        vessel's labels to ``"aorta_points"``. ``"aorta_points"`` stays disjoint
        from ``"rca_removed_points"`` and ``"lca_removed_points"``, as returned
        by :func:`label_geometry`.

    Raises
    ------
    ValueError
        If the branches were not labelled, *branch_id* is not a labelled
        branch, or the labels hold branches *centerline* does not have.
    """
    vessel = vessel_of_centerline(results, centerline)
    branches = vessel_branches(results, vessel)
    if branch_id not in branches:
        raise ValueError(
            f"branch {branch_id} is not labelled for {vessel.upper()} "
            f"(branches {sorted(branches)})"
        )
    unknown = sorted(set(branches) - set(centerline.branch_ids()))
    if unknown:
        raise ValueError(
            f"{vessel.upper()} branches {unknown} are not in the centerline. Pass the "
            "centerline the branches were labelled with, not get_branch(...)"
        )

    (
        proximal_points_raw,
        distal_points_raw,
        anomalous_points_raw,
        off_path_points,
    ) = find_points_by_cl_region(centerline, branch_id, frames, branches)

    # find_points_by_cl_region goes by distance alone, so a region can include
    # points not mesh-connected to its main cluster: keep the largest component.
    mesh = results["mesh"]
    vertices = [tuple(v) for v in mesh.vertices.tolist()]
    faces = mesh.faces.tolist()
    proximal_points = keep_largest_connected_component(
        vertices, faces, proximal_points_raw
    )
    distal_points = keep_largest_connected_component(vertices, faces, distal_points_raw)
    anomalous_points = keep_largest_connected_component(
        vertices, faces, anomalous_points_raw
    )

    # Island points dropped above are still nominally part of the vessel's
    # labels, which would otherwise keep them out of the aorta_points complement
    # below, stranding them in no region at all, so no scaling step ever touches
    # them.  Reassign them fully to aorta by removing them from the vessel's
    # labels too, mirroring how final_reclassification's Logic A reassigns
    # isolated vertices to aorta rather than leaving them ambiguous.
    dropped_points = (
        (set(proximal_points_raw) - set(proximal_points))
        | (set(distal_points_raw) - set(distal_points))
        | (set(anomalous_points_raw) - set(anomalous_points))
    )
    points_key = f"{vessel}_points"
    if dropped_points:
        results[points_key] = [
            p for p in results[points_key] if p not in dropped_points
        ]
        results[branches_key(vessel)] = {
            k: [p for p in pts if p not in dropped_points]
            for k, pts in branches.items()
        }
        print(
            f"  {len(dropped_points)} island point(s) reassigned from "
            f"'{points_key}' sub-regions to aorta_points"
        )

    results["proximal_points"] = proximal_points
    results["distal_points"] = distal_points
    results["anomalous_points"] = anomalous_points
    results["off_path_points"] = off_path_points

    # The occlusion-removed points are excluded too: label_geometry returns them
    # as their own classes, disjoint from aorta_points, and callers that want the
    # whole aortic wall add them explicitly (aorta_points + rca_removed_points).
    non_aorta = (
        set(results.get("rca_points", []))
        | set(results.get("lca_points", []))
        | set(results.get("rca_removed_points", []))
        | set(results.get("lca_removed_points", []))
        | set(proximal_points)
        | set(distal_points)
        | set(anomalous_points)
    )
    results["aorta_points"] = [
        tuple(v) for v in results["mesh"].vertices if tuple(v) not in non_aorta
    ]

    print(
        f"\nApplying anomalous labeling on {vessel.upper()} branch {branch_id} "
        "based on aligned intravascular frames..."
    )
    print(f"proximal_points: {len(results['proximal_points'])}")
    print(f"distal_points: {len(results['distal_points'])}")
    print(f"anomalous_points: {len(results['anomalous_points'])}")
    print(f"off_path_points: {len(results['off_path_points'])}")

    if debug_plot:
        plot_results_key(
            results=results,
            aorta_points=False,
            rca_points=False,
            lca_points=False,
            rca_removed_points=False,
            proximal_points=True,
            distal_points=True,
            anomalous_points=True,
            cl_rca=centerline,
            cl_lca=None,
            cl_aorta=None,
        )

    return results


def label_branches(
    centerline,
    results: dict,
    vessel: str | None = None,
    bounding_sphere_radius_mm: float = 3.0,
) -> dict:
    """Split a coronary's vertices into one label per centerline branch.

    Every vertex of ``results["{vessel}_points"]`` is assigned to exactly one
    branch of *centerline*.  A rolling sphere along each branch collects its
    candidates.  A vertex several branches collect goes to the one closest to the
    main vessel in the branch tree (a parent before its children, so the parent
    keeps its wall at a junction), then to the nearest of those.  A vertex no
    sphere reaches goes to the nearest branch.

    Parameters
    ----------
    centerline : PyCenterline
        Centerline of the whole coronary, with its branches.
    results : dict
        Labelled results dictionary (e.g. from :func:`label_geometry`).
    vessel : {"rca", "lca"}, optional
        Vessel *centerline* belongs to. By default the one whose vertices lie
        nearest to it.
    bounding_sphere_radius_mm : float, optional
        Radius of the rolling sphere used to collect candidate vertices.
        Default is ``3.0``.

    Returns
    -------
    dict
        *results* with ``"{vessel}_branches"``: a dict from every branch id of
        *centerline* (``centerline.branch_ids()``, 0 being the main vessel) to
        the list of its vertices.  The lists partition
        ``results["{vessel}_points"]``.
    """
    if vessel is None:
        vessel = vessel_of_centerline(results, centerline)
    points = results[f"{vessel}_points"]
    ids = centerline.branch_ids()
    parents = centerline.branch_parents()

    def depth(k: int) -> int:
        d = 0
        while (p := parents[k]) is not None:
            d, k = d + 1, p
        return d

    claims: dict[tuple, list[int]] = {}
    for k in ids:
        found = find_centerline_bounded_points(
            centerline.get_branch(k), points, bounding_sphere_radius_mm
        )
        for p in found:
            claims.setdefault(tuple(p), []).append(k)

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)

    def distance_to_branch(k: int) -> np.ndarray:
        axis = np.asarray(centerline.get_branch(k).points_as_tuples())
        return cKDTree(axis).query(pts)[0]

    distance = {k: distance_to_branch(k) for k in ids}
    depths = {k: depth(k) for k in ids}

    branches: dict[int, list] = {k: [] for k in ids}
    for i, p in enumerate(points):
        candidates = claims.get(tuple(p))
        if candidates:
            owner = min(candidates, key=lambda k: (depths[k], distance[k][i]))
        else:
            owner = min(ids, key=lambda k: distance[k][i])
        branches[owner].append(p)
    results[branches_key(vessel)] = branches

    print(f"\nBranch labeling for '{vessel}_points':")
    for k, branch_points in branches.items():
        print(f"  branch {k}: {len(branch_points)}")

    return results
