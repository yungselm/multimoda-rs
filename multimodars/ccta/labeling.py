from __future__ import annotations


from pathlib import Path
from typing import Any
import trimesh

from ..multimodars import (
    find_centerline_bounded_points_simple,
    find_faces_near_points,
    find_aortic_points,
    final_reclassification,
    remove_occluded_points_ray_triangle,
    clean_outlier_points,
    find_points_by_cl_region,
    keep_largest_connected_component,
    PyCenterline,
)
from ..io.read_geometrical import read_mesh
from .debug_plots import plot_results_key


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
    rca_points_found = find_centerline_bounded_points_simple(
        centerline=centerline_rca,
        points=points_list,
        radius=bounding_sphere_radius_mm_rca,
    )
    lca_points_found = find_centerline_bounded_points_simple(
        centerline=centerline_lca,
        points=points_list,
        radius=bounding_sphere_radius_mm_lca,
    )

    print(f"\nRCA points found: {len(rca_points_found)}")
    print(f"LCA points found: {len(lca_points_found)}")

    if acute_takeoff_rca:
        rca_removed_points, final_rca_points_found = _apply_occlusion_removal(
            range_mm_takeoff_rca,
            step_size_mm,
            tolerance_float,
            centerline_aorta,
            centerline_rca,
            points_list,
            mesh_faces_list,
            rca_points_found,
            "RCA",
        )
    else:
        final_rca_points_found = rca_points_found.copy()
        rca_removed_points = []

    if acute_takeoff_lca:
        lca_removed_points, final_lca_points_found = _apply_occlusion_removal(
            range_mm_takeoff_lca,
            step_size_mm,
            tolerance_float,
            centerline_aorta,
            centerline_lca,
            points_list,
            mesh_faces_list,
            lca_points_found,
            "LCA",
        )
    else:
        final_lca_points_found = lca_points_found.copy()
        lca_removed_points = []

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


def _apply_occlusion_removal(
    range_mm_takeoff_rca: float,
    step_size_mm: float,
    tolerance_float: float,
    cl_aorta: PyCenterline,
    cl_rca: PyCenterline,
    points_list: list[tuple[float, float, float]],
    mesh_faces_list: list[list[int]],
    rca_points_found: list[tuple[float, float, float]],
    name: str,
):
    print(f"Applying occlusion removal for acute-takeoff {name}...")
    rca_faces_for_rust = find_faces_near_points(
        points_list, mesh_faces_list, rca_points_found, tolerance_float
    )
    final_rca_points_found = remove_occluded_points_ray_triangle(
        centerline_coronary=cl_rca,
        centerline_aorta=cl_aorta,
        range_mm=range_mm_takeoff_rca,
        points=rca_points_found,
        faces=rca_faces_for_rust,
        step_size_mm=step_size_mm,
    )
    final_rca_points_found_set = set(final_rca_points_found)
    rca_removed_points = [
        p for p in rca_points_found if p not in final_rca_points_found_set
    ]
    print(f"{name}: relabeled {len(rca_removed_points)} points in intramual course")
    return rca_removed_points, final_rca_points_found


def label_anomalous_region(
    centerline,
    frames,
    results: dict,
    results_key: str = "rca_points",
    debug_plot: bool = False,
) -> dict:
    """Partition a coronary region into proximal, anomalous, and distal sub-regions.

    Uses the intravascular imaging frames to determine where along the centerline
    the anomalous (intramural) segment begins and ends, then tags each mesh
    vertex accordingly.

    Parameters
    ----------
    centerline : PyCenterline
        Centerline of the coronary vessel of interest.
    frames : list of PyFrame
        Ordered list of intravascular imaging frames for the vessel.
    results : dict
        Labelled results dictionary (e.g. from :func:`label_geometry`).
        Must contain the key specified by *results_key*.
    results_key : str, optional
        Key in *results* whose point list is partitioned.
        Default is ``"rca_points"``.
    debug_plot : bool, optional
        When ``True`` opens an interactive visualisation of the three
        sub-regions.  Default is ``False``.

    Returns
    -------
    dict
        The input *results* dictionary extended with three new keys:

        * ``"proximal_points"`` - vertices proximal to the anomalous segment.
        * ``"distal_points"`` - vertices distal to the anomalous segment.
        * ``"anomalous_points"`` - vertices within the anomalous segment.
    """
    proximal_points_raw, distal_points_raw, anomalous_points_raw = (
        find_points_by_cl_region(
            centerline=centerline,
            frames=frames,
            points=results[results_key],
        )
    )

    mesh = results["mesh"]
    proximal_points = _keep_largest_connected_component(mesh, proximal_points_raw)
    distal_points = _keep_largest_connected_component(mesh, distal_points_raw)
    anomalous_points = _keep_largest_connected_component(mesh, anomalous_points_raw)

    # Island points dropped above are still nominally part of results[results_key]
    # (e.g. rca_points), which would otherwise keep them out of the aorta_points
    # complement below - stranding them in no region at all, so no scaling step
    # ever touches them.  Reassign them fully to aorta by removing them from
    # results[results_key] too, mirroring how final_reclassification's Logic A
    # reassigns isolated vertices to aorta rather than leaving them ambiguous.
    dropped_points = (
        (set(proximal_points_raw) - set(proximal_points))
        | (set(distal_points_raw) - set(distal_points))
        | (set(anomalous_points_raw) - set(anomalous_points))
    )
    if dropped_points:
        results[results_key] = [
            p for p in results[results_key] if p not in dropped_points
        ]
        print(
            f"  {len(dropped_points)} island point(s) reassigned from "
            f"'{results_key}' sub-regions to aorta_points"
        )

    results["proximal_points"] = proximal_points
    results["distal_points"] = distal_points
    results["anomalous_points"] = anomalous_points

    all_coronary = (
        set(results.get("rca_points", []))
        | set(results.get("lca_points", []))
        | set(proximal_points)
        | set(distal_points)
        | set(anomalous_points)
    )
    results["aorta_points"] = [
        tuple(v) for v in results["mesh"].vertices if tuple(v) not in all_coronary
    ]

    print("\nApplying anomalous labeling based on aligned intravascular frames...")
    print(f"proximal_points: {len(results['proximal_points'])}")
    print(f"distal_points: {len(results['distal_points'])}")
    print(f"anomalous_points: {len(results['anomalous_points'])}")

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


def _keep_largest_connected_component(
    mesh: trimesh.Trimesh, points: list[tuple[float, float, float]]
) -> list[tuple[float, float, float]]:
    """Keep only the largest mesh-connected component of *points*.

    ``find_points_by_cl_region`` classifies points using coordinate-only
    heuristics (nearest 3-D centerline point, axis-aligned proximal/distal
    split) with no notion of mesh topology.  That can leave a handful of
    points assigned to a region despite not being mesh-connected to its main
    cluster ("islands") - e.g. a point geometrically close to the anomalous
    segment but on a different, unconnected part of the vessel surface.

    This restricts the mesh's face-adjacency graph to *points* and keeps
    only the single largest connected component; the rest are dropped
    (callers typically let dropped points fall through to a different
    region via a complement/set-difference step, mirroring how
    :func:`final_reclassification` reassigns isolated vertices to aorta).
    """
    return keep_largest_connected_component(
        [tuple(v) for v in mesh.vertices], mesh.faces.tolist(), points
    )


def label_branches(
    centerline,
    results: dict,
    results_key: str = "rca_points",
    branch_id: int | list[int] = 0,
    bounding_sphere_radius_mm: float = 3.0,
) -> dict:
    """Partition a coronary region into main branch and per-side-branch point sets.

    Parameters
    ----------
    centerline : PyCenterline
        Centerline of the coronary vessel of interest.
    results : dict
        Labelled results dictionary (e.g. from :func:`label_geometry`).
        Must contain the key specified by *results_key*.
    results_key : str, optional
        Key in *results* whose point list is partitioned.
        Default is ``"rca_points"``.
    branch_id : int or list of int, optional
        Branch index or list of branch indices whose combined points form the
        main branch (e.g. ``[0, 1]`` for LAD + LCx).  Default is ``0``.
    bounding_sphere_radius_mm : float, optional
        Radius of the rolling sphere used to collect candidate vertices.
        Default is ``3.0``.

    Returns
    -------
    dict
        The input *results* dictionary extended with:

        * ``"{results_key}_main"`` - vertices along the main branch(es).
        * ``"{results_key}_side"`` - all remaining vertices (aggregate).
        * ``"{results_key}_side_{k}"`` - vertices near side branch *k*, one
          key per side branch discovered in *centerline*.  A point may appear
          in more than one side-branch set when it sits near a bifurcation;
          the Voronoi inside :func:`discretize_vessel` resolves the assignment.
    """
    branch_ids = [branch_id] if isinstance(branch_id, int) else list(branch_id)
    main_id_set = set(branch_ids)

    # Collect main-branch points.
    main_set: set = set()
    for bid in branch_ids:
        branch = centerline.get_branch(bid)
        points_found = find_centerline_bounded_points_simple(
            branch, results[results_key], bounding_sphere_radius_mm
        )
        main_set.update(points_found)

    main_points = [p for p in results[results_key] if p in main_set]
    side_points = [p for p in results[results_key] if p not in main_set]

    results[f"{results_key}_main"] = main_points
    results[f"{results_key}_side"] = side_points

    # Split side points per individual side branch.
    n_branches = len(centerline.branch_start_indices)
    side_branch_ids = [k for k in range(n_branches) if k not in main_id_set]

    print(f"\nBranch labeling for '{results_key}' (branch_ids={branch_ids}):")
    print(f"  {results_key}_main: {len(main_points)}")
    print(f"  {results_key}_side: {len(side_points)}")

    for k in side_branch_ids:
        branch_k = centerline.get_branch(k)
        branch_k_points = find_centerline_bounded_points_simple(
            branch_k, side_points, bounding_sphere_radius_mm
        )
        results[f"{results_key}_side_{k}"] = branch_k_points
        print(f"  {results_key}_side_{k}: {len(branch_k_points)}")

    return results
