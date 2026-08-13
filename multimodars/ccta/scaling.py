from __future__ import annotations

import numpy as np
import trimesh

from ..multimodars import (
    PyFrame,
    PyCenterline,
    adjust_diameter_centerline_morphing_simple,
    find_proximal_distal_scaling,
    find_aortic_scaling,
    find_aortic_wall_scaling as _find_aortic_wall_scaling,
)


def scale_region_centerline_morphing(
    mesh: trimesh.Trimesh,
    region_points: list,
    centerline,
    diameter_adjustment_mm: float,
) -> trimesh.Trimesh:
    """Scale a mesh region radially around its centerline.

    Each vertex in *region_points* is displaced along the direction from the
    nearest centerline point outward (positive *diameter_adjustment_mm*) or
    inward (negative).

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The original mesh.  A copy is returned; the input is not modified.
    region_points : list of tuple
        ``(x, y, z)`` coordinates of the vertices to be scaled.  Only vertices
        present in this list are moved.
    centerline : PyCenterline
        Centerline of the vessel region used as the morphing axis.
    diameter_adjustment_mm : float
        Diameter change in millimetres.  Positive values expand the lumen;
        negative values contract it.

    Returns
    -------
    trimesh.Trimesh
        A new mesh with the selected region scaled.

    Warns
    -----
    If no vertices matching *region_points* are found in the mesh, a warning
    is printed and the unmodified copy is returned.
    """
    scaled_mesh = mesh.copy()

    region_vertex_indices_list: list[int] = []
    region_set = set(region_points)

    for idx, vertex in enumerate(scaled_mesh.vertices):
        if tuple(vertex) in region_set:
            region_vertex_indices_list.append(idx)

    region_vertex_indices = np.array(region_vertex_indices_list)

    if len(region_vertex_indices) == 0:
        print("Warning: No vertices found for scaling region")
        return scaled_mesh

    print(f"\nScaling {len(region_vertex_indices)} vertices around {centerline}")
    print(f"Diameter adjustment: {np.round(diameter_adjustment_mm, 2)} mm")

    region_vertices_list = [
        tuple(vertex) for vertex in scaled_mesh.vertices[region_vertex_indices]
    ]
    adjusted_points = adjust_diameter_centerline_morphing_simple(
        centerline=centerline,
        points=region_vertices_list,
        diameter_adjustment_mm=diameter_adjustment_mm,
    )

    scaled_mesh.vertices[region_vertex_indices] = np.array(
        adjusted_points, dtype=np.float64
    )

    # Clear mesh cache since we modified vertices directly
    scaled_mesh.vertices.flags["WRITEABLE"] = False

    return scaled_mesh


def find_distal_and_proximal_scaling(
    frames,
    centerline,
    results: dict,
    dist_range: int = 3,
    prox_range: int = 2,
) -> tuple[float, float]:
    """Compute the optimal radial scaling factors for the proximal and distal segments.

    Collects lumen wall points from the first *prox_range* and last *dist_range*
    imaging frames as reference geometry, then calls the Rust
    ``find_proximal_distal_scaling`` routine to find the scaling factors that
    best match the anomalous segment endpoints to those references.

    Parameters
    ----------
    frames : list of PyFrame
        Ordered intravascular imaging frames for the vessel.
    centerline : PyCenterline
        Centerline of the vessel region.
    results : dict
        Labelled results dictionary containing ``"anomalous_points"``.
    dist_range : int, optional
        Number of frames from the distal end used as the distal reference.
        Default is ``3``.
    prox_range : int, optional
        Number of frames from the proximal end used as the proximal reference.
        Default is ``2``.

    Returns
    -------
    prox_scaling : float
        Optimal radial scaling factor for the proximal segment.
    dist_scaling : float
        Optimal radial scaling factor for the distal segment.
    """
    frame_points_dist = [
        (p.x, p.y, p.z) for f in frames[-dist_range:] for p in f.lumen.points
    ]
    frame_points_prox = [
        (p.x, p.y, p.z) for f in frames[0:prox_range] for p in f.lumen.points
    ]
    n_anomalous_points = len(results["anomalous_points"])
    n_section: int = int(np.ceil(0.25 * n_anomalous_points))

    print("\nFinding best proximal/distal radial scaling factors...")
    prox_scaling, dist_scaling = find_proximal_distal_scaling(
        results["anomalous_points"],
        n_section,
        n_section,
        centerline,
        frame_points_prox,
        frame_points_dist,
    )
    print(f"Proximal scaling: {np.round(prox_scaling, 2)} mm")
    print(f"Distal scaling: {np.round(dist_scaling, 2)} mm")

    return prox_scaling, dist_scaling


def find_aorta_scaling(
    frames: list[PyFrame],
    cl_aorta: PyCenterline,
    results: dict,
) -> float:
    """Compute the optimal radial scaling factor for the aortic region.

    Extracts reconstructed wall points from the intravascular frames (using
    ``aortic_thickness`` and the ``"Wall"`` extras) as a reference, then calls
    the Rust ``find_aortic_scaling`` routine to determine the factor that best
    aligns the removed RCA points to those references.

    Parameters
    ----------
    frames : list of PyFrame
        Intravascular imaging frames containing ``aortic_thickness`` and
        ``extras["Wall"]`` data.
    cl_aorta : PyCenterline
        Centerline of the aortic region.
    results : dict
        Labelled results dictionary containing ``"rca_removed_points"``.
    debug_plot : bool, optional
        Reserved for future use; currently unused.  Default is ``True``.

    Returns
    -------
    float
        Optimal radial scaling factor for the aortic segment.
    """
    reference_points = _extract_wall_from_frames(frames)
    if reference_points is None:
        raise ValueError("No aortic wall points found in frames for scaling reference")

    print("\nFinding best aortic radial scaling factor...")
    scaling = find_aortic_scaling(
        results["rca_removed_points"],  # For now work with removed points
        reference_points,
        cl_aorta,
    )
    print(f"Aortic scaling: {np.round(scaling, 2)} mm")

    return scaling


def find_aortic_wall_scaling(
    frames: list[PyFrame],
    cl_aorta: PyCenterline,
    results: dict,
) -> float:
    """Compute the optimal radial scaling factor for the aortic wall region.
    This is created for anomalous coronaries, and tries to optimize the aortic wall
    to the point on the first quarter towards the aortic wall of the first round
    lumen (marking the end of the intramural course).

    End of the intramural course is defined as the first lumen with an elliptic ratio <1.3

    Parameters
    ----------
    frames : list of PyFrame
        Intravascular imaging frames.
    cl_aorta : PyCenterline
        Centerline of the aortic region.
    results : dict
        Labelled results dictionary containing ``"rca_removed_points"``.

    Returns
    -------
    float
        Optimal radial scaling factor for the aortic wall.
    """
    ref_point = None

    print("\nFinding best aortic wall radial scaling factor...")
    for frame in frames:
        elliptic_ratio = frame.lumen.get_elliptic_ratio()
        if elliptic_ratio < 1.3:
            print(f"elliptic ratio <1.3 for frame index {frame.id}")
            point_idx = len(frame.lumen) // 4
            ref_point_ir = frame.lumen.points[point_idx]
            ref_point = (ref_point_ir.x, ref_point_ir.y, ref_point_ir.z)
            break
        else:
            continue

    if ref_point is None:
        raise ValueError("No coronary reference point found")
    scaling = _find_aortic_wall_scaling(cl_aorta, ref_point, results["aorta_points"])
    print(f"Aortic wall scaling: {np.round(scaling, 2)} mm")

    return scaling


def _extract_wall_from_frames(frames) -> list[tuple[float, float, float]] | None:
    """Extract the straight-wall (coronary-side) points from intravascular frames.

    ``create_aortic_wall`` in ``wall.rs`` builds the ``"Wall"`` extra contour
    in two halves:

    * **Straight wall** - ``point_index`` 0 to ``n // 2`` (exclusive): the lumen
      contour offset outward by 1 mm, following the true circular/elliptic vessel
      geometry on the coronary side.
    * **Aortic wall** - ``point_index`` ``n // 2`` to ``n``: the rectangular
      aortic-thickness shape constructed from ``aortic_thickness``.

    Only the straight-wall half is returned because it preserves the actual vessel
    cross-section shape and is therefore a stable reference for radial scaling.
    Assumes an even number of points per frame (the standard 500-point geometry).

    Parameters
    ----------
    frames : list of PyFrame
        Intravascular imaging frames.  Frames without ``aortic_thickness`` are
        skipped.

    Returns
    -------
    list of tuple
        ``(x, y, z)`` tuples of straight-wall points from the last eligible frame.
        Returns ``None`` if no eligible frame is found.

    Raises
    ------
    ValueError
        If an eligible frame is missing the ``"Wall"`` extras entry or that
        entry is empty.
    """
    n_points = len(frames[0].lumen.points)
    half = n_points // 2

    reference_points = None

    for frame in frames:
        if frame.lumen.aortic_thickness is None:
            continue
        wall = frame.extras.get("Wall")
        if wall is None:
            raise ValueError(
                f"No Wall extras found for frame {getattr(frame, 'frame', '?')}"
            )
        if not wall.points:
            raise ValueError(
                f"Empty Wall extras for frame {getattr(frame, 'frame', '?')}"
            )

        # Straight wall: coronary-side offset lumen, point_index 0..half.
        # Aortic wall:   rectangular aortic-thickness shape, point_index half..n_points.
        reference_points = [
            (p.x, p.y, p.z) for p in wall.points if p.point_index < half
        ]

    return reference_points


def sync_results_to_mesh(
    results: dict,
    old_mesh: trimesh.Trimesh,
    new_mesh: trimesh.Trimesh,
) -> dict:
    """Update all coordinate lists in *results* after vertices have been moved.

    Use this after :func:`scale_region_centerline_morphing` to keep the stored
    point lists consistent with the new vertex positions.  The two meshes must
    have the same vertex count and ordering (only positions change, no vertices
    are added or removed).

    Parameters
    ----------
    results : dict
        Results dict whose coordinate lists should be refreshed.
    old_mesh : trimesh.Trimesh
        The mesh whose vertex positions match the current coordinate lists.
    new_mesh : trimesh.Trimesh
        The mesh with updated vertex positions (same indices, new coordinates).

    Returns
    -------
    dict
        Updated *results* with ``"mesh"`` replaced by *new_mesh* and all
        coordinate lists remapped to the new vertex positions.
    """
    old_coord_to_idx = {tuple(v): i for i, v in enumerate(old_mesh.vertices)}

    updated = dict(results)
    updated["mesh"] = new_mesh

    for key in (
        "aorta_points",
        "rca_points",
        "lca_points",
        "rca_removed_points",
        "lca_removed_points",
        "proximal_points",
        "distal_points",
        "anomalous_points",
        "boundary_points",
        # Per-ring boundary lists, so they stay in step with "boundary_points".
        *sorted(k for k in updated if k.startswith("boundary_points_")),
    ):
        if key not in updated or not updated[key]:
            continue
        indices = [old_coord_to_idx.get(tuple(p)) for p in updated[key]]
        updated[key] = [tuple(new_mesh.vertices[i]) for i in indices if i is not None]

    return updated
