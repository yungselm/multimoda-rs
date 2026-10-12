from __future__ import annotations

from itertools import pairwise

import numpy as np
import trimesh

from .stitching.boundary import order_points_list

try:
    import pymeshlab
except ImportError:
    pymeshlab = None  # type: ignore[assignment]


def manual_hole_fill(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Fill holes by fan-triangulating each boundary loop to its centroid."""
    outline = mesh.outline()

    new_vertices = list(mesh.vertices)
    new_faces = list(mesh.faces)

    # Build a lookup from coordinate tuple -> vertex index (original mesh)
    coord_to_idx = {tuple(v): i for i, v in enumerate(mesh.vertices)}

    for entity in outline.entities:
        pts = outline.vertices[entity.points]

        ordered = order_points_list(mesh, pts)
        if len(ordered) < 3:
            continue

        ordered_arr = np.array(ordered)
        centroid = ordered_arr.mean(axis=0)

        centroid_idx = len(new_vertices)
        new_vertices.append(centroid)

        n = len(ordered)
        for i in range(n):
            i0 = coord_to_idx.get(tuple(ordered[i]))
            i1 = coord_to_idx.get(tuple(ordered[(i + 1) % n]))
            if i0 is not None and i1 is not None:
                new_faces.append([i0, i1, centroid_idx])

    result = trimesh.Trimesh(
        vertices=np.array(new_vertices),
        faces=np.array(new_faces),
        process=False,
    )
    result.fix_normals()
    return result


def _closest_on_polyline(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Closest point on the open *polyline* (M, 3) for each of *points* (N, 3)."""
    if len(polyline) == 1:
        return np.repeat(polyline, len(points), axis=0)
    a, b = polyline[:-1], polyline[1:]
    ab = b - a
    len_sq = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-24)
    # (N, M-1) segment parameters, clamped to the segment.
    t = np.einsum("nmk,mk->nm", points[:, None, :] - a[None], ab) / len_sq
    t = np.clip(t, 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    nearest = np.argmin(np.linalg.norm(proj - points[:, None, :], axis=2), axis=1)
    return proj[np.arange(len(points)), nearest]


def fill_ring_on_vessel_wall(
    mesh: trimesh.Trimesh,
    ring: list[int],
    axis: np.ndarray,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Close one boundary ring with a patch that follows the wall of a vessel.

    Meant for the hole a removed side branch leaves on its parent vessel: a
    flat fan would cut a chord through the parent's lumen.  The ring is shrunk
    towards its centroid in concentric layers, and every layer vertex is pushed
    radially away from the parent centerline *axis* to the wall radius,
    interpolated from the ring's own distance to the axis.  Only this ring is
    closed; the mesh's other open boundaries are left alone.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh with the open ring.
    ring : list of int
        Vertex indices of the ring in walk order.
    axis : (M, 3) array
        Centerline points of the parent vessel, in order.

    Returns
    -------
    (trimesh.Trimesh, (K, 3) array)
        The patched mesh, and the coordinates of the vertices the patch added.
    """
    ring_pts = mesh.vertices[ring]
    n = len(ring_pts)
    if n < 3:
        return mesh, np.empty((0, 3))
    axis = np.asarray(axis, dtype=np.float64)

    ring_foot = _closest_on_polyline(ring_pts, axis)
    ring_radius = np.linalg.norm(ring_pts - ring_foot, axis=1)
    mean_radius = ring_radius.mean()
    # Fallback direction where a vertex sits on the axis (an ostium as wide as
    # its parent): the ring's mean outward direction.
    outward = (ring_pts - ring_foot) / np.maximum(ring_radius, 1e-12)[:, None]
    mean_outward = outward.mean(axis=0)
    mean_outward /= max(np.linalg.norm(mean_outward), 1e-12)

    centroid = ring_pts.mean(axis=0)
    edge = np.linalg.norm(np.roll(ring_pts, -1, axis=0) - ring_pts, axis=1).mean()
    span = np.linalg.norm(ring_pts - centroid, axis=1).mean()
    n_layers = int(np.clip(round(span / max(edge, 1e-12)), 1, 30))

    def onto_wall(flat: np.ndarray, radius: np.ndarray) -> np.ndarray:
        foot = _closest_on_polyline(flat, axis)
        d = flat - foot
        norm = np.linalg.norm(d, axis=1)
        d = np.where(
            (norm > 0.25 * radius)[:, None],
            d / np.maximum(norm, 1e-12)[:, None],
            mean_outward,
        )
        return foot + radius[:, None] * d

    new_vertices = []
    for j in range(1, n_layers):
        t = 1.0 - j / n_layers
        flat = centroid + t * (ring_pts - centroid)
        new_vertices.append(onto_wall(flat, t * ring_radius + (1.0 - t) * mean_radius))
    new_vertices.append(onto_wall(centroid[None], np.array([mean_radius])))
    added = np.vstack(new_vertices)

    base = len(mesh.vertices)
    layers = [np.asarray(ring)] + [
        base + (j - 1) * n + np.arange(n) for j in range(1, n_layers)
    ]
    center = base + (n_layers - 1) * n
    tris: list[list[int]] = []
    for outer, inner in pairwise(layers):
        for i in range(n):
            a, b = outer[i], outer[(i + 1) % n]
            c, d = inner[i], inner[(i + 1) % n]
            tris += [[a, b, d], [a, d, c]]
    last = layers[-1]
    tris += [[last[i], last[(i + 1) % n], center] for i in range(n)]
    faces = np.array(tris, dtype=np.int64)

    # The patch walks the ring edge ring[0] -> ring[1]; a consistently oriented
    # mesh must walk it the other way in the face already on that edge.
    r0, r1 = ring[0], ring[1]
    on_edge = mesh.faces[
        (mesh.faces == r0).any(axis=1) & (mesh.faces == r1).any(axis=1)
    ]
    if any((f[k], f[(k + 1) % 3]) == (r0, r1) for f in on_edge for k in range(3)):
        faces = faces[:, ::-1]

    patched = trimesh.Trimesh(
        vertices=np.vstack([mesh.vertices, added]),
        faces=np.vstack([mesh.faces, faces]),
        process=False,
    )
    return patched, added


def postprocess_stitched_mesh(
    mesh: trimesh.Trimesh,
    *,
    target_edge_length_mm: float | None = None,
    remesh_iterations: int = 10,
    lamb: float = 0.5,
    nu: float = 0.5,
    verbose: bool = False,
    **kwargs,
) -> trimesh.Trimesh:
    """Remesh and smooth a stitched mesh.

    A separate, explicit step from :func:`~multimodars.ccta.stitch` - call this
    on its result when the heavier remesh/smooth pass is wanted.  Requires
    pymeshlab to be installed.

    Parameters
    ----------
    mesh:
        Input stitched mesh.
    target_edge_length_mm:
        Passed to :func:`fix_and_remesh_stitched_mesh`.
    remesh_iterations:
        Passed to :func:`fix_and_remesh_stitched_mesh`.
    verbose:
        Passed to :func:`fix_and_remesh_stitched_mesh`.
    **kwargs:
        Additional keyword arguments passed to :func:`fix_and_remesh_stitched_mesh`.
    """
    mesh = fix_and_remesh_stitched_mesh(
        mesh,
        target_edge_length_mm=target_edge_length_mm,
        remesh_iterations=remesh_iterations,
        verbose=verbose,
        **kwargs,
    )
    trimesh.smoothing.filter_taubin(mesh, lamb=lamb, nu=nu)
    return mesh


def _trimesh_to_meshset(mesh: trimesh.Trimesh):
    ms = pymeshlab.MeshSet()
    m = pymeshlab.Mesh(
        vertex_matrix=mesh.vertices.astype(np.float64),
        face_matrix=mesh.faces.astype(np.int32),
    )
    ms.add_mesh(m)
    return ms


def _meshset_to_trimesh(ms) -> trimesh.Trimesh:
    m = ms.current_mesh()
    return trimesh.Trimesh(
        vertices=m.vertex_matrix(),
        faces=m.face_matrix(),
        process=False,
    )


def fix_and_remesh_stitched_mesh(
    mesh: trimesh.Trimesh,
    *,
    target_edge_length_mm: float | None = None,
    remesh_iterations: int = 10,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Fill holes and remesh a stitched mesh, replicating Meshmixer workflow.

    Steps
    -----
    1. Repair non-manifold edges/vertices.
    2. Close holes - flat fill (all holes).
    3. Isotropic remesh to ``target_edge_length_mm``.

    Parameters
    ----------
    mesh:
        Input trimesh (the stitched surface).
    target_edge_length_mm:
        Target edge length in mm for the isotropic remesh.  If ``None``,
        uses the 25th-percentile edge length of the input mesh (preserves
        the fine IV-mesh resolution as reference).
    remesh_iterations:
        Number of isotropic remeshing iterations (default 10).
    verbose:
        Print progress info.
    """

    if pymeshlab is None:
        raise ImportError(
            "pymeshlab is required for fix_and_remesh_stitched_mesh. "
            "Install it with: pip install 'multimodars[meshlab]'"
        )

    def _log(label: str, m: trimesh.Trimesh) -> None:
        if verbose:
            print(
                f"[{label:35s}] verts={len(m.vertices):>7,}  "
                f"faces={len(m.faces):>7,}  "
                f"watertight={m.is_watertight}"
            )

    _log("input", mesh)

    # Use the fine end of the edge-length distribution as reference so that
    # the IV mesh resolution drives the target (not the coarser CCTA edges).
    if target_edge_length_mm is None:
        target_edge_length_mm = float(np.percentile(mesh.edges_unique_length, 25))
        if verbose:
            print(f"  auto target edge length = {target_edge_length_mm:.4f} mm (P25)")

    ms = _trimesh_to_meshset(mesh)

    # ------------------------------------------------------------------
    # 0.  Repair non-manifold geometry (required before hole filling)
    # ------------------------------------------------------------------
    ms.meshing_repair_non_manifold_edges(method=0)  # 0 = remove faces
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_null_faces()
    if verbose:
        print("  non-manifold edges/vertices repaired")

    # ------------------------------------------------------------------
    # 1.  Fill holes – flat fill
    # ------------------------------------------------------------------
    ms.meshing_close_holes(
        maxholesize=1000,
        selfintersection=False,
    )
    if verbose:
        print("  holes closed")

    mesh_filled = _meshset_to_trimesh(ms)
    _log("after hole fill", mesh_filled)

    # ------------------------------------------------------------------
    # 2.  Isotropic remesh
    #     targetlen / maxsurfdist are expressed as % of bbox diagonal.
    # ------------------------------------------------------------------
    bbox_diag = float(np.linalg.norm(mesh_filled.bounding_box.extents))
    targetlen_pct = (target_edge_length_mm / bbox_diag) * 100.0
    maxsurfdist_pct = targetlen_pct * 0.5

    if verbose:
        print(
            f"  target edge={target_edge_length_mm:.4f} mm  "
            f"({targetlen_pct:.4f}% of bbox diag={bbox_diag:.2f} mm)"
        )

    ms2 = _trimesh_to_meshset(mesh_filled)
    ms2.meshing_isotropic_explicit_remeshing(
        targetlen=pymeshlab.PercentageValue(targetlen_pct),
        iterations=remesh_iterations,
        adaptive=False,
        selectedonly=False,
        checksurfdist=True,
        maxsurfdist=pymeshlab.PercentageValue(maxsurfdist_pct),
        splitflag=True,
        collapseflag=True,
        swapflag=True,
        smoothflag=True,
        reprojectflag=True,
    )

    mesh_remeshed = _meshset_to_trimesh(ms2)
    mesh_remeshed.fix_normals()
    _log("after remesh", mesh_remeshed)

    # ------------------------------------------------------------------
    # 3.  Post-remesh cleanup: remeshing can open small holes; close them.
    # ------------------------------------------------------------------
    if not mesh_remeshed.is_watertight:
        ms3 = _trimesh_to_meshset(mesh_remeshed)
        ms3.meshing_repair_non_manifold_edges(method=0)
        ms3.meshing_repair_non_manifold_vertices()
        ms3.meshing_remove_duplicate_faces()
        ms3.meshing_remove_null_faces()
        ms3.meshing_close_holes(maxholesize=1000, selfintersection=False)
        mesh_remeshed = _meshset_to_trimesh(ms3)
        mesh_remeshed.fix_normals()
        _log("after post-remesh fix", mesh_remeshed)

    return mesh_remeshed
