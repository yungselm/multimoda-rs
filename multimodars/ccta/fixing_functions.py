from __future__ import annotations

import numpy as np
import trimesh
from ..ccta.manipulating import order_points_list

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
        target_edge_length_mm = float(
            np.percentile(mesh.edges_unique_length, 25)
        )
        if verbose:
            print(f"  auto target edge length = {target_edge_length_mm:.4f} mm (P25)")

    ms = _trimesh_to_meshset(mesh)

    # ------------------------------------------------------------------
    # 0.  Repair non-manifold geometry (required before hole filling)
    # ------------------------------------------------------------------
    ms.meshing_repair_non_manifold_edges(method=0)   # 0 = remove faces
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
