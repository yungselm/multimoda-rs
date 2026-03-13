from __future__ import annotations

from . import manipulating
from . import labeling
from . import debug_plots
from . import fixing_functions

from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
import trimesh

if TYPE_CHECKING:
    from ..multimodars import PyCenterline, PyFrame, PyGeometry

# -------------------------------
# Convenience Functions
# -------------------------------
def label(
    path_ccta_geometry: Path | str | trimesh.Trimesh,
    path_centerline_aorta: Path | str | PyCenterline,
    path_centerline_rca: Path | str | PyCenterline,
    path_centerline_lca: Path | str | PyCenterline,
    aligned_frames: list[PyFrame],
    anomalous_rca: bool = False,
    anomalous_lca: bool = False,
    n_points_intramural: int = 120,
    bounding_sphere_radius_mm: float = 3.0,
    tolerance_float: float = 1e-6,
    control_plot: bool = True,
) -> tuple[dict, tuple[PyCenterline, PyCenterline, PyCenterline]]:
    """Label CCTA mesh vertices as aorta, RCA, or LCA using centerline-based region detection.

    Loads a 3-D surface mesh and three centerlines (aorta, RCA, LCA), then assigns
    each mesh vertex to one of the anatomical regions. For anomalous vessels an
    additional occlusion-removal step uses ray-triangle intersection to strip
    intramural segments, followed by adjacency-map reclassification to clean up
    isolated mis-labelled vertices. Herfore, a ray is cast from every aorta point to the centerline
    points of the anomalous section and if 3 faces are intersected by the ray the points from
    the first face must correspond to the intramural section.

    Additionally partition a coronary region into proximal, anomalous, and distal sub-regions.

    Uses the intravascular imaging frames to determine where along the centerline
    the anomalous (intramural) segment begins and ends, then tags each mesh
    vertex accordingly.

    Parameters
    ----------
    path_ccta_geometry : Path or str
        Path to the CCTA surface mesh file (any format supported by
        :func:`multimodars.io.read_geometrical.read_mesh`).
    path_centerline_aorta : Path or str
        Path to a CSV file containing the aortic centerline (comma-delimited,
        columns: x, y, z, …).
    path_centerline_rca : Path or str
        Path to a CSV file containing the RCA centerline.
    path_centerline_lca : Path or str
        Path to a CSV file containing the LCA centerline.
    aligned_frames : list of PyFrame
        Ordered list of intravascular imaging frames for the vessel.
    anomalous_rca : bool, optional
        When ``True`` applies ray-triangle occlusion removal to the RCA region
        to handle anomalous (intramural) courses.  Default is ``False``.
    anomalous_lca : bool, optional
        When ``True`` applies ray-triangle occlusion removal to the LCA region.
        Default is ``False``.
    n_points_intramural : int, optional
        Number of coronary centerline points examined during occlusion removal
        (the intramural segment length).  Default is ``120``.
    bounding_sphere_radius_mm : float, optional
        Radius in millimetres of the rolling sphere used to collect candidate
        mesh vertices around each centerline point.  Default is ``3.0``.
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

    centerlines : tuple
        A 3-tuple ``(cl_rca, cl_lca, cl_aorta)`` of ``PyCenterline`` objects.

    Raises
    ------
    Exception
        Re-raises any error that occurs while reading the mesh or centerline
        files, after printing a descriptive message.
    """
    results, (rca_cl, lca_cl, ao_cl) = labeling.label_geometry(
        path_ccta_geometry,
        path_centerline_aorta,
        path_centerline_rca,
        path_centerline_lca,
        anomalous_rca,
        anomalous_lca,
        n_points_intramural,
        bounding_sphere_radius_mm,
        tolerance_float,
        control_plot,
    )

    if anomalous_rca or anomalous_lca:
        if anomalous_rca:
            key='rca_points'
            cl=rca_cl
        else:
            key='lca_points'
            cl=lca_cl
        
        results = labeling.label_anomalous_region(
            centerline=cl,
            frames=aligned_frames,
            results=results,
            results_key=key,
        )
    
    return results, (rca_cl, lca_cl, ao_cl)


def scale(
    results: dict,
    cl_vessel: PyCenterline,
    cl_aorta: PyCenterline,
    aligned_frames: list[PyFrame],
) -> dict:
    """Scale the distal, proximal, and aortic regions of the vessel mesh.

    1. Computes proximal and distal radial scaling factors via
       :func:`manipulating.find_distal_and_proximal_scaling`, which matches
       the anomalous segment endpoints to lumen wall points from the first and
       last intravascular frames.
    2. Computes an aortic radial scaling factor via
       :func:`manipulating.find_aorta_scaling`, which aligns removed RCA points
       to reconstructed aortic wall points from the frames.
    3. Applies the scaling in sequence — distal, then aortic (``aorta_points``
       + ``rca_removed_points``), then proximal — using
       :func:`manipulating.scale_region_centerline_morphing`, which displaces
       each vertex radially around the nearest centerline point.
    4. After each aortic/proximal scaling step,
       :func:`manipulating.sync_results_to_mesh` remaps all coordinate lists
       in *results* to the updated vertex positions.

    Parameters
    ----------
    results : dict
        Labelled results dictionary containing at minimum:

        * ``"mesh"`` - the :class:`trimesh.Trimesh` to scale.
        * ``"anomalous_points"`` - points in the anomalous segment.
        * ``"distal_points"`` - vertices of the distal region.
        * ``"proximal_points"`` - vertices of the proximal region.
        * ``"aorta_points"`` - vertices of the aortic wall region.
        * ``"rca_removed_points"`` - RCA vertices removed by occlusion detection.

    cl_vessel : PyCenterline
        Centerline of the vessel (used for proximal/distal scaling).
    cl_aorta : PyCenterline
        Centerline of the aorta (used for aortic scaling).
    aligned_frames : list of PyFrame
        Ordered intravascular imaging frames, used as the reference geometry
        for computing all three scaling factors.

    Returns
    -------
    dict
        Updated *results* with ``"mesh"`` replaced by the fully scaled mesh
        and all coordinate lists remapped to the new vertex positions.
    """
    prox_scaling, distal_scaling = manipulating.find_distal_and_proximal_scaling(
        frames=aligned_frames,
        centerline=cl_vessel,
        results=results,
    )

    aortic_scaling = manipulating.find_aorta_scaling(
        frames=aligned_frames,
        centerline=cl_aorta,
        results=results,
    )

    scaled_distal = manipulating.scale_region_centerline_morphing(
        mesh=results['mesh'],
        region_points=results['distal_points'],
        centerline=cl_vessel,
        diameter_adjustment_mm=distal_scaling,
    )
    results = manipulating.sync_results_to_mesh(results, results['mesh'], scaled_distal)

    scaled_distal_aortic = manipulating.scale_region_centerline_morphing(
        mesh=results['mesh'],
        region_points=results['aorta_points'] + results['rca_removed_points'],
        centerline=cl_aorta,
        diameter_adjustment_mm=aortic_scaling,
    )
    results = manipulating.sync_results_to_mesh(results, results['mesh'], scaled_distal_aortic)

    scaled_proximal = manipulating.scale_region_centerline_morphing(
        mesh=results['mesh'],
        region_points=results['proximal_points'],
        centerline=cl_vessel,
        diameter_adjustment_mm=prox_scaling,
    )
    results = manipulating.sync_results_to_mesh(results, results['mesh'], scaled_proximal)

    return results

def stitch(
    results: dict,
    geometry: PyGeometry,
    postprocessing: bool = False,
    **postprocessing_kwargs,
) -> dict:
    """Stitch a CCTA mesh to the intravascular geometry and optionally remesh.

    Removes labeled anatomical regions from the CCTA mesh, then stitches the
    remaining surface to the intravascular geometry reconstructed from
    *geometry*.  When *postprocessing* is ``True`` **and** pymeshlab is
    installed, the stitched mesh is repaired, isotropically remeshed, and
    smoothed with a Taubin filter before being returned.

    Parameters
    ----------
    results : dict
        Labelled results dictionary (output of :func:`scale`), containing at
        minimum ``"mesh"`` and the point-label lists produced by
        :func:`label`.
    geometry : PyGeometry
        Intravascular imaging geometry whose contours define the vessel lumen
        used as the stitching target.
    postprocessing : bool, optional
        When ``True``, run :func:`fixing_functions.fix_and_remesh_stitched_mesh`
        followed by Taubin smoothing on the stitched mesh.  Silently skipped
        if pymeshlab is not installed.  Default is ``False``.
    **postprocessing_kwargs
        Keyword arguments forwarded to
        :func:`fixing_functions.fix_and_remesh_stitched_mesh`, e.g.
        ``target_edge_length_mm``, ``remesh_iterations``, ``verbose``.

    Returns
    -------
    dict
        Stitched results dictionary with the same structure as *results*, where
        ``"mesh"`` is the stitched (and optionally postprocessed) surface.
    """
    if postprocessing and fixing_functions.pymeshlab is None:
        raise ImportError(
            "postprocessing=True requires pymeshlab. "
            "Install it with: pip install 'multimodars[meshlab]'"
        )

    updated_results = manipulating.remove_labeled_points_from_mesh(results)

    stitched = manipulating.stitch_ccta_to_intravascular(
        geometry,
        updated_results['mesh'],
        updated_results,
    )

    stitched['mesh'] = fixing_functions.manual_hole_fill(stitched['mesh'])

    stitched['mesh'] = fixing_functions.postprocess_stitched_mesh(
        stitched['mesh'],
        postprocessing=postprocessing,
        **postprocessing_kwargs,
    )

    return stitched


def _extract_region_with_border_faces(
    mesh: trimesh.Trimesh,
    region_points: list,
) -> trimesh.Trimesh:
    """Return a sub-mesh containing every face that touches at least one vertex
    in *region_points*.

    Unlike :func:`manipulating.keep_labeled_points_from_mesh`, which only keeps
    faces whose *all* vertices belong to the region, this function uses an
    **at-least-one-vertex** criterion.  The result therefore includes the thin
    ring of adjacent-region vertices that share a face with the target region,
    giving seamless overlapping boundaries when meshes of different labels are
    exported side-by-side.
    """
    coord_to_idx = {tuple(v): i for i, v in enumerate(mesh.vertices)}
    keep_indices = np.array(
        [coord_to_idx[tuple(p)] for p in region_points if tuple(p) in coord_to_idx],
        dtype=np.int64,
    )
    if keep_indices.size == 0:
        return trimesh.Trimesh()

    face_mask = np.isin(mesh.faces, keep_indices).any(axis=1)
    selected_faces = mesh.faces[face_mask]

    used = np.unique(selected_faces)
    remap = np.full(len(mesh.vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)

    return trimesh.Trimesh(
        vertices=mesh.vertices[used],
        faces=remap[selected_faces],
        process=False,
    )


def export_section_stl(
    results: dict,
    type: str = "all",
    output_dir: Path | str | None = None,
) -> None:
    """Export the mesh (or a labeled sub-region) as an STL file.

    Parameters
    ----------
    results : dict
        Labeled results dictionary containing ``"mesh"`` and the point-label
        lists produced by :func:`label` / :func:`scale`.
    type : str, optional
        Which region to export.  One of:

        * ``"all"``   - the full mesh as-is.
        * ``"aorta"`` - only the aorta region.
        * ``"rca"``   - only the RCA region (includes adjacent aorta ring).
        * ``"lca"``   - only the LCA region (includes adjacent aorta ring).

        Default is ``"all"``.
    output_dir : Path, str, or None, optional
        Directory in which to write the STL file.  Defaults to the current
        working directory when ``None``.
    """
    output_dir = Path(output_dir) if output_dir is not None else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh: trimesh.Trimesh = results["mesh"]

    _REGION_KEYS = {
        "aorta": "aorta_points",
        "rca":   "rca_points",
        "lca":   "lca_points",
    }

    if type == "all":
        mesh.export(str(output_dir / "all.stl"))
    elif type in _REGION_KEYS:
        region_points = results.get(_REGION_KEYS[type], [])
        sub_mesh = _extract_region_with_border_faces(mesh, region_points)
        sub_mesh.export(str(output_dir / f"{type}.stl"))
    else:
        raise ValueError(
            f"Unknown export type {type!r}. "
            f"Choose one of: 'all', 'aorta', 'rca', 'lca'."
        )
