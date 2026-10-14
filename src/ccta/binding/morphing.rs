//! Bindings for morphing a mesh: changing its shape by scaling regions along a centerline,
//! fixing face winding and closing holes.
use crate::ccta::discretizing::SurfaceMesh;
use crate::ccta::morphing::{fill_holes, scaling, winding};
use crate::types::binding::PyCenterline;
use pyo3::prelude::*;

type Point3D = (f64, f64, f64);

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(adjust_diameter_centerline_morphing, m)?)?;
    m.add_function(wrap_pyfunction!(find_proximal_distal_scaling, m)?)?;
    m.add_function(wrap_pyfunction!(find_aortic_scaling, m)?)?;
    m.add_function(wrap_pyfunction!(find_aortic_wall_scaling, m)?)?;
    m.add_function(wrap_pyfunction!(fix_mesh_winding, m)?)?;
    m.add_function(wrap_pyfunction!(fill_holes_fan, m)?)?;
    Ok(())
}

/// Adjust the vessel diameter by morphing points outward or inward along the centerline.
///
/// Parameters
/// ----------
/// centerline : PyCenterline
///     Centerline of the vessel.
/// points : list of tuple of float
///     Input ``(x, y, z)`` point coordinates.
/// diameter_adjustment_mm : float
///     Amount to adjust the diameter in mm.  Positive values expand the
///     vessel; negative values contract it.
///
/// Returns
/// -------
/// adjusted_points : list of tuple of float
///     Morphed ``(x, y, z)`` point coordinates.
#[pyfunction]
pub fn adjust_diameter_centerline_morphing(
    centerline: PyCenterline,
    points: Vec<Point3D>,
    diameter_adjustment_mm: f64,
) -> PyResult<Vec<Point3D>> {
    let rust_centerline = centerline.to_rust_centerline();

    let result_points = scaling::centerline_based_diameter_morphing(
        &rust_centerline,
        &points,
        diameter_adjustment_mm,
    );

    Ok(result_points)
}

/// Find the optimal diameter scaling for the proximal and distal regions.
///
/// Parameters
/// ----------
/// anomalous_points : list of tuple of float
///     ``(x, y, z)`` coordinates of the anomalous vessel region.
/// n_proximal : int
///     Number of proximal points used for comparison.
/// n_distal : int
///     Number of distal points used for comparison.
/// centerline : PyCenterline
///     Centerline of the vessel region.
/// proximal_reference : list of tuple of float
///     Reference ``(x, y, z)`` points from the CCTA mesh for the proximal region.
/// distal_reference : list of tuple of float
///     Reference ``(x, y, z)`` points for the distal region.
///
/// Returns
/// -------
/// proximal_scaling : float
///     Optimal scaling distance for the proximal region.
/// distal_scaling : float
///     Optimal scaling distance for the distal region.
///
/// Examples
/// --------
/// >>> import multimodars as mm
#[pyfunction]
pub fn find_proximal_distal_scaling(
    anomalous_points: Vec<Point3D>,
    n_proximal: usize,
    n_distal: usize,
    centerline: PyCenterline,
    proximal_reference: Vec<Point3D>,
    distal_reference: Vec<Point3D>,
) -> PyResult<(f64, f64)> {
    let rust_centerline = centerline.to_rust_centerline();
    let (prox_dist, distal_dist) = scaling::centerline_based_diameter_optimization(
        &anomalous_points,
        n_proximal,
        n_distal,
        &rust_centerline,
        &proximal_reference,
        &distal_reference,
    );
    Ok((prox_dist, distal_dist))
}

/// Find the optimal aortic diameter scaling using the intramural region.
///
/// Parameters
/// ----------
/// intramural_points : list of tuple of float
///     ``(x, y, z)`` coordinates of the intramural vessel region.
/// reference_points : list of tuple of float
///     Reference ``(x, y, z)`` points from the CCTA mesh.
/// centerline : PyCenterline
///     Centerline of the aorta.
///
/// Returns
/// -------
/// scaling : float
///     Optimal scaling distance for the aortic region.
///
/// Examples
/// --------
/// >>> import multimodars as mm
#[pyfunction]
pub fn find_aortic_scaling(
    intramural_points: Vec<Point3D>,
    reference_points: Vec<Point3D>,
    centerline: PyCenterline,
) -> PyResult<f64> {
    let rust_centerline = centerline.to_rust_centerline();
    let dist = scaling::centerline_based_aortic_diameter_optimization(
        &intramural_points,
        &reference_points,
        &rust_centerline,
    );
    Ok(dist)
}

/// Find the optimal aortic wall scaling for coronary artery anomalies
/// using aortic points and the aortic centerline.
/// Additionally needs a reference point on the first quarter of the first round lumen.
/// Projects a vector from the centerline point to the reference point on the coronary
/// and finds the best scaling along this vector
///
/// Parameters
/// ----------
/// centerline : PyCenterline
///     Centerline of the aorta.
/// ref_pt_coronary : tuple of float
///     ``(x, y, z)`` coordinates of the first round ref point.
/// aortic_pts : list of tuple of float
///     Reference ``(x, y, z)`` points from the CCTA mesh.
///
/// Returns
/// -------
/// scaling : float
///     Optimal scaling distance for the aortic wall.
///
/// Examples
/// --------
/// >>> import multimodars as mm
/// >>>
#[pyfunction]
pub fn find_aortic_wall_scaling(
    cl_aorta: PyCenterline,
    ref_pt_coronary: Point3D,
    aortic_pts: Vec<Point3D>,
) -> PyResult<f64> {
    let rust_centerline = cl_aorta.to_rust_centerline();
    let dist = scaling::centerline_based_wall_diameter_optimization(
        &rust_centerline,
        &ref_pt_coronary,
        &aortic_pts,
    );
    Ok(dist)
}

/// Fix face winding so that every pair of adjacent faces traverses their
/// shared edge in opposite directions (consistent orientation).
///
/// Faithful port of `trimesh.repair.fix_winding` — a BFS over the
/// face-adjacency graph, once per connected component, propagating a
/// consistent orientation outward from an arbitrary root face per component.
/// trimesh's own implementation does this as a Python/NetworkX loop over
/// every face-adjacency edge with several small numpy allocations per
/// iteration; this Rust version does the same traversal without that
/// per-edge interpreter overhead.
///
/// Parameters
/// ----------
/// faces : list of list of int
///     Mesh face vertex-index triples, e.g. ``mesh.faces.tolist()``.
///
/// Returns
/// -------
/// faces : list of list of int
///     The same faces, each either left as-is or with its vertex order
///     reversed so that winding is consistent within each connected
///     component of the mesh.
#[pyfunction]
pub fn fix_mesh_winding(faces: Vec<[usize; 3]>) -> Vec<[usize; 3]> {
    winding::fix_mesh_winding(faces)
}

/// Close every open boundary loop with a fan of triangles around its centroid.
///
/// Each loop gets one new vertex, its edge-length-weighted centroid, and one
/// triangle per loop edge, wound like the faces around the loop. A closed mesh
/// comes back unchanged.
///
/// Parameters
/// ----------
/// vertices : list of (float, float, float)
///     Mesh vertices.
/// faces : list of [int, int, int]
///     Mesh faces.
///
/// Returns
/// -------
/// (list of (float, float, float), list of [int, int, int])
///     Vertices (the input ones first, then one centroid per loop) and faces
///     (the input ones first, then the fans).
#[pyfunction]
pub fn fill_holes_fan(
    vertices: Vec<Point3D>,
    faces: Vec<[usize; 3]>,
) -> PyResult<(Vec<Point3D>, Vec<[usize; 3]>)> {
    let mesh =
        SurfaceMesh::new(&vertices, faces).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let (v, f) = fill_holes::fill_holes_fan(&mesh.vertices, &mesh.faces);
    Ok((v.iter().map(|p| (p.x, p.y, p.z)).collect(), f))
}
