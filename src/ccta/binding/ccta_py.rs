// src/ccta/binding/label_py.rs
use std::collections::{HashMap, HashSet};

use crate::ccta::adjust_mesh::label_coronary;
use crate::ccta::adjust_mesh::label_coronary::Triangle;
use crate::ccta::adjust_mesh::scale_coronary;
use crate::ccta::discretizing;
use crate::ccta::discretizing::vessel_tree::DiscretizedVesselTree;
use crate::intravascular::binding::classes::{PyCenterline, PyContour, PyFrame};
use pyo3::prelude::*;

type Point3D = (f64, f64, f64);
type TriangleTuple = (Point3D, Point3D, Point3D);
/// (main_ref, clock_ref, counter_clock_ref) — all three as plain xyz tuples.
type RefTriplet = (Point3D, Point3D, Point3D);

// ─── PyDiscretizedVesselTree ──────────────────────────────────────────────────

/// Fully discretized coronary vessel tree (aorta + RCA + LCA + side branches).
///
/// Attributes
/// ----------
/// discretized_aorta : list of PyContour
///     Cross-sectional contours along the aortic centerline.
/// discretized_rca_main : list of PyContour
///     Cross-sectional contours along the RCA main vessel.
/// discretized_lca_main : list of PyContour
///     Cross-sectional contours along the LCA main vessel.
/// rca_branches : list of list of PyContour
///     Per-side-branch contour lists for the RCA.  ``rca_branches[i]``
///     corresponds to RCA branch_id ``i + 1``.
/// lca_branches : list of list of PyContour
///     Per-side-branch contour lists for the LCA.
/// rca_references : list of tuple
///     Orientation triplets ``(main_ref, clock_ref, counter_clock_ref)`` along
///     the RCA, sorted proximal → distal.  Each element is a 3-tuple of
///     ``(x, y, z)`` coordinate tuples.
/// lca_references : list of tuple
///     Same structure for the LCA.
/// ao_rca : tuple of float
///     Centroid ``(x, y, z)`` of the aorta slice closest to the RCA ostium.
/// ao_lca : tuple of float
///     Centroid ``(x, y, z)`` of the aorta slice closest to the LCA ostium.
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyDiscretizedVesselTree {
    #[pyo3(get, set)]
    pub discretized_aorta: Vec<PyContour>,
    #[pyo3(get, set)]
    pub discretized_rca_main: Vec<PyContour>,
    #[pyo3(get, set)]
    pub discretized_lca_main: Vec<PyContour>,
    #[pyo3(get, set)]
    pub rca_branches: Vec<Vec<PyContour>>,
    #[pyo3(get, set)]
    pub lca_branches: Vec<Vec<PyContour>>,
    #[pyo3(get)]
    pub rca_references: Vec<RefTriplet>,
    #[pyo3(get)]
    pub lca_references: Vec<RefTriplet>,
    #[pyo3(get, set)]
    pub ao_rca: Point3D,
    #[pyo3(get, set)]
    pub ao_lca: Point3D,
}

#[pymethods]
impl PyDiscretizedVesselTree {
    fn __repr__(&self) -> String {
        format!(
            "DiscretizedVesselTree(ao={}, rca_main={}, lca_main={}, rca_branches={}, lca_branches={}, rca_refs={}, lca_refs={})",
            self.discretized_aorta.len(),
            self.discretized_rca_main.len(),
            self.discretized_lca_main.len(),
            self.rca_branches.len(),
            self.lca_branches.len(),
            self.rca_references.len(),
            self.lca_references.len(),
        )
    }

    /// Recompute orientation reference triplets and aortic ostium centroids
    /// from the current contour data.
    ///
    /// Call this after replacing contours (e.g. with B-spline fits) so that
    /// ``rca_references``, ``lca_references``, ``ao_rca``, and ``ao_lca``
    /// reflect the updated geometry.
    pub fn calculate_ref_pts(&mut self) -> PyResult<()> {
        let convert =
            |contours: &[PyContour]| -> PyResult<Vec<crate::intravascular::io::geometry::Contour>> {
                contours.iter().map(|c| c.to_rust_contour()).collect()
            };

        let rust_aorta = convert(&self.discretized_aorta)?;
        let rust_rca = convert(&self.discretized_rca_main)?;
        let rust_lca = convert(&self.discretized_lca_main)?;
        let rust_rca_branches: PyResult<Vec<Vec<_>>> =
            self.rca_branches.iter().map(|b| convert(b)).collect();
        let rust_lca_branches: PyResult<Vec<Vec<_>>> =
            self.lca_branches.iter().map(|b| convert(b)).collect();

        let tree = DiscretizedVesselTree {
            discretized_aorta: rust_aorta,
            discretized_rca_main: rust_rca,
            discretized_lca_main: rust_lca,
            rca_branches: rust_rca_branches?,
            lca_branches: rust_lca_branches?,
            spacing: 1.0,
            rca_references: vec![],
            lca_references: vec![],
            ao_rca: (0.0, 0.0, 0.0),
            ao_lca: (0.0, 0.0, 0.0),
            pts_cusp_rcc: None,
            pts_cusp_lcc: None,
            pts_cusp_acc: None,
            index_stj_slice: None,
            index_aa: None,
        };

        let updated = tree.calculate_ref_pts();
        self.rca_references = updated
            .rca_references
            .into_iter()
            .map(|r| (r.main_ref, r.clock_ref, r.counter_clock_ref))
            .collect();
        self.lca_references = updated
            .lca_references
            .into_iter()
            .map(|r| (r.main_ref, r.clock_ref, r.counter_clock_ref))
            .collect();
        self.ao_rca = updated.ao_rca;
        self.ao_lca = updated.ao_lca;

        Ok(())
    }
}

impl From<DiscretizedVesselTree> for PyDiscretizedVesselTree {
    fn from(t: DiscretizedVesselTree) -> Self {
        Self {
            discretized_aorta: t.discretized_aorta.iter().map(PyContour::from).collect(),
            discretized_rca_main: t.discretized_rca_main.iter().map(PyContour::from).collect(),
            discretized_lca_main: t.discretized_lca_main.iter().map(PyContour::from).collect(),
            rca_branches: t
                .rca_branches
                .iter()
                .map(|b| b.iter().map(PyContour::from).collect())
                .collect(),
            lca_branches: t
                .lca_branches
                .iter()
                .map(|b| b.iter().map(PyContour::from).collect())
                .collect(),
            rca_references: t
                .rca_references
                .into_iter()
                .map(|r| (r.main_ref, r.clock_ref, r.counter_clock_ref))
                .collect(),
            lca_references: t
                .lca_references
                .into_iter()
                .map(|r| (r.main_ref, r.clock_ref, r.counter_clock_ref))
                .collect(),
            ao_rca: t.ao_rca,
            ao_lca: t.ao_lca,
        }
    }
}

/// Find points bounded by spheres along a coronary vessel centerline.
///
/// This version accepts and returns simple Python lists of tuples.
///
/// Parameters
/// ----------
/// centerline : PyCenterline
///     Centerline of the vessel.
/// points : list of tuple of float
///     List of ``(x, y, z)`` point coordinates.
/// radius : float
///     Radius of the bounding spheres around each centerline point.
///
/// Returns
/// -------
/// bounded_points : list of tuple of float
///     Filtered points that are inside the bounding spheres.
///
/// Examples
/// --------
/// >>> import multimodars as mm
/// >>>
/// >>> # Load centerline and point cloud
/// >>> centerline = mm.load_centerline("path/to/centerline.json")
/// >>> points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), ...]  # or mesh.vertices.tolist()
/// >>>
/// >>> # Find points bounded by centerline spheres
/// >>> bounded_points = mm.find_centerline_bounded_points(centerline, points, 2.0)
/// >>> print(f"Found {len(bounded_points)} points inside vessel bounds")
#[pyfunction]
pub fn find_centerline_bounded_points_simple(
    centerline: PyCenterline,
    points: Vec<Point3D>,
    radius: f64,
) -> PyResult<Vec<Point3D>> {
    let rust_centerline = centerline.to_rust_centerline();

    // Call Rust function directly - no complex conversions needed
    let result_points =
        label_coronary::find_centerline_bounded_points(rust_centerline, &points, radius);

    Ok(result_points)
}

/// Remove occluded points using ray-triangle intersection testing.
///
/// For each centerline point, candidate points that are occluded by the mesh
/// surface (represented as triangles) are removed via ray casting.  This is
/// intended to run as a cleanup step after bounding-sphere selection.
///
/// Parameters
/// ----------
/// centerline_coronary : PyCenterline
///     Centerline of the coronary vessel used as vantage points.
/// centerline_aorta : PyCenterline
///     Centerline of the aorta used as additional vantage points.
/// range_coronary : int
///     Number of centerline points considered around each coronary point.
/// points : list of tuple of float
///     Candidate ``(x, y, z)`` points, e.g. output of
///     :func:`find_centerline_bounded_points`.
/// faces : list of tuple of tuple of float
///     Triangle faces as ``((v0x, v0y, v0z), (v1x, v1y, v1z), (v2x, v2y, v2z))``
///     triples representing the mesh surface.
///
/// Returns
/// -------
/// filtered_points : list of tuple of float
///     Points with occluded entries removed.
#[pyfunction]
pub fn remove_occluded_points_ray_triangle(
    centerline_coronary: PyCenterline,
    centerline_aorta: PyCenterline,
    range_coronary: usize,
    points: Vec<Point3D>,
    faces: Vec<TriangleTuple>,
) -> PyResult<Vec<Point3D>> {
    let rust_centerline_coronary = centerline_coronary.to_rust_centerline();
    let rust_centerline_aorta = centerline_aorta.to_rust_centerline();

    let triangles: Vec<Triangle> = faces
        .into_iter()
        .map(|(v0, v1, v2)| Triangle::new(v0, v1, v2))
        .collect();

    let result = label_coronary::remove_occluded_points_ray_triangle_rust(
        &rust_centerline_coronary,
        &rust_centerline_aorta,
        range_coronary,
        &points,
        &triangles,
    );
    Ok(result)
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
pub fn adjust_diameter_centerline_morphing_simple(
    centerline: PyCenterline,
    points: Vec<Point3D>,
    diameter_adjustment_mm: f64,
) -> PyResult<Vec<Point3D>> {
    let rust_centerline = centerline.to_rust_centerline();

    let result_points = scale_coronary::centerline_based_diameter_morphing(
        &rust_centerline,
        &points,
        diameter_adjustment_mm,
    );

    Ok(result_points)
}

/// Find points that lie within a specified frame region along the centerline.
///
/// Parameters
/// ----------
/// centerline : PyCenterline
///     Centerline of the vessel.
/// frames : list of PyFrame
///     Frames defining the region of interest along the centerline.
/// points : list of tuple of float
///     Input ``(x, y, z)`` point coordinates.
///
/// Returns
/// -------
/// proximal_points : list of tuple of float
///     Points located before the first frame in the region.
/// distal_points : list of tuple of float
///     Points located after the last frame in the region.
/// points_between : list of tuple of float
///     Points located within the frame region.
///
/// Examples
/// --------
/// >>> import multimodars as mm
#[pyfunction]
pub fn find_points_by_cl_region(
    centerline: PyCenterline,
    frames: Vec<PyFrame>,
    points: Vec<Point3D>,
) -> PyResult<(Vec<Point3D>, Vec<Point3D>, Vec<Point3D>)> {
    let rust_centerline = centerline.to_rust_centerline();
    let rust_frames: Vec<crate::intravascular::io::geometry::Frame> = frames
        .into_iter()
        .map(|f| f.to_rust_frame())
        .collect::<Result<_, _>>()?;

    let result_points =
        scale_coronary::find_points_by_cl_region_rs(&rust_centerline, &rust_frames, &points);

    Ok(result_points)
}

/// Clean up outlier points based on neighbourhood density and reference points.
///
/// Parameters
/// ----------
/// points_to_cleanup : list of tuple of float
///     Input ``(x, y, z)`` points to be filtered.
/// reference_points : list of tuple of float
///     Reference ``(x, y, z)`` points used to absorb outliers.
/// neighborhood_radius : float
///     Radius used to determine the neighbourhood for density estimation.
/// min_neigbor_ratio : float
///     Minimum fraction of neighbours required to keep a point.
///
/// Returns
/// -------
/// cleaned_points : list of tuple of float
///     Points from ``points_to_cleanup`` with outliers removed.
/// augmented_reference : list of tuple of float
///     Reference points augmented with the removed outliers.
///
/// Examples
/// --------
/// >>> import multimodars as mm
#[pyfunction]
pub fn clean_outlier_points(
    points_to_cleanup: Vec<Point3D>,
    reference_points: Vec<Point3D>,
    neighborhood_radius: f64,
    min_neigbor_ratio: f64,
) -> PyResult<(Vec<Point3D>, Vec<Point3D>)> {
    let result_points = scale_coronary::clean_up_non_section_points(
        points_to_cleanup,
        reference_points,
        neighborhood_radius,
        min_neigbor_ratio,
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
    let (prox_dist, distal_dist) = scale_coronary::centerline_based_diameter_optimization(
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
    let dist = scale_coronary::centerline_based_aortic_diameter_optimization(
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
    let dist = scale_coronary::centerline_based_wall_diameter_optimization(
        &rust_centerline,
        &ref_pt_coronary,
        &aortic_pts,
    );
    Ok(dist)
}

/// Build a vertex adjacency map from a triangle mesh face list.
///
/// For each triangle face, all three undirected edges are recorded so that
/// every vertex maps to the set of vertices it shares an edge with.
///
/// Parameters
/// ----------
/// faces : list of list of int
///     Triangle faces, each represented as a three-element array of vertex
///     indices ``[v0, v1, v2]``.
///
/// Returns
/// -------
/// adjacency_map : dict of int to set of int
///     Mapping from each vertex index to the set of its directly connected
///     neighbour vertex indices.
///
/// Examples
/// --------
/// >>> import multimodars as mm
/// >>>
/// >>> faces = [[0, 1, 2], [1, 2, 3]]
/// >>> adj = mm.build_adjacency_map(faces)
/// >>> print(adj[1])  # {0, 2, 3}
#[pyfunction]
pub fn build_adjacency_map(faces: Vec<[usize; 3]>) -> HashMap<usize, HashSet<usize>> {
    let mut adjacency: HashMap<usize, HashSet<usize>> = HashMap::new();

    for face in faces {
        let v0 = face[0];
        let v1 = face[1];
        let v2 = face[2];

        // Add connections for all three edges of the triangle
        let edges = [(v0, v1), (v1, v2), (v2, v0)];

        for &(a, b) in &edges {
            adjacency.entry(a).or_default().insert(b);
            adjacency.entry(b).or_default().insert(a);
        }
    }

    adjacency
}

/// Discretize a coronary vessel into uniform cross-sectional contours.
///
/// Walks ``branch_id`` of ``centerline`` at uniform arc-length intervals of
/// ``step_size``, projects the supplied mesh ``points`` onto the perpendicular
/// plane at each position, filters out empty and incomplete (half-circle) slices,
/// and resamples each surviving slice to exactly ``n_points`` evenly-spaced
/// points via a closed Catmull-Rom spline.
///
/// Parameters
/// ----------
/// centerline : PyCenterline
///     Centerline of the vessel to discretize.
/// points : list of tuple of float
///     ``(x, y, z)`` surface point cloud of the vessel (e.g. mesh vertices).
/// branch_id : int
///     Branch of the centerline to walk (0 = main vessel, 1+ = side branches).
/// step_size : float
///     Arc-length step between successive cross-sections in mm.  The slab
///     half-thickness used for point selection is ``step_size / 2``.
/// n_points : int
///     Number of evenly-spaced points on each output contour.
///
/// Returns
/// -------
/// contours : list of PyContour
///     One uniformly-sampled closed contour per valid cross-section.
///
/// Examples
/// --------
/// >>> import multimodars as mm
/// >>> centerline = mm.load_centerline("vessel.json")
/// >>> vertices = mesh.vertices.tolist()
/// >>> contours = mm.discretize_vessel(centerline, vertices, 0, 0.5, 200)
/// >>> print(f"Got {len(contours)} cross-sections")
#[pyfunction]
pub fn discretize_vessel(
    centerline: PyCenterline,
    points: Vec<Point3D>,
    branch_id: u32,
    step_size: f64,
    n_points: usize,
) -> PyResult<Vec<PyContour>> {
    let rust_centerline = centerline.to_rust_centerline();
    let contours = discretizing::discretize_vessel_rs(
        &rust_centerline,
        &points,
        branch_id,
        step_size,
        n_points,
    );
    Ok(contours.iter().map(PyContour::from).collect())
}

/// Smooth per-vertex mesh labels by majority voting over neighbours.
///
/// In each iteration every vertex whose label differs from the unanimous
/// majority of its neighbours is reassigned to that majority label.  Only
/// flips where *all* neighbours agree are applied, making the smoothing
/// conservative.
///
/// Parameters
/// ----------
/// labels : list of int
///     Per-vertex label values (``u8``), indexed by vertex index.
/// adjacency_map : dict of int to set of int
///     Vertex adjacency map as returned by :func:`build_adjacency_map`.
/// iterations : int
///     Number of smoothing passes to perform.
///
/// Returns
/// -------
/// smoothed_labels : list of int
///     Updated per-vertex label values after smoothing.
///
/// Examples
/// --------
/// >>> import multimodars as mm
/// >>>
/// >>> faces = [[0, 1, 2], [1, 2, 3]]
/// >>> adj = mm.build_adjacency_map(faces)
/// >>> labels = [0, 1, 0, 0]
/// >>> smoothed = mm.smooth_mesh_labels(labels, adj, iterations=3)
#[pyfunction]
pub fn smooth_mesh_labels(
    labels: Vec<u8>,
    adjacency_map: HashMap<usize, HashSet<usize>>,
    iterations: usize,
) -> Vec<u8> {
    let mut current_labels = labels;
    let n = current_labels.len();

    for _ in 0..iterations {
        let mut next_labels = current_labels.clone();

        for i in 0..n {
            if let Some(neighbors) = adjacency_map.get(&i) {
                if neighbors.is_empty() {
                    continue;
                }

                let my_label = current_labels[i];

                // Count occurrences of neighbor labels
                let mut counts = HashMap::new();
                for &neighbor_idx in neighbors {
                    let label = current_labels[neighbor_idx];
                    *counts.entry(label).or_insert(0) += 1;
                }

                // Find the most frequent label among neighbors
                let (&majority_label, &max_count) =
                    counts.iter().max_by_key(|&(_, count)| count).unwrap();

                // If I am different from the majority and the majority is strong
                // (You can adjust this logic: e.g., only flip if 100% of neighbors are different)
                if max_count == neighbors.len() && my_label != majority_label {
                    next_labels[i] = majority_label;
                }
            }
        }
        current_labels = next_labels;
    }

    current_labels
}

/// Discretize the full coronary vessel tree and compute orientation references.
///
/// Runs :func:`discretize_vessel` for every branch (aorta, RCA main, LCA main,
/// and each side branch), smoothes all centerlines with a Gaussian kernel
/// (σ = 2.5 points) beforehand, and then computes orientation reference triplets
/// at the ostium and every side-branch bifurcation.
///
/// Parameters
/// ----------
/// ao_cl : PyCenterline
///     Aortic centerline (branch 0 only).
/// rca_cl : PyCenterline
///     RCA centerline with all branches calculated.
/// lca_cl : PyCenterline
///     LCA centerline with all branches calculated.
/// points_ao : list of tuple of float
///     Surface mesh points ``(x, y, z)`` of the aorta.
/// points_rca_main : list of tuple of float
///     Surface mesh points for the RCA main vessel.
/// points_lca_main : list of tuple of float
///     Surface mesh points for the LCA main vessel.
/// side_branches_rca : list of list of tuple of float
///     One point list per RCA side branch, ordered by branch_id
///     (``side_branches_rca[0]`` → branch_id 1, etc.).
///     Pass ``results["rca_points_side_1"]``, ``["rca_points_side_2"]``, … in order.
/// side_branches_lca : list of list of tuple of float
///     Same structure for LCA.
/// branch_id_rca : int
///     Branch ID of the RCA main vessel (almost always ``0``).
/// branch_id_lca : int
///     Branch ID of the LCA main vessel (almost always ``0``).
/// step_size : float
///     Arc-length step between cross-sections in mm.
/// n_points : int
///     Number of evenly-spaced points per output contour.
///
/// Returns
/// -------
/// PyDiscretizedVesselTree
///     Fully populated vessel tree including orientation references.
///
/// Examples
/// --------
/// >>> import multimodars as mm
/// >>> results = mm.label_branches(rca_cl, results)
/// >>> results = mm.label_branches(lca_cl, results, results_key="lca_points")
/// >>> side_rca = [results["rca_points_side_1"], results["rca_points_side_2"]]
/// >>> side_lca = [results["lca_points_side_1"]]
/// >>> tree = mm.discretize_vessel_tree(
/// ...     ao_cl, rca_cl, lca_cl,
/// ...     results["aorta_points"],
/// ...     results["rca_points_main"],
/// ...     results["lca_points_main"],
/// ...     side_rca, side_lca,
/// ...     branch_id_rca=0, branch_id_lca=0,
/// ...     step_size=1.0, n_points=100,
/// ... )
#[pyfunction]
#[pyo3(signature = (
    ao_cl, rca_cl, lca_cl,
    points_ao, points_rca_main, points_lca_main,
    side_branches_rca, side_branches_lca,
    branch_id_rca = 0, branch_id_lca = 0,
    step_size = 1.0, n_points = 100,
    calculate_ref_pts=true,
))]
pub fn discretize_vessel_tree(
    ao_cl: PyCenterline,
    rca_cl: PyCenterline,
    lca_cl: PyCenterline,
    points_ao: Vec<Point3D>,
    points_rca_main: Vec<Point3D>,
    points_lca_main: Vec<Point3D>,
    side_branches_rca: Vec<Vec<Point3D>>,
    side_branches_lca: Vec<Vec<Point3D>>,
    branch_id_rca: u32,
    branch_id_lca: u32,
    step_size: f64,
    n_points: usize,
    calculate_ref_pts: bool,
) -> PyResult<PyDiscretizedVesselTree> {
    let mut tree = DiscretizedVesselTree::from_results_dict(
        &ao_cl.to_rust_centerline(),
        &rca_cl.to_rust_centerline(),
        &lca_cl.to_rust_centerline(),
        &points_ao,
        &points_rca_main,
        &points_lca_main,
        side_branches_rca,
        side_branches_lca,
        branch_id_rca,
        branch_id_lca,
        step_size,
        n_points,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    tree = if calculate_ref_pts {
        tree.calculate_ref_pts()
    } else {
        tree
    };

    Ok(PyDiscretizedVesselTree::from(tree))
}
