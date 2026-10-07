pub mod anchors;
pub mod resampling;
pub mod slicing;
pub mod vessel_tree;

use crate::types::native::{Centerline, Contour};
use nalgebra::Vector3;
use std::borrow::Cow;
use std::collections::HashSet;

type Coords3 = (f64, f64, f64);

pub struct SurfaceMesh {
    pub vertices: Vec<Vector3<f64>>,
    pub faces: Vec<[usize; 3]>,
}

impl SurfaceMesh {
    /// Fails when a face references a missing vertex.
    pub fn new(vertices: &[Coords3], faces: Vec<[usize; 3]>) -> Result<Self, String> {
        if let Some(face) = faces
            .iter()
            .find(|f| f.iter().any(|&i| i >= vertices.len()))
        {
            return Err(format!(
                "face {face:?} references a vertex index >= {} (the vertex count)",
                vertices.len()
            ));
        }
        Ok(Self {
            vertices: vertices
                .iter()
                .map(|&(x, y, z)| Vector3::new(x, y, z))
                .collect(),
            faces,
        })
    }

    /// Faces with at least two vertices among `region_points` (matched by exact coordinates), so
    /// each border triangle belongs to exactly one of two adjacent regions.
    pub fn region_faces(&self, region_points: &[Coords3]) -> Vec<[usize; 3]> {
        let labelled: HashSet<[u64; 3]> = region_points
            .iter()
            .map(|&(x, y, z)| coord_key(x, y, z))
            .collect();
        let in_region: Vec<bool> = self
            .vertices
            .iter()
            .map(|v| labelled.contains(&coord_key(v.x, v.y, v.z)))
            .collect();
        self.faces
            .iter()
            .filter(|f| f.iter().filter(|&&i| in_region[i]).count() >= 2)
            .copied()
            .collect()
    }
}

/// Bit pattern of a coordinate triple. `+ 0.0` folds `-0.0` into `0.0`.
pub(crate) fn coord_key(x: f64, y: f64, z: f64) -> [u64; 3] {
    [
        (x + 0.0).to_bits(),
        (y + 0.0).to_bits(),
        (z + 0.0).to_bits(),
    ]
}

/// Cuts the mesh every `step_size` along branch `branch_id`, drops incomplete end slices and
/// resamples each outline to `n_points` evenly spaced points. Only faces of the `region_points`
/// region are cut (`None` cuts all). `centerline` must already be smoothed and resampled.
///
/// Fails on input that would otherwise silently give no contours. An empty result means the
/// input was valid but no slice was complete.
pub fn discretize_vessel_rs(
    centerline: &Centerline,
    mesh: &SurfaceMesh,
    region_points: Option<&[Coords3]>,
    branch_id: u32,
    step_size: f64,
    n_points: usize,
) -> Result<Vec<Contour>, String> {
    if !(step_size.is_finite() && step_size > 0.0) {
        return Err(format!(
            "step_size must be a positive number, got {step_size}"
        ));
    }
    if n_points < 3 {
        return Err(format!("n_points must be at least 3, got {n_points}"));
    }
    if !centerline.points.iter().any(|p| p.branch_id == branch_id) {
        return Err(format!("centerline has no branch {branch_id}"));
    }
    let faces = match region_points {
        Some(points) => {
            let faces = mesh.region_faces(points);
            if faces.is_empty() {
                return Err(format!(
                    "region_points ({} points) select no mesh faces, are the labels from this mesh?",
                    points.len()
                ));
            }
            Cow::Owned(faces)
        }
        None => Cow::Borrowed(&mesh.faces),
    };

    let anchors = anchors::branch_anchors(centerline, branch_id, step_size);
    let raw = slicing::slice_mesh(&anchors, &mesh.vertices, &faces);
    Ok(resampling::create_uniform_contours(raw, n_points))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::native::{CenterlinePoint, ContourPoint, DiscretizedVesselTree};
    use std::f64::consts::TAU;

    fn z_centerline(n: usize) -> Centerline {
        Centerline {
            points: (0..n)
                .map(|i| CenterlinePoint {
                    contour_point: ContourPoint {
                        frame_index: i as u32,
                        point_index: i as u32,
                        x: 0.0,
                        y: 0.0,
                        z: i as f64,
                        aortic: false,
                    },
                    tangent: Vector3::z(),
                    branch_id: 0,
                    radius: 0.0,
                })
                .collect(),
            branch_start_indices: vec![0],
        }
    }

    /// Circular tube of radius 2 along z with `m` vertices per ring, rings every 0.4 from -1 to 7.
    fn tube_mesh(m: usize) -> (Vec<Coords3>, Vec<[usize; 3]>) {
        let rings = 21;
        let vertices: Vec<Coords3> = (0..rings)
            .flat_map(|r| {
                (0..m).map(move |i| {
                    let a = TAU * i as f64 / m as f64;
                    (2.0 * a.cos(), 2.0 * a.sin(), -1.0 + 0.4 * r as f64)
                })
            })
            .collect();
        let mut faces = Vec::new();
        for r in 0..rings - 1 {
            for i in 0..m {
                let (a, b) = (r * m + i, r * m + (i + 1) % m);
                faces.push([a, b, b + m]);
                faces.push([a, b + m, a + m]);
            }
        }
        (vertices, faces)
    }

    #[test]
    fn test_invalid_face_index_is_rejected() {
        let vertices = vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)];
        assert!(SurfaceMesh::new(&vertices, vec![[0, 1, 2]]).is_ok());
        assert!(SurfaceMesh::new(&vertices, vec![[0, 1, 3]]).is_err());
    }

    #[test]
    fn test_region_faces_need_two_labelled_vertices() {
        let vertices = vec![
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (2.0, 1.0, 0.0),
        ];
        let mesh = SurfaceMesh::new(&vertices, vec![[0, 1, 2], [1, 3, 2], [1, 4, 3]]).unwrap();
        // Labelled: 0, 1 and -0.0-signed copy of vertex 2 → faces with ≥ 2 labelled vertices.
        let region = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-0.0, 1.0, 0.0)];
        assert_eq!(mesh.region_faces(&region), vec![[0, 1, 2], [1, 3, 2]]);
    }

    #[test]
    fn test_discretize_tube_end_to_end() {
        let (vertices, faces) = tube_mesh(30);
        let mesh = SurfaceMesh::new(&vertices, faces).unwrap();
        let contours = discretize_vessel_rs(&z_centerline(7), &mesh, None, 0, 1.0, 50).unwrap();
        assert_eq!(contours.len(), 7);
        for c in &contours {
            assert_eq!(c.points.len(), 50);
            let (cx, cy, cz) = c.centroid.unwrap();
            for p in &c.points {
                assert!((p.z - cz).abs() < 1e-9);
                assert!(((p.x - cx).hypot(p.y - cy) - 2.0).abs() < 0.02);
            }
        }
    }

    #[test]
    fn test_discretize_only_cuts_labelled_region() {
        // Label only the rings with z <= 3: slices beyond that are empty and dropped.
        let (vertices, faces) = tube_mesh(30);
        let region: Vec<Coords3> = vertices.iter().copied().filter(|v| v.2 <= 3.0).collect();
        let mesh = SurfaceMesh::new(&vertices, faces).unwrap();
        let contours =
            discretize_vessel_rs(&z_centerline(7), &mesh, Some(&region), 0, 1.0, 50).unwrap();
        let max_z = contours
            .iter()
            .map(|c| c.centroid.unwrap().2)
            .fold(f64::MIN, f64::max);
        assert!(max_z <= 3.0, "slice at z={max_z} lies outside the region");
        assert!(contours.len() >= 3);
    }

    #[test]
    fn test_invalid_input_is_rejected() {
        let (vertices, faces) = tube_mesh(30);
        let mesh = SurfaceMesh::new(&vertices, faces).unwrap();
        let cl = z_centerline(7);
        let run = |region: Option<&[Coords3]>, branch_id, step, n| {
            discretize_vessel_rs(&cl, &mesh, region, branch_id, step, n)
        };
        assert!(run(None, 0, 0.0, 50).is_err());
        assert!(run(None, 0, f64::NAN, 50).is_err());
        assert!(run(None, 0, 1.0, 2).is_err());
        assert!(run(None, 5, 1.0, 50).is_err());
        assert!(run(Some(&[(99.0, 99.0, 99.0)]), 0, 1.0, 50).is_err());
        assert!(run(Some(&[]), 0, 1.0, 50).is_err());
        assert!(run(None, 0, 1.0, 50).is_ok());
    }

    #[test]
    fn test_tree_skips_failing_side_branch_but_not_main() {
        let (vertices, faces) = tube_mesh(30);
        let mesh = SurfaceMesh::new(&vertices, faces).unwrap();
        let cl = z_centerline(7);
        let bad = vec![(99.0, 99.0, 99.0)];
        let build = |main: &[Coords3], side: Vec<Coords3>| {
            DiscretizedVesselTree::from_results_dict(
                &cl,
                &cl,
                &cl,
                &mesh,
                &vertices,
                main,
                &vertices,
                vec![side],
                vec![],
                0,
                0,
                1.0,
                50,
            )
        };

        let tree = build(&vertices, bad.clone()).unwrap();
        assert!(!tree.discretized_rca_main.is_empty());
        assert_eq!(tree.rca_branches.len(), 1);
        assert!(tree.rca_branches[0].is_empty());

        let err = build(&bad, vec![]).unwrap_err().to_string();
        assert!(err.starts_with("RCA main:"), "{err}");
    }
}
