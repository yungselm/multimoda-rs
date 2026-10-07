pub mod anchors;
pub mod resampling;
pub mod slicing;
pub mod vessel_tree;

use crate::types::native::{Centerline, Contour};
use nalgebra::Vector3;
use std::collections::HashSet;

type Coords3 = (f64, f64, f64);

pub struct SurfaceMesh {
    pub vertices: Vec<Vector3<f64>>,
    pub faces: Vec<[usize; 3]>,
}

impl SurfaceMesh {
    /// Fails when a face references a vertex index that does not exist.
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

    /// Faces belonging to the region labelled by `region_points`: those with at least two of their
    /// three vertices among the labelled points. Points are matched to vertices by exact
    /// coordinates, as label sets are copies of the mesh's own vertices.
    ///
    /// The two-of-three rule assigns each triangle on the border between two adjacent regions to
    /// exactly one of them, so neighbouring regions neither overlap nor leave a gap.
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

/// Bit pattern of a coordinate triple; `+ 0.0` folds `-0.0` into `0.0` so both match.
pub(crate) fn coord_key(x: f64, y: f64, z: f64) -> [u64; 3] {
    [
        (x + 0.0).to_bits(),
        (y + 0.0).to_bits(),
        (z + 0.0).to_bits(),
    ]
}

/// Walk `branch_id` of `centerline` at uniform `step_size` intervals, cut the mesh with the
/// plane perpendicular to the centerline at each position, filter incomplete slices, and
/// resample each surviving outline to exactly `n_points` evenly spaced points.
///
/// Only the faces of the region labelled by `region_points` are cut; `None` cuts the whole mesh.
///
/// `centerline` is used as-is — callers must smooth/resample/orient it beforehand
/// (e.g. via `Centerline::smooth`); this no longer re-smooths internally.
pub fn discretize_vessel_rs(
    centerline: &Centerline,
    mesh: &SurfaceMesh,
    region_points: Option<&[Coords3]>,
    branch_id: u32,
    step_size: f64,
    n_points: usize,
) -> Vec<Contour> {
    let anchors = anchors::branch_anchors(centerline, branch_id, step_size);
    let raw = match region_points {
        Some(points) => slicing::slice_mesh(&anchors, &mesh.vertices, &mesh.region_faces(points)),
        None => slicing::slice_mesh(&anchors, &mesh.vertices, &mesh.faces),
    };
    resampling::create_uniform_contours(raw, n_points)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::native::{CenterlinePoint, ContourPoint};
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
        let contours = discretize_vessel_rs(&z_centerline(7), &mesh, None, 0, 1.0, 50);
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
        let contours = discretize_vessel_rs(&z_centerline(7), &mesh, Some(&region), 0, 1.0, 50);
        let max_z = contours
            .iter()
            .map(|c| c.centroid.unwrap().2)
            .fold(f64::MIN, f64::max);
        assert!(max_z <= 3.0, "slice at z={max_z} lies outside the region");
        assert!(contours.len() >= 3);
    }
}
