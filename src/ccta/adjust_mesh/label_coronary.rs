use crate::ccta::binding::ccta_py::build_adjacency_map;
use crate::types::native::Centerline;
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq)]
pub struct Triangle {
    pub v0: (f64, f64, f64),
    pub v1: (f64, f64, f64),
    pub v2: (f64, f64, f64),
}

impl Triangle {
    pub fn new(v0: (f64, f64, f64), v1: (f64, f64, f64), v2: (f64, f64, f64)) -> Self {
        Self { v0, v1, v2 }
    }

    fn points(&self) -> [Point3<f64>; 3] {
        [
            Point3::new(self.v0.0, self.v0.1, self.v0.2),
            Point3::new(self.v1.0, self.v1.1, self.v1.2),
            Point3::new(self.v2.0, self.v2.1, self.v2.2),
        ]
    }
}

// Ray-Triangle intersection using Möller–Trumbore algorithm
fn ray_triangle_intersection(
    ray_origin: &Point3<f64>,
    ray_direction: &Vector3<f64>,
    triangle: &Triangle,
) -> Option<f64> {
    let eps = 1e-8;

    let [v0, v1, v2] = triangle.points();
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;

    let h = ray_direction.cross(&edge2);
    let a = edge1.dot(&h);
    if a.abs() < eps {
        return None; // Ray is parallel to triangle
    }

    let f = 1.0 / a;
    let s = ray_origin - v0;

    let u = f * s.dot(&h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let q = s.cross(&edge1);

    let v = f * ray_direction.dot(&q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    // Compute t to find intersection point
    let t = f * edge2.dot(&q);
    if t > eps {
        Some(t)
    } else {
        None
    }
}

pub fn remove_occluded_points_ray_triangle_rust(
    centerline_coronary: &Centerline,
    centerline_aorta: &Centerline,
    range_coronary: usize,
    points: &[(f64, f64, f64)],
    faces: &[Triangle],
    step_size_mm: f64,
) -> Vec<(f64, f64, f64)> {
    if points.is_empty() || faces.is_empty() {
        return points.to_vec();
    }

    let checked_cl_coronary = check_centerline(centerline_coronary.clone());
    let checked_cl_aorta = check_centerline(centerline_aorta.clone());
    let spacing = (centerline_aorta.mean_spacing() + centerline_coronary.mean_spacing()) / 2.0;
    let step_cl_points = (step_size_mm / spacing).ceil() as usize;

    // Parallelize over aorta points (75 items): each thread owns 100 sequential coronary
    // iterations against faces — coarse enough to avoid scheduler overhead from nested parallelism.
    let faces_to_exclude: HashSet<usize> = checked_cl_aorta
        .points
        .par_iter()
        .flat_map_iter(|aorta_point| {
            let aorta_coord = Point3::new(
                aorta_point.contour_point.x,
                aorta_point.contour_point.y,
                aorta_point.contour_point.z,
            );

            let mut local_excluded: Vec<usize> = Vec::new();

            for coronary_point in checked_cl_coronary
                .points
                .iter()
                .take(range_coronary)
                .step_by(step_cl_points)
            {
                let coronary_coord = Point3::new(
                    coronary_point.contour_point.x,
                    coronary_point.contour_point.y,
                    coronary_point.contour_point.z,
                );

                let ray_direction = coronary_coord - aorta_coord;

                let mut intersecting_faces: Vec<(usize, f64)> = faces
                    .iter()
                    .enumerate()
                    .filter_map(|(face_idx, face)| {
                        ray_triangle_intersection(&aorta_coord, &ray_direction, face)
                            .map(|t| (face_idx, t))
                    })
                    .collect();

                intersecting_faces.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                if intersecting_faces.len() >= 3 {
                    if let Some((closest_face_idx, _)) = intersecting_faces.first() {
                        local_excluded.push(*closest_face_idx);
                    }
                }
            }

            local_excluded
        })
        .collect();

    println!("Total faces to exclude: {}", faces_to_exclude.len());

    const DISTANCE_THRESHOLD: f64 = 0.5; // This only works for mm, just needs to be sufficiently small

    // R-tree over the vertices of only the excluded faces (small, typically a few hundred):
    // each mesh point then costs O(log k) instead of a linear scan over every excluded face,
    // so this pass is O(points * log k) instead of O(points * faces_to_exclude). Checking "any
    // excluded face's vertex within threshold" is exactly equivalent to the original per-face
    // min-of-3-vertices check, since the vertex pool covers the same 3 vertices per face.
    // `locate_within_distance` compares squared distance, matching the pre-existing (squared)
    // semantics of `DISTANCE_THRESHOLD` here.
    let excluded_vertices: Vec<[f64; 3]> = faces_to_exclude
        .iter()
        .filter_map(|&face_idx| faces.get(face_idx))
        .flat_map(|face| {
            [
                [face.v0.0, face.v0.1, face.v0.2],
                [face.v1.0, face.v1.1, face.v1.2],
                [face.v2.0, face.v2.1, face.v2.2],
            ]
        })
        .collect();

    let points_to_remove: HashSet<usize> = if excluded_vertices.is_empty() {
        HashSet::new()
    } else {
        let excluded_tree = rstar::RTree::bulk_load(excluded_vertices);
        points
            .par_iter()
            .enumerate()
            .filter_map(|(point_idx, point)| {
                excluded_tree
                    .locate_within_distance([point.0, point.1, point.2], DISTANCE_THRESHOLD)
                    .next()
                    .is_some()
                    .then_some(point_idx)
            })
            .collect()
    };

    let filtered_points: Vec<(f64, f64, f64)> = points
        .iter()
        .enumerate()
        .filter(|(idx, _)| !points_to_remove.contains(idx))
        .map(|(_, point)| *point)
        .collect();

    println!(
        "Excluded {} faces, removed {} points (filtered from {} to {} points)",
        faces_to_exclude.len(),
        points_to_remove.len(),
        points.len(),
        filtered_points.len()
    );

    filtered_points
}

pub fn find_centerline_bounded_points(
    centerline: Centerline,
    points: &[(f64, f64, f64)],
    radius: f64,
) -> Vec<(f64, f64, f64)> {
    let checked_centerline = check_centerline(centerline);
    if points.is_empty() || checked_centerline.points.is_empty() {
        return Vec::new();
    }

    // R-tree over the centerline points (the small side, typically ~1000 points):
    // each mesh point then costs O(log M) instead of a scan over all M centerline
    // points, so the whole query is O(N log M) instead of O(N * M).
    let cl_coords: Vec<[f64; 3]> = checked_centerline
        .points
        .iter()
        .map(|p| [p.contour_point.x, p.contour_point.y, p.contour_point.z])
        .collect();
    let tree = rstar::RTree::bulk_load(cl_coords);

    let radius_sq = radius * radius;
    points
        .iter()
        .filter(|p| {
            tree.locate_within_distance([p.0, p.1, p.2], radius_sq)
                .next()
                .is_some()
        })
        .copied()
        .collect()
}

/// Find mesh faces that reference any vertex coincident (within `tol`) with one of
/// `points`. Replaces the old pure-Python `_find_faces_for_points` +
/// `_prepare_faces_for_rust`, which scanned every mesh vertex per point
/// (O(points * n_vertices)) and then every mesh face (O(n_faces)) in Python. Here an
/// R-tree over `vertices` brings the per-point lookup to O(log n_vertices), and the
/// face scan stays O(n_faces) but runs as a single Rust pass instead of a Python loop.
pub fn find_faces_near_points(
    vertices: &[(f64, f64, f64)],
    faces: &[[usize; 3]],
    points: &[(f64, f64, f64)],
    tol: f64,
) -> Vec<Triangle> {
    if points.is_empty() || vertices.is_empty() || faces.is_empty() {
        return Vec::new();
    }

    let tagged_vertices: Vec<rstar::primitives::GeomWithData<[f64; 3], usize>> = vertices
        .iter()
        .enumerate()
        .map(|(idx, v)| rstar::primitives::GeomWithData::new([v.0, v.1, v.2], idx))
        .collect();
    let vertex_tree = rstar::RTree::bulk_load(tagged_vertices);

    let tol_sq = tol * tol;
    // Every vertex within `tol` is matched (not just the single nearest), which is a
    // slightly more thorough than the original "closest vertex only" Python logic —
    // relevant only if the mesh has coincident/duplicate vertices, in which case this
    // correctly includes faces touching every one of them instead of just one.
    let matched_vertex_indices: HashSet<usize> = points
        .par_iter()
        .flat_map_iter(|p| {
            vertex_tree
                .locate_within_distance([p.0, p.1, p.2], tol_sq)
                .map(|item| item.data)
        })
        .collect();

    if matched_vertex_indices.is_empty() {
        return Vec::new();
    }

    faces
        .iter()
        .filter(|[a, b, c]| {
            matched_vertex_indices.contains(a)
                || matched_vertex_indices.contains(b)
                || matched_vertex_indices.contains(c)
        })
        .map(|&[a, b, c]| Triangle::new(vertices[a], vertices[b], vertices[c]))
        .collect()
}

/// Exact-match key for a mesh-vertex coordinate. Vertex-derived point lists passed
/// around this pipeline (e.g. `rca_points`, `lca_points`) are always bit-identical
/// copies of the originating `vertices` entries (no arithmetic in between), so an
/// exact bit-pattern key is the correct tool here — unlike a radius/nearest-neighbor
/// query, this is a plain exact-membership test, so no spatial index is needed.
fn bits_key(p: &(f64, f64, f64)) -> (u64, u64, u64) {
    (p.0.to_bits(), p.1.to_bits(), p.2.to_bits())
}

/// Vertices present in neither `points_a` nor `points_b` (exact-match set
/// difference). Replaces the old pure-Python `_find_aortic_points`, which did the
/// same set-based filtering but paid per-vertex Python-loop overhead.
pub fn find_aortic_points(
    vertices: &[(f64, f64, f64)],
    points_a: &[(f64, f64, f64)],
    points_b: &[(f64, f64, f64)],
) -> Vec<(f64, f64, f64)> {
    let mut excluded: HashSet<(u64, u64, u64)> =
        HashSet::with_capacity(points_a.len() + points_b.len());
    excluded.extend(points_a.iter().map(bits_key));
    excluded.extend(points_b.iter().map(bits_key));

    vertices
        .iter()
        .filter(|v| !excluded.contains(&bits_key(v)))
        .copied()
        .collect()
}

/// Output of [`final_reclassification`]: the five vessel-region point lists after
/// adjacency-based label smoothing.
pub struct ReclassifiedLabels {
    pub aorta_points: Vec<(f64, f64, f64)>,
    pub rca_points: Vec<(f64, f64, f64)>,
    pub lca_points: Vec<(f64, f64, f64)>,
    pub rca_removed_points: Vec<(f64, f64, f64)>,
    pub lca_removed_points: Vec<(f64, f64, f64)>,
}

/// Refine vertex labels using a mesh adjacency map. Replaces the old pure-Python
/// `_final_reclassification`, which did the same per-vertex adjacency traversal but
/// paid Python-loop overhead for every one of up to tens of thousands of vertices.
///
/// Applies two adjacency-based correction rules, identical to the original:
/// * Logic A - an isolated RCA/LCA vertex (no same-label neighbours) is reassigned
///   to the aorta class.
/// * Logic B - a vertex removed by occlusion detection but whose neighbours are
///   predominantly (> 70%) the corresponding coronary label is restored to that
///   label.
pub fn final_reclassification(
    vertices: &[(f64, f64, f64)],
    faces: &[[usize; 3]],
    rca_points: &[(f64, f64, f64)],
    lca_points: &[(f64, f64, f64)],
    rca_removed_points: &[(f64, f64, f64)],
    lca_removed_points: &[(f64, f64, f64)],
) -> ReclassifiedLabels {
    let n_vertices = vertices.len();

    // Forward insertion so a duplicate coordinate keeps the *last* matching index,
    // mirroring Python's `{tuple(coord): i for i, coord in enumerate(...)}`.
    let mut coord_to_idx: HashMap<(u64, u64, u64), usize> = HashMap::with_capacity(n_vertices);
    for (i, v) in vertices.iter().enumerate() {
        coord_to_idx.insert(bits_key(v), i);
    }

    let mut labels: Vec<u8> = vec![0; n_vertices];
    for pt in rca_points {
        if let Some(&idx) = coord_to_idx.get(&bits_key(pt)) {
            labels[idx] = 1;
        }
    }
    for pt in lca_points {
        if let Some(&idx) = coord_to_idx.get(&bits_key(pt)) {
            labels[idx] = 2;
        }
    }
    for pt in rca_removed_points {
        if let Some(&idx) = coord_to_idx.get(&bits_key(pt)) {
            labels[idx] = 3;
        }
    }
    for pt in lca_removed_points {
        if let Some(&idx) = coord_to_idx.get(&bits_key(pt)) {
            labels[idx] = 4;
        }
    }

    let adjacency = build_adjacency_map(faces.to_vec());

    let mut new_labels = labels.clone();
    for i in 0..n_vertices {
        let Some(neighbors) = adjacency.get(&i) else {
            continue;
        };
        if neighbors.is_empty() {
            continue;
        }

        let current_label = labels[i];
        let neighbor_labels: Vec<u8> = neighbors.iter().map(|&n| labels[n]).collect();

        match current_label {
            // LOGIC A: isolated RCA/LCA -> aorta
            1 if !neighbor_labels.contains(&1) => new_labels[i] = 0,
            2 if !neighbor_labels.contains(&2) => new_labels[i] = 0,
            // LOGIC B: removed RCA/LCA point with >70% same-label neighbours -> restored
            3 => {
                let rca_neighbors = neighbor_labels.iter().filter(|&&l| l == 1).count();
                if (rca_neighbors as f64) > (neighbors.len() as f64 * 0.7) {
                    new_labels[i] = 1;
                }
            }
            4 => {
                let lca_neighbors = neighbor_labels.iter().filter(|&&l| l == 2).count();
                if (lca_neighbors as f64) > (neighbors.len() as f64 * 0.7) {
                    new_labels[i] = 2;
                }
            }
            _ => {}
        }
    }

    let mut result = ReclassifiedLabels {
        aorta_points: Vec::new(),
        rca_points: Vec::new(),
        lca_points: Vec::new(),
        rca_removed_points: Vec::new(),
        lca_removed_points: Vec::new(),
    };
    for (i, &label) in new_labels.iter().enumerate() {
        match label {
            0 => result.aorta_points.push(vertices[i]),
            1 => result.rca_points.push(vertices[i]),
            2 => result.lca_points.push(vertices[i]),
            3 => result.rca_removed_points.push(vertices[i]),
            4 => result.lca_removed_points.push(vertices[i]),
            _ => unreachable!(),
        }
    }
    result
}

/// Check that the centerline is sorted by z-value (distal to proximal)
/// and ensure the last point has the lowest z-value
fn check_centerline(centerline: Centerline) -> Centerline {
    let mut points = centerline.points.clone();

    points.sort_by(|a, b| b.contour_point.z.partial_cmp(&a.contour_point.z).unwrap());

    let branch_start_indices = if points.is_empty() { vec![] } else { vec![0] };
    Centerline {
        points,
        branch_start_indices,
    }
}

#[cfg(test)]
mod test_find_cl_bounded_points {
    use super::*;
    use crate::types::native::ContourPoint;

    #[test]
    fn test_find_points_simple_geometry() {
        let points_inside = vec![
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.5, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.5, 1.0, 1.0),
            (0.0, 0.0, 2.0),
            (1.0, 0.0, 2.0),
            (0.5, 1.0, 2.0),
        ];
        let points_outside = vec![
            (-1.0, -1.0, 0.5),
            (2.0, -1.0, 0.5),
            (0.5, 2.0, 0.5),
            (-1.0, -1.0, 1.5),
            (2.0, -1.0, 1.5),
            (0.5, 2.0, 1.5),
            (-1.0, -1.0, 2.5),
            (2.0, -1.0, 2.5),
            (0.5, 2.0, 2.5),
        ];
        let cl_raw_points: Vec<ContourPoint> = vec![
            ContourPoint {
                frame_index: 879,
                point_index: 0,
                x: 0.5,
                y: 0.5,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 212,
                point_index: 1,
                x: 0.5,
                y: 0.5,
                z: 1.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 3657,
                point_index: 2,
                x: 0.5,
                y: 0.5,
                z: 2.0,
                aortic: false,
            },
        ];
        let cl = Centerline::from_contour_points(cl_raw_points);

        // Combine inside and outside points
        let all_points: Vec<(f64, f64, f64)> = points_inside
            .iter()
            .chain(points_outside.iter())
            .cloned()
            .collect();

        let result = find_centerline_bounded_points(cl, &all_points, 1.0);

        // The result should contain all the points that were inside our test spheres
        // Since our spheres have radius 1.0 and are centered at (0.5, 0.5, z),
        // all points within distance 1.0 should be included
        assert_eq!(result.len(), points_inside.len());

        // Verify that all expected inside points are in the result
        for expected_point in &points_inside {
            assert!(
                result.contains(expected_point),
                "Missing point: {expected_point:?}"
            );
        }

        // Verify that no outside points are in the result
        for outside_point in &points_outside {
            assert!(
                !result.contains(outside_point),
                "Unexpected point: {outside_point:?}"
            );
        }
    }

    #[test]
    fn test_single_ray_triangle_intersection() {
        // Test a single specific ray and triangle
        let ray_origin = Point3::new(0.0, 0.0, 0.0);
        let ray_direction = Vector3::new(1.0, 0.0, 0.0); // Ray along x-axis

        // Triangle in the yz-plane at x=1.0
        let triangle = Triangle::new((1.0, -1.0, -1.0), (1.0, 1.0, -1.0), (1.0, 0.0, 1.0));

        let result = ray_triangle_intersection(&ray_origin, &ray_direction, &triangle);

        println!("=== Single Ray-Triangle Test ===");
        println!("Ray origin: {ray_origin:?}");
        println!("Ray direction: {ray_direction:?}");
        println!("Triangle: {triangle:?}");
        println!("Intersection result: {result:?}");

        assert!(result.is_some(), "Ray should intersect triangle");
        assert!(
            (result.unwrap() - 1.0).abs() < 1e-6,
            "Intersection should be at t=1.0"
        );
    }

    #[test]
    fn test_find_faces_near_points_matches_only_touching_faces() {
        let vertices = vec![
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ];
        let faces = vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];

        // Only vertex 0 matches -> faces referencing it: (0,1,2), (0,1,3), (0,2,3).
        // Face (1,2,3) doesn't touch vertex 0 and must be excluded.
        let points = vec![(0.0, 0.0, 0.0)];
        let result = find_faces_near_points(&vertices, &faces, &points, 1e-6);
        assert_eq!(result.len(), 3);
        assert!(!result.contains(&Triangle::new(vertices[1], vertices[2], vertices[3])));
        assert!(result.contains(&Triangle::new(vertices[0], vertices[1], vertices[2])));
        assert!(result.contains(&Triangle::new(vertices[0], vertices[1], vertices[3])));
        assert!(result.contains(&Triangle::new(vertices[0], vertices[2], vertices[3])));
    }

    #[test]
    fn test_find_faces_near_points_no_match_returns_empty() {
        let vertices = vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)];
        let faces = vec![[0, 1, 2]];
        let points = vec![(5.0, 5.0, 5.0)];
        let result = find_faces_near_points(&vertices, &faces, &points, 1e-6);
        assert!(result.is_empty());
    }

    #[test]
    fn test_find_aortic_points_basic_set_difference() {
        let vertices = vec![
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        ];
        let a = vec![vertices[0]];
        let b = vec![vertices[1]];
        let result = find_aortic_points(&vertices, &a, &b);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&vertices[2]));
        assert!(result.contains(&vertices[3]));
    }

    #[test]
    fn test_find_aortic_points_empty_exclusions_returns_all() {
        let vertices = vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)];
        let result = find_aortic_points(&vertices, &[], &[]);
        assert_eq!(result.len(), 2);
    }

    type GridMeshFixture = (Vec<(f64, f64, f64)>, Vec<[usize; 3]>);

    // 3x3 grid mesh (9 vertices, 8 faces, z=0 plane), mirroring
    // tests/test_ccta.py::_make_grid_mesh. Vertex 4 (centre) is adjacent to
    // {1, 2, 3, 5, 6, 7}; vertex 0 (corner) is adjacent to {1, 3}.
    fn grid_mesh_fixture() -> GridMeshFixture {
        let vertices = vec![
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (2.0, 1.0, 0.0),
            (0.0, 2.0, 0.0),
            (1.0, 2.0, 0.0),
            (2.0, 2.0, 0.0),
        ];
        let faces = vec![
            [0, 1, 3],
            [1, 4, 3],
            [1, 2, 4],
            [2, 5, 4],
            [3, 4, 6],
            [4, 7, 6],
            [4, 5, 7],
            [5, 8, 7],
        ];
        (vertices, faces)
    }

    #[test]
    fn test_final_reclassification_isolated_rca_becomes_aorta() {
        let (vertices, faces) = grid_mesh_fixture();
        // vertex 0 labelled RCA; its neighbours (1, 3) are aorta -> reclassified.
        let rca_points = vec![vertices[0]];
        let result = final_reclassification(&vertices, &faces, &rca_points, &[], &[], &[]);
        assert!(!result.rca_points.contains(&vertices[0]));
        assert!(result.aorta_points.contains(&vertices[0]));
    }

    #[test]
    fn test_final_reclassification_non_isolated_rca_stays() {
        let (vertices, faces) = grid_mesh_fixture();
        // vertex 0 and neighbour 1 are both RCA -> vertex 0 keeps its label.
        let rca_points = vec![vertices[0], vertices[1]];
        let result = final_reclassification(&vertices, &faces, &rca_points, &[], &[], &[]);
        assert!(result.rca_points.contains(&vertices[0]));
    }

    #[test]
    fn test_final_reclassification_removed_rca_restored_when_majority_rca() {
        let (vertices, faces) = grid_mesh_fixture();
        // vertex 4 is RCA_REMOVED; all 6 neighbours (1,2,3,5,6,7) are RCA (100% > 70%).
        let rca_points = vec![
            vertices[1],
            vertices[2],
            vertices[3],
            vertices[5],
            vertices[6],
            vertices[7],
        ];
        let rca_removed_points = vec![vertices[4]];
        let result = final_reclassification(
            &vertices,
            &faces,
            &rca_points,
            &[],
            &rca_removed_points,
            &[],
        );
        assert!(result.rca_points.contains(&vertices[4]));
        assert!(!result.rca_removed_points.contains(&vertices[4]));
    }

    #[test]
    fn test_final_reclassification_vertex_count_conserved() {
        let (vertices, faces) = grid_mesh_fixture();
        let rca_points = vec![vertices[0], vertices[1]];
        let lca_points = vec![vertices[2], vertices[3]];
        let result = final_reclassification(&vertices, &faces, &rca_points, &lca_points, &[], &[]);
        let total = result.aorta_points.len()
            + result.rca_points.len()
            + result.lca_points.len()
            + result.rca_removed_points.len()
            + result.lca_removed_points.len();
        assert_eq!(total, vertices.len());
    }
}
