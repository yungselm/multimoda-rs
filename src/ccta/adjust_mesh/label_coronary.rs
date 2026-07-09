use crate::types::native::{Centerline, CenterlinePoint};
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;
use std::collections::HashSet;

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

    let mut points_to_remove: HashSet<usize> = HashSet::new();
    const DISTANCE_THRESHOLD: f64 = 0.5; // This only works for mm, just needs to be sufficiently small

    for (point_idx, point) in points.iter().enumerate() {
        for &face_idx in &faces_to_exclude {
            if let Some(face) = faces.get(face_idx) {
                let distance = point_to_triangle_distance(point, face);
                if distance < DISTANCE_THRESHOLD {
                    points_to_remove.insert(point_idx);
                    break; // No need to check other faces for this point
                }
            }
        }
    }

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

fn point_to_triangle_distance(point: &(f64, f64, f64), triangle: &Triangle) -> f64 {
    let p = Point3::new(point.0, point.1, point.2);
    triangle
        .points()
        .iter()
        .map(|v| nalgebra::distance_squared(&p, v))
        .fold(f64::INFINITY, f64::min)
}

#[derive(Debug)]
struct BoundingSphere {
    center: Point3<f64>,
    radius: f64,
}

pub fn find_centerline_bounded_points(
    centerline: Centerline,
    points: &[(f64, f64, f64)],
    radius: f64,
) -> Vec<(f64, f64, f64)> {
    let checked_centerline = check_centerline(centerline);
    let mut all_points_inside: Vec<(f64, f64, f64)> = Vec::new();

    for cl_point in checked_centerline.points.iter().rev() {
        let bounding_sphere = create_bounding_sphere(cl_point, radius);
        let mut points_inside: Vec<(f64, f64, f64)> = Vec::new();

        for point in points.iter() {
            if find_points_inside_of_sphere(&bounding_sphere, point) {
                points_inside.push(*point);
            }
        }

        if points_inside.is_empty() {
            continue;
        }

        // Local filtering: use distances from the sphere center and discard
        // distance outliers (within 2*std). Skip filter if there are too few points.
        let filtered_local = if points_inside.len() < 3 {
            points_inside
        } else {
            let dists: Vec<f64> = points_inside
                .iter()
                .map(|p| nalgebra::distance(&Point3::new(p.0, p.1, p.2), &bounding_sphere.center))
                .collect();

            let mean = dists.iter().sum::<f64>() / dists.len() as f64;
            let var = dists.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / dists.len() as f64;
            let std = var.sqrt();

            if std == 0.0 {
                points_inside
            } else {
                points_inside
                    .into_iter()
                    .zip(dists)
                    .filter(|(_p, d)| (d - mean).abs() <= 2.0 * std)
                    .map(|(p, _d)| p)
                    .collect()
            }
        };

        all_points_inside.extend(filtered_local);
    }

    all_points_inside.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap()
            .then(a.1.partial_cmp(&b.1).unwrap())
            .then(a.2.partial_cmp(&b.2).unwrap())
    });

    const EPS: f64 = 1e-6;
    all_points_inside.dedup_by(|a, b| {
        (a.0 - b.0).abs() < EPS && (a.1 - b.1).abs() < EPS && (a.2 - b.2).abs() < EPS
    });

    all_points_inside
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

fn create_bounding_sphere(cl_point: &CenterlinePoint, radius: f64) -> BoundingSphere {
    BoundingSphere {
        center: Point3::new(
            cl_point.contour_point.x,
            cl_point.contour_point.y,
            cl_point.contour_point.z,
        ),
        radius,
    }
}

fn find_points_inside_of_sphere(sphere: &BoundingSphere, point: &(f64, f64, f64)) -> bool {
    let p = Point3::new(point.0, point.1, point.2);
    nalgebra::distance_squared(&p, &sphere.center) <= sphere.radius * sphere.radius
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
}
