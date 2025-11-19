use crate::intravascular::io::input::{CenterlinePoint, Centerline};

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
            let dists: Vec<f64> = points_inside.iter().map(|p| {
                let dx = p.0 - bounding_sphere.center.0;
                let dy = p.1 - bounding_sphere.center.1;
                let dz = p.2 - bounding_sphere.center.2;
                (dx*dx + dy*dy + dz*dz).sqrt()
            }).collect();

            let mean = dists.iter().sum::<f64>() / dists.len() as f64;
            let var = dists.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / dists.len() as f64;
            let std = var.sqrt();

            if std == 0.0 {
                points_inside
            } else {
                points_inside.into_iter()
                    .zip(dists.into_iter())
                    .filter(|(_p, d)| (d - mean).abs() <= 2.0 * std)
                    .map(|(p, _d)| p)
                    .collect()
            }
        };

        all_points_inside.extend(filtered_local);
    }

    all_points_inside.sort_by(|a, b| {
        a.0.partial_cmp(&b.0).unwrap()
            .then(a.1.partial_cmp(&b.1).unwrap())
            .then(a.2.partial_cmp(&b.2).unwrap())
    });

    const EPS: f64 = 1e-6;
    all_points_inside.dedup_by(|a, b| {
        (a.0 - b.0).abs() < EPS && (a.1 - b.1).abs() < EPS && (a.2 - b.2).abs() < EPS
    });

    all_points_inside
}

fn check_centerline(centerline: Centerline) -> Centerline {
    // Check that the centerline is sorted by z-value (distal to proximal)
    // and ensure the last point has the lowest z-value
    let mut points = centerline.points.clone();
    
    points.sort_by(|a, b| b.contour_point.z.partial_cmp(&a.contour_point.z).unwrap());
    
    Centerline { points }
}

fn create_bounding_sphere(cl_point: &CenterlinePoint, radius: f64) -> BoundingSphere {
    let radius = radius;
    
    BoundingSphere {
        center: (cl_point.contour_point.x, cl_point.contour_point.y, cl_point.contour_point.z),
        radius,
    }
}

fn find_points_inside_of_sphere(sphere: &BoundingSphere, point: &(f64, f64, f64)) -> bool {
    let dx = point.0 - sphere.center.0;
    let dy = point.1 - sphere.center.1;
    let dz = point.2 - sphere.center.2;
    let distance_squared = dx * dx + dy * dy + dz * dz;
    
    distance_squared <= sphere.radius * sphere.radius
}

#[derive(Debug)]
struct BoundingSphere {
    center: (f64, f64, f64),
    radius: f64,
}

#[cfg(test)]
mod test_find_cl_bounded_points {
    use super::*;
    use crate::intravascular::io::input::ContourPoint;

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
                aortic: false
            },
            ContourPoint {
                frame_index: 212,
                point_index: 1,
                x: 0.5,
                y: 0.5,
                z: 1.0,
                aortic: false
            },
            ContourPoint {
                frame_index: 3657,
                point_index: 2,
                x: 0.5,
                y: 0.5,
                z: 2.0,
                aortic: false
            }
        ];
        let cl = Centerline::from_contour_points(cl_raw_points);
        
        // Combine inside and outside points
        let all_points: Vec<(f64, f64, f64)> = points_inside.iter()
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
            assert!(result.contains(expected_point), "Missing point: {:?}", expected_point);
        }
        
        // Verify that no outside points are in the result
        for outside_point in &points_outside {
            assert!(!result.contains(outside_point), "Unexpected point: {:?}", outside_point);
        }
    }
}