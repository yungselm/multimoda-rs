use crate::intravascular::io::input::{CenterlinePoint, Centerline};

pub fn centerline_based_diameter_morphing(
    centerline: &Centerline,
    points: &[(f64, f64, f64)],
    diameter_adjustment_mm: f64,
) -> Vec<(f64, f64, f64)> {
    let mut result_points = Vec::with_capacity(points.len());

    for point in points.iter() {
        let closest_cl_point = find_closest_centerline_point_optimized(centerline, *point);

        let vector = (
            point.0 - closest_cl_point.contour_point.x,
            point.1 - closest_cl_point.contour_point.y,
            point.2 - closest_cl_point.contour_point.z,
        );

        let magnitude = (vector.0 * vector.0 + vector.1 * vector.1 + vector.2 * vector.2).sqrt();

        if magnitude > 0.0 {
            let normalized_vector = (
                vector.0 / magnitude,
                vector.1 / magnitude,
                vector.2 / magnitude,
            );
            
            let new_point = (
                point.0 + normalized_vector.0 * diameter_adjustment_mm,
                point.1 + normalized_vector.1 * diameter_adjustment_mm,
                point.2 + normalized_vector.2 * diameter_adjustment_mm,
            );
            
            result_points.push(new_point);
        } else {
            // If point is exactly on centerline, we can't determine direction
            // Just keep the original point
            result_points.push(*point);
        }
    }
    
    result_points
}

fn find_closest_centerline_point_optimized(centerline: &Centerline, point: (f64, f64, f64)) -> &CenterlinePoint {
    let mut min_distance_squared = f64::MAX;
    let mut closest_point = &centerline.points[0];
    
    for centerline_point in &centerline.points {
        let distance_squared = calculate_squared_distance(point, centerline_point);
        if distance_squared < min_distance_squared {
            min_distance_squared = distance_squared;
            closest_point = centerline_point;
        }
    }
    
    closest_point
}

fn calculate_squared_distance(point: (f64, f64, f64), centerline_point: &CenterlinePoint) -> f64 {
    let dx = point.0 - centerline_point.contour_point.x;
    let dy = point.1 - centerline_point.contour_point.y;
    let dz = point.2 - centerline_point.contour_point.z;
    dx * dx + dy * dy + dz * dz
}

#[cfg(test)]
mod tests {
    use crate::intravascular::io::input::{CenterlinePoint, ContourPoint};
    use nalgebra::Vector3;

    use super::*;

    #[test]
    fn test_centerline_based_diameter_morphing() {
        let centerline = Centerline {
            points: vec![
                CenterlinePoint { 
                    contour_point: ContourPoint {
                        frame_index: 0, 
                        point_index: 0, 
                        x: 0.0, 
                        y: 0.0, 
                        z: 0.0, 
                        aortic: false
                    },
                    normal: Vector3::new(1.0, 0.0, 0.0).into()
                },
                CenterlinePoint { 
                    contour_point: ContourPoint {
                        frame_index: 1, 
                        point_index: 1, 
                        x: 1.0, 
                        y: 0.0, 
                        z: 0.0, 
                        aortic: false
                    },
                    normal: Vector3::new(1.0, 0.0, 0.0).into()
                },
            ],
        };

        let points = vec![
            (1.0, 1.0, 0.0),  // Point at (1,1,0) - closest to centerline point (1,0,0)
        ];

        let result = centerline_based_diameter_morphing(&centerline, &points, 1.0);
        
        // The point should move from (1,1,0) to (1,2,0) - same direction but 1 unit further
        assert_eq!(result.len(), 1);
        let new_point = result[0];
        assert!((new_point.0 - 1.0).abs() < 1e-6);
        assert!((new_point.1 - 2.0).abs() < 1e-6);
        assert!((new_point.2 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_negative_adjustment() {
        let centerline = Centerline {
            points: vec![
                CenterlinePoint { 
                    contour_point: ContourPoint {
                        frame_index: 0, 
                        point_index: 0, 
                        x: 0.0, 
                        y: 0.0, 
                        z: 0.0, 
                        aortic: false
                    },
                    normal: Vector3::new(1.0, 0.0, 0.0).into()
                },
            ],
        };

        let points = vec![
            (2.0, 0.0, 0.0),  // Point at (2,0,0)
        ];

        let result = centerline_based_diameter_morphing(&centerline, &points, -0.5);
        
        // Should move toward centerline by 0.5 units
        let new_point = result[0];
        assert!((new_point.0 - 1.5).abs() < 1e-6);
        assert!((new_point.1 - 0.0).abs() < 1e-6);
        assert!((new_point.2 - 0.0).abs() < 1e-6);
    }
}