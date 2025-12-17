use crate::intravascular::io::input::{CenterlinePoint, Centerline};
use crate::intravascular::io::geometry::Frame;

#[allow(dead_code)]
pub fn centerline_based_diameter_optimization(
    centerline: &Centerline,
    points: &[(f64, f64, f64)],
    reference_points: &[(f64, f64, f64)],
) -> f64 {
    let initial_distance = todo!("function that calcualtes distance between the two sets");
    todo!()
}

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

pub fn find_points_by_cl_region_rs(
    centerline: &Centerline,
    frames: &[Frame],
    points: &[(f64, f64, f64)],
) -> (Vec<(f64, f64, f64)>, Vec<(f64, f64, f64)>, Vec<(f64, f64, f64)>) {
    let mut cumulative_z_dist_frames = 0.0;
    for i in 1..frames.len() {
        cumulative_z_dist_frames += (frames[i].centroid.2 - frames[i-1].centroid.2).abs();
    }
    cumulative_z_dist_frames /= (frames.len() -1) as f64;

    let centroids_to_match = frames.iter().map(|f| f.centroid).collect::<Vec<(f64, f64, f64)>>();
    let cl_points_indices: Vec<usize> = find_cl_points_in_range(
        centerline,
        &centroids_to_match,
        cumulative_z_dist_frames,
    );

    // needed for proximal/distal classification
    let dist_ref = centroids_to_match[centroids_to_match.len() -1];

    let mut proximal_points: Vec<(f64, f64, f64)> = Vec::new();
    let mut distal_points: Vec<(f64, f64, f64)> = Vec::new();
    let mut points_between: Vec<(f64, f64, f64)> = Vec::new();

    let mut remaining_points = points.to_vec();
    
    // First pass: find all points between centerline regions
    remaining_points.retain(|point| {
        let closest_cl_point = find_closest_centerline_point_optimized(&centerline, *point);
        let cl_idx = closest_cl_point.contour_point.frame_index as usize;
        
        if cl_points_indices.contains(&cl_idx) {
            points_between.push(*point);
            false // remove from remaining
        } else {
            true // keep in remaining
        }
    });
    
    // Second pass: classify remaining points as proximal or distal
    for point in remaining_points.iter() {
        if point.0 > dist_ref.0 && point.1 > dist_ref.1 && point.2 > dist_ref.2 {
            proximal_points.push(*point);
        } else {
            distal_points.push(*point);
        }
    }
    let (proximal_points, points_between) = clean_up_non_section_points(proximal_points, points_between, 1.0, 0.6);
    let (distal_points, points_between) = clean_up_non_section_points(distal_points, points_between, 1.0, 0.6);
    (proximal_points, distal_points, points_between)
}

fn find_cl_points_in_range(
    centerline: &Centerline,
    points: &[(f64, f64, f64)],
    search_radius: f64,
) -> Vec<usize> {
    let mut selected_points = Vec::new();

    for point in points.iter() {
        for cl_point in centerline.points.iter() {
            let dx = point.0 - cl_point.contour_point.x;
            let dy = point.1 - cl_point.contour_point.y;
            let dz = point.2 - cl_point.contour_point.z;
            let distance_squared = dx * dx + dy * dy + dz * dz;
            if distance_squared <= search_radius * search_radius {
                selected_points.push(cl_point);
            }
        }
    }

    // remove duplicates
    selected_points.sort_by_key(|p| p.contour_point.frame_index);
    selected_points.dedup_by_key(|p| p.contour_point.frame_index);
    let mut final_points = Vec::new();
    for p in selected_points {
        final_points.push(p.contour_point.frame_index as usize);
    }
    final_points
}

pub fn clean_up_non_section_points(
    points_to_cleanup: Vec<(f64, f64, f64)>,
    reference_points: Vec<(f64, f64, f64)>,
    neighborhood_radius: f64,
    min_neigbor_ratio: f64,
) -> (Vec<(f64, f64, f64)>, Vec<(f64, f64, f64)>) {   
    let neighborhood_radius_sq = neighborhood_radius * neighborhood_radius;

    let mut cleaned_points = Vec::new();
    let mut reassigned_points = reference_points.clone();
    
    for point in points_to_cleanup.iter() {
        let mut ref_neighbors = 0;
        let mut total_neighbors = 0;
        
        for ref_point in reference_points.iter() {
            let dx = point.0 - ref_point.0;
            let dy = point.1 - ref_point.1;
            let dz = point.2 - ref_point.2;
            let distance_squared = dx * dx + dy * dy + dz * dz;
            
            if distance_squared <= neighborhood_radius_sq {
                ref_neighbors += 1;
                total_neighbors += 1;
            }
        }
        
        for other_point in points_to_cleanup.iter() {
            if std::ptr::eq(point, other_point) {
                continue; // Skip the point itself
            }
            
            let dx = point.0 - other_point.0;
            let dy = point.1 - other_point.1;
            let dz = point.2 - other_point.2;
            let distance_squared = dx * dx + dy * dy + dz * dz;
            
            if distance_squared <= neighborhood_radius_sq {
                total_neighbors += 1;
            }
        }
        
        // Decision logic: if most neighbors are reference points, reassign
        if total_neighbors > 0 {
            let ref_ratio = ref_neighbors as f64 / total_neighbors as f64;
            if ref_ratio >= min_neigbor_ratio {
                // Reassign to reference_points (anomalous)
                reassigned_points.push(*point);
            } else {
                // Keep in cleaned_points (proximal/distal)
                cleaned_points.push(*point);
            }
        } else {
            // If no neighbors in range, keep original classification
            cleaned_points.push(*point);
        }
    }
    
    (cleaned_points, reassigned_points)
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
            (2.0, 0.0, 0.0),
        ];

        let result = centerline_based_diameter_morphing(&centerline, &points, -0.5);
        
        // Should move toward centerline by 0.5 units
        let new_point = result[0];
        assert!((new_point.0 - 1.5).abs() < 1e-6);
        assert!((new_point.1 - 0.0).abs() < 1e-6);
        assert!((new_point.2 - 0.0).abs() < 1e-6);
    }
}