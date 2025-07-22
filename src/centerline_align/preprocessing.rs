use nalgebra::Vector3;

use crate::io::input::{Centerline, CenterlinePoint, ContourPoint};
use crate::io::Geometry;
use crate::processing::geometries::GeometryPair;

pub fn ensure_descending_z(centerline: &mut Centerline) {
    if !centerline.points.is_empty() {
        let first_z = centerline.points[0].contour_point.z;
        let last_z = centerline.points.last().unwrap().contour_point.z;
        if first_z < last_z {
            centerline.points.reverse();
        }
    }
}

pub fn remove_leading_points_cl(
    mut centerline: Centerline,
    reference_point: &(f64, f64, f64),
) -> Centerline {
    centerline.points.retain(|p| 
        !p.contour_point.x.is_nan() && 
        !p.contour_point.y.is_nan() && 
        !p.contour_point.z.is_nan()
    );
    
    if centerline.points.is_empty() {
        return centerline;
    }

    // Find closest point to reference
    let closest_pt = centerline
        .points
        .iter()
        .min_by(|a, b| {
            distance_sq(&a.contour_point, reference_point)
                .partial_cmp(&distance_sq(&b.contour_point, reference_point))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    let start_frame = closest_pt.contour_point.frame_index;

    println!("Index of closest point: {:?}", closest_pt.contour_point.frame_index);

    // Remove points before closest point
    let mut remaining: Vec<_> = centerline
        .points
        .into_iter()
        .filter(|p| p.contour_point.frame_index >= start_frame)
        .collect();

    // 4) Re-sort by frame_index to restore z-order
    remaining.sort_by_key(|p| p.contour_point.frame_index);

    // Reindex starting from 0
    for (i, pt) in remaining.iter_mut().enumerate() {
        pt.contour_point.frame_index = i as u32;
        pt.contour_point.point_index = i as u32;
    }

    Centerline { points: remaining }
}

/// Helper function to calculate squared distance between two points
fn distance_sq(a: &ContourPoint, b: &(f64, f64, f64)) -> f64 {
    let dx = a.x - b.0;
    let dy = a.y - b.1;
    let dz = a.z - b.2;
    dx * dx + dy * dy + dz * dz
}

pub fn resample_centerline_by_contours(centerline: &Centerline, ref_mesh: &Geometry) -> Centerline {
    if centerline.points.is_empty() {
        return Centerline { points: Vec::new() };
    }

    // Extract and sort z_refs (descending)
    let mut z_refs: Vec<f64> = ref_mesh.contours.iter().map(|c| c.centroid.2).collect();
    z_refs.sort_by(|a, b| b.partial_cmp(a).unwrap()); // Descending
    z_refs.dedup();

    let min_z = centerline.points.last().unwrap().contour_point.z;
    let max_z = centerline.points[0].contour_point.z;
    
    // Filter z_refs within centerline's z-range
    let z_refs: Vec<f64> = z_refs
        .into_iter()
        .filter(|&z| z <= max_z && z >= min_z)
        .collect();

    if z_refs.is_empty() {
        return centerline.clone();
    }

    let mut i = 0; // Index in z_refs
    let mut j = 0; // Index in centerline segments
    let mut new_points = Vec::new();

    while i < z_refs.len() && j < centerline.points.len() - 1 {
        let z_target = z_refs[i];
        let p0 = &centerline.points[j];
        let p1 = &centerline.points[j + 1];
        let p0_z = p0.contour_point.z;
        let p1_z = p1.contour_point.z;

        if z_target > p0_z {
            // Above current segment - skip
            i += 1;
        } else if z_target < p1_z {
            // Below current segment - move to next segment
            j += 1;
        } else {
            // Within segment - interpolate
            let t = (p0_z - z_target) / (p0_z - p1_z);
            let x = p0.contour_point.x + t * (p1.contour_point.x - p0.contour_point.x);
            let y = p0.contour_point.y + t * (p1.contour_point.y - p0.contour_point.y);
            
            new_points.push(CenterlinePoint {
                contour_point: ContourPoint {
                    frame_index: new_points.len() as u32,
                    point_index: new_points.len() as u32,
                    x,
                    y,
                    z: z_target,
                    aortic: false,
                },
                normal: Vector3::zeros(),
            });
            i += 1;
        }
    }

    Centerline { points: new_points }
}

pub fn prepare_geometry_alignment(mut geom_pair: GeometryPair) -> GeometryPair {
    fn align_geometry(mut geom: Geometry) -> Geometry {
        geom.contours.reverse();
        for (index, contour) in geom.contours.iter_mut().enumerate() {
            contour.id = index as u32;
            for point in &mut contour.points {
                point.frame_index = index as u32;
            }
        }

        geom.catheter.reverse();
        for (index, catheter) in geom.catheter.iter_mut().enumerate() {
            catheter.id = index as u32;
            for point in &mut catheter.points {
                point.frame_index = index as u32;
            }
        }

        geom.walls.reverse();
        for (index, contour) in geom.walls.iter_mut().enumerate() {
            contour.id = index as u32;
            for point in &mut contour.points {
                point.frame_index = index as u32;
            }
        }

        geom.reference_point.frame_index = (geom.contours.len() - 1)
            .saturating_sub(geom.reference_point.frame_index as usize) as u32; // correct method?

        geom
    }

    geom_pair.dia_geom = align_geometry(geom_pair.dia_geom);
    geom_pair.sys_geom = align_geometry(geom_pair.sys_geom);

    geom_pair
}
