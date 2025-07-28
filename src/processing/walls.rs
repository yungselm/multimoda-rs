use std::ops::RangeInclusive;

use crate::io::input::{Contour, ContourPoint};
use crate::io::Geometry;

pub fn create_wall_geometry(geometry: &Geometry, with_pulmonary: bool) -> Geometry {
    let mut new_contours = Vec::new();

    for contour in &geometry.contours {
        let new_contour = if with_pulmonary {
            create_wall_contour_with_pulmonary(contour)
        } else {
            create_wall_contour_aortic_only(contour)
        };

        new_contours.push(new_contour);
    }

    let new_geometry = Geometry {
        contours: geometry.contours.clone(),
        catheter: geometry.catheter.clone(),
        walls: new_contours,
        reference_point: geometry.reference_point.clone(),
        label: geometry.label.clone(),
    };

    new_geometry
}

fn create_wall_contour_aortic_only(contour: &Contour) -> Contour {
    if contour.aortic_thickness.is_none() {
        let new_contour = offset_contour(contour, 1.0, None);
        new_contour
    } else {
        let new_contour = create_aortic_wall(contour);
        new_contour
    }
}

#[allow(dead_code)]
fn create_wall_contour_with_pulmonary(_contour: &Contour) -> Contour {
    todo!("not yet implemented");
}

/// Offsets every point away from the centroid by exactly `distance` units,
/// only for those whose `point_index` is in the optional `point_range`.
/// If `point_range` is `None`, all points get offset.
pub fn offset_contour(
    contour: &Contour,
    distance: f64,
    point_range: Option<RangeInclusive<u32>>,
) -> Contour {
    let (cx, cy, cz) = contour.centroid;

    let new_points = contour.points.iter().map(|pt| {
        let mut p = pt.clone();

        let do_offset = point_range
            .as_ref()
            .map(|r| r.contains(&(pt.point_index as u32)))
            .unwrap_or(true);

        if do_offset {
            // vector from centroid → point
            let dx = pt.x - cx;
            let dy = pt.y - cy;
            let dz = pt.z - cz;

            let len = (dx * dx + dy * dy + dz * dz).sqrt();
            if len > std::f64::EPSILON {
                // build a unit vector, then add `distance` along it
                let ux = dx / len;
                let uy = dy / len;
                let uz = dz / len;

                p.x += ux * distance;
                p.y += uy * distance;
                p.z += uz * distance;
            }
        }

        p
    });

    Contour {
        id: contour.id,
        points: new_points.collect(),
        centroid: contour.centroid,
        aortic_thickness: contour.aortic_thickness,
        pulmonary_thickness: contour.pulmonary_thickness,
    }
}

/// Create aortic wall contour with the thickness from IVUS images
/// assumption: using preexisting point indices since the contour is already aligned
/// 0 is the highest point, 250 is the lowest point, point 375 used to build the thickness
/// by adding the distance to point x-coordinates. y-coordinates aortic side are +1.5
/// to the y-coodinate of point 0 and -1.5 to the y-coordinates of point 250. The side
/// with indices 0-250 will use the enlarge contour function to create a coronary wall.
/// For easier geometry creation again 500 points are used.
fn create_aortic_wall(contour: &Contour) -> Contour {
    let n = contour.points.len();
    let first_quarter = n / 4;
    let half = n / 2;
    let third_quarter = first_quarter * 3;

    let ref_pt = &contour.points[third_quarter];
    let thickness = contour
        .aortic_thickness
        .expect("aortic_thickness must be present for this contour");
    let outer_x = ref_pt.x + thickness;
    let z = ref_pt.z;

    // Define key points (x, y)
    let up_mid = (contour.points[0].x, contour.points[0].y + 1.0);
    let up_right = (outer_x, up_mid.1);
    let low_mid = (contour.points[half].x, contour.points[half].y - 1.0);
    let low_right = (outer_x, low_mid.1);

    // Calculate segment lengths for point distribution
    let dist_up = (up_right.0 - up_mid.0).abs();
    let dist_right = (up_right.1 - low_right.1).abs();
    let dist_low = (low_right.0 - low_mid.0).abs();
    let total_dist = dist_up + dist_right + dist_low;

    // Allocate points proportionally (sum must be 250)
    let n_points_up = (dist_up / total_dist * half as f64).round() as usize;
    let n_points_mid = (dist_right / total_dist * half as f64).round() as usize;
    let mut n_points_low = half - n_points_up - n_points_mid;

    // Ensure we have exactly half points total
    let total = n_points_up + n_points_mid + n_points_low;
    if total != half {
        n_points_low += half - total; // Distribute remaining points to low segment
    }

    // Generate points for each segment
    let mut right_points = Vec::with_capacity(half);

    // 1. Horizontal line: low_mid to low_right
    for i in 0..n_points_low {
        let t = i as f64 / (n_points_low - 1) as f64;
        let x = low_mid.0 + t * (low_right.0 - low_mid.0);
        right_points.push((x, low_mid.1));
    }

    // 2. Vertical line: low_right to up_right
    for i in 0..n_points_mid {
        let t = i as f64 / (n_points_mid - 1) as f64;
        let y = low_right.1 + t * (up_right.1 - low_right.1);
        right_points.push((low_right.0, y));
    }

    // 3. Horizontal line: up_right to up_mid
    for i in 0..n_points_up {
        let t = i as f64 / (n_points_up.max(1) - 1) as f64;
        let x = up_right.0 - t * (up_right.0 - up_mid.0);
        right_points.push((x, up_right.1));
    }

    // Create the contour points
    let mut left_wall = offset_contour(contour, 1.0, Some(0..=half as u32)).points;
    if left_wall.len() % 2 != 0 {
        left_wall.truncate(half + 1); // + 1 for uneven numbers of points
    } else {
        left_wall.truncate(half)
    };
    let left_len = left_wall.len();

    let mut right_wall = Vec::with_capacity(half);
    for (i, (x, y)) in right_points.into_iter().enumerate() {
        // Use safe index calculation
        let src_index = left_len + i;
        assert!(
            src_index < contour.points.len(),
            "Index out of bounds: {} >= {}",
            src_index,
            contour.points.len()
        );

        let src = &contour.points[src_index];
        right_wall.push(ContourPoint {
            frame_index: src.frame_index,
            point_index: src.point_index,
            x,
            y,
            z,
            aortic: src.aortic,
        });
    }

    let mut new_points = Vec::with_capacity(contour.points.len());
    new_points.extend(left_wall);
    new_points.extend(right_wall);

    Contour {
        id: contour.id,
        points: new_points,
        centroid: contour.centroid,
        aortic_thickness: contour.aortic_thickness,
        pulmonary_thickness: contour.pulmonary_thickness,
    }
}
