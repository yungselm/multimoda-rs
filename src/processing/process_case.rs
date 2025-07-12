use anyhow::bail;

use crate::io::input::{Contour, ContourPoint};
use crate::io::output::{write_geometry_vec_to_obj, GeometryType};
use crate::io::Geometry;
use crate::processing::geometries::GeometryPair;
use crate::processing::walls::create_wall_geometry;
use crate::texture::{write_mtl_geometry, write_mtl_wall};

pub fn create_geometry_pair(
    case_name: String,
    input_dir: &str,
    steps_best_rotation: usize,
    range_rotation_rad: f64,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    align_inside: bool,
) -> anyhow::Result<GeometryPair> {
    let geometries =
        GeometryPair::new(input_dir, case_name.clone(), image_center, radius, n_points)?;
    let mut geometries = geometries.adjust_z_coordinates();

    geometries = geometries.process_geometry_pair(steps_best_rotation, range_rotation_rad, align_inside);
    geometries = geometries.trim_geometries_same_length();
    geometries = geometries.thickness_adjustment();

    let dia_geom = geometries.dia_geom;
    let dia_geom = dia_geom.smooth_contours();
    let sys_geom = geometries.sys_geom;
    let sys_geom = sys_geom.smooth_contours();

    Ok(GeometryPair {
        dia_geom: dia_geom,
        sys_geom: sys_geom,
    })
}

/// Processes a given case by reading diastolic and systolic contours, aligning them,
/// computing displacements and UV coordinates, and finally writing out the OBJ, MTL, and texture files.
/// Additionally it can be specified how many interpolation steps should be used to generate the final meshes
/// used for the animation in blender.
pub fn process_case(
    case_name: &str,
    geometries: GeometryPair,
    output_dir: &str,
    interpolation_steps: usize,
) -> anyhow::Result<GeometryPair> {
    std::fs::create_dir_all(output_dir)?;

    let dia_geom = geometries.dia_geom;
    let sys_geom = geometries.sys_geom;

    // Interpolate between two geometrys by creating new geometries with coordinates
    // in between the two geometries.
    let interpolated_geometries =
        interpolate_contours(&dia_geom, &sys_geom, interpolation_steps.clone())?;

    let (uv_coords_contours, uv_coords_catheter) =
        write_mtl_geometry(&interpolated_geometries, output_dir, case_name);

    // Write contours (mesh) and catheter using the enum
    write_geometry_vec_to_obj(
        GeometryType::Contour,
        case_name,
        output_dir,
        &interpolated_geometries,
        &uv_coords_contours,
    )?;

    if !dia_geom.catheter.is_empty() & !sys_geom.catheter.is_empty() {
        write_geometry_vec_to_obj(
            GeometryType::Catheter,
            case_name,
            output_dir,
            &interpolated_geometries,
            &uv_coords_catheter,
        )?;
    }

    // test wall mesh
    let dia_wall = create_wall_geometry(&dia_geom, false);
    let sys_wall = create_wall_geometry(&sys_geom, false);

    let interpolated_walls =
        interpolate_contours(&dia_wall, &sys_wall, interpolation_steps.clone())?;

    let uv_coords_walls = write_mtl_wall(&interpolated_walls, output_dir, case_name);

    write_geometry_vec_to_obj(
        GeometryType::Wall,
        case_name,
        output_dir,
        &interpolated_walls,
        &uv_coords_walls,
    )?;

    Ok(GeometryPair { dia_geom, sys_geom })
}

/// Interpolates between two aligned Geometry configurations with number of steps
/// used to visualize deformation over a cardiac cycle.
pub fn interpolate_contours(
    start: &Geometry,
    end: &Geometry,
    steps: usize,
) -> anyhow::Result<Vec<Geometry>> {
    use std::cmp::min;
    let n = min(start.contours.len(), end.contours.len());

    // Check catheter lengths
    let use_catheter = !start.catheter.is_empty() && !end.catheter.is_empty();

    let mut geoms = Vec::with_capacity(steps + 2);
    geoms.push(start.clone());

    for step in 0..steps {
        let t = step as f64 / (steps - 1) as f64;

        let contours = interp_contour_list(&start.contours[..n], &end.contours[..n], t);

        let catheter = if use_catheter {
            interp_contour_list(&start.catheter[..n], &end.catheter[..n], t)
                .into_iter()
                // catheter contours don’t carry thickness
                .map(|mut c| { c.aortic_thickness = None; c.pulmonary_thickness = None; c })
                .collect()
        } else {
            Vec::new()
        };

        // now simply interpolate walls exactly the same way:
        let walls = interp_contour_list(&start.walls[..n], &end.walls[..n], t);

        geoms.push(Geometry {
            contours,
            catheter,
            walls,
            reference_point: start.reference_point.clone(),
            label: format!("{}_inter_{}", start.label, step),
        });
    }

    geoms.push(end.clone());
    Ok(geoms)
}

/// Interpolate two aligned Vec<Contour> slices.
fn interp_contour_list(
    a: &[Contour],
    b: &[Contour],
    t: f64,
) -> Vec<Contour> {
    a.iter().zip(b)
     .map(|(s, e)| Contour {
         id: s.id,
         points: s.points.iter()
              .zip(&e.points)
              .map(|(ps, pe)| ContourPoint {
                  frame_index: ps.frame_index,
                  point_index: ps.point_index,
                  x: ps.x * (1.0 - t) + pe.x * t,
                  y: ps.y * (1.0 - t) + pe.y * t,
                  z: ps.z * (1.0 - t) + pe.z * t,
                  aortic: ps.aortic,
              })
              .collect(),
         centroid: (
             s.centroid.0 * (1.0 - t) + e.centroid.0 * t,
             s.centroid.1 * (1.0 - t) + e.centroid.1 * t,
             s.centroid.2 * (1.0 - t) + e.centroid.2 * t,
         ),
         aortic_thickness: interpolate_thickness(&s.aortic_thickness, &e.aortic_thickness, t),
         pulmonary_thickness: interpolate_thickness(&s.pulmonary_thickness, &e.pulmonary_thickness, t),
     })
     .collect()
}

/// Helper function to interpolate Vec<Contour>
#[allow(dead_code)]
fn interpolate_points(
    start_contours: &[Contour],
    end_contours: &[Contour],
    steps: usize,
    n: usize,
) -> anyhow::Result<Vec<Contour>> {
    let mut intermediate_contours = Vec::with_capacity(n);

    for step in 0..steps {
        let t = step as f64 / (steps - 1) as f64;

        // Pair matching contours between start and end geometries
        for (start_contour, end_contour) in start_contours.iter().zip(end_contours.iter()) {
            if start_contour.id != end_contour.id {
                bail!("Contour IDs do not match between start and end geometries");
            }
            if start_contour.points.len() != end_contour.points.len() {
                bail!("Contour point counts do not match between start and end");
            }

            // Interpolate points between contours
            let interp_points: Vec<ContourPoint> = start_contour
                .points
                .iter()
                .zip(end_contour.points.iter())
                .map(|(p_start, p_end)| {
                    // Linear interpolation for each coordinate
                    ContourPoint {
                        frame_index: p_start.frame_index,
                        point_index: p_start.point_index,
                        x: p_start.x * (1.0 - t) + p_end.x * t,
                        y: p_start.y * (1.0 - t) + p_end.y * t,
                        z: p_start.z * (1.0 - t) + p_end.z * t,
                        aortic: p_start.aortic,
                    }
                })
                .collect();

            // Create new Contour with interpolated values
            let interp_contour = Contour {
                id: start_contour.id,
                points: interp_points,
                centroid: (
                    start_contour.centroid.0 * (1.0 - t) + end_contour.centroid.0 * t,
                    start_contour.centroid.1 * (1.0 - t) + end_contour.centroid.1 * t,
                    start_contour.centroid.2 * (1.0 - t) + end_contour.centroid.2 * t,
                ),
                aortic_thickness: interpolate_thickness(
                    &start_contour.aortic_thickness,
                    &end_contour.aortic_thickness,
                    t,
                ),
                pulmonary_thickness: interpolate_thickness(
                    &start_contour.pulmonary_thickness,
                    &end_contour.pulmonary_thickness,
                    t,
                ),
            };

            intermediate_contours.push(interp_contour);
        }
    }

    Ok(intermediate_contours)
}

/// Interpolate two optional thickness values at fraction `t` (0.0..1.0).
fn interpolate_thickness(start: &Option<f64>, end: &Option<f64>, t: f64) -> Option<f64> {
    match (start, end) {
        (Some(s), Some(e)) => Some(s * (1.0 - t) + e * t),
        _ => None,
    }
}

#[cfg(test)]
mod process_tests {
    use super::*;
    use crate::io::input::{Contour, ContourPoint};
    use approx::assert_relative_eq; // Add approx = "1.4" to Cargo.toml

    // Helper to create mock geometry
    fn mock_geometry(label: &str) -> Geometry {
        Geometry {
            contours: vec![Contour {
                id: 1,
                points: vec![
                    ContourPoint {
                        frame_index: 0,
                        point_index: 0,
                        x: 1.0,
                        y: 2.0,
                        z: 3.0,
                        aortic: true,
                    },
                    ContourPoint {
                        frame_index: 0,
                        point_index: 1,
                        x: 4.0,
                        y: 5.0,
                        z: 6.0,
                        aortic: true,
                    },
                ],
                centroid: (0.5, 1.0, 1.5),
                aortic_thickness: Some(1.0),
                pulmonary_thickness: Some(2.0),
            }],
            catheter: vec![Contour {
                id: 2,
                points: vec![ContourPoint {
                    frame_index: 0,
                    point_index: 0,
                    x: 10.0,
                    y: 20.0,
                    z: 30.0,
                    aortic: true,
                }],
                centroid: (5.0, 10.0, 15.0),
                aortic_thickness: None,
                pulmonary_thickness: None,
            }],
            walls: vec![],
            reference_point: ContourPoint {
                frame_index: 0,
                point_index: 0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                aortic: true,
            },
            label: label.to_string(),
        }
    }

    #[test]
    fn test_interpolate_contours_basic() {
        let start = mock_geometry("start");
        let end = mock_geometry("end");
        let steps = 2;

        let result = interpolate_contours(&start, &end, steps).unwrap();

        // Should have start + steps interpolated + end
        assert_eq!(result.len(), steps + 2);

        // Verify start is unchanged
        assert_eq!(result[0].label, "start");
        assert_eq!(result[0].contours[0].points[0].x, 1.0);

        // Verify end is unchanged
        assert_eq!(result[result.len() - 1].label, "end");
        assert_eq!(result[result.len() - 1].contours[0].points[0].x, 1.0);

        // Verify interpolation at midpoint
        let mid = &result[1];
        assert_eq!(mid.label, "start_inter_0");

        // Point interpolation
        assert_relative_eq!(mid.contours[0].points[0].x, 1.0, epsilon = 1e-5);
        assert_relative_eq!(mid.contours[0].points[1].y, 5.0, epsilon = 1e-5);

        // Centroid interpolation
        assert_relative_eq!(mid.contours[0].centroid.0, 0.5, epsilon = 1e-5);

        // Catheter interpolation
        assert_relative_eq!(mid.catheter[0].points[0].z, 30.0, epsilon = 1e-5);
    }

    #[test]
    fn test_interpolate_contours_different_lengths() {
        let start = mock_geometry("start");
        let mut end = mock_geometry("end");

        // Add extra contour to end geometry
        end.contours.push(Contour {
            id: 3,
            points: vec![ContourPoint {
                frame_index: 0,
                point_index: 0,
                x: 100.0,
                y: 200.0,
                z: 300.0,
                aortic: true,
            }],
            centroid: (50.0, 100.0, 150.0),
            aortic_thickness: Some(10.0),
            pulmonary_thickness: None,
        });

        let result = interpolate_contours(&start, &end, 1).unwrap();

        // Verify frame counts
        assert_eq!(result.len(), 3);

        // Start frame has original 1 contour
        assert_eq!(result[0].contours.len(), 1);

        // Interpolated frame uses min contours (1)
        assert_eq!(result[1].contours.len(), 1);

        // End frame retains its original 2 contours
        assert_eq!(result[2].contours.len(), 2);
    }

    #[test]
    fn test_interpolate_thickness() {
        // Both present
        assert_eq!(
            interpolate_thickness(&Some(1.0), &Some(3.0), 0.5),
            Some(2.0)
        );

        // Start missing
        assert_eq!(interpolate_thickness(&None, &Some(3.0), 0.5), None);

        // End missing
        assert_eq!(interpolate_thickness(&Some(1.0), &None, 0.5), None);

        // Both missing
        assert_eq!(interpolate_thickness(&None, &None, 0.5), None);
    }

    #[test]
    fn test_interpolate_contours_zero_steps() {
        let start = mock_geometry("start");
        let end = mock_geometry("end");

        let result = interpolate_contours(&start, &end, 0).unwrap();

        // Should still have start and end frames
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].label, "start");
        assert_eq!(result[1].label, "end");
    }

    #[test]
    fn test_interpolate_contours_mismatched_ids() {
        let start = mock_geometry("start");
        let mut end = mock_geometry("end");

        // Create ID mismatch
        end.contours[0].id = 99;

        let result = interpolate_contours(&start, &end, 1);
        assert!(result.is_ok(), "Should handle ID mismatch gracefully");
    }
}
