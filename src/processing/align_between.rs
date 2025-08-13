use rayon::prelude::*;
use anyhow::Context;

use crate::io::Geometry;
use crate::io::input::{Contour, ContourPoint};
use crate::processing::process_utils::downsample_contour_points;

#[derive(Clone, Debug)]
pub struct GeometryPair {
    pub dia_geom: Geometry,
    pub sys_geom: Geometry,
}

pub fn get_geometry_pair(
    case_name: String,
    input_dir: &str,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
) -> anyhow::Result<GeometryPair> {
    let mut geometries = 
    GeometryPair::new(
        input_dir,
        case_name.clone(),
        image_center,
        radius,
        n_points)
        .context(format!("Creating GeometryPair({}) failed", case_name))?;
    geometries = geometries.adjust_z_coordinates();
    geometries = geometries.trim_geometries_same_length();
    
    Ok(geometries)
}

impl GeometryPair {
    pub fn new(
        input_dir: &str,
        label: String,
        image_center: (f64, f64),
        radius: f64,
        n_points: u32,
    ) -> anyhow::Result<GeometryPair> {
        let dia_geom = Geometry::new(
            input_dir,
            label.clone(),
            true,
            image_center,
            radius,
            n_points,
        )?;
        println!("geometry pair: diastolic geometry generated");
        let sys_geom = Geometry::new(input_dir, label, false, image_center, radius, n_points)?;
        println!("geometry pair: systolic geometry generated");
        Ok(GeometryPair { dia_geom, sys_geom })
    }

    /// Translates systolic contours to the diastolic contours, aligns z-coordinates and
    /// finds the best rotation and trims them to same length.    
    pub fn align_between_geometries(
        mut self,
        steps_rotation_deg: f64,
        range_rotation_deg: f64,
    ) -> anyhow::Result<GeometryPair> {
        Self::translate_contours_to_match(&self.dia_geom, &mut self.sys_geom);
    
        // Adjust the z-coordinates of systolic contours. (later replaceed by adjust_z_coordinates)
        Self::apply_z_transformation(&self.dia_geom, &mut self.sys_geom);
    
        let best_rotation_angle = find_best_rotation_all(
            &self.dia_geom,
            &self.sys_geom,
            steps_rotation_deg, // number of candidate steps (e.g. 200 or 400)
            range_rotation_deg,  // rotation range (e.g. 1.05 for ~±60°)
        );
    
        for ref mut contour in self.sys_geom
            .contours
            .iter_mut()
            .chain(self.sys_geom.catheter.iter_mut())
        {
            contour.rotate_contour(best_rotation_angle);
        };
        Ok(GeometryPair {
            dia_geom: self.dia_geom,
            sys_geom: self.sys_geom,
        })
    }

    pub fn translate_contours_to_match(dia: &Geometry, sys: &mut Geometry) {
        let dia_ref = dia.contours.last().unwrap().centroid;
        let sys_ref = sys.contours.last().unwrap().centroid;
        let offset = (dia_ref.0 - sys_ref.0, dia_ref.1 - sys_ref.1);

        for contour in &mut sys.contours {
            contour.translate_contour((offset.0, offset.1, 0.0));
        }

        for catheter in &mut sys.catheter {
            catheter.translate_contour((offset.0, offset.1, 0.0));
        }
    }

    pub fn apply_z_transformation(dia: &Geometry, sys: &mut Geometry) {
        let dia_last_z = dia.contours.last().unwrap().centroid.2;
        let sys_last_z = sys.contours.last().unwrap().centroid.2;
        let z_offset = dia_last_z - sys_last_z;

        for contour in &mut sys.contours {
            contour.points.iter_mut().for_each(|p| p.z += z_offset);
            contour.centroid.2 += z_offset;
        }

        for catheter in &mut sys.catheter {
            catheter.points.iter_mut().for_each(|p| p.z += z_offset);
            catheter.centroid.2 += z_offset;
        }
    }

    pub fn adjust_z_coordinates(mut self) -> GeometryPair {
        let mut z_coords_dia: Vec<f64> = self
            .dia_geom
            .contours
            .iter()
            .skip(1) // Skip the first entry since 0.0
            .map(|contour| contour.centroid.2)
            .collect();

        let mut z_coords_sys: Vec<f64> = self
            .sys_geom
            .contours
            .iter()
            .skip(1) // Skip the first entry sicne 0.0
            .map(|contour| contour.centroid.2)
            .collect();

        for i in (0..z_coords_dia.len()).rev() {
            z_coords_dia[i] /= (i + 1) as f64;
        }

        for i in (0..z_coords_sys.len()).rev() {
            z_coords_sys[i] /= (i + 1) as f64;
        }

        let mut z_coords = z_coords_sys;
        z_coords.extend(z_coords_dia);

        let mean_z_coords = z_coords.iter().sum::<f64>() / z_coords.len() as f64;

        // If there are missing frames in between this will create false results, but probably
        // still more accurate then taking the actual frame position due to breathing artefacts
        // and the resampling performed in combined_sorted_manual to counter this.
        let n_slices = self
            .dia_geom
            .contours
            .len()
            .max(self.sys_geom.contours.len())
            .max(self.dia_geom.catheter.len())
            .max(self.sys_geom.catheter.len());

        let mut current_z = 0.0;
        for i in 0..n_slices {
            // helper to set z for a mutable slice element
            let assign_z = |cont_opt: Option<&mut Contour>| {
                if let Some(cont) = cont_opt {
                    cont.centroid.2 = current_z;
                    for pt in &mut cont.points {
                        pt.z = current_z;
                    }
                }
            };

            assign_z(self.dia_geom.contours.get_mut(i));
            assign_z(self.sys_geom.contours.get_mut(i));
            assign_z(self.dia_geom.catheter.get_mut(i));
            assign_z(self.sys_geom.catheter.get_mut(i));

            current_z += mean_z_coords;
        }

        self
    }

    pub fn trim_geometries_same_length(mut self) -> GeometryPair {
        // Process contours
        let min_contours =
            std::cmp::min(self.dia_geom.contours.len(), self.sys_geom.contours.len());

        if self.dia_geom.contours.len() > min_contours {
            let remove_count = self.dia_geom.contours.len() - min_contours;
            self.dia_geom.contours.drain(0..remove_count);
            for contour in self.dia_geom.contours.iter_mut() {
                contour.id -= remove_count as u32;
                for point in contour.points.iter_mut() {
                    point.frame_index -= remove_count as u32;
                }
            }
        }

        if self.sys_geom.contours.len() > min_contours {
            let remove_count = self.sys_geom.contours.len() - min_contours;
            self.sys_geom.contours.drain(0..remove_count);
            for contour in self.sys_geom.contours.iter_mut() {
                contour.id -= remove_count as u32;
                for point in contour.points.iter_mut() {
                    point.frame_index -= remove_count as u32;
                }
            }
        }

        // Process catheter points
        let min_catheter =
            std::cmp::min(self.dia_geom.catheter.len(), self.sys_geom.catheter.len());

        if self.dia_geom.catheter.len() > min_catheter {
            let remove_count = self.dia_geom.catheter.len() - min_catheter;
            self.dia_geom.catheter.drain(0..remove_count);
            for catheter in self.dia_geom.catheter.iter_mut() {
                catheter.id -= remove_count as u32;
                for point in catheter.points.iter_mut() {
                    point.frame_index -= remove_count as u32;
                }
            }
        }

        if self.sys_geom.catheter.len() > min_catheter {
            let remove_count = self.sys_geom.catheter.len() - min_catheter;
            self.sys_geom.catheter.drain(0..remove_count);
            for catheter in self.sys_geom.catheter.iter_mut() {
                catheter.id -= remove_count as u32;
                for point in catheter.points.iter_mut() {
                    point.frame_index -= remove_count as u32;
                }
            }
        }
        self
    }

    /// Adjusts the aortic and pulmonary thicknesses of the contours in both geometries
    /// to be the average of the two. This is done for each contour in both geometries.
    /// The function ensures that the lengths of the thickness vectors are equal by resizing
    /// them to the maximum length found in either geometry. The average is calculated
    /// for each corresponding element in the vectors.
    /// If one of the elements is None, it takes the value from the other element.
    /// If both are None, it remains None.
    /// This function is called after the geometries have been aligned and translated.
    /// It is important to ensure that the geometries are aligned before calling this function.
    /// The function assumes that the contours in both geometries are in the same order.
    /// It does not check for matching IDs, so it is the caller's responsibility to ensure
    /// that the contours correspond to the same anatomical structures.
    pub fn thickness_adjustment(mut self) -> GeometryPair {
        let min_contours =
            std::cmp::min(self.dia_geom.contours.len(), self.sys_geom.contours.len());
        for i in 0..min_contours {
            let dia = &mut self.dia_geom.contours[i];
            let sys = &mut self.sys_geom.contours[i];

            let combined_aortic = match (dia.aortic_thickness, sys.aortic_thickness) {
                (Some(a), Some(b)) => Some((a + b) / 2.0),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            };
            dia.aortic_thickness = combined_aortic;
            sys.aortic_thickness = combined_aortic;

            let combined_pulmonary = match (dia.pulmonary_thickness, sys.pulmonary_thickness) {
                (Some(a), Some(b)) => Some((a + b) / 2.0),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            };
            dia.pulmonary_thickness = combined_pulmonary;
            sys.pulmonary_thickness = combined_pulmonary;
        }
        self
    }
}

// pub fn find_best_rotation_all(
//     diastole: &Geometry,
//     systole: &Geometry,
//     step_rotation_deg: f64,
//     range_deg: f64,
// ) -> f64 {
//     println!(
//         "---------------------Finding optimal rotation {:?}/{:?}---------------------",
//         &diastole.label, &systole.label
//     );
//     let steps: usize = ((2.0 * range_deg) / step_rotation_deg).round() as usize;
//     let range = range_deg.to_radians();
//     let increment = step_rotation_deg.to_radians();

//     let mut new_dia_cont = Vec::new();
//     let mut new_sys_cont = Vec::new();
//     for contour in &diastole.contours {
//         let tmp_points = align_within::downsample_contour_points(&contour.points, 200);
//         let new_cont = Contour {
//             id: contour.id,
//             points: tmp_points,
//             centroid: contour.centroid,
//             aortic_thickness: contour.aortic_thickness,
//             pulmonary_thickness: contour.pulmonary_thickness,
//         };
//         new_dia_cont.push(new_cont)
//     };
//     for contour in &systole.contours {
//         let tmp_points = align_within::downsample_contour_points(&contour.points, 200);
//         let new_cont = Contour {
//             id: contour.id,
//             points: tmp_points,
//             centroid: contour.centroid,
//             aortic_thickness: contour.aortic_thickness,
//             pulmonary_thickness: contour.pulmonary_thickness,
//         };
//         new_sys_cont.push(new_cont)
//     };

//     let mut new_dia = diastole.clone();
//     let mut new_sys = systole.clone();
//     new_dia.contours = new_dia_cont;
//     new_sys.contours = new_sys_cont;

//     let results: Vec<(f64, f64)> = (0..=steps)
//         .into_par_iter()
//         .map(|i| {
//             let angle = -range + i as f64 * increment;
//             let total_distance: f64 = new_dia
//                 .contours
//                 .par_iter()
//                 .zip(new_sys.contours.par_iter())
//                 .map(|(d_contour, s_contour)| {
//                     assert_eq!(d_contour.id, s_contour.id, "Mismatched contour IDs");

//                     // Rotate each point in systole contour
//                     let rotated_points: Vec<ContourPoint> = s_contour
//                         .points
//                         .iter()
//                         .map(|p| {
//                             let x = p.x * angle.cos() - p.y * angle.sin();
//                             let y = p.x * angle.sin() + p.y * angle.cos();
//                             ContourPoint { x, y, ..*p }
//                         })
//                         .collect();

//                     hausdorff_distance(&d_contour.points, &rotated_points)
//                 })
//                 .sum();

//             let avg_distance = total_distance / diastole.contours.len() as f64;

//             (angle, avg_distance)
//         })
//         .collect();

//     let (best_angle, best_dist) = results
//         .into_iter()
//         .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
//         .unwrap_or((0.0, std::f64::INFINITY));

//     // 3) Print a tiny table
//     println!();
//     println!(
//         "{:>20} | {:>20} | {:>15} | {:>12}",
//         "Geometry A", "Geometry B", "Best Distance", "Best Angle (°)"
//     );
//     println!("{:-<75}", "");
//     println!(
//         "{:>20} | {:>20} | {:>15.3} | {:>12.3}",
//         diastole.label,
//         systole.label,
//         best_dist,
//         best_angle.to_degrees(),
//     );
//     println!();

//     best_angle
// }

pub fn find_best_rotation_all(
    diastole: &Geometry,
    systole: &Geometry,
    step_rotation_deg: f64,
    range_deg: f64,
) -> f64 {
    println!(
        "---------------------Finding optimal rotation {:?}/{:?}---------------------",
        &diastole.label, &systole.label
    );
    
    // Downsample contours once instead of cloning entire geometries
    let dia_contours: Vec<_> = diastole.contours.iter()
        .map(|c| {
            let points = downsample_contour_points(&c.points, 200);
            (c.id, points, c.centroid)
        })
        .collect();
    
    let sys_contours: Vec<_> = systole.contours.iter()
        .map(|c| {
            let points = downsample_contour_points(&c.points, 200);
            let rel_points: Vec<_> = points.iter()
                .map(|p| (p.x - c.centroid.0, p.y - c.centroid.1))
                .collect();
            (c.id, rel_points, c.centroid)
        })
        .collect();

    // Multi-resolution search parameters
    let coarse_step = 1.0_f64.to_radians();
    let medium_step = 0.1_f64.to_radians();
    let fine_step = step_rotation_deg.to_radians();
    
    let range_rad = range_deg.to_radians();
    
    // Coarse search (1° steps)
    let coarse_angle = search_rotation(
        &dia_contours,
        &sys_contours,
        -range_rad,
        range_rad,
        coarse_step
    );
    
    // Medium search (0.1° steps in ±10° window)
    let medium_angle = search_rotation(
        &dia_contours,
        &sys_contours,
        coarse_angle - 10.0_f64.to_radians(),
        coarse_angle + 10.0_f64.to_radians(),
        medium_step
    );
    
    // Fine search (original steps in ±1° window)
    let best_angle = search_rotation(
        &dia_contours,
        &sys_contours,
        medium_angle - 1.0_f64.to_radians(),
        medium_angle + 1.0_f64.to_radians(),
        fine_step
    );

    // Calculate best distance for reporting
    let best_dist = calculate_avg_distance(&dia_contours, &sys_contours, best_angle);

    // Print results
    println!();
    println!(
        "{:>20} | {:>20} | {:>15} | {:>12}",
        "Geometry A", "Geometry B", "Best Distance", "Best Angle (°)"
    );
    println!("{:-<75}", "");
    println!(
        "{:>20} | {:>20} | {:>15.3} | {:>12.3}",
        diastole.label,
        systole.label,
        best_dist,
        best_angle.to_degrees(),
    );
    println!();

    best_angle
}

/// Helper function for rotation search in a given range
fn search_rotation(
    dia_contours: &[(u32, Vec<ContourPoint>, (f64, f64, f64))],
    sys_contours: &[(u32, Vec<(f64, f64)>, (f64, f64, f64))],
    start_angle: f64,
    end_angle: f64,
    step: f64,
) -> f64 {
    let steps = ((end_angle - start_angle) / step).ceil() as usize;
    
    (0..=steps)
        .into_par_iter()
        .map(|i| {
            let angle = start_angle + (i as f64) * step;
            (angle, calculate_avg_distance(dia_contours, sys_contours, angle))
        })
        .min_by(|&(_, a), &(_, b)| a.partial_cmp(&b).unwrap())
        .map(|(angle, _)| angle)
        .unwrap_or(0.0)
}

/// Calculate average Hausdorff distance for a given rotation angle
fn calculate_avg_distance(
    dia_contours: &[(u32, Vec<ContourPoint>, (f64, f64, f64))],
    sys_contours: &[(u32, Vec<(f64, f64)>, (f64, f64, f64))],
    angle: f64,
) -> f64 {
    let total: f64 = dia_contours
        .par_iter()
        .zip(sys_contours.par_iter())
        .map(|((dia_id, dia_points, _), (sys_id, sys_rel, centroid))| {
            assert_eq!(dia_id, sys_id, "Mismatched contour IDs");
            
            // Rotate points using precomputed relative coordinates
            let rotated: Vec<ContourPoint> = sys_rel
                .iter()
                .map(|(dx, dy)| {
                    let x = dx * angle.cos() - dy * angle.sin() + centroid.0;
                    let y = dx * angle.sin() + dy * angle.cos() + centroid.1;
                    ContourPoint { 
                        frame_index: 0,
                        point_index: 0,
                        x: x, 
                        y: y, 
                        z: 0.0,
                        aortic: false }
                })
                .collect();
            
            // hausdorff_distance(dia_points, &rotated)
            dia_points.iter()
                .zip(rotated.iter())
                .map(|(d, s)| {
                    (d.x - s.x).hypot(d.y - s.y)  // Euclidean distance
                })
                .sum::<f64>()
              / (dia_points.len() as f64)
        })
        .sum();

    total / dia_contours.len() as f64
}

#[cfg(test)]
mod geometry_pair_tests {
    use super::*;

    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    use crate::io::input::{Contour, ContourPoint};
    use crate::io::Geometry;
    use crate::utils::test_utils::new_dummy_contour;

    /// Helper: build a simple geometry with one contour of three points
    fn simple_geometry(
        offset: (f64, f64),
        z_offset: f64,
        thickness: (Option<f64>, Option<f64>),
    ) -> Geometry {
        let p1 = ContourPoint {
            frame_index: 0,
            point_index: 0,
            x: 0.0 + offset.0,
            y: 0.0 + offset.1,
            z: 0.0 + z_offset,
            aortic: false,
        };
        let p2 = ContourPoint {
            frame_index: 0,
            point_index: 1,
            x: 1.0 + offset.0,
            y: 0.0 + offset.1,
            z: 1.0 + z_offset,
            aortic: false,
        };
        let p3 = ContourPoint {
            frame_index: 0,
            point_index: 2,
            x: 2.0 + offset.0,
            y: 0.0 + offset.1,
            z: 2.0 + z_offset,
            aortic: false,
        };
        let mut contour = Contour {
            id: 0,
            points: vec![p1.clone(), p2.clone(), p3.clone()],
            centroid: (
                (0.0 + 1.0 + 2.0) / 3.0 + offset.0,
                offset.1,
                (0.0 + 1.0 + 2.0) / 3.0 + z_offset,
            ),
            aortic_thickness: thickness.0,
            pulmonary_thickness: thickness.1,
        };
        contour.sort_contour_points();
        Geometry {
            contours: vec![contour],
            catheter: vec![],
            walls: vec![],
            reference_point: p1,
            label: "test".into(),
        }
    }

    #[test]
    fn test_translate_contours_to_match() {
        let mut gp = GeometryPair {
            dia_geom: simple_geometry((5.0, 5.0), 0.0, (None, None)),
            sys_geom: simple_geometry((0.0, 0.0), 0.0, (None, None)),
        };
        gp = gp.align_between_geometries(1.0, 0.0).unwrap();
        let dia_centroid = gp.dia_geom.contours[0].centroid;
        let sys_centroid = gp.sys_geom.contours[0].centroid;
        assert_relative_eq!(dia_centroid.0, sys_centroid.0, epsilon = 1e-6);
        assert_relative_eq!(dia_centroid.1, sys_centroid.1, epsilon = 1e-6);
    }

    #[test]
    fn test_apply_z_and_adjust_z_coordinates() {
        let dia = simple_geometry((0.0, 0.0), 0.0, (None, None));
        let mut gp = GeometryPair {
            dia_geom: dia.clone(),
            sys_geom: simple_geometry((0.0, 0.0), 2.0, (None, None)),
        };
        gp = gp.align_between_geometries(1.0, 0.0).unwrap();
        gp = gp.adjust_z_coordinates();
        for contour in gp.dia_geom.contours.iter() {
            assert!(contour.centroid.2.is_finite());
        }
        for contour in gp.sys_geom.contours.iter() {
            assert!(contour.centroid.2.is_finite());
        }
    }

    #[test]
    fn test_adjust_z_coordinates_mean_spacing() {
        // helper to build a Geometry with contours at the given centroid z-positions
        fn geom_from_zs(zs: &[f64]) -> Geometry {
            let mut contours = Vec::new();
            for (i, &z) in zs.iter().enumerate() {
                let p = ContourPoint {
                    frame_index: i as u32,
                    point_index: 0,
                    x: 0.0,
                    y: 0.0,
                    z,
                    aortic: false,
                };
                let mut cont = Contour {
                    id: i as u32,
                    points: vec![p.clone()],
                    centroid: (0.0, 0.0, z),
                    aortic_thickness: None,
                    pulmonary_thickness: None,
                };
                // ensure sorting if your implementation expects it
                cont.sort_contour_points();
                contours.push(cont);
            }
            Geometry {
                contours,
                catheter: vec![],
                walls: vec![],
                reference_point: ContourPoint {
                    frame_index: 0,
                    point_index: 0,
                    x: 0.0,
                    y: 0.0,
                    z: zs[0],
                    aortic: false,
                },
                label: "test".into(),
            }
        }

        // dia with 1.0 spacing (0,1,2,3)
        let dia = geom_from_zs(&[0.0, 1.0, 2.0, 3.0]);
        // sys with 2.0 spacing (0,2,4,6)
        let sys = geom_from_zs(&[0.0, 2.0, 4.0, 6.0]);

        let gp = GeometryPair {
            dia_geom: dia,
            sys_geom: sys,
        };

        let gp = gp.adjust_z_coordinates();

        // expected mean spacing = (1.0 + 2.0) / 2.0 = 1.5
        let expected_spacing = 1.5;
        // check dia spacing
        for window in gp.dia_geom.contours.windows(2) {
            let d = window[1].centroid.2 - window[0].centroid.2;
            assert_relative_eq!(d, expected_spacing, epsilon = 1e-9);
        }
        // check sys spacing
        for window in gp.sys_geom.contours.windows(2) {
            let d = window[1].centroid.2 - window[0].centroid.2;
            assert_relative_eq!(d, expected_spacing, epsilon = 1e-9);
        }

        // spot-check first and last positions (optional)
        let n = gp.dia_geom.contours.len();
        assert_relative_eq!(gp.dia_geom.contours[0].centroid.2, 0.0, epsilon = 1e-9);
        assert_relative_eq!(
            gp.dia_geom.contours[n - 1].centroid.2,
            expected_spacing * ((n - 1) as f64),
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_trim_geometries_same_length() {
        let mut gp = GeometryPair {
            dia_geom: simple_geometry((0.0, 0.0), 0.0, (None, None)),
            sys_geom: simple_geometry((0.0, 0.0), 0.0, (None, None)),
        };

        // Clear initial contours from simple_geometry
        gp.dia_geom.contours.clear();
        gp.dia_geom.catheter.clear();
        gp.sys_geom.contours.clear();
        gp.sys_geom.catheter.clear();

        // Create contours with increasing IDs starting from 0
        gp.dia_geom.contours.push(new_dummy_contour(0));
        gp.dia_geom.contours.push(new_dummy_contour(1));
        gp.dia_geom.contours.push(new_dummy_contour(2));
        gp.dia_geom.catheter.push(new_dummy_contour(0));
        gp.dia_geom.catheter.push(new_dummy_contour(1));
        gp.dia_geom.catheter.push(new_dummy_contour(2));
        gp.sys_geom.contours.push(new_dummy_contour(0));
        gp.sys_geom.contours.push(new_dummy_contour(1));
        gp.sys_geom.catheter.push(new_dummy_contour(0));
        gp.sys_geom.catheter.push(new_dummy_contour(1));

        println!("Contours dia geom: {:?}", gp.dia_geom.contours);
        let trimmed = gp.trim_geometries_same_length();

        assert_eq!(
            trimmed.dia_geom.contours.len(),
            trimmed.sys_geom.contours.len()
        );
        assert_eq!(
            trimmed.dia_geom.catheter.len(),
            trimmed.sys_geom.catheter.len()
        );

        // Verify IDs start at 0 and are consecutive
        for (i, contour) in trimmed.dia_geom.contours.iter().enumerate() {
            assert_eq!(contour.id, i as u32);
        }
        for (i, contour) in trimmed.sys_geom.contours.iter().enumerate() {
            assert_eq!(contour.id, i as u32);
        }
        for (i, catheter) in trimmed.dia_geom.catheter.iter().enumerate() {
            assert_eq!(catheter.id, i as u32);
        }
        for (i, catheter) in trimmed.sys_geom.catheter.iter().enumerate() {
            assert_eq!(catheter.id, i as u32);
        }
    }

    #[test]
    fn test_thickness_adjustment() {
        let dia = simple_geometry((0.0, 0.0), 0.0, (Some(2.0), None));
        let sys = simple_geometry((0.0, 0.0), 0.0, (None, Some(4.0)));
        let gp = GeometryPair {
            dia_geom: dia.clone(),
            sys_geom: sys.clone(),
        }
        .thickness_adjustment();
        let d = &gp.dia_geom.contours[0];
        let s = &gp.sys_geom.contours[0];
        assert_eq!(d.aortic_thickness.unwrap(), 2.0);
        assert_eq!(s.aortic_thickness.unwrap(), 2.0);
        assert_eq!(d.pulmonary_thickness.unwrap(), 4.0);
        assert_eq!(s.pulmonary_thickness.unwrap(), 4.0);
    }

    #[test]
    fn test_find_best_rotation_all_simple() {
        let dia = simple_geometry((0.0, 0.0), 0.0, (None, None));
        let p1 = ContourPoint {
            frame_index: 0,
            point_index: 0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            aortic: false,
        };
        let p2 = ContourPoint {
            frame_index: 0,
            point_index: 1,
            x: 1.0,
            y: 0.0,
            z: 1.0,
            aortic: false,
        };
        let p3 = ContourPoint {
            frame_index: 0,
            point_index: 2,
            x: 2.0,
            y: 0.0,
            z: 2.0,
            aortic: false,
        };
        let mut angle = -PI / 2.0;
        angle = angle;
        let rotate = |p: ContourPoint| ContourPoint {
            x: p.x * angle.cos() - p.y * angle.sin(),
            y: p.x * angle.sin() + p.y * angle.cos(),
            ..p
        };
        let mut sys_contour = Contour {
            id: 0,
            points: vec![rotate(p1), rotate(p2), rotate(p3)],
            centroid: (0.0, 0.0, 1.0),
            aortic_thickness: None,
            pulmonary_thickness: None,
        };
        sys_contour.sort_contour_points();
        let sys = Geometry {
            contours: vec![sys_contour],
            catheter: vec![],
            walls: vec![],
            reference_point: p1,
            label: "".into(),
        };

        let best = find_best_rotation_all(&dia, &sys, 0.01, PI.to_degrees() / 2.0);
        let expected = PI / 2.0;
        assert_relative_eq!(best.rem_euclid(2.0 * PI), expected, epsilon = 0.4);
    }
}
