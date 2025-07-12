use rayon::prelude::*;

use crate::io::Geometry;
use crate::processing::contours::hausdorff_distance;

use super::contours::align_frames_in_geometry;
use crate::io::input::{Contour, ContourPoint};

#[derive(Clone)]
pub struct GeometryPair {
    pub dia_geom: Geometry,
    pub sys_geom: Geometry,
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

    /// aligns the frames in each geomtery by rotating them based on best Hausdorff distance
    /// then translates systolic contours to the diastolic contours, aligns z-coordinates and
    /// trims them to same length.
    pub fn process_geometry_pair(
        self,
        steps_best_rotation: usize,
        range_rotation_rad: f64,
        align_inside: bool,
    ) -> GeometryPair {
        let diastole = if align_inside {
            align_frames_in_geometry(self.dia_geom, steps_best_rotation, range_rotation_rad)
        } else {
            self.dia_geom
        };
        let mut systole = if align_inside {
            align_frames_in_geometry(self.sys_geom, steps_best_rotation, range_rotation_rad)
        } else {
            self.sys_geom
        };

        Self::translate_contours_to_match(&diastole, &mut systole);

        // Adjust the z-coordinates of systolic contours. (later replaceed by adjust_z_coordinates)
        Self::apply_z_transformation(&diastole, &mut systole);

        let best_rotation_angle = find_best_rotation_all(
            &diastole,
            &systole,
            steps_best_rotation, // number of candidate steps (e.g. 200 or 400)
            range_rotation_rad,  // rotation range (e.g. 1.05 for ~±60°)
        );

        for ref mut contour in systole
            .contours
            .iter_mut()
            .chain(systole.catheter.iter_mut())
        {
            contour.rotate_contour(best_rotation_angle);
        }
        GeometryPair {
            dia_geom: diastole,
            sys_geom: systole,
        }
    }

    fn translate_contours_to_match(dia: &Geometry, sys: &mut Geometry) {
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

    fn apply_z_transformation(dia: &Geometry, sys: &mut Geometry) {
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
        let dia_len = self.dia_geom.contours.len();
        let sys_len = self.sys_geom.contours.len();
        let min_len = std::cmp::min(dia_len, sys_len);

        if dia_len > min_len {
            self.dia_geom.contours.drain(0..(dia_len - min_len));
        }
        if sys_len > min_len {
            self.sys_geom.contours.drain(0..(sys_len - min_len));
        }

        let dia_catheter_len = self.dia_geom.catheter.len();
        let sys_catheter_len = self.sys_geom.catheter.len();
        let min_catheter_len = std::cmp::min(dia_catheter_len, sys_catheter_len);

        if dia_catheter_len > min_catheter_len {
            self.dia_geom
                .catheter
                .drain(0..(dia_catheter_len - min_catheter_len));
        }
        if sys_catheter_len > min_catheter_len {
            self.sys_geom
                .catheter
                .drain(0..(sys_catheter_len - min_catheter_len));
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

pub fn find_best_rotation_all(
    diastole: &Geometry,
    systole: &Geometry,
    steps: usize,
    range: f64,
) -> f64 {
    println!("---------------------Finding optimal rotation Diastole/Systole---------------------");
    let increment = (2.0 * range) / steps as f64;

    (0..=steps)
        .into_par_iter()
        .map(|i| {
            let angle = -range + i as f64 * increment;
            let total_distance: f64 = diastole
                .contours
                .par_iter()
                .zip(systole.contours.par_iter())
                .map(|(d_contour, s_contour)| {
                    assert_eq!(d_contour.id, s_contour.id, "Mismatched contour IDs");

                    // Rotate each point in systole contour
                    let rotated_points: Vec<ContourPoint> = s_contour
                        .points
                        .iter()
                        .map(|p| {
                            let x = p.x * angle.cos() - p.y * angle.sin();
                            let y = p.x * angle.sin() + p.y * angle.cos();
                            ContourPoint { x, y, ..*p }
                        })
                        .collect();

                    hausdorff_distance(&d_contour.points, &rotated_points)
                })
                .sum();

            let avg_distance = total_distance / diastole.contours.len() as f64;

            println!("Angle: {:.3} rad, Avg Distance: {:.3}", angle, avg_distance);
            (angle, avg_distance)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(angle, _)| angle)
        .unwrap_or(0.0)
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
        gp = gp.process_geometry_pair(1, 0.0, true);
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
        gp = gp.process_geometry_pair(1, 0.0, true).adjust_z_coordinates();
        for contour in gp.dia_geom.contours.iter() {
            assert!(contour.centroid.2.is_finite());
        }
        for contour in gp.sys_geom.contours.iter() {
            assert!(contour.centroid.2.is_finite());
        }
    }

    #[test]
    fn test_trim_geometries_same_length() {
        let mut gp = GeometryPair {
            dia_geom: simple_geometry((0.0, 0.0), 0.0, (None, None)),
            sys_geom: simple_geometry((0.0, 0.0), 0.0, (None, None)),
        };
        gp.dia_geom.contours.push(new_dummy_contour(1));
        gp.sys_geom.contours.push(new_dummy_contour(1));
        gp.dia_geom.contours.push(new_dummy_contour(2));
        gp.dia_geom.catheter.push(new_dummy_contour(0));
        gp.sys_geom.catheter.push(new_dummy_contour(0));
        gp.sys_geom.catheter.push(new_dummy_contour(1));

        let trimmed = gp.trim_geometries_same_length();
        assert_eq!(
            trimmed.dia_geom.contours.len(),
            trimmed.sys_geom.contours.len()
        );
        assert_eq!(
            trimmed.dia_geom.catheter.len(),
            trimmed.sys_geom.catheter.len()
        );
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
        let angle = -PI / 2.0;
        let rotate = |p: ContourPoint| ContourPoint {
            x: p.x * angle.cos() - p.y * angle.sin(),
            y: p.x * angle.sin() + p.y * angle.cos(),
            ..p
        };
        let mut sys_contour = Contour {
            id: 0,
            points: vec![rotate(p1), rotate(p2), rotate(p3)],
            centroid: (0.0, 1.0, 1.0),
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

        let best = find_best_rotation_all(&dia, &sys, 4, PI);
        let expected = PI / 2.0;
        assert_relative_eq!(best.rem_euclid(2.0 * PI), expected, epsilon = 0.4);
    }
}
