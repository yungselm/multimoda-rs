use super::contour::{Contour, ContourType};
use super::contour_point::ContourPoint;
use std::collections::HashMap;
use std::f64::consts::PI;

#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub id: u32,
    pub centroid: (f64, f64, f64),
    // groundtruth, must exist!
    pub lumen: Contour,
    pub extras: HashMap<ContourType, Contour>,
    pub reference_point: Option<ContourPoint>,
}

impl Frame {
    /// Sets a value for all contours and reference point in the frame.
    /// You can set `id`, `original_frame`, `points`, or `centroid` for all contours and reference point.
    pub fn set_value(
        &mut self,
        id: Option<u32>,
        lumen_points: Option<Vec<ContourPoint>>,
        centroid: Option<(f64, f64, f64)>,
        z_value: Option<f64>,
    ) {
        if let Some(new_id) = id {
            self.id = new_id;
            self.lumen.id = new_id;
            for contour in self.extras.values_mut() {
                contour.id = new_id;
            }
        }
        if let Some(new_points) = lumen_points.clone() {
            self.lumen.points = new_points.clone();
            for contour in self.extras.values_mut() {
                contour.points = new_points.clone();
            }
        }
        if let Some(new_centroid) = centroid {
            self.lumen.centroid = Some(new_centroid);
            for contour in self.extras.values_mut() {
                contour.centroid = Some(new_centroid);
            }
            self.centroid = new_centroid;
        }
        if let Some(z_value) = z_value {
            for point in &mut self.lumen.points {
                point.z = z_value;
            }

            if let Some(ref mut centroid) = self.lumen.centroid {
                centroid.2 = z_value;
            }

            for contour in self.extras.values_mut() {
                for point in &mut contour.points {
                    point.z = z_value;
                }

                if let Some(ref mut centroid) = contour.centroid {
                    centroid.2 = z_value;
                }
            }

            if let Some(ref mut point) = self.reference_point {
                point.z = z_value;
            }

            self.centroid.2 = z_value;
        }
    }

    pub fn rotate_frame(&mut self, angle: f64) {
        if angle == 0.0 {
            return;
        }
        let center = (self.centroid.0, self.centroid.1);

        self.lumen.points = self
            .lumen
            .points
            .iter()
            .map(|p| p.rotate_point(angle, center))
            .collect();

        for contour in self.extras.values_mut() {
            contour.points = contour
                .points
                .iter()
                .map(|p| p.rotate_point(angle, center))
                .collect();
        }

        if let Some(ref_point) = &mut self.reference_point {
            *ref_point = ref_point.rotate_point(angle, center);
        }
    }

    pub fn translate_frame(&mut self, translation: (f64, f64, f64)) {
        let (dx, dy, dz) = translation;

        for p in self.lumen.points.iter_mut() {
            p.x += dx;
            p.y += dy;
            p.z += dz;
        }

        self.lumen.compute_centroid();

        for contour in self.extras.values_mut() {
            for p in contour.points.iter_mut() {
                p.x += dx;
                p.y += dy;
                p.z += dz;
            }

            contour.compute_centroid();
        }

        if let Some(ref_point) = &mut self.reference_point {
            ref_point.x += dx;
            ref_point.y += dy;
            ref_point.z += dz;
        }

        self.centroid.0 += dx;
        self.centroid.1 += dy;
        self.centroid.2 += dz;
    }

    pub fn sort_frame_points(&mut self) {
        self.lumen.sort_contour_points();

        for contour in self.extras.values_mut() {
            contour.sort_contour_points();
        }
    }

    pub fn rotate_frame_around_point(&mut self, angle_rad: f64, center: (f64, f64, f64)) {
        let cos_angle = angle_rad.cos();
        let sin_angle = angle_rad.sin();

        // Rotate lumen points
        for point in &mut self.lumen.points {
            let translated_x = point.x - center.0;
            let translated_y = point.y - center.1;

            point.x = center.0 + translated_x * cos_angle - translated_y * sin_angle;
            point.y = center.1 + translated_x * sin_angle + translated_y * cos_angle;
        }

        // Rotate frame centroid
        let translated_cx = self.centroid.0 - center.0;
        let translated_cy = self.centroid.1 - center.1;

        self.centroid.0 = center.0 + translated_cx * cos_angle - translated_cy * sin_angle;
        self.centroid.1 = center.1 + translated_cx * sin_angle + translated_cy * cos_angle;

        // Rotate extras if needed
        for contour in self.extras.values_mut() {
            for point in &mut contour.points {
                let translated_x = point.x - center.0;
                let translated_y = point.y - center.1;

                point.x = center.0 + translated_x * cos_angle - translated_y * sin_angle;
                point.y = center.1 + translated_x * sin_angle + translated_y * cos_angle;
            }
        }
    }

    pub fn create_catheter_points(
        points: &Vec<ContourPoint>,
        image_center: (f64, f64),
        radius: f64,
        n_points: u32,
    ) -> Vec<ContourPoint> {
        // Map to store unique frame indices and one associated z coordinate per frame.
        let mut frame_z: HashMap<u32, f64> = HashMap::new();
        for point in points {
            // Use the first encountered z-coordinate for each frame index.
            frame_z.entry(point.frame_index).or_insert(point.z);
        }

        let mut catheter_points = Vec::new();
        // Sort the frame indices to ensure a predictable order.
        let mut frames: Vec<u32> = frame_z.keys().cloned().collect();
        frames.sort();

        // Parameters for the catheter circle.
        let center_x = image_center.0;
        let center_y = image_center.1;
        let num_points = n_points;

        // For each unique frame, generate 20 catheter points around a circle.
        for frame in frames {
            let z = frame_z[&frame];
            for i in 0..num_points {
                let angle = 2.0 * PI * (i as f64) / (num_points as f64);
                let x = center_x + radius * angle.cos();
                let y = center_y + radius * angle.sin();
                catheter_points.push(ContourPoint {
                    frame_index: frame,
                    point_index: i,
                    x,
                    y,
                    z,
                    aortic: false,
                });
            }
        }
        catheter_points
    }
}

#[cfg(test)]
mod frame_tests {
    use super::*;
    use std::collections::HashMap;
    use std::f64::consts::PI;

    #[test]
    fn test_frame_rotate_with_eem_90deg() {
        // Build a frame with lumen and eem contours
        let mut extras: HashMap<ContourType, Contour> = HashMap::new();

        let eem = Contour {
            id: 2,
            original_frame: 2,
            points: vec![
                ContourPoint {
                    frame_index: 2,
                    point_index: 0,
                    x: -1.0,
                    y: 2.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 2,
                    point_index: 1,
                    x: 2.0,
                    y: 5.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 2,
                    point_index: 2,
                    x: 5.0,
                    y: 2.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 2,
                    point_index: 3,
                    x: 0.0,
                    y: -1.0,
                    z: 0.0,
                    aortic: false,
                },
            ],
            centroid: Some((2.0, 2.0, 0.0)),
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Eem,
        };

        extras.insert(ContourType::Eem, eem);

        let mut frame = Frame {
            id: 1,
            centroid: (1.0, 1.0, 0.0),
            lumen: Contour {
                id: 1,
                original_frame: 1,
                points: vec![
                    ContourPoint {
                        frame_index: 1,
                        point_index: 0,
                        x: 0.0,
                        y: 2.0,
                        z: 0.0,
                        aortic: false,
                    },
                    ContourPoint {
                        frame_index: 1,
                        point_index: 1,
                        x: 2.0,
                        y: 4.0,
                        z: 0.0,
                        aortic: false,
                    },
                    ContourPoint {
                        frame_index: 1,
                        point_index: 2,
                        x: 4.0,
                        y: 2.0,
                        z: 0.0,
                        aortic: false,
                    },
                    ContourPoint {
                        frame_index: 1,
                        point_index: 3,
                        x: 2.0,
                        y: 0.0,
                        z: 0.0,
                        aortic: false,
                    },
                ],
                centroid: Some((2.0, 2.0, 0.0)),
                aortic_thickness: None,
                pulmonary_thickness: None,
                kind: ContourType::Lumen,
            },
            extras,
            reference_point: Some(ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 0.0,
                y: 4.0,
                z: 0.0,
                aortic: false,
            }),
        };

        // --- store originals before any rotation ---
        let original_lumen: Vec<(f64, f64)> =
            frame.lumen.points.iter().map(|p| (p.x, p.y)).collect();
        let original_eem: Vec<(f64, f64)> = frame
            .extras
            .get(&ContourType::Eem)
            .expect("eem should exist")
            .points
            .iter()
            .map(|p| (p.x, p.y))
            .collect();
        let original_ref = frame.reference_point.map(|rp| (rp.x, rp.y));

        // Rotate by 90 degrees (pi/2) about frame.centroid (1.0,1.0)
        frame.rotate_frame(PI / 2.0);

        // Expected lumen points after rotation around (1,1):
        let expected_lumen = [(0.0, 0.0), (-2.0, 2.0), (0.0, 4.0), (2.0, 2.0)];

        // Expected eem points after rotation around (1,1):
        let expected_eem = [(0.0, -1.0), (-3.0, 2.0), (0.0, 5.0), (3.0, 0.0)];

        let eps = 1e-6;

        // Check lumen after first rotation
        for (i, p) in frame.lumen.points.iter().enumerate() {
            assert!(
                (p.x - expected_lumen[i].0).abs() < eps,
                "lumen x[{}] mismatch: {} vs {}",
                i,
                p.x,
                expected_lumen[i].0
            );
            assert!(
                (p.y - expected_lumen[i].1).abs() < eps,
                "lumen y[{}] mismatch: {} vs {}",
                i,
                p.y,
                expected_lumen[i].1
            );
        }

        // Retrieve rotated eem from extras and check
        let rotated_eem = frame
            .extras
            .get(&ContourType::Eem)
            .expect("eem contour should exist in frame.extras");

        for (i, p) in rotated_eem.points.iter().enumerate() {
            assert!(
                (p.x - expected_eem[i].0).abs() < eps,
                "eem x[{}] mismatch: {} vs {}",
                i,
                p.x,
                expected_eem[i].0
            );
            assert!(
                (p.y - expected_eem[i].1).abs() < eps,
                "eem y[{}] mismatch: {} vs {}",
                i,
                p.y,
                expected_eem[i].1
            );
        }

        // Also check reference point rotated correctly (optional)
        if let Some(rp) = frame.reference_point {
            // original ref point was (0,4): after rotation:
            // x' = 1 - (4-1) = -2 ; y' = 1 + (0-1) = 0  => (-2, 0)
            assert!((rp.x - (-2.0)).abs() < eps, "reference_point x mismatch");
            assert!((rp.y - 0.0).abs() < eps, "reference_point y mismatch");
        }

        // --- rotate back by -90 degrees and check we return to originals ---
        frame.rotate_frame(-(PI / 2.0));

        // Check lumen returned to original positions
        for (i, p) in frame.lumen.points.iter().enumerate() {
            let (ox, oy) = original_lumen[i];
            assert!(
                (p.x - ox).abs() < eps,
                "lumen after back-rotation x[{}] mismatch: {} vs original {}",
                i,
                p.x,
                ox
            );
            assert!(
                (p.y - oy).abs() < eps,
                "lumen after back-rotation y[{}] mismatch: {} vs original {}",
                i,
                p.y,
                oy
            );
        }

        // Check eem returned to original positions
        let eem_after = frame.extras.get(&ContourType::Eem).expect("eem must exist");
        for (i, p) in eem_after.points.iter().enumerate() {
            let (ox, oy) = original_eem[i];
            assert!(
                (p.x - ox).abs() < eps,
                "eem after back-rotation x[{}] mismatch: {} vs original {}",
                i,
                p.x,
                ox
            );
            assert!(
                (p.y - oy).abs() < eps,
                "eem after back-rotation y[{}] mismatch: {} vs original {}",
                i,
                p.y,
                oy
            );
        }

        // Check reference point returned to original (if present)
        match (original_ref, frame.reference_point) {
            (Some((ox, oy)), Some(rp)) => {
                assert!((rp.x - ox).abs() < eps, "reference_point x not restored");
                assert!((rp.y - oy).abs() < eps, "reference_point y not restored");
            }
            (None, None) => {}
            _ => panic!("reference_point presence/absence changed during rotations"),
        }
    }

    #[test]
    fn test_frame_rotate_around_point() {
        let mut frame = Frame {
            id: 1,
            centroid: (0.0, 0.0, 0.0),
            lumen: Contour {
                id: 1,
                original_frame: 1,
                points: vec![
                    ContourPoint {
                        frame_index: 1,
                        point_index: 0,
                        x: 1.0,
                        y: 0.0,
                        z: 0.0,
                        aortic: false,
                    },
                    ContourPoint {
                        frame_index: 1,
                        point_index: 1,
                        x: 0.0,
                        y: 1.0,
                        z: 0.0,
                        aortic: false,
                    },
                    ContourPoint {
                        frame_index: 1,
                        point_index: 2,
                        x: -1.0,
                        y: 0.0,
                        z: 0.0,
                        aortic: false,
                    },
                    ContourPoint {
                        frame_index: 1,
                        point_index: 3,
                        x: 0.0,
                        y: -1.0,
                        z: 0.0,
                        aortic: false,
                    },
                ],
                centroid: Some((0.0, 0.0, 0.0)),
                aortic_thickness: None,
                pulmonary_thickness: None,
                kind: ContourType::Lumen,
            },
            extras: HashMap::new(),
            reference_point: None,
        };

        // Rotate 180 degrees (PI) around point (1, 1)
        frame.rotate_frame_around_point(PI, (1.0, 1.0, 0.0));

        let expected_points = [
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 1.0,
                y: 2.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 2.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 2,
                x: 3.0,
                y: 2.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 3,
                x: 2.0,
                y: 3.0,
                z: 0.0,
                aortic: false,
            },
        ];

        for (i, point) in frame.lumen.points.iter().enumerate() {
            assert!(
                (point.x - expected_points[i].x).abs() < 1e-6,
                "x mismatch at {}: expected {}, got {}",
                i,
                expected_points[i].x,
                point.x
            );
            assert!(
                (point.y - expected_points[i].y).abs() < 1e-6,
                "y mismatch at {}: expected {}, got {}",
                i,
                expected_points[i].y,
                point.y
            );
        }
    }

    #[test]
    fn test_frame_translate_with_eem_and_reference() {
        let mut extras: HashMap<ContourType, Contour> = HashMap::new();

        let eem = Contour {
            id: 2,
            original_frame: 2,
            points: vec![
                ContourPoint {
                    frame_index: 2,
                    point_index: 0,
                    x: -1.0,
                    y: 2.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 2,
                    point_index: 1,
                    x: 2.0,
                    y: 5.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 2,
                    point_index: 2,
                    x: 5.0,
                    y: 2.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 2,
                    point_index: 3,
                    x: 0.0,
                    y: -1.0,
                    z: 0.0,
                    aortic: false,
                },
            ],
            centroid: Some((1.5, 2.0, 0.0)),
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Eem,
        };
        extras.insert(ContourType::Eem, eem);

        let mut frame = Frame {
            id: 1,
            centroid: (1.0, 1.0, 0.0),
            lumen: Contour {
                id: 1,
                original_frame: 1,
                points: vec![
                    ContourPoint {
                        frame_index: 1,
                        point_index: 0,
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                        aortic: false,
                    },
                    ContourPoint {
                        frame_index: 1,
                        point_index: 1,
                        x: 2.0,
                        y: 0.0,
                        z: 0.0,
                        aortic: false,
                    },
                    ContourPoint {
                        frame_index: 1,
                        point_index: 2,
                        x: 2.0,
                        y: 2.0,
                        z: 0.0,
                        aortic: false,
                    },
                    ContourPoint {
                        frame_index: 1,
                        point_index: 3,
                        x: 0.0,
                        y: 2.0,
                        z: 0.0,
                        aortic: false,
                    },
                ],
                centroid: Some((1.0, 1.0, 0.0)),
                aortic_thickness: None,
                pulmonary_thickness: None,
                kind: ContourType::Lumen,
            },
            extras,
            reference_point: Some(ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 0.5,
                y: -0.5,
                z: 0.0,
                aortic: false,
            }),
        };

        frame.translate_frame((1.0, 2.0, 3.0));
        assert_eq!(frame.centroid, (2.0, 3.0, 3.0));

        let expected_lumen = [
            (1.0, 2.0, 3.0),
            (3.0, 2.0, 3.0),
            (3.0, 4.0, 3.0),
            (1.0, 4.0, 3.0),
        ];
        for (i, p) in frame.lumen.points.iter().enumerate() {
            assert_eq!(
                (p.x, p.y, p.z),
                expected_lumen[i],
                "lumen point {i} mismatch"
            );
        }

        let rotated_eem = frame
            .extras
            .get(&ContourType::Eem)
            .expect("eem contour present");
        let expected_eem = [
            (0.0, 4.0, 3.0),
            (3.0, 7.0, 3.0),
            (6.0, 4.0, 3.0),
            (1.0, 1.0, 3.0),
        ];
        for (i, p) in rotated_eem.points.iter().enumerate() {
            assert_eq!((p.x, p.y, p.z), expected_eem[i], "eem point {i} mismatch");
        }

        // Reference point should also have been translated
        if let Some(rp) = &frame.reference_point {
            assert_eq!((rp.x, rp.y, rp.z), (1.5, 1.5, 3.0));
        } else {
            panic!("reference_point expected but missing");
        }
    }

    #[test]
    fn test_create_catheter_points() {
        let points = vec![ContourPoint {
            frame_index: 1,
            point_index: 0,
            x: 0.0,
            y: 0.0,
            z: 5.0,
            aortic: false,
        }];

        let catheter_points = Frame::create_catheter_points(&points, (4.5, 4.5), 0.5, 20);
        assert_eq!(catheter_points.len(), 20);

        for point in catheter_points {
            assert_eq!(point.frame_index, 1);
            assert_eq!(point.z, 5.0);
            let dx = point.x - 4.5;
            let dy = point.y - 4.5;
            let dist = (dx * dx + dy * dy).sqrt();
            assert!((dist - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_frame_set_value_updates_all_targets() {
        let mut extras: HashMap<ContourType, Contour> = HashMap::new();
        let initial_eem = Contour {
            id: 7,
            original_frame: 723,
            points: vec![ContourPoint {
                frame_index: 723,
                point_index: 0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            }],
            centroid: Some((0.0, 0.0, 0.0)),
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Eem,
        };
        extras.insert(ContourType::Eem, initial_eem);

        let mut frame = Frame {
            id: 99,
            centroid: (0.0, 0.0, 0.0),
            lumen: Contour {
                id: 1,
                original_frame: 723,
                points: vec![ContourPoint {
                    frame_index: 1,
                    point_index: 0,
                    x: 10.0,
                    y: 10.0,
                    z: 10.0,
                    aortic: false,
                }],
                centroid: Some((10.0, 10.0, 10.0)),
                aortic_thickness: None,
                pulmonary_thickness: None,
                kind: ContourType::Lumen,
            },
            extras,
            reference_point: Some(ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 5.0,
                y: 5.0,
                z: 5.0,
                aortic: false,
            }),
        };

        // New values to set
        let new_id = 42;
        let new_points = vec![
            ContourPoint {
                frame_index: 5,
                point_index: 0,
                x: 1.0,
                y: 2.0,
                z: 3.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 5,
                point_index: 1,
                x: 4.0,
                y: 5.0,
                z: 6.0,
                aortic: false,
            },
        ];
        let new_centroid = (7.0_f64, 8.0_f64, 9.0_f64);
        let new_z_value = 123.0_f64;

        frame.set_value(
            Some(new_id),
            Some(new_points.clone()),
            Some(new_centroid),
            Some(new_z_value),
        );

        assert_eq!(frame.id, 42);
        assert_eq!(frame.lumen.id, new_id);
        assert_eq!(frame.lumen.original_frame, 723);

        let eem = frame.extras.get(&ContourType::Eem).expect("eem present");
        assert_eq!(eem.id, new_id);
        assert_eq!(eem.original_frame, 723);

        assert_eq!(frame.lumen.points.len(), new_points.len());
        for (i, p) in frame.lumen.points.iter().enumerate() {
            assert_eq!(p.x, new_points[i].x);
            assert_eq!(p.y, new_points[i].y);
            assert_eq!(p.z, new_z_value);
        }

        let eem_points = &eem.points;
        assert_eq!(eem_points.len(), new_points.len());
        for (i, p) in eem_points.iter().enumerate() {
            assert_eq!(p.x, new_points[i].x);
            assert_eq!(p.y, new_points[i].y);
            assert_eq!(p.z, new_z_value);
        }

        assert_eq!(
            frame.lumen.centroid.unwrap(),
            (new_centroid.0, new_centroid.1, new_z_value)
        );
        assert_eq!(
            frame.centroid,
            (new_centroid.0, new_centroid.1, new_z_value)
        );
        let eem_centroid = eem.centroid.expect("eem centroid present");
        assert_eq!(eem_centroid, (new_centroid.0, new_centroid.1, new_z_value));

        let rp = frame.reference_point.expect("reference point present");
        assert_eq!(rp.z, new_z_value);
    }
}
