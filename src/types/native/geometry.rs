use super::contour::{Contour, ContourType};
use super::contour_point::ContourPoint;
use super::frame::Frame;
use super::record::Record;
use super::Transform;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct Geometry {
    pub frames: Vec<Frame>,
    pub label: String,
}

impl Transform for Geometry {
    fn translate(mut self, dx: f64, dy: f64, dz: f64) -> Self {
        self.frames = self
            .frames
            .into_iter()
            .map(|f| f.translate(dx, dy, dz))
            .collect();
        self
    }

    fn rotate(mut self, angle: f64, center: (f64, f64)) -> Self {
        if angle == 0.0 {
            return self;
        }
        self.frames = self
            .frames
            .into_iter()
            .map(|f| f.rotate(angle, center))
            .collect();
        self
    }
}

impl Geometry {
    pub fn new() -> anyhow::Result<Self> {
        todo!()
    }

    pub fn find_proximal_end_idx(&self) -> usize {
        let n = self.frames.len();
        if n == 0 {
            return 0;
        }

        if n == 1 {
            return self.frames[0].lumen.id as usize;
        }

        // simple check for now, just take the frame from two ends with highest original frame id
        let proximal_idx =
            if self.frames[0].lumen.original_frame > self.frames[n - 1].lumen.original_frame {
                self.frames[0].lumen.id
            } else {
                self.frames[n - 1].lumen.id
            };
        proximal_idx as usize
    }

    pub fn find_ref_frame_idx(&self) -> anyhow::Result<usize> {
        for frame in self.frames.iter() {
            if frame.reference_point.is_some() {
                return Ok(frame.id as usize);
            }
        }
        Err(anyhow::anyhow!("No reference point found in any frame"))
    }

    /// Reorder frames based on order of Vec<Record>
    pub fn reorder_frames(&mut self, records: &[Record], diastole: bool) {
        use std::mem;

        let filtered = Self::filter_records(records, diastole);

        let mut orig_z_map: HashMap<u32, f64> = HashMap::new();
        for fr in &self.frames {
            let orig = fr.lumen.original_frame;
            if let Some(p) = fr.lumen.points.first() {
                orig_z_map.entry(orig).or_insert(p.z);
            }
        }

        // Move existing frames into a lookup by original_frame for reordering
        let old_frames = mem::take(&mut self.frames);
        let mut frame_map: HashMap<u32, Frame> = HashMap::new();
        for f in old_frames.into_iter() {
            frame_map.insert(f.lumen.original_frame, f);
        }

        let mut new_frames: Vec<Frame> = Vec::with_capacity(frame_map.len());
        let mut used_originals: HashSet<u32> = HashSet::new();

        for &orig_id in &filtered {
            if let Some(frame) = frame_map.remove(&orig_id) {
                new_frames.push(frame);
                used_originals.insert(orig_id);
            }
        }

        let mut remaining_frames: Vec<Frame> = frame_map.into_values().collect();
        remaining_frames.sort_by_key(|frame| frame.lumen.original_frame);
        new_frames.extend(remaining_frames);

        for (new_idx, frame) in new_frames.iter_mut().enumerate() {
            let new_id = new_idx as u32;

            let orig = frame.lumen.original_frame;
            let z_value = *orig_z_map.get(&orig).unwrap_or(&(new_id as f64));

            frame.id = new_id;

            frame.lumen.id = new_id;
            for p in frame.lumen.points.iter_mut() {
                p.frame_index = new_id;
                p.z = z_value;
            }
            if let Some(ref mut c) = frame.lumen.centroid {
                c.2 = z_value;
            }

            // Extras contours
            for contour in frame.extras.values_mut() {
                contour.id = new_id;
                for p in contour.points.iter_mut() {
                    p.frame_index = new_id;
                    p.z = z_value;
                }
                if let Some(ref mut c) = contour.centroid {
                    c.2 = z_value;
                }
            }

            // Reference point
            if let Some(ref mut rp) = frame.reference_point {
                rp.z = z_value;
            }

            frame.centroid.2 = z_value;
        }

        self.frames = new_frames;
    }

    fn filter_records(records: &[Record], diastole: bool) -> Vec<u32> {
        let phase = if diastole { "D" } else { "S" };
        let filtered: Vec<u32> = records
            .iter()
            .filter(|r| r.phase == phase)
            .map(|r| r.frame)
            .collect();

        filtered
    }

    /// Smooths the x and y coordinates of the lumen, eem, and wall using a 3‐point moving average.
    ///
    /// For each point i in contour j, the new x and y values are computed as:
    ///     new_x = (prev_contour[i].x + current_contour[i].x + next_contour[i].x) / 3.0
    ///     new_y = (prev_contour[i].y + current_contour[i].y + next_contour[i].y) / 3.0
    /// while the z coordinate remains unchanged (taken from the current contour).
    ///
    /// For the first and last contours, the current contour is used twice to simulate a mirror effect.
    pub fn smooth_frames(mut self) -> Geometry {
        let mut smoothed_frames = Vec::with_capacity(self.frames.len());

        for i in 0..self.frames.len() {
            let mut current_frame = self.frames[i].clone();
            let point_count = current_frame.lumen.points.len();

            // Helper closure to smooth a contour
            let smooth_contour = |current: &Contour, prev: &Contour, next: &Contour| -> Contour {
                let mut new_points = Vec::with_capacity(point_count);

                for j in 0..point_count {
                    let curr_point = &current.points[j];
                    let prev_point = &prev.points[j];
                    let next_point = &next.points[j];

                    let avg_x = (prev_point.x + curr_point.x + next_point.x) / 3.0;
                    let avg_y = (prev_point.y + curr_point.y + next_point.y) / 3.0;

                    new_points.push(ContourPoint {
                        frame_index: curr_point.frame_index,
                        point_index: curr_point.point_index,
                        x: avg_x,
                        y: avg_y,
                        z: curr_point.z,
                        aortic: curr_point.aortic,
                    });
                }

                let mut cont = Contour {
                    id: current.id,
                    original_frame: current.original_frame,
                    points: new_points,
                    centroid: None,
                    aortic_thickness: current.aortic_thickness,
                    pulmonary_thickness: current.pulmonary_thickness,
                    kind: current.kind,
                };

                cont.compute_centroid();
                cont
            };

            // Smooth lumen contour
            let prev_frame = if i == 0 {
                &self.frames[i]
            } else {
                &self.frames[i - 1]
            };
            let next_frame = if i == self.frames.len() - 1 {
                &self.frames[i]
            } else {
                &self.frames[i + 1]
            };
            current_frame.lumen =
                smooth_contour(&current_frame.lumen, &prev_frame.lumen, &next_frame.lumen);

            // Smooth EEM and Wall contours if they exist
            for kind in [ContourType::Eem, ContourType::Wall] {
                if let Some(current_contour) = current_frame.extras.get(&kind) {
                    if let (Some(prev_contour), Some(next_contour)) =
                        (prev_frame.extras.get(&kind), next_frame.extras.get(&kind))
                    {
                        let smoothed = smooth_contour(current_contour, prev_contour, next_contour);
                        current_frame.extras.insert(kind, smoothed);
                    }
                }
            }

            smoothed_frames.push(current_frame);
        }

        self.frames = smoothed_frames;
        self
    }

    pub fn rotate_geometry(&mut self, angle_rad: f64) {
        if angle_rad == 0.0 {
            return;
        }
        for frame in self.frames.iter_mut() {
            let center = (frame.centroid.0, frame.centroid.1);
            frame.rotate_mut(angle_rad, center);
            frame.sort_frame_points();
        }
    }

    /// Rotate all frame contour point Vecs so that the point with the highest
    /// Z-value in frame 0's lumen becomes Vec index 0.  The same rotation
    /// offset is applied to every contour in every frame, and `point_index`
    /// fields are reassigned sequentially (0, 1, 2, …).
    /// X/Y/Z coordinates are never modified.
    pub fn sort_frame_points_by_z(&mut self) {
        let shift = match self.frames.first() {
            Some(f) => f
                .lumen
                .points
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.z.partial_cmp(&b.z).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0),
            None => return,
        };

        for frame in self.frames.iter_mut() {
            frame.lumen.rotate_and_reindex(shift);
            for contour in frame.extras.values_mut() {
                contour.rotate_and_reindex(shift);
            }
        }
    }

    pub fn translate_geometry(&mut self, translation: (f64, f64, f64)) {
        let (dx, dy, dz) = translation;
        for frame in self.frames.iter_mut() {
            frame.translate_mut(dx, dy, dz);
        }
    }

    pub fn insert_frame(&mut self, frame: Frame, idx: Option<usize>) {
        // Find insertion position based on z coordinate
        let z = frame.centroid.2;
        let pos: usize = if let Some(i) = idx {
            i
        } else {
            self.frames
                .iter()
                .position(|f| f.centroid.2 > z)
                .unwrap_or(self.frames.len())
        };

        self.frames.insert(pos, frame);

        for (idx, frame) in self.frames.iter_mut().enumerate() {
            let new_id = idx as u32;

            frame.id = new_id;

            frame.lumen.id = new_id;
            for point in &mut frame.lumen.points {
                point.frame_index = new_id;
            }

            for contour in frame.extras.values_mut() {
                contour.id = new_id;
                for point in &mut contour.points {
                    point.frame_index = new_id;
                }
            }
            if let Some(ref mut rp) = frame.reference_point {
                rp.frame_index = new_id;
            }
        }
    }

    /// Rotate frames so proximal end is at index 0, then reassign z-values
    /// (smallest z goes to frame 0) and propagate z/frame indices to all points
    /// and contour centroids. Reassigns contour ids sequentially (but preserves
    /// Contour::original_frame). Renumbers Frame.id to match new vector index.
    pub fn ensure_proximal_at_position_zero(&mut self) {
        use std::cmp::Ordering;

        let n = self.frames.len();
        if n == 0 {
            return;
        }

        let proximal_idx = self.find_proximal_end_idx();
        let proximal_idx = proximal_idx.min(self.frames.len() - 1);

        if proximal_idx != 0 {
            let mut temp_frames = self.frames.clone();
            temp_frames.reverse();

            self.frames = temp_frames;
        }

        let mut zs: Vec<f64> = self.frames.iter().map(|f| f.centroid.2).collect();
        zs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let mut next_contour_id: u32 = 0;

        for (idx, frame) in self.frames.iter_mut().enumerate() {
            let new_frame_id = idx as u32;
            frame.id = new_frame_id;

            let assigned_z = zs.get(idx).copied().unwrap_or(frame.centroid.2);
            frame.centroid.2 = assigned_z;

            frame.lumen.id = next_contour_id;
            next_contour_id = next_contour_id.wrapping_add(1);

            for p in frame.lumen.points.iter_mut() {
                p.z = assigned_z;
            }
            if let Some(ref mut c) = frame.lumen.centroid {
                c.2 = assigned_z;
            }

            for contour in frame.extras.values_mut() {
                contour.id = next_contour_id;
                next_contour_id = next_contour_id.wrapping_add(1);

                for p in contour.points.iter_mut() {
                    p.z = assigned_z;
                }
                if let Some(ref mut c) = contour.centroid {
                    c.2 = assigned_z;
                }
            }

            if let Some(ref mut rp) = frame.reference_point {
                rp.z = assigned_z;
            }
        }
    }

    pub fn center_to_contour(&mut self, contour_type: ContourType) {
        let n = self.frames.len();
        if n == 0 {
            return;
        }

        let first_frame = &mut self.frames[0];
        let mut reference_centroid = match contour_type {
            ContourType::Lumen => {
                first_frame.lumen.compute_centroid();
                first_frame.lumen.centroid.unwrap_or(first_frame.centroid)
            }
            _ => {
                if let Some(contour) = first_frame.extras.get_mut(&contour_type) {
                    contour.compute_centroid();
                    contour.centroid.unwrap_or(first_frame.centroid)
                } else {
                    first_frame.centroid
                }
            }
        };

        for i in 1..n {
            let current_frame = &mut self.frames[i];

            let current_centroid = match contour_type {
                ContourType::Lumen => {
                    current_frame.lumen.compute_centroid();
                    current_frame
                        .lumen
                        .centroid
                        .unwrap_or(current_frame.centroid)
                }
                _ => {
                    if let Some(contour) = current_frame.extras.get_mut(&contour_type) {
                        contour.compute_centroid();
                        contour.centroid.unwrap_or(current_frame.centroid)
                    } else {
                        current_frame.centroid
                    }
                }
            };

            let translation = (
                reference_centroid.0 - current_centroid.0,
                reference_centroid.1 - current_centroid.1,
                0.0,
            );

            let (dx, dy, dz) = translation;
            current_frame.translate_mut(dx, dy, dz);

            reference_centroid = (
                reference_centroid.0,
                reference_centroid.1,
                reference_centroid.2,
            );
        }
    }
}

#[cfg(test)]
mod geometry_tests {
    use super::*;
    use crate::intravascular::utils::test_utils::dummy_geometry;
    use approx::assert_relative_eq;

    #[test]
    fn rotate_contour_back_and_forth() {
        let geometry = dummy_geometry();
        let mut geometry_rotate = geometry.clone();
        let rotation_deg: f64 = 15.0;

        geometry_rotate.rotate_geometry(rotation_deg.to_radians());
        geometry_rotate.rotate_geometry(-rotation_deg.to_radians());

        assert_eq!(
            geometry.frames[0].lumen.points[0],
            geometry_rotate.frames[0].lumen.points[0]
        );

        let center = (
            geometry_rotate.frames[0].centroid.0,
            geometry_rotate.frames[0].centroid.1,
        );
        geometry_rotate.frames[0].rotate_mut(rotation_deg.to_radians(), center);
        geometry_rotate.frames[0].rotate_mut(-rotation_deg.to_radians(), center);

        assert_eq!(
            geometry.frames[0].lumen.points[0],
            geometry_rotate.frames[0].lumen.points[0]
        );

        geometry_rotate.frames[0].lumen.compute_centroid();
        let lumen_center = {
            let (cx, cy, _) = geometry_rotate.frames[0].lumen.centroid.unwrap();
            (cx, cy)
        };
        geometry_rotate.frames[0]
            .lumen
            .rotate_mut(rotation_deg.to_radians(), lumen_center);
        geometry_rotate.frames[0]
            .lumen
            .rotate_mut(-rotation_deg.to_radians(), lumen_center);

        assert_eq!(
            geometry.frames[0].lumen.points[0],
            geometry_rotate.frames[0].lumen.points[0]
        );

        let center = geometry_rotate.frames[0].centroid;
        geometry_rotate.frames[0].lumen.points[0]
            .rotate_mut(rotation_deg.to_radians(), (center.0, center.1));
        geometry_rotate.frames[0].lumen.points[0]
            .rotate_mut(-rotation_deg.to_radians(), (center.0, center.1));

        assert_eq!(
            geometry.frames[0].lumen.points[0],
            geometry_rotate.frames[0].lumen.points[0]
        );
    }

    #[test]
    fn test_geometry_idx_and_ensure() {
        let frames = vec![
            Frame {
                id: 0,
                centroid: (1.0, 1.0, 0.0),
                lumen: Contour {
                    id: 0,
                    original_frame: 621,
                    points: Vec::new(),
                    centroid: None,
                    aortic_thickness: None,
                    pulmonary_thickness: None,
                    kind: ContourType::Lumen,
                },
                extras: HashMap::new(),
                reference_point: None,
            },
            Frame {
                id: 1,
                centroid: (1.0, 1.0, 1.0),
                lumen: Contour {
                    id: 1,
                    original_frame: 678,
                    points: Vec::new(),
                    centroid: None,
                    aortic_thickness: None,
                    pulmonary_thickness: None,
                    kind: ContourType::Lumen,
                },
                extras: HashMap::new(),
                reference_point: Some(ContourPoint {
                    frame_index: 678,
                    point_index: 2,
                    x: 1.0,
                    y: 3.0,
                    z: 2.0,
                    aortic: false,
                }),
            },
            Frame {
                id: 2,
                centroid: (1.0, 1.0, 2.0),
                lumen: Contour {
                    id: 2,
                    original_frame: 717,
                    points: Vec::new(),
                    centroid: None,
                    aortic_thickness: None,
                    pulmonary_thickness: None,
                    kind: ContourType::Lumen,
                },
                extras: HashMap::new(),
                reference_point: None,
            },
        ];

        let mut geom = Geometry {
            frames,
            label: "test".to_string(),
        };
        let prox_idx = geom.find_proximal_end_idx();
        let ref_idx = geom.find_ref_frame_idx().unwrap();

        assert_eq!(prox_idx, 2);
        assert_eq!(geom.frames[prox_idx].lumen.original_frame, 717);
        assert_eq!(geom.frames[prox_idx].centroid.2, 2.0);
        assert_eq!(ref_idx, 1);
        assert_eq!(geom.frames[ref_idx].lumen.original_frame, 678);
        assert_eq!(geom.frames[ref_idx].centroid.2, 1.0);

        geom.ensure_proximal_at_position_zero();
        let prox_idx = geom.find_proximal_end_idx();
        let ref_idx = geom.find_ref_frame_idx().unwrap();
        assert_eq!(prox_idx, 0);
        assert_eq!(geom.frames[prox_idx].lumen.original_frame, 717);
        assert_eq!(geom.frames[prox_idx].centroid.2, 0.0);
        assert_eq!(ref_idx, 1);
        assert_eq!(geom.frames[ref_idx].lumen.original_frame, 678);
        assert_eq!(geom.frames[ref_idx].centroid.2, 1.0);
    }

    #[test]
    fn test_reorder_geometry() {
        use crate::types::native::record::Record;
        let frames = vec![
            Frame {
                id: 0,
                centroid: (1.0, 1.0, 0.0),
                lumen: Contour {
                    id: 0,
                    original_frame: 621,
                    points: Vec::new(),
                    centroid: None,
                    aortic_thickness: None,
                    pulmonary_thickness: None,
                    kind: ContourType::Lumen,
                },
                extras: HashMap::new(),
                reference_point: None,
            },
            Frame {
                id: 1,
                centroid: (1.0, 1.0, 1.0),
                lumen: Contour {
                    id: 1,
                    original_frame: 678,
                    points: Vec::new(),
                    centroid: None,
                    aortic_thickness: None,
                    pulmonary_thickness: None,
                    kind: ContourType::Lumen,
                },
                extras: HashMap::new(),
                reference_point: Some(ContourPoint {
                    frame_index: 678,
                    point_index: 2,
                    x: 1.0,
                    y: 3.0,
                    z: 2.0,
                    aortic: false,
                }),
            },
            Frame {
                id: 2,
                centroid: (1.0, 1.0, 2.0),
                lumen: Contour {
                    id: 2,
                    original_frame: 717,
                    points: Vec::new(),
                    centroid: None,
                    aortic_thickness: None,
                    pulmonary_thickness: None,
                    kind: ContourType::Lumen,
                },
                extras: HashMap::new(),
                reference_point: None,
            },
        ];
        let mut geom = Geometry {
            frames,
            label: "test".to_string(),
        };

        let records = vec![
            Record {
                frame: 678,
                phase: "S".to_string(),
                measurement_1: Some(1.1),
                measurement_2: Some(2.3),
            },
            Record {
                frame: 717,
                phase: "S".to_string(),
                measurement_1: Some(1.2),
                measurement_2: None,
            },
            Record {
                frame: 621,
                phase: "S".to_string(),
                measurement_1: None,
                measurement_2: None,
            },
            Record {
                frame: 999,
                phase: "D".to_string(),
                measurement_1: Some(1.5),
                measurement_2: Some(2.1),
            },
        ];
        geom.reorder_frames(&records, false);

        assert_eq!(geom.frames[0].lumen.original_frame, 678);
        assert_eq!(geom.frames[1].lumen.original_frame, 717);
        assert_eq!(geom.frames[2].lumen.original_frame, 621);

        assert_eq!(geom.frames[0].id, 0);
        assert_eq!(geom.frames[1].id, 1);
        assert_eq!(geom.frames[2].id, 2);

        assert_eq!(geom.frames[0].lumen.id, 0);
        assert_eq!(geom.frames[1].lumen.id, 1);
        assert_eq!(geom.frames[2].lumen.id, 2);

        assert_eq!(geom.frames[0].centroid.2, 0.0);
        assert_eq!(geom.frames[1].centroid.2, 1.0);
        assert_eq!(geom.frames[2].centroid.2, 2.0);

        assert!(geom.frames[0].reference_point.is_some());
        assert_eq!(geom.frames[0].reference_point.as_ref().unwrap().z, 0.0);
        assert_eq!(
            geom.frames[0].reference_point.as_ref().unwrap().frame_index,
            678
        );
    }

    #[test]
    fn test_smoothing_effect() -> anyhow::Result<()> {
        use std::collections::HashMap;

        // Create 3 frames with square contours of different sizes
        let mut frames = Vec::new();

        // Frame 0: Square with edge length 2, centered at (0,0)
        let frame0_points = vec![
            ContourPoint {
                frame_index: 0,
                point_index: 0,
                x: -1.0,
                y: -1.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 0,
                point_index: 1,
                x: 1.0,
                y: -1.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 0,
                point_index: 2,
                x: 1.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 0,
                point_index: 3,
                x: -1.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let mut lumen0 = Contour {
            id: 0,
            original_frame: 0,
            points: frame0_points,
            centroid: None,
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        };
        lumen0.compute_centroid();

        let frame0 = Frame {
            id: 0,
            centroid: lumen0.centroid.unwrap(),
            lumen: lumen0,
            extras: HashMap::new(),
            reference_point: None,
        };

        // Frame 1: Square with edge length 1, centered at (0,0)
        let frame1_points = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: -0.5,
                y: -0.5,
                z: 1.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 0.5,
                y: -0.5,
                z: 1.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 2,
                x: 0.5,
                y: 0.5,
                z: 1.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 3,
                x: -0.5,
                y: 0.5,
                z: 1.0,
                aortic: false,
            },
        ];

        let mut lumen1 = Contour {
            id: 1,
            original_frame: 1,
            points: frame1_points,
            centroid: None,
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        };
        lumen1.compute_centroid();

        let frame1 = Frame {
            id: 1,
            centroid: lumen1.centroid.unwrap(),
            lumen: lumen1,
            extras: HashMap::new(),
            reference_point: None,
        };

        // Frame 2: Square with edge length 2, centered at (0,0) - same as frame 0
        let frame2_points = vec![
            ContourPoint {
                frame_index: 2,
                point_index: 0,
                x: -1.0,
                y: -1.0,
                z: 2.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 2,
                point_index: 1,
                x: 1.0,
                y: -1.0,
                z: 2.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 2,
                point_index: 2,
                x: 1.0,
                y: 1.0,
                z: 2.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 2,
                point_index: 3,
                x: -1.0,
                y: 1.0,
                z: 2.0,
                aortic: false,
            },
        ];

        let mut lumen2 = Contour {
            id: 2,
            original_frame: 2,
            points: frame2_points,
            centroid: None,
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        };
        lumen2.compute_centroid();

        let frame2 = Frame {
            id: 2,
            centroid: lumen2.centroid.unwrap(),
            lumen: lumen2,
            extras: HashMap::new(),
            reference_point: None,
        };

        frames.push(frame0);
        frames.push(frame1);
        frames.push(frame2);

        let geometry = Geometry {
            frames,
            label: "test_smoothing".to_string(),
        };

        // Apply smoothing
        let smoothed_geometry = geometry.clone().smooth_frames();

        // Verify results

        // Frame 0 (first frame) - should be average of frame0, frame0, frame1
        // For first point (-1,-1): (-1 + -1 + -0.5)/3 = -2.5/3 ≈ -0.833
        let frame0_smoothed = &smoothed_geometry.frames[0];
        assert_relative_eq!(frame0_smoothed.lumen.points[0].x, -0.8333, epsilon = 0.01);
        assert_relative_eq!(frame0_smoothed.lumen.points[0].y, -0.8333, epsilon = 0.01);

        // Frame 1 (middle frame) - should be average of frame0, frame1, frame2
        // For first point: (-1 + -0.5 + -1)/3 = -2.5/3 ≈ -0.833
        let frame1_smoothed = &smoothed_geometry.frames[1];
        assert_relative_eq!(frame1_smoothed.lumen.points[0].x, -0.8333, epsilon = 0.01);
        assert_relative_eq!(frame1_smoothed.lumen.points[0].y, -0.8333, epsilon = 0.01);

        // Frame 2 (last frame) - should be average of frame1, frame2, frame2
        // For first point: (-0.5 + -1 + -1)/3 = -2.5/3 ≈ -0.833
        let frame2_smoothed = &smoothed_geometry.frames[2];
        assert_relative_eq!(frame2_smoothed.lumen.points[0].x, -0.8333, epsilon = 0.01);
        assert_relative_eq!(frame2_smoothed.lumen.points[0].y, -0.8333, epsilon = 0.01);

        // Verify that z-coordinates remain unchanged
        for (i, frame) in smoothed_geometry.frames.iter().enumerate() {
            assert_relative_eq!(frame.centroid.2, i as f64, epsilon = 1e-6);
            for point in &frame.lumen.points {
                assert_relative_eq!(point.z, i as f64, epsilon = 1e-6);
            }
        }

        // Verify that all points in the same frame moved consistently
        let frame1_points: Vec<(f64, f64)> = smoothed_geometry.frames[1]
            .lumen
            .points
            .iter()
            .map(|p| (p.x, p.y))
            .collect();

        // All points in frame1 should have moved toward the average of frame0 and frame2
        for point in frame1_points {
            // Original frame1 points were at ±0.5, now should be around ±0.833
            assert!(point.0.abs() > 0.5 && point.0.abs() < 1.0);
            assert!(point.1.abs() > 0.5 && point.1.abs() < 1.0);
        }

        Ok(())
    }

    #[test]
    fn test_smoothing_with_eem_and_wall() -> anyhow::Result<()> {
        use std::collections::HashMap;

        // Create a simple geometry with EEM and Wall contours to test they get smoothed too
        let mut frames = Vec::new();

        // Create frame with lumen, EEM, and wall
        let lumen_points = vec![
            ContourPoint {
                frame_index: 0,
                point_index: 0,
                x: -1.0,
                y: -1.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 0,
                point_index: 1,
                x: 1.0,
                y: -1.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 0,
                point_index: 2,
                x: 1.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 0,
                point_index: 3,
                x: -1.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let eem_points = vec![
            ContourPoint {
                frame_index: 0,
                point_index: 0,
                x: -2.0,
                y: -2.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 0,
                point_index: 1,
                x: 2.0,
                y: -2.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 0,
                point_index: 2,
                x: 2.0,
                y: 2.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 0,
                point_index: 3,
                x: -2.0,
                y: 2.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let wall_points = vec![
            ContourPoint {
                frame_index: 0,
                point_index: 0,
                x: -1.5,
                y: -1.5,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 0,
                point_index: 1,
                x: 1.5,
                y: -1.5,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 0,
                point_index: 2,
                x: 1.5,
                y: 1.5,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 0,
                point_index: 3,
                x: -1.5,
                y: 1.5,
                z: 0.0,
                aortic: false,
            },
        ];

        let mut lumen = Contour {
            id: 0,
            original_frame: 0,
            points: lumen_points,
            centroid: None,
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        };
        lumen.compute_centroid();

        let mut eem = Contour {
            id: 0,
            original_frame: 0,
            points: eem_points,
            centroid: None,
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Eem,
        };
        eem.compute_centroid();

        let mut wall = Contour {
            id: 0,
            original_frame: 0,
            points: wall_points,
            centroid: None,
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Wall,
        };
        wall.compute_centroid();

        let mut extras = HashMap::new();
        extras.insert(ContourType::Eem, eem);
        extras.insert(ContourType::Wall, wall);

        let frame = Frame {
            id: 0,
            centroid: lumen.centroid.unwrap(),
            lumen,
            extras,
            reference_point: None,
        };

        frames.push(frame);

        // Create two more similar frames with slight variations
        for i in 1..3 {
            let lumen_points = vec![
                ContourPoint {
                    frame_index: i,
                    point_index: 0,
                    x: -1.0 + i as f64 * 0.1,
                    y: -1.0,
                    z: i as f64,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: i,
                    point_index: 1,
                    x: 1.0,
                    y: -1.0,
                    z: i as f64,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: i,
                    point_index: 2,
                    x: 1.0,
                    y: 1.0,
                    z: i as f64,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: i,
                    point_index: 3,
                    x: -1.0,
                    y: 1.0,
                    z: i as f64,
                    aortic: false,
                },
            ];

            let eem_points = vec![
                ContourPoint {
                    frame_index: i,
                    point_index: 0,
                    x: -2.0 + i as f64 * 0.1,
                    y: -2.0,
                    z: i as f64,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: i,
                    point_index: 1,
                    x: 2.0,
                    y: -2.0,
                    z: i as f64,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: i,
                    point_index: 2,
                    x: 2.0,
                    y: 2.0,
                    z: i as f64,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: i,
                    point_index: 3,
                    x: -2.0,
                    y: 2.0,
                    z: i as f64,
                    aortic: false,
                },
            ];

            let wall_points = vec![
                ContourPoint {
                    frame_index: i,
                    point_index: 0,
                    x: -1.5 + i as f64 * 0.1,
                    y: -1.5,
                    z: i as f64,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: i,
                    point_index: 1,
                    x: 1.5,
                    y: -1.5,
                    z: i as f64,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: i,
                    point_index: 2,
                    x: 1.5,
                    y: 1.5,
                    z: i as f64,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: i,
                    point_index: 3,
                    x: -1.5,
                    y: 1.5,
                    z: i as f64,
                    aortic: false,
                },
            ];

            let mut lumen = Contour {
                id: i,
                original_frame: i,
                points: lumen_points,
                centroid: None,
                aortic_thickness: None,
                pulmonary_thickness: None,
                kind: ContourType::Lumen,
            };
            lumen.compute_centroid();

            let mut eem = Contour {
                id: i,
                original_frame: i,
                points: eem_points,
                centroid: None,
                aortic_thickness: None,
                pulmonary_thickness: None,
                kind: ContourType::Eem,
            };
            eem.compute_centroid();

            let mut wall = Contour {
                id: i,
                original_frame: i,
                points: wall_points,
                centroid: None,
                aortic_thickness: None,
                pulmonary_thickness: None,
                kind: ContourType::Wall,
            };
            wall.compute_centroid();

            let mut extras = HashMap::new();
            extras.insert(ContourType::Eem, eem);
            extras.insert(ContourType::Wall, wall);

            let frame = Frame {
                id: i,
                centroid: lumen.centroid.unwrap(),
                lumen,
                extras,
                reference_point: None,
            };

            frames.push(frame);
        }

        let geometry = Geometry {
            frames,
            label: "test_smoothing_with_extras".to_string(),
        };

        let smoothed_geometry = geometry.smooth_frames();

        // Verify that EEM and Wall contours were also smoothed
        for frame in &smoothed_geometry.frames {
            // Check EEM exists and was smoothed
            if let Some(eem) = frame.extras.get(&ContourType::Eem) {
                // EEM points should have been averaged with adjacent frames
                assert!(eem.points[0].x > -2.1 && eem.points[0].x < 2.1);
            }

            // Check Wall exists and was smoothed
            if let Some(wall) = frame.extras.get(&ContourType::Wall) {
                // Wall points should have been averaged with adjacent frames
                assert!(wall.points[0].x > -1.6 && wall.points[0].x < 1.6);
            }
        }

        Ok(())
    }
}
