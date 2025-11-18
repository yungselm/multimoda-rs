use super::input::{ContourPoint, Record};
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContourType {
    Lumen,
    Eem,
    Calcification,
    Sidebranch,
    Catheter,
    Wall,
}

impl fmt::Display for ContourType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ContourType::Lumen => "Lumen",
                ContourType::Eem => "Eem",
                ContourType::Calcification => "Calcification",
                ContourType::Sidebranch => "Sidebranch",
                ContourType::Catheter => "Catheter",
                ContourType::Wall => "Wall",
            }
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Contour {
    pub id: u32,
    pub original_frame: u32,
    pub points: Vec<ContourPoint>,
    pub centroid: Option<(f64, f64, f64)>,
    pub aortic_thickness: Option<f64>,
    pub pulmonary_thickness: Option<f64>,
    pub kind: ContourType,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub id: u32,
    pub centroid: (f64, f64, f64),
    // groundtruth, must exist!
    pub lumen: Contour,
    pub extras: HashMap<ContourType, Contour>,
    pub reference_point: Option<ContourPoint>,
}

#[derive(Debug, Clone)]
pub struct Geometry {
    pub frames: Vec<Frame>,
    pub label: String,
}

impl Contour {
    pub fn new(
        id: u32,
        points: Vec<ContourPoint>,
        records: Option<Vec<Record>>,
        kind: ContourType,
    ) -> Self {
        let (aortic, pulmonary) = if let Some(ref records) = records {
            records
                .iter()
                .find(|r| r.frame == id)
                .map(|r| (Some(r.measurement_1), Some(r.measurement_2)))
                .unwrap_or((None, None))
        } else {
            (None, None)
        };

        Contour {
            id,
            original_frame: id,
            points,
            centroid: None,
            aortic_thickness: aortic.flatten(),
            pulmonary_thickness: pulmonary.flatten(),
            kind,
        }
    }

    pub fn build_contour(
        points: Vec<ContourPoint>,
        records: Option<Vec<Record>>,
        kind: ContourType,
    ) -> anyhow::Result<Vec<Contour>> {
        let mut groups: HashMap<u32, Vec<ContourPoint>> = HashMap::new();
        for p in points {
            groups.entry(p.frame_index).or_default().push(p);
        }

        let mut contours = Vec::new();

        // Only process measurements for Lumen contour type
        let measurements = if kind == ContourType::Lumen {
            let mut meas = HashMap::new();
            if let Some(records) = records {
                for record in records {
                    meas.insert(record.frame, (record.measurement_1, record.measurement_2));
                }
            }
            Some(meas)
        } else {
            None
        };

        let mut sorted_groups: Vec<(u32, Vec<ContourPoint>)> = groups.into_iter().collect();
        sorted_groups.sort_by_key(|(frame_idx, _)| *frame_idx);

        for (i, (frame_idx, points)) in sorted_groups.into_iter().enumerate() {
            let (aortic, pulmonary) = if let Some(ref meas) = measurements {
                meas.get(&frame_idx)
                    .map(|&(m1, m2)| (Some(m1), Some(m2)))
                    .unwrap_or((None, None))
            } else {
                (None, None)
            };

            contours.push(Contour {
                id: i as u32,
                original_frame: frame_idx,
                points,
                centroid: None,
                aortic_thickness: aortic.flatten(),
                pulmonary_thickness: pulmonary.flatten(),
                kind,
            });
        }

        Ok(contours)
    }

    pub fn build_contour_with_mapping(
        points: Vec<ContourPoint>,
        records: Option<Vec<Record>>,
        kind: ContourType,
        frame_mapping: &HashMap<u32, u32>,
    ) -> anyhow::Result<Vec<Contour>> {
        let mut groups: HashMap<u32, Vec<ContourPoint>> = HashMap::new();
        for p in points {
            groups.entry(p.frame_index).or_default().push(p);
        }

        let mut contours = Vec::new();

        let measurements = if kind == ContourType::Lumen {
            let mut meas = HashMap::new();
            if let Some(records) = records {
                for record in records {
                    meas.insert(record.frame, (record.measurement_1, record.measurement_2));
                }
            }
            Some(meas)
        } else {
            None
        };

        let mut sorted_groups: Vec<(u32, Vec<ContourPoint>)> = groups.into_iter().collect();
        sorted_groups.sort_by_key(|(frame_idx, _)| *frame_idx);

        for (original_frame_idx, points) in sorted_groups {
            let sequential_id = *frame_mapping.get(&original_frame_idx).ok_or_else(|| {
                anyhow::anyhow!("No mapping found for original frame {}", original_frame_idx)
            })?;

            let (aortic, pulmonary) = if let Some(ref meas) = measurements {
                meas.get(&original_frame_idx)
                    .map(|&(m1, m2)| (Some(m1), Some(m2)))
                    .unwrap_or((None, None))
            } else {
                (None, None)
            };

            contours.push(Contour {
                id: sequential_id,                  // Use the shared sequential ID
                original_frame: original_frame_idx, // Keep original frame for reference
                points,
                centroid: None,
                aortic_thickness: aortic.flatten(),
                pulmonary_thickness: pulmonary.flatten(),
                kind,
            });
        }

        Ok(contours)
    }

    pub fn compute_centroid(&mut self) {
        if self.points.is_empty() {
            self.centroid = None;
            return;
        }

        let (sum_x, sum_y, sum_z) = self.points.iter().fold((0.0, 0.0, 0.0), |(sx, sy, sz), p| {
            (sx + p.x, sy + p.y, sz + p.z)
        });
        let n = self.points.len() as f64;
        self.centroid = Some((sum_x / n, sum_y / n, sum_z / n));
    }

    /// Finds the pair of farthest points in the current contour.
    pub fn find_farthest_points(&self) -> ((&ContourPoint, &ContourPoint), f64) {
        let mut max_dist = 0.0;
        let mut farthest_pair = (&self.points[0], &self.points[0]);

        for i in 0..self.points.len() {
            for j in i + 1..self.points.len() {
                let dx = self.points[i].x - self.points[j].x;
                let dy = self.points[i].y - self.points[j].y;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist > max_dist {
                    max_dist = dist;
                    farthest_pair = (&self.points[i], &self.points[j]);
                }
            }
        }

        (farthest_pair, max_dist)
    }

    /// Find the pair of points whose coord is the smallest diameter,
    /// by matching each point to the one whose angle (about the centroid)
    /// differs by as close to π radians as possible.
    pub fn find_closest_opposite(&self) -> ((&ContourPoint, &ContourPoint), f64) {
        let n = self.points.len();
        assert!(n > 2, "Need at least 3 points");

        let (cx, cy, _) = if let Some(c) = self.centroid {
            c
        } else {
            let (sum_x, sum_y, sum_z) =
                self.points.iter().fold((0.0, 0.0, 0.0), |(sx, sy, sz), p| {
                    (sx + p.x, sy + p.y, sz + p.z)
                });
            let n_f = n as f64;
            (sum_x / n_f, sum_y / n_f, sum_z / n_f)
        };

        let thetas: Vec<f64> = self
            .points
            .iter()
            .map(|p| {
                let mut t = (p.y - cy).atan2(p.x - cx);
                if t < 0.0 {
                    t += 2.0 * PI;
                }
                t
            })
            .collect();

        let mut min_dist = f64::MAX;
        let mut best_pair = (&self.points[0], &self.points[1]);

        // 3) Brute‐force: for each i, find j that best approximates θi+π
        for i in 0..n {
            let mut best_angle_diff = f64::MAX;
            let mut best_j = i;

            for j in 0..n {
                if j == i {
                    continue;
                }
                // compute angular separation in [0,2π)
                let mut delta = (thetas[j] - thetas[i]).abs();
                if delta > PI {
                    delta = 2.0 * PI - delta;
                }
                let diff = (delta - PI).abs();
                if diff < best_angle_diff {
                    best_angle_diff = diff;
                    best_j = j;
                }
            }

            // 4) Compute chord length between i and best_j
            let pi = &self.points[i];
            let pj = &self.points[best_j];
            let dx = pi.x - pj.x;
            let dy = pi.y - pj.y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < min_dist {
                min_dist = dist;
                best_pair = (pi, pj);
            }
        }

        (best_pair, min_dist)
    }

    pub fn elliptic_ratio(&self) -> f64 {
        let major_length = self.find_farthest_points().1;
        let minor_length = self.find_closest_opposite().1;
        if major_length < minor_length {
            minor_length / major_length
        } else {
            major_length / minor_length
        }
    }

    pub fn area(&self) -> f64 {
        let n = self.points.len();
        if n < 3 {
            return 0.0;
        }

        let mut sum = 0.0_f64;
        for i in 0..n {
            let p1 = &self.points[i];
            let p2 = &self.points[(i + 1) % n];
            sum += p1.x * p2.y - p2.x * p1.y;
        }
        0.5 * sum.abs()
    }

    /// Reorders `self.points` so that:
    /// 1) They’re sorted counterclockwise around the centroid,
    /// 2) The point with the highest Y-value is at index 0,
    /// 3) Each `point.point_index` matches its position in the Vec.
    pub fn sort_contour_points(&mut self) {
        let n = self.points.len() as f64;
        if n == 0.0 {
            return;
        }

        // 1) Compute centroid (cx, cy)
        let (sum_x, sum_y) = self
            .points
            .iter()
            .fold((0.0, 0.0), |(sx, sy), p| (sx + p.x, sy + p.y));
        let cx = sum_x / n;
        let cy = sum_y / n;

        // 2) Sort by *descending* angle around centroid (clockwise sweep)
        self.points.sort_by(|a, b| {
            let angle_a = (a.y - cy).atan2(a.x - cx);
            let angle_b = (b.y - cy).atan2(b.x - cx);
            // flip the comparison order ?
            angle_a.partial_cmp(&angle_b).unwrap()
        });

        // 3) Find the index of the highest y-coord point and rotate it to front
        if let Some(start_idx) = self
            .points
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.y.partial_cmp(&b.y).unwrap())
            .map(|(i, _)| i)
        {
            self.points.rotate_left(start_idx);
        }

        // 4) Re-index in array order
        for (i, pt) in self.points.iter_mut().enumerate() {
            pt.point_index = i as u32;
        }
    }

    /// Rotates a contour around its centroid by the specified angle (in radians)
    pub fn rotate_contour(&mut self, angle: f64) {
        if angle == 0.0 {
            return;
        }
        // Get centroid or compute if not present
        self.compute_centroid();
        let (cx, cy, _) = self.centroid.unwrap();

        // Rotate each point around the centroid
        for point in self.points.iter_mut() {
            let x = point.x - cx;
            let y = point.y - cy;
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            point.x = x * cos_a - y * sin_a + cx;
            point.y = x * sin_a + y * cos_a + cy;
        }
    }

    pub fn translate_contour(&mut self, translation: (f64, f64, f64)) {
        let (dx, dy, dz) = translation;
        for p in self.points.iter_mut() {
            p.x += dx;
            p.y += dy;
            p.z += dz;
        }
    }
}

impl Frame {
    pub fn new() {
        todo!()
    }

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
        let radius = radius;
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

        let mut remaining_frames: Vec<Frame> =
            frame_map.into_iter().map(|(_, frame)| frame).collect();
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
            frame.rotate_frame(angle_rad);
            frame.sort_frame_points();
        }
    }

    pub fn translate_geometry(&mut self, translation: (f64, f64, f64)) {
        for frame in self.frames.iter_mut() {
            frame.translate_frame(translation);
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

            current_frame.translate_frame(translation);

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
    use std::f64::consts::PI;

    // test for contours
    #[test]
    fn test_build_contour_groups_by_frame() -> anyhow::Result<()> {
        use super::*;

        // create points: two points for frame 1, one point for frame 2
        let pts = vec![
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
                x: 1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 2,
                point_index: 0,
                x: 2.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let contours = Contour::build_contour(pts, None, ContourType::Lumen)?;
        // sort by id for deterministic assertions
        let mut contours = contours;
        contours.sort_by_key(|c| c.id);

        assert_eq!(
            contours.len(),
            2,
            "should produce two contours (frames 1 and 2)"
        );
        assert_eq!(contours[0].id, 0);
        assert_eq!(contours[0].original_frame, 1);
        assert_eq!(
            contours[0].points.len(),
            2,
            "frame 1 should have two points"
        );
        assert_eq!(contours[1].id, 1);
        assert_eq!(contours[1].original_frame, 2);
        assert_eq!(contours[1].points.len(), 1, "frame 2 should have one point");

        Ok(())
    }

    #[test]
    fn test_build_contour_attaches_measurements_for_lumen() -> anyhow::Result<()> {
        use super::*;

        // points for frame 1
        let pts = vec![ContourPoint {
            frame_index: 1,
            point_index: 0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            aortic: false,
        }];

        // create a Record that contains measurement_1 and measurement_2 for frame 1
        let records = vec![Record {
            frame: 1,
            phase: "systolic".to_string(),
            measurement_1: Some(1.23),
            measurement_2: Some(4.56),
        }];

        let mut contours = Contour::build_contour(pts, Some(records), ContourType::Lumen)?;
        assert_eq!(contours.len(), 1);
        let c = contours.pop().unwrap();

        assert_eq!(c.id, 0);
        assert_eq!(c.aortic_thickness, Some(1.23));
        assert_eq!(c.pulmonary_thickness, Some(4.56));

        Ok(())
    }

    #[test]
    fn test_build_contour_ignores_measurements_for_non_lumen() -> anyhow::Result<()> {
        use super::*;

        // points for frame 1
        let pts = vec![ContourPoint {
            frame_index: 1,
            point_index: 0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            aortic: false,
        }];

        // create a Record that contains measurement_1 and measurement_2 for frame 1
        let records = vec![Record {
            frame: 1,
            phase: "systolic".to_string(),
            measurement_1: Some(7.0),
            measurement_2: Some(8.0),
        }];

        // Use a non-lumen kind — measurements should not be applied
        let mut contours = Contour::build_contour(pts, Some(records), ContourType::Eem)?;
        assert_eq!(contours.len(), 1);
        let c = contours.pop().unwrap();

        assert_eq!(c.id, 0);
        assert_eq!(
            c.aortic_thickness, None,
            "non-lumen contours shouldn't pick up measurements"
        );
        assert_eq!(c.pulmonary_thickness, None);

        Ok(())
    }

    #[test]
    fn test_compute_centroid() {
        let points = vec![
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
        ];

        let mut contour = Contour {
            id: 1,
            original_frame: 1,
            points,
            centroid: None,
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        };

        contour.compute_centroid();
        assert_eq!(contour.centroid, Some((1.0, 1.0, 0.0)));
    }

    #[test]
    fn test_find_farthest_points() {
        let contour = Contour {
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
        };

        let (pair, distance) = contour.find_farthest_points();
        assert!((distance - (8.0_f64).sqrt()).abs() < 1e-6);
        assert!(
            (pair.0.x == 0.0 && pair.0.y == 0.0 && pair.1.x == 2.0 && pair.1.y == 2.0)
                || (pair.0.x == 2.0 && pair.0.y == 2.0 && pair.1.x == 0.0 && pair.1.y == 0.0)
        );
    }

    #[test]
    fn test_find_closest_opposite() {
        let points = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 0.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 2,
                x: 0.0,
                y: -0.5,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 3,
                x: -1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let contour = Contour {
            id: 1,
            original_frame: 1,
            points,
            centroid: Some((0.0, 0.125, 0.0)), // Pre-computed centroid
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        };

        let (pair, distance) = contour.find_closest_opposite();
        assert!((distance - 1.5).abs() < 1e-6);
        let p0 = &contour.points[0];
        let p2 = &contour.points[2];
        assert!((pair.0 == p0 && pair.1 == p2) || (pair.0 == p2 && pair.1 == p0));
    }

    #[test]
    fn rotate_contour_back_and_forth() {
        let geometry = dummy_geometry();
        let mut geometry_rotate = geometry.clone();
        let rotation_deg: f64 = 15.0;

        geometry_rotate.rotate_geometry(rotation_deg.to_radians());
        geometry_rotate.rotate_geometry(-1.0 * rotation_deg.to_radians());

        assert_eq!(
            geometry.frames[0].lumen.points[0],
            geometry_rotate.frames[0].lumen.points[0]
        );

        geometry_rotate.frames[0].rotate_frame(rotation_deg.to_radians());
        geometry_rotate.frames[0].rotate_frame(-1.0 * rotation_deg.to_radians());

        assert_eq!(
            geometry.frames[0].lumen.points[0],
            geometry_rotate.frames[0].lumen.points[0]
        );

        geometry_rotate.frames[0]
            .lumen
            .rotate_contour(rotation_deg.to_radians());
        geometry_rotate.frames[0]
            .lumen
            .rotate_contour(-1.0 * rotation_deg.to_radians());

        assert_eq!(
            geometry.frames[0].lumen.points[0],
            geometry_rotate.frames[0].lumen.points[0]
        );

        let center = geometry_rotate.frames[0].centroid;
        geometry_rotate.frames[0].lumen.points[0]
            .rotate_point(rotation_deg.to_radians(), (center.0, center.1));
        geometry_rotate.frames[0].lumen.points[0]
            .rotate_point(-1.0 * rotation_deg.to_radians(), (center.0, center.1));

        assert_eq!(
            geometry.frames[0].lumen.points[0],
            geometry_rotate.frames[0].lumen.points[0]
        );
    }

    #[test]
    fn test_frame_rotate_with_eem_90deg() {
        use super::*;
        use std::collections::HashMap;
        use std::f64::consts::PI;

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
        let expected_lumen = vec![(0.0, 0.0), (-2.0, 2.0), (0.0, 4.0), (2.0, 2.0)];

        // Expected eem points after rotation around (1,1):
        let expected_eem = vec![(0.0, -1.0), (-3.0, 2.0), (0.0, 5.0), (3.0, 0.0)];

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

        let expected_points = vec![
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
    fn test_sort_contour_points() {
        let mut contour = Contour {
            id: 1,
            original_frame: 1,
            points: vec![
                ContourPoint {
                    frame_index: 1,
                    point_index: 0,
                    x: -2.0,
                    y: 0.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 1,
                    point_index: 1,
                    x: 0.0,
                    y: 2.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 1,
                    point_index: 2,
                    x: 2.0,
                    y: 0.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 1,
                    point_index: 3,
                    x: 0.0,
                    y: -2.0,
                    z: 0.0,
                    aortic: false,
                },
            ],
            centroid: Some((0.0, 0.0, 0.0)),
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        };

        contour.sort_contour_points();

        for (i, point) in contour.points.iter().enumerate() {
            match i {
                0 => {
                    assert!((point.x - 0.0).abs() < 1e-6);
                    assert!((point.y - 2.0).abs() < 1e-6);
                }
                1 => {
                    assert!((point.x - (-2.0)).abs() < 1e-6);
                    assert!((point.y - 0.0).abs() < 1e-6);
                }
                2 => {
                    assert!((point.x - 0.0).abs() < 1e-6);
                    assert!((point.y - (-2.0)).abs() < 1e-6);
                }
                3 => {
                    assert!((point.x - 2.0).abs() < 1e-6);
                    assert!((point.y - 0.0).abs() < 1e-6);
                }
                _ => panic!("Unexpected index"),
            }
        }
    }

    #[test]
    fn test_frame_translate_with_eem_and_reference() {
        use std::collections::HashMap;

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

        let expected_lumen = vec![
            (1.0, 2.0, 3.0),
            (3.0, 2.0, 3.0),
            (3.0, 4.0, 3.0),
            (1.0, 4.0, 3.0),
        ];
        for (i, p) in frame.lumen.points.iter().enumerate() {
            assert_eq!(
                (p.x, p.y, p.z),
                expected_lumen[i],
                "lumen point {} mismatch",
                i
            );
        }

        let rotated_eem = frame
            .extras
            .get(&ContourType::Eem)
            .expect("eem contour present");
        let expected_eem = vec![
            (0.0, 4.0, 3.0),
            (3.0, 7.0, 3.0),
            (6.0, 4.0, 3.0),
            (1.0, 1.0, 3.0),
        ];
        for (i, p) in rotated_eem.points.iter().enumerate() {
            assert_eq!((p.x, p.y, p.z), expected_eem[i], "eem point {} mismatch", i);
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
    fn test_area_triangle() {
        // Right triangle with base 3 and height 4 -> area = 6
        let points = vec![
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
                x: 3.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 2,
                x: 0.0,
                y: 4.0,
                z: 0.0,
                aortic: false,
            },
        ];
        let c = Contour {
            id: 1,
            original_frame: 1,
            points,
            centroid: None,
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        };

        let area = c.area();
        assert!(
            (area - 6.0).abs() < 1e-6,
            "triangle area should be 6.0, got {}",
            area
        );
    }

    #[test]
    fn test_area_square_ccw_and_cw() {
        let pts_ccw = vec![
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
                x: 1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 2,
                x: 1.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 3,
                x: 0.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
        ];
        let c_ccw = Contour {
            id: 1,
            original_frame: 1,
            points: pts_ccw,
            centroid: None,
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        };
        assert!(
            (c_ccw.area() - 1.0).abs() < 1e-6,
            "ccw square area should be 1.0"
        );

        let pts_cw = c_ccw.points.iter().rev().cloned().collect::<Vec<_>>();
        let c_cw = Contour {
            points: pts_cw,
            ..c_ccw.clone()
        };
        assert!(
            (c_cw.area() - 1.0).abs() < 1e-6,
            "cw square area should be 1.0"
        );
    }

    #[test]
    fn test_area_less_than_three_points_is_zero() {
        let empty = Contour {
            id: 0,
            original_frame: 0,
            points: vec![],
            centroid: None,
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        };
        assert_eq!(empty.area(), 0.0);

        let two = Contour {
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
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                    aortic: false,
                },
            ],
            centroid: None,
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        };
        assert_eq!(two.area(), 0.0);
    }

    #[test]
    fn test_elliptic_ratio_and_area() {
        let contour = Contour {
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
                    y: 2.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 1,
                    point_index: 2,
                    x: 1.0,
                    y: 4.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 1,
                    point_index: 3,
                    x: 2.0,
                    y: 2.0,
                    z: 0.0,
                    aortic: false,
                },
            ],
            centroid: Some((1.0, 2.0, 0.0)),
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        };

        let ratio = contour.elliptic_ratio();
        let area = contour.area();

        // For a 4x2 rectangle, elliptic ratio should be ~2.0
        assert!((ratio - 2.0).abs() < 1e-6);
        // Area should be 8.0
        assert!((area - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_frame_set_value_updates_all_targets() {
        use std::collections::HashMap;

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
            frames: frames,
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
        use crate::intravascular::io::input::Record;
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
            frames: frames,
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
        use crate::intravascular::io::geometry::{Contour, ContourType, Frame, Geometry};
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

        println!("Before smoothing:");
        for (i, frame) in geometry.frames.iter().enumerate() {
            println!("Frame {}:", i);
            for point in &frame.lumen.points {
                println!("  Point: ({:.2}, {:.2}, {:.2})", point.x, point.y, point.z);
            }
        }

        // Apply smoothing
        let smoothed_geometry = geometry.clone().smooth_frames();

        println!("\nAfter smoothing:");
        for (i, frame) in smoothed_geometry.frames.iter().enumerate() {
            println!("Frame {}:", i);
            for point in &frame.lumen.points {
                println!("  Point: ({:.2}, {:.2}, {:.2})", point.x, point.y, point.z);
            }
        }

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
        // (all should have the same amount of smoothing applied)
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
        use crate::intravascular::io::geometry::{Contour, ContourType, Frame, Geometry};
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
                id: i as u32,
                original_frame: i as u32,
                points: lumen_points,
                centroid: None,
                aortic_thickness: None,
                pulmonary_thickness: None,
                kind: ContourType::Lumen,
            };
            lumen.compute_centroid();

            let mut eem = Contour {
                id: i as u32,
                original_frame: i as u32,
                points: eem_points,
                centroid: None,
                aortic_thickness: None,
                pulmonary_thickness: None,
                kind: ContourType::Eem,
            };
            eem.compute_centroid();

            let mut wall = Contour {
                id: i as u32,
                original_frame: i as u32,
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
                id: i as u32,
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
