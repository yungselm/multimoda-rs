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

        for (frame_idx, points) in groups {
            let (aortic, pulmonary) = if let Some(ref meas) = measurements {
                meas.get(&frame_idx)
                    .map(|&(m1, m2)| (Some(m1), Some(m2)))
                    .unwrap_or((None, None))
            } else {
                (None, None)
            };

            contours.push(Contour {
                id: frame_idx,
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

    pub fn compute_centroid(&self) -> Self {
        let (sum_x, sum_y, sum_z) = self.points.iter().fold((0.0, 0.0, 0.0), |(sx, sy, sz), p| {
            (sx + p.x, sy + p.y, sz + p.z)
        });
        let n = self.points.len() as f64;
        Self {
            id: self.id,
            original_frame: self.original_frame,
            points: self.points.clone(),
            centroid: Some((sum_x / n, sum_y / n, sum_z / n)),
            aortic_thickness: self.aortic_thickness,
            pulmonary_thickness: self.pulmonary_thickness,
            kind: self.kind,
        }
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

        // 1) Compute centroid (x0,y0). If missing, compute it first.
        let (cx, cy, _) = match self.centroid {
            Some(c) => c,
            None => {
                let computed = self.compute_centroid();
                computed.centroid.unwrap()
            }
        };

        // 2) Precompute angles
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
        let mut area = 0.0;
        for i in 0..n {
            let p1 = &self.points[i];
            let p2 = &self.points[(i + 1) % n];
            area += p1.x * p2.y - p2.x * p1.y;
        }
        0.5 * area.abs()
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
        // Get centroid or compute if not present
        let (cx, cy, _) = match self.centroid {
            Some(c) => c,
            None => {
                let computed = self.compute_centroid();
                computed.centroid.unwrap()
            }
        };

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
        original_frame: Option<u32>,
        points: Option<Vec<ContourPoint>>,
        centroid: Option<(f64, f64, f64)>,
        z_value: Option<f64>,
    ) {
        if let Some(new_id) = id {
            self.lumen.id = new_id;
            for contour in self.extras.values_mut() {
                contour.id = new_id;
            }
        }
        if let Some(new_frame) = original_frame {
            self.lumen.original_frame = new_frame;
            for contour in self.extras.values_mut() {
                contour.original_frame = new_frame;
            }
            if let Some(ref mut rp) = self.reference_point {
                rp.frame_index = new_frame;
            }
        }
        if let Some(new_points) = points.clone() {
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

        for contour in self.extras.values_mut() {
            for p in contour.points.iter_mut() {
                p.x += dx;
                p.y += dy;
                p.z += dz;
            }
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

    pub fn rotate_frame_around_point(&mut self, angle: f64, center: (f64, f64)) {
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

        // Update centroid
        let (cx, cy) = center;
        let current_centroid = self.centroid;
        let x = current_centroid.0 - cx;
        let y = current_centroid.1 - cy;
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        self.centroid.0 = x * cos_a - y * sin_a + cx;
        self.centroid.1 = x * sin_a + y * cos_a + cy;
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

    pub fn create_wall_contour() -> () {
        todo!()
    }
}

impl Geometry {
    pub fn new() -> anyhow::Result<Self> {
        todo!("from files and from arrays")
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

        for (_orig, frame) in frame_map.into_iter() {
            new_frames.push(frame);
        }

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
                rp.frame_index = new_id;
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

                Contour {
                    id: current.id,
                    original_frame: current.original_frame,
                    points: new_points,
                    centroid: None,
                    aortic_thickness: current.aortic_thickness,
                    pulmonary_thickness: current.pulmonary_thickness,
                    kind: current.kind,
                }
                .compute_centroid()
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
        for frame in self.frames.iter_mut() {
            frame.rotate_frame(angle_rad);
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
}

#[cfg(test)]
mod geometry_tests {
    use super::*;
    use std::f64::consts::PI;

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

        let contour = Contour {
            id: 1,
            original_frame: 1,
            points,
            centroid: None,
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        };

        let contour_with_centroid = contour.compute_centroid();
        assert_eq!(contour_with_centroid.centroid, Some((1.0, 1.0, 0.0)));
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
    fn test_frame_rotate() {
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
                        x: 2.0,
                        y: 1.0,
                        z: 0.0,
                        aortic: false,
                    },
                    ContourPoint {
                        frame_index: 1,
                        point_index: 1,
                        x: 1.0,
                        y: 2.0,
                        z: 0.0,
                        aortic: false,
                    },
                    ContourPoint {
                        frame_index: 1,
                        point_index: 2,
                        x: 0.0,
                        y: 1.0,
                        z: 0.0,
                        aortic: false,
                    },
                    ContourPoint {
                        frame_index: 1,
                        point_index: 3,
                        x: 1.0,
                        y: 0.0,
                        z: 0.0,
                        aortic: false,
                    },
                ],
                centroid: Some((1.0, 1.0, 0.0)),
                aortic_thickness: None,
                pulmonary_thickness: None,
                kind: ContourType::Lumen,
            },
            extras: HashMap::new(),
            reference_point: None,
        };

        frame.rotate_frame(PI / 2.0);

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
                x: 0.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 2,
                x: 1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 3,
                x: 2.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
        ];

        for (i, point) in frame.lumen.points.iter().enumerate() {
            assert!((point.x - expected_points[i].x).abs() < 1e-6);
            assert!((point.y - expected_points[i].y).abs() < 1e-6);
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
        frame.rotate_frame_around_point(PI, (1.0, 1.0));

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
    fn test_frame_translate() {
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
            extras: HashMap::new(),
            reference_point: None,
        };

        frame.translate_frame((1.0, 2.0, 3.0));

        assert_eq!(frame.centroid, (2.0, 3.0, 3.0));
        for point in &frame.lumen.points {
            assert_eq!(point.z, 3.0);
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
    fn test_elliptic_ratio_and_area() {
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
                    x: 4.0,
                    y: 0.0,
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
                    x: 0.0,
                    y: 2.0,
                    z: 0.0,
                    aortic: false,
                },
            ],
            centroid: Some((2.0, 1.0, 0.0)),
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        };

        let ratio = contour.elliptic_ratio();
        let area = contour.area();

        // For a 4x2 rectangle, elliptic ratio should be ~2.0
        assert!((ratio - 2.0).abs() < 0.1);
        // Area should be 8.0
        assert!((area - 8.0).abs() < 1e-6);
    }
}
