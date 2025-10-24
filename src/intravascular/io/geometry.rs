use super::input::{ContourPoint, Record};
use super::wall;
use std::collections::HashMap;
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContourType {
    Lumen,
    Eem,
    Calcification,
    Sidebranch,
    Catheter,
    Wall,
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
    pub extras:  HashMap<ContourType, Contour>,
    pub reference_point: Option<ContourPoint>,
}

#[derive(Debug, Clone)]
pub struct Geometry {
    pub frames: Vec<Frame>,
    pub label: String,
    // pointer to proximal end
}

impl Contour {
    // pub fn new(
    //     points: Vec<ContourPoint>,  
    //     records: Option<Vec<Record>>, 
    //     label: ContourType) -> Self {
    //     Self {
    //         id,
    //         original_frame,

    //     }
    // }
    pub fn create_contour() {
        
    }

    fn build_contour_records(&mut self) -> () {
        todo!()
    }

    fn build_contour(&mut self) -> () {
        todo!()
    }

    /// Compute centroid by calculating mean of x-, y- and z-coordinates
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

        // 1) Compute centroid (x0,y0) if missing
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
}

impl Frame {
    pub fn new() {
        todo!()
    }

    pub fn rotate_frame(&mut self, angle: f64) {
        let center = (self.centroid.0, self.centroid.1);

        self.lumen.points = self.lumen
            .points
            .iter()
            .map(|p| p.rotate_point(angle, center))
            .collect();

        for contour in self.extras.values_mut() {
            contour.points = contour.points
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

    fn create_catheter_contour(
        contour: &Contour,
        image_center: (f64, f64),
        radius: f64,
        n_points: u32,
    ) -> Contour {
        let centroid = match contour.centroid {
            Some(c) => c,
            None => {
                let computed = contour.compute_centroid();
                computed.centroid.unwrap()
            }
        };

        let mut catheter_points = Vec::new();

        let center_x = image_center.0;
        let center_y = image_center.1;

        // Generate n_points around a circle
        for i in 0..n_points {
            let angle = 2.0 * PI * (i as f64) / (n_points as f64);
            let x = center_x + radius * angle.cos();
            let y = center_y + radius * angle.sin();
            catheter_points.push(ContourPoint {
                frame_index: contour.id,
                point_index: i,
                x,
                y,
                z: centroid.2, // Use frame's z coordinate
                aortic: false,
            });
        }

        Contour {
            id: contour.id,
            original_frame: contour.id,
            points: catheter_points,
            centroid: Some((image_center.0, image_center.1, centroid.2)),
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Catheter,
        }
    }

    pub fn create_catheter_frame(
        &mut self, 
        image_center: (f64, f64), 
        radius: f64, 
        n_points: u32) -> Self {
        
        let catheter = Frame::create_catheter_contour(&self.lumen, image_center, radius, n_points);
        
        let mut new_extras = self.extras.clone();
        new_extras.insert(ContourType::Catheter, catheter);

        Self {
            id: self.id,
            centroid: self.centroid,
            lumen: self.lumen.clone(),
            extras: new_extras,
            reference_point: self.reference_point,
        }
    }
}

impl Geometry {
    pub fn new() -> anyhow::Result<Self> {
        todo!("from files and from arrays")
    }

    pub fn reorder_frames() {
        todo!()
    }


    /// Smooths the x and y coordinates using a 3‐point moving average for the following `ContourTypes`:
    ///     - Lumen
    ///     - Wall
    ///     - EEM
    ///
    /// For each point i in contour j, the new x and y values are computed as:
    ///     new_x = (prev_contour[i].x + current_contour[i].x + next_contour[i].x) / 3.0
    ///     new_y = (prev_contour[i].y + current_contour[i].y + next_contour[i].y) / 3.0
    /// while the z coordinate remains unchanged (taken from the current contour).
    ///
    /// For the first and last contours, the current contour is used twice to simulate a mirror effect.
    pub fn smooth_frames(mut self) -> Geometry {
        let n = self.frames.len();
        if n == 0 {
            return self;
        }

        // Helper function to smooth a single contour type across frames
        fn smooth_contour_type(frames: &mut [Frame], n: usize, contour_type: ContourType) {
            for j in 0..n {
                let mut new_points = Vec::new();
                
                // Get current frame's contour points
                let current_points = match contour_type {
                    ContourType::Lumen => &frames[j].lumen.points,
                    _ => match &frames[j].extras.get(&contour_type) {
                        Some(c) => &c.points,
                        None => continue,
                    }
                };

                for i in 0..current_points.len() {
                    // Get previous and next frame points
                    let (prev_points, next_points) = if j == 0 {
                        // First frame: use current for previous
                        (current_points, match contour_type {
                            ContourType::Lumen => &frames[j + 1].lumen.points,
                            _ => match &frames[j + 1].extras.get(&contour_type) {
                                Some(c) => &c.points,
                                None => current_points,
                            }
                        })
                    } else if j == n - 1 {
                        // Last frame: use current for next
                        (match contour_type {
                            ContourType::Lumen => &frames[j - 1].lumen.points,
                            _ => match &frames[j - 1].extras.get(&contour_type) {
                                Some(c) => &c.points,
                                None => current_points,
                            }
                        }, current_points)
                    } else {
                        (match contour_type {
                            ContourType::Lumen => &frames[j - 1].lumen.points,
                            _ => match &frames[j - 1].extras.get(&contour_type) {
                                Some(c) => &c.points,
                                None => current_points,
                            }
                        }, match contour_type {
                            ContourType::Lumen => &frames[j + 1].lumen.points,
                            _ => match &frames[j + 1].extras.get(&contour_type) {
                                Some(c) => &c.points,
                                None => current_points,
                            }
                        })
                    };

                    let curr_point = &current_points[i];
                    let prev_point = &prev_points[i];
                    let next_point = &next_points[i];

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

                // Update the points in the frame
                match contour_type {
                    ContourType::Lumen => frames[j].lumen.points = new_points,
                    _ => if let Some(contour) = frames[j].extras.get_mut(&contour_type) {
                        contour.points = new_points;
                    }
                }
            }
        }

        // Only smooth these contour types
        let smooth_types = [ContourType::Lumen, ContourType::Eem, ContourType::Wall];
        
        for contour_type in smooth_types.iter() {
            smooth_contour_type(&mut self.frames, n, *contour_type);
        }

        self
    }

    pub fn create_walls(&mut self, pulmonary: bool) -> anyhow::Result<Self> {
        let new_frames = wall::create_wall_frames(&self.frames, pulmonary);

        // Simple check for the moment
        if new_frames.len() != self.frames.len() {
            return Err(anyhow::anyhow!(
                "wall::create_wall_frames returned {} walls for {} frames",
                new_frames.len(),
                self.frames.len()
            ));
        }

        // Return a new Geometry with updated frames and the same label.
        Ok(Self {
            frames: new_frames,
            label: self.label.clone(),
        })
    }
}