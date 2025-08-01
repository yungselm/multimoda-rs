pub mod input;
pub mod load_geometry;
pub mod output;

use anyhow::Context;
use input::{read_records, Contour, ContourPoint, Record};
use std::path::Path;
use std::collections::HashMap;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ContourKind {
    Lumen,
    Wall,
    Catheter,
    // EEM, Plaque, Stent, … future shapes
}

pub struct ContourGroup {
    pub kind: ContourKind,
    pub contours: Vec<Contour>,
}

pub struct Geometry {
    pub groups: HashMap<ContourKind, ContourGroup>,
    pub label: String,
}

impl Geometry {
    /// Creates a new Geometry instance by loading all required data files
    pub fn new(
        input_dir: &str,
        contour_file: &str,
        reference_file: &str,
        records_file: &str,
        label: String,
        diastole: bool,
        image_center: (f64, f64),
        radius: f64,
        n_points: u32,
    ) -> anyhow::Result<Self> {
        let base = Path::new(input_dir);

        let reference_path = base.join(reference_file);
        let contour_path  = base.join(contour_file);
        let records_path  = base.join(records_file);

        // load reference point
        let ref_pts = ContourPoint::read_reference_points(&reference_path)
            .with_context(|| format!("failed to load reference: {}", reference_path.display()))?;

        // load any existing measurements
        let records = read_records(&records_path).unwrap_or_default();

        // load raw contour points and infer records if needed
        let raw_points = ContourPoint::read_contour_data(&contour_path)?;
        let inferred = if records.is_empty() {
            Self::infer_records(&raw_points, diastole)
        } else {
            records
        };

        // finally build contours
        let mut contours = Contour::create_contours(raw_points, inferred.clone(), &ref_pts)
            .with_context(|| format!("Failed to build contours from {}", contour_path.display()))?;

        // since reordeing the frames, destroys the z-coordinates of everyframe they need to be stored here
        // and then be reused after reordering them
        let z_coords = Self::extract_zs(&contours);
        Self::reorder(&mut contours, &inferred, diastole, &z_coords);

        let mut catheter = if n_points > 0 {
            Contour::create_catheter_contours(
                &contours.iter().flat_map(|c| c.points.clone()).collect(),
                image_center,
                radius,
                n_points,
            )
            .unwrap_or_default()
        } else {
            Vec::new()
        };

        //sort catheter in ascending order
        catheter.sort_by_key(|c| c.id);

        let mut groups: HashMap<ContourKind, ContourGroup> = HashMap::new();
        groups.insert(
            ContourKind::Lumen,
            ContourGroup {
                kind: ContourKind::Lumen,
                contours: contours.clone(),
            }
        );
        groups.insert(
            ContourKind::Catheter,
            ContourGroup {
                kind: ContourKind::Catheter,
                contours: catheter,
            }
        );
        groups.insert(
            ContourKind::Wall,
            ContourGroup {
                kind: ContourKind::Wall,
                contours: Vec::new(), // Walls are calculated at the end of the pipeline
            }
        );

        let contours_loaded = !contours.is_empty();
        let reference_loaded = !ref_pts.is_empty();
        let records_loaded = !inferred.is_empty();

        println!("Generating geometry for {:?}", input_dir);
        println!("{:<50} {}", "file/path", "loaded");
        println!("{:<50} {}", contour_path.display(), contours_loaded);
        println!("{:<50} {}", reference_path.display(), reference_loaded);
        println!("{:<50} {}", records_path.display(), records_loaded);

        Ok(Geometry { groups, label})
    }

    fn infer_records(points: &[ContourPoint], diastole: bool) -> Vec<Record> {
        let phase = if diastole { "D" } else { "S" };
        let mut seen = std::collections::HashSet::new();
        points.iter().filter_map(|p| {
            if seen.insert(p.frame_index) {
                Some(Record { frame: p.frame_index, phase: phase.into(), measurement_1: None, measurement_2: None })
            } else { None }
        }).collect()
    }

    fn extract_zs(contours: &[Contour]) -> Vec<f64> {
        let mut zs: Vec<f64> = contours.iter().map(|c| c.centroid.2).collect();
        zs.sort_by(|a,b| a.partial_cmp(b).unwrap());
        zs
    }

    /// Reorders contours by record frame order, updates z-coordinates and ids
    pub fn reorder(
        contours: &mut Vec<Contour>,
        records: &[Record],
        diastole: bool,
        z_coords: &[f64],
    ) {
        if records.is_empty() {
            return;
        }

        // reorder contours by records frame order, first filter only phase == 'D' if diastole true
        // otherwise only phase == 'S'
        let phase = if diastole { "D" } else { "S" };
        let filtered: Vec<u32> = records
            .iter()
            .filter(|r| r.phase == phase)
            .map(|r| r.frame)
            .collect();

        // Sort contours to match filtered record frame order
        contours.sort_by_key(|c| {
            filtered
                .iter()
                .position(|&f| f == c.id as u32)
                .unwrap_or(usize::MAX)
        });

        // Update the z-coordinates of contours and their points using z_coords
        for (i, contour) in contours.iter_mut().enumerate() {
            contour.centroid.2 = z_coords[i];

            for pt in contour.points.iter_mut() {
                pt.z = z_coords[i];
            }
        }

        // Reassign indices for contours and update their points' frame_index accordingly.
        for (new_id, contour) in contours.iter_mut().enumerate() {
            contour.id = new_id as u32;
            for pt in contour.points.iter_mut() {
                pt.frame_index = new_id as u32;
            }
        } // new order has now highest index for the ostium
    }

    /// Smooths the x and y coordinates of the contours using a 3‐point moving average.
    ///
    /// For each point i in contour j, the new x and y values are computed as:
    ///     new_x = (prev_contour[i].x + current_contour[i].x + next_contour[i].x) / 3.0
    ///     new_y = (prev_contour[i].y + current_contour[i].y + next_contour[i].y) / 3.0
    /// while the z coordinate remains unchanged (taken from the current contour).
    ///
    /// For the first and last contours, the current contour is used twice to simulate a mirror effect.
    pub fn smooth(&mut self, kind: ContourKind) {
        if kind == ContourKind::Catheter {
            return;
        }
        if let Some(group) = self.groups.get_mut(&kind) {
            let n = group.contours.len();
            if n == 0 {
                return;
            }
            // Ensure uniform point count
            let m = group.contours[0].points.len();
            assert!(
                group.contours.iter().all(|c| c.points.len() == m),
                "All contours in {:?} must have the same number of points",
                kind
            );
            let mut smoothed = Vec::with_capacity(n);
            for j in 0..n {
                let curr = &group.contours[j];
                let prev = &group.contours[if j == 0 { 0 } else { j - 1 }];
                let next = &group.contours[if j + 1 == n { j } else { j + 1 }];
                let mut pts = Vec::with_capacity(m);
                for i in 0..m {
                    let p_prev = &prev.points[i];
                    let p_curr = &curr.points[i];
                    let p_next = &next.points[i];
                    let new_x = (p_prev.x + p_curr.x + p_next.x) / 3.0;
                    let new_y = (p_prev.y + p_curr.y + p_next.y) / 3.0;
                    pts.push(ContourPoint {
                        frame_index: p_curr.frame_index,
                        point_index: p_curr.point_index,
                        x: new_x,
                        y: new_y,
                        z: p_curr.z,
                        aortic: p_curr.aortic,
                    });
                }
                let centroid = Contour::compute_centroid(&pts);
                smoothed.push(Contour {
                    id: curr.id,
                    points: pts,
                    centroid,
                    reference_point: curr.reference_point,
                    aortic_thickness: curr.aortic_thickness,
                    pulmonary_thickness: curr.pulmonary_thickness,
                });
            }
            group.contours = smoothed;
        }
    }

    /// Rotate a single contour by its ID. Lumens rotate around image center,
    /// others rotate around each contour's centroid.
    pub fn rotate_contour_by_id(
        &mut self,
        id: u32,
        angle: f64,
        image_center: (f64, f64),
    ) {
        // Find the centroid of the Lumen contour (if present)
        let lumen_centroid = self.groups.get(&ContourKind::Lumen)
            .and_then(|group| group.contours.iter().find(|c| c.id == id))
            .map(|c| (c.centroid.0, c.centroid.1));

        for (kind, group) in self.groups.iter_mut() {
            if let Some(contour) = group.contours.iter_mut().find(|c| c.id == id) {
                match kind {
                    ContourKind::Lumen => {
                        contour.rotate_contour(angle);
                    }
                    _ => {
                        // Use Lumen centroid if available, otherwise fallback to image_center
                        let pivot = lumen_centroid.unwrap_or(image_center);
                        contour.rotate_contour_around_point(angle, pivot);
                    }
                }
                break;
            }
        }
    }

    /// Rotate every contour:
    /// - Lumen contours use their own centroid
    /// - Other contours use the Lumen centroid if found, otherwise `image_center`
    pub fn rotate_all(&mut self, angle: f64, image_center: (f64, f64)) {
        // Look up one Lumen contour’s centroid to use as pivot for the others
        let lumen_pivot = self.groups.get(&ContourKind::Lumen)
            .and_then(|g| g.contours.first())
            .map(|c| (c.centroid.0, c.centroid.1));
        
        // Parallel over each group
        self.groups.par_iter_mut().for_each(|(kind, group)| {
            group.contours.par_iter_mut().for_each(|contour| {
                match kind {
                    ContourKind::Lumen => {
                        // Rotate around its own centroid
                        contour.rotate_contour(angle);
                    }
                    _ => {
                        // Use the cached Lumen pivot if present, else fallback
                        let pivot = lumen_pivot.unwrap_or(image_center);
                        contour.rotate_contour_around_point(angle, pivot);
                    }
                }
            });
        });
    }

    /// Translate all contours and reference points by (dx,dy,dz) in parallel
    pub fn translate_all(&mut self, dx: f64, dy: f64, dz: f64) {
        // Parallel translate each group's contours
        self.groups.par_iter_mut().for_each(|(_, group)| {
            group.contours.par_iter_mut().for_each(|contour| {
                contour.translate_contour((dx, dy, dz));
            });
        });
    }

    /// Sort points in all contours counterclockwise and reindex, done in parallel
    pub fn sort_all_points(&mut self) {
        self.groups.par_iter_mut().for_each(|(_, group)| {
            group.contours.par_iter_mut().for_each(|contour| {
                contour.sort_contour_points();
            });
        });
    }

    /// Remap a single contour ID across *all* kinds.
    ///
    /// Every contour whose `id == old_id` will be changed to `new_id`, and
    /// every ContourPoint (in `points` and `reference_point`) will have its
    /// `frame_index` updated to `new_id`.
    pub fn remap_contour_id_across_kinds(&mut self, old_id: u32, new_id: u32) {
        for group in self.groups.values_mut() {
            for contour in &mut group.contours {
                if contour.id == old_id {
                    // 1) Update the contour’s own ID
                    contour.id = new_id;

                    // 2) Patch all its points
                    for pt in &mut contour.points {
                        pt.frame_index = new_id;
                    }

                    // 3) Patch the optional reference point
                    if let Some(ref mut rp) = contour.reference_point {
                        rp.frame_index = new_id;
                    }
                    // (We *don’t* break here, since there might be
                    // multiple groups all containing old_id)
                }
            }
        }
    }
}

#[cfg(test)]
mod geometry_tests {
    use super::*;
    use approx::assert_relative_eq;
    use serde_json::Value;
    use std::fs::File;

    const NUM_POINTS_CATHETER: usize = 20;

    fn load_test_manifest(mode: &str) -> Value {
        let manifest_path = format!("data/fixtures/{}_csv_files/test_manifest.json", mode);
        let file = File::open(manifest_path).expect("Failed to open manifest");
        serde_json::from_reader(file).expect("Failed to parse manifest")
    }

    #[test]
    fn test_reorder_matches_manifest_indices() {
        let mode = "rest";
        let manifest = load_test_manifest(mode);
        let dia_expected: Vec<u32> = manifest["dia"]["expected_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();

        // Load raw records and geometry
        let input_dir = format!("data/fixtures/{0}_csv_files", mode);
        let geometry = Geometry::new(
            &input_dir, 
            "diastolic_contours.csv",
            "diastolic_reference_points.csv",
            "combined_sorted_manual.csv",
            "test".into(), 
            true,
            (4.5, 4.5), 
            0.5, 
            20).unwrap();
        let records =
            read_records(&Path::new(&input_dir).join("combined_sorted_manual.csv"))
                .unwrap();
        let filtered: Vec<u32> = records
            .into_iter()
            .filter(|r| r.phase == "D")
            .map(|r| r.frame)
            .collect();

        let lumen_contours = &geometry.groups.get(&ContourKind::Lumen).unwrap().contours;
        // Map reordered contours back to original frame indices
        let actual_sequence: Vec<u32> = lumen_contours
            .iter()
            .map(|c| filtered[c.id as usize])
            .collect();

        assert_eq!(actual_sequence, dia_expected);
    }

    #[test]
    fn test_rest_diastolic_config_match() {
        let geometry = Geometry::new(
            "data/fixtures/rest_csv_files", 
            "diastolic_contours.csv",
            "diastolic_reference_points.csv",
            "combined_sorted_manual.csv",
            "test".into(), 
            true,
            (4.5, 4.5), 
            0.5, 
            20,
        )
        .expect("Failed to load geometry");

        let manifest = load_test_manifest("rest");
        let dia_config = &manifest["dia"];

        let lumen_contours = &geometry.groups.get(&ContourKind::Lumen).unwrap().contours;
        
        assert_eq!(
            lumen_contours.len(),
            dia_config["num_contours"].as_u64().unwrap() as usize,
            "Contour count mismatch"
        );

        let expected_indices: Vec<u32> = (0..lumen_contours.len() as u32).collect();
        let actual_indices: Vec<u32> = lumen_contours.iter().map(|c| c.id).collect();

        assert_eq!(
            actual_indices, expected_indices,
            "Frame indices ordering mismatch"
        );
    }

    #[test]
    fn test_contour_property_consistency() {
        let geometry = Geometry::new(
            "data/fixtures/rest_csv_files", 
            "diastolic_contours.csv",
            "diastolic_reference_points.csv",
            "combined_sorted_manual.csv",
            "test".into(), 
            true,
            (4.5, 4.5), 
            0.5, 
            20,
        )
        .expect("Failed to load geometry");

        let manifest = load_test_manifest("rest");
        let dia_config = &manifest["dia"];

        // Access lumen contours
        let lumen_contours = &geometry.groups.get(&ContourKind::Lumen).unwrap().contours;
        
        for (i, contour) in lumen_contours.iter().enumerate() {
            // Verify elliptic ratio
            let expected_ratio = dia_config["elliptic_ratios"][i].as_f64().unwrap();
            assert_relative_eq!(
                contour.elliptic_ratio(),
                expected_ratio,
                epsilon = 0.1
            );

            // Verify area
            let expected_area = dia_config["areas"][i].as_f64().unwrap();
            assert_relative_eq!(contour.area(), expected_area, epsilon = 3.0);

            // Verify aortic thickness
            let expected_thickness = match dia_config["aortic_thickness"][i].as_f64() {
                Some(v) => Some(v),
                None => None,
            };
            assert_eq!(
                contour.aortic_thickness, expected_thickness,
                "Aortic thickness mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_catheter_contour_properties() {
        let geometry = Geometry::new(
            "data/fixtures/rest_csv_files", 
            "diastolic_contours.csv",
            "diastolic_reference_points.csv",
            "combined_sorted_manual.csv",
            "test".into(),
            true,
            (4.5, 4.5), 
            0.5, 
            20,
        )
        .expect("Failed to load geometry");

        // Access catheter contours
        let catheter_contours = &geometry.groups.get(&ContourKind::Catheter).unwrap().contours;
        
        // Verify number of catheter points per contour
        for catheter_contour in catheter_contours {
            assert_eq!(
                catheter_contour.points.len(),
                NUM_POINTS_CATHETER,
                "Incorrect number of catheter points"
            );
        }

        // Access lumen contours for z-coordinate comparison
        let lumen_contours = &geometry.groups.get(&ContourKind::Lumen).unwrap().contours;
        
        // Verify z-coordinate consistency
        for (lumen_contour, catheter_contour) in lumen_contours.iter().zip(catheter_contours) {
            assert_relative_eq!(catheter_contour.centroid.2, lumen_contour.centroid.2, epsilon = 1e-6);
        }
    }
}
