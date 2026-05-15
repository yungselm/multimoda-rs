use super::geometry::ContourType;

use anyhow::{anyhow, Context, Result};
use csv::ReaderBuilder;
use nalgebra::Vector3;
use serde::Deserialize;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

const RECORD_FILE_NAME: &str = "combined_sorted_manual.csv";

/// InputData stores raw data from typical AIVUS-CAA output
/// where dataframes have the following order
///
/// .. text::
///     
///     frame_idx, x, y, z
///
/// not automatically kept in sync with geometry.rs
#[derive(Debug, Clone)]
pub struct InputData {
    pub lumen: Vec<ContourPoint>,
    pub eem: Option<Vec<ContourPoint>>,
    pub calcification: Option<Vec<ContourPoint>>,
    pub sidebranch: Option<Vec<ContourPoint>>,
    pub record: Option<Vec<Record>>,
    pub ref_point: ContourPoint,
    pub diastole: bool,
    pub label: String,
}

impl InputData {
    pub fn new(
        lumen: Vec<ContourPoint>,
        eem: Option<Vec<ContourPoint>>,
        calcification: Option<Vec<ContourPoint>>,
        sidebranch: Option<Vec<ContourPoint>>,
        record: Option<Vec<Record>>,
        ref_point: ContourPoint,
        diastole: bool,
        label: String,
    ) -> anyhow::Result<InputData> {
        let input = InputData {
            lumen,
            eem,
            calcification,
            sidebranch,
            record,
            ref_point,
            diastole,
            label,
        };
        Ok(input)
    }

    // #[allow(unused_assignments)]
    pub fn process_directory<P: AsRef<Path>>(
        path: P,
        names: HashMap<ContourType, &str>,
        diastole: bool,
        label: &str,
    ) -> anyhow::Result<InputData> {
        let path = path.as_ref();
        let label_string = label.to_string();

        let mut eem: Option<Vec<ContourPoint>> = None;
        let mut calcification: Option<Vec<ContourPoint>> = None;
        let mut sidebranch: Option<Vec<ContourPoint>> = None;
        let mut record: Option<Vec<Record>> = None;

        let phase = if diastole { "diastolic" } else { "systolic" };

        // Read required files - these will crash if missing
        let contours_fname = format!("{phase}_contours.csv");
        let contours_path = path.join(&contours_fname);
        let lumen = if contours_path.exists() {
            ContourPoint::read_contour_data(&contours_path)
                .with_context(|| format!("reading {}", contours_path.display()))?
        } else {
            return Err(anyhow::anyhow!(
                "required contours file missing: {contours_path:?}"
            ));
        };

        let ref_fname = format!("{phase}_reference_points.csv");
        let ref_path = path.join(&ref_fname);
        let ref_point = if ref_path.exists() {
            ContourPoint::read_reference_point(&ref_path)
                .with_context(|| format!("reading {}", ref_path.display()))?
        } else {
            return Err(anyhow::anyhow!(
                "required reference-point file missing: {ref_path:?}"
            ));
        };

        for (_ctype, raw_name) in names.iter() {
            let name = raw_name.trim().to_lowercase();
            match name.as_str() {
                "" | "lumen" => {
                    // Already handled above, skip
                }

                "branch" | "sidebranch" => {
                    let fname = format!("{}_{}_contours.csv", "branch", phase);
                    let p = path.join(&fname);
                    if p.exists() {
                        sidebranch = Some(
                            ContourPoint::read_contour_data(&p)
                                .with_context(|| format!("reading {}", p.display()))?,
                        );
                    } else {
                        eprintln!("sidebranch file not found, skipping: {p:?}");
                    }
                }

                "calcium" | "calcification" => {
                    let fname = format!("{}_{}_contours.csv", "calcium", phase);
                    let p = path.join(&fname);
                    if p.exists() {
                        calcification = Some(
                            ContourPoint::read_contour_data(&p)
                                .with_context(|| format!("reading {}", p.display()))?,
                        );
                    } else {
                        eprintln!("calcification file not found, skipping: {p:?}");
                    }
                }

                "eem" | "e_e_m" => {
                    let fname = format!("{}_{}_contours.csv", "eem", phase);
                    let p = path.join(&fname);
                    if p.exists() {
                        eem = Some(
                            ContourPoint::read_contour_data(&p)
                                .with_context(|| format!("reading {}", p.display()))?,
                        );
                    } else {
                        eprintln!("eem file not found, skipping: {p:?}");
                    }
                }

                "records" | "record" | "phases" => {
                    let fname = "combined_sorted_manual.csv";
                    let p = path.join(fname);
                    if p.exists() {
                        record = Some(
                            read_records(&p).with_context(|| format!("reading {}", p.display()))?,
                        );
                    } else {
                        eprintln!("records file not found, skipping: {p:?}");
                    }
                }

                other => {
                    eprintln!("process_directory: unknown mapping name '{other}', skipping");
                }
            }
        }

        // Also attempt to read records.csv if present even if not requested explicitly
        if record.is_none() {
            let maybe_records = path.join(RECORD_FILE_NAME);
            if maybe_records.exists() {
                record = Some(read_records(&maybe_records).with_context(|| {
                    format!("reading optional records file {}", maybe_records.display())
                })?);
            }
        }

        let input = InputData {
            lumen,
            eem,
            calcification,
            sidebranch,
            record,
            ref_point,
            diastole,
            label: label_string,
        };

        Ok(input)
    }

    // fn quick_check_integrity(&self) {
    //     todo!()
    // }
}

/// Utility: detect whether the file uses comma or tab as delimiter.
fn detect_delimiter<P: AsRef<Path>>(path: P) -> Result<u8> {
    let file = File::open(&path).with_context(|| {
        format!(
            "failed to open file for delimiter sniffing: {:?}",
            path.as_ref()
        )
    })?;
    let mut reader = BufReader::new(file);
    let mut first_line = String::new();
    reader
        .read_line(&mut first_line)
        .with_context(|| "failed to read first line for delimiter detection")?;

    // Count occurrences
    let tabs = first_line.matches('\t').count();
    let commas = first_line.matches(',').count();

    if tabs > commas {
        Ok(b'\t')
    } else if commas > tabs {
        Ok(b',')
    } else {
        // default to comma
        Ok(b',')
    }
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq)]
pub struct ContourPoint {
    pub frame_index: u32,

    #[serde(default, skip_deserializing)]
    pub point_index: u32,

    pub x: f64,
    pub y: f64,
    pub z: f64,

    #[serde(default)]
    pub aortic: bool,
}

impl ContourPoint {
    /// Reads contour points from a CSV file.
    pub fn read_contour_data<P: AsRef<Path> + std::fmt::Debug + Clone>(
        path: P,
    ) -> anyhow::Result<Vec<ContourPoint>> {
        let delim = detect_delimiter(&path)?;
        let file = File::open(path)?;
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .delimiter(delim)
            .from_reader(file);

        let mut points = Vec::new();
        for result in rdr.records() {
            match result {
                Ok(record) => match record.deserialize(None) {
                    Ok(point) => points.push(point),
                    Err(e) => eprintln!("Skipping invalid record: {e:?}"),
                },
                Err(e) => eprintln!("Skipping invalid row: {e:?}"),
            }
        }

        Ok(points)
    }

    pub fn read_reference_point<P: AsRef<Path>>(path: P) -> Result<ContourPoint> {
        let delim = detect_delimiter(&path)?;
        // 1) Open the file, with context on failur
        let file = File::open(&path)
            .with_context(|| format!("failed to open reference-point file {:?}", path.as_ref()))?;

        // 2) Build a TSV reader
        let mut rdr = ReaderBuilder::new()
            .has_headers(false)
            .delimiter(delim)
            .from_reader(file);

        // 3) Grab the first record (if any)…
        let first = rdr
            .deserialize()
            .next()
            // if there was literally no row at all:
            .ok_or_else(|| {
                anyhow!(
                    "reference-point file {:?} was empty — this data is required",
                    path.as_ref()
                )
            })?;

        // 4) And now propagate any parse / I/O error with its own context:
        let point: ContourPoint =
            first.with_context(|| "failed to deserialize first reference-point record")?;
        Ok(point)
    }

    /// Computes the Euclidean distance between two contour points.
    pub fn distance_to(&self, other: &ContourPoint) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Rotates a single point about a given center (cx, cy) by a specified angle (in radians).
    pub fn rotate_point(&self, angle: f64, center: (f64, f64)) -> ContourPoint {
        if angle == 0.0 {
            return *self;
        }
        let (cx, cy) = center;
        let x = self.x - cx;
        let y = self.y - cy;
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        ContourPoint {
            frame_index: self.frame_index,
            point_index: self.point_index,
            x: x * cos_a - y * sin_a + cx,
            y: x * sin_a + y * cos_a + cy,
            z: self.z,
            aortic: self.aortic,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Record {
    pub frame: u32,
    pub phase: String,
    #[serde(deserialize_with = "csv::invalid_option")]
    pub measurement_1: Option<f64>,
    #[serde(deserialize_with = "csv::invalid_option")]
    pub measurement_2: Option<f64>,
}

pub fn read_records<P: AsRef<Path>>(path: P) -> anyhow::Result<Vec<Record>> {
    let delim = detect_delimiter(&path)?;
    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new()
        .delimiter(delim)
        .has_headers(true)
        .from_reader(file);

    let mut records = Vec::new();
    for result in reader.deserialize() {
        let record: Record = result?;
        records.push(record);
    }
    Ok(records)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Centerline {
    pub points: Vec<CenterlinePoint>,
    /// First index in `points` for each branch (branch 0 = main vessel).
    pub branch_start_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CenterlinePoint {
    pub contour_point: ContourPoint,
    pub normal: Vector3<f64>,
    /// 0 = main vessel, 1+ = side branches ordered by descending length.
    pub branch_id: u32,
}

impl Centerline {
    pub fn from_contour_points(contour_points: Vec<ContourPoint>) -> Self {
        let mut points: Vec<CenterlinePoint> = Vec::with_capacity(contour_points.len());

        // Calculate normals for all but the last point.
        for i in 0..contour_points.len() {
            let current = &contour_points[i];
            let normal = if i < contour_points.len() - 1 {
                let next = &contour_points[i + 1];
                Vector3::new(next.x - current.x, next.y - current.y, next.z - current.z).normalize()
            } else if !contour_points.is_empty() {
                points[i - 1].normal
            } else {
                Vector3::zeros()
            };

            points.push(CenterlinePoint {
                contour_point: *current,
                normal,
                branch_id: 0,
            });
        }

        let branch_start_indices = if points.is_empty() { vec![] } else { vec![0] };
        Centerline {
            points,
            branch_start_indices,
        }
    }

    /// Retrieves a centerline point by matching frame index.
    pub fn get_by_frame(&self, frame_index: u32) -> Option<&CenterlinePoint> {
        self.points
            .iter()
            .find(|p| p.contour_point.frame_index == frame_index)
    }

    /// Finds the index of the centerline point closest to the reference point
    pub fn find_reference_cl_point_idx(&self, reference_point: &(f64, f64, f64)) -> usize {
        // Helper function to calculate squared distance (avoids sqrt for performance)
        fn distance_sq(contour_point: &ContourPoint, reference: &(f64, f64, f64)) -> f64 {
            let dx = contour_point.x - reference.0;
            let dy = contour_point.y - reference.1;
            let dz = contour_point.z - reference.2;
            dx * dx + dy * dy + dz * dz
        }

        self.points
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                distance_sq(&a.contour_point, reference_point)
                    .partial_cmp(&distance_sq(&b.contour_point, reference_point))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap()
    }

    /// Partition the centerline into anatomical branches using the tree-diameter algorithm.
    ///
    /// Raw centerline data concatenates vessel segments end-to-end with large
    /// positional jumps (26–86 mm for coronary data) at segment boundaries, while
    /// branches share a bifurcation point with the main vessel at ≈ 0 mm distance.
    ///
    /// A SPARSE TREE adjacency is built to avoid the cycles that arise from a
    /// dense O(n²) graph near bifurcation clusters:
    ///   • within each segment: consecutive edges only
    ///   • between each pair of segments: exactly one edge at the closest point pair
    ///
    /// The tree diameter (double BFS) then gives the longest vessel path = main
    /// branch.  Remaining connected components are side branches.  Tiny components
    /// (< MIN_BRANCH_SIZE pts) are artefacts and are merged into branch 0.
    pub fn calculate_branches(&mut self, spacing_tolerance: f64) {
        const MIN_BRANCH_SIZE: usize = 5;

        let n = self.points.len();
        if n == 0 {
            self.branch_start_indices = vec![];
            return;
        }

        let threshold = self.p95_consecutive_spacing() * spacing_tolerance;

        // Identify segment boundaries (large consecutive gaps).
        let mut seg_starts: Vec<usize> = vec![0];
        for i in 1..n {
            if self.points[i - 1]
                .contour_point
                .distance_to(&self.points[i].contour_point)
                > threshold
            {
                seg_starts.push(i);
            }
        }
        seg_starts.push(n); // sentinel
        let num_segs = seg_starts.len() - 1;

        // Build sparse tree adjacency.
        let mut adj: Vec<Vec<usize>> = vec![vec![]; n];

        // Within-segment: consecutive edges.
        for i in 1..n {
            if self.points[i - 1]
                .contour_point
                .distance_to(&self.points[i].contour_point)
                <= threshold
            {
                adj[i - 1].push(i);
                adj[i].push(i - 1);
            }
        }

        // Between segments: single edge at the closest point pair.
        for si in 0..num_segs {
            let (s0, s1) = (seg_starts[si], seg_starts[si + 1]);
            for sj in (si + 1)..num_segs {
                let (t0, t1) = (seg_starts[sj], seg_starts[sj + 1]);
                let mut best_d = f64::INFINITY;
                let mut best_pi = s0;
                let mut best_pj = t0;
                for pi in s0..s1 {
                    for pj in t0..t1 {
                        let d = self.points[pi]
                            .contour_point
                            .distance_to(&self.points[pj].contour_point);
                        if d < best_d {
                            best_d = d;
                            best_pi = pi;
                            best_pj = pj;
                        }
                    }
                }
                if best_d <= threshold {
                    adj[best_pi].push(best_pj);
                    adj[best_pj].push(best_pi);
                }
            }
        }

        // Double BFS on the tree to find the diameter (longest path = main branch).
        let (a, _) = Self::bfs_farthest(&adj, n, 0);
        let (b, prev) = Self::bfs_farthest(&adj, n, a);
        let main_path = Self::trace_path(b, a, &prev);

        let mut in_main = vec![false; n];
        for &idx in &main_path {
            in_main[idx] = true;
        }

        // BFS connected components of nodes not on the main path.
        let mut visited = in_main.clone();
        let mut side_components: Vec<Vec<usize>> = Vec::new();
        for start in 0..n {
            if visited[start] {
                continue;
            }
            let mut comp = Vec::new();
            let mut q = std::collections::VecDeque::new();
            q.push_back(start);
            visited[start] = true;
            while let Some(node) = q.pop_front() {
                comp.push(node);
                for &nb in &adj[node] {
                    if !visited[nb] {
                        visited[nb] = true;
                        q.push_back(nb);
                    }
                }
            }
            side_components.push(comp);
        }

        // Tiny components are artefacts; merge into branch 0 instead of own branch.
        let mut artefacts: Vec<Vec<usize>> = Vec::new();
        let mut real_branches: Vec<Vec<usize>> = Vec::new();
        for comp in side_components {
            if comp.len() < MIN_BRANCH_SIZE {
                artefacts.push(comp);
            } else {
                real_branches.push(comp);
            }
        }
        real_branches.sort_by_key(|b| std::cmp::Reverse(b.len()));

        let mut new_points: Vec<CenterlinePoint> = Vec::with_capacity(n);
        let mut branch_start_indices: Vec<usize> = Vec::new();

        branch_start_indices.push(0);
        for &idx in &main_path {
            let mut pt = self.points[idx].clone();
            pt.branch_id = 0;
            new_points.push(pt);
        }
        for comp in &artefacts {
            for idx in Self::order_chain(comp, &adj) {
                let mut pt = self.points[idx].clone();
                pt.branch_id = 0;
                new_points.push(pt);
            }
        }

        for (i, comp) in real_branches.iter().enumerate() {
            branch_start_indices.push(new_points.len());
            for idx in Self::order_chain(comp, &adj) {
                let mut pt = self.points[idx].clone();
                pt.branch_id = (i + 1) as u32;
                new_points.push(pt);
            }
        }

        self.points = new_points;
        self.branch_start_indices = branch_start_indices;
        self.recompute_normals();
    }

    /// BFS from `start`; returns the farthest reachable node and a predecessor array.
    fn bfs_farthest(adj: &[Vec<usize>], n: usize, start: usize) -> (usize, Vec<Option<usize>>) {
        let mut dist = vec![usize::MAX; n];
        let mut prev: Vec<Option<usize>> = vec![None; n];
        let mut q = std::collections::VecDeque::new();
        dist[start] = 0;
        q.push_back(start);
        let mut farthest = start;
        while let Some(u) = q.pop_front() {
            for &v in &adj[u] {
                if dist[v] == usize::MAX {
                    dist[v] = dist[u] + 1;
                    prev[v] = Some(u);
                    q.push_back(v);
                    if dist[v] > dist[farthest] {
                        farthest = v;
                    }
                }
            }
        }
        (farthest, prev)
    }

    /// Trace the path from `from` back to `to` using the predecessor array.
    fn trace_path(from: usize, to: usize, prev: &[Option<usize>]) -> Vec<usize> {
        let mut path = Vec::new();
        let mut cur = from;
        loop {
            path.push(cur);
            if cur == to {
                break;
            }
            match prev[cur] {
                Some(p) => cur = p,
                None => break,
            }
        }
        path
    }

    /// 95th-percentile of consecutive-point spacings — O(n).
    ///
    /// Operates only on adjacent pairs in the original ordering so large
    /// inter-segment jumps in the CSV do not inflate the estimate.
    fn p95_consecutive_spacing(&self) -> f64 {
        let n = self.points.len();
        if n < 2 {
            return 1.0;
        }
        let mut spacings: Vec<f64> = (1..n)
            .map(|i| {
                self.points[i - 1]
                    .contour_point
                    .distance_to(&self.points[i].contour_point)
            })
            .collect();
        spacings.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        spacings[(spacings.len() * 95) / 100]
    }

    /// Walk a connected component as a linear chain from a degree-1 endpoint.
    fn order_chain(component: &[usize], adj: &[Vec<usize>]) -> Vec<usize> {
        if component.is_empty() {
            return vec![];
        }
        let in_comp: std::collections::HashSet<usize> = component.iter().copied().collect();
        let &start = component
            .iter()
            .find(|&&idx| adj[idx].iter().filter(|&&nb| in_comp.contains(&nb)).count() <= 1)
            .unwrap_or(&component[0]);
        let mut ordered = Vec::with_capacity(component.len());
        let mut seen = std::collections::HashSet::new();
        let mut current = start;
        loop {
            ordered.push(current);
            seen.insert(current);
            match adj[current]
                .iter()
                .find(|&&nb| in_comp.contains(&nb) && !seen.contains(&nb))
            {
                Some(&next) => current = next,
                None => break,
            }
        }
        for &idx in component {
            if !seen.contains(&idx) {
                ordered.push(idx);
            }
        }
        ordered
    }

    /// Recompute normals after points have been reordered.
    ///
    /// Normals are not computed across branch boundaries so each branch's
    /// last point inherits the direction of its penultimate point.
    fn recompute_normals(&mut self) {
        let n = self.points.len();
        for i in 0..n {
            let normal = if i + 1 < n && self.points[i].branch_id == self.points[i + 1].branch_id {
                let a = &self.points[i].contour_point;
                let b = &self.points[i + 1].contour_point;
                Vector3::new(b.x - a.x, b.y - a.y, b.z - a.z).normalize()
            } else if i > 0 && self.points[i - 1].branch_id == self.points[i].branch_id {
                self.points[i - 1].normal
            } else {
                Vector3::zeros()
            };
            self.points[i].normal = normal;
        }
    }
}

#[cfg(test)]
mod input_tests {
    use super::*;

    #[test]
    fn test_process_directory_runs_with_example_data() -> anyhow::Result<()> {
        use crate::intravascular::io::geometry::ContourType;

        let mut names: HashMap<ContourType, &str> = HashMap::new();
        names.insert(ContourType::Lumen, "lumen");
        names.insert(ContourType::Eem, "eem");
        names.insert(ContourType::Calcification, "calcification");
        names.insert(ContourType::Sidebranch, "sidebranch");

        let data_path = Path::new("./data/fixtures/idealized_geometry");
        let input = InputData::process_directory(data_path, names, true, "")?;

        assert!(
            !input.lumen.is_empty(),
            "lumen contour vector should not be empty"
        );
        assert!(
            input.eem.is_some(),
            "eem contour vector should not be empty"
        );
        assert!(
            input.calcification.is_some(),
            "calcification contour vector should not be empty"
        );
        assert!(input.record.is_none(), "record vector should be empty");
        assert!(input.ref_point.x > 0.0, "ref_point should not be empty");

        Ok(())
    }

    #[test]
    fn test_contour_point_distance() {
        let p1 = ContourPoint {
            frame_index: 1,
            point_index: 0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            aortic: false,
        };
        let p2 = ContourPoint {
            frame_index: 1,
            point_index: 1,
            x: 3.0,
            y: 4.0,
            z: 0.0,
            aortic: false,
        };
        assert!((p1.distance_to(&p2) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_rotate_point() {
        use std::f64::consts::PI;

        let p = ContourPoint {
            frame_index: 1,
            point_index: 0,
            x: 1.0,
            y: 0.0,
            z: 0.0,
            aortic: false,
        };
        let rotated = p.rotate_point(PI / 2.0, (0.0, 0.0));
        assert!((rotated.x - 0.0).abs() < 1e-6);
        assert!((rotated.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_centerline_normals() {
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
                frame_index: 2,
                point_index: 1,
                x: 1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 3,
                point_index: 2,
                x: 2.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
        ];
        let centerline = Centerline::from_contour_points(points);
        assert_eq!(centerline.points[0].normal, Vector3::new(1.0, 0.0, 0.0));
        assert_eq!(centerline.points[1].normal, Vector3::new(1.0, 0.0, 0.0));
        assert_eq!(centerline.points[2].normal, Vector3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn test_cl_find_ref_pt() {
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
                frame_index: 2,
                point_index: 1,
                x: 1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 3,
                point_index: 2,
                x: 2.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
        ];
        let centerline = Centerline::from_contour_points(points);
        let ref_pt = (0.0, 0.0, 0.0);
        let ref_id = centerline.find_reference_cl_point_idx(&ref_pt);
        assert_eq!(centerline.points[0], centerline.points[ref_id]);
    }
}
