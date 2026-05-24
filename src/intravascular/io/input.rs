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
    /// (< MIN_BRANCH_SIZE pts) are artefacts and are discarded.
    pub fn calculate_branches(&mut self, spacing_tolerance: f64) {
        const MIN_BRANCH_SIZE: usize = 5;

        let n_points = self.points.len();
        if n_points == 0 {
            self.branch_start_indices = vec![];
            return;
        }

        let threshold = self.p95_consecutive_spacing() * spacing_tolerance;

        // Identify segment boundaries (large consecutive gaps).
        let mut seg_starts: Vec<usize> = vec![0];
        for i in 1..n_points {
            if self.points[i - 1]
                .contour_point
                .distance_to(&self.points[i].contour_point)
                > threshold
            {
                seg_starts.push(i);
            }
        }
        seg_starts.push(n_points); // sentinel

        let adj_map = self.build_sparse_tree_adjacency(seg_starts, n_points, threshold);

        let (main_path, side_components) = self.identify_components_with_bfs(&adj_map, n_points);

        // Tiny components are artefacts; discard instead of treating as own branch.
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

        let ordered_real_branches: Vec<Vec<usize>> = real_branches
            .into_iter()
            .map(|b| Self::order_chain(&b, &adj_map))
            .collect();

        let mut new_points: Vec<CenterlinePoint> = Vec::with_capacity(n_points);
        let mut branch_start_indices: Vec<usize> = Vec::new();
        let mut global_idx: u32 = 0;

        branch_start_indices.push(0);
        for &idx in &main_path {
            let mut pt = self.points[idx].clone();
            pt.branch_id = 0;
            pt.contour_point.point_index = global_idx;
            global_idx += 1;
            new_points.push(pt);
        }
        // Artefacts (< MIN_BRANCH_SIZE pts) are disconnected noise; drop them entirely
        // rather than merging into branch 0, where they would corrupt arc-length
        // calculations and z-based reversal in the alignment pipeline.
        let _ = artefacts;

        for (i, branch) in ordered_real_branches.iter().enumerate() {
            branch_start_indices.push(new_points.len());
            // branch is already spatially ordered from the pre-sort above
            for &idx in branch {
                let mut pt = self.points[idx].clone();
                pt.branch_id = (i + 1) as u32;
                pt.contour_point.point_index = global_idx;
                global_idx += 1;
                new_points.push(pt);
            }
        }

        self.points = new_points;
        self.branch_start_indices = branch_start_indices;
        self.recompute_normals();
    }

    /// Build a sparse tree adjacency map. Within the segment and then between segments
    fn build_sparse_tree_adjacency(
        &self,
        seg_starts: Vec<usize>,
        n_points: usize,
        threshold: f64,
    ) -> Vec<Vec<usize>> {
        let num_segs = seg_starts.len() - 1;

        let mut adj: Vec<Vec<usize>> = vec![vec![]; n_points];

        // Within-segment: consecutive edges.
        for i in 1..n_points {
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
        adj
    }

    fn identify_components_with_bfs(
        &self,
        adj_map: &[Vec<usize>],
        n_points: usize,
    ) -> (Vec<usize>, Vec<Vec<usize>>) {
        // Double BFS on the tree to find the diameter (longest path = main branch).
        let (a, _) = Self::bfs_farthest(adj_map, n_points, 0);
        let (b, prev) = Self::bfs_farthest(adj_map, n_points, a);
        let main_path = Self::trace_path(b, a, &prev);

        let mut in_main_branch = vec![false; n_points];
        for &idx in &main_path {
            in_main_branch[idx] = true;
        }

        // BFS connected components of nodes not on the main path.
        let mut visited = in_main_branch.clone();
        let mut side_components: Vec<Vec<usize>> = Vec::new();
        for start in 0..n_points {
            if visited[start] {
                continue;
            }
            let mut comp = Vec::new();
            let mut q = std::collections::VecDeque::new();
            q.push_back(start);
            visited[start] = true;
            while let Some(node) = q.pop_front() {
                comp.push(node);
                for &nb in &adj_map[node] {
                    if !visited[nb] {
                        visited[nb] = true;
                        q.push_back(nb);
                    }
                }
            }
            side_components.push(comp);
        }
        (main_path, side_components)
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

    /// Decompose the flat points Vec into one Vec per branch.
    fn branches_as_vecs(&self) -> Vec<Vec<CenterlinePoint>> {
        let n = self.branch_start_indices.len();
        (0..n)
            .map(|i| {
                let start = self.branch_start_indices[i];
                let end = if i + 1 < n {
                    self.branch_start_indices[i + 1]
                } else {
                    self.points.len()
                };
                self.points[start..end].to_vec()
            })
            .collect()
    }

    /// Rebuild the flat points Vec and branch_start_indices from a list of branch segments,
    /// reassigning branch_id and point_index sequentially.
    fn rebuild_from_branches(&mut self, branches: Vec<Vec<CenterlinePoint>>) {
        let total: usize = branches.iter().map(|b| b.len()).sum();
        let mut new_points: Vec<CenterlinePoint> = Vec::with_capacity(total);
        let mut branch_start_indices: Vec<usize> = Vec::with_capacity(branches.len());
        let mut global_idx: u32 = 0;

        for (branch_id, branch) in branches.into_iter().enumerate() {
            branch_start_indices.push(new_points.len());
            for mut pt in branch {
                pt.branch_id = branch_id as u32;
                pt.contour_point.point_index = global_idx;
                global_idx += 1;
                new_points.push(pt);
            }
        }

        self.points = new_points;
        self.branch_start_indices = branch_start_indices;
        self.recompute_normals();
    }

    /// Return local positions (0-indexed within the branch) of interior points
    /// where the opening angle satisfies `cos_angle > cos_threshold`.
    /// Use `cos_threshold = 0.0` for < 90°, `0.5` for < 60°, etc.
    pub fn find_sharp_angles(&self, branch_id: u32, cos_threshold: f64) -> Vec<usize> {
        let idx = branch_id as usize;
        let n = self.branch_start_indices.len();
        if idx >= n {
            return vec![];
        }
        let start = self.branch_start_indices[idx];
        let end = if idx + 1 < n {
            self.branch_start_indices[idx + 1]
        } else {
            self.points.len()
        };
        let branch = &self.points[start..end];

        (1..branch.len().saturating_sub(1))
            .filter(|&i| {
                let prev = &branch[i - 1].contour_point;
                let curr = &branch[i].contour_point;
                let next = &branch[i + 1].contour_point;
                let v1 = Vector3::new(prev.x - curr.x, prev.y - curr.y, prev.z - curr.z);
                let v2 = Vector3::new(next.x - curr.x, next.y - curr.y, next.z - curr.z);
                let n1 = v1.norm();
                let n2 = v2.norm();
                if n1 < 1e-10 || n2 < 1e-10 {
                    return false;
                }
                v1.dot(&v2) / (n1 * n2) > cos_threshold
            })
            .collect()
    }

    /// Split the branch at `local_pos` (0-indexed within the branch).
    /// Both resulting segments include the split point.
    /// If `branch_id == 0` the longer segment stays as branch 0.
    /// For side branches the first segment keeps its slot; the second is appended.
    pub fn split_branch(&mut self, branch_id: u32, local_pos: usize) {
        let mut branches = self.branches_as_vecs();
        let idx = branch_id as usize;
        if idx >= branches.len() {
            return;
        }
        let branch = branches.remove(idx);
        if local_pos == 0 || local_pos >= branch.len().saturating_sub(1) {
            branches.insert(idx, branch);
            return;
        }

        let seg_a = branch[..=local_pos].to_vec();
        let seg_b = branch[local_pos..].to_vec();

        if branch_id == 0 {
            let (main_seg, other_seg) = if seg_a.len() >= seg_b.len() {
                (seg_a, seg_b)
            } else {
                (seg_b, seg_a)
            };
            branches.insert(0, main_seg);
            branches.push(other_seg);
        } else {
            branches.insert(idx, seg_a);
            branches.push(seg_b);
        }

        self.rebuild_from_branches(branches);
    }

    /// Merge two branches into one. Endpoints are matched by minimum distance
    /// so the segments are concatenated in the correct spatial order.
    /// If either branch is the main branch (id 0) the merged result is branch 0.
    pub fn merge_branches(&mut self, branch_id_a: u32, branch_id_b: u32) {
        let mut branches = self.branches_as_vecs();
        let idx_a = branch_id_a as usize;
        let idx_b = branch_id_b as usize;
        if idx_a == idx_b || idx_a >= branches.len() || idx_b >= branches.len() {
            return;
        }

        let (low, high) = if idx_a < idx_b {
            (idx_a, idx_b)
        } else {
            (idx_b, idx_a)
        };
        let b_high = branches.remove(high);
        let b_low = branches.remove(low);

        let lf = &b_low[0].contour_point;
        let ll = &b_low[b_low.len() - 1].contour_point;
        let hf = &b_high[0].contour_point;
        let hl = &b_high[b_high.len() - 1].contour_point;

        // Find the orientation that puts the closest endpoints adjacent.
        let d_ll_hf = ll.distance_to(hf);
        let d_ll_hl = ll.distance_to(hl);
        let d_lf_hf = lf.distance_to(hf);
        let d_lf_hl = lf.distance_to(hl);
        let min_d = d_ll_hf.min(d_ll_hl).min(d_lf_hf).min(d_lf_hl);

        let merged: Vec<CenterlinePoint> = if (min_d - d_ll_hf).abs() < 1e-12 {
            b_low.into_iter().chain(b_high).collect()
        } else if (min_d - d_ll_hl).abs() < 1e-12 {
            b_low.into_iter().chain(b_high.into_iter().rev()).collect()
        } else if (min_d - d_lf_hf).abs() < 1e-12 {
            b_high.into_iter().rev().chain(b_low).collect()
        } else {
            b_high.into_iter().chain(b_low).collect()
        };

        let result_is_main = low == 0 || high == 0;
        if result_is_main {
            branches.insert(0, merged);
        } else {
            branches.insert(low, merged);
        }

        self.rebuild_from_branches(branches);
    }

    /// Ensure consistent ordering across all branches:
    ///
    /// * **Branch 0** – the point with the highest z-coordinate must be at index 0.
    ///   If it is not, the entire branch is reversed.
    /// * **Side branches (1 … n-1)** – the endpoint that is closest to any point on
    ///   branch 0 must be the *first* point of the branch.  If the last point is
    ///   closer to branch 0 than the first, the branch is reversed.
    pub fn check_centerline(&mut self) {
        let n = self.branch_start_indices.len();
        if n == 0 {
            return;
        }

        let mut branches = self.branches_as_vecs();

        // --- Branch 0: highest-z point must be at index 0 ---
        if !branches[0].is_empty() {
            let max_z_idx = branches[0]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.contour_point
                        .z
                        .partial_cmp(&b.contour_point.z)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            if max_z_idx != 0 {
                branches[0].reverse();
            }
        }

        // --- Side branches: first point must be the one closest to branch 0 ---
        for k in 1..n {
            if branches[k].is_empty() || branches[0].is_empty() {
                continue;
            }

            // Clone endpoints so we can borrow branches[0] freely afterwards.
            let first_pt = branches[k][0].contour_point;
            let last_pt = branches[k].last().unwrap().contour_point;

            let dist_first = branches[0]
                .iter()
                .map(|p| p.contour_point.distance_to(&first_pt))
                .fold(f64::INFINITY, f64::min);

            let dist_last = branches[0]
                .iter()
                .map(|p| p.contour_point.distance_to(&last_pt))
                .fold(f64::INFINITY, f64::min);

            if dist_last < dist_first {
                branches[k].reverse();
            }
        }

        self.rebuild_from_branches(branches);
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

    fn cl_from_coords(coords: &[(f64, f64, f64)]) -> Centerline {
        let points = coords
            .iter()
            .enumerate()
            .map(|(i, &(x, y, z))| ContourPoint {
                frame_index: i as u32,
                point_index: i as u32,
                x,
                y,
                z,
                aortic: false,
            })
            .collect();
        Centerline::from_contour_points(points)
    }

    #[test]
    fn test_find_sharp_angles_straight() {
        let cl = cl_from_coords(&[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
        ]);
        assert!(cl.find_sharp_angles(0, 0.0).is_empty());
    }

    #[test]
    fn test_find_sharp_angles_v_shape() {
        // At position 3 the path makes a sharp turn back; cos ≈ 0.707 (> 0.0).
        let cl = cl_from_coords(&[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0), // elbow
            (2.5, 0.5, 0.0),
            (2.0, 1.0, 0.0),
        ]);
        assert_eq!(cl.find_sharp_angles(0, 0.0), vec![3]);
        // With threshold 0.8 (cos(36°)≈0.81) the ≈0.707 elbow is not detected.
        assert!(cl.find_sharp_angles(0, 0.8).is_empty());
        // Out-of-range branch returns empty.
        assert!(cl.find_sharp_angles(5, 0.0).is_empty());
    }

    #[test]
    fn test_split_branch_main_longer_stays() {
        // 9 points, split at pos 3 → seg_a=4pts, seg_b=6pts → seg_b (longer) is branch 0.
        let mut cl = cl_from_coords(&[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            (5.0, 0.0, 0.0),
            (6.0, 0.0, 0.0),
            (7.0, 0.0, 0.0),
            (8.0, 0.0, 0.0),
        ]);
        cl.split_branch(0, 3);
        assert_eq!(cl.branch_start_indices.len(), 2);
        assert_eq!(cl.points.len(), 10); // split point shared
        let len0 = cl.branch_start_indices[1];
        let len1 = cl.points.len() - cl.branch_start_indices[1];
        assert_eq!(len0, 6, "longer segment must be branch 0");
        assert_eq!(len1, 4);
        // branch_id and point_index are sequential
        assert!(cl.points.iter().enumerate().all(|(i, p)| {
            p.branch_id == if i < 6 { 0 } else { 1 } && p.contour_point.point_index == i as u32
        }));
    }

    #[test]
    fn test_split_branch_equal_length_first_is_main() {
        // 5 points, split at pos 2 → both segments 3 pts → first (seg_a) stays as branch 0.
        let mut cl = cl_from_coords(&[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
        ]);
        cl.split_branch(0, 2);
        assert_eq!(cl.branch_start_indices.len(), 2);
        assert_eq!(cl.branch_start_indices[1], 3, "branch 0 has 3 pts");
    }

    #[test]
    fn test_merge_branches_result_is_main() {
        // Split a 5-point line into [0,1,2] and [2,3,4], then merge back.
        // The shared endpoint causes d_ll_hf == 0 → b_low + b_high concatenation.
        let mut cl = cl_from_coords(&[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
        ]);
        cl.split_branch(0, 2);
        assert_eq!(cl.branch_start_indices.len(), 2);

        cl.merge_branches(0, 1);
        assert_eq!(cl.branch_start_indices.len(), 1);
        assert_eq!(cl.points.len(), 6); // 3+3, shared point duplicated
        assert!(cl.points.iter().all(|p| p.branch_id == 0));
        // point_index must be 0..5 sequentially
        for (i, p) in cl.points.iter().enumerate() {
            assert_eq!(p.contour_point.point_index, i as u32);
        }
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
