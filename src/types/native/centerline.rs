use super::centerline_point::CenterlinePoint;
use super::contour_point::ContourPoint;
use nalgebra::Vector3;

#[derive(Debug, Clone, PartialEq)]
pub struct Centerline {
    pub points: Vec<CenterlinePoint>,
    /// First index in `points` for each branch (branch 0 = main vessel).
    pub branch_start_indices: Vec<usize>,
}

impl Centerline {
    pub fn from_contour_points(contour_points: Vec<ContourPoint>) -> Self {
        let mut points: Vec<CenterlinePoint> = Vec::with_capacity(contour_points.len());

        // Calculate normals for all but the last point.
        for i in 0..contour_points.len() {
            let current = &contour_points[i];
            let tangent = if i < contour_points.len() - 1 {
                let next = &contour_points[i + 1];
                Vector3::new(next.x - current.x, next.y - current.y, next.z - current.z).normalize()
            } else if !contour_points.is_empty() {
                points[i - 1].tangent
            } else {
                Vector3::zeros()
            };

            points.push(CenterlinePoint {
                contour_point: *current,
                tangent,
                branch_id: 0,
                radius: 0.0,
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
        self.recompute_tangents();
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

    /// Mean arc-length spacing between consecutive points of branch 0.
    ///
    /// Only intra-branch pairs are considered.  Returns `1.0` if branch 0
    /// has fewer than two points.
    pub fn mean_spacing(&self) -> f64 {
        let end = self
            .branch_start_indices
            .get(1)
            .copied()
            .unwrap_or(self.points.len());
        let main = &self.points[..end];
        if main.len() < 2 {
            return 1.0;
        }
        let sum: f64 = main
            .windows(2)
            .map(|w| w[0].contour_point.distance_to(&w[1].contour_point))
            .sum();
        sum / (main.len() - 1) as f64
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
    fn recompute_tangents(&mut self) {
        let n = self.points.len();
        for i in 0..n {
            let tangent = if i + 1 < n && self.points[i].branch_id == self.points[i + 1].branch_id {
                let a = &self.points[i].contour_point;
                let b = &self.points[i + 1].contour_point;
                Vector3::new(b.x - a.x, b.y - a.y, b.z - a.z).normalize()
            } else if i > 0 && self.points[i - 1].branch_id == self.points[i].branch_id {
                self.points[i - 1].tangent
            } else {
                Vector3::zeros()
            };
            self.points[i].tangent = tangent;
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
        self.recompute_tangents();
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

    /// Remove the run-alongside-main-branch prefix from every side branch and
    /// optionally strip the inlet region from branch 0.
    ///
    /// VTP files export every branch starting from the vessel origin, so side
    /// branches share a common prefix with branch 0.  For each side branch this
    /// method trims the contiguous leading prefix whose points all lie within
    /// one mean inter-point spacing of branch 0 of at least one main-branch
    /// point.  The last point of the trimmed prefix is kept as the bifurcation
    /// junction.  Branches whose entire extent lies within that buffer are
    /// dropped completely.
    ///
    /// If `rm_start_mm > 0`, the leading points of branch 0 are also removed
    /// up to `rm_start_mm` arc-length from its first point.  This is useful
    /// when the main branch starts at the aortic inlet and the proximal region
    /// is outside the region of interest.
    pub fn cleanup_vtp_data(&mut self, rm_start_mm: f64) {
        if self.branch_start_indices.is_empty() {
            return;
        }

        let buffer = self.mean_spacing();
        let buffer_sq = buffer * buffer;

        let mut branches = self.branches_as_vecs();

        if branches.len() > 1 {
            let main_pts: Vec<(f64, f64, f64)> = branches[0]
                .iter()
                .map(|p| (p.contour_point.x, p.contour_point.y, p.contour_point.z))
                .collect();

            let close_to_main = |pt: &CenterlinePoint| -> bool {
                let (x, y, z) = (pt.contour_point.x, pt.contour_point.y, pt.contour_point.z);
                main_pts.iter().any(|&(mx, my, mz)| {
                    (x - mx).powi(2) + (y - my).powi(2) + (z - mz).powi(2) <= buffer_sq
                })
            };

            for branch in branches.iter_mut().skip(1) {
                let first_outside = branch.iter().position(|pt| !close_to_main(pt));
                match first_outside {
                    None => branch.clear(),
                    Some(0) => {}
                    Some(i) => {
                        branch.drain(..i - 1);
                    }
                }
            }

            branches.retain(|b| !b.is_empty());
        }

        if rm_start_mm > 0.0 && branches[0].len() > 1 {
            let mut arc = 0.0;
            let mut trim_idx = 0;
            for i in 1..branches[0].len() {
                arc += branches[0][i - 1]
                    .contour_point
                    .distance_to(&branches[0][i].contour_point);
                if arc <= rm_start_mm {
                    trim_idx = i;
                } else {
                    break;
                }
            }
            if trim_idx > 0 {
                branches[0].drain(..trim_idx);
            }
        }

        self.rebuild_from_branches(branches);
    }
}

#[cfg(test)]
mod centerline_tests {
    use super::*;

    fn make_multi_branch(branches: &[&[(f64, f64, f64)]]) -> Centerline {
        let mut points: Vec<CenterlinePoint> = vec![];
        let mut branch_start_indices: Vec<usize> = vec![];
        for (bid, coords) in branches.iter().enumerate() {
            branch_start_indices.push(points.len());
            for &(x, y, z) in *coords {
                let i = points.len() as u32;
                points.push(CenterlinePoint {
                    contour_point: ContourPoint {
                        frame_index: i,
                        point_index: i,
                        x,
                        y,
                        z,
                        aortic: false,
                    },
                    tangent: Vector3::zeros(),
                    radius: 0.0,
                    branch_id: bid as u32,
                });
            }
        }
        Centerline {
            points,
            branch_start_indices,
        }
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
        let cl = cl_from_coords(&[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (2.5, 0.5, 0.0),
            (2.0, 1.0, 0.0),
        ]);
        assert_eq!(cl.find_sharp_angles(0, 0.0), vec![3]);
        assert!(cl.find_sharp_angles(0, 0.8).is_empty());
        assert!(cl.find_sharp_angles(5, 0.0).is_empty());
    }

    #[test]
    fn test_split_branch_main_longer_stays() {
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
        assert_eq!(cl.points.len(), 10);
        let len0 = cl.branch_start_indices[1];
        let len1 = cl.points.len() - cl.branch_start_indices[1];
        assert_eq!(len0, 6, "longer segment must be branch 0");
        assert_eq!(len1, 4);
        assert!(cl.points.iter().enumerate().all(|(i, p)| {
            p.branch_id == if i < 6 { 0 } else { 1 } && p.contour_point.point_index == i as u32
        }));
    }

    #[test]
    fn test_split_branch_equal_length_first_is_main() {
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
        assert_eq!(cl.points.len(), 6);
        assert!(cl.points.iter().all(|p| p.branch_id == 0));
        for (i, p) in cl.points.iter().enumerate() {
            assert_eq!(p.contour_point.point_index, i as u32);
        }
    }

    #[test]
    fn test_centerline_tangents() {
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
        assert_eq!(centerline.points[0].tangent, Vector3::new(1.0, 0.0, 0.0));
        assert_eq!(centerline.points[1].tangent, Vector3::new(1.0, 0.0, 0.0));
        assert_eq!(centerline.points[2].tangent, Vector3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn test_cleanup_vtp_trims_overlap_prefix() {
        // Main: straight along x, spacing = 1.0.
        // Side: first 3 pts lie on main, then diverges by 1.5 (> 1 spacing) in y.
        let main = &[
            (0., 0., 0.),
            (1., 0., 0.),
            (2., 0., 0.),
            (3., 0., 0.),
            (4., 0., 0.),
        ];
        let side = &[
            (0., 0., 0.),
            (1., 0., 0.),
            (2., 0., 0.),
            (2., 1.5, 0.),
            (2., 3., 0.),
        ];
        let mut cl = make_multi_branch(&[main, side]);
        cl.cleanup_vtp_data(0.0);

        let branches = cl.branches_as_vecs();
        assert_eq!(branches.len(), 2, "side branch must survive");
        assert_eq!(branches[0].len(), 5, "main branch unchanged");
        // Junction (2,0,0) + 2 diverged points.
        assert_eq!(branches[1].len(), 3);
        let j = &branches[1][0].contour_point;
        assert!((j.x - 2.0).abs() < 1e-9 && j.y.abs() < 1e-9);
    }

    #[test]
    fn test_cleanup_vtp_drops_fully_overlapping_branch() {
        let main = &[(0., 0., 0.), (1., 0., 0.), (2., 0., 0.)];
        // Side branch lies entirely on main within buffer=0.5.
        let side = &[(0., 0., 0.), (1., 0., 0.)];
        let mut cl = make_multi_branch(&[main, side]);
        cl.cleanup_vtp_data(0.0);

        assert_eq!(
            cl.branch_start_indices.len(),
            1,
            "fully-overlapping branch must be dropped"
        );
    }

    #[test]
    fn test_cleanup_vtp_inlet_trim() {
        // Main: spacing = 1.0, 6 points → trim first 3 mm → keep from point 3 onwards.
        let main = &[
            (0., 0., 0.),
            (1., 0., 0.),
            (2., 0., 0.),
            (3., 0., 0.),
            (4., 0., 0.),
            (5., 0., 0.),
        ];
        let mut cl = make_multi_branch(&[main]);
        cl.cleanup_vtp_data(3.0);

        assert_eq!(cl.branch_start_indices.len(), 1);
        // arc ≤ 3.0 covers points at 0, 1, 2, 3 mm → trim_idx = 3, keep from 3 onwards
        assert_eq!(cl.points.len(), 3);
        assert!((cl.points[0].contour_point.x - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_cleanup_vtp_no_overlap_leaves_branch_intact() {
        let main = &[(0., 0., 0.), (1., 0., 0.), (2., 0., 0.)];
        // Side branch diverges from the very first point.
        let side = &[(0., 5., 0.), (0., 6., 0.), (0., 7., 0.)];
        let mut cl = make_multi_branch(&[main, side]);
        cl.cleanup_vtp_data(0.0);

        let branches = cl.branches_as_vecs();
        assert_eq!(branches.len(), 2);
        assert_eq!(branches[1].len(), 3, "no trimming when no overlap");
    }
}
