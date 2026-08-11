use super::centerline_point::CenterlinePoint;
use super::contour_point::ContourPoint;
use super::Point3D;
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

    pub fn get_by_frame(&self, frame_index: u32) -> Option<&CenterlinePoint> {
        self.points
            .iter()
            .find(|p| p.contour_point.frame_index == frame_index)
    }

    /// Finds the index of the centerline point closest to the reference point
    pub fn find_reference_cl_point_idx(&self, reference_point: &(f64, f64, f64)) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f64::INFINITY;
        for (idx, p) in self.points.iter().enumerate() {
            let dist = p.contour_point.distance_to(reference_point);
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }
        best_idx
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
        // Double BFS on the tree to find the diameter (longest path by arc length = main branch).
        let (a, _) = self.bfs_farthest(adj_map, n_points, 0);
        let (b, prev) = self.bfs_farthest(adj_map, n_points, a);
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

    /// BFS from `start` over the tree, accumulating real arc-length distance
    /// (not hop count) so that non-uniformly sampled branches don't skew
    /// which node is found to be "farthest". Returns the farthest reachable
    /// node and a predecessor array.
    fn bfs_farthest(
        &self,
        adj: &[Vec<usize>],
        n: usize,
        start: usize,
    ) -> (usize, Vec<Option<usize>>) {
        let mut dist = vec![f64::INFINITY; n];
        let mut prev: Vec<Option<usize>> = vec![None; n];
        let mut q = std::collections::VecDeque::new();
        dist[start] = 0.0;
        q.push_back(start);
        let mut farthest = start;
        while let Some(u) = q.pop_front() {
            for &v in &adj[u] {
                if dist[v].is_infinite() {
                    dist[v] = dist[u]
                        + self.points[u]
                            .contour_point
                            .distance_to(&self.points[v].contour_point);
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

    /// Return global `point_index` values (as in `ContourPoint::point_index`, i.e.
    /// indices into the flat `points` Vec) of interior points on `branch_id` where
    /// the opening angle satisfies `cos_angle > cos_threshold`.
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
            .map(|local_pos| start + local_pos)
            .collect()
    }

    /// Split `branch_id` at `point_index` (a global index into `points`, as
    /// returned by [`Centerline::find_sharp_angles`] - must fall within
    /// `branch_id`'s own range). Both resulting segments include the split
    /// point. Branches are re-sorted by descending length afterwards, so
    /// branch 0 is always the longest overall, matching the invariant
    /// established by [`Centerline::calculate_branches`].
    pub fn split_branch(&mut self, branch_id: u32, point_index: usize) {
        let idx = branch_id as usize;
        let n = self.branch_start_indices.len();
        if idx >= n {
            return;
        }
        let start = self.branch_start_indices[idx];
        let end = if idx + 1 < n {
            self.branch_start_indices[idx + 1]
        } else {
            self.points.len()
        };
        if point_index < start || point_index >= end {
            return;
        }
        let local_pos = point_index - start;

        let mut branches = self.branches_as_vecs();
        let branch = branches.remove(idx);
        if local_pos == 0 || local_pos >= branch.len().saturating_sub(1) {
            branches.insert(idx, branch);
            return;
        }

        let seg_a = branch[..=local_pos].to_vec();
        let seg_b = branch[local_pos..].to_vec();
        branches.push(seg_a);
        branches.push(seg_b);

        Self::sort_branches_by_length(&mut branches);
        self.rebuild_from_branches(branches);
    }

    /// Merge two branches into one. Endpoints are matched by minimum distance
    /// so the segments are concatenated in the correct spatial order. Branches
    /// are re-sorted by descending length afterwards, so branch 0 is always
    /// the longest overall, matching the invariant established by
    /// [`Centerline::calculate_branches`].
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

        branches.push(merged);
        Self::sort_branches_by_length(&mut branches);
        self.rebuild_from_branches(branches);
    }

    /// Sort branches by descending length, so branch 0 ends up the longest -
    /// the same invariant [`Centerline::calculate_branches`] establishes.
    /// Ties keep their relative order (stable sort).
    fn sort_branches_by_length(branches: &mut [Vec<CenterlinePoint>]) {
        branches.sort_by_key(|b| std::cmp::Reverse(b.len()));
    }

    /// Reverse branch 0 in place if its highest-z point is not already at its start,
    /// then apply the same "closer end goes first" rule to every side branch,
    /// using branch 0 (post-reversal) as the reference.
    ///
    /// Intended for a centerline with no anatomical reference to orient against
    /// (e.g. the aorta) — use [`Centerline::orient_to_reference`] instead whenever
    /// one is available. Only correct under the standard CT/DICOM convention where
    /// z increases toward the head, so the aortic root/valve is the highest-z point.
    pub fn orient_by_max_z(&mut self) {
        let n = self.branch_start_indices.len();
        if n == 0 {
            return;
        }
        let mut branches = self.branches_as_vecs();
        if Self::should_reverse_by_max_z(&branches[0]) {
            branches[0].reverse();
        }
        if let Some((first, rest)) = branches.split_first_mut() {
            for branch in rest.iter_mut() {
                if Self::should_reverse_relative_to(branch, first.as_slice()) {
                    branch.reverse();
                }
            }
        }
        self.rebuild_from_branches(branches);
    }

    /// Reverse branch 0 in place if its last point is closer to `reference`'s
    /// branch 0 than its first point is, then apply the same rule to every side
    /// branch — so every branch of `self` starts at whichever end is nearer
    /// `reference`.
    ///
    /// Any side branches `reference` has are ignored so a stray one can't skew
    /// the distance check — only `reference`'s branch 0 is ever measured
    /// against. Distance to `reference` is the minimum distance to any point of
    /// its branch 0, not a single fixed point — e.g. for a coronary centerline,
    /// `reference` would be the aorta centerline, not one ostium point.
    pub fn orient_to_reference(&mut self, reference: &Centerline) {
        let n = self.branch_start_indices.len();
        if n == 0 {
            return;
        }
        let mut branches = self.branches_as_vecs();
        let ref_branch_0 = reference.branch_0();
        if Self::should_reverse_relative_to(&branches[0], ref_branch_0) {
            branches[0].reverse();
        }
        for branch in branches.iter_mut().skip(1) {
            if Self::should_reverse_relative_to(branch, ref_branch_0) {
                branch.reverse();
            }
        }
        self.rebuild_from_branches(branches);
    }

    /// Branch 0's points, or all points if `self` has no branch structure yet.
    fn branch_0(&self) -> &[CenterlinePoint] {
        let end = self
            .branch_start_indices
            .get(1)
            .copied()
            .unwrap_or(self.points.len());
        &self.points[..end]
    }

    /// `true` if the point with the maximum z-coordinate in `points` is not at index 0.
    fn should_reverse_by_max_z(points: &[CenterlinePoint]) -> bool {
        if points.is_empty() {
            return false;
        }
        let max_z_idx = points
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
        max_z_idx != 0
    }

    /// `true` if the last point of `points` is closer to `reference` (minimum distance
    /// to any of its points) than the first point of `points` is.
    fn should_reverse_relative_to(
        points: &[CenterlinePoint],
        reference: &[CenterlinePoint],
    ) -> bool {
        if points.is_empty() || reference.is_empty() {
            return false;
        }

        let first_pt = points[0].contour_point;
        let last_pt = points.last().unwrap().contour_point;

        let dist_first = reference
            .iter()
            .map(|p| p.contour_point.distance_to(&first_pt))
            .fold(f64::INFINITY, f64::min);

        let dist_last = reference
            .iter()
            .map(|p| p.contour_point.distance_to(&last_pt))
            .fold(f64::INFINITY, f64::min);

        dist_last < dist_first
    }

    /// Remove the run-alongside-main-branch prefix duplicated by every side branch.
    ///
    /// Some centerline export formats (e.g. VTP) write every branch starting from
    /// the vessel origin, so side branches share a common prefix with branch 0.
    /// For each side branch this trims the contiguous leading prefix whose points
    /// all lie within one mean inter-point spacing of branch 0 of at least one
    /// main-branch point. The last point of the trimmed prefix is kept as the
    /// bifurcation junction. Branches whose entire extent lies within that buffer
    /// are dropped completely.
    pub fn remove_branch_overlap(&mut self) {
        if self.branch_start_indices.is_empty() {
            return;
        }

        let buffer = self.mean_spacing();
        let mut branches = self.branches_as_vecs();

        Self::remove_overlapping(&mut branches, buffer * buffer);

        self.rebuild_from_branches(branches);
    }

    /// Trim `mm` of arc length off the start of branch 0.
    ///
    /// Useful when the main branch starts at the aortic inlet and the proximal
    /// region is outside the region of interest.
    pub fn trim_start(&mut self, mm: f64) {
        if mm <= 0.0 || self.branch_start_indices.is_empty() {
            return;
        }

        let mut branches = self.branches_as_vecs();

        Self::remove_trailing_start(&mut branches, mm);

        self.rebuild_from_branches(branches);
    }

    /// Resample every branch independently to even arc-length spacing.
    ///
    /// Interior points are linearly interpolated (position and radius) between the
    /// two nearest original points; tangents are recomputed afterwards. No
    /// interpolation occurs across a bifurcation. `frame_index`/`point_index` are
    /// reassigned sequentially per branch since resampled points no longer
    /// correspond 1:1 with source frames.
    pub fn resample(&mut self, spacing_mm: f64) {
        if self.points.is_empty() || spacing_mm <= 1e-12 {
            return;
        }

        let mut branches = self.branches_as_vecs();
        for branch in branches.iter_mut() {
            *branch = Self::resample_branch(branch, spacing_mm);
        }
        self.rebuild_from_branches(branches);
    }

    /// Resample one branch's points to even arc-length spacing via linear interpolation.
    fn resample_branch(points: &[CenterlinePoint], spacing_mm: f64) -> Vec<CenterlinePoint> {
        if points.len() < 2 {
            return points.to_vec();
        }

        let mut cum = vec![0.0f64; points.len()];
        for i in 1..points.len() {
            cum[i] = cum[i - 1]
                + points[i - 1]
                    .contour_point
                    .distance_to(&points[i].contour_point);
        }
        let total = cum[points.len() - 1];
        if total < 1e-12 {
            return points.to_vec();
        }

        let mut targets = Vec::new();
        let mut s = 0.0;
        while s < total {
            targets.push(s);
            s += spacing_mm;
        }
        targets.push(total);

        let mut seg = 0usize;
        targets
            .iter()
            .enumerate()
            .map(|(sample_index, &t)| {
                while seg < points.len() - 2 && cum[seg + 1] < t {
                    seg += 1;
                }
                let (s0, s1) = (cum[seg], cum[seg + 1]);
                let frac = if (s1 - s0).abs() < 1e-12 {
                    0.0
                } else {
                    (t - s0) / (s1 - s0)
                };

                let p0 = &points[seg].contour_point;
                let p1 = &points[seg + 1].contour_point;
                let r0 = points[seg].radius;
                let r1 = points[seg + 1].radius;

                CenterlinePoint {
                    contour_point: ContourPoint {
                        frame_index: sample_index as u32,
                        point_index: sample_index as u32,
                        x: p0.x + frac * (p1.x - p0.x),
                        y: p0.y + frac * (p1.y - p0.y),
                        z: p0.z + frac * (p1.z - p0.z),
                        aortic: p0.aortic,
                    },
                    tangent: Vector3::zeros(),
                    branch_id: points[seg].branch_id,
                    radius: r0 + frac * (r1 - r0),
                }
            })
            .collect()
    }

    /// Smooth centerline positions with a Gaussian kernel (per branch) and recompute tangents.
    ///
    /// `sigma` is the half-width in number of centerline points.  A value of 1.0 is a gentle
    /// neighbourhood average; 3–5 removes noise while keeping the overall vessel path; larger
    /// values heavily round corners.  Branches are processed independently so no smoothing
    /// bleeds across the bifurcation.
    pub fn smooth(&mut self, sigma: f64) {
        if self.points.is_empty() || sigma < 1e-12 {
            return;
        }

        let n = self.points.len();
        let max_branch = self.points.iter().map(|p| p.branch_id).max().unwrap_or(0);

        let mut sx = vec![0.0f64; n];
        let mut sy = vec![0.0f64; n];
        let mut sz = vec![0.0f64; n];

        for branch_id in 0..=max_branch {
            let indices: Vec<usize> = self
                .points
                .iter()
                .enumerate()
                .filter(|(_, p)| p.branch_id == branch_id)
                .map(|(i, _)| i)
                .collect();

            if indices.is_empty() {
                continue;
            }

            // Truncate kernel at 3σ to avoid O(n²) cost on long vessels.
            let radius = (3.0 * sigma).ceil() as usize;

            for (li, &gi) in indices.iter().enumerate() {
                // Symmetric truncation: equal radius on both sides so that a
                // linear trend is preserved exactly (weighted mean of symmetric
                // neighbours always equals the centre value).
                let sym_r = li.min(radius).min(indices.len() - 1 - li);
                let j_start = li - sym_r;
                let j_end = li + sym_r + 1;
                let (mut wx, mut wy, mut wz, mut wt) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);

                for (k, &gi_j) in indices[j_start..j_end].iter().enumerate() {
                    let j = j_start + k;
                    let diff = (li as f64) - (j as f64);
                    let w = (-0.5 * diff * diff / (sigma * sigma)).exp();
                    let pt = &self.points[gi_j].contour_point;
                    wx += w * pt.x;
                    wy += w * pt.y;
                    wz += w * pt.z;
                    wt += w;
                }

                if wt > 1e-12 {
                    sx[gi] = wx / wt;
                    sy[gi] = wy / wt;
                    sz[gi] = wz / wt;
                } else {
                    let pt = &self.points[gi].contour_point;
                    sx[gi] = pt.x;
                    sy[gi] = pt.y;
                    sz[gi] = pt.z;
                }
            }
        }

        for (i, p) in self.points.iter_mut().enumerate() {
            p.contour_point.x = sx[i];
            p.contour_point.y = sy[i];
            p.contour_point.z = sz[i];
        }

        self.recompute_tangents();
    }

    /// Trims the leading overlap each branch shares with the branches already cleaned.
    ///
    /// VTP centerline branch data often starts by duplicating a stretch of an earlier
    /// branch before diverging at the true bifurcation; this drops that duplicated
    /// prefix. Branches are ordered longest-first with branch 0 as the main vessel, so
    /// a smaller side branch may actually bifurcate off another (already-trimmed) side
    /// branch rather than directly off the main vessel. Processing branches in order
    /// and growing the reference point set as each branch is cleaned handles that case
    /// too, instead of only ever comparing against branch 0.
    fn remove_overlapping(branches: &mut Vec<Vec<CenterlinePoint>>, buffer_sq: f64) {
        if branches.len() <= 1 {
            return;
        }

        let mut known_pts: Vec<(f64, f64, f64)> = branches[0]
            .iter()
            .map(|p| (p.contour_point.x, p.contour_point.y, p.contour_point.z))
            .collect();

        for branch in branches.iter_mut().skip(1) {
            let close_to_known = |pt: &CenterlinePoint| -> bool {
                let (x, y, z) = (pt.contour_point.x, pt.contour_point.y, pt.contour_point.z);
                known_pts.iter().any(|&(mx, my, mz)| {
                    // avoid distance_to's sqrt in this O(branch_points * known_points) check
                    (x - mx).powi(2) + (y - my).powi(2) + (z - mz).powi(2) <= buffer_sq
                })
            };

            let first_outside = branch.iter().position(|pt| !close_to_known(pt));
            match first_outside {
                None => branch.clear(),
                Some(0) => {}
                Some(j) => {
                    branch.drain(..j - 1);
                }
            }

            known_pts.extend(
                branch
                    .iter()
                    .map(|p| (p.contour_point.x, p.contour_point.y, p.contour_point.z)),
            );
        }

        branches.retain(|b| !b.is_empty());
    }

    /// Trims points off the start of the main branch (branch 0) until `rm_start_mm` of
    /// arc length has been removed.
    fn remove_trailing_start(branches: &mut [Vec<CenterlinePoint>], rm_start_mm: f64) {
        if rm_start_mm <= 0.0 || branches[0].len() <= 1 {
            return;
        }

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
    fn test_find_sharp_angles_returns_point_index_not_local_pos() {
        // Side branch (branch_id=1) starts at global point_index 3, after the 3-point
        // main branch. The V-shape's sharp corner sits at local position 3 within the
        // side branch, so find_sharp_angles must report global point_index 6 (3 + 3),
        // not local position 3.
        let main = &[(0., 0., 0.), (0., 0., 1.), (0., 0., 2.)];
        let side = &[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (2.5, 0.5, 0.0),
            (2.0, 1.0, 0.0),
        ];
        let cl = make_multi_branch(&[main, side]);
        assert_eq!(cl.find_sharp_angles(1, 0.0), vec![6]);
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
    fn test_split_branch_uses_point_index_not_local_pos() {
        // Side branch (branch_id=1) starts at global point_index 5, after the 5-point
        // main branch. Splitting at point_index=7 must land at local position 2 within
        // the side branch, not local position 7 (out of range for a 5-point branch).
        let main = &[
            (0., 0., 0.),
            (1., 0., 0.),
            (2., 0., 0.),
            (3., 0., 0.),
            (4., 0., 0.),
        ];
        let side = &[
            (10., 0., 0.),
            (11., 0., 0.),
            (12., 0., 0.),
            (13., 0., 0.),
            (14., 0., 0.),
        ];
        let mut cl = make_multi_branch(&[main, side]);
        cl.split_branch(1, 7);

        assert_eq!(cl.branch_start_indices.len(), 3);
        let branches = cl.branches_as_vecs();
        assert_eq!(
            branches[0].len(),
            5,
            "main branch unaffected, still longest"
        );
        assert_eq!(branches[1].len(), 3);
        assert_eq!(branches[2].len(), 3);
        assert_eq!(branches[1][0].contour_point.x, 10.0);
        assert_eq!(branches[2][0].contour_point.x, 12.0);
    }

    #[test]
    fn test_split_branch_resorts_by_length_among_side_branches() {
        // main(10) > side1(6, split into 4+3) > side2(2). After the split, both new
        // pieces must sort ahead of the untouched, shorter side2 - not just slot into
        // side1's old position while side2 stays where it was.
        let main: Vec<(f64, f64, f64)> = (0..10).map(|i| (i as f64, 0., 0.)).collect();
        let side1: Vec<(f64, f64, f64)> = (0..6).map(|i| (i as f64, 1., 0.)).collect();
        let side2: Vec<(f64, f64, f64)> = (0..2).map(|i| (i as f64, 2., 0.)).collect();
        let mut cl = make_multi_branch(&[main.as_slice(), side1.as_slice(), side2.as_slice()]);

        // side1 (branch_id=1) starts at point_index 10; split at local position 3 -> point_index 13.
        cl.split_branch(1, 13);

        let branches = cl.branches_as_vecs();
        assert_eq!(branches.len(), 4);
        assert_eq!(branches[0].len(), 10, "main unaffected, stays longest");
        assert_eq!(branches[1].len(), 4, "larger new piece from the split");
        assert_eq!(branches[2].len(), 3, "smaller new piece from the split");
        assert_eq!(
            branches[3].len(),
            2,
            "side2 sorted last: shorter than both new pieces"
        );
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
    fn test_merge_branches_promotes_to_main_when_result_is_longest() {
        // main(5) < side1(4) + side2(4) merged = 8. Even though neither merged branch
        // was originally the main vessel, the merged result is now the longest branch
        // and must become the new branch 0.
        let main: Vec<(f64, f64, f64)> = (0..5).map(|i| (i as f64, 0., 0.)).collect();
        let side1 = &[(0., 1., 0.), (1., 1., 0.), (2., 1., 0.), (3., 1., 0.)];
        let side2 = &[(3., 1., 0.), (4., 1., 0.), (5., 1., 0.), (6., 1., 0.)];
        let mut cl = make_multi_branch(&[main.as_slice(), side1, side2]);

        cl.merge_branches(1, 2);

        assert_eq!(cl.branch_start_indices.len(), 2);
        let branches = cl.branches_as_vecs();
        assert_eq!(
            branches[0].len(),
            8,
            "merged side branches are now the longest, must be branch 0"
        );
        assert_eq!(branches[1].len(), 5, "old main demoted to a side branch");
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
    fn test_remove_branch_overlap_trims_prefix() {
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
        cl.remove_branch_overlap();

        let branches = cl.branches_as_vecs();
        assert_eq!(branches.len(), 2, "side branch must survive");
        assert_eq!(branches[0].len(), 5, "main branch unchanged");
        // Junction (2,0,0) + 2 diverged points.
        assert_eq!(branches[1].len(), 3);
        let j = &branches[1][0].contour_point;
        assert!((j.x - 2.0).abs() < 1e-9 && j.y.abs() < 1e-9);
    }

    #[test]
    fn test_remove_branch_overlap_drops_fully_overlapping_branch() {
        let main = &[(0., 0., 0.), (1., 0., 0.), (2., 0., 0.)];
        // Side branch lies entirely on main within buffer=0.5.
        let side = &[(0., 0., 0.), (1., 0., 0.)];
        let mut cl = make_multi_branch(&[main, side]);
        cl.remove_branch_overlap();

        assert_eq!(
            cl.branch_start_indices.len(),
            1,
            "fully-overlapping branch must be dropped"
        );
    }

    #[test]
    fn test_trim_start_removes_inlet() {
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
        cl.trim_start(3.0);

        assert_eq!(cl.branch_start_indices.len(), 1);
        // arc ≤ 3.0 covers points at 0, 1, 2, 3 mm → trim_idx = 3, keep from 3 onwards
        assert_eq!(cl.points.len(), 3);
        assert!((cl.points[0].contour_point.x - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_remove_branch_overlap_no_overlap_leaves_branch_intact() {
        let main = &[(0., 0., 0.), (1., 0., 0.), (2., 0., 0.)];
        // Side branch diverges from the very first point.
        let side = &[(0., 5., 0.), (0., 6., 0.), (0., 7., 0.)];
        let mut cl = make_multi_branch(&[main, side]);
        cl.remove_branch_overlap();

        let branches = cl.branches_as_vecs();
        assert_eq!(branches.len(), 2);
        assert_eq!(branches[1].len(), 3, "no trimming when no overlap");
    }

    #[test]
    fn straight_line_smooth_is_unchanged() {
        // A perfectly straight line should not move after smoothing.
        let pts: Vec<(f64, f64, f64)> = (0..20).map(|i| (i as f64, 0.0, 0.0)).collect();
        let mut cl = cl_from_coords(&pts);
        let original = cl.clone();
        cl.smooth(3.0);

        for (orig, sm) in original.points.iter().zip(cl.points.iter()) {
            let dx = (orig.contour_point.x - sm.contour_point.x).abs();
            let dy = (orig.contour_point.y - sm.contour_point.y).abs();
            let dz = (orig.contour_point.z - sm.contour_point.z).abs();
            assert!(
                dx < 1e-10 && dy < 1e-10 && dz < 1e-10,
                "straight line moved"
            );
        }
    }

    #[test]
    fn smooth_damps_spike() {
        // Insert a sharp lateral spike at position 7 in an otherwise straight line.
        let mut pts: Vec<(f64, f64, f64)> = (0..15).map(|i| (i as f64, 0.0, 0.0)).collect();
        pts[7] = (7.0, 5.0, 0.0);
        let mut cl = cl_from_coords(&pts);
        cl.smooth(2.0);

        let spike_y = cl.points[7].contour_point.y;
        assert!(spike_y < 5.0, "spike should be damped, got y = {spike_y}");
        assert!(spike_y > 0.0, "spike should not be fully erased");
    }

    #[test]
    fn smooth_produces_unit_tangents() {
        let mut pts: Vec<(f64, f64, f64)> = (0..20).map(|i| (i as f64, 0.0, 0.0)).collect();
        pts[10] = (10.0, 3.0, 0.0);
        let mut cl = cl_from_coords(&pts);
        cl.smooth(2.0);

        for p in &cl.points {
            let len = p.tangent.norm();
            assert!(
                (len - 1.0).abs() < 1e-10 || len < 1e-12,
                "tangent not unit: {len}"
            );
        }
    }

    #[test]
    fn smooth_sigma_zero_is_noop() {
        let pts: Vec<(f64, f64, f64)> = (0..10).map(|i| (i as f64, 0.0, 0.0)).collect();
        let mut cl = cl_from_coords(&pts);
        let original = cl.clone();
        cl.smooth(0.0);
        assert_eq!(cl, original);
    }

    #[test]
    fn test_resample_produces_even_spacing() {
        let mut cl = cl_from_coords(&[(0., 0., 0.), (10., 0., 0.)]);
        cl.resample(2.5);

        assert_eq!(cl.points.len(), 5);
        for (i, p) in cl.points.iter().enumerate() {
            assert!((p.contour_point.x - i as f64 * 2.5).abs() < 1e-9);
        }
        assert!((cl.points.last().unwrap().contour_point.x - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_resample_does_not_cross_branch_boundary() {
        let main = &[(0., 0., 0.), (10., 0., 0.)];
        let side = &[(10., 0., 0.), (10., 5., 0.)];
        let mut cl = make_multi_branch(&[main, side]);
        cl.resample(2.0);

        assert_eq!(cl.branch_start_indices.len(), 2);
        let branches = cl.branches_as_vecs();
        assert!(branches[0].iter().all(|p| p.branch_id == 0));
        assert!(branches[1].iter().all(|p| p.branch_id == 1));
        assert!((branches[1][0].contour_point.y).abs() < 1e-9);
    }

    #[test]
    fn test_orient_by_max_z_reverses_branch_0_only() {
        // Highest z (2.0) is at the end of branch 0, must end up at its start;
        // the side branch must be left completely untouched.
        let main = &[(0., 0., 0.), (0., 0., 1.), (0., 0., 2.)];
        let side = &[(0., 0., 2.), (5., 0., 2.)];
        let mut cl = make_multi_branch(&[main, side]);
        cl.orient_by_max_z();

        let branches = cl.branches_as_vecs();
        assert_eq!(branches[0][0].contour_point.z, 2.0);
        assert_eq!(branches[0][2].contour_point.z, 0.0);
        assert_eq!(branches[1][0].contour_point.x, 0.0);
        assert_eq!(branches[1][1].contour_point.x, 5.0);
        assert!(cl
            .points
            .iter()
            .enumerate()
            .all(|(i, p)| p.contour_point.point_index == i as u32));
    }

    #[test]
    fn test_orient_by_max_z_leaves_already_correct_untouched() {
        let mut cl = cl_from_coords(&[(0., 0., 2.), (0., 0., 1.), (0., 0., 0.)]);
        cl.orient_by_max_z();
        assert_eq!(cl.points[0].contour_point.z, 2.0);
    }

    #[test]
    fn test_orient_by_max_z_reorients_side_branches() {
        // Side branch's last point ends up nearer branch 0's (post-reversal)
        // start than its first point is, so it must be reversed too.
        let main = &[(0., 0., 0.), (0., 0., 1.), (0., 0., 2.)];
        let side = &[(5., 0., 2.), (0., 0., 2.)];
        let mut cl = make_multi_branch(&[main, side]);
        cl.orient_by_max_z();

        let branches = cl.branches_as_vecs();
        assert_eq!(branches[0][0].contour_point.z, 2.0);
        assert_eq!(branches[1][0].contour_point.x, 0.0);
        assert_eq!(branches[1][1].contour_point.x, 5.0);
    }

    #[test]
    fn test_orient_to_reference_reverses_branch_0_only() {
        // Branch 0's last point (10,0,0) is close to `reference`; its first (0,0,0) is not.
        let main = &[(0., 0., 0.), (5., 0., 0.), (10., 0., 0.)];
        let side = &[(0., 0., 0.), (0., 5., 0.)];
        let mut cl = make_multi_branch(&[main, side]);
        let reference = cl_from_coords(&[(10., 1., 0.), (20., 1., 0.)]);
        cl.orient_to_reference(&reference);

        let branches = cl.branches_as_vecs();
        assert_eq!(branches[0][0].contour_point.x, 10.0);
        assert_eq!(branches[0][2].contour_point.x, 0.0);
        assert_eq!(branches[1][0].contour_point.x, 0.0);
        assert_eq!(branches[1][1].contour_point.y, 5.0);
    }

    #[test]
    fn test_orient_to_reference_leaves_already_correct_untouched() {
        // `cl`'s first point (10,0,0) is already closest to `reference`.
        let mut cl = cl_from_coords(&[(10., 0., 0.), (5., 0., 0.), (0., 0., 0.)]);
        let reference = cl_from_coords(&[(10., 1., 0.), (20., 1., 0.)]);
        cl.orient_to_reference(&reference);
        assert_eq!(cl.points[0].contour_point.x, 10.0);
    }

    #[test]
    fn test_orient_to_reference_ignores_references_side_branches() {
        // `cl`'s first point (0,0,0) is already closest to `reference`'s branch 0.
        // `reference` also has a side branch sitting right next to `cl`'s last
        // point (10,0,0) — that must NOT cause a reversal.
        let mut cl = cl_from_coords(&[(0., 0., 0.), (5., 0., 0.), (10., 0., 0.)]);
        let ref_main = &[(0., 1., 0.), (1., 1., 0.)];
        let ref_side = &[(10., 1., 0.), (11., 1., 0.)];
        let reference = make_multi_branch(&[ref_main, ref_side]);
        cl.orient_to_reference(&reference);
        assert_eq!(cl.points[0].contour_point.x, 0.0);
    }

    #[test]
    fn test_orient_to_reference_reorients_side_branches_toward_reference() {
        // Own branch 0 gives no preference for the side branch (it's equidistant
        // from both its endpoints), but `reference` is much closer to the side
        // branch's last point — the side branch must be reversed to match
        // `reference`, not left alone based on `self`'s own branch 0.
        let main = &[(0., 0., 0.), (5., 0., 0.), (10., 0., 0.)];
        let side = &[(10., 5., 0.), (0., 5., 0.)];
        let mut cl = make_multi_branch(&[main, side]);
        let reference = cl_from_coords(&[(0., 1., 0.), (-10., 1., 0.)]);
        cl.orient_to_reference(&reference);

        let branches = cl.branches_as_vecs();
        assert_eq!(branches[0][0].contour_point.x, 0.0, "branch 0 unchanged");
        assert_eq!(
            branches[1][0].contour_point.x, 0.0,
            "side branch reversed toward reference"
        );
        assert_eq!(branches[1][1].contour_point.x, 10.0);
    }

    #[test]
    fn test_orient_to_reference_side_branches_ignore_references_side_branches() {
        // `cl`'s side branch's first point (0,0,0) is already closest to
        // `reference`'s branch 0. `reference` also has a side branch sitting
        // right next to `cl`'s side branch's last point (10,5,0) — that must
        // NOT cause a reversal.
        let main = &[(0., 0., 0.), (5., 0., 0.), (10., 0., 0.)];
        let side = &[(0., 0., 0.), (10., 5., 0.)];
        let mut cl = make_multi_branch(&[main, side]);
        let ref_main = &[(0., 1., 0.), (1., 1., 0.)];
        let ref_side = &[(10., 5., 0.), (11., 5., 0.)];
        let reference = make_multi_branch(&[ref_main, ref_side]);
        cl.orient_to_reference(&reference);

        let branches = cl.branches_as_vecs();
        assert_eq!(branches[1][0].contour_point.x, 0.0);
    }
}
