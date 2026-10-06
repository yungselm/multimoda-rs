use crate::types::native::{Centerline, CenterlinePoint, Frame};
use nalgebra::Point3;
use rayon::prelude::*;
use rstar::primitives::GeomWithData;
use std::collections::HashSet;
use std::ops::Range;

type Coords3 = (f64, f64, f64);
/// (proximal, distal, between) point lists.
type RegionSplit = (Vec<Coords3>, Vec<Coords3>, Vec<Coords3>);

pub fn centerline_based_wall_diameter_optimization(
    centerline: &Centerline,
    ref_point_coronary: &Coords3,
    aortic_points: &[Coords3],
) -> f64 {
    if centerline.points.is_empty() || aortic_points.is_empty() {
        return 0.0;
    }

    let closest_cl = match centerline.points.iter().min_by(|a, b| {
        let dist_a = super::calculate_squared_distance(*a, ref_point_coronary);
        let dist_b = super::calculate_squared_distance(*b, ref_point_coronary);
        dist_a
            .partial_cmp(&dist_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        Some(pt) => pt,
        None => return 0.0,
    };

    let closest_aortic: &Coords3 = match aortic_points.iter().min_by(|a, b| {
        let dist_a = super::calculate_squared_distance(*a, ref_point_coronary);
        let dist_b = super::calculate_squared_distance(*b, ref_point_coronary);
        dist_a
            .partial_cmp(&dist_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        Some(pt) => pt,
        None => return 0.0,
    };

    let ref_point = Point3::new(
        ref_point_coronary.0,
        ref_point_coronary.1,
        ref_point_coronary.2,
    );
    let closest_cl_point = Point3::new(
        closest_cl.contour_point.x,
        closest_cl.contour_point.y,
        closest_cl.contour_point.z,
    );
    let closest_aortic_point = Point3::new(closest_aortic.0, closest_aortic.1, closest_aortic.2);

    // vector centerline point to coronary point
    let vector = ref_point - closest_cl_point;
    let Some(unit) = vector.try_normalize(0.0) else {
        return 0.0;
    };

    // project (ref_point_coronary - closest_aortic) onto unit direction
    // minimizes ||closest_aortic + t*unit - ref_point_coronary||², constrained t >= 0
    let to_ref = ref_point - closest_aortic_point;
    let t = to_ref.dot(&unit);

    t.max(0.0)
}

pub fn centerline_based_aortic_diameter_optimization(
    intramural_points: &[Coords3],
    reference_points: &[Coords3],
    centerline: &Centerline,
) -> f64 {
    let start = -2.0f64;
    let end = 2.0f64;
    let step = 0.1f64;
    let steps = ((end - start) / step).round() as i32; // (4.0 / 0.1) => 40

    let mut min_dist = f64::MAX;
    let mut scaling_best = f64::MAX;

    for i in 0..=steps {
        let x = start + i as f64 * step;
        let temp_points = centerline_based_diameter_morphing(centerline, intramural_points, x);
        let dist = symmetric_nn_distance(reference_points, &temp_points);
        if dist < min_dist {
            min_dist = dist;
            scaling_best = x;
        }
    }
    scaling_best
}

pub fn centerline_based_diameter_optimization(
    anomalous_points: &[Coords3],
    n_proximal: usize,
    n_distal: usize,
    centerline: &Centerline,
    proximal_reference: &[Coords3],
    distal_reference: &[Coords3],
) -> (f64, f64) {
    let (proximal_points, anomalous_points_new) =
        find_region_points(anomalous_points, proximal_reference, n_proximal);
    let (distal_points, _) = find_region_points(&anomalous_points_new, distal_reference, n_distal);

    let start = -2.0f64;
    let end = 2.0f64;
    let step = 0.1f64;
    let steps = ((end - start) / step).round() as i32; // (4.0 / 0.1) => 40

    let mut min_dist_proximal = f64::MAX;
    let mut min_dist_distal = f64::MAX;
    let mut prox_scaling_best = f64::MAX;
    let mut dist_scaling_best = f64::MAX;

    for i in 0..=steps {
        let x = start + i as f64 * step;
        let temp_points = centerline_based_diameter_morphing(centerline, &proximal_points, x);
        let dist = symmetric_nn_distance(proximal_reference, &temp_points);
        if dist < min_dist_proximal {
            min_dist_proximal = dist;
            prox_scaling_best = x;
        }
    }
    for i in 0..=steps {
        let x = start + i as f64 * step;
        let temp_points = centerline_based_diameter_morphing(centerline, &distal_points, x);
        let dist = symmetric_nn_distance(distal_reference, &temp_points);
        if dist < min_dist_distal {
            min_dist_distal = dist;
            dist_scaling_best = x;
        }
    }
    (prox_scaling_best, dist_scaling_best)
}

fn find_region_points(
    anomalous_points: &[Coords3],
    reference_points: &[Coords3],
    n_points: usize,
) -> (Vec<Coords3>, Vec<Coords3>) {
    if anomalous_points.is_empty() || reference_points.is_empty() || n_points == 0 {
        return (Vec::new(), anomalous_points.to_vec());
    }

    let mut indexed_dists: Vec<(usize, f64)> = anomalous_points
        .iter()
        .enumerate()
        .map(|(i, a_pt)| {
            let min_sq = reference_points
                .iter()
                .map(|r_pt| super::calculate_squared_distance(a_pt, r_pt))
                .fold(f64::INFINITY, |m, v| if v < m { v } else { m });
            (i, min_sq)
        })
        .collect();

    indexed_dists.sort_by(|(i1, d1), (i2, d2)| {
        d1.partial_cmp(d2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| i1.cmp(i2))
    });

    let take = n_points.min(anomalous_points.len());
    let selected_slice = &indexed_dists[..take];

    let selected_indices: HashSet<usize> = selected_slice.iter().map(|(i, _)| *i).collect();

    let selected_points: Vec<Coords3> = selected_slice
        .iter()
        .map(|(i, _)| anomalous_points[*i])
        .collect();

    let remaining_points: Vec<Coords3> = anomalous_points
        .iter()
        .enumerate()
        .filter_map(|(i, pt)| {
            if selected_indices.contains(&i) {
                None
            } else {
                Some(*pt)
            }
        })
        .collect();

    (selected_points, remaining_points)
}

/// Symmetric average nearest-neighbor distance between two point sets.
/// Returns sqrt of average squared distance for readability (i.e., RMS of nearest neighbor distances).
/// If either set is empty returns f64::INFINITY.
fn symmetric_nn_distance(a: &[Coords3], b: &[Coords3]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::INFINITY;
    }

    let sum_a_to_b: f64 = a
        .par_iter()
        .map(|pa| {
            b.iter()
                .map(|pb| super::calculate_squared_distance(pa, pb))
                .fold(f64::INFINITY, |m, v| if v < m { v } else { m })
        })
        .sum();

    let avg_a_to_b = sum_a_to_b / (a.len() as f64);

    let sum_b_to_a: f64 = b
        .par_iter()
        .map(|pb| {
            a.iter()
                .map(|pa| super::calculate_squared_distance(pb, pa))
                .fold(f64::INFINITY, |m, v| if v < m { v } else { m })
        })
        .sum();

    let avg_b_to_a = sum_b_to_a / (b.len() as f64);

    ((avg_a_to_b + avg_b_to_a) / 2.0).sqrt()
}

pub fn centerline_based_diameter_morphing(
    centerline: &Centerline,
    points: &[Coords3],
    diameter_adjustment_mm: f64,
) -> Vec<Coords3> {
    points
        .par_iter()
        .map(|point| {
            let closest_cl_point = find_closest_centerline_point_optimized(centerline, *point);
            let p = Point3::new(point.0, point.1, point.2);
            let cl_point = Point3::new(
                closest_cl_point.contour_point.x,
                closest_cl_point.contour_point.y,
                closest_cl_point.contour_point.z,
            );

            match (p - cl_point).try_normalize(0.0) {
                Some(unit) => {
                    let moved = p + unit * diameter_adjustment_mm;
                    (moved.x, moved.y, moved.z)
                }
                None => *point,
            }
        })
        .collect()
}

fn find_closest_centerline_point_optimized(
    centerline: &Centerline,
    point: Coords3,
) -> &CenterlinePoint {
    let mut min_distance_squared = f64::MAX;
    let mut closest_point = &centerline.points[0];

    for centerline_point in &centerline.points {
        let distance_squared = super::calculate_squared_distance(&point, centerline_point);
        if distance_squared < min_distance_squared {
            min_distance_squared = distance_squared;
            closest_point = centerline_point;
        }
    }

    closest_point
}

/// Split `points` into (proximal, distal, between) by where they sit along the branch of
/// `centerline` the `frames` lie on (the pullback branch), relative to the segment the
/// frames cover.
///
/// The pullback branch is the branch most frame centroids are nearest to; each frame is
/// snapped to it, and the lowest and highest positions bound the segment. Each point is
/// placed at its nearest centerline point and walked up the branch tree to the pullback
/// branch: a point on a branch that leaves the pullback branch takes the position where
/// that branch joins it. Positions below the segment are proximal, above it distal, and
/// inside it "between". Points outside the pullback branch's subtree (its parent vessel
/// and sibling branches) are distal.
///
/// Positions count from the end of a branch that attaches to its parent; branch 0 counts
/// from its first point, which must be the ostium. A side branch's parent is the earlier
/// branch nearest to either of its ends, the same order `remove_branch_overlap` trims
/// branches in. Pass the whole vessel centerline: with a single branch there is no tree
/// to walk, and every point is placed on that branch.
///
/// Positions are point indices, not `frame_index` values: `resample` restarts
/// `frame_index` at 0 on every branch, so it is not unique across branches.
pub fn find_points_by_cl_region_rs(
    centerline: &Centerline,
    frames: &[Frame],
    points: &[Coords3],
) -> Result<RegionSplit, String> {
    if frames.is_empty() {
        return Err("find_points_by_cl_region requires at least one frame".to_string());
    }
    if centerline.points.is_empty() {
        return Err("find_points_by_cl_region requires a non-empty centerline".to_string());
    }

    let n_cl = centerline.points.len();
    let mut branch_starts: Vec<usize> = centerline
        .branch_start_indices
        .iter()
        .copied()
        .filter(|&s| s < n_cl)
        .collect();
    branch_starts.push(0);
    branch_starts.sort_unstable();
    branch_starts.dedup();
    let ranges: Vec<Range<usize>> = branch_starts
        .iter()
        .enumerate()
        .map(|(b, &start)| start..branch_starts.get(b + 1).copied().unwrap_or(n_cl))
        .collect();

    let coords = |g: usize| {
        let p = &centerline.points[g].contour_point;
        [p.x, p.y, p.z]
    };
    let tagged_tree = |indices: Vec<usize>| -> CenterlineTree {
        rstar::RTree::bulk_load(
            indices
                .into_iter()
                .map(|g| GeomWithData::new(coords(g), g))
                .collect(),
        )
    };
    let branch_trees: Vec<CenterlineTree> = ranges
        .iter()
        .map(|r| tagged_tree(r.clone().collect()))
        .collect();
    let links = link_branches(&ranges, &branch_trees, coords);
    let branch_of = |g: usize| links.partition_point(|l| l.range.start <= g) - 1;

    // A side branch's attachment point duplicates a point on its parent; leaving it out
    // keeps the parent's wall around the junction on the parent.
    let attach_points: HashSet<usize> = links.iter().skip(1).map(|l| l.attach_index()).collect();
    let all_tree = tagged_tree((0..n_cl).filter(|g| !attach_points.contains(g)).collect());

    let frame_coords: Vec<[f64; 3]> = frames
        .iter()
        .map(|f| [f.centroid.0, f.centroid.1, f.centroid.2])
        .collect();
    let mut votes = vec![0usize; links.len()];
    for &c in &frame_coords {
        if let Some(n) = all_tree.nearest_neighbor(c) {
            votes[branch_of(n.data)] += 1;
        }
    }
    // Ties go to the lower branch index.
    let pullback = (0..links.len())
        .max_by_key(|&b| (votes[b], std::cmp::Reverse(b)))
        .expect("centerline has at least one branch");
    let (seg_start, seg_end) = frame_coords
        .iter()
        .filter_map(|&c| branch_trees[pullback].nearest_neighbor(c))
        .map(|n| links[pullback].position(n.data))
        .fold((usize::MAX, 0), |(lo, hi), p| (lo.min(p), hi.max(p)));

    let mut proximal_points: Vec<Coords3> = Vec::new();
    let mut distal_points: Vec<Coords3> = Vec::new();
    let mut points_between: Vec<Coords3> = Vec::new();

    for point in points {
        let g = all_tree
            .nearest_neighbor([point.0, point.1, point.2])
            .map_or(0, |n| n.data);
        let mut branch = branch_of(g);
        let mut pos = links[branch].position(g);
        // Parents always have a lower branch index, so this terminates.
        let pos_on_pullback = loop {
            if branch == pullback {
                break Some(pos);
            }
            match links[branch].parent {
                Some((parent, parent_pos)) => {
                    branch = parent;
                    pos = parent_pos;
                }
                None => break None,
            }
        };

        match pos_on_pullback {
            Some(p) if p < seg_start => proximal_points.push(*point),
            Some(p) if p <= seg_end => points_between.push(*point),
            _ => distal_points.push(*point),
        }
    }

    let (proximal_points, points_between) =
        clean_up_non_section_points(proximal_points, points_between, 1.0, 0.6);
    let (distal_points, points_between) =
        clean_up_non_section_points(distal_points, points_between, 1.0, 0.6);
    Ok((proximal_points, distal_points, points_between))
}

type CenterlineTree = rstar::RTree<GeomWithData<[f64; 3], usize>>;

/// One branch of a centerline tree: its slice of `centerline.points`, which end attaches
/// to its parent, and where on the parent it attaches.
struct BranchLink {
    range: Range<usize>,
    /// The branch attaches to its parent at its last point instead of its first.
    reversed: bool,
    /// Parent branch and the position on it where this branch attaches; `None` for
    /// branch 0.
    parent: Option<(usize, usize)>,
}

impl BranchLink {
    /// Position of centerline index `g` along this branch, counted from the end that
    /// attaches to the parent.
    fn position(&self, g: usize) -> usize {
        if self.reversed {
            self.range.end - 1 - g
        } else {
            g - self.range.start
        }
    }

    fn attach_index(&self) -> usize {
        if self.reversed {
            self.range.end - 1
        } else {
            self.range.start
        }
    }
}

/// Link every side branch to the earlier branch nearest to either of its ends.
fn link_branches(
    ranges: &[Range<usize>],
    trees: &[CenterlineTree],
    coords: impl Fn(usize) -> [f64; 3],
) -> Vec<BranchLink> {
    let mut links: Vec<BranchLink> = Vec::with_capacity(ranges.len());
    for (b, range) in ranges.iter().enumerate() {
        if b == 0 {
            links.push(BranchLink {
                range: range.clone(),
                reversed: false,
                parent: None,
            });
            continue;
        }
        let nearest_earlier = |g: usize| {
            trees[..b]
                .iter()
                .filter_map(|t| t.nearest_neighbor_with_distance_2(coords(g)))
                .min_by(|x, y| x.1.total_cmp(&y.1))
                .map(|(n, d2)| (n.data, d2))
                .expect("branch 0 is non-empty")
        };
        let first = nearest_earlier(range.start);
        let last = nearest_earlier(range.end - 1);
        let (reversed, (joint, _)) = if last.1 < first.1 {
            (true, last)
        } else {
            (false, first)
        };
        let parent = links.partition_point(|l| l.range.start <= joint) - 1;
        let parent_pos = links[parent].position(joint);
        links.push(BranchLink {
            range: range.clone(),
            reversed,
            parent: Some((parent, parent_pos)),
        });
    }
    links
}

pub fn clean_up_non_section_points(
    points_to_cleanup: Vec<Coords3>,
    reference_points: Vec<Coords3>,
    neighborhood_radius: f64,
    min_neigbor_ratio: f64,
) -> (Vec<Coords3>, Vec<Coords3>) {
    let neighborhood_radius_sq = neighborhood_radius * neighborhood_radius;

    let mut cleaned_points = Vec::new();
    let mut reassigned_points = reference_points.clone();

    if points_to_cleanup.is_empty() {
        return (cleaned_points, reassigned_points);
    }

    // R-trees over both point sets: each point's ref/self neighbor counts then
    // cost O(log n + k) instead of a full linear scan, so the whole function is
    // O((P+R) log(P+R)) instead of O(P * (P+R)).
    let ref_tree = rstar::RTree::bulk_load(
        reference_points
            .iter()
            .map(|p| [p.0, p.1, p.2])
            .collect::<Vec<_>>(),
    );
    let cleanup_tree = rstar::RTree::bulk_load(
        points_to_cleanup
            .iter()
            .map(|p| [p.0, p.1, p.2])
            .collect::<Vec<_>>(),
    );

    for point in points_to_cleanup.iter() {
        let query = [point.0, point.1, point.2];

        let ref_neighbors = ref_tree
            .locate_within_distance(query, neighborhood_radius_sq)
            .count();
        // The point itself is always within its own radius, so subtract 1 to
        // match the original "skip self" behavior (duplicate points at the same
        // coordinate still count as neighbors, exactly one instance is excluded).
        let self_neighbors = cleanup_tree
            .locate_within_distance(query, neighborhood_radius_sq)
            .count()
            .saturating_sub(1);
        let total_neighbors = ref_neighbors + self_neighbors;

        // Decision logic: if most neighbors are reference points, reassign
        if total_neighbors > 0 {
            let ref_ratio = ref_neighbors as f64 / total_neighbors as f64;
            if ref_ratio >= min_neigbor_ratio {
                // Reassign to reference_points (anomalous)
                reassigned_points.push(*point);
            } else {
                // Keep in cleaned_points (proximal/distal)
                cleaned_points.push(*point);
            }
        } else {
            // If no neighbors in range, keep original classification
            cleaned_points.push(*point);
        }
    }

    (cleaned_points, reassigned_points)
}

#[cfg(test)]
mod tests {
    use crate::types::native::{CenterlinePoint, Contour, ContourPoint};
    use nalgebra::Vector3;

    use super::*;

    #[test]
    fn test_centerline_based_diameter_morphing() {
        let centerline = Centerline {
            points: vec![
                CenterlinePoint {
                    contour_point: ContourPoint {
                        frame_index: 0,
                        point_index: 0,
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                        aortic: false,
                    },
                    tangent: Vector3::new(1.0, 0.0, 0.0),
                    branch_id: 0,
                    radius: 0.0,
                },
                CenterlinePoint {
                    contour_point: ContourPoint {
                        frame_index: 1,
                        point_index: 1,
                        x: 1.0,
                        y: 0.0,
                        z: 0.0,
                        aortic: false,
                    },
                    tangent: Vector3::new(1.0, 0.0, 0.0),
                    branch_id: 0,
                    radius: 0.0,
                },
            ],
            branch_start_indices: vec![0],
        };

        let points = vec![
            (1.0, 1.0, 0.0), // Point at (1,1,0) - closest to centerline point (1,0,0)
        ];

        let result = centerline_based_diameter_morphing(&centerline, &points, 1.0);

        // The point should move from (1,1,0) to (1,2,0) - same direction but 1 unit further
        assert_eq!(result.len(), 1);
        let new_point = result[0];
        assert!((new_point.0 - 1.0).abs() < 1e-6);
        assert!((new_point.1 - 2.0).abs() < 1e-6);
        assert!((new_point.2 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_negative_adjustment() {
        let centerline = Centerline {
            points: vec![CenterlinePoint {
                contour_point: ContourPoint {
                    frame_index: 0,
                    point_index: 0,
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    aortic: false,
                },
                tangent: Vector3::new(1.0, 0.0, 0.0),
                branch_id: 0,
                radius: 0.0,
            }],
            branch_start_indices: vec![0],
        };

        let points = vec![(2.0, 0.0, 0.0)];

        let result = centerline_based_diameter_morphing(&centerline, &points, -0.5);

        // Should move toward centerline by 0.5 units
        let new_point = result[0];
        assert!((new_point.0 - 1.5).abs() < 1e-6);
        assert!((new_point.1 - 0.0).abs() < 1e-6);
        assert!((new_point.2 - 0.0).abs() < 1e-6);
    }

    // A centerline in the z = 0 plane; one (x, y) list per branch, branch 0 first. Each
    // branch numbers its frame_index from 0, as `resample` does.
    fn tree_centerline(branches: &[Vec<(f64, f64)>]) -> Centerline {
        let mut points = Vec::new();
        let mut branch_start_indices = Vec::new();
        for (b, branch) in branches.iter().enumerate() {
            branch_start_indices.push(points.len());
            points.extend(
                branch
                    .iter()
                    .enumerate()
                    .map(|(i, &(x, y))| CenterlinePoint {
                        contour_point: ContourPoint {
                            frame_index: i as u32,
                            point_index: i as u32,
                            x,
                            y,
                            z: 0.0,
                            aortic: false,
                        },
                        tangent: Vector3::zeros(),
                        branch_id: b as u32,
                        radius: 1.0,
                    }),
            );
        }
        Centerline {
            points,
            branch_start_indices,
        }
    }

    // Main branch: 21 points along x (index == x).
    fn main_branch() -> Vec<(f64, f64)> {
        (0..21).map(|i| (i as f64, 0.0)).collect()
    }

    // `n` points going +y from (x, 1).
    fn side_branch_at(x: f64, n: usize) -> Vec<(f64, f64)> {
        (0..n).map(|i| (x, 1.0 + i as f64)).collect()
    }

    // Main branch plus a 10-point side branch at `side_junction_x`, whose local
    // frame_index 0..9 collides with main-branch indices.
    fn branched_centerline(side_junction_x: f64) -> Centerline {
        tree_centerline(&[main_branch(), side_branch_at(side_junction_x, 10)])
    }

    fn frames_at(centroids: impl Iterator<Item = (f64, f64, f64)>) -> Vec<Frame> {
        centroids
            .enumerate()
            .map(|(k, centroid)| Frame {
                id: k as u32,
                centroid,
                lumen: Contour {
                    id: k as u32,
                    original_frame: k as u32,
                    points: Vec::new(),
                    centroid: None,
                    aortic_thickness: None,
                    pulmonary_thickness: None,
                    kind: crate::types::native::ContourType::Lumen,
                },
                extras: std::collections::HashMap::new(),
                reference_point: None,
            })
            .collect()
    }

    // Frames 0.7 mm apart along x from 3.0 to 7.9, all at z = 0 (a horizontal vessel),
    // offset 0.3 mm from the centerline like real aligned centroids.
    fn horizontal_frames() -> Vec<Frame> {
        frames_at((0..8).map(|k| (3.0 + 0.7 * k as f64, 0.3, 0.0)))
    }

    // Frames 0.7 mm apart along the side branch at x = 5, from y = 4.0 to 9.6.
    fn side_branch_frames() -> Vec<Frame> {
        frames_at((0..9).map(|k| (5.3, 4.0 + 0.7 * k as f64, 0.0)))
    }

    // 8 points on a circle around `center`, perpendicular to the x axis (or the y axis);
    // the half-step phase keeps them off exact ties between two centerline points.
    fn ring(center: (f64, f64, f64), axis_is_x: bool, r: f64) -> Vec<Coords3> {
        (0..8)
            .map(|k| {
                let a = (k as f64 + 0.5) * std::f64::consts::FRAC_PI_4;
                let (u, v) = (r * a.cos(), r * a.sin());
                if axis_is_x {
                    (center.0, center.1 + u, center.2 + v)
                } else {
                    (center.0 + u, center.1, center.2 + v)
                }
            })
            .collect()
    }

    fn all_in(rings: &[Vec<Coords3>], set: &[Coords3]) -> bool {
        rings.iter().flatten().all(|p| set.contains(p))
    }

    // Rings around the main branch (x = 0..=20) and around the x = 5 side branch
    // (y = 0..n, indexed by y; rings at y < 2 are left empty, too close to the junction).
    fn rings_for_side_pullback(n_side: usize) -> (Vec<Vec<Coords3>>, Vec<Vec<Coords3>>) {
        let main = (0..21)
            .map(|x| ring((x as f64, 0.0, 0.0), true, 1.0))
            .collect();
        let side = (0..=n_side)
            .map(|y| {
                if y < 2 {
                    Vec::new()
                } else {
                    ring((5.0, y as f64, 0.0), false, 0.5)
                }
            })
            .collect();
        (main, side)
    }

    #[test]
    fn test_find_points_by_cl_region_splits_by_main_branch_index() {
        let centerline = branched_centerline(15.0);
        let main_rings: Vec<Vec<Coords3>> = (0..21)
            .map(|x| ring((x as f64, 0.0, 0.0), true, 1.0))
            .collect();
        let side_points: Vec<Coords3> = (2..10)
            .flat_map(|y| ring((15.0, y as f64, 0.0), false, 0.5))
            .collect();
        let mut points: Vec<Coords3> = main_rings.concat();
        points.extend(&side_points);

        let (proximal, distal, between) =
            find_points_by_cl_region_rs(&centerline, &horizontal_frames(), &points).unwrap();
        assert_eq!(proximal.len() + distal.len() + between.len(), points.len());

        assert!(all_in(&main_rings[0..=1], &proximal));
        // Frames cover main-branch indices 3..=8; check away from the boundary rings.
        assert!(all_in(&main_rings[4..=7], &between));
        assert!(all_in(&main_rings[10..=20], &distal));
        // Side-branch points follow their junction (x = 15, distal), not their own
        // colliding frame_index values.
        assert!(side_points.iter().all(|p| distal.contains(p)));
    }

    #[test]
    fn test_find_points_by_cl_region_side_branch_inside_segment_is_between() {
        let centerline = branched_centerline(5.0);
        let side_points: Vec<Coords3> = (2..10)
            .flat_map(|y| ring((5.0, y as f64, 0.0), false, 0.5))
            .collect();
        let (proximal, distal, between) =
            find_points_by_cl_region_rs(&centerline, &horizontal_frames(), &side_points).unwrap();
        assert!(proximal.is_empty() && distal.is_empty());
        assert_eq!(between.len(), side_points.len());
    }

    #[test]
    fn test_find_points_by_cl_region_rejects_empty_input() {
        let centerline = branched_centerline(15.0);
        assert!(find_points_by_cl_region_rs(&centerline, &[], &[(0.0, 0.0, 0.0)]).is_err());
        let empty = Centerline {
            points: Vec::new(),
            branch_start_indices: Vec::new(),
        };
        assert!(find_points_by_cl_region_rs(&empty, &horizontal_frames(), &[]).is_err());
    }

    #[test]
    fn test_find_points_by_cl_region_pullback_on_side_branch() {
        // Side branch at x = 5 with 16 points (y = 1..=16, positions 0..=15); the frames
        // cover positions 3..=9 (y = 4..=10).
        let centerline = tree_centerline(&[main_branch(), side_branch_at(5.0, 16)]);
        let (main_rings, side_rings) = rings_for_side_pullback(16);
        let points: Vec<Coords3> = main_rings
            .iter()
            .chain(&side_rings)
            .flatten()
            .copied()
            .collect();

        let (proximal, distal, between) =
            find_points_by_cl_region_rs(&centerline, &side_branch_frames(), &points).unwrap();

        // The parent vessel is kept, including its wall at the junction (x = 4..=6).
        assert!(all_in(&main_rings, &distal));
        assert!(all_in(&side_rings[2..=2], &proximal));
        assert!(all_in(&side_rings[5..=9], &between));
        assert!(all_in(&side_rings[12..=16], &distal));
    }

    #[test]
    fn test_find_points_by_cl_region_side_branch_stored_distal_end_first() {
        let mut side = side_branch_at(5.0, 16);
        side.reverse();
        let centerline = tree_centerline(&[main_branch(), side]);
        let (main_rings, side_rings) = rings_for_side_pullback(16);
        let points: Vec<Coords3> = main_rings
            .iter()
            .chain(&side_rings)
            .flatten()
            .copied()
            .collect();

        let (proximal, distal, between) =
            find_points_by_cl_region_rs(&centerline, &side_branch_frames(), &points).unwrap();

        assert!(all_in(&main_rings, &distal));
        assert!(all_in(&side_rings[2..=2], &proximal));
        assert!(all_in(&side_rings[5..=9], &between));
        assert!(all_in(&side_rings[12..=16], &distal));
    }

    #[test]
    fn test_find_points_by_cl_region_nested_branch_follows_its_parent() {
        // Branch 2 leaves the side branch (not the main branch) at y = 3, position 2,
        // before the frames start, so it is proximal. Placing it by the nearest
        // main-branch point instead would make it distal.
        let nested: Vec<(f64, f64)> = (0..9).map(|i| (6.0 + i as f64, 3.0)).collect();
        let centerline = tree_centerline(&[main_branch(), side_branch_at(5.0, 16), nested]);
        let nested_rings: Vec<Vec<Coords3>> = (8..=13)
            .map(|x| ring((x as f64, 3.0, 0.0), true, 0.5))
            .collect();
        let points: Vec<Coords3> = nested_rings.concat();

        let (proximal, _, _) =
            find_points_by_cl_region_rs(&centerline, &side_branch_frames(), &points).unwrap();
        assert!(all_in(&nested_rings, &proximal));
    }
}
