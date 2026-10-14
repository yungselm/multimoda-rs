use crate::types::native::{Centerline, CenterlinePoint, Frame};
use rstar::primitives::GeomWithData;
use std::collections::{BTreeMap, HashMap};
use std::ops::RangeInclusive;

type Coords3 = (f64, f64, f64);
/// R-tree over centerline points, each tagged with its coordinates' bit pattern.
type CenterlineTree = rstar::RTree<GeomWithData<[f64; 3], [u64; 3]>>;

/// Points of a vessel split by where they sit relative to the pullback segment.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PullbackRegions {
    /// On the pullback path, before the segment the frames cover.
    pub proximal: Vec<Coords3>,
    /// On the pullback path, after the segment.
    pub distal: Vec<Coords3>,
    /// On the pullback path, inside the segment.
    pub anomalous: Vec<Coords3>,
    /// Not on the pullback path: other branches, and parent vessels beyond the junction
    /// the path leaves them at.
    pub off_path: Vec<Coords3>,
}

/// Split the labelled points of a vessel by where they sit along the pullback path of
/// branch `branch_id`, relative to the segment the `frames` cover.
///
/// The pullback path is [`Centerline::branch_path`]: from the ostium through the parent
/// branches (each up to the junction the path leaves it at) to the tip of `branch_id`.
/// `branch_points` holds the vertices labelled for each branch id. Only vertices of branches
/// on the path are placed on it, at the nearest point of their own branch. A vertex whose
/// nearest point is past the junction, and every vertex of another branch, is off the path.
/// Each frame is snapped to the path, and the lowest and highest path positions bound the
/// segment: vertices on the path are proximal below it, anomalous inside it, distal above.
pub fn find_points_by_cl_region_rs(
    centerline: &Centerline,
    branch_id: u32,
    frames: &[Frame],
    branch_points: &BTreeMap<u32, Vec<Coords3>>,
) -> Result<PullbackRegions, String> {
    if frames.is_empty() {
        return Err("find_points_by_cl_region requires at least one frame".to_string());
    }
    if centerline.points.is_empty() {
        return Err("find_points_by_cl_region requires a non-empty centerline".to_string());
    }
    let path = centerline.branch_path(branch_id)?;

    let key = |x: f64, y: f64, z: f64| [x.to_bits(), y.to_bits(), z.to_bits()];
    let path_position: HashMap<[u64; 3], usize> = path
        .points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            (
                key(p.contour_point.x, p.contour_point.y, p.contour_point.z),
                i,
            )
        })
        .collect();
    let tree_of = |pts: &[CenterlinePoint]| -> CenterlineTree {
        rstar::RTree::bulk_load(
            pts.iter()
                .map(|p| {
                    let c = &p.contour_point;
                    GeomWithData::new([c.x, c.y, c.z], key(c.x, c.y, c.z))
                })
                .collect(),
        )
    };
    let position_of = |tree: &CenterlineTree, p: [f64; 3]| {
        tree.nearest_neighbor(p)
            .and_then(|n| path_position.get(&n.data).copied())
    };

    let path_tree = tree_of(&path.points);
    let (seg_start, seg_end) = frames
        .iter()
        .filter_map(|f| position_of(&path_tree, [f.centroid.0, f.centroid.1, f.centroid.2]))
        .fold((usize::MAX, 0), |(lo, hi), p| (lo.min(p), hi.max(p)));

    // Branches on the path: `branch_id` and its ancestors.
    let parents = centerline.branch_parents();
    let mut on_path_branches = vec![branch_id];
    while let Some(Some(p)) = parents.get(*on_path_branches.last().unwrap() as usize) {
        on_path_branches.push(*p);
    }

    let segment = seg_start..=seg_end;

    let mut regions = PullbackRegions::default();
    for (&b, points) in branch_points {
        let range = centerline.branch_range(b).unwrap_or_default();
        let branch_tree = (on_path_branches.contains(&b) && !range.is_empty())
            .then(|| tree_of(&centerline.points[range]));
        for &point in points {
            let position = branch_tree
                .as_ref()
                .and_then(|tree| position_of(tree, [point.0, point.1, point.2]));
            regions.region_for(position, &segment).push(point);
        }
    }

    let (proximal, anomalous) =
        clean_up_non_section_points(regions.proximal, regions.anomalous, 1.0, 0.6);
    let (distal, anomalous) = clean_up_non_section_points(regions.distal, anomalous, 1.0, 0.6);
    Ok(PullbackRegions {
        proximal,
        distal,
        anomalous,
        off_path: regions.off_path,
    })
}

impl PullbackRegions {
    fn region_for(
        &mut self,
        position: Option<usize>,
        segment: &RangeInclusive<usize>,
    ) -> &mut Vec<Coords3> {
        match position {
            None => &mut self.off_path,
            Some(p) if p < *segment.start() => &mut self.proximal,
            Some(p) if segment.contains(&p) => &mut self.anomalous,
            Some(_) => &mut self.distal,
        }
    }
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

    fn labels(branches: &[&[Vec<Coords3>]]) -> BTreeMap<u32, Vec<Coords3>> {
        branches
            .iter()
            .enumerate()
            .map(|(b, rings)| (b as u32, rings.concat()))
            .collect()
    }

    fn total(r: &PullbackRegions) -> usize {
        r.proximal.len() + r.distal.len() + r.anomalous.len() + r.off_path.len()
    }

    #[test]
    fn test_find_points_by_cl_region_splits_main_branch_and_leaves_side_off_path() {
        let centerline = branched_centerline(15.0);
        let main_rings: Vec<Vec<Coords3>> = (0..21)
            .map(|x| ring((x as f64, 0.0, 0.0), true, 1.0))
            .collect();
        let side_rings: Vec<Vec<Coords3>> = (2..10)
            .map(|y| ring((15.0, y as f64, 0.0), false, 0.5))
            .collect();
        let points = labels(&[&main_rings, &side_rings]);

        let r = find_points_by_cl_region_rs(&centerline, 0, &horizontal_frames(), &points).unwrap();
        assert_eq!(total(&r), points.values().map(Vec::len).sum::<usize>());

        assert!(all_in(&main_rings[0..=1], &r.proximal));
        // Frames cover main-branch indices 3..=8; check away from the boundary rings.
        assert!(all_in(&main_rings[4..=7], &r.anomalous));
        assert!(all_in(&main_rings[10..=20], &r.distal));
        // The side branch is not on the pullback path, whatever its frame_index values.
        assert_eq!(r.off_path, side_rings.concat());
    }

    #[test]
    fn test_find_points_by_cl_region_side_branch_leaving_segment_is_off_path() {
        let centerline = branched_centerline(5.0);
        let side_rings: Vec<Vec<Coords3>> = (2..10)
            .map(|y| ring((5.0, y as f64, 0.0), false, 0.5))
            .collect();
        let points = labels(&[&[], &side_rings]);
        let r = find_points_by_cl_region_rs(&centerline, 0, &horizontal_frames(), &points).unwrap();
        assert!(r.anomalous.is_empty() && r.proximal.is_empty() && r.distal.is_empty());
        assert_eq!(r.off_path.len(), 8 * 8);
    }

    #[test]
    fn test_find_points_by_cl_region_rejects_bad_input() {
        let centerline = branched_centerline(15.0);
        let points = labels(&[&[ring((0.0, 0.0, 0.0), true, 1.0)]]);
        assert!(find_points_by_cl_region_rs(&centerline, 0, &[], &points).is_err());
        assert!(
            find_points_by_cl_region_rs(&centerline, 7, &horizontal_frames(), &points).is_err(),
            "unknown pullback branch"
        );
        let empty = Centerline {
            points: Vec::new(),
            branch_start_indices: Vec::new(),
        };
        assert!(find_points_by_cl_region_rs(&empty, 0, &horizontal_frames(), &points).is_err());
    }

    #[test]
    fn test_find_points_by_cl_region_pullback_on_side_branch() {
        // Side branch at x = 5 with 16 points (y = 1..=16). The path is main x = 0..=5, then
        // the side branch. The frames cover y = 4..=10.
        let centerline = tree_centerline(&[main_branch(), side_branch_at(5.0, 16)]);
        let (main_rings, side_rings) = rings_for_side_pullback(16);
        let points = labels(&[&main_rings, &side_rings]);

        let r =
            find_points_by_cl_region_rs(&centerline, 1, &side_branch_frames(), &points).unwrap();

        // The parent vessel up to the junction is proximal, beyond it off the path.
        assert!(all_in(&main_rings[0..=4], &r.proximal));
        assert!(all_in(&main_rings[7..=20], &r.off_path));
        assert!(all_in(&side_rings[2..=2], &r.proximal));
        assert!(all_in(&side_rings[5..=9], &r.anomalous));
        assert!(all_in(&side_rings[12..=16], &r.distal));
    }

    #[test]
    fn test_find_points_by_cl_region_side_branch_stored_distal_end_first() {
        let mut side = side_branch_at(5.0, 16);
        side.reverse();
        let centerline = tree_centerline(&[main_branch(), side]);
        let (main_rings, side_rings) = rings_for_side_pullback(16);
        let points = labels(&[&main_rings, &side_rings]);

        let r =
            find_points_by_cl_region_rs(&centerline, 1, &side_branch_frames(), &points).unwrap();

        assert!(all_in(&main_rings[0..=4], &r.proximal));
        assert!(all_in(&main_rings[7..=20], &r.off_path));
        assert!(all_in(&side_rings[2..=2], &r.proximal));
        assert!(all_in(&side_rings[5..=9], &r.anomalous));
        assert!(all_in(&side_rings[12..=16], &r.distal));
    }

    #[test]
    fn test_find_points_by_cl_region_nested_branch_is_off_path() {
        // Branch 2 leaves the pullback branch 1 at y = 3, so it is not on the path.
        let nested: Vec<(f64, f64)> = (0..9).map(|i| (6.0 + i as f64, 3.0)).collect();
        let centerline = tree_centerline(&[main_branch(), side_branch_at(5.0, 16), nested]);
        let nested_rings: Vec<Vec<Coords3>> = (8..=13)
            .map(|x| ring((x as f64, 3.0, 0.0), true, 0.5))
            .collect();
        let points = labels(&[&[], &[], &nested_rings]);

        let r =
            find_points_by_cl_region_rs(&centerline, 1, &side_branch_frames(), &points).unwrap();
        assert_eq!(r.off_path, nested_rings.concat());
    }

    #[test]
    fn test_find_points_by_cl_region_frames_continue_into_parent() {
        // Frames aligned with `get_branch(1, with_parents=True)`: main branch x = 2..=5, then
        // the side branch at x = 5 up to y = 9.
        let centerline = tree_centerline(&[main_branch(), side_branch_at(5.0, 16)]);
        let frames = frames_at(
            (2..=4)
                .map(|x| (x as f64, 0.3, 0.0))
                .chain((1..=9).map(|y| (5.3, y as f64, 0.0))),
        );
        let (main_rings, side_rings) = rings_for_side_pullback(16);
        let points = labels(&[&main_rings, &side_rings]);

        let r = find_points_by_cl_region_rs(&centerline, 1, &frames, &points).unwrap();

        assert!(all_in(&main_rings[0..=0], &r.proximal));
        assert!(all_in(&main_rings[3..=3], &r.anomalous));
        assert!(all_in(&side_rings[3..=7], &r.anomalous));
        assert!(all_in(&side_rings[12..=16], &r.distal));
        // The main branch beyond the junction is not on the path.
        assert!(all_in(&main_rings[8..=20], &r.off_path));
    }
}
