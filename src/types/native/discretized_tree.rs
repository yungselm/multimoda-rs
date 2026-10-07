use crate::types::native::Contour;
use nalgebra::Vector3;

#[derive(Debug, Clone)]
pub struct ReferenceTriplet {
    pub main_ref: (f64, f64, f64),
    pub counter_clock_ref: (f64, f64, f64), // view from proximal to distal, former upper_ref
    pub clock_ref: (f64, f64, f64),         // former lower_ref
}

#[derive(Debug, Clone)]
pub struct DiscretizedVesselTree {
    pub discretized_aorta: Vec<Contour>,
    pub discretized_rca_main: Vec<Contour>,
    pub discretized_lca_main: Vec<Contour>,
    pub spacing: f64,
    /// Index `i` is RCA side branch `i + 1`.
    pub rca_branches: Vec<Vec<Contour>>,
    /// Index `i` is LCA side branch `i + 1`.
    pub lca_branches: Vec<Vec<Contour>>,
    /// Ostium triplet, then one per branch leaving the main vessel.
    pub rca_references: Vec<ReferenceTriplet>,
    pub lca_references: Vec<ReferenceTriplet>,
    /// Per side branch (as in `rca_branches`): its ostium, then one per branch leaving it.
    pub rca_branch_references: Vec<Vec<ReferenceTriplet>>,
    pub lca_branch_references: Vec<Vec<ReferenceTriplet>>,
    /// Centroid of the aorta slice closest to the RCA ostium.
    pub ao_rca: (f64, f64, f64),
    /// Centroid of the aorta slice closest to the LCA ostium.
    pub ao_lca: (f64, f64, f64),
    pub pts_cusp_rcc: Option<Vec<(f64, f64, f64)>>,
    pub pts_cusp_lcc: Option<Vec<(f64, f64, f64)>>,
    pub pts_cusp_acc: Option<Vec<(f64, f64, f64)>>,
    pub index_stj_slice: Option<usize>,
    pub index_aa: Option<usize>,
}

impl DiscretizedVesselTree {
    /// Computes `ao_rca`/`ao_lca` (aorta slice centroid nearest each main vessel's start) and the
    /// reference triplets of every branch, sorted proximal → distal:
    /// - ostium, on the first contour: `main_ref` faces the vessel the branch leaves (aorta or
    ///   parent branch), the side refs lie a quarter ring either side
    /// - one per branch leaving this one, on the nearest contour: `main_ref` is the child's
    ///   start, the side refs lie a quarter ring either side of the point closest to it
    ///
    /// A side branch's parent is the earlier branch with a contour nearest its start.
    pub fn calculate_ref_pts(mut self) -> Self {
        if let Some(refs) = coronary_references(
            &self.discretized_aorta,
            &self.discretized_rca_main,
            &self.rca_branches,
        ) {
            self.ao_rca = refs.ao_centroid;
            self.rca_references = refs.main;
            self.rca_branch_references = refs.branches;
        }
        if let Some(refs) = coronary_references(
            &self.discretized_aorta,
            &self.discretized_lca_main,
            &self.lca_branches,
        ) {
            self.ao_lca = refs.ao_centroid;
            self.lca_references = refs.main;
            self.lca_branch_references = refs.branches;
        }
        self
    }
}

struct CoronaryReferences {
    ao_centroid: (f64, f64, f64),
    main: Vec<ReferenceTriplet>,
    branches: Vec<Vec<ReferenceTriplet>>,
}

struct BranchSlices<'a> {
    contours: &'a [Contour],
    centroids: Vec<Vector3<f64>>,
}

fn coronary_references(
    aorta: &[Contour],
    main: &[Contour],
    side_branches: &[Vec<Contour>],
) -> Option<CoronaryReferences> {
    let first_main = contour_centroid(main.first()?);
    let ao_centroid = aorta
        .iter()
        .map(contour_centroid)
        .min_by(|a, b| (a - first_main).norm().total_cmp(&(b - first_main).norm()))?;

    // Index 0 is the main vessel, index `j` side branch `j` (branch_id `j`).
    let branches: Vec<BranchSlices> = std::iter::once(main)
        .chain(side_branches.iter().map(Vec::as_slice))
        .map(|contours| BranchSlices {
            contours,
            centroids: contours.iter().map(contour_centroid).collect(),
        })
        .collect();
    let parents: Vec<Option<(usize, usize)>> = (0..branches.len())
        .map(|b| find_parent(b, &branches))
        .collect();

    let references_of = |b: usize| -> Vec<ReferenceTriplet> {
        let branch = &branches[b];
        let Some(&first) = branch.centroids.first() else {
            return vec![];
        };
        let origin = match parents[b] {
            Some((p, slice)) => branches[p].centroids[slice],
            None => ao_centroid,
        };
        let up_hint = (first - origin)
            .try_normalize(1e-12)
            .unwrap_or(Vector3::z());

        // Facing preference: the vessel left behind, back up the parent (a branch leaving at
        // ~90° has the parent's centre straight behind it), then down.
        let mut facing = vec![origin - first];
        if let Some((p, slice)) = parents[b] {
            facing.push(-branch_direction(&branches[p], slice, ao_centroid));
        }
        facing.push(-Vector3::z());

        let mut tagged: Vec<(usize, ReferenceTriplet)> = Vec::new();
        if let Some(r) = ostium_reference(origin, branch, &facing, up_hint) {
            tagged.push((0, r));
        }
        for (child, parent) in parents.iter().enumerate() {
            if let Some((p, slice)) = *parent {
                if p == b {
                    if let Some(r) =
                        bifurcation_reference(origin, branch, slice, &branches[child], up_hint)
                    {
                        tagged.push((slice, r));
                    }
                }
            }
        }
        tagged.sort_by_key(|(k, _)| *k);
        tagged.into_iter().map(|(_, r)| r).collect()
    };

    Some(CoronaryReferences {
        ao_centroid: (ao_centroid.x, ao_centroid.y, ao_centroid.z),
        main: references_of(0),
        branches: (1..branches.len()).map(references_of).collect(),
    })
}

/// Parent of side branch `b` (main or lower id) and the parent slice nearest `b`'s start.
fn find_parent(b: usize, branches: &[BranchSlices]) -> Option<(usize, usize)> {
    if b == 0 {
        return None;
    }
    let start = *branches[b].centroids.first()?;
    branches[..b]
        .iter()
        .enumerate()
        .flat_map(|(p, branch)| {
            branch
                .centroids
                .iter()
                .enumerate()
                .map(move |(slice, c)| (p, slice, (c - start).norm()))
        })
        .min_by(|a, b| a.2.total_cmp(&b.2))
        .map(|(p, slice, _)| (p, slice))
}

/// Unit direction of the branch at `slice` (from `origin` for a single contour).
fn branch_direction(branch: &BranchSlices, slice: usize, origin: Vector3<f64>) -> Vector3<f64> {
    let c = &branch.centroids;
    let dir = if slice + 1 < c.len() {
        c[slice + 1] - c[slice]
    } else if slice > 0 {
        c[slice] - c[slice - 1]
    } else {
        c[slice] - origin
    };
    dir.try_normalize(1e-12).unwrap_or(Vector3::z())
}

fn closest_point_index(contour: &Contour, target: Vector3<f64>) -> Option<usize> {
    contour
        .points
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (Vector3::new(a.x, a.y, a.z) - target)
                .norm()
                .total_cmp(&(Vector3::new(b.x, b.y, b.z) - target).norm())
        })
        .map(|(i, _)| i)
}

/// Points a quarter ring either side of `idx`, as (counter_clock, clock).
fn quarter_points(
    contour: &Contour,
    idx: usize,
    centroid: Vector3<f64>,
    normal: Vector3<f64>,
    up_hint: Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let n = contour.points.len();
    let quarter = n / 4;
    let pp = &contour.points[(idx + quarter) % n];
    let pm = &contour.points[(idx + n - quarter) % n];
    assign_cc_clock(
        Vector3::new(pp.x, pp.y, pp.z),
        Vector3::new(pm.x, pm.y, pm.z),
        centroid,
        normal,
        up_hint,
    )
}

/// A facing direction must be this far off the axis (sin ≈ 11.5°) to project reliably.
const MIN_OFF_AXIS: f64 = 0.2;

/// Ostium triplet. `main_ref` faces the first usable direction in `facing`, which unlike the
/// contour shape stays stable on round ostia.
fn ostium_reference(
    origin: Vector3<f64>,
    branch: &BranchSlices,
    facing: &[Vector3<f64>],
    up_hint: Vector3<f64>,
) -> Option<ReferenceTriplet> {
    let first = branch.contours.first()?;
    if first.points.len() < 4 {
        return None;
    }
    let centroid = branch.centroids[0];
    let normal = branch_direction(branch, 0, origin);
    let facing = facing.iter().find_map(|d| {
        let in_plane = d - normal * d.dot(&normal);
        (in_plane.norm() >= MIN_OFF_AXIS * d.norm()).then_some(in_plane)
    })?;
    let idx = first
        .points
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let cos = |p: &crate::types::native::ContourPoint| {
                (Vector3::new(p.x, p.y, p.z) - centroid)
                    .try_normalize(1e-12)
                    .map_or(-1.0, |d| d.dot(&facing))
            };
            cos(a).total_cmp(&cos(b))
        })
        .map(|(i, _)| i)?;

    let main_ref = &first.points[idx];
    let (cc, cl) = quarter_points(first, idx, centroid, normal, up_hint);
    Some(ReferenceTriplet {
        main_ref: (main_ref.x, main_ref.y, main_ref.z),
        counter_clock_ref: (cc.x, cc.y, cc.z),
        clock_ref: (cl.x, cl.y, cl.z),
    })
}

/// Triplet on `branch`'s contour `slice`, where `child` leaves it.
fn bifurcation_reference(
    origin: Vector3<f64>,
    branch: &BranchSlices,
    slice: usize,
    child: &BranchSlices,
    up_hint: Vector3<f64>,
) -> Option<ReferenceTriplet> {
    let child_start = *child.centroids.first()?;
    let contour = &branch.contours[slice];
    if contour.points.len() < 4 {
        return None;
    }
    let normal = branch_direction(branch, slice, origin);
    let idx = closest_point_index(contour, child_start)?;
    let (cc, cl) = quarter_points(contour, idx, branch.centroids[slice], normal, up_hint);
    Some(ReferenceTriplet {
        main_ref: (child_start.x, child_start.y, child_start.z),
        counter_clock_ref: (cc.x, cc.y, cc.z),
        clock_ref: (cl.x, cl.y, cl.z),
    })
}

/// Orders two points as (counter_clock, clock) viewed proximal → distal: counter_clock is left
/// of the vessel, i.e. negative along `up × normal` with "up" taken from `up_hint`.
fn assign_cc_clock(
    p1: Vector3<f64>,
    p2: Vector3<f64>,
    centroid: Vector3<f64>,
    normal: Vector3<f64>,
    up_hint: Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let up_perp = (up_hint - normal * up_hint.dot(&normal))
        .try_normalize(1e-12)
        .unwrap_or(Vector3::zeros());

    let right = up_perp.cross(&normal);

    if (p1 - centroid).dot(&right) < 0.0 {
        (p1, p2)
    } else {
        (p2, p1)
    }
}

fn contour_centroid(c: &Contour) -> Vector3<f64> {
    if let Some((x, y, z)) = c.centroid {
        return Vector3::new(x, y, z);
    }
    let n = c.points.len() as f64;
    let sum = c
        .points
        .iter()
        .fold(Vector3::zeros(), |acc, p| acc + Vector3::new(p.x, p.y, p.z));
    sum / n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::native::{ContourPoint, ContourType};
    use std::f64::consts::TAU;

    /// Round contour of radius `r` around `center`, in the plane perpendicular to `dir`.
    fn ring(id: u32, center: Vector3<f64>, dir: Vector3<f64>, r: f64) -> Contour {
        let dir = dir.normalize();
        let u = dir.cross(&Vector3::new(0.3, 0.5, 0.8)).normalize();
        let v = dir.cross(&u);
        Contour {
            id,
            original_frame: id,
            centroid: Some((center.x, center.y, center.z)),
            points: (0..40)
                .map(|i| {
                    let a = TAU * i as f64 / 40.0;
                    let p = center + (u * a.cos() + v * a.sin()) * r;
                    ContourPoint {
                        frame_index: id,
                        point_index: i,
                        x: p.x,
                        y: p.y,
                        z: p.z,
                        aortic: false,
                    }
                })
                .collect(),
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        }
    }

    /// Straight vessel of `n` round slices from `start` along `dir`, 1 mm apart.
    fn vessel(start: Vector3<f64>, dir: Vector3<f64>, n: usize, r: f64) -> Vec<Contour> {
        let d = dir.normalize();
        (0..n)
            .map(|i| ring(i as u32, start + d * i as f64, d, r))
            .collect()
    }

    fn v(t: (f64, f64, f64)) -> Vector3<f64> {
        Vector3::new(t.0, t.1, t.2)
    }

    /// Aorta along z at y = −8, main vessel along +x from (15, 0, 0), side branch 1 off main
    /// slice 10 along +y, side branch 2 off side branch 1 slice 8 along +z. Both leave at 90°.
    fn tree() -> DiscretizedVesselTree {
        let aorta = vessel(Vector3::new(0.0, -8.0, -10.0), Vector3::z(), 21, 12.0);
        let main = vessel(Vector3::new(15.0, 0.0, 0.0), Vector3::x(), 30, 2.0);
        let side1 = vessel(Vector3::new(25.0, 4.0, 0.0), Vector3::y(), 20, 1.5);
        let side2 = vessel(Vector3::new(25.0, 12.0, 3.0), Vector3::z(), 10, 1.0);
        DiscretizedVesselTree {
            discretized_aorta: aorta,
            discretized_rca_main: vec![],
            discretized_lca_main: main,
            spacing: 1.0,
            rca_branches: vec![],
            lca_branches: vec![side1, side2],
            rca_references: vec![],
            lca_references: vec![],
            rca_branch_references: vec![],
            lca_branch_references: vec![],
            ao_rca: (0.0, 0.0, 0.0),
            ao_lca: (0.0, 0.0, 0.0),
            pts_cusp_rcc: None,
            pts_cusp_lcc: None,
            pts_cusp_acc: None,
            index_stj_slice: None,
            index_aa: None,
        }
        .calculate_ref_pts()
    }

    #[test]
    fn test_ostium_main_ref_faces_aorta_on_round_contour() {
        let t = tree();
        let ostium = &t.lca_references[0];
        // First main contour: centre (15, 0, 0), radius 2, in the yz plane. The aorta centre
        // (0, −8, 0) projects onto −y, so main_ref ≈ (15, −2, 0). Ring points are 9° apart, so
        // the nearest one is within 2·sin(4.5°) ≈ 0.16 mm.
        let m = v(ostium.main_ref);
        assert!(
            (m - Vector3::new(15.0, -2.0, 0.0)).norm() < 0.16,
            "main_ref {m}"
        );
        // Side points a quarter turn away, on opposite sides (±z).
        let (cc, cl) = (v(ostium.counter_clock_ref), v(ostium.clock_ref));
        assert!(cc.y.abs() < 0.16 && cl.y.abs() < 0.16);
        assert!((cc - cl).norm() > 3.9);
    }

    #[test]
    fn test_references_are_placed_on_the_parent_branch() {
        let t = tree();
        // Main vessel: ostium + side branch 1 only (side branch 2 leaves side branch 1).
        assert_eq!(t.lca_references.len(), 2);
        assert!((v(t.lca_references[1].main_ref) - Vector3::new(25.0, 4.0, 0.0)).norm() < 1e-9);
        // Side branch 1 has its ostium and side branch 2's bifurcation, side branch 2 its ostium.
        assert_eq!(t.lca_branch_references.len(), 2);
        assert_eq!(t.lca_branch_references[0].len(), 2);
        assert!(
            (v(t.lca_branch_references[0][1].main_ref) - Vector3::new(25.0, 12.0, 3.0)).norm()
                < 1e-9
        );
        assert_eq!(t.lca_branch_references[1].len(), 1);
        // Side branch 1 leaves at 90°: the main vessel's centre is straight behind it, so its
        // main_ref faces back up the main vessel (−x): ≈ (23.5, 4, 0).
        let m = v(t.lca_branch_references[0][0].main_ref);
        assert!(
            (m - Vector3::new(23.5, 4.0, 0.0)).norm() < 0.12,
            "main_ref {m}"
        );
        // Side branch 2 likewise faces back up side branch 1 (−y): ≈ (25, 11, 3).
        let m = v(t.lca_branch_references[1][0].main_ref);
        assert!(
            (m - Vector3::new(25.0, 11.0, 3.0)).norm() < 0.08,
            "main_ref {m}"
        );
    }
}
