use crate::types::native::Contour;
use nalgebra::Vector3;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct ReferenceTriplet {
    pub main_ref: (f64, f64, f64),
    pub counter_clock_ref: (f64, f64, f64), // view from proximal to distal, former upper_ref
    pub clock_ref: (f64, f64, f64),         // former lower_ref
}

/// One branch of a discretized coronary.
#[derive(Debug, Clone, Default)]
pub struct DiscretizedBranch {
    pub contours: Vec<Contour>,
    /// Parent branch id (from the centerline) and the parent contour nearest this branch's
    /// start. `None` for the main vessel (branch 0).
    pub parent: Option<(u32, usize)>,
    /// Triplet at this branch's ostium, on its first contour.
    pub ostium: Option<ReferenceTriplet>,
    /// Triplet on this branch where each child branch leaves it, by child branch id.
    pub bifurcations: BTreeMap<u32, ReferenceTriplet>,
}

#[derive(Debug, Clone)]
pub struct DiscretizedVesselTree {
    pub discretized_aorta: Vec<Contour>,
    pub spacing: f64,
    /// RCA branches by branch id, 0 being the main vessel.
    pub rca: BTreeMap<u32, DiscretizedBranch>,
    /// LCA branches by branch id, 0 being the main vessel.
    pub lca: BTreeMap<u32, DiscretizedBranch>,
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
    /// Computes `ao_rca`/`ao_lca` (aorta slice centroid nearest each main vessel's start), each
    /// branch's junction contour on its parent, and the reference triplets:
    /// - ostium, on the first contour: `main_ref` faces the vessel the branch leaves (aorta or
    ///   parent branch), the side refs lie a quarter ring either side
    /// - one per child branch, on the parent contour nearest the child's start: `main_ref` is
    ///   the child's start, the side refs lie a quarter ring either side of the point closest
    ///   to it
    ///
    /// Parent ids are kept as they are (they come from the centerline). A parent missing from
    /// the map is treated like the aorta.
    pub fn calculate_ref_pts(mut self) -> Self {
        if let Some(ao) = coronary_references(&self.discretized_aorta, &mut self.rca) {
            self.ao_rca = ao;
        }
        if let Some(ao) = coronary_references(&self.discretized_aorta, &mut self.lca) {
            self.ao_lca = ao;
        }
        self
    }
}

struct BranchSlices<'a> {
    contours: &'a [Contour],
    centroids: Vec<Vector3<f64>>,
}

/// Fills in the junction contour and reference triplets of every branch in `branches` and
/// returns the centroid of the aorta slice nearest the main vessel's start.
fn coronary_references(
    aorta: &[Contour],
    branches: &mut BTreeMap<u32, DiscretizedBranch>,
) -> Option<(f64, f64, f64)> {
    let first_main = contour_centroid(branches.get(&0)?.contours.first()?);
    let ao_centroid = aorta
        .iter()
        .map(contour_centroid)
        .min_by(|a, b| (a - first_main).norm().total_cmp(&(b - first_main).norm()))?;

    let slices: BTreeMap<u32, BranchSlices> = branches
        .iter()
        .map(|(&id, b)| {
            let slices = BranchSlices {
                contours: &b.contours,
                centroids: b.contours.iter().map(contour_centroid).collect(),
            };
            (id, slices)
        })
        .collect();
    // Parent id and the parent contour nearest the branch's start.
    let parents: BTreeMap<u32, Option<(u32, usize)>> = branches
        .iter()
        .map(|(&id, b)| {
            let junction = b.parent.and_then(|(p, _)| {
                let start = *slices[&id].centroids.first()?;
                let slice = nearest_centroid(&slices.get(&p)?.centroids, start)?;
                Some((p, slice))
            });
            (id, junction)
        })
        .collect();

    let references_of = |id: u32| -> (Option<ReferenceTriplet>, BTreeMap<u32, ReferenceTriplet>) {
        let branch = &slices[&id];
        let Some(&first) = branch.centroids.first() else {
            return (None, BTreeMap::new());
        };
        let origin = match parents[&id] {
            Some((p, slice)) => slices[&p].centroids[slice],
            None => ao_centroid,
        };
        let up_hint = (first - origin)
            .try_normalize(1e-12)
            .unwrap_or(Vector3::z());

        // Facing preference: the vessel left behind, back up the parent (a branch leaving at
        // ~90° has the parent's centre straight behind it), then down.
        let mut facing = vec![origin - first];
        if let Some((p, slice)) = parents[&id] {
            facing.push(-branch_direction(&slices[&p], slice, ao_centroid));
        }
        facing.push(-Vector3::z());

        let ostium = ostium_reference(origin, branch, &facing, up_hint);
        let bifurcations = parents
            .iter()
            .filter_map(|(&child, parent)| match *parent {
                Some((p, slice)) if p == id => {
                    bifurcation_reference(origin, branch, slice, &slices[&child], up_hint)
                        .map(|r| (child, r))
                }
                _ => None,
            })
            .collect();
        (ostium, bifurcations)
    };
    let references: Vec<_> = slices.keys().map(|&id| (id, references_of(id))).collect();

    for (id, (ostium, bifurcations)) in references {
        let branch = branches.get_mut(&id).expect("id comes from the map");
        if let Some(junction) = parents[&id] {
            branch.parent = Some(junction);
        }
        branch.ostium = ostium;
        branch.bifurcations = bifurcations;
    }
    Some((ao_centroid.x, ao_centroid.y, ao_centroid.z))
}

/// Index of the centroid in `centroids` nearest `target`.
fn nearest_centroid(centroids: &[Vector3<f64>], target: Vector3<f64>) -> Option<usize> {
    centroids
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (*a - target).norm().total_cmp(&(*b - target).norm()))
        .map(|(i, _)| i)
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

    fn branch(contours: Vec<Contour>, parent: Option<u32>) -> DiscretizedBranch {
        DiscretizedBranch {
            contours,
            parent: parent.map(|p| (p, 0)),
            ..Default::default()
        }
    }

    /// Aorta along z at y = −8, main vessel along +x from (15, 0, 0), side branch 1 off main
    /// slice 10 along +y, side branch 3 off side branch 1 slice 8 along +z. Both leave at 90°.
    /// Branch 2 was removed, so ids have a gap.
    fn tree() -> DiscretizedVesselTree {
        let aorta = vessel(Vector3::new(0.0, -8.0, -10.0), Vector3::z(), 21, 12.0);
        let main = vessel(Vector3::new(15.0, 0.0, 0.0), Vector3::x(), 30, 2.0);
        let side1 = vessel(Vector3::new(25.0, 4.0, 0.0), Vector3::y(), 20, 1.5);
        let side3 = vessel(Vector3::new(25.0, 12.0, 3.0), Vector3::z(), 10, 1.0);
        DiscretizedVesselTree {
            discretized_aorta: aorta,
            spacing: 1.0,
            rca: BTreeMap::new(),
            lca: BTreeMap::from([
                (0, branch(main, None)),
                (1, branch(side1, Some(0))),
                (3, branch(side3, Some(1))),
            ]),
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
        let ostium = t.lca[&0].ostium.as_ref().unwrap();
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
    fn test_references_are_keyed_by_child_id_on_the_parent() {
        let t = tree();
        // Main vessel: side branch 1 only (side branch 3 leaves side branch 1).
        let main = &t.lca[&0];
        assert_eq!(
            main.bifurcations.keys().copied().collect::<Vec<_>>(),
            vec![1]
        );
        assert!((v(main.bifurcations[&1].main_ref) - Vector3::new(25.0, 4.0, 0.0)).norm() < 1e-9);
        // Side branch 1 has side branch 3's bifurcation, side branch 3 has none.
        let side1 = &t.lca[&1];
        assert_eq!(side1.parent, Some((0, 10)));
        assert_eq!(
            side1.bifurcations.keys().copied().collect::<Vec<_>>(),
            vec![3]
        );
        assert!((v(side1.bifurcations[&3].main_ref) - Vector3::new(25.0, 12.0, 3.0)).norm() < 1e-9);
        assert_eq!(t.lca[&3].parent, Some((1, 8)));
        assert!(t.lca[&3].bifurcations.is_empty());
        // Side branch 1 leaves at 90°: the main vessel's centre is straight behind it, so its
        // main_ref faces back up the main vessel (−x): ≈ (23.5, 4, 0).
        let m = v(side1.ostium.as_ref().unwrap().main_ref);
        assert!(
            (m - Vector3::new(23.5, 4.0, 0.0)).norm() < 0.12,
            "main_ref {m}"
        );
        // Side branch 3 likewise faces back up side branch 1 (−y): ≈ (25, 11, 3).
        let m = v(t.lca[&3].ostium.as_ref().unwrap().main_ref);
        assert!(
            (m - Vector3::new(25.0, 11.0, 3.0)).norm() < 0.08,
            "main_ref {m}"
        );
    }

    #[test]
    fn test_parent_comes_from_the_branch_not_the_geometry() {
        // Branch 3 starts right next to the main vessel, but its parent is branch 1: the
        // bifurcation goes on branch 1, the nearest contour of branch 1.
        let mut t = tree();
        let near_main = vessel(Vector3::new(20.0, 2.5, 0.0), Vector3::y(), 5, 0.5);
        t.lca.insert(3, branch(near_main, Some(1)));
        let t = t.calculate_ref_pts();
        assert!(!t.lca[&0].bifurcations.contains_key(&3));
        assert!(t.lca[&1].bifurcations.contains_key(&3));
        assert_eq!(t.lca[&3].parent.map(|(p, _)| p), Some(1));
    }
}
