use crate::intravascular::io::geometry::Contour;
use crate::intravascular::io::input::Centerline;
use anyhow::{Ok, Result};

use super::discretize_vessel_rs;
use super::utils::smooth_centerline;

pub struct ReferenceTriplet {
    pub main_ref: (f64, f64, f64),
    pub counter_clock_ref: (f64, f64, f64), // view from proximal to distal, former upper_ref
    pub clock_ref: (f64, f64, f64),         // former lower_ref
}

pub struct DiscretizedVesselTree {
    pub discretized_aorta: Vec<Contour>,
    pub discretized_rca_main: Vec<Contour>,
    pub discretized_lca_main: Vec<Contour>,
    pub spacing: f64,
    /// One entry per RCA side branch; index `i` → branch_id `i + 1`.
    pub rca_branches: Vec<Vec<Contour>>,
    /// One entry per LCA side branch; index `i` → branch_id `i + 1`.
    pub lca_branches: Vec<Vec<Contour>>,
    pub rca_references: Vec<ReferenceTriplet>,
    pub lca_references: Vec<ReferenceTriplet>,
    pub index_ao_lca: usize,
    pub index_ao_rca: usize,
    pub pts_cusp_rcc: Option<Vec<(f64, f64, f64)>>,
    pub pts_cusp_lcc: Option<Vec<(f64, f64, f64)>>,
    pub pts_cusp_acc: Option<Vec<(f64, f64, f64)>>,
    pub index_stj_slice: Option<usize>,
    pub index_aa: Option<usize>,
}

impl DiscretizedVesselTree {
    pub fn new(
        discretized_aorta: Vec<Contour>,
        discretized_rca_main: Vec<Contour>,
        discretized_lca_main: Vec<Contour>,
        spacing: f64,
        rca_branches: Vec<Vec<Contour>>,
        lca_branches: Vec<Vec<Contour>>,
        rca_references: Vec<ReferenceTriplet>,
        lca_references: Vec<ReferenceTriplet>,
        index_ao_lca: usize,
        index_ao_rca: usize,
        pts_cusp_rcc: Option<Vec<(f64, f64, f64)>>,
        pts_cusp_lcc: Option<Vec<(f64, f64, f64)>>,
        pts_cusp_acc: Option<Vec<(f64, f64, f64)>>,
        index_stj_slice: Option<usize>,
        index_aa: Option<usize>,
    ) -> anyhow::Result<DiscretizedVesselTree> {
        let tree = DiscretizedVesselTree {
            discretized_aorta,
            discretized_rca_main,
            discretized_lca_main,
            spacing,
            rca_branches,
            lca_branches,
            rca_references,
            lca_references,
            index_ao_lca,
            index_ao_rca,
            pts_cusp_rcc,
            pts_cusp_lcc,
            pts_cusp_acc,
            index_stj_slice,
            index_aa,
        };
        Ok(tree)
    }

    /// Discretize the full vessel tree from pre-labelled point sets.
    ///
    /// Smooths all three centerlines with a Gaussian (σ = 2.5 points) before
    /// discretizing, then processes:
    /// - aorta (branch 0)
    /// - RCA main vessel (`branch_id_rca`, usually 0)
    /// - LCA main vessel (`branch_id_lca`, usually 0)
    /// - every RCA side branch: `side_branches_rca[i]` → branch_id `i + 1`
    /// - every LCA side branch: `side_branches_lca[i]` → branch_id `i + 1`
    ///
    /// `side_branches_rca` / `side_branches_lca` come from `label_branches` results:
    /// `results["rca_points_side_1"]`, `results["rca_points_side_2"]`, …, in order.
    pub fn from_results_dict(
        ao_cl: &Centerline,
        rca_cl: &Centerline,
        lca_cl: &Centerline,
        points_ao: &[(f64, f64, f64)],
        points_rca_main: &[(f64, f64, f64)],
        points_lca_main: &[(f64, f64, f64)],
        side_branches_rca: Vec<Vec<(f64, f64, f64)>>,
        side_branches_lca: Vec<Vec<(f64, f64, f64)>>,
        branch_id_rca: u32,
        branch_id_lca: u32,
        step_size: f64,
        n_points: usize,
    ) -> Result<DiscretizedVesselTree> {
        const SMOOTH_SIGMA: f64 = 2.5;

        let ao_cl_s = smooth_centerline(ao_cl, SMOOTH_SIGMA);
        let rca_cl_s = smooth_centerline(rca_cl, SMOOTH_SIGMA);
        let lca_cl_s = smooth_centerline(lca_cl, SMOOTH_SIGMA);

        let discretized_aorta = discretize_vessel_rs(&ao_cl_s, points_ao, 0, step_size, n_points);
        let discretized_rca_main = discretize_vessel_rs(
            &rca_cl_s,
            points_rca_main,
            branch_id_rca,
            step_size,
            n_points,
        );
        let discretized_lca_main = discretize_vessel_rs(
            &lca_cl_s,
            points_lca_main,
            branch_id_lca,
            step_size,
            n_points,
        );

        // side_branches_rca[i] carries the surface points for branch_id i+1.
        let rca_branches: Vec<Vec<Contour>> = side_branches_rca
            .iter()
            .enumerate()
            .map(|(i, pts)| {
                discretize_vessel_rs(&rca_cl_s, pts, (i + 1) as u32, step_size, n_points)
            })
            .collect();

        let lca_branches: Vec<Vec<Contour>> = side_branches_lca
            .iter()
            .enumerate()
            .map(|(i, pts)| {
                discretize_vessel_rs(&lca_cl_s, pts, (i + 1) as u32, step_size, n_points)
            })
            .collect();

        Ok(DiscretizedVesselTree {
            discretized_aorta,
            discretized_rca_main,
            discretized_lca_main,
            spacing: step_size,
            rca_branches,
            lca_branches,
            rca_references: vec![],
            lca_references: vec![],
            index_ao_lca: 0,
            index_ao_rca: 0,
            pts_cusp_rcc: None,
            pts_cusp_lcc: None,
            pts_cusp_acc: None,
            index_stj_slice: None,
            index_aa: None,
        })
    }

    /// Compute `index_ao_rca`, `index_ao_lca`, `rca_references`, and `lca_references`.
    ///
    /// **`index_ao_*`** – index into `discretized_aorta` whose centroid is closest to
    /// the first contour of the respective main-vessel discretization.
    ///
    /// **Reference triplets** are built at two kinds of landmarks, then sorted
    /// proximal → distal by their position on the main-vessel contour list:
    ///
    /// 1. *Ostium* (always the most proximal):
    ///    - `main_ref`         = the point on the minor-axis pair (`find_closest_opposite_3d`)
    ///      of the first coronary contour that is closer to the aorta centroid
    ///    - `counter_clock_ref`/ `clock_ref` = the two major-axis points
    ///      (`find_farthest_points`), assigned left/right when viewing proximal→distal
    ///
    /// 2. *Side-branch bifurcations* (one per non-empty side branch):
    ///    - Find the main-vessel contour whose centroid is closest to the side
    ///      branch's first-contour centroid
    ///    - `main_ref`         = side branch first-contour centroid
    ///    - `counter_clock_ref`/ `clock_ref` = the two points ±¼ of the contour
    ///      ring away from the closest point on the main-vessel contour,
    ///      assigned left/right by the same cross-product rule
    ///
    /// Left/right is determined by an "up" hint = direction from aorta centroid to
    /// first main-vessel centroid, projected perpendicular to the local vessel normal.
    pub fn calculate_ref_pts(mut self) -> Self {
        // Compute index_ao_rca / index_ao_lca: aorta slice closest to each vessel's first contour.
        if !self.discretized_aorta.is_empty() {
            if !self.discretized_rca_main.is_empty() {
                let c0 = contour_centroid(&self.discretized_rca_main[0]);
                self.index_ao_rca = self
                    .discretized_aorta
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        v3_dist(contour_centroid(a), c0)
                            .partial_cmp(&v3_dist(contour_centroid(b), c0))
                            .unwrap()
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0);
            }
            if !self.discretized_lca_main.is_empty() {
                let c0 = contour_centroid(&self.discretized_lca_main[0]);
                self.index_ao_lca = self
                    .discretized_aorta
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        v3_dist(contour_centroid(a), c0)
                            .partial_cmp(&v3_dist(contour_centroid(b), c0))
                            .unwrap()
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0);
            }
        }

        if !self.discretized_rca_main.is_empty() && self.index_ao_rca < self.discretized_aorta.len()
        {
            let ao_centroid = contour_centroid(&self.discretized_aorta[self.index_ao_rca]);
            self.rca_references =
                vessel_references(ao_centroid, &self.discretized_rca_main, &self.rca_branches);
        }
        if !self.discretized_lca_main.is_empty() && self.index_ao_lca < self.discretized_aorta.len()
        {
            let ao_centroid = contour_centroid(&self.discretized_aorta[self.index_ao_lca]);
            self.lca_references =
                vessel_references(ao_centroid, &self.discretized_lca_main, &self.lca_branches);
        }
        self
    }
}

// ─── private helpers ────────────────────────────────────────────────────────

/// Build a sorted (proximal→distal) reference-triplet list for one vessel.
fn vessel_references(
    ao_centroid: (f64, f64, f64),
    main: &[Contour],
    side_branches: &[Vec<Contour>],
) -> Vec<ReferenceTriplet> {
    // Pre-compute all main-contour centroids once.
    let main_centroids: Vec<(f64, f64, f64)> = main.iter().map(contour_centroid).collect();

    // "Up" hint: direction from aorta centroid to first main-vessel centroid,
    // constant for the whole vessel to keep all refs in the same orientation frame.
    let up_hint = v3_normalize(v3_sub(main_centroids[0], ao_centroid));

    // (sorting_key, ReferenceTriplet) – key = index in main contour list
    let mut tagged: Vec<(usize, ReferenceTriplet)> = Vec::new();

    // ── 1. Ostium reference ──────────────────────────────────────────────────
    if let Some(first) = main.first() {
        if first.points.len() > 2 {
            // Normal at the ostium: toward the second contour
            let normal = if main.len() > 1 {
                v3_normalize(v3_sub(main_centroids[1], main_centroids[0]))
            } else {
                v3_normalize(v3_sub(main_centroids[0], ao_centroid))
            };

            // main_ref: minor-axis point closer to the aorta
            let ((pa, pb), _) = first.find_closest_opposite_3d();
            let pta = (pa.x, pa.y, pa.z);
            let ptb = (pb.x, pb.y, pb.z);
            let main_ref = if v3_dist(pta, ao_centroid) <= v3_dist(ptb, ao_centroid) {
                pta
            } else {
                ptb
            };

            // counter_clock / clock: major-axis pair
            let ((p1, p2), _) = first.find_farthest_points();
            let (counter_clock_ref, clock_ref) = assign_cc_clock(
                (p1.x, p1.y, p1.z),
                (p2.x, p2.y, p2.z),
                main_centroids[0],
                normal,
                up_hint,
            );

            tagged.push((
                0,
                ReferenceTriplet {
                    main_ref,
                    counter_clock_ref,
                    clock_ref,
                },
            ));
        }
    }

    // ── 2. Side-branch bifurcation references ────────────────────────────────
    for branch_contours in side_branches {
        if branch_contours.is_empty() {
            continue;
        }
        let side_c0 = contour_centroid(&branch_contours[0]);

        // Closest main-vessel contour by centroid distance
        let (bifurc_idx, _) = main_centroids
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                v3_dist(**a, side_c0)
                    .partial_cmp(&v3_dist(**b, side_c0))
                    .unwrap()
            })
            .unwrap_or((0, &main_centroids[0]));

        let bifurc_centroid = main_centroids[bifurc_idx];

        // Local proximal→distal normal at the bifurcation
        let normal = if bifurc_idx + 1 < main.len() {
            v3_normalize(v3_sub(main_centroids[bifurc_idx + 1], bifurc_centroid))
        } else if bifurc_idx > 0 {
            v3_normalize(v3_sub(bifurc_centroid, main_centroids[bifurc_idx - 1]))
        } else {
            v3_normalize(v3_sub(bifurc_centroid, ao_centroid))
        };

        let bifurc_contour = &main[bifurc_idx];
        let n_pts = bifurc_contour.points.len();
        if n_pts < 4 {
            continue;
        }

        // Point on main contour closest to the side-branch centroid
        let closest_idx = bifurc_contour
            .points
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                v3_dist((a.x, a.y, a.z), side_c0)
                    .partial_cmp(&v3_dist((b.x, b.y, b.z), side_c0))
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        // ±¼ around the contour ring
        let quarter = n_pts / 4;
        let idx_plus = (closest_idx + quarter) % n_pts;
        let idx_minus = (closest_idx + n_pts - quarter) % n_pts;
        let pp = &bifurc_contour.points[idx_plus];
        let pm = &bifurc_contour.points[idx_minus];

        let (counter_clock_ref, clock_ref) = assign_cc_clock(
            (pp.x, pp.y, pp.z),
            (pm.x, pm.y, pm.z),
            bifurc_centroid,
            normal,
            up_hint,
        );

        tagged.push((
            bifurc_idx,
            ReferenceTriplet {
                main_ref: side_c0,
                counter_clock_ref,
                clock_ref,
            },
        ));
    }

    // Sort proximal → distal
    tagged.sort_by_key(|(k, _)| *k);
    tagged.into_iter().map(|(_, r)| r).collect()
}

/// Assign the two points to (counter_clock, clock) when viewing proximal→distal.
///
/// "Left = counter_clock" convention (user-specified):
/// - Compute an "up" direction perpendicular to `normal` from `up_hint`
/// - `right = up_perp × normal` (right-hand rule: up × forward = right)
/// - `p` is counter_clock (left) if its component along `right` is negative
fn assign_cc_clock(
    p1: (f64, f64, f64),
    p2: (f64, f64, f64),
    centroid: (f64, f64, f64),
    normal: (f64, f64, f64),
    up_hint: (f64, f64, f64),
) -> ((f64, f64, f64), (f64, f64, f64)) {
    // Project up_hint perpendicular to normal
    let up_perp = v3_normalize(v3_sub(up_hint, v3_scale(normal, v3_dot(up_hint, normal))));

    // right = up_perp × normal  (when looking in +normal, right is clockwise)
    let right = v3_cross(up_perp, normal);

    let v1 = v3_sub(p1, centroid);
    if v3_dot(v1, right) < 0.0 {
        (p1, p2) // p1 is to the left → counter_clock
    } else {
        (p2, p1) // p2 is to the left → counter_clock
    }
}

// ─── 3-D vector micro-utilities (tuple-based, no nalgebra dependency) ────────

fn contour_centroid(c: &Contour) -> (f64, f64, f64) {
    if let Some(cen) = c.centroid {
        return cen;
    }
    let n = c.points.len() as f64;
    let (sx, sy, sz) = c.points.iter().fold((0.0, 0.0, 0.0), |(sx, sy, sz), p| {
        (sx + p.x, sy + p.y, sz + p.z)
    });
    (sx / n, sy / n, sz / n)
}

#[inline]
fn v3_sub(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (a.0 - b.0, a.1 - b.1, a.2 - b.2)
}
#[inline]
fn v3_scale(v: (f64, f64, f64), s: f64) -> (f64, f64, f64) {
    (v.0 * s, v.1 * s, v.2 * s)
}
#[inline]
fn v3_dot(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}
#[inline]
fn v3_cross(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (
        a.1 * b.2 - a.2 * b.1,
        a.2 * b.0 - a.0 * b.2,
        a.0 * b.1 - a.1 * b.0,
    )
}
#[inline]
fn v3_dist(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    let d = v3_sub(a, b);
    (d.0 * d.0 + d.1 * d.1 + d.2 * d.2).sqrt()
}
#[inline]
fn v3_normalize(v: (f64, f64, f64)) -> (f64, f64, f64) {
    let len = (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).sqrt();
    if len > 1e-12 {
        (v.0 / len, v.1 / len, v.2 / len)
    } else {
        (0.0, 0.0, 0.0)
    }
}
