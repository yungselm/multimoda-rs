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
    /// One entry per RCA side branch; index `i` → branch_id `i + 1`.
    pub rca_branches: Vec<Vec<Contour>>,
    /// One entry per LCA side branch; index `i` → branch_id `i + 1`.
    pub lca_branches: Vec<Vec<Contour>>,
    pub rca_references: Vec<ReferenceTriplet>,
    pub lca_references: Vec<ReferenceTriplet>,
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
    pub fn new(
        discretized_aorta: Vec<Contour>,
        discretized_rca_main: Vec<Contour>,
        discretized_lca_main: Vec<Contour>,
        spacing: f64,
        rca_branches: Vec<Vec<Contour>>,
        lca_branches: Vec<Vec<Contour>>,
        rca_references: Vec<ReferenceTriplet>,
        lca_references: Vec<ReferenceTriplet>,
        ao_lca: (f64, f64, f64),
        ao_rca: (f64, f64, f64),
        pts_cusp_rcc: Option<Vec<(f64, f64, f64)>>,
        pts_cusp_lcc: Option<Vec<(f64, f64, f64)>>,
        pts_cusp_acc: Option<Vec<(f64, f64, f64)>>,
        index_stj_slice: Option<usize>,
        index_aa: Option<usize>,
    ) -> anyhow::Result<Self> {
        Ok(DiscretizedVesselTree {
            discretized_aorta,
            discretized_rca_main,
            discretized_lca_main,
            spacing,
            rca_branches,
            lca_branches,
            rca_references,
            lca_references,
            ao_lca,
            ao_rca,
            pts_cusp_rcc,
            pts_cusp_lcc,
            pts_cusp_acc,
            index_stj_slice,
            index_aa,
        })
    }

    /// Compute `ao_rca`, `ao_lca`, `rca_references`, and `lca_references`.
    ///
    /// **`ao_rca` / `ao_lca`** – centroid of the aorta slice whose centroid is closest to
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
        if !self.discretized_aorta.is_empty() {
            if !self.discretized_rca_main.is_empty() {
                let c0 = contour_centroid(&self.discretized_rca_main[0]);
                if let Some(closest) = self.discretized_aorta.iter().min_by(|a, b| {
                    (contour_centroid(a) - c0)
                        .norm()
                        .partial_cmp(&(contour_centroid(b) - c0).norm())
                        .unwrap()
                }) {
                    let ao_centroid = contour_centroid(closest);
                    self.ao_rca = (ao_centroid.x, ao_centroid.y, ao_centroid.z);
                    self.rca_references = vessel_references(
                        ao_centroid,
                        &self.discretized_rca_main,
                        &self.rca_branches,
                    );
                }
            }
            if !self.discretized_lca_main.is_empty() {
                let c0 = contour_centroid(&self.discretized_lca_main[0]);
                if let Some(closest) = self.discretized_aorta.iter().min_by(|a, b| {
                    (contour_centroid(a) - c0)
                        .norm()
                        .partial_cmp(&(contour_centroid(b) - c0).norm())
                        .unwrap()
                }) {
                    let ao_centroid = contour_centroid(closest);
                    self.ao_lca = (ao_centroid.x, ao_centroid.y, ao_centroid.z);
                    self.lca_references = vessel_references(
                        ao_centroid,
                        &self.discretized_lca_main,
                        &self.lca_branches,
                    );
                }
            }
        }
        self
    }
}

fn vessel_references(
    ao_centroid: Vector3<f64>,
    main: &[Contour],
    side_branches: &[Vec<Contour>],
) -> Vec<ReferenceTriplet> {
    let main_centroids: Vec<Vector3<f64>> = main.iter().map(contour_centroid).collect();

    let up_hint = (main_centroids[0] - ao_centroid)
        .try_normalize(1e-12)
        .unwrap_or(Vector3::z());

    let mut tagged: Vec<(usize, ReferenceTriplet)> = Vec::new();

    if let Some(entry) = ostium_reference(ao_centroid, main, &main_centroids, up_hint) {
        tagged.push(entry);
    }
    for branch_contours in side_branches {
        if let Some(entry) =
            sidebranch_reference(ao_centroid, main, &main_centroids, branch_contours, up_hint)
        {
            tagged.push(entry);
        }
    }

    tagged.sort_by_key(|(k, _)| *k);
    tagged.into_iter().map(|(_, r)| r).collect()
}

fn ostium_reference(
    ao_centroid: Vector3<f64>,
    main: &[Contour],
    main_centroids: &[Vector3<f64>],
    up_hint: Vector3<f64>,
) -> Option<(usize, ReferenceTriplet)> {
    let first = main.first()?;
    if first.points.len() <= 2 {
        return None;
    }

    let normal = if main.len() > 1 {
        (main_centroids[1] - main_centroids[0])
            .try_normalize(1e-12)
            .unwrap_or(Vector3::z())
    } else {
        (main_centroids[0] - ao_centroid)
            .try_normalize(1e-12)
            .unwrap_or(Vector3::z())
    };

    let ((pa, pb), _) = first.find_closest_opposite_3d();
    let pta = Vector3::new(pa.x, pa.y, pa.z);
    let ptb = Vector3::new(pb.x, pb.y, pb.z);
    let main_ref_v = if (pta - ao_centroid).norm() <= (ptb - ao_centroid).norm() {
        pta
    } else {
        ptb
    };

    let ((p1, p2), _) = first.find_farthest_points();
    let (cc, cl) = assign_cc_clock(
        Vector3::new(p1.x, p1.y, p1.z),
        Vector3::new(p2.x, p2.y, p2.z),
        main_centroids[0],
        normal,
        up_hint,
    );

    Some((
        0,
        ReferenceTriplet {
            main_ref: (main_ref_v.x, main_ref_v.y, main_ref_v.z),
            counter_clock_ref: (cc.x, cc.y, cc.z),
            clock_ref: (cl.x, cl.y, cl.z),
        },
    ))
}

fn sidebranch_reference(
    ao_centroid: Vector3<f64>,
    main: &[Contour],
    main_centroids: &[Vector3<f64>],
    branch_contours: &[Contour],
    up_hint: Vector3<f64>,
) -> Option<(usize, ReferenceTriplet)> {
    if branch_contours.is_empty() {
        return None;
    }
    let side_c0 = contour_centroid(&branch_contours[0]);

    let (bifurc_idx, _) = main_centroids.iter().enumerate().min_by(|(_, a), (_, b)| {
        (*a - side_c0)
            .norm()
            .partial_cmp(&(*b - side_c0).norm())
            .unwrap()
    })?;

    let bifurc_centroid = main_centroids[bifurc_idx];

    let normal = if bifurc_idx + 1 < main.len() {
        (main_centroids[bifurc_idx + 1] - bifurc_centroid)
            .try_normalize(1e-12)
            .unwrap_or(Vector3::z())
    } else if bifurc_idx > 0 {
        (bifurc_centroid - main_centroids[bifurc_idx - 1])
            .try_normalize(1e-12)
            .unwrap_or(Vector3::z())
    } else {
        (bifurc_centroid - ao_centroid)
            .try_normalize(1e-12)
            .unwrap_or(Vector3::z())
    };

    let bifurc_contour = &main[bifurc_idx];
    let n_pts = bifurc_contour.points.len();
    if n_pts < 4 {
        return None;
    }

    let closest_idx = bifurc_contour
        .points
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (Vector3::new(a.x, a.y, a.z) - side_c0)
                .norm()
                .partial_cmp(&(Vector3::new(b.x, b.y, b.z) - side_c0).norm())
                .unwrap()
        })
        .map(|(i, _)| i)?;

    let quarter = n_pts / 4;
    let idx_plus = (closest_idx + quarter) % n_pts;
    let idx_minus = (closest_idx + n_pts - quarter) % n_pts;
    let pp = &bifurc_contour.points[idx_plus];
    let pm = &bifurc_contour.points[idx_minus];

    let (cc, cl) = assign_cc_clock(
        Vector3::new(pp.x, pp.y, pp.z),
        Vector3::new(pm.x, pm.y, pm.z),
        bifurc_centroid,
        normal,
        up_hint,
    );

    Some((
        bifurc_idx,
        ReferenceTriplet {
            main_ref: (side_c0.x, side_c0.y, side_c0.z),
            counter_clock_ref: (cc.x, cc.y, cc.z),
            clock_ref: (cl.x, cl.y, cl.z),
        },
    ))
}

/// Assign the two points to (counter_clock, clock) when viewing proximal→distal.
///
/// "Left = counter_clock" convention (user-specified):
/// - Compute an "up" direction perpendicular to `normal` from `up_hint`
/// - `right = up_perp × normal` (right-hand rule: up × forward = right)
/// - `p` is counter_clock (left) if its component along `right` is negative
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
