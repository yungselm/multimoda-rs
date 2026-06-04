use crate::types::native::{Centerline, Contour, DiscretizedVesselTree};
use anyhow::Result;

impl DiscretizedVesselTree {
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
        let discretized_aorta =
            super::discretize_vessel_rs(ao_cl, points_ao, 0, step_size, n_points);
        let discretized_rca_main = super::discretize_vessel_rs(
            rca_cl,
            points_rca_main,
            branch_id_rca,
            step_size,
            n_points,
        );
        let discretized_lca_main = super::discretize_vessel_rs(
            lca_cl,
            points_lca_main,
            branch_id_lca,
            step_size,
            n_points,
        );

        let rca_branches: Vec<Vec<Contour>> = side_branches_rca
            .iter()
            .enumerate()
            .map(|(i, pts)| {
                super::discretize_vessel_rs(rca_cl, pts, (i + 1) as u32, step_size, n_points)
            })
            .collect();

        let lca_branches: Vec<Vec<Contour>> = side_branches_lca
            .iter()
            .enumerate()
            .map(|(i, pts)| {
                super::discretize_vessel_rs(lca_cl, pts, (i + 1) as u32, step_size, n_points)
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
            ao_lca: (0.0, 0.0, 0.0),
            ao_rca: (0.0, 0.0, 0.0),
            pts_cusp_rcc: None,
            pts_cusp_lcc: None,
            pts_cusp_acc: None,
            index_stj_slice: None,
            index_aa: None,
        })
    }
}
