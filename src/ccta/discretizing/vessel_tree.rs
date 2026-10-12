use super::SurfaceMesh;
use crate::types::native::{Centerline, Contour, DiscretizedBranch, DiscretizedVesselTree};
use anyhow::{anyhow, Result};
use rayon::prelude::*;
use std::collections::BTreeMap;

type Coords3 = (f64, f64, f64);

impl DiscretizedVesselTree {
    /// Cuts `mesh` along every labelled branch, using only each branch's own labelled faces
    /// (see `SurfaceMesh::region_faces`). `rca_branches` / `lca_branches` map branch ids to their
    /// vertices (`results["rca_branches"]`); every id must be a branch of its centerline. Each
    /// branch's parent comes from [`Centerline::branch_parents`]. Fails if the aorta or a main
    /// vessel (branch 0) fails; a failing side branch is left out with a warning.
    pub fn from_results_dict(
        ao_cl: &Centerline,
        rca_cl: &Centerline,
        lca_cl: &Centerline,
        mesh: &SurfaceMesh,
        points_ao: &[Coords3],
        rca_branches: &BTreeMap<u32, Vec<Coords3>>,
        lca_branches: &BTreeMap<u32, Vec<Coords3>>,
        step_size: f64,
        n_points: usize,
    ) -> Result<DiscretizedVesselTree> {
        let discretized_aorta =
            super::discretize_vessel_rs(ao_cl, mesh, Some(points_ao), 0, step_size, n_points)
                .map_err(|e| anyhow!("aorta: {e}"))?;
        let rca = vessel_branches("RCA", rca_cl, mesh, rca_branches, step_size, n_points)?;
        let lca = vessel_branches("LCA", lca_cl, mesh, lca_branches, step_size, n_points)?;

        Ok(DiscretizedVesselTree {
            discretized_aorta,
            spacing: step_size,
            rca,
            lca,
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

/// Discretizes every branch in `points` (by branch id). Branch 0 must succeed; a failing or
/// unlabelled side branch is left out, with a warning if it failed. Parents are set from the
/// centerline (junction contour 0 until `calculate_ref_pts`), skipping ancestors that are not
/// in the result.
fn vessel_branches(
    vessel: &str,
    centerline: &Centerline,
    mesh: &SurfaceMesh,
    points: &BTreeMap<u32, Vec<Coords3>>,
    step_size: f64,
    n_points: usize,
) -> Result<BTreeMap<u32, DiscretizedBranch>> {
    let known = centerline.branch_ids();
    if let Some(id) = points.keys().find(|id| !known.contains(id)) {
        return Err(anyhow!(
            "{vessel}: labelled branch {id} is not in the centerline (branches {known:?})"
        ));
    }
    if points.get(&0).is_none_or(Vec::is_empty) {
        return Err(anyhow!("{vessel} main: no vertices labelled for branch 0"));
    }

    let discretized: Vec<(u32, Result<Vec<Contour>, String>)> = points
        .par_iter()
        .filter(|(_, pts)| !pts.is_empty())
        .map(|(&id, pts)| {
            let contours =
                super::discretize_vessel_rs(centerline, mesh, Some(pts), id, step_size, n_points);
            (id, contours)
        })
        .collect();

    let mut branches = BTreeMap::new();
    for (id, contours) in discretized {
        match contours {
            Ok(contours) => {
                branches.insert(
                    id,
                    DiscretizedBranch {
                        contours,
                        ..Default::default()
                    },
                );
            }
            Err(e) if id == 0 => return Err(anyhow!("{vessel} main: {e}")),
            Err(e) => eprintln!("Warning: skipping {vessel} side branch {id}: {e}"),
        }
    }

    let parents = centerline.branch_parents();
    let present: Vec<u32> = branches.keys().copied().collect();
    for &id in &present {
        let parent = present_ancestor(&parents, id, |p| present.contains(&p));
        if let Some(branch) = branches.get_mut(&id) {
            branch.parent = parent.map(|p| (p, 0));
        }
    }
    Ok(branches)
}

/// Nearest ancestor of `id` in `parents` (indexed by branch id) for which `present` holds.
fn present_ancestor(
    parents: &[Option<u32>],
    id: u32,
    present: impl Fn(u32) -> bool,
) -> Option<u32> {
    let mut parent = parents.get(id as usize).copied().flatten();
    while let Some(p) = parent {
        if present(p) {
            return Some(p);
        }
        parent = parents.get(p as usize).copied().flatten();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_present_ancestor_skips_missing_parents() {
        // 3 -> 2 -> 1 -> 0, branch 2 missing (removed or failed).
        let parents = vec![None, Some(0), Some(1), Some(2)];
        let present = |p: u32| p != 2;
        assert_eq!(present_ancestor(&parents, 3, present), Some(1));
        assert_eq!(present_ancestor(&parents, 1, present), Some(0));
        assert_eq!(present_ancestor(&parents, 0, present), None);
    }
}
