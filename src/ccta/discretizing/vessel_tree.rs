use super::{coord_key, SurfaceMesh};
use crate::types::native::{Centerline, Contour, DiscretizedVesselTree, Point3D};
use anyhow::{anyhow, Result};
use rayon::prelude::*;
use std::collections::HashMap;

type Coords3 = (f64, f64, f64);

impl DiscretizedVesselTree {
    /// Cuts `mesh` along every centerline branch, using only each vessel's own labelled faces
    /// (see `SurfaceMesh::region_faces`). `side_branches_rca[i]` is branch_id `i + 1`, i.e.
    /// `results["rca_points_side_{i + 1}"]`, likewise for the LCA. Shared side-branch labels are
    /// resolved first (see `exclusive_side_labels`). Fails if the aorta or a main vessel fails,
    /// a failing side branch is skipped with a warning.
    pub fn from_results_dict(
        ao_cl: &Centerline,
        rca_cl: &Centerline,
        lca_cl: &Centerline,
        mesh: &SurfaceMesh,
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
        let side_branches_rca = exclusive_side_labels(rca_cl, side_branches_rca);
        let side_branches_lca = exclusive_side_labels(lca_cl, side_branches_lca);
        let main = |name: &str, cl: &Centerline, points: &[Coords3], branch_id: u32| {
            super::discretize_vessel_rs(cl, mesh, Some(points), branch_id, step_size, n_points)
                .map_err(|e| anyhow!("{name}: {e}"))
        };
        let discretized_aorta = main("aorta", ao_cl, points_ao, 0)?;
        let discretized_rca_main = main("RCA main", rca_cl, points_rca_main, branch_id_rca)?;
        let discretized_lca_main = main("LCA main", lca_cl, points_lca_main, branch_id_lca)?;

        let rca_branches =
            side_branches("RCA", rca_cl, mesh, &side_branches_rca, step_size, n_points);
        let lca_branches =
            side_branches("LCA", lca_cl, mesh, &side_branches_lca, step_size, n_points);

        Ok(DiscretizedVesselTree {
            discretized_aorta,
            discretized_rca_main,
            discretized_lca_main,
            spacing: step_size,
            rca_branches,
            lca_branches,
            rca_references: vec![],
            lca_references: vec![],
            rca_branch_references: vec![],
            lca_branch_references: vec![],
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

/// Discretizes every side branch. A failing branch is skipped with a warning and left empty,
/// so indices stay aligned with branch ids.
fn side_branches(
    vessel: &str,
    centerline: &Centerline,
    mesh: &SurfaceMesh,
    sides: &[Vec<Coords3>],
    step_size: f64,
    n_points: usize,
) -> Vec<Vec<Contour>> {
    sides
        .par_iter()
        .enumerate()
        .map(|(i, pts)| {
            let branch_id = (i + 1) as u32;
            super::discretize_vessel_rs(centerline, mesh, Some(pts), branch_id, step_size, n_points)
                .unwrap_or_else(|e| {
                    eprintln!("Warning: skipping {vessel} side branch {branch_id}: {e}");
                    vec![]
                })
        })
        .collect()
}

/// Keeps each vertex shared by several side branches only for the one with the nearest
/// centerline. Otherwise a branch growing out of another claims its parent's wall and its first
/// slices cut the parent's tube. `sides[i]` is branch_id `i + 1`.
fn exclusive_side_labels(centerline: &Centerline, sides: Vec<Vec<Coords3>>) -> Vec<Vec<Coords3>> {
    let mut claims: HashMap<[u64; 3], Vec<usize>> = HashMap::new();
    for (i, pts) in sides.iter().enumerate() {
        for &(x, y, z) in pts {
            let owners = claims.entry(coord_key(x, y, z)).or_default();
            if owners.last() != Some(&i) {
                owners.push(i);
            }
        }
    }

    let dist_to_branch = |p: &Coords3, branch_id: u32| -> f64 {
        centerline
            .points
            .iter()
            .filter(|c| c.branch_id == branch_id)
            .map(|c| c.contour_point.distance_to(p))
            .fold(f64::INFINITY, f64::min)
    };
    let mut owner: HashMap<[u64; 3], usize> = HashMap::new();
    for pts in &sides {
        for p in pts {
            let key = coord_key(p.0, p.1, p.2);
            let candidates = &claims[&key];
            if candidates.len() > 1 && !owner.contains_key(&key) {
                let nearest = candidates
                    .iter()
                    .copied()
                    .min_by(|&a, &b| {
                        dist_to_branch(p, a as u32 + 1).total_cmp(&dist_to_branch(p, b as u32 + 1))
                    })
                    .unwrap_or(candidates[0]);
                owner.insert(key, nearest);
            }
        }
    }

    sides
        .into_iter()
        .enumerate()
        .map(|(i, pts)| {
            pts.into_iter()
                .filter(|p| owner.get(&coord_key(p.0, p.1, p.2)).is_none_or(|&o| o == i))
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::native::{CenterlinePoint, ContourPoint};
    use nalgebra::Vector3;

    fn cl_pt(branch_id: u32, x: f64, y: f64) -> CenterlinePoint {
        CenterlinePoint {
            contour_point: ContourPoint {
                frame_index: 0,
                point_index: 0,
                x,
                y,
                z: 0.0,
                aortic: false,
            },
            tangent: Vector3::x(),
            branch_id,
            radius: 0.0,
        }
    }

    #[test]
    fn test_shared_side_labels_go_to_nearest_branch() {
        // Branch 1 along x at y = 0, branch 2 along x at y = 10.
        let centerline = Centerline {
            points: (0..10)
                .map(|i| cl_pt(1, i as f64, 0.0))
                .chain((0..10).map(|i| cl_pt(2, i as f64, 10.0)))
                .collect(),
            branch_start_indices: vec![0, 10],
        };
        let near_1 = (3.0, 2.0, 0.0);
        let near_2 = (3.0, 8.0, 0.0);
        let only_1 = (5.0, -2.0, 0.0);
        let sides = vec![vec![near_1, near_2, only_1], vec![near_1, near_2]];
        let out = exclusive_side_labels(&centerline, sides);
        assert_eq!(out[0], vec![near_1, only_1]);
        assert_eq!(out[1], vec![near_2]);
    }
}
