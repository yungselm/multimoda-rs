use crate::io::Geometry;
use crate::io::input::{ContourPoint, Contour, Record};
use crate::processing::contours::{align_frames_in_geometry, hausdorff_distance};

pub fn geometry_from_array_rs(
    contours: Vec<Contour>,
    reference_point: ContourPoint,
    steps: usize,
    range: f64,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    label: &str,
    records: Option<Vec<Record>>,
    delta: f64,
    max_rounds: usize,
    diastole: bool,
    sort: bool,
) -> Result<Geometry, Box<dyn std::error::Error>> {
    let mut contours = contours;
    let label = label.to_string();

    // Build catheter contours, propagating any errors
    let mut catheter = if n_points == 0 {
        Vec::new()
    } else {
        Contour::create_catheter_contours(
            &contours.iter().flat_map(|c| c.points.clone()).collect::<Vec<_>>(),
            image_center,
            radius,
            n_points,
        )?
    };

    // Initial geometry setup, possibly reordering by records
    let mut geometry = if let Some(recs) = records.as_ref() {
        let mut z_coords: Vec<f64> = contours.iter().map(|c| c.centroid.2).collect();
        z_coords.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        Geometry::reorder_contours(&mut contours, recs, diastole, &z_coords);
        catheter.sort_by_key(|c| c.id);

        Geometry {
            contours,
            catheter,
            reference_point,
            label: label.clone(),
        }
    } else {
        Geometry { contours, catheter, reference_point, label }
    };

    // Optionally align and refine ordering
    if sort {
        let aligned = align_frames_in_geometry(geometry, steps, range);
        geometry = refine_ordering(aligned, delta, max_rounds, steps, range);
    } else {
        geometry = align_frames_in_geometry(geometry, steps, range);
    }

    Ok(geometry)
}


fn refine_ordering(mut geom: Geometry, delta: f64, max_rounds: usize, steps: usize, range: f64) -> Geometry {
    let mut last_order = Vec::new();
    for _round in 0..max_rounds {
        let cost = build_cost_matrix(&geom.contours, delta);
        // Choose DP or heuristic based on geom.contours.len()
        let order = if geom.contours.len() <= 15 {
            held_karp(&cost)
        } else {
            two_opt(&cost, 500)
        };
        if order == last_order {
            break;  // converged
        }
        last_order = order.clone();
        geom = reorder_geometry(geom, &order);
        geom = align_frames_in_geometry(geom, steps, range)
    }
    geom
}

fn reorder_geometry(mut geom: Geometry, new_order: &[usize]) -> Geometry {
    let reordered_contours = new_order.iter()
        .map(|&i| geom.contours[i].clone())
        .collect();
    let reordered_catheter = new_order.iter()
        .map(|&i| geom.catheter[i].clone())
        .collect();
    geom.contours = reordered_contours;
    geom.catheter = reordered_catheter;
    geom
}

fn build_cost_matrix(contours: &[Contour], delta: f64) -> Vec<Vec<f64>> {
    let n = contours.len();
    let mut cost = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                cost[i][j] = 0.0;
            } else {
                let h = hausdorff_distance(&contours[i].points, &contours[j].points);
                let jump = (contours[i].id as i32 - contours[j].id as i32).abs() as f64;
                cost[i][j] = h + delta * jump;
            }
        }
    }
    cost
}

/// Solve the shortest Hamiltonian path from 0 through all nodes exactly.
fn held_karp(cost: &[Vec<f64>]) -> Vec<usize> {
    let n = cost.len();
    let full_mask = (1 << n) - 1;
    // dp[mask][j] = best cost to start at 0, visit mask, end at j
    let mut dp = vec![vec![f64::INFINITY; n]; 1 << n];
    let mut parent = vec![vec![None; n]; 1 << n];
    
    dp[1 << 0][0] = 0.0;
    for mask in 1..=full_mask {
        if (mask & 1) == 0 { continue } // must always include node 0
        for j in 0..n {
            if mask & (1 << j) == 0 { continue; }
            if j == 0 && mask != (1 << 0) { continue; }
            let prev_mask = mask ^ (1 << j);
            for i in 0..n {
                if prev_mask & (1 << i) == 0 { continue; }
                let cost_ij = cost[i][j];
                let cand = dp[prev_mask][i] + cost_ij;
                if cand < dp[mask][j] {
                    dp[mask][j] = cand;
                    parent[mask][j] = Some(i);
                }
            }
        }
    }
    // Pick endpoint with minimal cost
    let (mut end, _) = (1..n)
        .map(|j| (j, dp[full_mask][j]))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    // Reconstruct path
    let mut order = Vec::with_capacity(n);
    let mut mask = full_mask;
    while let Some(p) = parent[mask][end] {
        order.push(end);
        mask ^= 1 << end;
        end = p;
    }
    order.push(0);
    order.reverse();
    order
}

/// Start with identity ordering, then do repeated 2‑opt swaps
fn two_opt(cost: &[Vec<f64>], max_iters: usize) -> Vec<usize> {
    let n = cost.len();
    let mut order: Vec<usize> = (0..n).collect();
    
    let mut best_impr = true;
    for _ in 0..max_iters {
        if !best_impr { break; }
        best_impr = false;
        for a in 1..n - 1 {
            for b in a + 1..n {
                // compute delta cost of reversing order[a..=b]
                let i = order[a - 1];
                let j = order[a];
                let k = order[b];
                let l = if b + 1 < n { order[b + 1] } else { usize::MAX };
                
                let before = cost[i][j]
                           + if l != usize::MAX { cost[k][l] } else { 0.0 };
                let after  = cost[i][k]
                           + if l != usize::MAX { cost[j][l] } else { 0.0 };
                if after + 1e-9 < before {
                    order[a..=b].reverse();
                    best_impr = true;
                }
            }
        }
    }
    order
}
