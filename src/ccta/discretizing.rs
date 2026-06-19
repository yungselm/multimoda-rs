pub mod projecting;
pub mod resampling;
pub mod vessel_tree;

use crate::types::native::{Centerline, Contour};
use crate::types::utils;

/// Walk `branch_id` of `centerline` at uniform `step_size` intervals, project the supplied
/// mesh `points` onto each perpendicular cross-section, filter incomplete slices, and
/// resample each surviving slice to exactly `n_points` via a closed Catmull-Rom spline.
pub fn discretize_vessel_rs(
    centerline: &Centerline,
    points: &[(f64, f64, f64)],
    branch_id: u32,
    step_size: f64,
    n_points: usize,
) -> Vec<Contour> {
    const SMOOTH_SIGMA: f64 = 2.5;

    let cl_smooth = utils::smooth_centerline(centerline, SMOOTH_SIGMA);
    let raw = projecting::walk_centerline_slices(&cl_smooth, points, branch_id, step_size);
    resampling::create_uniform_contours(raw, n_points)
}
