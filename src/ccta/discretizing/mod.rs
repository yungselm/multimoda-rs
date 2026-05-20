pub mod creating;
pub mod data;
pub mod projecting;
mod utils;

use crate::ccta::discretizing::creating::create_uniform_contours;
use crate::ccta::discretizing::projecting::walk_centerline_slices;
use crate::intravascular::io::geometry::Contour;
use crate::intravascular::io::input::Centerline;

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
    let raw = walk_centerline_slices(centerline, points, branch_id, step_size);
    create_uniform_contours(raw, n_points)
}
