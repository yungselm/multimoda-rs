use crate::types::native::contour::{Contour, ContourType};
use crate::types::native::geometry::Geometry;
use crate::types::native::ContourPoint;
use rayon::prelude::*;
use std::f64::consts::PI;

pub fn extract_contours_by_type(geometry: &Geometry, contour_type: ContourType) -> Vec<Contour> {
    match contour_type {
        ContourType::Lumen => geometry
            .frames
            .iter()
            .map(|frame| frame.lumen.clone())
            .collect(),
        _ => geometry
            .frames
            .iter()
            .filter_map(|frame| frame.extras.get(&contour_type).cloned())
            .collect(),
    }
}

pub fn get_contour_type_name(contour_type: ContourType) -> &'static str {
    match contour_type {
        ContourType::Lumen => "lumen",
        ContourType::Eem => "eem",
        ContourType::Calcification => "calcification",
        ContourType::Sidebranch => "sidebranch",
        ContourType::Catheter => "catheter",
        ContourType::Wall => "wall",
    }
}

pub fn search_range<F>(
    cost_fn: F,
    step_deg: f64,
    range_deg: f64,
    center_angle: Option<f64>,
    limes_deg: f64,
) -> f64
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    let range_rad = range_deg.to_radians();
    let step_rad = step_deg.to_radians();

    // Handle edge case: zero or negative step
    if step_rad <= 0.0 {
        return center_angle.unwrap_or(0.0);
    }

    let center = center_angle.unwrap_or(0.0);
    let limes = limes_deg.to_radians();

    let start_angle = (center - range_rad).max(-limes);
    let stop_angle = (center + range_rad).min(limes);

    if stop_angle <= start_angle {
        return center;
    }

    let steps = (((stop_angle - start_angle) / step_rad).ceil() as usize).max(1);

    let angles: Vec<f64> = (0..=steps)
        .map(|i| start_angle + (i as f64) * step_rad)
        .take_while(|&a| a <= stop_angle)
        .map(|a| ((a + PI).rem_euclid(2.0 * PI)) - PI)
        .collect();

    angles
        .par_iter()
        .map(|&angle| (angle, cost_fn(angle)))
        .reduce_with(|a, b| if b.1 < a.1 { b } else { a })
        .map(|(angle, _)| angle)
        .unwrap_or(center)
}

/// Computes the **in-plane (xy)** Hausdorff distance between two point sets.
///
/// The z coordinate is deliberately ignored: every caller of this function scores
/// candidate in-plane rotations of one frame's contour against another frame's
/// contour, and those frames sit at different z. Including z would add a constant
/// offset that swamps the shape term the rotation search is trying to minimise.
///
/// For genuinely three-dimensional comparisons (e.g. a whole geometry against a
/// point cloud) use [`hausdorff_sq_3d_grid`] instead.
pub fn hausdorff_distance(set1: &[ContourPoint], set2: &[ContourPoint]) -> f64 {
    let forward = directed_hausdorff(set1, set2);
    let backward = directed_hausdorff(set2, set1);
    forward.max(backward)
}

/// Bare xyz coordinates, for the distance kernels.
///
/// A [`ContourPoint`] is 40 bytes, of which the inner loop reads 24. Packing the
/// coordinates cuts the memory traffic of the O(n·m) scan accordingly and lets the
/// compiler vectorise it.
pub type Xyz = [f64; 3];

/// Strips a contour point slice down to bare coordinates.
pub fn to_xyz(points: &[ContourPoint]) -> Vec<Xyz> {
    points.iter().map(|p| [p.x, p.y, p.z]).collect()
}

/// A uniform 3D bucket grid over a point set, for nearest-neighbour queries.
///
/// Brute-force Hausdorff is O(n·m), which dominates any alignment search over many
/// candidates. Bucketing one side turns each nearest-neighbour lookup into a scan of
/// a handful of nearby cells, taking the pair cost to roughly O(n + m).
///
/// Points are stored reordered by cell (a counting sort) so that a cell's points are
/// contiguous in memory.
pub struct SpatialGrid {
    cell: f64,
    inv_cell: f64,
    min: [f64; 3],
    dims: [usize; 3],
    /// CSR offsets: cell `c` owns `points[cell_start[c]..cell_start[c + 1]]`.
    cell_start: Vec<u32>,
    points: Vec<Xyz>,
}

impl SpatialGrid {
    /// Buckets `points`, choosing a cell size that averages a few points per cell.
    pub fn build(points: &[Xyz]) -> Self {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for p in points {
            for axis in 0..3 {
                min[axis] = min[axis].min(p[axis]);
                max[axis] = max[axis].max(p[axis]);
            }
        }
        if points.is_empty() {
            min = [0.0; 3];
            max = [0.0; 3];
        }

        let extent = [
            (max[0] - min[0]).max(1e-9),
            (max[1] - min[1]).max(1e-9),
            (max[2] - min[2]).max(1e-9),
        ];

        // Target a handful of points per occupied cell. Contours are surfaces, so
        // the occupied fraction of the bounding box is well under 1 and the true
        // occupancy runs higher than this estimate — which is the safe direction.
        let volume = extent[0] * extent[1] * extent[2];
        let n = points.len().max(1) as f64;
        let mut cell = (volume / n).cbrt().max(1e-6);

        // Keep the cell count bounded regardless of how degenerate the extent is.
        let max_cells = (8 * points.len().max(1)).min(1 << 22) as f64;
        loop {
            let dims: Vec<f64> = (0..3).map(|a| (extent[a] / cell).ceil().max(1.0)).collect();
            if dims[0] * dims[1] * dims[2] <= max_cells {
                break;
            }
            cell *= 1.5;
        }

        let dims = [
            (extent[0] / cell).ceil().max(1.0) as usize,
            (extent[1] / cell).ceil().max(1.0) as usize,
            (extent[2] / cell).ceil().max(1.0) as usize,
        ];
        let inv_cell = 1.0 / cell;
        let n_cells = dims[0] * dims[1] * dims[2];

        let cell_of = |p: &Xyz| -> usize {
            let ix = (((p[0] - min[0]) * inv_cell) as usize).min(dims[0] - 1);
            let iy = (((p[1] - min[1]) * inv_cell) as usize).min(dims[1] - 1);
            let iz = (((p[2] - min[2]) * inv_cell) as usize).min(dims[2] - 1);
            (iz * dims[1] + iy) * dims[0] + ix
        };

        // Counting sort of the points into their cells.
        let mut cell_start = vec![0u32; n_cells + 1];
        for p in points {
            cell_start[cell_of(p) + 1] += 1;
        }
        for c in 0..n_cells {
            cell_start[c + 1] += cell_start[c];
        }
        let mut cursor = cell_start.clone();
        let mut sorted = vec![[0.0; 3]; points.len()];
        for p in points {
            let c = cell_of(p);
            sorted[cursor[c] as usize] = *p;
            cursor[c] += 1;
        }

        SpatialGrid {
            cell,
            inv_cell,
            min,
            dims,
            cell_start,
            points: sorted,
        }
    }

    /// Squared distance from `q` to the nearest stored point.
    ///
    /// Returns a value `v` that is either exactly the squared nearest distance, or —
    /// when the search can prove the answer is no greater than `floor_sq` — some
    /// `v <= floor_sq`. That is precisely the contract a max-of-mins needs: a point
    /// whose nearest neighbour is already within the running maximum cannot raise it,
    /// so the exact value is not worth finding. Pass `0.0` for an always-exact query.
    pub fn nearest_sq(&self, q: &Xyz, floor_sq: f64) -> f64 {
        if self.points.is_empty() {
            return f64::INFINITY;
        }

        // Cell containing q, clamped into the grid (q may lie outside the bounds).
        let mut base = [0isize; 3];
        for axis in 0..3 {
            let raw = ((q[axis] - self.min[axis]) * self.inv_cell).floor();
            base[axis] = raw.clamp(0.0, (self.dims[axis] - 1) as f64) as isize;
        }

        let mut best_sq = f64::INFINITY;
        let mut radius = 0isize;

        loop {
            // Scan the Chebyshev shell at `radius`, skipping the already-scanned interior.
            let lo = [
                (base[0] - radius).max(0),
                (base[1] - radius).max(0),
                (base[2] - radius).max(0),
            ];
            let hi = [
                (base[0] + radius).min(self.dims[0] as isize - 1),
                (base[1] + radius).min(self.dims[1] as isize - 1),
                (base[2] + radius).min(self.dims[2] as isize - 1),
            ];

            for iz in lo[2]..=hi[2] {
                for iy in lo[1]..=hi[1] {
                    // A cell is scanned at the radius equal to its Chebyshev distance
                    // from `base`. If y or z already achieves `radius`, the whole x row
                    // is new; otherwise only the two x faces are.
                    let yz_achieves_radius = radius == 0
                        || (iz - base[2]).abs() == radius
                        || (iy - base[1]).abs() == radius;

                    if yz_achieves_radius {
                        for ix in lo[0]..=hi[0] {
                            self.scan_cell(ix, iy, iz, q, &mut best_sq);
                        }
                    } else {
                        for &ix in &[base[0] - radius, base[0] + radius] {
                            if ix >= lo[0] && ix <= hi[0] {
                                self.scan_cell(ix, iy, iz, q, &mut best_sq);
                            }
                        }
                    }
                }
            }

            if best_sq <= floor_sq {
                return best_sq;
            }

            // Smallest distance from q to anything outside the scanned box. Faces that
            // sit on the grid boundary have nothing beyond them, so they do not limit us.
            let mut outside_dist = f64::INFINITY;
            let mut any_open = false;
            for axis in 0..3 {
                if base[axis] - radius > 0 {
                    any_open = true;
                    let face = self.min[axis] + ((base[axis] - radius) as f64) * self.cell;
                    outside_dist = outside_dist.min((q[axis] - face).abs());
                }
                if base[axis] + radius < self.dims[axis] as isize - 1 {
                    any_open = true;
                    let face = self.min[axis] + ((base[axis] + radius + 1) as f64) * self.cell;
                    outside_dist = outside_dist.min((face - q[axis]).abs());
                }
            }

            if !any_open {
                return best_sq; // Whole grid scanned.
            }
            if best_sq <= outside_dist * outside_dist {
                return best_sq;
            }

            radius += 1;
        }
    }

    #[inline]
    fn scan_cell(&self, ix: isize, iy: isize, iz: isize, q: &Xyz, best_sq: &mut f64) {
        let c = ((iz as usize) * self.dims[1] + iy as usize) * self.dims[0] + ix as usize;
        let (start, end) = (self.cell_start[c] as usize, self.cell_start[c + 1] as usize);
        for p in &self.points[start..end] {
            let dx = q[0] - p[0];
            let dy = q[1] - p[1];
            let dz = q[2] - p[2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 < *best_sq {
                *best_sq = d2;
            }
        }
    }
}

/// Squared 3D Hausdorff distance using prebuilt grids, abandoned early once it
/// provably exceeds `bound_sq`.
///
/// O(n + m) rather than the O(n·m) of a brute-force scan.
/// Supply the grid built over each set; when one side is reused across many
/// candidates (as in an alignment search) its grid should be built once and shared.
pub fn hausdorff_sq_3d_grid(
    set1: &[Xyz],
    grid1: &SpatialGrid,
    set2: &[Xyz],
    grid2: &SpatialGrid,
    bound_sq: f64,
) -> Option<f64> {
    if set1.is_empty() || set2.is_empty() {
        return Some(0.0);
    }
    let forward = directed_hausdorff_sq_grid(set1, grid2, bound_sq)?;
    let backward = directed_hausdorff_sq_grid(set2, grid1, bound_sq)?;
    Some(forward.max(backward))
}

fn directed_hausdorff_sq_grid(a: &[Xyz], b: &SpatialGrid, bound_sq: f64) -> Option<f64> {
    let threads = rayon::current_num_threads().max(1);
    let chunk_size = a.len().div_ceil(threads * 4).max(1);

    a.par_chunks(chunk_size)
        .map(|chunk| {
            let mut local_max_sq = 0.0_f64;
            for pa in chunk {
                // Points already within the running maximum need no exact answer.
                let min_sq = b.nearest_sq(pa, local_max_sq);
                if min_sq > bound_sq {
                    return None;
                }
                if min_sq > local_max_sq {
                    local_max_sq = min_sq;
                }
            }
            Some(local_max_sq)
        })
        .try_reduce(|| 0.0_f64, |x, y| Some(x.max(y)))
}

fn directed_hausdorff(contour_a: &[ContourPoint], contour_b: &[ContourPoint]) -> f64 {
    // Keep behavior simple for empty inputs (match prior behavior -> 0.0)
    if contour_a.is_empty() || contour_b.is_empty() {
        return 0.0;
    }

    // Decide chunk size based on number of threads to create many tasks but not too many
    let threads = rayon::current_num_threads().max(1);
    // make several chunks per thread for load balancing
    let chunks_per_thread = 4;
    let chunk_size = contour_a.len().div_ceil(threads * chunks_per_thread).max(1);

    // For each chunk, compute the local maximum of the minimum squared distances
    let max_sq = contour_a
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut local_max_sq = 0.0_f64;
            for pa in chunk {
                // find min squared distance from pa to any pb (sequential inside chunk)
                let mut min_sq = f64::INFINITY;
                for pb in contour_b.iter() {
                    let dx = pa.x - pb.x;
                    let dy = pa.y - pb.y;
                    let d2 = dx * dx + dy * dy;
                    if d2 < min_sq {
                        min_sq = d2;
                    }
                }
                if min_sq.is_finite() && min_sq > local_max_sq {
                    local_max_sq = min_sq;
                }
            }
            local_max_sq
        })
        .reduce(|| 0.0_f64, f64::max);

    max_sq.sqrt()
}

// TODO: Move the interpolation to process_utils

#[cfg(test)]
mod process_utils_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_search_range_quadratic_function() {
        // Test with a simple quadratic cost function: (angle - 0.5)^2
        let cost_fn = |angle: f64| (angle - 0.5).powi(2);

        let result = search_range(cost_fn, 1.0, 180.0, None, 180.0);

        // Should find minimum near 0.5 radians
        assert_relative_eq!(result, 0.5, epsilon = 1.0_f64.to_radians());
    }

    #[test]
    fn test_search_range_with_center_angle() {
        // Test with center angle provided
        let cost_fn = |angle: f64| (angle - 1.0).powi(2);

        let result = search_range(cost_fn, 0.5, 45.0, Some(0.8), 180.0);

        // Should find minimum near 1.0 radians, but constrained by center and range
        assert_relative_eq!(result, 1.0, epsilon = 0.5_f64.to_radians());
    }

    #[test]
    fn test_search_range_sine_function() {
        // Test with sine function - multiple minima, but should find one in range
        let cost_fn = |angle: f64| angle.sin();

        let result = search_range(cost_fn, 1.0, 90.0, None, 180.0);

        // Sine is minimized at -π/2, -5π/2, etc. Within [-π, π], minimum is -π/2 ≈ -1.57
        assert!(result <= 0.0); // Should find a negative value where sine is minimal
    }

    #[test]
    fn test_search_range_edge_cases() {
        // Test with zero step - should return center angle
        let cost_fn = |_angle: f64| 1.0;
        let result = search_range(cost_fn, 0.0, 90.0, Some(1.0), 180.0);
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);

        // Test with very small range
        let cost_fn = |angle: f64| (angle - 0.1).powi(2);
        let result = search_range(cost_fn, 1.0, 1.0, Some(0.0), 180.0);
        // With small range (1°) around center 0.0, we can't reach 0.1 rad (~5.7°)
        // So should find minimum at the boundary closest to 0.1
        let expected_min = 1.0_f64.to_radians(); // 1° in radians ≈ 0.01745
        assert_relative_eq!(result, expected_min, epsilon = 0.5_f64.to_radians());

        // Test with range beyond limes
        let cost_fn = |angle: f64| (angle - 2.0).powi(2); // 2.0 rad ≈ 114.6°
        let result = search_range(cost_fn, 1.0, 180.0, None, 90.0); // limes = 90° ≈ 1.57 rad
                                                                    // Should be constrained by limes, so minimum at limes boundary (1.57)
        assert_relative_eq!(result, 1.57, epsilon = 1.0_f64.to_radians());

        // Test with negative step - should return center angle
        let result = search_range(cost_fn, -1.0, 90.0, Some(0.5), 180.0);
        assert_relative_eq!(result, 0.5, epsilon = 1e-10);

        // Test with no center angle provided
        let cost_fn = |angle: f64| (angle - 0.5).powi(2);
        let result = search_range(cost_fn, 0.0, 90.0, None, 180.0);
        assert_relative_eq!(result, 0.0, epsilon = 1e-10); // Should return default 0.0
    }

    #[test]
    fn test_search_range_small_range() {
        // Test with range too small to reach the true minimum
        let cost_fn = |angle: f64| (angle - 0.5).powi(2); // Minimum at 0.5 rad

        // Search with small range around 0.0 that doesn't include 0.5
        // Range: 0.2° ≈ 0.00349 rad, step: 0.1° ≈ 0.001745 rad
        let result = search_range(cost_fn, 0.1, 0.2, Some(0.0), 180.0);

        // Should find the best angle within the search range [0.0-0.00349, 0.0+0.00349]
        // The best in this range is the upper boundary 0.00349 rad
        let expected = 0.2_f64.to_radians(); // 0.0034906585 rad
        assert_relative_eq!(result, expected, epsilon = 0.1_f64.to_radians());

        // Test with range that exactly includes the minimum
        let result = search_range(cost_fn, 0.1, 30.0, Some(0.0), 180.0); // 30° ≈ 0.5236 rad
                                                                         // Should find something close to 0.5 (within the step size)
        assert_relative_eq!(result, 0.5, epsilon = 0.1_f64.to_radians());
    }

    #[test]
    fn test_hausdorff_distance_identical_sets() {
        let points = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 2,
                x: 0.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let distance = hausdorff_distance(&points, &points);
        assert_relative_eq!(distance, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hausdorff_distance_shifted_sets() {
        let set1 = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let set2 = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 2.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 3.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let distance = hausdorff_distance(&set1, &set2);
        // The farthest points are (0,0) to (3,0) = 3.0, and (1,0) to (2,0) = 1.0
        // But Hausdorff takes max of directed distances, so should be 2.0
        // (from (0,0) to nearest in set2 is (2,0) = 2.0, from (1,0) to nearest is (2,0) = 1.0)
        // and similarly from set2 to set1 gives 2.0
        assert_relative_eq!(distance, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hausdorff_distance_different_sizes() {
        let set1 = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 3.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let set2 = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 2.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 2,
                x: 4.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let distance = hausdorff_distance(&set1, &set2);
        // set1 to set2:
        //   (0,0) -> nearest is (1,0) = 1.0
        //   (3,0) -> nearest is (2,0) = 1.0
        // set2 to set1:
        //   (1,0) -> nearest is (0,0) = 1.0
        //   (2,0) -> nearest is (3,0) = 1.0
        //   (4,0) -> nearest is (3,0) = 1.0
        // So Hausdorff distance should be 1.0
        assert_relative_eq!(distance, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hausdorff_distance_empty_sets() {
        let empty: Vec<ContourPoint> = Vec::new();
        let points = vec![ContourPoint {
            frame_index: 1,
            point_index: 0,
            x: 1.0,
            y: 1.0,
            z: 0.0,
            aortic: false,
        }];

        // Empty set to non-empty should return 0.0 (as per current implementation)
        let distance1 = hausdorff_distance(&empty, &points);
        assert_relative_eq!(distance1, 0.0, epsilon = 1e-10);

        let distance2 = hausdorff_distance(&points, &empty);
        assert_relative_eq!(distance2, 0.0, epsilon = 1e-10);

        let distance3 = hausdorff_distance(&empty, &empty);
        assert_relative_eq!(distance3, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hausdorff_distance_complex_shapes() {
        // Create a square and a diamond that partially overlap
        let square = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 2.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 2,
                x: 2.0,
                y: 2.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 3,
                x: 0.0,
                y: 2.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let diamond = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 2.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 2,
                x: 1.0,
                y: 2.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 3,
                x: 0.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let distance = hausdorff_distance(&square, &diamond);

        // The farthest points should be from square corners to diamond
        // Let's verify it's a reasonable value
        assert!(distance > 0.0);
        assert!(distance < 2.0); // Should be less than the diagonal
    }

    #[test]
    fn test_directed_hausdorff_consistency() {
        // Test that the directed Hausdorff is consistent with the full Hausdorff
        let set1 = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let set2 = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 2.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 3.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let full_distance = hausdorff_distance(&set1, &set2);
        let directed_1_to_2 = directed_hausdorff(&set1, &set2);
        let directed_2_to_1 = directed_hausdorff(&set2, &set1);

        // Hausdorff distance should be the maximum of the two directed distances
        assert_relative_eq!(
            full_distance,
            directed_1_to_2.max(directed_2_to_1),
            epsilon = 1e-10
        );

        // For this case, both directed distances should be 2.0
        assert_relative_eq!(directed_1_to_2, 2.0, epsilon = 1e-10);
        assert_relative_eq!(directed_2_to_1, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_performance_hausdorff_large_sets() {
        // Create larger point sets to test performance
        let mut set1 = Vec::new();
        let mut set2 = Vec::new();

        for i in 0..100 {
            set1.push(ContourPoint {
                frame_index: 1,
                point_index: i,
                x: i as f64,
                y: 0.0,
                z: 0.0,
                aortic: false,
            });

            set2.push(ContourPoint {
                frame_index: 2,
                point_index: i,
                x: i as f64 + 0.5,
                y: 0.0,
                z: 0.0,
                aortic: false,
            });
        }

        // This should complete quickly with the parallel implementation
        let distance = hausdorff_distance(&set1, &set2);

        // Distance should be 0.5 (the constant offset)
        assert_relative_eq!(distance, 0.5, epsilon = 1e-10);
    }

    /// Straightforward O(n·m) 3D reference to validate the bounded kernel against.
    fn brute_force_hausdorff_3d(a: &[Xyz], b: &[Xyz]) -> f64 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        let directed = |from: &[Xyz], to: &[Xyz]| {
            from.iter()
                .map(|p| {
                    to.iter()
                        .map(|q| {
                            (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)
                        })
                        .fold(f64::INFINITY, f64::min)
                })
                .fold(0.0_f64, f64::max)
        };
        directed(a, b).max(directed(b, a))
    }

    /// Deterministic pseudo-random cloud (no rand dependency in the test path).
    fn pseudo_random_cloud(n: usize, seed: u64) -> Vec<Xyz> {
        let mut state = seed | 1;
        let mut next = || {
            // xorshift64*
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| [next() * 20.0, next() * 20.0, next() * 60.0])
            .collect()
    }

    #[test]
    fn test_hausdorff_3d_matches_brute_force() {
        let a = pseudo_random_cloud(300, 12345);
        let b = pseudo_random_cloud(250, 67890);

        let (ga, gb) = (SpatialGrid::build(&a), SpatialGrid::build(&b));
        let expected = brute_force_hausdorff_3d(&a, &b);
        let actual = hausdorff_sq_3d_grid(&a, &ga, &b, &gb, f64::MAX)
            .expect("unbounded must return a value");

        assert_relative_eq!(actual, expected, epsilon = 1e-9);
    }

    #[test]
    fn test_hausdorff_3d_is_symmetric() {
        let a = pseudo_random_cloud(200, 11);
        let b = pseudo_random_cloud(150, 22);

        let (ga, gb) = (SpatialGrid::build(&a), SpatialGrid::build(&b));
        let forward = hausdorff_sq_3d_grid(&a, &ga, &b, &gb, f64::MAX).unwrap();
        let backward = hausdorff_sq_3d_grid(&b, &gb, &a, &ga, f64::MAX).unwrap();

        assert_relative_eq!(forward, backward, epsilon = 1e-12);
    }

    #[test]
    fn test_hausdorff_3d_bound_prunes_without_changing_the_winner() {
        let a = pseudo_random_cloud(300, 999);
        let b = pseudo_random_cloud(300, 1000);
        let (ga, gb) = (SpatialGrid::build(&a), SpatialGrid::build(&b));
        let truth = brute_force_hausdorff_3d(&a, &b);

        // A bound at or above the true value must return the exact value.
        assert_relative_eq!(
            hausdorff_sq_3d_grid(&a, &ga, &b, &gb, truth).unwrap(),
            truth,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            hausdorff_sq_3d_grid(&a, &ga, &b, &gb, truth * 2.0).unwrap(),
            truth,
            epsilon = 1e-9
        );

        // A bound below it must prune.
        assert!(hausdorff_sq_3d_grid(&a, &ga, &b, &gb, truth * 0.5).is_none());
        assert!(hausdorff_sq_3d_grid(&a, &ga, &b, &gb, 0.0).is_none());
    }

    #[test]
    fn test_hausdorff_3d_accounts_for_z() {
        // Two identical squares separated purely in z. The 2D kernel must see them
        // as coincident; the 3D kernel must report the separation.
        let flat: Vec<ContourPoint> = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            .iter()
            .enumerate()
            .map(|(i, &(x, y))| ContourPoint {
                frame_index: 0,
                point_index: i as u32,
                x,
                y,
                z: 0.0,
                aortic: false,
            })
            .collect();
        let raised: Vec<ContourPoint> =
            flat.iter().map(|p| ContourPoint { z: 7.0, ..*p }).collect();

        assert_relative_eq!(hausdorff_distance(&flat, &raised), 0.0, epsilon = 1e-12);

        let (fxyz, rxyz) = (to_xyz(&flat), to_xyz(&raised));
        let (gf, gr) = (SpatialGrid::build(&fxyz), SpatialGrid::build(&rxyz));
        let distance_3d = hausdorff_sq_3d_grid(&fxyz, &gf, &rxyz, &gr, f64::MAX)
            .unwrap()
            .sqrt();
        assert_relative_eq!(distance_3d, 7.0, epsilon = 1e-12);
    }

    #[test]
    fn test_grid_nearest_matches_brute_force() {
        let cloud = pseudo_random_cloud(500, 31337);
        let grid = SpatialGrid::build(&cloud);

        // Queries inside the cloud, and well outside it on every side.
        let mut queries = pseudo_random_cloud(200, 424242);
        queries.extend([
            [-50.0, -50.0, -50.0],
            [100.0, 100.0, 100.0],
            [10.0, 10.0, -80.0],
            [-80.0, 10.0, 30.0],
            [0.0, 0.0, 0.0],
        ]);

        for q in &queries {
            let expected = cloud
                .iter()
                .map(|p| (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2) + (q[2] - p[2]).powi(2))
                .fold(f64::INFINITY, f64::min);
            // floor_sq = 0.0 forces an exact answer.
            assert_relative_eq!(grid.nearest_sq(q, 0.0), expected, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_grid_nearest_respects_floor() {
        let cloud = pseudo_random_cloud(400, 5150);
        let grid = SpatialGrid::build(&cloud);

        for q in &pseudo_random_cloud(100, 6161) {
            let exact = grid.nearest_sq(q, 0.0);
            // With a generous floor the query may bail out early, but never above it
            // and never below the true value.
            let approx = grid.nearest_sq(q, exact * 4.0 + 1.0);
            assert!(approx >= exact - 1e-9, "{approx} < {exact}");
            assert!(approx <= exact * 4.0 + 1.0 + 1e-9);
        }
    }

    #[test]
    fn test_grid_handles_degenerate_clouds() {
        // Single point, and a set collapsed onto a line — both give zero extent on
        // at least one axis, which the cell sizing has to survive.
        for cloud in [
            vec![[1.0, 2.0, 3.0]],
            (0..50).map(|i| [0.0, 0.0, i as f64]).collect::<Vec<_>>(),
            vec![[4.0, 4.0, 4.0]; 20],
        ] {
            let grid = SpatialGrid::build(&cloud);
            let q = [1.0, 1.0, 1.0];
            let expected = cloud
                .iter()
                .map(|p| (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2) + (q[2] - p[2]).powi(2))
                .fold(f64::INFINITY, f64::min);
            assert_relative_eq!(grid.nearest_sq(&q, 0.0), expected, epsilon = 1e-9);
        }

        // An empty grid has no nearest neighbour at all.
        assert_eq!(
            SpatialGrid::build(&[]).nearest_sq(&[0.0; 3], 0.0),
            f64::INFINITY
        );
    }

    #[test]
    fn test_hausdorff_3d_grid_matches_brute_force() {
        let a = pseudo_random_cloud(400, 2024);
        let b = pseudo_random_cloud(350, 2025);
        let (ga, gb) = (SpatialGrid::build(&a), SpatialGrid::build(&b));

        let expected = brute_force_hausdorff_3d(&a, &b);
        let actual = hausdorff_sq_3d_grid(&a, &ga, &b, &gb, f64::MAX).unwrap();

        assert_relative_eq!(actual, expected, epsilon = 1e-9);
    }

    #[test]
    fn test_hausdorff_3d_grid_bound_prunes() {
        let a = pseudo_random_cloud(300, 88);
        let b = pseudo_random_cloud(300, 99);
        let (ga, gb) = (SpatialGrid::build(&a), SpatialGrid::build(&b));
        let truth = brute_force_hausdorff_3d(&a, &b);

        assert_relative_eq!(
            hausdorff_sq_3d_grid(&a, &ga, &b, &gb, truth).unwrap(),
            truth,
            epsilon = 1e-9
        );
        assert!(hausdorff_sq_3d_grid(&a, &ga, &b, &gb, truth * 0.5).is_none());
    }

    #[test]
    fn test_hausdorff_3d_grid_disjoint_clouds() {
        // Far-apart clouds: every query lands outside the other grid's bounds, which
        // exercises the clamped-cell and boundary-face logic.
        let a: Vec<Xyz> = (0..100).map(|i| [i as f64 * 0.1, 0.0, 0.0]).collect();
        let b: Vec<Xyz> = (0..100).map(|i| [i as f64 * 0.1, 0.0, 500.0]).collect();
        let (ga, gb) = (SpatialGrid::build(&a), SpatialGrid::build(&b));

        let actual = hausdorff_sq_3d_grid(&a, &ga, &b, &gb, f64::MAX).unwrap();
        assert_relative_eq!(actual, brute_force_hausdorff_3d(&a, &b), epsilon = 1e-9);
        assert_relative_eq!(actual.sqrt(), 500.0, epsilon = 1e-9);
    }

    #[test]
    fn test_hausdorff_3d_empty_sets() {
        let points = pseudo_random_cloud(10, 7);
        let empty: Vec<Xyz> = Vec::new();
        let (gp, ge) = (SpatialGrid::build(&points), SpatialGrid::build(&empty));

        assert_eq!(
            hausdorff_sq_3d_grid(&empty, &ge, &points, &gp, f64::MAX),
            Some(0.0)
        );
        assert_eq!(
            hausdorff_sq_3d_grid(&points, &gp, &empty, &ge, f64::MAX),
            Some(0.0)
        );
        assert_eq!(
            hausdorff_sq_3d_grid(&empty, &ge, &empty, &ge, f64::MAX),
            Some(0.0)
        );
    }
}
