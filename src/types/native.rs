pub mod centerline;
pub mod centerline_point;
pub mod contour;
pub mod contour_point;
pub mod discretized_tree;
pub mod frame;
pub mod geometry;
pub mod geometry_pair;
pub mod record;
pub mod structure;

pub use centerline::Centerline;
pub use centerline_point::CenterlinePoint;
pub use contour::{downsample_contour_points, Contour, ContourType};
pub use contour_point::ContourPoint;
pub use discretized_tree::{DiscretizedBranch, DiscretizedVesselTree, ReferenceTriplet};
pub use frame::Frame;
pub use geometry::Geometry;
pub use geometry_pair::GeometryPair;
pub use record::Record;
pub use structure::{CloudParams, Structure, StructureKind};

pub trait Point3D {
    fn x(&self) -> f64;
    fn y(&self) -> f64;
    fn z(&self) -> f64;

    /// Computes the Euclidean 3-D distance to another point.
    fn distance_to(&self, other: &impl Point3D) -> f64 {
        let dx = self.x() - other.x();
        let dy = self.y() - other.y();
        let dz = self.z() - other.z();
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Computes the 2-D (XY-plane) distance to another point.
    fn distance_2d_to(&self, other: &impl Point3D) -> f64 {
        let dx = self.x() - other.x();
        let dy = self.y() - other.y();
        (dx * dx + dy * dy).sqrt()
    }
}

impl<P: Point3D + ?Sized> Point3D for &P {
    fn x(&self) -> f64 {
        (**self).x()
    }
    fn y(&self) -> f64 {
        (**self).y()
    }
    fn z(&self) -> f64 {
        (**self).z()
    }
}

impl Point3D for nalgebra::Vector3<f64> {
    fn x(&self) -> f64 {
        self[0]
    }
    fn y(&self) -> f64 {
        self[1]
    }
    fn z(&self) -> f64 {
        self[2]
    }
}

/// Cumulative arc length along the polyline `points`, starting at 0.
///
/// Returns one entry per point (empty for empty input); the last entry is the
/// total polyline length.
pub fn cumulative_arc_length<P: Point3D>(points: &[P]) -> Vec<f64> {
    if points.is_empty() {
        return Vec::new();
    }
    std::iter::once(0.0)
        .chain(points.windows(2).scan(0.0, |acc, w| {
            *acc += w[0].distance_to(&w[1]);
            Some(*acc)
        }))
        .collect()
}

/// Median of `values`, the mean of the two middle values for an even count. `None` if empty.
pub fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    Some(if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    })
}

/// Distances between neighbouring elements: frames along a pullback, points of a centerline.
pub trait Spacing {
    /// Distances between consecutive elements, empty with fewer than two.
    fn spacings(&self) -> Vec<f64>;

    /// Median of `spacings`, robust to holes and outliers. `None` with fewer than two elements.
    fn median_spacing(&self) -> Option<f64> {
        median(&self.spacings())
    }
}

pub trait Transform: Sized + Clone {
    fn translate(self, dx: f64, dy: f64, dz: f64) -> Self;
    fn rotate(self, angle: f64, center: (f64, f64)) -> Self;

    fn translate_mut(&mut self, dx: f64, dy: f64, dz: f64) {
        *self = self.clone().translate(dx, dy, dz);
    }
    fn rotate_mut(&mut self, angle: f64, center: (f64, f64)) {
        *self = self.clone().rotate(angle, center);
    }
}

#[cfg(test)]
mod native_tests {
    use super::*;

    #[test]
    fn test_cumulative_arc_length() {
        let pts = [(0.0, 0.0, 0.0), (3.0, 4.0, 0.0), (3.0, 4.0, 2.0)];
        assert_eq!(cumulative_arc_length(&pts), vec![0.0, 5.0, 7.0]);
        assert_eq!(cumulative_arc_length(&pts[..1]), vec![0.0]);
        assert!(cumulative_arc_length::<(f64, f64, f64)>(&[]).is_empty());
    }

    #[test]
    fn test_median() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), Some(2.5));
        assert_eq!(median(&[]), None);
    }
}
