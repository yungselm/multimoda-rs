pub mod centerline;
pub mod centerline_point;
pub mod contour;
pub mod contour_point;
pub mod discretized_tree;
pub mod frame;
pub mod geometry;
pub mod geometry_pair;
pub mod record;

pub use centerline::Centerline;
pub use centerline_point::CenterlinePoint;
pub use contour::{downsample_contour_points, Contour, ContourType};
pub use contour_point::ContourPoint;
pub use discretized_tree::{DiscretizedVesselTree, ReferenceTriplet};
pub use frame::Frame;
pub use geometry::Geometry;
pub use geometry_pair::GeometryPair;
pub use record::Record;

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
