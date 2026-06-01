pub mod label_coronary;
pub mod scale_coronary;
use crate::intravascular::io::input::CenterlinePoint;

trait Point3D {
    fn x(&self) -> f64;
    fn y(&self) -> f64;
    fn z(&self) -> f64;
}

impl Point3D for (f64, f64, f64) {
    fn x(&self) -> f64 {
        self.0
    }
    fn y(&self) -> f64 {
        self.1
    }
    fn z(&self) -> f64 {
        self.2
    }
}

impl Point3D for CenterlinePoint {
    fn x(&self) -> f64 {
        self.contour_point.x
    }
    fn y(&self) -> f64 {
        self.contour_point.y
    }
    fn z(&self) -> f64 {
        self.contour_point.z
    }
}

fn calculate_squared_distance<A: Point3D, B: Point3D>(a: &A, b: &B) -> f64 {
    let dx = a.x() - b.x();
    let dy = a.y() - b.y();
    let dz = a.z() - b.z();
    dx * dx + dy * dy + dz * dz
}
