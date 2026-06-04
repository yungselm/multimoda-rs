pub mod label_coronary;
pub mod scale_coronary;
use crate::types::native::Point3D;

pub(super) fn calculate_squared_distance<A: Point3D, B: Point3D>(a: &A, b: &B) -> f64 {
    let dx = a.x() - b.x();
    let dy = a.y() - b.y();
    let dz = a.z() - b.z();
    dx * dx + dy * dy + dz * dz
}
