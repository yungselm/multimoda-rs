use super::{contour_point, Point3D};
use nalgebra::Vector3;

#[derive(Debug, Clone, PartialEq)]
pub struct CenterlinePoint {
    pub contour_point: contour_point::ContourPoint,
    pub tangent: Vector3<f64>,
    pub radius: f64,
    /// 0 = main vessel, 1+ = side branches ordered by descending length.
    pub branch_id: u32,
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
