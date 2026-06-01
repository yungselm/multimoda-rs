use super::contour_point::ContourPoint;
use nalgebra::Vector3;

#[derive(Debug, Clone, PartialEq)]
pub struct CenterlinePoint {
    pub contour_point: ContourPoint,
    pub normal: Vector3<f64>,
    /// 0 = main vessel, 1+ = side branches ordered by descending length.
    pub branch_id: u32,
}
