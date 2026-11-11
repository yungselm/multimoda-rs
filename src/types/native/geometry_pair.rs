use super::geometry::Geometry;
use anyhow::Result;

#[derive(Clone, Debug)]
pub struct GeometryPair {
    pub geom_a: Geometry,
    pub geom_b: Geometry,
    pub label: String,
}

impl GeometryPair {
    pub fn new(geom_a: Geometry, geom_b: Geometry) -> Result<Self> {
        let label = format!("{} - {}", geom_a.label, geom_b.label);
        Ok(Self {
            geom_a,
            geom_b,
            label,
        })
    }

    /// Re-anchors `geom_a` only (see [`Geometry::set_reference_frame`]). Centerline alignment
    /// places `geom_a` and moves `geom_b` with the same per-frame transforms, so only the
    /// anchor of `geom_a` matters there.
    pub fn set_reference_frame(&mut self, frame: u32) -> Result<()> {
        self.geom_a.set_reference_frame(frame)
    }
}
