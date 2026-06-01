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
}
