use super::geometry::Geometry;
use anyhow::Result;
use std::ops::Range;

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

    /// Index ranges of the frames of A and B that face each other.
    ///
    /// Frames are matched by their offset from each geometry's master reference, so frame
    /// `i` of A faces frame `i - master_a + master_b` of B. Both ranges have the same length.
    pub fn overlap(&self) -> Result<(Range<usize>, Range<usize>)> {
        let ia = self.geom_a.master_frame_idx()?;
        let ib = self.geom_b.master_frame_idx()?;
        let before = ia.min(ib);
        let after = (self.geom_a.frames.len() - ia).min(self.geom_b.frames.len() - ib);
        Ok((ia - before..ia + after, ib - before..ib + after))
    }

    /// Index in B of the frame facing frame `idx_a` of A, `None` outside the overlap.
    pub fn counterpart_in_b(&self, idx_a: usize) -> Result<Option<usize>> {
        let (range_a, range_b) = self.overlap()?;
        Ok(range_a
            .contains(&idx_a)
            .then(|| idx_a - range_a.start + range_b.start))
    }

    /// A copy holding only the overlap, frames renumbered from 0 (analyses, reports, export).
    pub fn overlapping(&self) -> Result<GeometryPair> {
        let (range_a, range_b) = self.overlap()?;
        Ok(GeometryPair {
            geom_a: slice_geometry(&self.geom_a, range_a),
            geom_b: slice_geometry(&self.geom_b, range_b),
            label: self.label.clone(),
        })
    }
}

/// Frames in `range`, with frame and contour ids renumbered from 0.
fn slice_geometry(geometry: &Geometry, range: Range<usize>) -> Geometry {
    let frames = geometry.frames[range]
        .iter()
        .cloned()
        .enumerate()
        .map(|(new_id, mut frame)| {
            let id = new_id as u32;
            frame.id = id;
            frame.lumen.id = id;
            for contour in frame.extras.values_mut() {
                contour.id = id;
            }
            frame
        })
        .collect();
    Geometry {
        frames,
        label: geometry.label.clone(),
        master_reference: geometry.master_reference,
    }
}

#[cfg(test)]
mod geometry_pair_tests {
    use super::*;
    use crate::intravascular::utils::test_utils::dummy_geometry_custom;
    use crate::types::native::ContourPoint;

    /// `n` frames with references on the given positions.
    fn geometry_with_refs(label: &str, n: usize, refs: &[usize]) -> Geometry {
        let mut geom = dummy_geometry_custom(1.0, n);
        geom.label = label.to_string();
        for (i, frame) in geom.frames.iter_mut().enumerate() {
            frame.lumen.original_frame = 100 + i as u32;
            frame.reference_point = refs.contains(&i).then_some(ContourPoint {
                frame_index: 100 + i as u32,
                point_index: 0,
                x: 1.0,
                y: 0.0,
                z: i as f64,
                aortic: false,
            });
        }
        geom
    }

    #[test]
    fn test_overlap_follows_the_masters() -> Result<()> {
        // A: 6 frames, master at 1. B: 5 frames, master at 3.
        let a = geometry_with_refs("a", 6, &[1, 4]);
        let mut b = geometry_with_refs("b", 5, &[0, 3]);
        b.set_master_reference(103)?;
        let pair = GeometryPair::new(a, b)?;

        let (ra, rb) = pair.overlap()?;
        assert_eq!((ra.clone(), rb.clone()), (0..3, 2..5));
        assert_eq!(pair.counterpart_in_b(1)?, Some(3));
        assert_eq!(pair.counterpart_in_b(3)?, None);

        let cut = pair.overlapping()?;
        assert_eq!(cut.geom_a.frames.len(), 3);
        assert_eq!(cut.geom_b.frames.len(), 3);
        assert_eq!(cut.geom_b.frames[0].lumen.original_frame, 102);
        assert_eq!(cut.geom_b.frames[0].id, 0);
        // the full pair keeps every frame
        assert_eq!(pair.geom_a.frames.len(), 6);

        // another master in A moves the overlap
        let mut pair = pair;
        pair.geom_a.set_master_reference(104)?;
        assert_eq!(pair.overlap()?, (1..6, 0..5));
        Ok(())
    }

    #[test]
    fn test_master_must_carry_a_reference() {
        let mut a = geometry_with_refs("a", 4, &[2]);
        assert!(a.set_master_reference(101).is_err());
        assert!(a.set_master_reference(102).is_ok());
        assert_eq!(a.master_frame_idx().unwrap(), 2);
        // a master whose reference is gone is reported, not silently replaced
        a.frames[2].reference_point = None;
        assert!(a.master_frame_idx().is_err());
    }
}
