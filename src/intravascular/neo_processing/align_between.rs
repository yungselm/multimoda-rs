use anyhow::{Result, Context, anyhow};
use rayon::prelude::*;
use std::f64::consts::PI;

use crate::intravascular::io::input::ContourPoint;
use crate::intravascular::io::geometry::{Contour, ContourType, Frame, Geometry};
use crate::intravascular::neo_processing::process_utils::downsample_contour_points;

#[derive(Clone, Debug)]
pub struct GeometryPair {
    pub geom_a: Geometry,
    pub geom_b: Geometry,
    pub label: String,
}

impl GeometryPair {
    pub fn new(
        geom_a: Geometry,
        geom_b: Geometry,
    ) -> Result<Self> {
        let label = format!("{} - {}", geom_a.label, geom_b.label);
        Ok(Self { geom_a, geom_b, label })
    }
}

pub fn align_between_geometries(
    geom_a: Geometry, 
    geom_b: Geometry,
    rot_deg: f64,
    step_rot_deg: f64,
    sample_size: usize,
) -> Result<GeometryPair> {
    let ref_frame_a_idx = geom_a
        .find_ref_frame_idx()
        .unwrap_or(geom_a.find_proximal_end_idx()) as usize;
    let ref_frame_b_idx = geom_b
        .find_ref_frame_idx()
        .unwrap_or(geom_b.find_proximal_end_idx()) as usize;
    let ref_frame_a = geom_a.frames[ref_frame_a_idx].clone();
    let ref_frame_b = geom_b.frames[ref_frame_b_idx].clone();
    let translation = (
        ref_frame_a.centroid.0 - ref_frame_b.centroid.0,
        ref_frame_a.centroid.0 - ref_frame_b.centroid.1,
        ref_frame_a.centroid.0 - ref_frame_b.centroid.1,        
    );
    geom_b.translate_geometry(translation);
    
    // TODO: Check sample_rate of the two geoms, if different interpolate between frames
    
    // TODO: equalize spacing between frames to be same in both geoms

    // TODO: trim to same number of contours, by matching on the reference contour

    // TODO: create two dummy geometries for finding best rotation, downsample both leave only
    // lumen contours, find minimal hausdorffdistance over all frames, return best angle
    // rotate geom_b with this angle

    todo!()
}