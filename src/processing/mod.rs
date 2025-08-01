pub mod resampling;
pub mod align_within;
pub mod align_between;
pub mod process_case;
pub mod walls;

use crate::io::{Geometry, ContourGroup, ContourKind};
use crate::processing::align_within::align_frames_in_geometry;

pub fn align_frames_and_states(
    mut geometry_a: Option<Geometry>, 
    mut geometry_b: Option<Geometry>, 
    mut geometry_c: Option<Geometry>, 
    mut geometry_d: Option<Geometry>,
    steps: usize,
    range_deg: f64,
) {
    if let (mut geometry_a, mut logs_a) = if Some(geometry_a) {
        align_frames_in_geometry(geometry_a, steps, range_deg);
    };
    let (mut geometry_a, mut logs_a) = if Some(geometry_a) {
        align_frames_in_geometry(geometry_a, steps, range_deg);
    };
    let (mut geometry_a, mut logs_a) = if Some(geometry_a) {
        align_frames_in_geometry(geometry_a, steps, range_deg);
    };
    let (mut geometry_a, mut logs_a) = if Some(geometry_a) {
        align_frames_in_geometry(geometry_a, steps, range_deg);
    };  
    todo!()
}