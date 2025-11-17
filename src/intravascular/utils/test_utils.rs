use crate::intravascular::io::geometry::{Contour, ContourType, Frame, Geometry};
use crate::intravascular::io::input::ContourPoint;
use std::collections::HashMap;

pub fn dummy_geometry_custom(z_spacing: f64, n_frames: usize) -> Geometry {
    let mut new_frames: Vec<Frame> = Vec::new();

    for i in 0..n_frames {
        let z_coord = i as f64 * z_spacing;
        let idx = i as u32;
        let ref_pos = n_frames / 2;
        let has_ref_pt = if i as usize == ref_pos { true } else { false };

        let frame = new_frame(idx, z_coord, has_ref_pt);
        new_frames.push(frame)
    }

    Geometry {
        frames: new_frames,
        label: "dummy_geom".to_string(),
    }
}

fn new_frame(frame_index: u32, z_coord: f64, has_ref_pt: bool) -> Frame {
    let points = vec![
        ContourPoint {
            frame_index: frame_index,
            point_index: 0,
            x: 1.0,
            y: 3.0,
            z: z_coord,
            aortic: false,
        },
        ContourPoint {
            frame_index: frame_index,
            point_index: 1,
            x: 0.0,
            y: 2.0,
            z: z_coord,
            aortic: false,
        },
        ContourPoint {
            frame_index: frame_index,
            point_index: 2,
            x: 0.0,
            y: 0.0,
            z: z_coord,
            aortic: false,
        },
        ContourPoint {
            frame_index: frame_index,
            point_index: 3,
            x: 1.0,
            y: 0.0,
            z: z_coord,
            aortic: false,
        },
        ContourPoint {
            frame_index: frame_index,
            point_index: 4,
            x: 2.0,
            y: 0.0,
            z: z_coord,
            aortic: false,
        },
        ContourPoint {
            frame_index: frame_index,
            point_index: 5,
            x: 2.0,
            y: 2.0,
            z: z_coord,
            aortic: false,
        },
    ];
    let mut contour = Contour {
        id: frame_index,
        original_frame: 999,
        points: points,
        centroid: Some((1.0, 1.0, z_coord)),
        aortic_thickness: None,
        pulmonary_thickness: None,
        kind: ContourType::Lumen,
    };
    contour.compute_centroid();

    let ref_point = if has_ref_pt {
        Some(ContourPoint {
            frame_index: frame_index,
            point_index: 0,
            x: 3.0,
            y: 1.0,
            z: z_coord,
            aortic: false,
        })
    } else {
        None
    };

    Frame {
        id: contour.id,
        centroid: contour.centroid.unwrap(),
        lumen: contour,
        extras: HashMap::new(),
        reference_point: ref_point,
    }
}

pub fn dummy_geometry() -> Geometry {
    let points_a = vec![
        ContourPoint {
            frame_index: 1,
            point_index: 0,
            x: 1.0,
            y: 3.0,
            z: 0.0,
            aortic: false,
        },
        ContourPoint {
            frame_index: 1,
            point_index: 1,
            x: 0.0,
            y: 2.0,
            z: 0.0,
            aortic: false,
        },
        ContourPoint {
            frame_index: 1,
            point_index: 2,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            aortic: false,
        },
        ContourPoint {
            frame_index: 1,
            point_index: 3,
            x: 1.0,
            y: 0.0,
            z: 0.0,
            aortic: false,
        },
        ContourPoint {
            frame_index: 1,
            point_index: 4,
            x: 2.0,
            y: 0.0,
            z: 0.0,
            aortic: false,
        },
        ContourPoint {
            frame_index: 1,
            point_index: 5,
            x: 2.0,
            y: 2.0,
            z: 0.0,
            aortic: false,
        },
    ];
    let points_b = vec![
        ContourPoint {
            frame_index: 2,
            point_index: 0,
            x: 1.0,
            y: 3.0,
            z: 1.0,
            aortic: false,
        },
        ContourPoint {
            frame_index: 2,
            point_index: 1,
            x: 0.0,
            y: 2.0,
            z: 1.0,
            aortic: false,
        },
        ContourPoint {
            frame_index: 2,
            point_index: 2,
            x: 0.0,
            y: 0.0,
            z: 1.0,
            aortic: false,
        },
        ContourPoint {
            frame_index: 2,
            point_index: 3,
            x: 1.0,
            y: 0.0,
            z: 1.0,
            aortic: false,
        },
        ContourPoint {
            frame_index: 2,
            point_index: 4,
            x: 2.0,
            y: 0.0,
            z: 1.0,
            aortic: false,
        },
        ContourPoint {
            frame_index: 2,
            point_index: 5,
            x: 2.0,
            y: 2.0,
            z: 1.0,
            aortic: false,
        },
    ];
    let points_c = vec![
        ContourPoint {
            frame_index: 3,
            point_index: 0,
            x: 1.0,
            y: 3.0,
            z: 2.0,
            aortic: false,
        },
        ContourPoint {
            frame_index: 3,
            point_index: 1,
            x: 0.0,
            y: 2.0,
            z: 2.0,
            aortic: false,
        },
        ContourPoint {
            frame_index: 3,
            point_index: 2,
            x: 0.0,
            y: 0.0,
            z: 2.0,
            aortic: false,
        },
        ContourPoint {
            frame_index: 3,
            point_index: 3,
            x: 1.0,
            y: 0.0,
            z: 2.0,
            aortic: false,
        },
        ContourPoint {
            frame_index: 3,
            point_index: 4,
            x: 2.0,
            y: 0.0,
            z: 2.0,
            aortic: false,
        },
        ContourPoint {
            frame_index: 3,
            point_index: 5,
            x: 2.0,
            y: 2.0,
            z: 2.0,
            aortic: false,
        },
    ];
    let mut contour_a = Contour {
        id: 0,
        original_frame: 1,
        points: points_a,
        centroid: Some((1.0, 1.0, 0.0)),
        aortic_thickness: None,
        pulmonary_thickness: None,
        kind: ContourType::Lumen,
    };
    let mut contour_b = Contour {
        id: 1,
        original_frame: 2,
        points: points_b,
        centroid: Some((1.0, 1.0, 0.0)),
        aortic_thickness: None,
        pulmonary_thickness: None,
        kind: ContourType::Lumen,
    };
    let mut contour_c = Contour {
        id: 2,
        original_frame: 3,
        points: points_c,
        centroid: Some((1.0, 1.0, 0.0)),
        aortic_thickness: None,
        pulmonary_thickness: None,
        kind: ContourType::Lumen,
    };
    let rotation: f64 = 15.0;
    contour_a.compute_centroid();
    contour_b.compute_centroid();
    contour_c.compute_centroid();
    contour_b.translate_contour((1.0, 1.0, 0.0));
    contour_b.rotate_contour(rotation.to_radians());
    contour_c.translate_contour((2.0, 2.0, 0.0));
    contour_c.rotate_contour(rotation.to_radians() * 2.0);

    let ref_point = ContourPoint {
        frame_index: 1,
        point_index: 0,
        x: 3.0,
        y: 1.0,
        z: 0.0,
        aortic: false,
    };

    let frame_a = Frame {
        id: contour_a.id,
        centroid: contour_a.centroid.unwrap(),
        lumen: contour_a,
        extras: HashMap::new(),
        reference_point: Some(ref_point),
    };
    let frame_b = Frame {
        id: contour_b.id,
        centroid: contour_b.centroid.unwrap(),
        lumen: contour_b,
        extras: HashMap::new(),
        reference_point: None,
    };
    let frame_c = Frame {
        id: contour_c.id,
        centroid: contour_c.centroid.unwrap(),
        lumen: contour_c,
        extras: HashMap::new(),
        reference_point: None,
    };

    Geometry {
        frames: vec![frame_a, frame_b, frame_c],
        label: "dummy_geometry".to_string(),
    }
}

#[allow(dead_code)]
pub fn dummy_geometry_aligned_short() -> Geometry {
    let mut geometry = dummy_geometry();

    let rotation_deg: f64 = -15.0;

    geometry.frames[1].translate_frame((-1.0, -1.0, 0.0));
    geometry.frames[2].translate_frame((-2.0, -2.0, 0.0));
    geometry.frames[1].rotate_frame(rotation_deg.to_radians());
    geometry.frames[2].rotate_frame(rotation_deg.to_radians() * 2.0);

    geometry
}

pub fn dummy_geometry_aligned_long() -> Geometry {
    let mut g1 = dummy_geometry();

    let rotation_deg: f64 = -15.0;

    g1.frames[1].translate_frame((-1.0, -1.0, 0.0));
    g1.frames[2].translate_frame((-2.0, -2.0, 0.0));
    g1.frames[1].rotate_frame(rotation_deg.to_radians());
    g1.frames[2].rotate_frame(rotation_deg.to_radians() * 2.0);

    let mut g2 = g1.clone();
    let translation = (0.0, 0.0, 4.0);
    for (i, frame) in g2.frames.iter_mut().enumerate() {
        let idx = i as u32 + 3;
        frame.translate_frame(translation);
        frame.lumen.compute_centroid();
        frame.set_value(Some(idx), None, frame.lumen.centroid, Some(idx as f64));
    }

    let mut frames = g1.frames;
    frames.extend(g2.frames.into_iter());

    frames[3].reference_point = None;

    Geometry {
        frames,
        label: "dummy_geometry_center_reference".to_string(),
    }
}

pub fn dummy_geometry_center_reference() -> Geometry {
    let g1 = dummy_geometry();
    let mut g2 = dummy_geometry();

    let translation = (0.0, 0.0, 4.0);
    for (i, frame) in g2.frames.iter_mut().enumerate() {
        let idx = i as u32 + 3;
        frame.translate_frame(translation);
        frame.lumen.compute_centroid();
        frame.set_value(Some(idx), None, frame.lumen.centroid, Some(idx as f64));
    }

    let mut frames = g1.frames;
    frames.extend(g2.frames.into_iter());

    let mid_idx = frames.len() / 2;
    let ref_frame_id = frames[mid_idx].lumen.original_frame;
    let ref_point = ContourPoint {
        frame_index: ref_frame_id,
        point_index: 0,
        x: 3.0,
        y: 1.0,
        z: frames[mid_idx].centroid.2,
        aortic: false,
    };

    frames[0].reference_point = None;
    frames[mid_idx].reference_point = Some(ref_point);

    Geometry {
        frames,
        label: "dummy_geometry_center_reference".to_string(),
    }
}

#[cfg(test)]
mod test_utils_tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_reverse_dummy_geometry() {
        let mut geometry = dummy_geometry();

        let rotation_deg: f64 = -15.0;

        geometry.frames[1].translate_frame((-1.0, -1.0, 0.0));
        geometry.frames[2].translate_frame((-2.0, -2.0, 0.0));
        geometry.frames[1].rotate_frame(rotation_deg.to_radians());
        geometry.frames[2].rotate_frame(rotation_deg.to_radians() * 2.0);

        assert_relative_eq!(geometry.frames[1].lumen.points[0].x, 1.0, epsilon = 1e-6);
        assert_relative_eq!(geometry.frames[1].lumen.points[0].y, 3.0, epsilon = 1e-6);
        assert_relative_eq!(geometry.frames[1].lumen.points[1].x, 0.0, epsilon = 1e-6);
        assert_relative_eq!(geometry.frames[1].lumen.points[1].y, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_dummy_geometry_ref_middle() {
        let geometry = dummy_geometry_center_reference();
        println!("Geometry: {:?}", geometry);
        assert_eq!(geometry.frames.len(), 6);
        assert!(geometry.frames[0].reference_point.is_none());

        let mid_idx = geometry.frames.len() / 2;
        let mid_frame = &geometry.frames[mid_idx];

        let rp = mid_frame
            .reference_point
            .as_ref()
            .expect("expected middle frame to have a reference_point");

        assert_eq!(rp.frame_index, mid_frame.lumen.original_frame);
        assert_eq!(rp.point_index, 0);

        assert_relative_eq!(rp.x, 3.0, epsilon = 1e-6);
        assert_relative_eq!(rp.y, 1.0, epsilon = 1e-6);

        assert_relative_eq!(rp.z, mid_frame.centroid.2, epsilon = 1e-6);
        assert_relative_eq!(mid_frame.centroid.2, 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_geometry_from_z_spacing() {
        let test_geom = dummy_geometry_custom(1.0, 3);

        assert_eq!(test_geom.frames.len(), 3);
        assert_eq!(test_geom.frames[0].centroid.2, 0.0);
        assert_eq!(test_geom.frames[2].lumen.points[0].z, 2.0);
    }
}
