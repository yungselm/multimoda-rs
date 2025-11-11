use crate::intravascular::io::geometry::{Contour, ContourType, Frame, Geometry};
use crate::intravascular::io::input::ContourPoint;
use std::collections::HashMap;
use std::f64::consts::PI;

pub fn dummy_geometry() -> Geometry {
    let points_a = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 0.0,
                y: 0.0,
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
                x: 3.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 3,
                x: 2.0,
                y: 2.0,
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
    ];
    let points_b = vec![
            ContourPoint {
                frame_index: 2,
                point_index: 0,
                x: 0.0,
                y: 0.0,
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
                x: 3.0,
                y: 1.0,
                z: 1.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 2,
                point_index: 3,
                x: 2.0,
                y: 2.0,
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
    ];
    let points_c = vec![
            ContourPoint {
                frame_index: 3,
                point_index: 0,
                x: 0.0,
                y: 0.0,
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
                x: 3.0,
                y: 1.0,
                z: 2.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 3,
                point_index: 3,
                x: 2.0,
                y: 2.0,
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
