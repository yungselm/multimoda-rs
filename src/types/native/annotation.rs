use super::contour_point::ContourPoint;
use super::Transform;

/// Free-form per-frame points (side-branch outline, plaque region, landmark).
///
/// Annotations follow every rigid transform of their frame but are never
/// sorted, smoothed, interpolated, meshed or used for alignment scoring.
#[derive(Debug, Clone, PartialEq)]
pub struct Annotation {
    pub label: String,
    pub group: u32,
    pub points: Vec<ContourPoint>,
}

impl Transform for Annotation {
    fn translate(mut self, dx: f64, dy: f64, dz: f64) -> Self {
        self.points = self
            .points
            .into_iter()
            .map(|p| p.translate(dx, dy, dz))
            .collect();
        self
    }

    fn rotate(mut self, angle: f64, center: (f64, f64)) -> Self {
        if angle == 0.0 {
            return self;
        }
        self.points = self
            .points
            .into_iter()
            .map(|p| p.rotate(angle, center))
            .collect();
        self
    }
}

impl Annotation {
    pub fn new(label: impl Into<String>, group: u32, points: Vec<ContourPoint>) -> Self {
        Annotation {
            label: label.into(),
            group,
            points,
        }
    }
}

#[cfg(test)]
mod annotation_tests {
    use super::*;
    use std::f64::consts::PI;

    fn pt(x: f64, y: f64, z: f64) -> ContourPoint {
        ContourPoint {
            frame_index: 0,
            point_index: 0,
            x,
            y,
            z,
            aortic: false,
        }
    }

    #[test]
    fn test_annotation_rotate_translate() {
        let a = Annotation::new("branch", 1, vec![pt(1.0, 0.0, 2.0), pt(0.0, 1.0, 2.0)]);
        let r = a.clone().rotate(PI / 2.0, (0.0, 0.0));
        assert!((r.points[0].x - 0.0).abs() < 1e-12);
        assert!((r.points[0].y - 1.0).abs() < 1e-12);
        assert!((r.points[1].x + 1.0).abs() < 1e-12);
        assert_eq!(r.points[0].z, 2.0);

        let t = a.translate(1.0, 2.0, 3.0);
        assert_eq!(
            (t.points[0].x, t.points[0].y, t.points[0].z),
            (2.0, 2.0, 5.0)
        );
        assert_eq!(t.label, "branch");
        assert_eq!(t.group, 1);
    }
}
