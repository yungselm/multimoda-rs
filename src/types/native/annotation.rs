use super::contour_point::ContourPoint;
use super::Transform;
use std::fmt;

/// What an annotation marks. `Custom` keeps user-defined kinds (e.g. "fibrous_cap").
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AnnotationKind {
    Sidebranch,
    Calcification,
    Lipid,
    Macrophage,
    Reference,
    Custom(String),
}

impl AnnotationKind {
    /// Parses a snake_case name, any unknown name becomes `Custom`.
    pub fn from_name(name: &str) -> Self {
        match name.trim().to_lowercase().as_str() {
            "sidebranch" => AnnotationKind::Sidebranch,
            "calcification" => AnnotationKind::Calcification,
            "lipid" => AnnotationKind::Lipid,
            "macrophage" => AnnotationKind::Macrophage,
            "reference" => AnnotationKind::Reference,
            _ => AnnotationKind::Custom(name.trim().to_string()),
        }
    }

    /// snake_case name, the custom name for `Custom`.
    pub fn name(&self) -> &str {
        match self {
            AnnotationKind::Sidebranch => "sidebranch",
            AnnotationKind::Calcification => "calcification",
            AnnotationKind::Lipid => "lipid",
            AnnotationKind::Macrophage => "macrophage",
            AnnotationKind::Reference => "reference",
            AnnotationKind::Custom(name) => name,
        }
    }
}

impl fmt::Display for AnnotationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Geometric role of the points of an annotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnnotationRole {
    Region,
    Landmark,
    AxisTip,
    DiameterEnd,
    Point,
}

impl AnnotationRole {
    pub const ALL: [AnnotationRole; 5] = [
        AnnotationRole::Region,
        AnnotationRole::Landmark,
        AnnotationRole::AxisTip,
        AnnotationRole::DiameterEnd,
        AnnotationRole::Point,
    ];

    /// Parses a snake_case name, `None` for unknown names.
    pub fn from_name(name: &str) -> Option<Self> {
        let name = name.trim().to_lowercase();
        Self::ALL.into_iter().find(|r| r.name() == name)
    }

    pub fn name(&self) -> &'static str {
        match self {
            AnnotationRole::Region => "region",
            AnnotationRole::Landmark => "landmark",
            AnnotationRole::AxisTip => "axis_tip",
            AnnotationRole::DiameterEnd => "diameter_end",
            AnnotationRole::Point => "point",
        }
    }
}

impl fmt::Display for AnnotationRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Per-frame points that are not pipe-like (side-branch outline, plaque region, landmark).
///
/// Annotations follow every rigid transform of their frame but are never
/// sorted, smoothed, interpolated, meshed or used for alignment scoring.
#[derive(Debug, Clone, PartialEq)]
pub struct Annotation {
    pub kind: AnnotationKind,
    pub role: AnnotationRole,
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
    pub fn new(
        kind: AnnotationKind,
        role: AnnotationRole,
        group: u32,
        points: Vec<ContourPoint>,
    ) -> Self {
        Annotation {
            kind,
            role,
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
        let a = Annotation::new(
            AnnotationKind::Sidebranch,
            AnnotationRole::Region,
            1,
            vec![pt(1.0, 0.0, 2.0), pt(0.0, 1.0, 2.0)],
        );
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
        assert_eq!(t.kind, AnnotationKind::Sidebranch);
        assert_eq!(t.group, 1);
    }

    #[test]
    fn test_kind_and_role_names_round_trip() {
        for name in [
            "sidebranch",
            "calcification",
            "lipid",
            "macrophage",
            "reference",
        ] {
            let kind = AnnotationKind::from_name(name);
            assert!(!matches!(kind, AnnotationKind::Custom(_)), "{name}");
            assert_eq!(kind.name(), name);
        }
        let custom = AnnotationKind::from_name("fibrous_cap");
        assert_eq!(custom, AnnotationKind::Custom("fibrous_cap".to_string()));
        assert_eq!(custom.name(), "fibrous_cap");

        for role in AnnotationRole::ALL {
            assert_eq!(AnnotationRole::from_name(role.name()), Some(role));
        }
        assert_eq!(
            AnnotationRole::from_name("Axis_Tip"),
            Some(AnnotationRole::AxisTip)
        );
        assert_eq!(AnnotationRole::from_name("outline"), None);
    }
}
