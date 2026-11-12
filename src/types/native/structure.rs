use super::contour_point::ContourPoint;
use super::Transform;
use std::fmt;

/// What a structure is. In-wall kinds become jittered point clouds, side branches show where
/// a vessel leaves. `Custom` keeps user-defined kinds (e.g. "fibrous_cap").
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StructureKind {
    Sidebranch,
    Calcification,
    Lipid,
    Macrophage,
    Custom(String),
}

impl StructureKind {
    /// Parses a snake_case name, `branch` and `calcium` are aliases. Unknown names become `Custom`.
    pub fn from_name(name: &str) -> Self {
        match name.trim().to_lowercase().as_str() {
            "sidebranch" | "branch" => StructureKind::Sidebranch,
            "calcification" | "calcium" => StructureKind::Calcification,
            "lipid" => StructureKind::Lipid,
            "macrophage" => StructureKind::Macrophage,
            _ => StructureKind::Custom(name.trim().to_string()),
        }
    }

    /// snake_case name, the custom name for `Custom`.
    pub fn name(&self) -> &str {
        match self {
            StructureKind::Sidebranch => "sidebranch",
            StructureKind::Calcification => "calcification",
            StructureKind::Lipid => "lipid",
            StructureKind::Macrophage => "macrophage",
            StructureKind::Custom(name) => name,
        }
    }
}

impl fmt::Display for StructureKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A non-pipe structure of one frame (calcium, lipid, side branch, ...).
///
/// Read from the input, `points` is the drawn contour (`closed` or an open arc). On a frame of
/// a geometry, `points` is the point cloud generated from it. The cloud follows every rigid
/// transform of its frame but is never sorted, smoothed, interpolated or used for alignment.
#[derive(Debug, Clone, PartialEq)]
pub struct Structure {
    pub kind: StructureKind,
    pub closed: bool,
    pub points: Vec<ContourPoint>,
}

impl Transform for Structure {
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

impl Structure {
    pub fn new(kind: StructureKind, closed: bool, points: Vec<ContourPoint>) -> Self {
        Structure {
            kind,
            closed,
            points,
        }
    }
}

/// Settings of the point clouds: jitter of closed contours, revolution of open arcs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloudParams {
    /// Jittered copies per contour point, the point itself is kept too.
    pub copies: usize,
    /// Standard deviation of the in-plane jitter [mm].
    pub sigma_xy: f64,
    pub seed: u64,
    /// Copies of an open arc turned around its axis over half a turn, the arc itself included.
    pub revolution_steps: usize,
}

impl Default for CloudParams {
    fn default() -> Self {
        CloudParams {
            copies: 4,
            sigma_xy: 0.05,
            seed: 0,
            revolution_steps: 12,
        }
    }
}

/// An open side-branch arc turned around its own axis, `steps` copies over half a turn.
///
/// The axis lies in the frame plane and runs from the lumen `center` to the arc's midpoint
/// (by arc length). The arc ends sit on the lumen border and sweep the ostium circle, the
/// result is a dome pointing away from the lumen. A degenerate axis keeps the arc as drawn.
pub fn revolve_arc(
    arc: &[ContourPoint],
    center: (f64, f64, f64),
    steps: usize,
) -> Vec<ContourPoint> {
    let Some(mid) = arc_midpoint(arc) else {
        return arc.to_vec();
    };
    let (ux, uy) = (mid.0 - center.0, mid.1 - center.1);
    let len = (ux * ux + uy * uy).sqrt();
    if len < 1e-9 || steps == 0 {
        return arc.to_vec();
    }
    let (ux, uy) = (ux / len, uy / len);

    // step 0 is the drawn arc itself, kept exactly
    let mut out = Vec::with_capacity(arc.len() * steps);
    out.extend_from_slice(arc);
    for k in 1..steps {
        let (sin, cos) = (k as f64 * std::f64::consts::PI / steps as f64).sin_cos();
        for p in arc {
            // split into the part along the axis and the in-plane offset to it, turn the
            // offset around the axis into the z direction
            let (dx, dy, dz) = (p.x - center.0, p.y - center.1, p.z - center.2);
            let along = dx * ux + dy * uy;
            let (ox, oy) = (dx - along * ux, dy - along * uy);
            // signed offset in the plane, perpendicular to the axis
            let side = ox * -uy + oy * ux;
            let r_plane = side * cos;
            out.push(ContourPoint {
                x: center.0 + along * ux - r_plane * uy,
                y: center.1 + along * uy + r_plane * ux,
                z: center.2 + dz + side * sin,
                ..*p
            });
        }
    }
    for (i, p) in out.iter_mut().enumerate() {
        p.point_index = i as u32;
    }
    out
}

/// Point halfway along the polyline `arc` (x, y), `None` for fewer than two points.
fn arc_midpoint(arc: &[ContourPoint]) -> Option<(f64, f64)> {
    if arc.len() < 2 {
        return None;
    }
    let seg =
        |a: &ContourPoint, b: &ContourPoint| ((b.x - a.x).powi(2) + (b.y - a.y).powi(2)).sqrt();
    let total: f64 = arc.windows(2).map(|w| seg(&w[0], &w[1])).sum();
    let mut left = total / 2.0;
    for w in arc.windows(2) {
        let l = seg(&w[0], &w[1]);
        if l >= left && l > 0.0 {
            let t = left / l;
            return Some((
                w[0].x + t * (w[1].x - w[0].x),
                w[0].y + t * (w[1].y - w[0].y),
            ));
        }
        left -= l;
    }
    arc.last().map(|p| (p.x, p.y))
}

/// The contour points plus `copies` jittered copies of each: Gaussian in-plane, uniform in z
/// within `±half_spacing`. Deterministic for the same `params` and `stream`.
pub fn jitter_cloud(
    contour: &[ContourPoint],
    half_spacing: f64,
    params: &CloudParams,
    stream: u64,
) -> Vec<ContourPoint> {
    let mut rng = SplitMix64::new(params.seed ^ stream.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut cloud = Vec::with_capacity(contour.len() * (params.copies + 1));
    cloud.extend_from_slice(contour);
    for p in contour {
        for _ in 0..params.copies {
            let (gx, gy) = rng.gaussian_pair();
            cloud.push(ContourPoint {
                x: p.x + params.sigma_xy * gx,
                y: p.y + params.sigma_xy * gy,
                z: p.z + half_spacing * (2.0 * rng.unit() - 1.0),
                ..*p
            });
        }
    }
    for (i, p) in cloud.iter_mut().enumerate() {
        p.point_index = i as u32;
    }
    cloud
}

/// Small deterministic generator (splitmix64), enough for jitter.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Two independent standard normal values (Box-Muller).
    fn gaussian_pair(&mut self) -> (f64, f64) {
        let u1 = 1.0 - self.unit(); // (0, 1], keeps ln finite
        let u2 = self.unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let t = 2.0 * std::f64::consts::PI * u2;
        (r * t.cos(), r * t.sin())
    }
}

#[cfg(test)]
mod structure_tests {
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
    fn test_structure_rotate_translate() {
        let s = Structure::new(
            StructureKind::Calcification,
            true,
            vec![pt(1.0, 0.0, 2.0), pt(0.0, 1.0, 2.0)],
        );
        let r = s.clone().rotate(PI / 2.0, (0.0, 0.0));
        assert!((r.points[0].x - 0.0).abs() < 1e-12);
        assert!((r.points[0].y - 1.0).abs() < 1e-12);
        assert!((r.points[1].x + 1.0).abs() < 1e-12);
        assert_eq!(r.points[0].z, 2.0);

        let t = s.translate(1.0, 2.0, 3.0);
        assert_eq!(
            (t.points[0].x, t.points[0].y, t.points[0].z),
            (2.0, 2.0, 5.0)
        );
        assert_eq!(t.kind, StructureKind::Calcification);
    }

    #[test]
    fn test_kind_names_round_trip() {
        for name in ["sidebranch", "calcification", "lipid", "macrophage"] {
            let kind = StructureKind::from_name(name);
            assert!(!matches!(kind, StructureKind::Custom(_)), "{name}");
            assert_eq!(kind.name(), name);
        }
        assert_eq!(
            StructureKind::from_name("Branch"),
            StructureKind::Sidebranch
        );
        assert_eq!(
            StructureKind::from_name("calcium"),
            StructureKind::Calcification
        );
        let custom = StructureKind::from_name("fibrous_cap");
        assert_eq!(custom, StructureKind::Custom("fibrous_cap".to_string()));
        assert_eq!(custom.name(), "fibrous_cap");
    }

    #[test]
    fn test_revolve_arc_sweeps_a_dome_around_the_branch_axis() {
        // arc from (1, -1) over (2, 0) to (1, 1), lumen center at the origin: axis is +x
        let arc: Vec<ContourPoint> = (0..=8)
            .map(|i| {
                let t = -PI / 2.0 + i as f64 * PI / 8.0;
                pt(1.0 + t.cos(), t.sin(), 3.0)
            })
            .collect();

        let cloud = revolve_arc(&arc, (0.0, 0.0, 3.0), 12);

        assert_eq!(cloud.len(), arc.len() * 12);
        let xyz = |p: &ContourPoint| (p.x, p.y, p.z);
        assert!(cloud[..arc.len()]
            .iter()
            .zip(&arc)
            .all(|(a, b)| xyz(a) == xyz(b)));
        for (k, p) in cloud.iter().enumerate() {
            let q = arc[k % arc.len()];
            // x along the axis is kept, the distance to the axis too
            assert!((p.x - q.x).abs() < 1e-12);
            let r = (p.y * p.y + (p.z - 3.0).powi(2)).sqrt();
            assert!((r - q.y.abs()).abs() < 1e-12);
        }
        // the arc ends sweep the ostium circle of radius 1 in the plane x = 1
        let top = cloud[6 * arc.len()];
        assert!((top.x - 1.0).abs() < 1e-12 && top.y.abs() < 1e-12);
        assert!(((top.z - 3.0).abs() - 1.0).abs() < 1e-12);
        assert!(cloud.iter().all(|p| (p.z - 3.0).abs() <= 1.0 + 1e-12));

        // a degenerate axis keeps the arc
        assert_eq!(revolve_arc(&arc[..1], (0.0, 0.0, 3.0), 12).len(), 1);
    }

    #[test]
    fn test_jitter_cloud_is_deterministic_and_bounded() {
        let contour: Vec<ContourPoint> = (0..8)
            .map(|i| {
                let t = i as f64 * PI / 4.0;
                pt(t.cos(), t.sin(), 5.0)
            })
            .collect();
        let params = CloudParams::default();

        let a = jitter_cloud(&contour, 0.25, &params, 7);
        let b = jitter_cloud(&contour, 0.25, &params, 7);
        let c = jitter_cloud(&contour, 0.25, &params, 8);
        assert_eq!(a, b);
        assert_ne!(a, c, "another stream gives another cloud");

        assert_eq!(a.len(), contour.len() * (params.copies + 1));
        let xyz = |p: &ContourPoint| (p.x, p.y, p.z);
        assert!(
            a.iter().zip(&contour).all(|(p, q)| xyz(p) == xyz(q)),
            "contour points kept"
        );
        assert!(a.iter().enumerate().all(|(i, p)| p.point_index == i as u32));
        for p in &a {
            assert!((p.z - 5.0).abs() <= 0.25);
            let r = (p.x * p.x + p.y * p.y).sqrt();
            assert!((r - 1.0).abs() < 6.0 * params.sigma_xy);
        }
        assert!(a.iter().any(|p| (p.z - 5.0).abs() > 0.05), "z is spread");
    }
}
