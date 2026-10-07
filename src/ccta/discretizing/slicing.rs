use super::anchors::AnchorIndex;
use crate::types::native::{CenterlinePoint, Contour, ContourPoint, ContourType};
use nalgebra::{Vector2, Vector3};
use std::collections::HashMap;

/// How many anchors on either side of a triangle's nearest anchors may still cut it. Triangles
/// are small compared with the anchor spacing, so ±2 covers the planes that pass through them,
/// including on bends where neighbouring planes fan out.
const PLANE_REACH: usize = 2;

/// Cuts the mesh with the plane perpendicular to each anchor and returns one raw `Contour` per
/// anchor: the closed outline of the vessel in that plane, ordered counter-clockwise around the
/// anchor tangent. Anchors where the plane does not cut the mesh get an empty contour.
///
/// Each triangle is only tested against the planes of anchors near its vertices, so a plane
/// never picks up a distant part of the vessel that it happens to intersect when extended.
///
/// The cut segments are joined into chains through the mesh edges they cross. Per anchor:
/// - a closed loop around the anchor is the outline (the largest one, if several);
/// - otherwise the open chains are joined in angular order around the anchor, bridging gaps
///   where the region has no triangles (e.g. a side-branch ostium cut out of the main vessel);
/// - otherwise the closed loop nearest the anchor is used (centerline outside the lumen).
pub fn slice_mesh(
    anchors: &[CenterlinePoint],
    vertices: &[Vector3<f64>],
    faces: &[[usize; 3]],
) -> Vec<Contour> {
    if anchors.is_empty() {
        return vec![];
    }
    let index = AnchorIndex::new(anchors);
    let mut nearest: Vec<Option<usize>> = vec![None; vertices.len()];
    let mut segments: Vec<Vec<Segment>> = vec![vec![]; anchors.len()];

    for face in faces {
        let mut lo = usize::MAX;
        let mut hi = 0;
        for &vi in face {
            let k = match nearest[vi] {
                Some(k) => k,
                None => {
                    let k = index.nearest(&vertices[vi]).unwrap_or(0);
                    nearest[vi] = Some(k);
                    k
                }
            };
            lo = lo.min(k);
            hi = hi.max(k);
        }
        let lo = lo.saturating_sub(PLANE_REACH);
        let hi = (hi + PLANE_REACH).min(anchors.len() - 1);
        for (k, anchor) in anchors.iter().enumerate().take(hi + 1).skip(lo) {
            if let Some(seg) = cut_triangle(face, vertices, anchor) {
                segments[k].push(seg);
            }
        }
    }

    let frames = plane_frames(anchors);
    anchors
        .iter()
        .zip(segments)
        .zip(frames)
        .enumerate()
        .map(|(k, ((anchor, segs), frame))| {
            let outline = select_outline(chain_segments(&segs), &frame);
            build_contour(k, anchor, &outline)
        })
        .collect()
}

/// Where the plane crosses one mesh edge. `edge` holds the vertex indices in ascending order, so
/// both triangles sharing the edge compute the identical position and the crossing links them.
#[derive(Clone, Copy)]
struct Crossing {
    edge: (usize, usize),
    pos: Vector3<f64>,
}

type Segment = [Crossing; 2];

/// Intersects one triangle with the anchor's plane. A vertex counts as above the plane when its
/// signed distance is > 0, so a vertex lying exactly on the plane is handled consistently by all
/// triangles that share it, and a triangle is crossed on exactly two edges or none.
fn cut_triangle(
    face: &[usize; 3],
    vertices: &[Vector3<f64>],
    anchor: &CenterlinePoint,
) -> Option<Segment> {
    let center = anchor_pos(anchor);
    let dist = |vi: usize| (vertices[vi] - center).dot(&anchor.tangent);

    let mut crossings = [None, None];
    let mut n = 0;
    for (a, b) in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
        let (i, j) = if a < b { (a, b) } else { (b, a) };
        let (di, dj) = (dist(i), dist(j));
        if (di > 0.0) != (dj > 0.0) {
            if n == 2 {
                return None;
            }
            let t = di / (di - dj);
            crossings[n] = Some(Crossing {
                edge: (i, j),
                pos: vertices[i] + (vertices[j] - vertices[i]) * t,
            });
            n += 1;
        }
    }
    match crossings {
        [Some(a), Some(b)] => Some([a, b]),
        _ => None,
    }
}

struct Chain {
    points: Vec<Vector3<f64>>,
    closed: bool,
}

/// Joins segments that share a crossed edge into polylines. On a manifold mesh every edge is
/// shared by at most two triangles, so each chain is a simple path or a closed loop.
fn chain_segments(segments: &[Segment]) -> Vec<Chain> {
    let mut by_edge: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let mut pos: HashMap<(usize, usize), Vector3<f64>> = HashMap::new();
    for (s, seg) in segments.iter().enumerate() {
        for c in seg {
            by_edge.entry(c.edge).or_default().push(s);
            pos.insert(c.edge, c.pos);
        }
    }

    let mut used = vec![false; segments.len()];
    let walk = |start_edge: (usize, usize), first_seg: usize, used: &mut Vec<bool>| {
        let mut edges = vec![start_edge];
        let mut current = start_edge;
        let mut seg = first_seg;
        let closed = loop {
            used[seg] = true;
            let [a, b] = segments[seg];
            current = if a.edge == current { b.edge } else { a.edge };
            if current == start_edge {
                break true;
            }
            edges.push(current);
            match by_edge[&current].iter().find(|&&s| !used[s]) {
                Some(&next) => seg = next,
                None => break false,
            }
        };
        Chain {
            points: edges.iter().map(|e| pos[e]).collect(),
            closed,
        }
    };

    let mut chains = Vec::new();
    // Open chains first, starting from an end (an edge crossed by only one segment), so each
    // comes out whole; whatever is left afterwards forms closed loops.
    let mut ends: Vec<(usize, usize)> = by_edge
        .iter()
        .filter(|(_, segs)| segs.len() == 1)
        .map(|(&e, _)| e)
        .collect();
    ends.sort_unstable();
    for edge in ends {
        let s = by_edge[&edge][0];
        if !used[s] {
            chains.push(walk(edge, s, &mut used));
        }
    }
    for s in 0..segments.len() {
        if !used[s] {
            chains.push(walk(segments[s][0].edge, s, &mut used));
        }
    }
    chains
}

/// In-plane coordinate frame of one anchor: origin, and two unit axes spanning the plane.
struct PlaneFrame {
    origin: Vector3<f64>,
    u: Vector3<f64>,
    v: Vector3<f64>,
}

impl PlaneFrame {
    fn to_2d(&self, p: &Vector3<f64>) -> Vector2<f64> {
        let d = p - self.origin;
        Vector2::new(d.dot(&self.u), d.dot(&self.v))
    }
}

/// Builds the in-plane axes for all anchors, carrying `u` from one anchor to the next (projected
/// onto the new plane) so the axes do not spin between neighbouring slices.
fn plane_frames(anchors: &[CenterlinePoint]) -> Vec<PlaneFrame> {
    let mut prev_u: Option<Vector3<f64>> = None;
    anchors
        .iter()
        .map(|a| {
            let t = a.tangent;
            let u = prev_u
                .map(|u| u - t * u.dot(&t))
                .and_then(|u| u.try_normalize(1e-9))
                .unwrap_or_else(|| any_perpendicular(&t));
            prev_u = Some(u);
            PlaneFrame {
                origin: anchor_pos(a),
                u,
                v: t.cross(&u),
            }
        })
        .collect()
}

fn any_perpendicular(t: &Vector3<f64>) -> Vector3<f64> {
    let axis = if t.x.abs() <= t.y.abs() && t.x.abs() <= t.z.abs() {
        Vector3::x()
    } else if t.y.abs() <= t.z.abs() {
        Vector3::y()
    } else {
        Vector3::z()
    };
    t.cross(&axis).try_normalize(1e-12).unwrap_or(Vector3::x())
}

/// Picks or assembles the outline for one anchor (see [`slice_mesh`]) and returns it as a closed
/// polygon (first point not repeated), counter-clockwise in the anchor's frame.
fn select_outline(chains: Vec<Chain>, frame: &PlaneFrame) -> Vec<Vector3<f64>> {
    let (loops, open): (Vec<Chain>, Vec<Chain>) = chains
        .into_iter()
        .filter(|c| c.points.len() >= 2)
        .partition(|c| c.closed && c.points.len() >= 3);

    let enclosing = loops
        .iter()
        .filter(|c| contains_origin(&to_2d(&c.points, frame)))
        .max_by(|a, b| {
            let area = |c: &Chain| signed_area(&to_2d(&c.points, frame)).abs();
            area(a).total_cmp(&area(b))
        });

    let outline = if let Some(c) = enclosing {
        c.points.clone()
    } else if !open.is_empty() {
        join_open_chains(open, frame)
    } else if let Some(c) = loops.iter().min_by(|a, b| {
        let dist = |c: &Chain| {
            to_2d(&c.points, frame)
                .iter()
                .map(|p| p.norm())
                .fold(f64::INFINITY, f64::min)
        };
        dist(a).total_cmp(&dist(b))
    }) {
        c.points.clone()
    } else {
        return vec![];
    };

    normalize_polygon(outline, frame)
}

/// Orients every open chain counter-clockwise around the anchor, sorts them by the angle they
/// start at and concatenates them; the closing polygon edges bridge the gaps between chains.
fn join_open_chains(chains: Vec<Chain>, frame: &PlaneFrame) -> Vec<Vector3<f64>> {
    let mut oriented: Vec<(f64, Vec<Vector3<f64>>)> = chains
        .into_iter()
        .map(|c| {
            let pts2 = to_2d(&c.points, frame);
            let sweep: f64 = pts2
                .windows(2)
                .map(|w| wrap_angle(angle(&w[1]) - angle(&w[0])))
                .sum();
            let mut points = c.points;
            if sweep < 0.0 {
                points.reverse();
            }
            (angle(&frame.to_2d(&points[0])), points)
        })
        .collect();
    oriented.sort_by(|a, b| a.0.total_cmp(&b.0));
    oriented.into_iter().flat_map(|(_, pts)| pts).collect()
}

/// Drops repeated points (a vertex lying exactly on the plane is reached through several edges),
/// makes the polygon counter-clockwise and starts it at the point nearest the frame's `u` axis,
/// so neighbouring slices start on the same side of the vessel.
fn normalize_polygon(mut pts: Vec<Vector3<f64>>, frame: &PlaneFrame) -> Vec<Vector3<f64>> {
    pts.dedup_by(|a, b| (*a - *b).norm() < 1e-12);
    while pts.len() > 1 && (pts[0] - pts[pts.len() - 1]).norm() < 1e-12 {
        pts.pop();
    }
    if pts.len() < 3 {
        return pts;
    }
    if signed_area(&to_2d(&pts, frame)) < 0.0 {
        pts.reverse();
    }
    let start = pts
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            angle(&frame.to_2d(a))
                .abs()
                .total_cmp(&angle(&frame.to_2d(b)).abs())
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    pts.rotate_left(start);
    pts
}

fn build_contour(k: usize, anchor: &CenterlinePoint, outline: &[Vector3<f64>]) -> Contour {
    Contour {
        id: k as u32,
        original_frame: anchor.contour_point.frame_index,
        centroid: Some((
            anchor.contour_point.x,
            anchor.contour_point.y,
            anchor.contour_point.z,
        )),
        points: outline
            .iter()
            .enumerate()
            .map(|(i, p)| ContourPoint {
                frame_index: k as u32,
                point_index: i as u32,
                x: p.x,
                y: p.y,
                z: p.z,
                aortic: false,
            })
            .collect(),
        aortic_thickness: None,
        pulmonary_thickness: None,
        kind: ContourType::Lumen,
    }
}

fn anchor_pos(a: &CenterlinePoint) -> Vector3<f64> {
    Vector3::new(a.contour_point.x, a.contour_point.y, a.contour_point.z)
}

fn to_2d(points: &[Vector3<f64>], frame: &PlaneFrame) -> Vec<Vector2<f64>> {
    points.iter().map(|p| frame.to_2d(p)).collect()
}

fn angle(p: &Vector2<f64>) -> f64 {
    p.y.atan2(p.x)
}

fn wrap_angle(a: f64) -> f64 {
    use std::f64::consts::{PI, TAU};
    (a + PI).rem_euclid(TAU) - PI
}

/// Shoelace formula; positive for a counter-clockwise polygon.
fn signed_area(pts: &[Vector2<f64>]) -> f64 {
    let n = pts.len();
    (0..n)
        .map(|i| {
            let (a, b) = (pts[i], pts[(i + 1) % n]);
            a.x * b.y - b.x * a.y
        })
        .sum::<f64>()
        / 2.0
}

/// Even-odd ray cast from the origin along +x.
fn contains_origin(pts: &[Vector2<f64>]) -> bool {
    let n = pts.len();
    let mut inside = false;
    for i in 0..n {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        if (a.y > 0.0) != (b.y > 0.0) {
            let x = a.x + (0.0 - a.y) * (b.x - a.x) / (b.y - a.y);
            if x > 0.0 {
                inside = !inside;
            }
        }
    }
    inside
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    fn z_anchors(n: usize, step: f64) -> Vec<CenterlinePoint> {
        (0..n)
            .map(|i| CenterlinePoint {
                contour_point: ContourPoint {
                    frame_index: i as u32,
                    point_index: i as u32,
                    x: 0.0,
                    y: 0.0,
                    z: i as f64 * step,
                    aortic: false,
                },
                tangent: Vector3::z(),
                branch_id: 0,
                radius: 0.0,
            })
            .collect()
    }

    /// Tube along +z whose cross-section is the closed polyline `profile` (shifted by `offset`
    /// in xy), with rings every `dz` from `z0` to `z1`.
    fn tube(
        profile: &[(f64, f64)],
        offset: (f64, f64),
        z0: f64,
        z1: f64,
        dz: f64,
    ) -> (Vec<Vector3<f64>>, Vec<[usize; 3]>) {
        let m = profile.len();
        let rings = ((z1 - z0) / dz).round() as usize + 1;
        let vertices: Vec<Vector3<f64>> = (0..rings)
            .flat_map(|r| {
                profile.iter().map(move |&(x, y)| {
                    Vector3::new(x + offset.0, y + offset.1, z0 + r as f64 * dz)
                })
            })
            .collect();
        let mut faces = Vec::new();
        for r in 0..rings - 1 {
            for i in 0..m {
                let (a, b) = (r * m + i, r * m + (i + 1) % m);
                let (c, d) = (a + m, b + m);
                faces.push([a, b, d]);
                faces.push([a, d, c]);
            }
        }
        (vertices, faces)
    }

    fn circle(r: f64, n: usize) -> Vec<(f64, f64)> {
        (0..n)
            .map(|i| {
                let a = TAU * i as f64 / n as f64;
                (r * a.cos(), r * a.sin())
            })
            .collect()
    }

    /// Closed polygon through `corners`, with extra points every `spacing` along each edge.
    fn densify(corners: &[(f64, f64)], spacing: f64) -> Vec<(f64, f64)> {
        let n = corners.len();
        (0..n)
            .flat_map(|i| {
                let (a, b) = (corners[i], corners[(i + 1) % n]);
                let len = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
                let steps = (len / spacing).ceil() as usize;
                (0..steps).map(move |s| {
                    let t = s as f64 / steps as f64;
                    (a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1))
                })
            })
            .collect()
    }

    fn xy_area(c: &Contour) -> f64 {
        let pts: Vec<Vector2<f64>> = c.points.iter().map(|p| Vector2::new(p.x, p.y)).collect();
        signed_area(&pts)
    }

    #[test]
    fn test_circular_tube_gives_closed_planar_rings() {
        let (r, n) = (2.0, 32);
        let (vertices, faces) = tube(&circle(r, n), (0.0, 0.0), -1.0, 6.0, 0.37);
        let contours = slice_mesh(&z_anchors(6, 1.0), &vertices, &faces);
        assert_eq!(contours.len(), 6);
        let r_min = r * (std::f64::consts::PI / n as f64).cos();
        for (k, c) in contours.iter().enumerate() {
            assert!(c.points.len() >= n, "slice {k}: {} points", c.points.len());
            for p in &c.points {
                assert!((p.z - k as f64).abs() < 1e-9, "slice {k}: point off plane");
                let rad = p.x.hypot(p.y);
                assert!(
                    rad > r_min - 1e-9 && rad < r + 1e-9,
                    "slice {k}: radius {rad}"
                );
            }
            assert!(
                xy_area(c) > 0.0,
                "slice {k} should be counter-clockwise about +z"
            );
        }
    }

    #[test]
    fn test_vertices_exactly_on_plane() {
        // Rings at integer z coincide with the cutting planes.
        let (vertices, faces) = tube(&circle(2.0, 24), (0.0, 0.0), -1.0, 5.0, 1.0);
        let contours = slice_mesh(&z_anchors(5, 1.0), &vertices, &faces);
        for (k, c) in contours.iter().enumerate() {
            assert_eq!(c.points.len(), 24, "slice {k}: one point per ring vertex");
            for p in &c.points {
                assert!((p.x.hypot(p.y) - 2.0).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_non_star_shaped_lumen_is_preserved() {
        // U-shaped lumen: rays from the anchor at angles ≈ 50–70° cross the wall three times, so
        // any angle-sorted outline would be wrong. The cut must reproduce the exact area (20).
        let u_shape = densify(
            &[
                (-3.0, -1.0),
                (3.0, -1.0),
                (3.0, 3.0),
                (1.0, 3.0),
                (1.0, 1.0),
                (-1.0, 1.0),
                (-1.0, 3.0),
                (-3.0, 3.0),
            ],
            0.25,
        );
        let (vertices, faces) = tube(&u_shape, (0.0, 0.0), -1.0, 4.0, 0.4);
        let contours = slice_mesh(&z_anchors(4, 1.0), &vertices, &faces);
        for c in &contours {
            assert!((xy_area(c) - 20.0).abs() < 1e-9, "area {}", xy_area(c));
        }
    }

    #[test]
    fn test_gap_in_region_is_bridged() {
        // Drop the faces in the first 60° sector, as when a side-branch ostium is cut out of the
        // main vessel: the cut is an open arc that must still come back as one closed outline.
        let n = 36;
        let (vertices, faces) = tube(&circle(2.0, n), (0.0, 0.0), -1.0, 4.0, 0.5);
        let kept: Vec<[usize; 3]> = faces
            .into_iter()
            .filter(|f| f.iter().all(|&i| i % n > 6))
            .collect();
        let contours = slice_mesh(&z_anchors(4, 1.0), &vertices, &kept);
        for c in &contours {
            let area = xy_area(c);
            let full = std::f64::consts::PI * 4.0;
            // The bridged chord removes a circular segment of ~0.36 mm² from the full disc.
            assert!(area > 0.9 * full && area < full, "area {area}");
            let m = c.points.len();
            let bridges = (0..m)
                .filter(|&i| {
                    let (a, b) = (&c.points[i], &c.points[(i + 1) % m]);
                    (a.x - b.x).hypot(a.y - b.y) > 1.0
                })
                .count();
            assert_eq!(bridges, 1, "exactly one edge should bridge the gap");
        }
    }

    #[test]
    fn test_neighbouring_vessel_is_ignored() {
        // A second tube 10 mm away is cut by the same planes; only the loop around the
        // centerline may be used.
        let (mut vertices, mut faces) = tube(&circle(2.0, 24), (0.0, 0.0), -1.0, 4.0, 0.5);
        let (v2, f2) = tube(&circle(2.0, 24), (10.0, 0.0), -1.0, 4.0, 0.5);
        let shift = vertices.len();
        vertices.extend(v2);
        faces.extend(f2.into_iter().map(|f| f.map(|i| i + shift)));
        let contours = slice_mesh(&z_anchors(4, 1.0), &vertices, &faces);
        for c in &contours {
            assert!(!c.points.is_empty());
            assert!(c.points.iter().all(|p| p.x.hypot(p.y) < 2.0 + 1e-9));
        }
    }

    #[test]
    fn test_plane_beyond_mesh_gives_empty_contour() {
        let (vertices, faces) = tube(&circle(2.0, 24), (0.0, 0.0), -1.0, 2.5, 0.5);
        let contours = slice_mesh(&z_anchors(6, 1.0), &vertices, &faces);
        assert!(!contours[2].points.is_empty());
        assert!(contours[4].points.is_empty() && contours[5].points.is_empty());
    }

    #[test]
    fn test_start_point_is_stable_between_slices() {
        let (vertices, faces) = tube(&circle(2.0, 40), (0.0, 0.0), -1.0, 6.0, 0.3);
        let contours = slice_mesh(&z_anchors(6, 1.0), &vertices, &faces);
        let starts: Vec<f64> = contours
            .iter()
            .map(|c| c.points[0].y.atan2(c.points[0].x))
            .collect();
        for w in starts.windows(2) {
            assert!((w[0] - w[1]).abs() < TAU / 40.0, "start angles {starts:?}");
        }
    }
}
