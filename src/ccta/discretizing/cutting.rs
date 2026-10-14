use super::slicing::{
    chain_segments, contains_origin, cut_triangle_by, signed_area, to_2d, Chain, PlaneFrame,
};
use super::SurfaceMesh;
use crate::ccta::morphing::fill_holes::fan_faces;
use crate::types::native::Contour;
use nalgebra::{Vector2, Vector3};
use std::collections::{HashMap, HashSet, VecDeque};

/// Which side of the cutting plane stays. The plane normal points distally (along the
/// centerline the contour was cut from).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keep {
    Proximal,
    Distal,
    /// The side with the larger surface area, whichever way the centerline runs.
    Larger,
}

/// A cutting plane taken from a discretized contour.
#[derive(Debug, Clone)]
pub struct CutPlane {
    pub origin: Vector3<f64>,
    /// Unit normal, pointing distally.
    pub normal: Vector3<f64>,
    /// Largest distance of a contour point from `origin`.
    pub radius: f64,
}

impl CutPlane {
    /// Plane of a planar contour from the discretized tree: through its centroid (the centerline
    /// anchor), with the Newell normal of its outline. Outlines run counter-clockwise about the
    /// centerline tangent, so the normal points distally.
    pub fn from_contour(contour: &Contour) -> Result<Self, String> {
        let points: Vec<Vector3<f64>> = contour
            .points
            .iter()
            .map(|p| Vector3::new(p.x, p.y, p.z))
            .collect();
        if points.len() < 3 {
            return Err(format!(
                "contour {} has {} points, a plane needs at least 3",
                contour.id,
                points.len()
            ));
        }
        let origin = match contour.centroid {
            Some((x, y, z)) => Vector3::new(x, y, z),
            None => points.iter().sum::<Vector3<f64>>() / points.len() as f64,
        };
        let n = points.len();
        let newell: Vector3<f64> = (0..n)
            .map(|i| (points[i] - origin).cross(&(points[(i + 1) % n] - origin)))
            .sum();
        let normal = newell
            .try_normalize(1e-12)
            .ok_or_else(|| format!("contour {} is degenerate (no area)", contour.id))?;
        let radius = points
            .iter()
            .map(|p| (p - origin).norm())
            .fold(0.0, f64::max);
        Ok(Self {
            origin,
            normal,
            radius,
        })
    }

    fn dist(&self, p: &Vector3<f64>) -> f64 {
        (p - self.origin).dot(&self.normal)
    }
}

/// A mesh cut by [`cut_mesh_at_plane`] and closed with a planar cap.
#[derive(Debug, Clone)]
pub struct MeshCut {
    pub vertices: Vec<Vector3<f64>>,
    /// The kept faces and the cap faces.
    pub faces: Vec<[usize; 3]>,
    /// Ring vertices on the plane, in the cap's winding order.
    pub ring: Vec<usize>,
    /// The cap's centre vertex.
    pub centroid: usize,
    /// The cap faces, also contained in `faces`.
    pub cap_faces: Vec<[usize; 3]>,
    /// New index of each input vertex, `None` where it was cut away.
    pub old_to_new: Vec<Option<usize>>,
    /// Whether the side along the plane normal (distal) was kept, as resolved for
    /// [`Keep::Larger`].
    pub kept_distal: bool,
}

/// Cuts `mesh` with `plane`, drops everything on the `keep` side's opposite and closes the cut
/// with a flat fan around the ring's centroid.
///
/// Only the closed ring around `plane.origin` within `search_radius` is cut, so another vessel
/// crossing the same plane is left alone. Ring vertices lie exactly on the plane: a mesh vertex
/// closer to the plane than `snap_fraction` of a crossed edge is moved onto it (no sliver
/// triangles), every other crossed edge is split. The side to drop is the part of the surface
/// the ring separates off, so branches beyond the cut go with it.
///
/// Fails if no closed ring encloses the origin (the plane cuts a junction, or the vessel is
/// wider than `search_radius`), if the ring does not separate the mesh, or if the ring is not
/// star-shaped around its centroid (the fan would fold over).
pub fn cut_mesh_at_plane(
    mesh: &SurfaceMesh,
    plane: &CutPlane,
    search_radius: f64,
    keep: Keep,
    snap_fraction: f64,
) -> Result<MeshCut, String> {
    if !(search_radius.is_finite() && search_radius > 0.0) {
        return Err(format!(
            "search_radius must be a positive number, got {search_radius}"
        ));
    }
    if !(0.0..0.5).contains(&snap_fraction) {
        return Err(format!(
            "snap_fraction must be in [0, 0.5), got {snap_fraction}"
        ));
    }
    let frame = PlaneFrame::new(plane.origin, &plane.normal);
    let mut vertices = mesh.vertices.clone();
    let mut dist: Vec<f64> = vertices.iter().map(|v| plane.dist(v)).collect();
    let r2 = search_radius * search_radius;
    let candidates: Vec<usize> = (0..mesh.faces.len())
        .filter(|&f| {
            mesh.faces[f]
                .iter()
                .any(|&v| (vertices[v] - plane.origin).norm_squared() <= r2)
        })
        .collect();

    // Snap vertices near the ring onto the plane, then find the ring again through them.
    let ring_chain = find_ring(&vertices, &dist, &mesh.faces, &candidates, &frame)?;
    for &(i, j) in &ring_chain.edges {
        let t = dist[i] / (dist[i] - dist[j]);
        let snap = if t <= 0.5 { i } else { j };
        if t.min(1.0 - t) < snap_fraction {
            vertices[snap] -= plane.normal * dist[snap];
            dist[snap] = 0.0;
        }
    }
    let ring_chain = find_ring(&vertices, &dist, &mesh.faces, &candidates, &frame)?;

    // One ring vertex per crossed edge: an endpoint on the plane, or a new vertex.
    let mut crossing_vertex: HashMap<(usize, usize), usize> = HashMap::new();
    let mut ring: Vec<usize> = Vec::with_capacity(ring_chain.edges.len());
    for (&(i, j), pos) in ring_chain.edges.iter().zip(&ring_chain.points) {
        let id = if dist[i] == 0.0 {
            i
        } else if dist[j] == 0.0 {
            j
        } else {
            vertices.push(pos - plane.normal * plane.dist(pos));
            dist.push(0.0);
            vertices.len() - 1
        };
        crossing_vertex.insert((i, j), id);
        if ring.last() != Some(&id) {
            ring.push(id);
        }
    }
    while ring.len() > 1 && ring.first() == ring.last() {
        ring.pop();
    }
    if ring.len() < 3 {
        return Err("the cut ring collapsed to fewer than 3 vertices".into());
    }

    // Split the faces the ring runs through.
    let mut faces: Vec<[usize; 3]> = Vec::with_capacity(mesh.faces.len() + ring.len());
    let mut is_split = vec![false; mesh.faces.len()];
    for &f in &candidates {
        let face = mesh.faces[f];
        let Some([a, b]) = cut_triangle_by(&face, &vertices, |v| dist[v]) else {
            continue;
        };
        let (Some(&x), Some(&y)) = (crossing_vertex.get(&a.edge), crossing_vertex.get(&b.edge))
        else {
            continue;
        };
        is_split[f] = true;
        faces.extend(split_face(face, &dist, (a.edge, x), (b.edge, y)));
    }
    faces.extend(
        mesh.faces
            .iter()
            .zip(&is_split)
            .filter(|(_, &split)| !split)
            .map(|(f, _)| *f),
    );

    // Drop what the ring separates off on the far side. The faces along the ring seed either
    // side, by the vertex opposite their ring edge.
    let barrier: HashSet<(usize, usize)> = (0..ring.len())
        .map(|k| sorted(ring[k], ring[(k + 1) % ring.len()]))
        .collect();
    let mut above_seeds = Vec::new();
    let mut below_seeds = Vec::new();
    for (f, &[a, b, c]) in faces.iter().enumerate() {
        for (u, w, opposite) in [(a, b, c), (b, c, a), (c, a, b)] {
            if barrier.contains(&sorted(u, w)) {
                match dist[opposite] {
                    d if d > 0.0 => above_seeds.push(f),
                    d if d < 0.0 => below_seeds.push(f),
                    _ => {}
                }
            }
        }
    }
    if above_seeds.is_empty() || below_seeds.is_empty() {
        return Err("the cut ring has faces on one side only".into());
    }
    let above = flood_fill(&faces, &above_seeds, &barrier);
    if below_seeds.iter().any(|&f| above[f]) {
        return Err(
            "the cut ring does not separate the mesh (both sides are still connected)".into(),
        );
    }
    let (dropped, kept_distal) = match keep {
        Keep::Proximal => (above, false),
        Keep::Distal => (flood_fill(&faces, &below_seeds, &barrier), true),
        Keep::Larger => {
            let below = flood_fill(&faces, &below_seeds, &barrier);
            let area = |side: &[bool]| -> f64 {
                faces
                    .iter()
                    .zip(side)
                    .filter(|(_, &on)| on)
                    .map(|(&[a, b, c], _)| {
                        (vertices[b] - vertices[a])
                            .cross(&(vertices[c] - vertices[a]))
                            .norm()
                    })
                    .sum()
            };
            if area(&above) < area(&below) {
                (above, false)
            } else {
                (below, true)
            }
        }
    };
    let mut kept: Vec<[usize; 3]> = faces
        .into_iter()
        .zip(dropped)
        .filter(|(_, d)| !d)
        .map(|(f, _)| f)
        .collect();

    // Wind the ring against the kept faces along it, so the cap continues their orientation.
    let directed: HashSet<(usize, usize)> = kept
        .iter()
        .flat_map(|&[a, b, c]| [(a, b), (b, c), (c, a)])
        .collect();
    if directed.contains(&(ring[0], ring[1])) {
        ring.reverse();
    }
    let ring_2d: Vec<Vector2<f64>> = ring.iter().map(|&i| frame.to_2d(&vertices[i])).collect();
    let center_2d = polygon_centroid(&ring_2d);
    check_star_shaped(&ring_2d, &center_2d)?;
    let centroid = vertices.len();
    vertices.push(frame.origin + frame.u * center_2d.x + frame.v * center_2d.y);
    let cap = fan_faces(&ring, centroid);
    kept.extend(&cap);

    Ok(MeshCut {
        kept_distal,
        ..compact(vertices, kept, ring, centroid, cap, mesh.vertices.len())
    })
}

/// The closed ring around the plane origin: the smallest enclosing loop, so a loop of a larger
/// structure around the vessel is never chosen.
fn find_ring(
    vertices: &[Vector3<f64>],
    dist: &[f64],
    faces: &[[usize; 3]],
    candidates: &[usize],
    frame: &PlaneFrame,
) -> Result<Chain, String> {
    let segments: Vec<_> = candidates
        .iter()
        .filter_map(|&f| cut_triangle_by(&faces[f], vertices, |v| dist[v]))
        .collect();
    chain_segments(&segments)
        .into_iter()
        .filter(|c| c.closed && c.points.len() >= 3)
        .map(|c| {
            let pts = to_2d(&c.points, frame);
            (contains_origin(&pts), signed_area(&pts).abs(), c)
        })
        .filter(|(inside, _, _)| *inside)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(_, _, c)| c)
        .ok_or_else(|| {
            "no closed ring around the slice centre: the plane may cut a junction, or the \
             vessel is wider than the search radius"
                .into()
        })
}

/// Pieces of `face` either side of the plane. The crossings are given as (crossed edge, ring
/// vertex). Degenerate pieces (where a crossing is a vertex) are left out.
fn split_face(
    face: [usize; 3],
    dist: &[f64],
    a: ((usize, usize), usize),
    b: ((usize, usize), usize),
) -> Vec<[usize; 3]> {
    let above = |v: usize| dist[v] > 0.0;
    // The lone vertex is the one on its own side; rotate it to the front, keeping the winding.
    let lone_pos = (0..3)
        .find(|&k| {
            above(face[k]) != above(face[(k + 1) % 3]) && above(face[k]) != above(face[(k + 2) % 3])
        })
        .unwrap_or(0);
    let [l, p, q] = [
        face[lone_pos],
        face[(lone_pos + 1) % 3],
        face[(lone_pos + 2) % 3],
    ];
    let on_edge = |u: usize, w: usize| {
        let e = sorted(u, w);
        if a.0 == e {
            a.1
        } else {
            b.1
        }
    };
    let x = on_edge(l, p);
    let y = on_edge(l, q);
    [[l, x, y], [x, p, q], [x, q, y]]
        .into_iter()
        .filter(|[i, j, k]| i != j && j != k && k != i)
        .collect()
}

/// Faces reachable from `seeds` across shared edges that are not in `barrier`.
fn flood_fill(
    faces: &[[usize; 3]],
    seeds: &[usize],
    barrier: &HashSet<(usize, usize)>,
) -> Vec<bool> {
    let mut by_edge: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (f, &[a, b, c]) in faces.iter().enumerate() {
        for (u, w) in [(a, b), (b, c), (c, a)] {
            by_edge.entry(sorted(u, w)).or_default().push(f);
        }
    }
    let mut reached = vec![false; faces.len()];
    let mut queue: VecDeque<usize> = seeds.iter().copied().collect();
    for &s in seeds {
        reached[s] = true;
    }
    while let Some(f) = queue.pop_front() {
        let [a, b, c] = faces[f];
        for (u, w) in [(a, b), (b, c), (c, a)] {
            let e = sorted(u, w);
            if barrier.contains(&e) {
                continue;
            }
            for &g in &by_edge[&e] {
                if !reached[g] {
                    reached[g] = true;
                    queue.push_back(g);
                }
            }
        }
    }
    reached
}

/// Area centroid of a simple polygon.
fn polygon_centroid(pts: &[Vector2<f64>]) -> Vector2<f64> {
    let n = pts.len();
    let mut c = Vector2::zeros();
    let mut area2 = 0.0;
    for i in 0..n {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        let cross = a.x * b.y - b.x * a.y;
        c += (a + b) * cross;
        area2 += cross;
    }
    if area2.abs() < 1e-12 {
        return pts.iter().sum::<Vector2<f64>>() / n as f64;
    }
    c / (3.0 * area2)
}

/// Every fan triangle around `center` turns the same way as the ring.
fn check_star_shaped(ring: &[Vector2<f64>], center: &Vector2<f64>) -> Result<(), String> {
    let n = ring.len();
    let orientation = signed_area(ring).signum();
    let folded = (0..n)
        .filter(|&i| {
            let (a, b) = (ring[i] - center, ring[(i + 1) % n] - center);
            (a.x * b.y - b.x * a.y) * orientation <= 0.0
        })
        .count();
    if folded > 0 {
        return Err(format!(
            "the cut ring is not star-shaped around its centroid ({folded} of {n} fan \
             triangles would fold over), try a neighbouring slice"
        ));
    }
    Ok(())
}

/// Drops unreferenced vertices and renumbers everything.
fn compact(
    vertices: Vec<Vector3<f64>>,
    faces: Vec<[usize; 3]>,
    ring: Vec<usize>,
    centroid: usize,
    cap: Vec<[usize; 3]>,
    n_input: usize,
) -> MeshCut {
    let mut new_index: Vec<Option<usize>> = vec![None; vertices.len()];
    let mut kept_vertices = Vec::new();
    for &v in faces.iter().flatten() {
        if new_index[v].is_none() {
            new_index[v] = Some(kept_vertices.len());
            kept_vertices.push(vertices[v]);
        }
    }
    let remap = |v: usize| new_index[v].expect("face vertices are kept");
    MeshCut {
        vertices: kept_vertices,
        faces: faces.iter().map(|f| f.map(remap)).collect(),
        ring: ring.iter().map(|&v| remap(v)).collect(),
        centroid: remap(centroid),
        cap_faces: cap.iter().map(|f| f.map(remap)).collect(),
        old_to_new: new_index[..n_input].to_vec(),
        kept_distal: false,
    }
}

fn sorted(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccta::morphing::fill_holes::boundary_loops;
    use std::f64::consts::{PI, TAU};

    /// Outward-wound open tube along +z: radius `r`, `m` vertices per ring, rings every `dz`.
    fn tube(r: f64, m: usize, z0: f64, z1: f64, dz: f64, x0: f64) -> SurfaceMesh {
        let rings = ((z1 - z0) / dz).round() as usize + 1;
        let vertices: Vec<(f64, f64, f64)> = (0..rings)
            .flat_map(|k| {
                (0..m).map(move |i| {
                    let a = TAU * i as f64 / m as f64;
                    (x0 + r * a.cos(), r * a.sin(), z0 + k as f64 * dz)
                })
            })
            .collect();
        let mut faces = Vec::new();
        for k in 0..rings - 1 {
            for i in 0..m {
                let (a, b) = (k * m + i, k * m + (i + 1) % m);
                faces.push([a, b, b + m]);
                faces.push([a, b + m, a + m]);
            }
        }
        SurfaceMesh::new(&vertices, faces).unwrap()
    }

    fn z_plane(z: f64) -> CutPlane {
        CutPlane {
            origin: Vector3::new(0.0, 0.0, z),
            normal: Vector3::z(),
            radius: 2.0,
        }
    }

    fn normal(cut: &MeshCut, f: &[usize; 3]) -> Vector3<f64> {
        let v = &cut.vertices;
        (v[f[1]] - v[f[0]]).cross(&(v[f[2]] - v[f[0]]))
    }

    /// Every undirected edge is shared by at most two faces, wound opposite ways.
    fn assert_consistent_manifold(faces: &[[usize; 3]]) {
        let mut seen = HashSet::new();
        for &[a, b, c] in faces {
            for e in [(a, b), (b, c), (c, a)] {
                assert!(seen.insert(e), "directed edge {e:?} used twice");
            }
        }
    }

    #[test]
    fn test_keep_proximal_gives_flat_capped_end() {
        let mesh = tube(2.0, 40, 0.0, 6.0, 0.5, 0.0);
        let cut = cut_mesh_at_plane(&mesh, &z_plane(3.2), 5.0, Keep::Proximal, 0.0).unwrap();
        assert!(cut.vertices.iter().all(|v| v.z <= 3.2 + 1e-12));
        // One ring point per crossed edge: 40 vertical edges and 40 quad diagonals.
        assert_eq!(cut.ring.len(), 80);
        for &i in &cut.ring {
            assert!((cut.vertices[i].z - 3.2).abs() < 1e-12);
        }
        assert!((cut.vertices[cut.centroid] - Vector3::new(0.0, 0.0, 3.2)).norm() < 1e-9);
        // The cap faces outward (+z) and closes the top: only the bottom stays open.
        assert_eq!(cut.cap_faces.len(), 80);
        for f in &cut.cap_faces {
            assert!(normal(&cut, f).z > 0.0);
        }
        assert_consistent_manifold(&cut.faces);
        assert_eq!(boundary_loops(&cut.faces).len(), 1);
        let cap_area: f64 = cut
            .cap_faces
            .iter()
            .map(|f| normal(&cut, f).norm() / 2.0)
            .sum();
        assert!((cap_area - PI * 4.0).abs() < 0.05 * PI * 4.0, "{cap_area}");
    }

    #[test]
    fn test_keep_distal_drops_the_proximal_side() {
        let mesh = tube(2.0, 40, 0.0, 6.0, 0.5, 0.0);
        let cut = cut_mesh_at_plane(&mesh, &z_plane(3.2), 5.0, Keep::Distal, 0.0).unwrap();
        assert!(cut.vertices.iter().all(|v| v.z >= 3.2 - 1e-12));
        for f in &cut.cap_faces {
            assert!(normal(&cut, f).z < 0.0);
        }
        assert_consistent_manifold(&cut.faces);
        // Vertices below the plane are gone, the ones above keep their position.
        for (old, new) in cut.old_to_new.iter().enumerate() {
            let z = mesh.vertices[old].z;
            assert_eq!(new.is_some(), z > 3.2);
            if let Some(n) = new {
                assert_eq!(cut.vertices[*n], mesh.vertices[old]);
            }
        }
    }

    #[test]
    fn test_keep_larger_follows_the_area() {
        let mesh = tube(2.0, 40, 0.0, 6.0, 0.5, 0.0);
        // Most of the tube lies above z = 1.2, whichever way the normal points.
        for normal in [Vector3::z(), -Vector3::z()] {
            let plane = CutPlane {
                normal,
                ..z_plane(1.2)
            };
            let cut = cut_mesh_at_plane(&mesh, &plane, 5.0, Keep::Larger, 0.0).unwrap();
            assert!(cut.vertices.iter().all(|v| v.z >= 1.2 - 1e-12));
            assert_eq!(cut.kept_distal, normal.z > 0.0);
            assert_consistent_manifold(&cut.faces);
        }
    }

    #[test]
    fn test_near_vertices_are_snapped_instead_of_split() {
        // The ring at z = 3.0 is 0.02 below the plane: snapped onto it, no new ring vertices.
        let mesh = tube(2.0, 40, 0.0, 6.0, 0.5, 0.0);
        let cut = cut_mesh_at_plane(&mesh, &z_plane(3.02), 5.0, Keep::Proximal, 0.2).unwrap();
        assert_eq!(cut.ring.len(), 40);
        let original = cut.old_to_new.iter().flatten().collect::<HashSet<_>>();
        assert!(cut.ring.iter().all(|i| original.contains(i)));
        assert!(cut
            .ring
            .iter()
            .all(|&i| (cut.vertices[i].z - 3.02).abs() < 1e-12));
        assert_consistent_manifold(&cut.faces);
        assert_eq!(boundary_loops(&cut.faces).len(), 1);
        // Without snapping, the thin band is split instead.
        let split = cut_mesh_at_plane(&mesh, &z_plane(3.02), 5.0, Keep::Proximal, 0.0).unwrap();
        assert!(split.vertices.len() > cut.vertices.len());
    }

    #[test]
    fn test_plane_through_mesh_vertices() {
        let mesh = tube(2.0, 24, 0.0, 6.0, 1.0, 0.0);
        let cut = cut_mesh_at_plane(&mesh, &z_plane(3.0), 5.0, Keep::Proximal, 0.0).unwrap();
        assert_eq!(cut.ring.len(), 24);
        assert_eq!(cut.vertices.len(), 4 * 24 + 1);
        assert_consistent_manifold(&cut.faces);
    }

    #[test]
    fn test_tilted_plane() {
        let mesh = tube(2.0, 48, 0.0, 10.0, 0.3, 0.0);
        let normal_dir = Vector3::new(0.3, 0.0, 1.0).normalize();
        let plane = CutPlane {
            origin: Vector3::new(0.0, 0.0, 5.0),
            normal: normal_dir,
            radius: 2.0,
        };
        let cut = cut_mesh_at_plane(&mesh, &plane, 5.0, Keep::Proximal, 0.15).unwrap();
        for v in &cut.vertices {
            assert!(plane.dist(v) <= 1e-9);
        }
        for &i in &cut.ring {
            assert!(plane.dist(&cut.vertices[i]).abs() < 1e-9);
        }
        assert_consistent_manifold(&cut.faces);
        assert_eq!(boundary_loops(&cut.faces).len(), 1);
    }

    #[test]
    fn test_neighbouring_vessel_is_not_cut() {
        let a = tube(2.0, 30, 0.0, 6.0, 0.5, 0.0);
        let b = tube(2.0, 30, 0.0, 6.0, 0.5, 10.0);
        let shift = a.vertices.len();
        let mesh = SurfaceMesh {
            vertices: a.vertices.iter().chain(&b.vertices).copied().collect(),
            faces: a
                .faces
                .iter()
                .copied()
                .chain(b.faces.iter().map(|f| f.map(|i| i + shift)))
                .collect(),
        };
        let cut = cut_mesh_at_plane(&mesh, &z_plane(3.2), 4.0, Keep::Proximal, 0.0).unwrap();
        // The second tube keeps every vertex and face.
        assert!(cut.old_to_new[shift..].iter().all(Option::is_some));
        assert_eq!(boundary_loops(&cut.faces).len(), 1 + 2);
    }

    #[test]
    fn test_no_ring_around_origin_is_an_error() {
        let mesh = tube(2.0, 30, 0.0, 6.0, 0.5, 0.0);
        let mut plane = z_plane(3.2);
        plane.origin.x = 5.0;
        assert!(cut_mesh_at_plane(&mesh, &plane, 2.0, Keep::Proximal, 0.0).is_err());
        // Search radius smaller than the vessel: the ring is incomplete.
        assert!(cut_mesh_at_plane(&mesh, &z_plane(3.2), 1.0, Keep::Proximal, 0.0).is_err());
        assert!(cut_mesh_at_plane(&mesh, &z_plane(3.2), 5.0, Keep::Proximal, 0.6).is_err());
    }

    #[test]
    fn test_plane_from_contour_points_distally() {
        let mesh = tube(2.0, 30, 0.0, 6.0, 0.5, 0.0);
        let cl = crate::types::native::Centerline {
            points: (0..7)
                .map(|i| crate::types::native::CenterlinePoint {
                    contour_point: crate::types::native::ContourPoint {
                        frame_index: i,
                        point_index: i,
                        x: 0.0,
                        y: 0.0,
                        z: i as f64,
                        aortic: false,
                    },
                    tangent: Vector3::z(),
                    branch_id: 0,
                    radius: 0.0,
                })
                .collect(),
            branch_start_indices: vec![0],
        };
        let contours = super::super::discretize_vessel_rs(&cl, &mesh, None, 0, 1.0, 50).unwrap();
        let plane = CutPlane::from_contour(&contours[3]).unwrap();
        assert!((plane.normal - Vector3::z()).norm() < 1e-9);
        assert!((plane.origin.z - contours[3].centroid.unwrap().2).abs() < 1e-12);
        assert!((plane.radius - 2.0).abs() < 0.01);
    }
}
