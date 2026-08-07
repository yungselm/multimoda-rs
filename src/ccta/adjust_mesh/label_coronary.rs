use crate::ccta::binding::ccta_py::build_adjacency_map;
use crate::types::native::Centerline;
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq)]
pub struct Triangle {
    pub v0: (f64, f64, f64),
    pub v1: (f64, f64, f64),
    pub v2: (f64, f64, f64),
}

impl Triangle {
    pub fn new(v0: (f64, f64, f64), v1: (f64, f64, f64), v2: (f64, f64, f64)) -> Self {
        Self { v0, v1, v2 }
    }

    fn points(&self) -> [Point3<f64>; 3] {
        [
            Point3::new(self.v0.0, self.v0.1, self.v0.2),
            Point3::new(self.v1.0, self.v1.1, self.v1.2),
            Point3::new(self.v2.0, self.v2.1, self.v2.2),
        ]
    }
}

// Ray-Triangle intersection using Möller–Trumbore algorithm
fn ray_triangle_intersection(
    ray_origin: &Point3<f64>,
    ray_direction: &Vector3<f64>,
    triangle: &Triangle,
) -> Option<f64> {
    let eps = 1e-8;

    let [v0, v1, v2] = triangle.points();
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;

    let h = ray_direction.cross(&edge2);
    let a = edge1.dot(&h);
    if a.abs() < eps {
        return None; // Ray is parallel to triangle
    }

    let f = 1.0 / a;
    let s = ray_origin - v0;

    let u = f * s.dot(&h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let q = s.cross(&edge1);

    let v = f * ray_direction.dot(&q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    // Compute t to find intersection point
    let t = f * edge2.dot(&q);
    if t > eps {
        Some(t)
    } else {
        None
    }
}

pub fn remove_occluded_points_ray_triangle_rust(
    centerline_coronary: &Centerline,
    centerline_aorta: &Centerline,
    range_mm: f64,
    points: &[(f64, f64, f64)],
    faces: &[Triangle],
    step_size_mm: f64,
) -> Vec<(f64, f64, f64)> {
    if points.is_empty() || faces.is_empty() {
        return points.to_vec();
    }

    // Callers must orient both centerlines before calling (`orient_by_max_z` for the
    // aorta, `orient_to_reference` for the coronary) — `.take(range_points)` below
    // assumes the proximal segment is arc-length-contiguous starting at index 0.
    let spacing = (centerline_aorta.mean_spacing() + centerline_coronary.mean_spacing()) / 2.0;
    let step_cl_points = (step_size_mm / spacing).ceil() as usize;
    // range_mm is a physical length, not a point count, so it stays correct
    // regardless of how finely the centerline was resampled beforehand.
    let range_points = (range_mm / spacing).ceil() as usize;

    // Parallelize over aorta points (75 items): each thread owns 100 sequential coronary
    // iterations against faces — coarse enough to avoid scheduler overhead from nested parallelism.
    let faces_to_exclude: HashSet<usize> = centerline_aorta
        .points
        .par_iter()
        .flat_map_iter(|aorta_point| {
            let aorta_coord = Point3::new(
                aorta_point.contour_point.x,
                aorta_point.contour_point.y,
                aorta_point.contour_point.z,
            );

            let mut local_excluded: Vec<usize> = Vec::new();

            for coronary_point in centerline_coronary
                .points
                .iter()
                .take(range_points)
                .step_by(step_cl_points)
            {
                let coronary_coord = Point3::new(
                    coronary_point.contour_point.x,
                    coronary_point.contour_point.y,
                    coronary_point.contour_point.z,
                );

                let ray_direction = coronary_coord - aorta_coord;

                let mut intersecting_faces: Vec<(usize, f64)> = faces
                    .iter()
                    .enumerate()
                    .filter_map(|(face_idx, face)| {
                        ray_triangle_intersection(&aorta_coord, &ray_direction, face)
                            .map(|t| (face_idx, t))
                    })
                    .collect();

                intersecting_faces.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                if intersecting_faces.len() >= 3 {
                    if let Some((closest_face_idx, _)) = intersecting_faces.first() {
                        local_excluded.push(*closest_face_idx);
                    }
                }
            }

            local_excluded
        })
        .collect();

    println!("Total faces to exclude: {}", faces_to_exclude.len());

    const DISTANCE_THRESHOLD: f64 = 0.5; // This only works for mm, just needs to be sufficiently small

    // R-tree over the vertices of only the excluded faces (small, typically a few hundred):
    // each mesh point then costs O(log k) instead of a linear scan over every excluded face,
    // so this pass is O(points * log k) instead of O(points * faces_to_exclude). Checking "any
    // excluded face's vertex within threshold" is exactly equivalent to the original per-face
    // min-of-3-vertices check, since the vertex pool covers the same 3 vertices per face.
    // `locate_within_distance` compares squared distance, matching the pre-existing (squared)
    // semantics of `DISTANCE_THRESHOLD` here.
    let excluded_vertices: Vec<[f64; 3]> = faces_to_exclude
        .iter()
        .filter_map(|&face_idx| faces.get(face_idx))
        .flat_map(|face| {
            [
                [face.v0.0, face.v0.1, face.v0.2],
                [face.v1.0, face.v1.1, face.v1.2],
                [face.v2.0, face.v2.1, face.v2.2],
            ]
        })
        .collect();

    let points_to_remove: HashSet<usize> = if excluded_vertices.is_empty() {
        HashSet::new()
    } else {
        let excluded_tree = rstar::RTree::bulk_load(excluded_vertices);
        points
            .par_iter()
            .enumerate()
            .filter_map(|(point_idx, point)| {
                excluded_tree
                    .locate_within_distance([point.0, point.1, point.2], DISTANCE_THRESHOLD)
                    .next()
                    .is_some()
                    .then_some(point_idx)
            })
            .collect()
    };

    let filtered_points: Vec<(f64, f64, f64)> = points
        .iter()
        .enumerate()
        .filter(|(idx, _)| !points_to_remove.contains(idx))
        .map(|(_, point)| *point)
        .collect();

    println!(
        "Excluded {} faces, removed {} points (filtered from {} to {} points)",
        faces_to_exclude.len(),
        points_to_remove.len(),
        points.len(),
        filtered_points.len()
    );

    filtered_points
}

/// Find mesh points that fall within a bounding sphere of any centerline point,
/// using a single fixed `radius` (mm) for every centerline point.
pub fn find_centerline_bounded_points(
    centerline: Centerline,
    points: &[(f64, f64, f64)],
    radius: f64,
) -> Result<Vec<(f64, f64, f64)>, String> {
    if points.is_empty() || centerline.points.is_empty() {
        return Err(
            "find_centerline_bounded_points failed because `Centerline` is empty".to_string(),
        );
    }

    // R-tree over the centerline points (the small side, typically ~1000 points).
    // Each mesh point then costs O(log M) instead of a scan over all M centerline
    // points, so the whole query is O(N log M) instead of O(N * M).
    let cl_coords: Vec<[f64; 3]> = centerline
        .points
        .iter()
        .map(|p| [p.contour_point.x, p.contour_point.y, p.contour_point.z])
        .collect();
    let tree = rstar::RTree::bulk_load(cl_coords);

    let radius_sq = radius * radius;
    let result = points
        .iter()
        .filter(|p| {
            tree.locate_within_distance([p.0, p.1, p.2], radius_sq)
                .next()
                .is_some()
        })
        .copied()
        .collect();

    Ok(result)
}

/// Find mesh faces that reference any vertex coincident (within `tol`) with one of
/// `points`. Replaces the old pure-Python `_find_faces_for_points` +
/// `_prepare_faces_for_rust`, which scanned every mesh vertex per point
/// (O(points * n_vertices)) and then every mesh face (O(n_faces)) in Python. Here an
/// R-tree over `vertices` brings the per-point lookup to O(log n_vertices), and the
/// face scan stays O(n_faces) but runs as a single Rust pass instead of a Python loop.
pub fn find_faces_near_points(
    vertices: &[(f64, f64, f64)],
    faces: &[[usize; 3]],
    points: &[(f64, f64, f64)],
    tol: f64,
) -> Vec<Triangle> {
    if points.is_empty() || vertices.is_empty() || faces.is_empty() {
        return Vec::new();
    }

    let tagged_vertices: Vec<rstar::primitives::GeomWithData<[f64; 3], usize>> = vertices
        .iter()
        .enumerate()
        .map(|(idx, v)| rstar::primitives::GeomWithData::new([v.0, v.1, v.2], idx))
        .collect();
    let vertex_tree = rstar::RTree::bulk_load(tagged_vertices);

    let tol_sq = tol * tol;
    // Every vertex within `tol` is matched (not just the single nearest), which is a
    // slightly more thorough than the original "closest vertex only" Python logic —
    // relevant only if the mesh has coincident/duplicate vertices, in which case this
    // correctly includes faces touching every one of them instead of just one.
    let matched_vertex_indices: HashSet<usize> = points
        .par_iter()
        .flat_map_iter(|p| {
            vertex_tree
                .locate_within_distance([p.0, p.1, p.2], tol_sq)
                .map(|item| item.data)
        })
        .collect();

    if matched_vertex_indices.is_empty() {
        return Vec::new();
    }

    faces
        .iter()
        .filter(|[a, b, c]| {
            matched_vertex_indices.contains(a)
                || matched_vertex_indices.contains(b)
                || matched_vertex_indices.contains(c)
        })
        .map(|&[a, b, c]| Triangle::new(vertices[a], vertices[b], vertices[c]))
        .collect()
}

/// Exact-match key for a mesh-vertex coordinate. Vertex-derived point lists passed
/// around this pipeline (e.g. `rca_points`, `lca_points`) are always bit-identical
/// copies of the originating `vertices` entries (no arithmetic in between), so an
/// exact bit-pattern key is the correct tool here — unlike a radius/nearest-neighbor
/// query, this is a plain exact-membership test, so no spatial index is needed.
pub(crate) fn bits_key(p: &(f64, f64, f64)) -> (u64, u64, u64) {
    (p.0.to_bits(), p.1.to_bits(), p.2.to_bits())
}

/// Vertices present in neither `points_a` nor `points_b` (exact-match set
/// difference). Replaces the old pure-Python `_find_aortic_points`, which did the
/// same set-based filtering but paid per-vertex Python-loop overhead.
pub fn find_aortic_points(
    vertices: &[(f64, f64, f64)],
    points_a: &[(f64, f64, f64)],
    points_b: &[(f64, f64, f64)],
) -> Vec<(f64, f64, f64)> {
    let mut excluded: HashSet<(u64, u64, u64)> =
        HashSet::with_capacity(points_a.len() + points_b.len());
    excluded.extend(points_a.iter().map(bits_key));
    excluded.extend(points_b.iter().map(bits_key));

    vertices
        .iter()
        .filter(|v| !excluded.contains(&bits_key(v)))
        .copied()
        .collect()
}

/// Output of [`final_reclassification`]: the five vessel-region point lists after
/// adjacency-based label smoothing.
pub struct ReclassifiedLabels {
    pub aorta_points: Vec<(f64, f64, f64)>,
    pub rca_points: Vec<(f64, f64, f64)>,
    pub lca_points: Vec<(f64, f64, f64)>,
    pub rca_removed_points: Vec<(f64, f64, f64)>,
    pub lca_removed_points: Vec<(f64, f64, f64)>,
}

/// Refine vertex labels using a mesh adjacency map. Replaces the old pure-Python
/// `_final_reclassification`, which did the same per-vertex adjacency traversal but
/// paid Python-loop overhead for every one of up to tens of thousands of vertices.
///
/// Applies two adjacency-based correction rules, identical to the original:
/// * Logic A - an isolated RCA/LCA vertex (no same-label neighbours) is reassigned
///   to the aorta class.
/// * Logic B - a vertex removed by occlusion detection but whose neighbours are
///   predominantly (> 70%) the corresponding coronary label is restored to that
///   label.
pub fn final_reclassification(
    vertices: &[(f64, f64, f64)],
    faces: &[[usize; 3]],
    rca_points: &[(f64, f64, f64)],
    lca_points: &[(f64, f64, f64)],
    rca_removed_points: &[(f64, f64, f64)],
    lca_removed_points: &[(f64, f64, f64)],
) -> ReclassifiedLabels {
    let n_vertices = vertices.len();

    // Forward insertion so a duplicate coordinate keeps the *last* matching index,
    // mirroring Python's `{tuple(coord): i for i, coord in enumerate(...)}`.
    let mut coord_to_idx: HashMap<(u64, u64, u64), usize> = HashMap::with_capacity(n_vertices);
    for (i, v) in vertices.iter().enumerate() {
        coord_to_idx.insert(bits_key(v), i);
    }

    let mut labels: Vec<u8> = vec![0; n_vertices];
    for pt in rca_points {
        if let Some(&idx) = coord_to_idx.get(&bits_key(pt)) {
            labels[idx] = 1;
        }
    }
    for pt in lca_points {
        if let Some(&idx) = coord_to_idx.get(&bits_key(pt)) {
            labels[idx] = 2;
        }
    }
    for pt in rca_removed_points {
        if let Some(&idx) = coord_to_idx.get(&bits_key(pt)) {
            labels[idx] = 3;
        }
    }
    for pt in lca_removed_points {
        if let Some(&idx) = coord_to_idx.get(&bits_key(pt)) {
            labels[idx] = 4;
        }
    }

    let adjacency = build_adjacency_map(faces.to_vec());

    let mut new_labels = labels.clone();

    // LOGIC A: minority components of a label - excluding the single largest,
    // presumed the legitimate main body - are reclassified to a neighbouring
    // label when their external boundary is >70% that label. Generalizes the
    // original per-vertex "isolated same-label neighbour" check to whole
    // mesh-connected islands (a single-vertex island is the size-1 special
    // case), symmetric to Logic B below.
    reclassify_minority_components(&adjacency, &labels, &mut new_labels, 0, &[1, 2]); // aorta islands -> RCA/LCA
    reclassify_minority_components(&adjacency, &labels, &mut new_labels, 1, &[0]); // RCA islands -> aorta
    reclassify_minority_components(&adjacency, &labels, &mut new_labels, 2, &[0]); // LCA islands -> aorta

    // LOGIC B: a removed RCA/LCA point is restored via majority-vote label
    // propagation from the mesh boundary - starting from each point's own
    // directly-adjacent real RCA/LCA vs "other" neighbours and spreading
    // inward round by round, so points near the true RCA border resolve to
    // RCA while points on the far, aorta-facing side of the SAME connected
    // removed patch correctly stay removed, without needing the whole patch
    // to share one aggregate majority. Ties (a point receiving equal
    // evidence from both sides in the same round) stay removed - the
    // conservative default. A single directly-resolved vertex is the
    // zero-propagation-rounds special case.
    restore_removed_by_propagation(&adjacency, &labels, &mut new_labels, 3, 1);
    restore_removed_by_propagation(&adjacency, &labels, &mut new_labels, 4, 2);

    let mut result = ReclassifiedLabels {
        aorta_points: Vec::new(),
        rca_points: Vec::new(),
        lca_points: Vec::new(),
        rca_removed_points: Vec::new(),
        lca_removed_points: Vec::new(),
    };
    for (i, &label) in new_labels.iter().enumerate() {
        match label {
            0 => result.aorta_points.push(vertices[i]),
            1 => result.rca_points.push(vertices[i]),
            2 => result.lca_points.push(vertices[i]),
            3 => result.rca_removed_points.push(vertices[i]),
            4 => result.lca_removed_points.push(vertices[i]),
            _ => unreachable!(),
        }
    }
    result
}

/// Connected components of `subset`, restricted to mesh adjacency: neighbour
/// traversal only follows vertices that are themselves in `subset` (islands
/// within the subset stay separate components). Shared by
/// [`restore_removed_components`] and the `keep_largest_connected_component`
/// PyO3 binding.
pub(crate) fn connected_components(
    adjacency: &HashMap<usize, HashSet<usize>>,
    subset: &HashSet<usize>,
) -> Vec<HashSet<usize>> {
    let mut remaining: HashSet<usize> = subset.clone();
    let mut components = Vec::new();

    while let Some(&start) = remaining.iter().next() {
        let mut stack = vec![start];
        let mut component = HashSet::new();
        while let Some(i) = stack.pop() {
            if !component.insert(i) {
                continue;
            }
            if let Some(neighbors) = adjacency.get(&i) {
                for &n in neighbors {
                    if remaining.contains(&n) && !component.contains(&n) {
                        stack.push(n);
                    }
                }
            }
        }
        remaining.retain(|i| !component.contains(i));
        components.push(component);
    }

    components
}

/// External boundary of `component`: every mesh-adjacent vertex not itself in
/// the component.
fn component_boundary(
    adjacency: &HashMap<usize, HashSet<usize>>,
    component: &HashSet<usize>,
) -> HashSet<usize> {
    let mut boundary = HashSet::new();
    for &i in component {
        if let Some(neighbors) = adjacency.get(&i) {
            for &n in neighbors {
                if !component.contains(&n) {
                    boundary.insert(n);
                }
            }
        }
    }
    boundary
}

/// For every connected component of `subject_label` vertices except the
/// single largest (presumed the legitimate main body of that label - always
/// excluded, even when it is the only component, since a real anatomical
/// region reliably forms one big connected body that must never be
/// reclassified wholesale just because its aggregate boundary happens to
/// lean toward one neighbouring label), checks whether the component's
/// external boundary is >70% one of `target_labels` (checked in order; the
/// first to clear the threshold wins) and, if so, reclassifies the whole
/// component to that label.
fn reclassify_minority_components(
    adjacency: &HashMap<usize, HashSet<usize>>,
    labels: &[u8],
    new_labels: &mut [u8],
    subject_label: u8,
    target_labels: &[u8],
) {
    let subset: HashSet<usize> = labels
        .iter()
        .enumerate()
        .filter(|&(_, &l)| l == subject_label)
        .map(|(i, _)| i)
        .collect();
    if subset.is_empty() {
        return;
    }

    let mut components = connected_components(adjacency, &subset);
    let largest_idx = components
        .iter()
        .enumerate()
        .max_by_key(|(_, c)| c.len())
        .map(|(idx, _)| idx)
        .expect("subset is non-empty, so at least one component exists");
    components.remove(largest_idx);

    for component in components {
        let boundary = component_boundary(adjacency, &component);
        if boundary.is_empty() {
            continue;
        }
        for &target in target_labels {
            let target_count = boundary.iter().filter(|&&n| labels[n] == target).count();
            if (target_count as f64) > (boundary.len() as f64 * 0.7) {
                for &i in &component {
                    new_labels[i] = target;
                }
                break;
            }
        }
    }
}

/// Restores connected `removed_label` vertices to `target_label` via
/// majority-vote label propagation rather than a single whole-component
/// vote: a genuinely mixed removed patch (one face hugging the real
/// coronary lumen, another hugging the aortic wall, mesh-connected through
/// the same fold) is resolved sub-region by sub-region instead of being
/// judged as one atomic blob.
///
/// Round 0: every removed vertex tallies its directly-adjacent REAL
/// (non-removed) neighbours - how many are `target_label` vs anything else
/// ("other"). A strict majority decides it immediately. Every following
/// round, each vertex decided in the *previous* round contributes one vote
/// to its still-undecided removed neighbours; a whole round's
/// contributions are applied together before checking for new majorities,
/// so a vertex genuinely equidistant from both sides (one vote of each in
/// the same round) ties and stays undecided - deterministic regardless of
/// processing order. Vertices that never accumulate a majority (including
/// a fully mesh-isolated removed component with no real neighbours at all)
/// are left unchanged.
fn restore_removed_by_propagation(
    adjacency: &HashMap<usize, HashSet<usize>>,
    labels: &[u8],
    new_labels: &mut [u8],
    removed_label: u8,
    target_label: u8,
) {
    let subset: HashSet<usize> = labels
        .iter()
        .enumerate()
        .filter(|&(_, &l)| l == removed_label)
        .map(|(i, _)| i)
        .collect();
    if subset.is_empty() {
        return;
    }

    let mut target_count: HashMap<usize, usize> = HashMap::with_capacity(subset.len());
    let mut other_count: HashMap<usize, usize> = HashMap::with_capacity(subset.len());
    let mut decided: HashMap<usize, bool> = HashMap::new(); // true = resolved to target_label

    for &v in &subset {
        let mut t = 0usize;
        let mut o = 0usize;
        if let Some(neighbors) = adjacency.get(&v) {
            for &nb in neighbors {
                if subset.contains(&nb) {
                    continue;
                }
                if labels[nb] == target_label {
                    t += 1;
                } else {
                    o += 1;
                }
            }
        }
        target_count.insert(v, t);
        other_count.insert(v, o);
    }

    let mut frontier: Vec<usize> = Vec::new();
    for &v in &subset {
        let (t, o) = (target_count[&v], other_count[&v]);
        if t != o && (t > 0 || o > 0) {
            decided.insert(v, t > o);
            frontier.push(v);
        }
    }

    while !frontier.is_empty() {
        let mut increments: HashMap<usize, (usize, usize)> = HashMap::new();
        for &v in &frontier {
            let is_target = decided[&v];
            let Some(neighbors) = adjacency.get(&v) else {
                continue;
            };
            for &nb in neighbors {
                if !subset.contains(&nb) || decided.contains_key(&nb) {
                    continue;
                }
                let entry = increments.entry(nb).or_insert((0, 0));
                if is_target {
                    entry.0 += 1;
                } else {
                    entry.1 += 1;
                }
            }
        }

        let mut next_frontier = Vec::new();
        for (v, (dt, do_)) in increments {
            *target_count.get_mut(&v).unwrap() += dt;
            *other_count.get_mut(&v).unwrap() += do_;
            let (t, o) = (target_count[&v], other_count[&v]);
            if t != o {
                decided.insert(v, t > o);
                next_frontier.push(v);
            }
        }
        frontier = next_frontier;
    }

    for (&v, &is_target) in &decided {
        if is_target {
            new_labels[v] = target_label;
        }
    }
}

#[cfg(test)]
mod test_find_cl_bounded_points {
    use super::*;
    use crate::types::native::ContourPoint;

    #[test]
    fn test_find_points_simple_geometry() {
        let points_inside = vec![
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.5, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.5, 1.0, 1.0),
            (0.0, 0.0, 2.0),
            (1.0, 0.0, 2.0),
            (0.5, 1.0, 2.0),
        ];
        let points_outside = vec![
            (-1.0, -1.0, 0.5),
            (2.0, -1.0, 0.5),
            (0.5, 2.0, 0.5),
            (-1.0, -1.0, 1.5),
            (2.0, -1.0, 1.5),
            (0.5, 2.0, 1.5),
            (-1.0, -1.0, 2.5),
            (2.0, -1.0, 2.5),
            (0.5, 2.0, 2.5),
        ];
        let cl_raw_points: Vec<ContourPoint> = vec![
            ContourPoint {
                frame_index: 879,
                point_index: 0,
                x: 0.5,
                y: 0.5,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 212,
                point_index: 1,
                x: 0.5,
                y: 0.5,
                z: 1.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 3657,
                point_index: 2,
                x: 0.5,
                y: 0.5,
                z: 2.0,
                aortic: false,
            },
        ];
        let cl = Centerline::from_contour_points(cl_raw_points);

        // Combine inside and outside points
        let all_points: Vec<(f64, f64, f64)> = points_inside
            .iter()
            .chain(points_outside.iter())
            .cloned()
            .collect();

        let result = find_centerline_bounded_points(cl, &all_points, 1.0).unwrap();

        // The result should contain all the points that were inside our test spheres
        // Since our spheres have radius 1.0 and are centered at (0.5, 0.5, z),
        // all points within distance 1.0 should be included
        assert_eq!(result.len(), points_inside.len());

        // Verify that all expected inside points are in the result
        for expected_point in &points_inside {
            assert!(
                result.contains(expected_point),
                "Missing point: {expected_point:?}"
            );
        }

        // Verify that no outside points are in the result
        for outside_point in &points_outside {
            assert!(
                !result.contains(outside_point),
                "Unexpected point: {outside_point:?}"
            );
        }
    }

    #[test]
    fn test_single_ray_triangle_intersection() {
        // Test a single specific ray and triangle
        let ray_origin = Point3::new(0.0, 0.0, 0.0);
        let ray_direction = Vector3::new(1.0, 0.0, 0.0); // Ray along x-axis

        // Triangle in the yz-plane at x=1.0
        let triangle = Triangle::new((1.0, -1.0, -1.0), (1.0, 1.0, -1.0), (1.0, 0.0, 1.0));

        let result = ray_triangle_intersection(&ray_origin, &ray_direction, &triangle);

        println!("=== Single Ray-Triangle Test ===");
        println!("Ray origin: {ray_origin:?}");
        println!("Ray direction: {ray_direction:?}");
        println!("Triangle: {triangle:?}");
        println!("Intersection result: {result:?}");

        assert!(result.is_some(), "Ray should intersect triangle");
        assert!(
            (result.unwrap() - 1.0).abs() < 1e-6,
            "Intersection should be at t=1.0"
        );
    }

    #[test]
    fn test_find_faces_near_points_matches_only_touching_faces() {
        let vertices = vec![
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ];
        let faces = vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];

        // Only vertex 0 matches -> faces referencing it: (0,1,2), (0,1,3), (0,2,3).
        // Face (1,2,3) doesn't touch vertex 0 and must be excluded.
        let points = vec![(0.0, 0.0, 0.0)];
        let result = find_faces_near_points(&vertices, &faces, &points, 1e-6);
        assert_eq!(result.len(), 3);
        assert!(!result.contains(&Triangle::new(vertices[1], vertices[2], vertices[3])));
        assert!(result.contains(&Triangle::new(vertices[0], vertices[1], vertices[2])));
        assert!(result.contains(&Triangle::new(vertices[0], vertices[1], vertices[3])));
        assert!(result.contains(&Triangle::new(vertices[0], vertices[2], vertices[3])));
    }

    #[test]
    fn test_find_faces_near_points_no_match_returns_empty() {
        let vertices = vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)];
        let faces = vec![[0, 1, 2]];
        let points = vec![(5.0, 5.0, 5.0)];
        let result = find_faces_near_points(&vertices, &faces, &points, 1e-6);
        assert!(result.is_empty());
    }

    #[test]
    fn test_find_aortic_points_basic_set_difference() {
        let vertices = vec![
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        ];
        let a = vec![vertices[0]];
        let b = vec![vertices[1]];
        let result = find_aortic_points(&vertices, &a, &b);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&vertices[2]));
        assert!(result.contains(&vertices[3]));
    }

    #[test]
    fn test_find_aortic_points_empty_exclusions_returns_all() {
        let vertices = vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)];
        let result = find_aortic_points(&vertices, &[], &[]);
        assert_eq!(result.len(), 2);
    }

    type GridMeshFixture = (Vec<(f64, f64, f64)>, Vec<[usize; 3]>);

    // 3x3 grid mesh (9 vertices, 8 faces, z=0 plane), mirroring
    // tests/test_ccta.py::_make_grid_mesh. Vertex 4 (centre) is adjacent to
    // {1, 2, 3, 5, 6, 7}; vertex 0 (corner) is adjacent to {1, 3}.
    fn grid_mesh_fixture() -> GridMeshFixture {
        let vertices = vec![
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (2.0, 1.0, 0.0),
            (0.0, 2.0, 0.0),
            (1.0, 2.0, 0.0),
            (2.0, 2.0, 0.0),
        ];
        let faces = vec![
            [0, 1, 3],
            [1, 4, 3],
            [1, 2, 4],
            [2, 5, 4],
            [3, 4, 6],
            [4, 7, 6],
            [4, 5, 7],
            [5, 8, 7],
        ];
        (vertices, faces)
    }

    #[test]
    fn test_final_reclassification_isolated_rca_becomes_aorta() {
        let (vertices, faces) = grid_mesh_fixture();
        // vertex 0 labelled RCA; its neighbours (1, 3) are aorta -> reclassified.
        // {6,7,8} form a separate, larger (size-3) RCA component elsewhere in the
        // grid, disconnected from vertex 0's {1,3} neighbourhood - needed so
        // vertex 0 is correctly the minority island rather than the sole (and
        // therefore protected) component.
        let rca_points = vec![vertices[0], vertices[6], vertices[7], vertices[8]];
        let result = final_reclassification(&vertices, &faces, &rca_points, &[], &[], &[]);
        assert!(!result.rca_points.contains(&vertices[0]));
        assert!(result.aorta_points.contains(&vertices[0]));
        assert!(result.rca_points.contains(&vertices[6]));
        assert!(result.rca_points.contains(&vertices[7]));
        assert!(result.rca_points.contains(&vertices[8]));
    }

    #[test]
    fn test_final_reclassification_non_isolated_rca_stays() {
        let (vertices, faces) = grid_mesh_fixture();
        // vertex 0 and neighbour 1 are both RCA -> vertex 0 keeps its label.
        let rca_points = vec![vertices[0], vertices[1]];
        let result = final_reclassification(&vertices, &faces, &rca_points, &[], &[], &[]);
        assert!(result.rca_points.contains(&vertices[0]));
    }

    #[test]
    fn test_final_reclassification_removed_rca_restored_when_majority_rca() {
        let (vertices, faces) = grid_mesh_fixture();
        // vertex 4 is RCA_REMOVED; all 6 neighbours (1,2,3,5,6,7) are RCA (100% > 70%).
        let rca_points = vec![
            vertices[1],
            vertices[2],
            vertices[3],
            vertices[5],
            vertices[6],
            vertices[7],
        ];
        let rca_removed_points = vec![vertices[4]];
        let result = final_reclassification(
            &vertices,
            &faces,
            &rca_points,
            &[],
            &rca_removed_points,
            &[],
        );
        assert!(result.rca_points.contains(&vertices[4]));
        assert!(!result.rca_removed_points.contains(&vertices[4]));
    }

    #[test]
    fn test_final_reclassification_vertex_count_conserved() {
        let (vertices, faces) = grid_mesh_fixture();
        let rca_points = vec![vertices[0], vertices[1]];
        let lca_points = vec![vertices[2], vertices[3]];
        let result = final_reclassification(&vertices, &faces, &rca_points, &lca_points, &[], &[]);
        let total = result.aorta_points.len()
            + result.rca_points.len()
            + result.lca_points.len()
            + result.rca_removed_points.len()
            + result.lca_removed_points.len();
        assert_eq!(total, vertices.len());
    }

    // Fixture for the component-level Logic A tests: a 2-vertex candidate
    // island {0,1} bordering a 6-vertex cluster {2..7} on one side, and a
    // fully separate, larger 4-vertex cluster {8..11} with zero connectivity
    // to the rest (so it's always the "largest" component when {0,1} is the
    // subject label, and never itself when {2..7} is).
    //
    //   0 --- 2 --- 6            8 --- 9
    //   |\    |     |            |     |
    //   | 1 - 3 --- 7           11 --- 10
    //   |/    |
    //   4 --- 5
    //
    // vertex 0's neighbours: {1,2,3,4}; vertex 1's neighbours: {0,2,4,5}.
    // Combined external boundary of component {0,1} is exactly {2,3,4,5}.
    fn island_fixture() -> GridMeshFixture {
        let vertices: Vec<(f64, f64, f64)> = (0..12).map(|i| (i as f64, 0.0, 0.0)).collect();
        let faces = vec![
            [0, 1, 2],
            [1, 4, 5],
            [0, 3, 4],
            [2, 3, 6],
            [4, 5, 7],
            [6, 7, 3],
            [8, 9, 10],
            [8, 10, 11],
        ];
        (vertices, faces)
    }

    #[test]
    fn test_final_reclassification_aorta_island_promoted_to_rca() {
        let (vertices, faces) = island_fixture();
        // {2..7} labelled RCA; {0,1} and {8..11} default to aorta. Aorta
        // splits into {0,1} (size 2) and {8..11} (size 4) - the larger is
        // excluded as the presumed main body, leaving {0,1}'s 100%-RCA
        // boundary {2,3,4,5} to promote it.
        let rca_points: Vec<_> = (2..8).map(|i| vertices[i]).collect();
        let result = final_reclassification(&vertices, &faces, &rca_points, &[], &[], &[]);
        assert!(result.rca_points.contains(&vertices[0]));
        assert!(result.rca_points.contains(&vertices[1]));
    }

    #[test]
    fn test_final_reclassification_aorta_island_promoted_to_lca() {
        let (vertices, faces) = island_fixture();
        let lca_points: Vec<_> = (2..8).map(|i| vertices[i]).collect();
        let result = final_reclassification(&vertices, &faces, &[], &lca_points, &[], &[]);
        assert!(result.lca_points.contains(&vertices[0]));
        assert!(result.lca_points.contains(&vertices[1]));
    }

    #[test]
    fn test_final_reclassification_aorta_island_stays_when_boundary_mixed() {
        let (vertices, faces) = island_fixture();
        // {0,1}'s boundary {2,3,4,5} is split 2 RCA / 2 LCA - neither clears
        // 70%, so the island must stay aorta.
        let rca_points = vec![vertices[2], vertices[3]];
        let lca_points = vec![vertices[4], vertices[5]];
        let result = final_reclassification(&vertices, &faces, &rca_points, &lca_points, &[], &[]);
        assert!(result.aorta_points.contains(&vertices[0]));
        assert!(result.aorta_points.contains(&vertices[1]));
        assert!(!result.rca_points.contains(&vertices[0]));
        assert!(!result.lca_points.contains(&vertices[0]));
    }

    #[test]
    fn test_final_reclassification_largest_component_never_reclassified() {
        let (vertices, faces) = island_fixture();
        // {0,1,8..11} labelled RCA, leaving {2..7} as the sole aorta
        // component (no islands at all) - even though it borders RCA
        // extensively, it must never be reclassified: it's the (only, and
        // therefore always "largest") component, which is exactly the
        // catastrophic-mass-reclassification risk the "always exclude
        // largest" rule guards against.
        let mut rca_points = vec![vertices[0], vertices[1]];
        rca_points.extend((8..12).map(|i| vertices[i]));
        let result = final_reclassification(&vertices, &faces, &rca_points, &[], &[], &[]);
        for v in &vertices[2..8] {
            assert!(result.aorta_points.contains(v));
            assert!(!result.rca_points.contains(v));
        }
    }

    // Fixture for the Logic B propagation tests: two removed vertices (1, 2)
    // sharing an aorta neighbour (0) but each also touching two distinct
    // "outer" vertices (3,4 for vertex 1; 5,6 for vertex 2), plus a fully
    // isolated removed pair (7, 8) with no external connectivity at all.
    //
    //        3   4                 5   6
    //         \ /                   \ /
    //          1 --- 0 (aorta) --- 2          7 === 8 (isolated island)
    //
    // vertex 1's own neighbours are {0, 2, 3, 4} and vertex 2's are
    // {0, 1, 5, 6} - each has its own direct real-neighbour tally, resolved
    // independently in round 0 of the propagation.
    fn restore_blob_fixture() -> GridMeshFixture {
        let vertices: Vec<(f64, f64, f64)> = (0..9).map(|i| (i as f64, 0.0, 0.0)).collect();
        let faces = vec![[1, 0, 2], [1, 3, 4], [2, 5, 6], [7, 8, 7]];
        (vertices, faces)
    }

    #[test]
    fn test_final_reclassification_restores_via_local_majority() {
        let (vertices, faces) = restore_blob_fixture();
        // Vertex 1's real neighbours are {0(aorta), 3(RCA), 4(RCA)} - 2/3
        // RCA, a local majority that decides it directly in round 0, with
        // no need to consult the rest of the component. Same for vertex 2
        // via {0(aorta), 5(RCA), 6(RCA)}.
        let rca_points = vec![vertices[3], vertices[4], vertices[5], vertices[6]];
        let rca_removed_points = vec![vertices[1], vertices[2]];
        let result = final_reclassification(
            &vertices,
            &faces,
            &rca_points,
            &[],
            &rca_removed_points,
            &[],
        );
        assert!(result.rca_points.contains(&vertices[1]));
        assert!(result.rca_points.contains(&vertices[2]));
        assert!(result.rca_removed_points.is_empty());
    }

    #[test]
    fn test_final_reclassification_keeps_removed_when_local_majority_aorta() {
        let (vertices, faces) = restore_blob_fixture();
        // Only vertices 4 and 6 are RCA (one per removed vertex); 0, 3 and 5
        // default to aorta, so vertex 1's real neighbours {0(aorta),
        // 3(aorta), 4(RCA)} and vertex 2's {0(aorta), 5(aorta), 6(RCA)} are
        // each individually 2/3 aorta -> both stay removed (a genuine
        // occlusion, not a false positive).
        let rca_points = vec![vertices[4], vertices[6]];
        let rca_removed_points = vec![vertices[1], vertices[2]];
        let result = final_reclassification(
            &vertices,
            &faces,
            &rca_points,
            &[],
            &rca_removed_points,
            &[],
        );
        assert!(result.rca_removed_points.contains(&vertices[1]));
        assert!(result.rca_removed_points.contains(&vertices[2]));
        assert!(!result.rca_points.contains(&vertices[1]));
        assert!(!result.rca_points.contains(&vertices[2]));
    }

    #[test]
    fn test_final_reclassification_keeps_fully_isolated_component_removed() {
        let (vertices, faces) = restore_blob_fixture();
        // Vertices 7 and 8 are only adjacent to each other, with zero
        // external mesh connectivity -> neither ever has a real neighbour to
        // seed round 0, so no propagation ever starts and both stay removed.
        let rca_removed_points = vec![vertices[7], vertices[8]];
        let result = final_reclassification(&vertices, &faces, &[], &[], &rca_removed_points, &[]);
        assert!(result.rca_removed_points.contains(&vertices[7]));
        assert!(result.rca_removed_points.contains(&vertices[8]));
    }

    // Fixture for the multi-round propagation test: a simple 6-vertex chain
    // 0(aorta) - 1(removed) - 2(removed) - 3(removed) - 4(removed) - 5(RCA).
    // Vertices 2 and 3 have zero direct real neighbours - they can only be
    // resolved by inheriting evidence from 1 and 4 respectively in round 1.
    // This is a minimal abstraction of a real intramural "box" patch: one
    // connected removed component with a large aorta-facing face and a
    // smaller RCA-facing face, joined through the same fold. The old
    // whole-component vote saw one aorta neighbour and one RCA neighbour
    // (50/50) and would have left all four removed; propagation correctly
    // splits the chain instead.
    fn chain_fixture() -> GridMeshFixture {
        let vertices: Vec<(f64, f64, f64)> = (0..6).map(|i| (i as f64, 0.0, 0.0)).collect();
        // Degenerate (repeated-vertex) triangles, one per desired edge, so
        // the graph is exactly the path 0-1-2-3-4-5 with no incidental
        // extra edges from grouping three vertices per face.
        let faces = vec![[0, 1, 1], [1, 2, 2], [2, 3, 3], [3, 4, 4], [4, 5, 5]];
        (vertices, faces)
    }

    #[test]
    fn test_final_reclassification_splits_chain_by_propagation() {
        let (vertices, faces) = chain_fixture();
        let rca_points = vec![vertices[5]];
        let rca_removed_points = vec![vertices[1], vertices[2], vertices[3], vertices[4]];
        let result = final_reclassification(
            &vertices,
            &faces,
            &rca_points,
            &[],
            &rca_removed_points,
            &[],
        );
        assert!(result.rca_removed_points.contains(&vertices[1]));
        assert!(result.rca_removed_points.contains(&vertices[2]));
        assert!(result.rca_points.contains(&vertices[3]));
        assert!(result.rca_points.contains(&vertices[4]));
    }
}
