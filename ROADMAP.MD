# Python package that has one purpose: Combining any intravascular image modality with CCTA meshes.

# Optimizations
- [] Optimize the ray intersection algorithm
- [] Better parallelization strategy for frames alignment by processing in batches (frames only have to be aligned relative to each other not sequentially)

## `label_geometry` scales badly with vertex count — profiling results (2026-07-09)

**Top-level timings, full `fullworkflow.py` run (189.9s total):**

| Step | Time |
|---|---|
| `label_geometry` (final, 57,248 verts) | 95.1s |
| `label_geometry` (initial, 25,171 verts) | 38.6s |
| `fix_and_remesh_stitched_mesh` | 17.6-17.8s |
| `align_combined` | 16.1s |
| `prepare_centerlines` | 5.3-5.8s |
| `stitch_ccta_to_intravascular` | 4.5s |
| `discretize_vessel_tree` | 4.3-4.5s |
| `label_branches` | 2.4s |
| `from_file_singlepair` | 1.8s |
| `find_sharp_angles` | 1.7s |
| `remove_labeled_points_from_mesh` | 0.5s |
| `label_anomalous_region` | 0.25s |
| everything else (scaling/sync steps, exports, etc.) | <0.11s each |

**Internal breakdown of `label_geometry` (initial call → final call):**

| Sub-step | Initial (25,171 verts) | Final (57,248 verts) | Input growth | Time growth |
|---|---|---|---|---|
| `find_centerline_bounded_points_simple` (Rust, rca+lca) | 22.4s | 51.1s | ~2.4-2.8x points | 2.3x |
| `_find_faces_for_points` (Python, via `_prepare_faces_for_rust`) | 3.1s | 19.5s | 2.8x points | 6.2x |
| `remove_occluded_points_ray_triangle` (Rust) | 4.4s | 11.3s | ~2.8x points | 2.5x |
| `clean_outlier_points` — lca (Rust) | 0.73s | 3.71s | 2.4x points | 5.1x |
| `clean_outlier_points` — rca (Rust) | 0.54s | 3.48s | 2.9x points | 6.4x |
| removed-points diff (`p for p in a if p not in b`, Python) | 0.18s | 1.45s | 2.8x points | 8.0x |
| `_final_reclassification` (Python, loop over mesh vertices) | 0.57s | 1.36s | 2.3x verts | 2.4x |
| `plot_results_key` (control_plot rendering) | 5.9s | 2.4s | — | n/a |

Timed a full `fullworkflow.py` run end to end (`examples/fullworkflow copy.py`, timers added
per top-level call) plus internal timers inside `label_geometry`
(`multimodars/ccta/labeling.py`). `label_geometry` is called twice —
once on the raw CCTA mesh (25,171 verts) and once on the remeshed mesh after
`fix_and_remesh_stitched_mesh` (57,248 verts, ~2.3x more vertices) — and cost jumped from
38.6s to 95.1s, i.e. worse than linear in vertex count. Breakdown of the internal timers
(initial call → final call):

- [x] `find_centerline_bounded_points_simple` (Rust, rca+lca combined): 22.4s → 51.1s.
  **FIXED (2026-07-09):** rewrote `find_centerline_bounded_points` in
  `src/ccta/adjust_mesh/label_coronary.rs` to build an `rstar::RTree` over the (small)
  centerline point set and query each mesh point via `locate_within_distance`, dropping
  complexity from O(N_points × M_centerline) to O((N+M) log M). Re-measured on the
  25,171-vertex mesh: **22.4s → 0.236s** (~95x). See the new audit section below for every
  other place in the codebase with the same fixable pattern.
- [] `_find_faces_for_points` (called via `_prepare_faces_for_rust`, pure Python in
  `labeling.py`): 3.1s → 19.5s — a 6.2x increase for only a 2.8x increase in input points.
  Root cause: for every occlusion-candidate point it computes `np.linalg.norm(verts - p,
  axis=1)` over **all** mesh vertices to find the nearest one — O(points_found × n_vertices).
  Should build a `scipy.spatial.cKDTree` (or reuse a vertex-coordinate hash map, since matches
  are expected to be exact/near-exact) once per call instead of a per-point full-mesh scan.
- [] `remove_occluded_points_ray_triangle` (Rust): 4.4s → 11.3s — scales with input size as
  expected, not currently a priority.
- [] `rca_removed_points` / `lca_removed_points` diff (`p for p in a if p not in b`, pure
  Python in `labeling.py`): 0.18s → 1.45s (8x for a 2.8x input increase). `b` is a plain list,
  so `not in` is a linear scan — O(n × m). Trivial fix: convert `final_rca_points_found` to a
  `set` before the membership test.
- [] `clean_outlier_points` (Rust, called twice per `label_geometry` call): lca 0.73s → 3.71s,
  rca 0.54s → 3.48s — roughly quadratic in point count (~5-6x time for ~2.4-2.9x more points),
  suggesting a brute-force nearest-neighbor search inside the Rust implementation with no
  spatial index.
- [] `_final_reclassification` (pure Python loop over every mesh vertex in
  `labeling.py`): 0.57s → 1.36s — scales roughly linearly with vertex count, acceptable for now
  but a candidate to push into Rust if further mesh sizes grow.

Net: three separate near-quadratic hot spots (`_find_faces_for_points`,
the removed-points diff, and `clean_outlier_points`) compound with the already-large linear
cost of `find_centerline_bounded_points_simple`, which is why `label_geometry` more than
doubles in time for a ~2.3x increase in mesh vertices. Fixing the two pure-Python spots
(KD-tree lookup + set-based diff) is low-risk/high-value; `clean_outlier_points` and
`find_centerline_bounded_points_simple` need attention on the Rust side.

Other steps outside `label_geometry` worth noting from the same profiling run (not
investigated further yet): `fix_and_remesh_stitched_mesh` 17.6-17.8s, `align_combined`
16.1s, `prepare_centerlines` 5.3-5.8s, `stitch_ccta_to_intravascular` 4.5s,
`discretize_vessel_tree` 4.3-4.5s, `label_branches` 2.4s, `find_sharp_angles` 1.7s — all far
below the `label_geometry` cost but candidates if the workflow needs to get faster overall.

## R-tree / Rust-migration audit — `src/ccta` and `multimodars/ccta` (2026-07-09)

Re-profiled `label_geometry` on the 25,171-vertex mesh after the `find_centerline_bounded_points`
fix above. New bottleneck order (table below is from a **debug** build — see the B1 entry further
down for why that matters and for release-build numbers after batch A+B1 landed):

| Step | Time |
|---|---|
| `plot_results_key` | 6.446s |
| `remove_occluded_points_ray_triangle` (rca) | 4.353s |
| `_prepare_faces_for_rust` (rca) | 3.184s |
| `clean_outlier_points` (lca) | 0.728s |
| `_final_reclassification` | 0.560s |
| `clean_outlier_points` (rca) | 0.545s |
| `load mesh` | 0.524s |
| `find_centerline_bounded_points_simple` (rca+lca) | 0.236s *(was 22.4s)* |
| `diff rca_removed_points` | 0.177s |
| `_find_aortic_points` (post/pre-clean) | 0.088s / 0.085s |
| **TOTAL** | **16.969s** *(was 38.6s on this mesh size before any fixes)* |

Ran a full audit of `src/ccta/**/*.rs` and `multimodars/ccta/**/*.py` for the same
O(N×M)/O(N²) brute-force pattern that `find_centerline_bounded_points` had, plus a check of
what pure-Python logic in `multimodars/ccta` should move into Rust. Prioritized by expected
impact:

### A. Small effort, ready to apply next
- [x] `rca_removed_points` / `lca_removed_points` diff (`labeling.py`, `p for p in a if p not in
  b`) — **FIXED:** now builds a `set` from the final points list before the membership test.
  O(n×m) → O(n).
- [x] `clean_up_non_section_points` (`src/ccta/adjust_mesh/scale_coronary.rs`, exposed as
  `clean_outlier_points`) — **FIXED:** builds R-trees over `reference_points` and
  `points_to_cleanup` and counts via `locate_within_distance` instead of two nested linear
  scans. O(cleanup×(cleanup+reference)) → O((cleanup+reference) log(...)).
- [x] `remove_occluded_points_ray_triangle_rust` Pass B (`label_coronary.rs`) — **FIXED:** pools
  the (small) excluded-faces' vertices into one R-tree, queried per mesh point.
  O(points × faces_to_exclude) → O(points × log k).

### B. Medium effort, high value
- [x] `_find_faces_for_points` / `_prepare_faces_for_rust` (`labeling.py`) — **FIXED
  (2026-07-09):** deleted both pure-Python functions entirely; added a new Rust binding
  `find_faces_near_points` (`label_coronary.rs` + `ccta_py.rs`) that builds an R-tree over mesh
  vertices (tagged with `GeomWithData`) and does the whole nearest-vertex-match +
  face-membership scan in one Rust pass. `labeling.py` now calls it directly, and
  `tests/test_ccta.py` was updated to test the new binding instead of the deleted functions.
  Re-measured on the 25,171-vertex mesh (release build): **3.184s → 0.025s**. Note: this
  measurement (and the ones below) used `maturin develop --release`; the earlier 3.184s/22.4s/
  4.353s numbers upstream in this doc were against a debug build (`cargo build`'s default) —
  debug-vs-release alone is a large multiplier for numeric Rust code, so don't compare across
  build types when re-profiling. Full `label_geometry` (anomalous_rca=True, `control_plot=False`,
  release build, same mesh) is now **1.378s** end to end, down from 16.969s pre-B1 (same
  build/config) — dominated by `load mesh` (0.685s, I/O) and `_final_reclassification` (0.293s,
  still pure Python).
- [] ~~`remove_occluded_points_ray_triangle_rust` Pass A~~ — **DECLINED (2026-07-09):** after
  batch A + B1 landed and were re-measured against a release build, this dropped to 0.042-0.044s
  on its own (see B1 entry re: debug-vs-release), so it's no longer a real bottleneck. Also
  deliberately leaving the nested-loop-over-all-faces form alone since it reads more clearly than
  a BVH/AABB-pruned version would — not worth the complexity for a step that isn't slow anymore.
- [] `voronoi_partition` (`src/ccta/discretizing/projecting.rs:64-103`) — per mesh point, full
  linear scan of `anchors` to find the nearest one: O(points × anchors). Called once per vessel
  segment from `discretize_vessel_tree`. Same R-tree-over-the-small-side recipe. **On hold** until
  the full `fullworkflow.py` is re-profiled under a release build — the old debug-build estimate
  that motivated this (`discretize_vessel_tree` 4.3-4.5s) may no longer be accurate.
- [] `find_closest_centerline_point_optimized`, `find_cl_points_in_range`, `find_region_points`,
  `symmetric_nn_distance` (all in `src/ccta/adjust_mesh/scale_coronary.rs`) — all do
  nearest/within-radius search against a *fixed* centerline or reference-point set, but worse:
  `centerline_based_diameter_optimization` / `centerline_based_aortic_diameter_optimization`
  call these ~40-80 times in a scaling-parameter sweep, rescanning the same unchanged set from
  scratch on every step. Building one R-tree over the fixed side *once*, outside the sweep,
  is a compounding win — likely the highest-leverage fix in this list per line of code changed,
  but no fresh timing yet since these aren't on the currently-profiled `label_geometry` path.
  **On hold** for the same reason as `voronoi_partition` above — re-profile the full workflow
  under release first.
- [x] `_final_reclassification` + `_find_aortic_points` (`labeling.py`) — **FIXED (2026-07-09):**
  deleted both pure-Python functions; added two new Rust bindings in `label_coronary.rs` +
  `ccta_py.rs`: `find_aortic_points` (exact-match set difference, keyed by `f64::to_bits()` since
  this is an exact-membership test, not a radius query — no R-tree needed here) and
  `final_reclassification` (same adjacency-based label-smoothing logic, reusing the existing
  `build_adjacency_map`). `labeling.py` and `tests/test_ccta.py` updated accordingly; 170 Rust +
  97 Python tests pass. Re-measured on the 25,171-vertex mesh (release build):
  `_final_reclassification` 0.293s → 0.043s (~7x); `_find_aortic_points` pre/post-clean 0.129s+
  0.088s → 0.005s+0.058s. Full `label_geometry` (release, `control_plot=False`, same mesh) is now
  **0.853s** total, of which 0.545s is unavoidable `trimesh` mesh I/O — our own code is under
  0.31s. `remove_labeled_points_from_mesh` / `keep_labeled_points_from_mesh` in `manipulating.py`
  are the same shape of fix but weren't touched — they're not on this profiled path, not
  measured, and out of scope for this pass; still worth doing if that code shows up as hot later.

### C. Known, lower priority for now
- [] `plot_results_key` is now the single biggest line item (6.446s) but it's rendering, not a
  spatial-search problem — separate investigation (e.g. downsampling points before scatter
  plotting) rather than an R-tree fix.
- [] `load mesh` (0.524s) is I/O via `trimesh`'s own loader — not much to do without changing
  file formats.
- [] Cross-file `coord_to_idx` (float-tuple-keyed vertex dict) gets rebuilt from scratch ~10
  times across a full label→scale→stitch pipeline run (`labeling.py:465`,
  `manipulating.py:489,601,955,1000`, `fixing_functions.py:21`, `__init__.py:322`) with no
  caching — architecture/caching cleanup rather than an algorithmic fix.
- [] `manual_hole_fill` (`fixing_functions.py:26`) rebuilds `coord_to_idx` and
  `build_adjacency_map` from scratch on every hole/boundary loop instead of once outside the
  loop — only matters if hole counts get large.
- [] `discretize_vessel_tree`'s side-branch loop (`src/ccta/discretizing/vessel_tree.rs:48-62`)
  isn't parallelized with rayon despite each branch's `discretize_vessel_rs` call being
  independent — same `.iter()` → `.par_iter()` swap already used elsewhere in this module.
- [] `remove_occluded_points_ray_triangle_rust` clones both centerlines twice (once explicitly,
  once again inside `check_centerline`) just to sort them — cheap to avoid, minor win.

## Full-pipeline re-profile + `stitch_ccta_to_intravascular` fix (2026-07-09)

Re-ran the full `fullworkflow.py` under a release build after batch A + B1 + B5 landed.
New bottleneck order (~27.2s total, down from the original 189.9s baseline):

| Step | Time |
|---|---|
| `fix_and_remesh_stitched_mesh` | 17.359s |
| `stitch_ccta_to_intravascular` | 4.090s |
| `align_combined` | 2.196s |
| `label_geometry` (final/initial) | 0.890s / 0.808s |
| `from_file_singlepair` | 0.626s |
| everything else | <0.22s each |

This fully confirmed skipping B2 and holding B3/B4: `discretize_vessel_tree` (contains the B3
`voronoi_partition` target) dropped to 0.203s and the B4 `scale_coronary.rs` sweep functions
combined to 0.157s, purely from the release build — nothing left to fix there. Marked both
resolved-by-context above; B2 marked declined (Pass A dropped to 0.042-0.044s on its own, and
the nested-loop form is more readable than a BVH-pruned version — not worth the complexity for
a step that isn't slow anymore).

- [x] `stitch_ccta_to_intravascular` (`manipulating.py`) — **FIXED:** added internal `_timed`
  instrumentation (same pattern as `label_geometry`) and found 95% of its 4.1-4.3s cost was one
  call: `mesh.fix_normals()` on the freshly-concatenated mesh (~26k verts, 52k faces). Traced
  into trimesh's own source (`trimesh/repair.py` v4.8.3) — `fix_winding` does a Python/NetworkX
  BFS over the face-adjacency graph with several small numpy allocations per edge (~78k edges
  here): the exact same "Python loop + per-item numpy overhead" anti-pattern already fixed
  elsewhere in this module today, just living inside a third-party library instead of our code
  (unlike `fix_and_remesh_stitched_mesh`, which delegates to compiled pymeshlab and was left
  alone). Added `fix_mesh_winding` — a faithful Rust port of the same BFS-consistency algorithm
  — in `ccta_py.rs`, verified byte-for-bit identical output against trimesh's actual
  `fix_winding` on both a consistent and an inconsistent test case, plus against a real
  `icosphere` mesh. Wired in via a new `_fast_fix_normals()` helper (Rust winding fix +
  trimesh's own cheap/vectorized `fix_inversion`) replacing both `fix_normals()` calls in
  `stitch_ccta_to_intravascular`. Result: `fix_normals` 3.898s → 0.105s (~37x);
  `stitch_ccta_to_intravascular` total 4.1-4.3s → **0.444s** (~9-10x). Full pipeline now
  **~23.7s** (~8x under the original baseline). `fix_and_remesh_stitched_mesh` (17.6s,
  pymeshlab, left alone per instruction) is now ~74% of total runtime and the only large item
  left.
- [] `fix_mesh_winding` is general mesh-topology logic (pure `faces` indices, no coordinates
  needed) — `fixing_functions.py:222`'s `mesh_remeshed.fix_normals()` inside
  `fix_and_remesh_stitched_mesh` calls the same slow trimesh path on an even bigger mesh
  (114k faces post-remesh) and could reuse this fix directly. Deliberately **not done** —
  out of scope per instruction to leave `fix_and_remesh_stitched_mesh` alone this pass.

# Mesh combination (v0.3.0/0.4.0)
## New/improved features
- [] `numpy_to_report` function that allows to read in lists for the four variables of interest (frame, phase, measurement_1, measurement_2)
- [] write metadata to .obj files so they can be reconstructed completely to geometry/geometrypair

## Fixes/Improvements
- [] Write a more stable function to find intramural points on aorta (for now works with excluded RCA points)
- [] fix centerline resampling

## Investigate/check logic
- [] When aligning geometry with wall has a rotation in the wall -> logic error for centerline alignment?

# CCTA clinical assessment (v0.4.0)
- [] Automatically sort Centerline in branches.
    - [] add indices to show start and end of a segment to PyCenterline
- [] Create an aortic and coronary wall with a consistent connection
    - [] Create cap for holes in aorta after removing coronaries (layered)
- [] Find bifurcation reference points automatically
    - [] Discretize CCTA geometry in layers following centerline(-branches)
    - [] automatically assess lumen area, minor-, major axis, mla, intramural length etc. from the CCTA mesh.

# Miscellaneous
- [] Integrate coverage measurement (`pytest-cov`) and upload reports to Coveralls or Codecov.
- [] Add a coverage status badge to the README.

# Centerline alignment module (v1.0.0)
- [] Create Centerline directly from mesh

# Myocardial Fusion v2.0.0
- [] Read in LGE/T2 etc. from MRI and map onto CCTA myocardium
- [] Read in Perfusion map from PET/SPECT and load onto CCTA myocardium
- [] Calculate perfusion territories based on coronary anatomy