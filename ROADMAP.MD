# Python package that has one purpose: Combining any intravascular image modality with CCTA meshes.

# Optimizations

Performance pass on `label_geometry` / `stitch_ccta_to_intravascular` (2026-07-09): full
pipeline dropped from ~190s to ~24s (release build). Full before/after numbers and root causes
are in the CHANGELOG. Considered this pass **closed** after two follow-up investigations both
came back negative-or-marginal — see below. A first round of candidates (`voronoi_partition`,
the `scale_coronary.rs` sweep functions, and the ray-triangle intersection pass) were dropped
earlier the same way, after re-profiling under a release build showed they were never real
bottlenecks.

- [x] ~~`fix_mesh_winding` reuse in `fix_and_remesh_stitched_mesh`~~ — **INVESTIGATED, NOT A
  BOTTLENECK (2026-07-09):** timed both `mesh_remeshed.fix_normals()` calls in
  `fixing_functions.py` against the real dataset — 0.161s out of 18.5s total, negligible.
  Reason: `trimesh.repair.fix_winding` early-exits when `mesh.is_winding_consistent` is already
  `True`, and pymeshlab's isotropic remeshing produces globally consistent winding by
  construction — unlike `stitch_ccta_to_intravascular`'s case, where concatenating several
  separately-oriented mesh pieces genuinely breaks consistency and triggers the expensive BFS.
  `fix_and_remesh_stitched_mesh`'s cost is confirmed to be pymeshlab's own C++ work, not a
  hidden Python bottleneck — the original call to leave it alone was correct.
- [x] ~~`remove_labeled_points_from_mesh` / `keep_labeled_points_from_mesh`~~ — **INVESTIGATED,
  NOT WORTH IT (2026-07-09):** confirmed real (same shape of pure-Python full-mesh adjacency
  loop as `_final_reclassification`) but only 0.211s of ~24s total (<1%), and
  `keep_labeled_points_from_mesh` isn't even called by the real workflow. Deliberately left
  alone — diminishing returns past this point in the pipeline.
- [] Cross-file `coord_to_idx` (float-tuple-keyed vertex dict) gets rebuilt from scratch ~10
  times across a full label→scale→stitch pipeline run with no caching — architecture/caching
  cleanup rather than an algorithmic fix; independent of build type.
- [] `manual_hole_fill` (`fixing_functions.py`) rebuilds `coord_to_idx` and `build_adjacency_map`
  from scratch on every hole/boundary loop instead of once outside the loop.
- [x] ~~`discretize_vessel_tree`'s side-branch loop wasn't parallelized with rayon~~ —
  **FIXED (2026-07-09):** `rca_branches` / `lca_branches` in
  `src/ccta/discretizing/vessel_tree.rs` now use `.par_iter()` instead of `.iter()`
  (`discretize_vessel_rs` is a pure function over `&Centerline` + owned data, safe to
  parallelize). No measurable win on the reference dataset (only ~4-9 side branches), but ready
  for larger vessel trees. Verified: same output values end to end, 175 Rust + 100 Python tests
  pass.
- [x] ~~`remove_occluded_points_ray_triangle_rust` clones both centerlines twice~~ —
  **FIXED (2026-07-09):** `check_centerline` now takes `&Centerline` instead of an owned
  `Centerline`, removing the caller-side `.clone()` at both call sites in
  `label_coronary.rs`; the one remaining `points.clone()` inside `check_centerline` is the only
  clone actually needed (to sort into a new order without mutating the caller's centerline).

# Correctness fixes

- [x] ~~VTP branch-0 ("main vessel") elected by point count instead of arc length~~ — **FIXED
  (2026-07-09):** `read_centerline_vtp` (`src/intravascular/io/input.rs`) sorted VTK `Lines`
  cells by raw point count to decide `branch_id = 0`; on VTP exports with non-uniform per-branch
  sampling density, a physically shorter side branch with more points could out-rank the true
  main vessel. Now sorts by summed 3D arc length instead. `cleanup_vtp_data`'s overlap-trimming
  only ever trims branches *against* `branches[0]` and never trims `branches[0]` itself, so a
  wrong branch-0 election silently truncated the true main vessel's proximal segment while
  leaving the wrongly-elected branch untouched — this was the root cause of reference-point
  selection only working near the first side branch, and of visibly uneven/duplicated centerline
  spacing after `cleanup_vtp_data(smooth=True)`. Applied the same fix (arc length instead of hop
  count) to `calculate_branches`'s tree-diameter BFS (`Centerline::bfs_farthest`), used by the
  non-VTP/CSV branch-detection path, since it had the identical category of bug. Added a
  regression test (`test_read_centerline_vtp_picks_longest_by_arc_length_not_point_count`)
  constructing a short-but-dense vs. long-but-sparse line pair; confirmed the long line now wins
  branch 0. All 176 Rust tests pass.

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