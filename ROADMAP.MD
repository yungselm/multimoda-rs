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

- [] `find_centerline_bounded_points_simple` (Rust, rca+lca combined): 22.4s → 51.1s.
  Largest single cost in both calls; roughly linear with `points_list × centerline points`,
  but still the top target for algorithmic improvement (spatial index instead of brute rolling
  sphere check).
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