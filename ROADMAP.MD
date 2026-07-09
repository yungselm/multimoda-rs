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
- [] `plot_results_key` (control-plot rendering) was `label_geometry`'s biggest line item in the
  audit table, but that number is from a debug build and `control_plot=False` has been used in
  every release-build measurement since — its real cost under release is unverified. Lowest
  priority of what's left; only worth profiling if `control_plot=True` usage becomes a
  complaint.
- [] Cross-file `coord_to_idx` (float-tuple-keyed vertex dict) gets rebuilt from scratch ~10
  times across a full label→scale→stitch pipeline run with no caching — architecture/caching
  cleanup rather than an algorithmic fix; independent of build type.
- [] `manual_hole_fill` (`fixing_functions.py`) rebuilds `coord_to_idx` and `build_adjacency_map`
  from scratch on every hole/boundary loop instead of once outside the loop.
- [] `discretize_vessel_tree`'s side-branch loop (`src/ccta/discretizing/vessel_tree.rs`) isn't
  parallelized with rayon despite each branch's call being independent.
- [] `remove_occluded_points_ray_triangle_rust` clones both centerlines twice unnecessarily just
  to sort them.
- [] Better parallelization strategy for frames alignment by processing in batches (frames only
  have to be aligned relative to each other not sequentially) — intravascular alignment, a
  different subsystem from the CCTA work above; unrelated and unexamined this pass.

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