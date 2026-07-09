# Python package that has one purpose: Combining any intravascular image modality with CCTA meshes.

# Optimizations

Performance pass on `label_geometry` / `stitch_ccta_to_intravascular` (2026-07-09): full
pipeline dropped from ~190s to ~24s (release build). Full before/after numbers and root causes
are in the CHANGELOG. Remaining candidates, deferred because they weren't real bottlenecks once
the release-build baseline was corrected:

- [] `voronoi_partition` (`src/ccta/discretizing/projecting.rs`) — full linear scan of `anchors`
  per mesh point; same R-tree-over-the-small-side recipe as the fixes already applied elsewhere.
- [] `find_closest_centerline_point_optimized`, `find_cl_points_in_range`, `find_region_points`,
  `symmetric_nn_distance` (`src/ccta/adjust_mesh/scale_coronary.rs`) — rescan a *fixed*
  centerline/reference set from scratch on every step of a ~40-80 iteration scaling sweep;
  building the R-tree once outside the loop would compound. Likely the highest-leverage fix left
  per line of code changed, if these ever show up as hot in a real profile.
- [] `fix_mesh_winding` (the Rust BFS winding-consistency fix added this pass) could also replace
  `mesh_remeshed.fix_normals()` inside `fix_and_remesh_stitched_mesh`
  (`multimodars/ccta/fixing_functions.py`) — same trimesh bottleneck, on an even bigger mesh
  (114k faces post-remesh). Deliberately not done — out of scope for that pass.
- [] `plot_results_key` (control-plot rendering) is `label_geometry`'s biggest line item once
  everything else is fixed, but it's rendering, not spatial search — separate investigation
  (e.g. downsampling points before scatter plotting).
- [] Cross-file `coord_to_idx` (float-tuple-keyed vertex dict) gets rebuilt from scratch ~10
  times across a full label→scale→stitch pipeline run with no caching — architecture/caching
  cleanup rather than an algorithmic fix.
- [] `manual_hole_fill` (`fixing_functions.py`) rebuilds `coord_to_idx` and `build_adjacency_map`
  from scratch on every hole/boundary loop instead of once outside the loop.
- [] `discretize_vessel_tree`'s side-branch loop (`src/ccta/discretizing/vessel_tree.rs`) isn't
  parallelized with rayon despite each branch's call being independent.
- [] `remove_occluded_points_ray_triangle_rust` clones both centerlines twice unnecessarily just
  to sort them.
- [] `remove_labeled_points_from_mesh` / `keep_labeled_points_from_mesh` (`manipulating.py`) are
  the same shape of pure-Python full-mesh adjacency loop as the ones already moved to Rust, but
  weren't touched since they weren't on the profiled hot path.
- [] Optimize the ray intersection algorithm
- [] Better parallelization strategy for frames alignment by processing in batches (frames only have to be aligned relative to each other not sequentially)

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