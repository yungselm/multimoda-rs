# Changelog multimodars

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/).

## [0.7.0] - 2026-08-13

### Added
- New `ccta/boundary.py` owning open-boundary ring extraction, shared by the trimming and debug-plot
  paths: `open_boundary_edges`, `order_boundary_rings` (read-only) and `clean_open_boundary`.
- `remove_labeled_points_from_mesh`/`keep_labeled_points_from_mesh` take `target_boundaries`
  (default 1) and now also store the boundary per ring as `boundary_points_1`, `boundary_points_2`,
  … in walk order. Removing a mid-vessel segment leaves two rings, so stitching needs
  `target_boundaries=2`.
- `stitch_ccta_to_intravascular` takes `boundary_point_ratio` (default 1.0): the boundary ring is
  densified to `round(ratio * n_points_iv_cont)` points, so 1.0 gives a 1:1 quad strip.
- `plot_boundary_edges` debug scene, drawing each ring's edges coloured red→blue by walk order.

### Changed
- Ring connectivity now comes from open boundary edges instead of full vertex adjacency, so an
  interior chord can't invent a junction or fragment a ring; rims are selected as whole connected
  components, so a hole merging with a pre-existing opening stays intact.
- Rim vertices that can't form a clean ring (stragglers, hairs, pinch junctions, spikes) are now
  deleted from the mesh, not just skipped in the ring list.
- `stitch_ccta_to_intravascular` assigns whole rings to the two intravascular ends by closest
  centroid pairing, instead of splitting a flat point list per point - which could tear one ring
  across both seams. Fewer than two rings raises `ValueError` instead of `ZeroDivisionError`.
- Both rings are now conditioned identically (projected, smoothed, respaced, densified); previously
  only the proximal ostium ring was, so expect a visible change at the distal seam.
- Anomalous ostia gain a whole-plane correction: when the boundary plane cuts the intravascular
  ostial frame, the plane is slid toward the aorta until it clears by `clamp_overshoot` mm.
- `order_points_list` no longer called during stitching - rings now arrive ordered.

### Fixed
- `_stitch_boundary_ring` emitted `n_iv` triangles where a closed annulus needs `n_boundary + n_iv`,
  leaving one hole per boundary segment at every ratio - the surface only closed because
  `fill_holes` ran afterwards. Replaced by `_stitch_rings`.
- The ostium plane was slid along the vessel axis as a proxy for "toward the aorta", but an
  intramural coronary runs inside the aortic wall, so the lumen lies roughly perpendicular to that
  axis and the ring could be pushed the wrong way. Direction now taken from `aorta_points`.
- Laplacian smoothing shrank boundary rings, worst on coarse ones (~16% calibre at 17 points),
  leaving a pinched distal rim. Smoothing now restores the ring's original calibre.
- Points inserted along a boundary edge left T-junctions; the adjacent face is now retriangulated
  onto its opposite vertex.
- `docs/api/adjust_ccta.rst` referenced the removed `ccta.manipulating` module, so the scaling
  functions were silently missing from the API reference.

## [0.6.1] - 2026-08-11

### Fix
- `remove_branch_overlap` only considered the main branch and didn't clean up the side branches.

## [0.6.0] - 2026-08-07

### Changed
- `Centerline`/`PyCenterline` API reorganised around a load → prepare → branch → order → smooth
  pipeline:
  - `cleanup_vtp_data` removed, replaced by `remove_branch_overlap()` and `trim_start(mm)`.
  - `smooth(sigma)` is now an inherent method (was the Rust-only `utils::smooth_centerline`).
  - New `resample(spacing_mm)` resamples every branch to even arc-length spacing.
  - New `orient_by_max_z()` / `orient_to_reference(reference)` replace the old ad-hoc z-based
    reversal heuristics (`label_coronary.rs`, `centerline_align/preprocessing.rs`,
    `Centerline::check_centerline`) and now also reorder every side branch, not just branch 0 -
    making `check_centerline` fully redundant, so it is removed; call `orient_by_max_z()` instead.
  - New Python helpers `load_centerline(source, name)` and `prepare_centerline(centerline,
    ref_centerline=None, ...)` replace `read_centerline_vtp(...).cleanup_vtp_data(smooth=True)`.
    `ref_centerline` doubles as the "is this a coronary?" signal, driving both which orientation
    method runs and whether branch detection runs.
- `discretize_vessel`/`discretize_vessel_tree` no longer smooth centerlines internally; prepare
  centerlines beforehand via `prepare_centerline` or `.smooth()`.
- `label_geometry` now takes prepared `PyCenterline` objects (`centerline_aorta`,
  `centerline_rca`, `centerline_lca`) instead of path/array unions, and no longer loads or
  orients centerlines internally.

## [0.5.8] - 2026-08-06

### Fixed
- `final_reclassification`'s Logic B judged each removed vertex only by its own immediate
  neighbours, so a contiguous falsely-removed patch - mostly bordered by other removed points, not
  real RCA/LCA - could never clear the 70% restore threshold. Now evaluated per connected
  component against its true external boundary.
- `results["rca_points"]` was built from the uncleaned `final_rca_points_found` instead of
  `clean_outlier_points`'s cleaned output (LCA was already correct) - RCA never got the spatial
  cleanup pass at all.
- That fix exposed a latent issue in `clean_outlier_points`: its purely Euclidean check (no mesh
  topology) can carve an "island" of aorta out of the middle of the RCA patch near an acute
  takeoff. Logic A now runs per connected component in both directions (aorta ↔ RCA/LCA), always
  excluding each label's single largest component (the legitimate main body) from reclassification.

### Internal
- New shared `connected_components`/`component_boundary` helpers and `reclassify_minority_components`
  in `label_coronary.rs`, plus a `keep_largest_connected_component` Rust binding now backing
  `_keep_largest_connected_component` (`labeling.py`).
- New/updated Rust and Python tests covering both fixes.

## [0.5.7] - 2026-08-05

### Changed
- `label`/`label_geometry` split `n_points_intramural` into separate `n_points_takeoff_rca` and `n_points_takeoff_lca` parameters, so the occlusion-removal range can be tuned independently per vessel.
- Renamed `anomalous_rca`/`anomalous_lca` to `acute_takeoff_rca`/`acute_takeoff_lca` (and `n_points_intramural_rca`/`n_points_intramural_lca` to `n_points_takeoff_rca`/`n_points_takeoff_lca`) — the occlusion-removal step also applies to normal coronaries with an acute-angle ostium, not only anomalous/intramural courses, so "anomalous" no longer described the parameter accurately.

### Fixed
- `_stitch_boundary_ring` could reference an out-of-range IV vertex index (`mid == n_iv`) when
  there are more CCTA boundary vertices than IV lumen points; the bridging-triangle index is
  now wrapped with `mid % n_iv`.

## [0.5.6] - 2026-08-03

### Changed
- type stubs were not up to date resulting in `align_manual`still giving the impression to work
 with radians instead of degree

## [0.5.5] - 2026-08-03

### Fixed
- `align_manual` no explicitly states `rotation_angle_deg` instead of `rotation_angle`

## Changed
- `align_manual`, `align_combined` and `align_three_points` all return their rotation angle now.

## [0.5.4] - 2026-08-03 Hotfix
Hotfix of `discretize_vessel_tree()` function, did not include the `removed_rca_points` resulting in weird geometries.

## [0.5.3] - 2026-07-31

Original investigated the possibility to use the radius of the `.vtp` file to run the 
rolling sphere algorithm. However, it was apparent that the radii are not precise, which
led to holes in the labelling process. Therefor aborted and decided to just use two fixed
sphere radii for LCA and RCA.

### Changed
- `find_centerline_bounded_points` now fails on empty centerlines
- `label_geometry` now takes a seperate RCA and LCA radius as input.

### Fixed
- the pre-commit hook now references the version in the current .venv, otherwise resulted in 
sporadic errors.

## [0.5.2] - 2026-07-09

### Fixed
- `label_anomalous_region` could leave a handful of mesh vertices mislabeled into
  `proximal_points` / `distal_points` / `anomalous_points` despite not being mesh-connected to
  that region's main cluster ("islands"), since `find_points_by_cl_region` classifies points
  using coordinate-only heuristics with no notion of mesh topology. Each island then made
  `remove_labeled_points_from_mesh` carve out an extra small hole, corrupting
  `stitch_ccta_to_intravascular`'s boundary-ring ordering. Added
  `_keep_largest_connected_component` (`labeling.py`): builds the mesh's adjacency graph and
  keeps only the largest connected component per region, reassigning dropped points to
  `aorta_points` (removing them from `results[results_key]` too, so they aren't stranded
  outside every scaling step). Verified on the reference dataset: 7 dropped points
  (`anomalous_points` 421 → 414) eliminated 27 extra boundary vertices (82 → 55), cut the
  post-stitch max edge length from 6.65mm to 4.69mm, and are now correctly counted in
  `aorta_points` instead of stranded in `rca_points`.
- `read_centerline_vtp` elected branch 0 ("main vessel") by raw point count instead of arc
  length, so a non-uniformly-sampled VTP export could mislabel a shorter side branch as main.
  Since `cleanup_vtp_data`'s overlap trimming only ever trims other branches against
  `branches[0]`, a wrong election truncated the true main vessel's proximal segment while
  leaving the wrong branch untouched — causing reference points to only make sense near the
  first side branch and visibly uneven/duplicated centerline spacing. Now sorts by summed 3D
  arc length; applied the same fix to `calculate_branches`'s tree-diameter BFS (CSV path).

### Performance
- `label_geometry` and `stitch_ccta_to_intravascular` sped up substantially: several O(N×M)
  brute-force point/face searches were replaced with R-tree-based lookups (new `rstar`
  dependency), and multiple pure-Python per-vertex/per-face loops (nearest-face lookup,
  aortic-point set difference, adjacency-based label reclassification, mesh face-winding
  correction) were moved into new Rust bindings. Full example pipeline
  (`examples/fullworkflow.py`) dropped from ~190s to ~24s on the reference dataset (release
  build). No behaviour change - outputs verified identical, including byte-for-bit parity
  against `trimesh`'s own `fix_winding` on the winding-correction port.

### Internal
- `find_centerline_bounded_points_simple` (`label_coronary::find_centerline_bounded_points`)
  now queries an `rstar::RTree` over the centerline points instead of a nested loop.
- `clean_outlier_points` (`scale_coronary::clean_up_non_section_points`) now uses R-trees
  instead of two nested linear scans.
- `remove_occluded_points_ray_triangle`'s point-vs-excluded-face distance pass now queries an
  R-tree over the excluded faces' vertices instead of scanning every excluded face per point.
- Added Rust bindings `find_faces_near_points`, `find_aortic_points`, and
  `final_reclassification`, replacing the pure-Python `_find_faces_for_points`,
  `_prepare_faces_for_rust`, `_find_aortic_points`, and `_final_reclassification` helpers
  (removed from `labeling.py`).
- Added `fix_mesh_winding`: a Rust port of `trimesh.repair.fix_winding`'s BFS-based
  face-winding-consistency algorithm, used via a new `_fast_fix_normals()` helper in
  `manipulating.py` as a drop-in replacement for `trimesh.Trimesh.fix_normals()` on large
  stitched meshes.
- Type stubs (`multimodars.pyi`) updated for all new Rust bindings.
- `DiscretizedVesselTree::from_results_dict` (`vessel_tree.rs`) now discretizes RCA/LCA side
  branches via `par_iter()` instead of a sequential loop; each branch's `discretize_vessel_rs`
  call is independent.
- `check_centerline` (`label_coronary.rs`) now takes `&Centerline` instead of an owned
  `Centerline`, removing an unnecessary clone at each of its two call sites in
  `remove_occluded_points_ray_triangle_rust`.

## [0.5.1] - 2026-07-09

### Changed
- Refactored `label_coronary` to use nalgebra
- `remove_occluded_points_ray_triangle` now has a defined step size
  with which points are iterated through

## [0.5.0] - 2026-07-07

### Added
**VTP centerline reader**

- `read_centerline_vtp(path)`: parses ASCII-format VTK PolyData (`.vtp`) files
  exported by VMTK and returns a `PyCenterline`.  Branch detection
  is performed automatically; branch 0 is always the longest.  Available at the
  top-level `multimodars` namespace.

**`PyCenterline.cleanup_vtp_data(rm_start_mm, smooth, smooth_sigma)`**

- Removes the run-alongside-main-branch prefix that VTP files attach to every
  side branch, keeping only the true bifurcation junction and the diverged
  portion.  Branches that overlap branch 0 entirely are dropped.
- `rm_start_mm` (default `5.0`): arc-length in mm to additionally trim from the
  start of branch 0 (the aortic inlet region).
- `smooth` / `smooth_sigma`: optional post-trim per-branch Gaussian smoothing via
  a symmetric truncated kernel; half-width `smooth_sigma` is in number of points.

**`CenterlinePoint.radius`**

- New `radius: f64` field on `CenterlinePoint` / `PyCenterlinePoint`, populated
  from the `MaximumInscribedSphereRadius` PointData array in VTP files.

**`prepare_centerlines` - VTP fast path**

- New `vtp_data: bool = False` parameter.  When `True`, `calculate_branches` is
  skipped (branch indices are already set from the VTP file) and only
  `check_centerline` runs to normalise ordering.

### Changed
- `CenterlinePoint.normal` **renamed** to `CenterlinePoint.tangent` (and
  `PyCenterlinePoint.tangent` / `.pyi` stub updated accordingly).  Any code
  accessing `.normal` on a centerline point must be updated.

### Fixed
- `refine_alignment_hausdorff`: when `index_search_range == 0` the function now
  runs a single angle-search pass at the fixed reference index instead of
  iterating over an empty delta range.  A latent negative-underflow bug (casting
  a negative `isize` delta to `usize`) was also fixed with a signed-before-cast
  guard.

### Internal
- Moved `InputData` helper struct to the `intravascular::io` module.
- Removed an unused `geometry` submodule.

## [0.4.3] - 2026-06-04
Small refactor based on cargo modules analysis (https://github.com/regexident/cargo-modules)

### Changed
- Moved downsample points to top layer module `types`
- Renamed duplicate functions and names

## [0.4.2] - 2026-06-04
Large internal refactor to improve readability and decouple modules. No behaviour change.

### Changed
- All native types and Python bindings extracted from scattered files into a unified `src/types/native/` and
  `src/types/binding/` module. 
- Several large files split into focused submodules (`binding/classes.rs` removed, `io/geometry.rs`, `centerline_align.rs`, `to_object.rs` split).
- `Point3D` trait added and shared across `ContourPoint`, `CenterlinePoint`, and `(f64, f64, f64)`, simplified some functions with this. 

## [0.4.1] - 2026-06-02

### Changed
- Rust `use` declarations now follow the idiomatic convention of importing the parent module for
  functions (e.g. `use crate::ccta::discretizing;` -> `discretizing::discretize_vessel_rs(...)`)
  and importing types/structs/enums directly.  No behaviour change; purely internal style cleanup
  across 14 source files.

## [0.4.0] - 2026-05-25

### Added
**Vessel-tree discretization pipeline**

- `prepare_centerlines(rca_cl, lca_cl, results_dict, branch_sigma, control_plot)`: one-call
  wrapper that runs `calculate_branches`, `check_centerline`, and `label_branches` on both
  coronary centerlines.  Returns updated centerlines and a `results` dict extended with
  `rca_points_main`, `rca_points_side_N`, `lca_points_main`, `lca_points_side_N` keys.
- `discretize_vessel_tree(ao_cl, rca_cl, lca_cl, results_dict, ...)`: slices each vessel along
  its centerline at fixed arc-length intervals (`step_size`) and samples `n_points` evenly-spaced
  points from each cross-sectional contour.  Returns a `PyDiscretizedVesselTree` whose
  `rca_references` / `lca_references` attributes hold orientation reference triplets at the
  ostium and every side-branch bifurcation.
- `b_spline` / `bspline_smoothing` / `bspline_degree` parameters in `discretize_vessel_tree`:
  when `b_spline=True`, each discretized contour is replaced with a closed periodic B-spline fit
  (via `scipy.interpolate.splprep`) before reference points are computed.  Smoothing strength is
  controlled by `bspline_smoothing` (0 = exact interpolation; ≈ n_points = gentle;
  ≈ 5 x n_points = strong).
- `find_sharp_angles(cl, branch_id, cos_threshold, control_plot)`: Python wrapper around the
  Rust-bound `PyCenterline.find_sharp_angles` that prints a summary and, when `control_plot=True`,
  opens a trimesh scene highlighting each flagged position in a distinct colour.

**Debug visualisations (trimesh-based)**

All debug plots now use `trimesh.Scene().show()` (native OpenGL) instead of Plotly, matching
the style of `label_geometry`'s control plot and working correctly in all environments
(VS Code, terminal, Jupyter).

- `plot_vessel_tree(tree, pts_per_contour)`: renders discretized contour rings, yellow centroid
  traces, and the full orientation reference triplet for RCA and LCA.  Color coding: silver =
  aorta; steel-blue = RCA main; coral = LCA main; shades of blue/orange = side branches;
  red/orange/magenta = main/CCW/CW reference points.
- `plot_centerline_branches(rca_cl, lca_cl, results_dict)`: shows each centerline branch in a
  distinct colour; optionally overlays the labeled surface-mesh points semi-transparently.
- `plot_centerline_edges(cl, cos_threshold)`: shows all branches with red dots at positions
  flagged by `find_sharp_angles` - useful for deciding where to call `split_branch`.
- `plot_sharp_angles(cl, branch_id, sharp_positions, context_pts)`: shows the full centerline
  dimmed in gray with each flagged position and its neighbours highlighted in a distinct colour.
- `plot_results_key`, `compare_centerline_scaling`: migrated from Plotly to trimesh.

### Changed
**Alignment reference point naming** (`align_three_point`, `align_combined`, `align_manual`)

The three landmark parameters have been renamed for anatomical clarity:

| Old name | New name |
|---|---|
| `aortic_ref_pt` | `main_ref_pt` |
| `upper_ref_pt` | `counterclockwise_ref_pt` |
| `lower_ref_pt` | `clockwise_ref_pt` |

The new names reflect the geometric meaning of each landmark (view from proximal to distal:
the CCW reference lies counter-clockwise of the vessel centre, the CW reference lies clockwise).
Updated across: Rust binding (`src/intravascular/binding/align.rs`), alignment algorithms
(`centerline_align/`), Python wrappers (`_processing.py`), type stubs (`multimodars.pyi`),
documentation (`tutorial_intravascular.rst`), and the CCTA notebook.

- `PyDiscretizedVesselTree.rca_references` / `lca_references` now store triplets as
  `(main_ref, counterclockwise_ref, clockwise_ref)` matching the new parameter order.
- `PyCenterline` struct gains a new read-only field `branch_start_indices : list[int]`
  (index into `points` where each branch begins; `branch_start_indices[0]` is always 0).
  The constructor signature is unchanged - `PyCenterline(points=[...])` populates the field
  automatically.  **Note:** objects pickled with 0.3.x cannot be unpickled under 0.4.x.
- CCTA tutorial (`docs/tutorial_ccta.rst`) updated with new section 2 covering
  `prepare_centerlines`, sharp-angle inspection/correction, and `discretize_vessel_tree`.
- CCTA notebook (`docs/notebooks/ccta_notebook.ipynb`) updated: section 2 added, section 3
  uses `tree.rca_references[0]` and `rca_cl.get_branch(0)` for alignment.

## [0.3.5] - 2026-05-15

### Added
- `clamp_overshoot` parameter in `stitch_ccta_to_intravascular` (default 0.5 mm): for anomalous
  ostia where the boundary-ring plane and the IV plane diverge by ≥ 45°, every proximal boundary
  point is projected onto the IV plane and then pushed at least `clamp_overshoot` mm away from it,
  creating an inward step that softens the stitching angle.
- `_enforce_layer_gap_from_plane`: the two mesh rings adjacent to the clamped boundary are
  pushed radially outward within the IV plane (ring 1: 0.1 mm, ring 2: 0.2 mm) to eliminate
  ridges at the clamping zone.
- Tests for `_clamp_to_plane`, `_enforce_layer_gap_from_plane`, and `_prepare_prox_dist_boundary_pts`
  with idealized geometries (`test_ccta.py`, 17 tests).

### Performance
- Ray-intersection labeling step parallelized via `par_iter()`, reducing wall time on large meshes.

## [0.3.4] - 2026-05-13

### Changed
- `align_three_point`, `align_manual` and `align_combined` are now all generic, taking either `PyGeometryPair` or `PyGeometry` as input
- Test file added `test_intravascular.py` to demonstrate that same results with both

## [0.3.3] - 2026-05-06 Release JOSS paper

### Added
- `CITATION.cff` with full author list, ORCIDs, and preferred citation for JOSS/Zenodo archival.
- Zenodo DOI badge added to README.
- Created a release coupled to JOSS publication

## [0.3.2] - 2026-05-05

### Added
- Benchmark suite (`benchmarks/`) with two scripts: `benchmark_bruteforce_stepsize.py`
  (step-size scaling, `from_file_full`) and `benchmark_cpu_scaling.py` (core-count
  scaling, `from_array_single`). Results saved to `benchmarks/results/`.
- Benchmark documentation page (`docs/benchmark.rst`) with result figures and table,
  linked in the ReadTheDocs table of contents.

### Changed
- `paper/paper.md` Performance section updated with quantified benchmark results and
  corrected parallelization description.

### Performance
- Intra-pullback alignment (`search_range`) now parallelizes over candidate rotation
  angles via `par_iter` instead of the point-rotation loop. The inner cost-function
  loop is now sequential. This provides enough rayon tasks per frame step to saturate
  all cores: bruteforce scales 6.5× and optimized 4.2× from 2 to 16 cores (previously
  ~1.4× for both). Combined algorithmic + parallelization gain reaches **38.5×** vs.
  brute-force at 2 cores.

## [0.3.1] - 2026-05-04

### Added
- Example data and Jupyter notebooks are now included directly in the repository under
  `examples/` - no longer distributed as a release attachment. Both tutorials link to
  the `examples/` directory and cross-reference the rendered notebook pages.
- CCTA and intravascular Jupyter notebooks (`docs/notebooks/`) are now rendered and served
  in the documentation via `myst-nb`, accessible as `:doc:` references from the tutorials.
- Detailed explanation of the rolling-sphere and ray-casting labeling algorithms added to
  the CCTA tutorial, including annotated figures.

### Changed
- README condensed: developer-setup section added, key workflow images included, installation
  and quickstart sections tightened.
- `plotly` added as a documentation dependency for interactive notebook figures.
- Improved all print statements in the whole package.

### Fixed
- CI pipeline now correctly installs optional dependency groups required for the full test
  suite.
- Notebook tests no longer fail with `AttributeError: MeshSet has no attribute
  'meshing_close_holes'` on bare Linux runners. pymeshlab silently skips plugin loading when
  `libopengl0` is absent, leaving `MeshSet` with no filter methods; the CI now installs
  `libopengl0` before the notebook test job runs.

### Performance
- Rust labeling step migrated to `par_iter()` for parallel vertex classification, reducing
  wall time on large meshes.

## [0.3.0] - 2026-04-30

### Added
- `stitch_ccta_to_intravascular()`: fuses an aligned intravascular `PyGeometry` with a CCTA
  trimesh by triangulating a patch between the open boundary ring of the CCTA mesh and the
  proximal/distal contour of the intravascular geometry. Supports two boundary-ring alignment
  modes (`"nearest_iv"` and `"highest_z"`) for proximal and distal ends independently.
- `fix_and_remesh_stitched_mesh()`: repairs the stitched surface (non-manifold edges/vertices,
  open holes) and applies isotropic remeshing via `pymeshlab`. Requires the optional
  `multimodars[meshlab]` extra.
- `manual_hole_fill()`: flat-fills remaining open holes in a trimesh after automatic repair.
- `remove_labeled_points_from_mesh()`: deletes one or more labeled vertex regions from the mesh, remaps faces, and returns the open boundary ring as `"boundary_points"` for downstream stitching.
- `keep_labeled_points_from_mesh()`: retains only the vertices of a specified labeled region,
  discarding all others.
- `sync_results_to_mesh()`: refreshes all coordinate lists in the results dictionary after
  `scale_region_centerline_morphing()` moves vertices, keeping labels consistent with the
  updated mesh.
- `find_aortic_wall_scaling()`: computes the optimal radial scaling factor for the aortic wall
  region by minimising the distance to the first intravascular frame whose lumen elliptic ratio
  drops below 1.3 (transition from intramural to free-segment lumen).
- `export_section_stl()`: exports the full mesh or a labeled sub-region (`"all"`, `"aorta"`,
  `"rca"`, `"lca"`) as an STL file.
- `plot_results_key()`: opens an interactive 3-D trimesh scene visualising selected labeled
  regions with colour coding (yellow = aorta, blue = RCA, green = LCA, red = removed/intramural,
  cyan = proximal, magenta = distal, orange = anomalous).
- Tests for all new CCTA functions.
- `mypy` added to pre-commit hooks for static type checking.

### Changed
- Rust-side CCTA STL reader (`src/ccta/io/input.rs`) removed; mesh I/O is now handled
  entirely via Python `trimesh`, eliminating the `stl_io` dependency.
- `pymeshlab` and `pyglet` are now optional dependencies; the core package installs without
  them. Install extras with `pip install 'multimodars[meshlab]'`.
- Debug-plot helpers moved to a dedicated `ccta.debug_plots` module.
- CCTA Python module refactored into focused sub-modules (`labeling`, `manipulating`,
  `fixing_functions`, `debug_plots`).
- Parameter order updated in several CCTA API calls for consistency.
- `diastole` argument removed from functions where it was not used.
- Labeling of aorta boundary points improved to avoid incorrect assignment at mesh edges.
- Type stubs updated and extended to cover all new public functions.
- Clippy lints applied across the Rust codebase (non-breaking).

### Fixed
- 3-D area and elliptic ratio calculation corrected for non-planar contours.
- Mesh face normals corrected after stitching to ensure consistent outward orientation.

### Documentation
- CCTA tutorial completely rewritten to cover the full stitching workflow:
  `remove_labeled_points_from_mesh` → `stitch_ccta_to_intravascular` →
  `fix_and_remesh_stitched_mesh` → Taubin smoothing → `export_section_stl` → re-labeling.
- Intravascular tutorial reviewed and completed: section numbering corrected (1–7),
  CSV-vs-numpy workflow note added, column descriptions with units added to data tables,
  Sphinx cross-references added throughout, `translate()` tuple-argument bug fixed.


## [0.2.3] - 2026-03-06

### Added
- Type stubs for the Rust-implemented interface (via maturin)
- `PyFrame.get_frame_at_z`: retrieve a frame by its z position
- `PyFrame.get_frame_at_index`: retrieve a frame by its index
- `PyFrame.replace_frame`: replace an existing frame with another

### Changed
- CI pipeline now tests against Python 3.10 through 3.13
- Updated documentation to reflect the current interface and Python version requirements
- Added direct download links to example data in the README and documentation
- Migrated to modern NumPy-style type annotations throughout the Python codebase
- All imports are now top-level across all Python modules

## [0.2.2] - 2026-03-03

### Changed
- Uploaded new example data
- Updated docs to match the new example data

## [0.2.1] - 2026-01-12

### Changed
- Automatic version update based on Cargo.toml

## [0.2.0] - 2026-01-11

### Added
- Functionality to read in CCTA .stl files and label them according to centerline object
- Functionality to label the CCTA geometry additionally by an aligned intravascular aligned geometry.
- Functionality to better align intravascular geometry to a centerline using additionally the CCTA vertices `align_combined`
- Functionality to radially scale different regions radially to their corresponding centerline
- Algorihtm for finding the best radial scaling factor for distal and proximal region compared to the aligned geometry
- Algorithm for finding the best radial scaling factor for the aorta to minimize the distance between "wall" of `PyGeometry` object and the identified intramural vertices

### Changed
- Updated documentation to include a tutorial for the new functionality
- Reduced the amount of testing data to reduce size of the package
- Removed the examples folder from the repository and coupled to release (.zip file)

## [0.1.2] - 2025-11-21

### Changed 
- intravascular::processing::align_within::fill_holes() now does not panic with gaps > 2
  frames, but instead still fixes the holes and returns a warning to the user to check integrity of the data.

## [0.1.1] - 2025-11-17

### Changed
- Update documentation to match new functionality

## [0.1.0] - 2025-11-17

### Added
- Fill-holes functionality: when a geometry has a large gap between frames,
  up to two frames will be inserted to improve continuity.
- Geometry integrity tests: after aligning frames within a geometry,
  several integrity checks are performed before continuing the pipeline.
- `InputData`: the entire pipeline can be driven from an `InputData` object
  that includes coordinates for different contours.
- Enum for contour types: `Lumen`, `EEM`, `sidebranch`, `calcification`,
  `wall`, `catheter`.
- `PyFrame` struct: each frame can hold multiple contour types which are
  processed in parallel.

### Changed
- Full architecture refactor:
  - Raw data is read and ordered into a `Geometry` struct.
  - The `Geometry` struct undergoes integrity checks to ensure data quality.
  - Alignment is performed first *within* geometries, then *between* geometries.
  - Modules are loosely coupled to improve maintainability.
- Parallelization model updated: each geometry is processed in parallel;
  within-geometry alignment runs first, then between-geometry alignment runs in parallel.
- Multiple bug fixes in the alignment algorithms.

### Fixed
- Various stability and indexing bugs discovered during alignment testing.
