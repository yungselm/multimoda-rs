# Changelog multimodars

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.1] - 2026-05-04

### Added
- Example data and Jupyter notebooks are now included directly in the repository under
  `examples/` — no longer distributed as a release attachment. Both tutorials link to
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
