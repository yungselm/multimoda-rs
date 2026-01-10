# Changelog multimodars

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/).

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
