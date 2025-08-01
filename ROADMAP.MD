# Python package that has one purpose: Combining any intravascular image modality with CCTA meshes.
# Base module (aligning frames withing IVUS/OCT)
- [x] Bind all Rust structs to python objects
- [x] Single mesh alignment (ignoring systole and diastole (for OCT))
- [x] Functions should return obj, as python objects
- [] Walls not rotated together with contours, and also catheter. 
    here a helper function that performs operations on all ContourPoints
    inside of a Geometry regardless of type, should solve the problem.

# Centerline alignment module
- [x] Centerline resampling based on geometry
- [x] 3-Point alignment versus manual (degree/centerline point)
- [] versus Hausdorff-alignment (v0.2.0)
(- [] Create Centerline directly from mesh) (v0.2.0)

# Mesh combination (v0.2.0)
- [] Read in CCTA mesh (STL/Obj)
- [] Assign regions based on Centerlines (RCA, LCA, Aorta)
- [] Shrink/Expand CCTA to best match Aortic wall and shrink/expand coronary to match iv images
- [] ensure works for IVUS and OCT

# v0.1.0 desired functionality:
- [x] .from_file(mode = {full, single}, "rest dir", "stress dir", "dia dir", "sys dir")
- [x] .from_array(rest, stress, dia, sys)
- [x] .reorder -> initial alignment, without breathing correction, reorder, checks for better order then reruns
- [x] .to_centerline(mode = {three_point, hausdorff, manual}) -> if three point provide three tuples, hausdorff nothing, manual provide centerline point and rotation (v0.0.2 for hausdorff)
- [] .get_array()
- [x] .to_array() for all datatypes (GeometryPair, Geometry, Contour, Point)
- [] .to_obj() writes contours/catheters/walls to .obj file

# Miscellaneous
- [x] change hardcoded catheter settings
- [x] better summarized print statement
- [] for fixture use a idealized geometry to check functionality
- [] reference point is not rotated and translated

# Prepublish checklist:
- [] tests/ with unit tests checking that importing and running every public function works
- [] Integrate coverage measurement (e.g. pytest-cov) and upload reports to Coverall or Codecov
- [] Status badge for coverage to README
- [] Add testing to the CI workflow
- [x] Auto-generated API docs
- [] High-level "Getting Started" guide
- [x] CONTRIBUTING.md

- [] Example scripts Jupyter notebook?
- [] mypy in pre-commit hooks