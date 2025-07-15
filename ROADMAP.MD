# Python package that has one purpose: Combining any intravascular image modality with CCTA meshes.
# Base module (aligning frames withing IVUS/OCT)
- [x] Bind all Rust structs to python objects
- [x] Single mesh alignment (ignoring systole and diastole (for OCT))
- [x] Functions should return obj, as python objects
- [] Walls not rotated together with contours, and also catheter. 
    here a helper function that performs operations on all ContourPoints
    inside of a Geometry regardless of type, should solve the problem.

# Centerline alignment module
- [] Centerline resampling based on geometry
- [] 3-Point alignment versus Hausdorff-alignment versus manual (degree/centerline point)
(- [] Create Centerline directly from mesh)

# Mesh combination
- [] Read in CCTA mesh (STL/Obj)
- [] Assign regions based on Centerlines (RCA, LCA, Aorta)
- [] Shrink/Expand CCTA to best match Aortic wall and shrink/expand coronary to match iv images
- [] ensure works for IVUS and OCT

# v0.1.0 desired functionality:
- [] .from_file(mode = {full, single}, "rest dir", "stress dir", "dia dir", "sys dir")
- [] .from_array(rest, stress, dia, sys)
- [] .frame_align / .frame_reorder -> initial alignment, without breathing correction, reorder, checks for better order then reruns
- [] .to_centerline(mode = {three_point, hausdorff, manual}) -> if three point provide three tuples, hausdorff nothing, manual provide centerline point and rotation
- [] .get_array()

# Miscellaneous
- [x] change hardcoded catheter settings
- [] better summarized print statement
-- [x] input/mod.rs
-- []

# Prepublish checklist:
- [] tests/ with unit tests checking that importing and running every public function works
- [] Integrate coverage measurement (e.g. pytest-cov) and upload reports to Coverall or Codecov
- [] Status badge for coverage to README
- [] Add testing to the CI workflow
- [] Auto-generated API docs
- [] High-level "Getting Started" guide
- [] CONTRIBUTING.md

- [] Example scripts Jupyter notebook?
- [] mypy in pre-commit hooks