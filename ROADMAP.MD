# Python package that has one purpose: Combining any intravascular image modality with CCTA meshes.
# Base module (aligning frames withing IVUS/OCT)
- [x] Bind all Rust structs to python objects
- [x] Single mesh alignment (ignoring systole and diastole (for OCT))
- [x] Functions should return obj, as python objects
- [x] Walls not rotated together with contours, and also catheter. 
    here a helper function that performs operations on all ContourPoints
    inside of a Geometry regardless of type, should solve the problem.
- [x] double check resampling for dia and sys

# Centerline alignment module
- [x] Centerline resampling based on geometry
- [x] 3-Point alignment versus manual (degree/centerline point)
- [x] versus Hausdorff-alignment (v0.2.0)
(- [] Create Centerline directly from mesh) (v1.0.0)

# Mesh combination (v0.2.0)
- [x] Read in CCTA mesh (STL/Obj)
- [x] Create closed mesh for intravascular segment
- [x] Assign regions based on Centerlines (RCA, LCA, Aorta)
- [x] ensure works for IVUS and OCT
- [x] Lable points inside of anomalous region, proximal and distal to region
- [x] use the cleanup_section_points() also on LCA points
- [x] Shrink/expand coronary to match intravascular images
- [x] Shrink/Expand CCTA to best match aortic wall
- [] Write a more stable clean up function by going along every face of a vertix adn then check if most vertices are from the same type
- [] Remove points inside of anomalous region, connect remaining points to proximal and distal points, without resampling of the anomalous region
- [] Create ostium frames for smoother connection to aorta (define range cl points, and function of lumen increase)
- [] Write Pytests for the new functions
- [] Write a more stable function to find intramural points on aorta (for now works with excluded RCA points)

# CCTA clinical assessment (v0.3.0)
- [] automatically find the required reference points for alignment
- [] automatically assess lumen area, minor-, major axis, mla, intramural length etc. from the CCTA mesh.
- [] return hausdorff distances per point between aligned ivus mesh and ccta mesh.

# v0.1.0 desired functionality:
- [x] .from_file(mode = {full, single}, "rest dir", "stress dir", "dia dir", "sys dir")
- [x] .from_array(rest, stress, dia, sys)
- [x] .reorder -> initial alignment, without breathing correction, reorder, checks for better order then reruns
- [x] .to_centerline(mode = {three_point, hausdorff, manual}) -> if three point provide three tuples, hausdorff nothing, manual provide centerline point and rotation (v0.0.2 for hausdorff)
- [x] .get_array()
- [x] .to_array() for all datatypes (GeometryPair, Geometry, Contour, Point)
- [x] .to_obj() writes contours/catheters/walls to .obj file

# Miscellaneous
- [x] change hardcoded catheter settings
- [x] better summarized print statement
- [x] for fixture use a idealized geometry to check functionality
- [x] reference point is not rotated and translated

# Prepublish checklist:
- [x] tests/ with unit tests checking that importing and running every public function works
- [x] Add testing to the CI workflow
- [x] Auto-generated API docs
- [x] High-level "Getting Started" guide
- [x] CONTRIBUTING.md
- [x] Example scripts Jupyter notebook?
- [] make stable version update by reading in a __version__ file
- [] Integrate coverage measurement (e.g. pytest-cov) and upload reports to Coverall or Codecov
- [] Status badge for coverage to README
- [] mypy in pre-commit hooks
- [] move example data to a downloadable folder in release and adjust tests to only use fixtures