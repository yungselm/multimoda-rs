# Python package that has one purpose: Combining any intravascular image modality with CCTA meshes.

# Optimizations
- [] Optimize the ray intersection algorithm
- [] Better parallelization strategy for frames alignment by processing in batches (frames only have to be aligned relative to each other not sequentially)

# Mesh combination (v0.3.0)
## New/improved features
- [] "Smooth" the aortic ring of boundary points first, by projecting on plane to shift outliers
- [] Write a more stable function to find intramural points on aorta (for now works with excluded RCA points)
- [] Create an aortic and coronary wall with a consistent connection
- [] Find bifurcation reference points automatically
- [] `numpy_to_report` function that allows to read in lists for the four variables of interest (frame, phase, measurement_1, measurement_2)
- [] write metadata to .obj files so they can be reconstructed completely to geometry/geometrypair

## Fixes/Improvements
- [] Write a more stable function to find intramural points on aorta (for now works with excluded RCA points)
- [] fix centerline resampling

## Investigate/check logic
- [] When aligning geometry with wall has a rotation in the wall -> logic error for centerline alignment?

# CCTA clinical assessment (v0.4.0)
- [] automatically find the required reference points for anomalous alignment
- [] automatically assess lumen area, minor-, major axis, mla, intramural length etc. from the CCTA mesh.
- [] return hausdorff distances per point between aligned ivus mesh and ccta mesh.
- [] function to automatically assign wall thickness to ccta and combined ivus/ccta mesh

# Miscellaneous
- [] Integrate coverage measurement (`pytest-cov`) and upload reports to Coveralls or Codecov.
- [] Add a coverage status badge to the README.

# Centerline alignment module (v1.0.0)
- [] Create Centerline directly from mesh

# Myocardial Fusion v2.0.0
- [] Read in LGE/T2 etc. from MRI and map onto CCTA myocardium
- [] Read in Perfusion map from PET/SPECT and load onto CCTA myocardium
- [] Calculate perfusion territories based on coronary anatomy
