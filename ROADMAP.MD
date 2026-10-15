# Python package that has one purpose: Combining any intravascular image modality with CCTA meshes.

# Optimizations

- [ ] Cross-file `coord_to_idx` (float-tuple-keyed vertex dict) gets rebuilt from scratch ~10
  times across a full label→scale→stitch pipeline run with no caching — architecture/caching
  cleanup rather than an algorithmic fix; independent of build type.
- [ ] `manual_hole_fill` (`fixing_functions.py`) rebuilds `coord_to_idx` and `build_adjacency_map`
  from scratch on every hole/boundary loop instead of once outside the loop.

# Rust core: mesh cutting and labelled-mesh operations

`cut_mesh_at_contour` ([src/ccta/binding/discretizing.rs](src/ccta/binding/discretizing.rs)) returns a
7-tuple of internal bookkeeping (`old_to_new`, `ring`, `centroid`, `cap_faces`, ...) that
`cut_at_slice` ([multimodars/ccta/mesh_regions.py](multimodars/ccta/mesh_regions.py)) finishes in
Python: label remapping, majority-neighbour labels for new vertices, centerline cut, cap extraction.

- [ ] Move the whole cut into Rust (`cut_vessel(mesh, labels, centerline, branch_id, plane, keep, ...)
      -> VesselCut { mesh, labels, centerline, cap, kept, removed_branches }`), labels as named
      index sets (e.g. `BTreeMap<String, Vec<usize>>`, branch maps flattened to `"rca_branches/1"`)
- [ ] Native `inherit_labels` helper (majority label of ring neighbours), reuse in `remove_branch`
- [ ] Drop redundant outputs (`cap_faces` = `fan_faces(ring, centroid)`; `plane.radius` recomputed in Python)
- [ ] End state: native `LabeledMesh { vertices, faces, labels }` pyclass so `sync_results_to_mesh`,
      `map_point_lists`, `remove_labeled_points_from_mesh`, `remove_branch` stop remapping by coordinates
- Open questions: labels as named sets vs per-vertex enum; take a `Contour` or a `CutPlane`

# Mesh combination (v0.3.0/0.4.0)
## New/improved features
- [ ] `numpy_to_report` function that allows to read in lists for the four variables of interest (frame, phase, measurement_1, measurement_2)
- [ ] write metadata to .obj files so they can be reconstructed completely to geometry/geometrypair

## Fixes/Improvements
- [ ] Write a more stable function to find intramural points on aorta (for now works with excluded RCA points)
- [ ] fix centerline resampling

## Investigate/check logic
- [ ] When aligning geometry with wall has a rotation in the wall -> logic error for centerline alignment?

# CCTA clinical assessment (v0.4.0)
- [ ] Automatically sort Centerline in branches.
    - [ ] add indices to show start and end of a segment to PyCenterline
- [x] ~~Create an aortic and coronary wall with a consistent connection~~ — **DONE (0.7.2):**
  anomalous/intramural ostia now get two-half ring conditioning
  (`_condition_ostium_ring_two_half`): the aorta-facing half is rebuilt from the IV frame's own
  contour scaled up by its measured wall thickness, the coronary-facing half is only lightly
  denoised, and the seam into the surrounding mesh is faded over a few layers. See CHANGELOG.
    - [ ] Create cap for holes in aorta after removing coronaries (layered)
- [ ] Find bifurcation reference points automatically
    - [ ] Discretize CCTA geometry in layers following centerline(-branches)
    - [ ] automatically assess lumen area, minor-, major axis, mla, intramural length etc. from the CCTA mesh.

# Miscellaneous
- [ ] Integrate coverage measurement (`pytest-cov`) and upload reports to Coveralls or Codecov.
- [ ] Add a coverage status badge to the README.

# Centerline alignment module (v1.0.0)
- [ ] Create Centerline directly from mesh

# Myocardial Fusion v2.0.0
- [ ] Read in LGE/T2 etc. from MRI and map onto CCTA myocardium
- [ ] Read in Perfusion map from PET/SPECT and load onto CCTA myocardium
- [ ] Calculate perfusion territories based on coronary anatomy
