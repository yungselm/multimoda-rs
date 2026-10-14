# Review: functions added on 2026-10-14

Everything new since `60b0093` (the last commit before today), including the uncommitted
restructuring. Functions that were only moved are under "Moved or rewritten", renames under
"Renamed".

## Cutting inlets and outlets

### Python
- [ ] `cut_at_slice` — [multimodars/ccta/mesh_regions.py:498](multimodars/ccta/mesh_regions.py#L498): cuts mesh + centerline at a tree slice, returns `(results, centerline, cap)`
- [ ] `manual_hole_fill` (now calls Rust) — [multimodars/ccta/postprocessing.py:16](multimodars/ccta/postprocessing.py#L16)

### Bindings
- [ ] `cut_mesh_at_contour` — [src/ccta/binding/discretizing.rs:227](src/ccta/binding/discretizing.rs#L227)
- [ ] `fill_holes_fan` — [src/ccta/binding/morphing.rs:220](src/ccta/binding/morphing.rs#L220)
- [ ] `PyCenterline.cut_branch` — [src/types/binding/py_centerline.rs:276](src/types/binding/py_centerline.rs#L276)

### Rust: `src/ccta/discretizing/cutting.rs`
- [ ] `enum Keep` — [cutting.rs:13](src/ccta/discretizing/cutting.rs#L13)
- [ ] `struct CutPlane` — [cutting.rs:22](src/ccta/discretizing/cutting.rs#L22)
- [ ] `CutPlane::from_contour` — [cutting.rs:34](src/ccta/discretizing/cutting.rs#L34)
- [ ] `CutPlane::dist` — [cutting.rs:69](src/ccta/discretizing/cutting.rs#L69)
- [ ] `struct MeshCut` — [cutting.rs:76](src/ccta/discretizing/cutting.rs#L76)
- [ ] `cut_mesh_at_plane` — [cutting.rs:105](src/ccta/discretizing/cutting.rs#L105)
- [ ] `find_ring` — [cutting.rs:276](src/ccta/discretizing/cutting.rs#L276)
- [ ] `split_face` — [cutting.rs:306](src/ccta/discretizing/cutting.rs#L306)
- [ ] `flood_fill` — [cutting.rs:341](src/ccta/discretizing/cutting.rs#L341)
- [ ] `polygon_centroid` — [cutting.rs:376](src/ccta/discretizing/cutting.rs#L376)
- [ ] `check_star_shaped` — [cutting.rs:393](src/ccta/discretizing/cutting.rs#L393)
- [ ] `compact` — [cutting.rs:412](src/ccta/discretizing/cutting.rs#L412)
- [ ] `sorted` — [cutting.rs:440](src/ccta/discretizing/cutting.rs#L440)

### Rust: `src/ccta/discretizing/slicing.rs`
- [ ] `cut_triangle_by` — [slicing.rs:91](src/ccta/discretizing/slicing.rs#L91)
- [ ] `PlaneFrame::new` — [slicing.rs:193](src/ccta/discretizing/slicing.rs#L193)
- [ ] `Chain.edges` field, and `Crossing`, `Segment`, `chain_segments`, `to_2d`, `signed_area`, `contains_origin` made `pub(super)`

### Rust: `src/ccta/morphing/fill_holes.rs`
- [ ] `boundary_loops` — [fill_holes.rs:7](src/ccta/morphing/fill_holes.rs#L7)
- [ ] `fan_faces` — [fill_holes.rs:48](src/ccta/morphing/fill_holes.rs#L48)
- [ ] `polyline_centroid` — [fill_holes.rs:57](src/ccta/morphing/fill_holes.rs#L57)
- [ ] `fill_holes_fan` — [fill_holes.rs:76](src/ccta/morphing/fill_holes.rs#L76)

### Rust: centerline
- [ ] `Centerline::cut_branch` — [src/types/native/centerline.rs:110](src/types/native/centerline.rs#L110)

## Writing centerlines as VTP

- [ ] `write_centerline_vtp` (Python wrapper) — [multimodars/_processing.py:1441](multimodars/_processing.py#L1441)
- [ ] `write_centerline_vtp` (binding) — [src/intravascular/binding/functions.rs:1580](src/intravascular/binding/functions.rs#L1580)
- [ ] `write_centerline_vtp` (Rust) — [src/intravascular/io/output.rs:320](src/intravascular/io/output.rs#L320)
- [ ] `VTP_RADIUS_ARRAY` — [src/intravascular/io/input.rs:261](src/intravascular/io/input.rs#L261)
- [ ] `read_ascii_vtp` — [src/intravascular/io/input.rs:265](src/intravascular/io/input.rs#L265)
- [ ] `split_cells` — [src/intravascular/io/input.rs:351](src/intravascular/io/input.rs#L351)
- [ ] `extract_section`, `dataarray_text`, `parse_nums`: pulled out of `read_centerline_vtp`, unchanged — [input.rs:298](src/intravascular/io/input.rs#L298)

## CCTA restructuring (uncommitted)

### New
- [ ] `remove_occluded_points` (binding) — [src/ccta/binding/labeling.rs:111](src/ccta/binding/labeling.rs#L111)
- [ ] `split_occluded_points` — [src/ccta/labeling/coronary.rs:205](src/ccta/labeling/coronary.rs#L205)
- [ ] `register` in each binding module — [labeling.rs:21](src/ccta/binding/labeling.rs#L21), [morphing.rs:10](src/ccta/binding/morphing.rs#L10), [discretizing.rs:13](src/ccta/binding/discretizing.rs#L13)
- [ ] `label_geometry` occlusion step inlined — [multimodars/ccta/labeling/core.py:156](multimodars/ccta/labeling/core.py#L156)
- [ ] `label_anomalous_region` largest-component step inlined — [multimodars/ccta/labeling/core.py:337](multimodars/ccta/labeling/core.py#L337)

### Moved or rewritten
- [ ] `build_adjacency_map` (native, was the binding body) — [src/ccta/labeling/topology.rs:15](src/ccta/labeling/topology.rs#L15)
- [ ] `keep_largest_connected_component` (native, was the binding body) — [src/ccta/labeling/topology.rs:60](src/ccta/labeling/topology.rs#L60)
- [ ] `bits_key`, `connected_components` (from `label_coronary.rs`) — [src/ccta/labeling/topology.rs:10](src/ccta/labeling/topology.rs#L10)
- [ ] `fix_mesh_winding` (native, was the binding body) — [src/ccta/morphing/winding.rs:6](src/ccta/morphing/winding.rs#L6)
- [ ] `find_points_by_cl_region_rs`, `PullbackRegions`, `clean_up_non_section_points` (from `scale_coronary.rs`) — [src/ccta/labeling/pullback.rs:34](src/ccta/labeling/pullback.rs#L34)
- [ ] Thin binding wrappers now calling native code: `build_adjacency_map` [labeling.rs:334](src/ccta/binding/labeling.rs#L334), `keep_largest_connected_component` [labeling.rs:359](src/ccta/binding/labeling.rs#L359), `fix_mesh_winding` [morphing.rs:197](src/ccta/binding/morphing.rs#L197)

### Renamed
- [ ] `find_centerline_bounded_points_simple` → `find_centerline_bounded_points` — [binding](src/ccta/binding/labeling.rs#L62), [Python wrapper](multimodars/_processing.py#L1480)
- [ ] `adjust_diameter_centerline_morphing_simple` → `adjust_diameter_centerline_morphing` — [src/ccta/binding/morphing.rs:37](src/ccta/binding/morphing.rs#L37)
- [ ] `remove_occluded_points_ray_triangle_rust` → `remove_occluded_points_ray_triangle` (native) — [src/ccta/labeling/coronary.rs](src/ccta/labeling/coronary.rs)

### Removed
- [ ] Bindings `find_faces_near_points`, `remove_occluded_points_ray_triangle`, `smooth_mesh_labels`
- [ ] `multimodars/ccta/labeling/helpers.py` (`_apply_occlusion_removal`, `_keep_largest_connected_component`)

## Tests

- [ ] `TestCutAtSlice` (7 tests) + `_ring_plane_residual` — [tests/test_ccta.py:2924](tests/test_ccta.py#L2924)
- [ ] `test_write_centerline_vtp_round_trip` — [tests/test_ccta.py:3026](tests/test_ccta.py#L3026)
- [ ] `_keep_largest_connected_component` test helper — [tests/test_ccta.py:527](tests/test_ccta.py#L527)
- [ ] Cutting (9 tests + `tube`, `z_plane`, `normal`, `assert_consistent_manifold`) — [src/ccta/discretizing/cutting.rs:449](src/ccta/discretizing/cutting.rs#L449)
- [ ] Fan fill (4 tests + `open_box`, `is_closed`) — [src/ccta/morphing/fill_holes.rs:92](src/ccta/morphing/fill_holes.rs#L92)
- [ ] `cut_branch` (4 tests + `cut_test_centerline`, `end_point`) — [src/types/native/centerline.rs:1555](src/types/native/centerline.rs#L1555)
- [ ] VTP writer (3 tests) — [src/intravascular/io/output.rs:420](src/intravascular/io/output.rs#L420)
- [ ] `test_split_occluded_points_removes_the_wall_nearest_the_aorta` + `line_centerline` — [src/ccta/labeling/coronary.rs:784](src/ccta/labeling/coronary.rs#L784)
