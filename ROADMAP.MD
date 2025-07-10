# Python package that has one purpose: Combining any intravascular image modality with CCTA meshes.
# Base module (aligning frames withing IVUS/OCT)
- [x] Bind all Rust structs to python objects
- [x] Single mesh alignment (ignoring systole and diastole (for OCT))
- [x] Functions should return obj, as python objects

# Centerline alignment module
- [] Centerline resampling based on geometry
- [] 3-Point alignment versus Hausdorff-alignment versus manual (degree/centerline point)
(- [] Create Centerline directly from mesh)

# Mesh combination
- [] Read in CCTA mesh (STL/Obj)
- [] Assign regions based on Centerlines (RCA, LCA, Aorta)
- [] Shrink/Expand CCTA to best match Aortic wall and shrink/expand coronary to match iv images
- [] ensure works for IVUS and OCT


# final functions list:
- [] .from_file(mode = {full, single}, "rest dir", "stress dir", "dia dir", "sys dir")
- [] .from_array(rest, stress, dia, sys)
- [] .frame_align / .frame_reorder -> initial alignment, without breathing correction, reorder, checks for better order then reruns
- [] .to_centerline(mode = {three_point, hausdorff, manual}) -> if three point provide three tuples, hausdorff nothing, manual provide centerline point and rotation
- [] .get_array()

# Miscellaneous
- [] change hardcoded catheter settings