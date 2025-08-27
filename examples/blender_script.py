# overhauled_blender_4x_animation.py
# Compatible with Blender 4.x (4.4)
import bpy
import os
import math
import bmesh
from mathutils import Vector, Matrix, Quaternion

# -----------------------------
# Configuration - update this
# -----------------------------
obj_directory = r"D:/00_coding/multimoda-rs/data/output/rest"
object_end_frame = 60
z_orbit_frames = 240
y_orbit_frames = 240
total_frames = object_end_frame + z_orbit_frames + y_orbit_frames
frame_rate = 60

# Lighting config
light_radius = 10.0
num_lights = 12
light_energy = 1000.0   # Blender 4's lights are more physically scaled; increase if too dim.

# -----------------------------
# Utilities
# -----------------------------
def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except Exception:
        pass

def clean_scene():
    """Remove all objects, lights, cameras and collections (fresh start)."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # Remove orphan collections (optional, keep the master collection)
    for coll in list(bpy.data.collections):
        if coll.name != "Collection":
            try:
                bpy.data.collections.remove(coll)
            except Exception:
                pass

def get_new_objects(before_set):
    """Return list of newly-created objects compared to before_set."""
    return [o for o in bpy.data.objects if o.name not in before_set]

# -----------------------------
# Normals flipping (BMesh)
# -----------------------------
def flip_normals_for_object(obj):
    """Use BMesh to flip normals robustly for a single mesh object."""
    if obj.type != 'MESH' or not obj.data.polygons:
        return

    mesh = obj.data
    # get a copy of one polygon normal before flipping (world space)
    try:
        normal_before = (obj.matrix_world @ mesh.polygons[0].normal).copy()
    except Exception:
        normal_before = None

    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        # Ensure lookup table and faces exist
        bm.faces.ensure_lookup_table()
        if len(bm.faces) > 0:
            # Reverse all faces (flip winding -> flips normals)
            bmesh.ops.reverse_faces(bm, faces=bm.faces)
            # Write back and update
            bm.to_mesh(mesh)
            mesh.update()
    finally:
        bm.free()

    try:
        normal_after = (obj.matrix_world @ mesh.polygons[0].normal).copy()
    except Exception:
        normal_after = None

    safe_print(f"[Normals] {obj.name} before: {normal_before}, after: {normal_after}")

def flip_normals_for_selected():
    """Flip normals for all selected mesh objects (safe)."""
    for obj in list(bpy.context.selected_objects):
        flip_normals_for_object(obj)

# -----------------------------
# Import helpers (Blender 4.x)
# -----------------------------
def import_obj_file(filepath):
    """
    Import a single OBJ using the new Blender 4.x operator.
    Returns list of objects created by the import.
    """
    if not os.path.exists(filepath):
        safe_print(f"[Import] File not found: {filepath}")
        return []

    before = {o.name for o in bpy.data.objects}
    try:
        # Call the new OBJ importer (Blender 4.x)
        # The operator is known as bpy.ops.wm.obj_import in 4.x
        bpy.ops.wm.obj_import(filepath=filepath)
    except Exception as e:
        safe_print(f"[Import] obj_import failed for {filepath}: {e}")
        # Try legacy-style import args (best-effort) — some builds may accept axis args
        try:
            bpy.ops.wm.obj_import(filepath=filepath)
        except Exception as e2:
            safe_print(f"[Import] fallback also failed: {e2}")
            return []

    # determine newly created objects
    new_objs = get_new_objects(before)
    return new_objs

def import_sequence(prefix="mesh", count=30):
    imported_groups = []
    for i in range(count):
        file_path = os.path.join(obj_directory, f"{prefix}_{i:03}_rest.obj")
        safe_print(f"[Import sequence] Importing {file_path} ...")
        new_objs = import_obj_file(file_path)

        if not new_objs:
            safe_print(f"[Import sequence] No objects imported for {file_path}")
            imported_groups.append([])
            continue

        # Select only the newly imported objects and flip normals on them
        bpy.ops.object.select_all(action='DESELECT')
        for o in new_objs:
            try:
                o.select_set(True)
            except Exception:
                pass
        # Make first new obj active
        try:
            bpy.context.view_layer.objects.active = new_objs[0]
        except Exception:
            pass

        flip_normals_for_selected()

        # Sort group by name for stable ordering and store
        group_sorted = sorted(new_objs, key=lambda obj: obj.name)
        imported_groups.append(group_sorted)

        # Deselect to keep a clean state
        bpy.ops.object.select_all(action='DESELECT')

    safe_print(f"[Import sequence] Imported groups: {len(imported_groups)}")
    return imported_groups

# -----------------------------
# Animation setup
# -----------------------------
def setup_animation(groups, catheter_groups, total_frames, frame_rate, object_end_frame):
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = total_frames
    scene.render.fps = frame_rate

    num_objects = len(groups)
    if num_objects == 0:
        return

    # initialize all objects to hidden at frame 0
    for mesh_group, catheter_group in zip(groups, catheter_groups):
        for obj in mesh_group + catheter_group:
            try:
                obj.hide_viewport = True
                obj.hide_render = True
                obj.keyframe_insert(data_path="hide_viewport", frame=0)
                obj.keyframe_insert(data_path="hide_render", frame=0)
            except Exception:
                pass

    for frame in range(total_frames + 1):
        current_index = frame % (2 * num_objects - 2) if (2 * num_objects - 2) > 0 else 0
        if current_index >= num_objects:
            current_index = 2 * num_objects - 2 - current_index

        # Hide all objects at this frame
        for mesh_group, catheter_group in zip(groups, catheter_groups):
            for obj in mesh_group + catheter_group:
                try:
                    obj.hide_viewport = True
                    obj.hide_render = True
                    obj.keyframe_insert(data_path="hide_viewport", frame=frame)
                    obj.keyframe_insert(data_path="hide_render", frame=frame)
                except Exception:
                    pass

        # Show only the current object pair
        for obj in groups[current_index] + catheter_groups[current_index]:
            try:
                obj.hide_viewport = False
                obj.hide_render = False
                obj.keyframe_insert(data_path="hide_viewport", frame=frame)
                obj.keyframe_insert(data_path="hide_render", frame=frame)
            except Exception:
                pass

    # Set interpolation to CONSTANT for the visibility F-curves
    for group in (groups + catheter_groups):
        for obj in group:
            try:
                if obj.animation_data and obj.animation_data.action:
                    for fcurve in obj.animation_data.action.fcurves:
                        if "hide_viewport" in fcurve.data_path or "hide_render" in fcurve.data_path:
                            for kp in fcurve.keyframe_points:
                                kp.interpolation = 'CONSTANT'
            except Exception:
                pass

# -----------------------------
# Render settings & camera
# -----------------------------
def setup_render_settings(output_name="animation.mp4"):
    scene = bpy.context.scene
    # prefer the new Eevee variant if present, fallback otherwise
    try:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
    except Exception:
        try:
            scene.render.engine = 'BLENDER_EEVEE'
        except Exception:
            safe_print("[Render] Could not set Eevee engine; leaving current engine.")

    scene.render.image_settings.file_format = 'FFMPEG'
    try:
        scene.render.ffmpeg.format = 'MPEG4'
        scene.render.ffmpeg.codec = 'H264'
        # use a high-quality CRF alternative if available
        try:
            scene.render.ffmpeg.constant_rate_factor = 'PERC_LOSSLESS'
        except Exception:
            pass
    except Exception:
        pass

    scene.render.filepath = os.path.join(obj_directory, output_name)
    scene.render.resolution_percentage = 100
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080

def setup_camera():
    # compute scene bounding box
    min_coord = Vector((float('inf'),) * 3)
    max_coord = Vector((float('-inf'),) * 3)
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            for corner in obj.bound_box:
                world_corner = obj.matrix_world @ Vector(corner)
                min_coord.x = min(min_coord.x, world_corner.x)
                min_coord.y = min(min_coord.y, world_corner.y)
                min_coord.z = min(min_coord.z, world_corner.z)
                max_coord.x = max(max_coord.x, world_corner.x)
                max_coord.y = max(max_coord.y, world_corner.y)
                max_coord.z = max(max_coord.z, world_corner.z)

    scene_center = (min_coord + max_coord) / 2
    scene_dimensions = max_coord - min_coord
    max_dimension = max(scene_dimensions.x, scene_dimensions.y, scene_dimensions.z) if max(scene_dimensions) != float('-inf') else 1.0

    # empty as camera target
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=scene_center)
    camera_target = bpy.context.object

    # add camera
    bpy.ops.object.camera_add(location=(scene_center.x, scene_center.y - max_dimension*2.5, scene_center.z + max_dimension*1.5))
    camera = bpy.context.object

    constraint = camera.constraints.new('TRACK_TO')
    constraint.target = camera_target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

    bpy.context.scene.camera = camera
    camera.data.lens = 35
    camera.data.clip_end = max_dimension * 10

    return camera, camera_target

def animate_camera(camera, target, start_frame, z_orbit_frames, y_orbit_frames):
    scene_center = target.location.copy()
    initial_offset = camera.location - scene_center

    # Phase 1: Horizontal orbit
    z_start = start_frame
    z_end = start_frame + z_orbit_frames - 1
    for f in range(z_start, z_end + 1):
        t = (f - z_start) / max(1, (z_orbit_frames - 1))
        angle = 2 * math.pi * t
        rot_z = Matrix.Rotation(angle, 4, 'Z')
        new_offset = rot_z @ initial_offset
        camera.location = scene_center + new_offset
        camera.keyframe_insert(data_path="location", frame=f)

    # Phase 2: vertical tilt broken into two segments + hold
    hold_frames = 60
    tilt_motion_frames = max(0, y_orbit_frames - hold_frames)
    if tilt_motion_frames <= 0:
        # just hold
        for f in range(z_end + 1, z_end + 1 + hold_frames):
            camera.keyframe_insert(data_path="location", frame=f)
        return

    # segment frame counts based on angle ratio
    seg1_frames = round(tilt_motion_frames * (89 / (89 + 179)))
    seg2_frames = tilt_motion_frames - seg1_frames

    right_vector = camera.matrix_world.to_quaternion() @ Vector((1, 0, 0))
    current_offset = camera.location - scene_center

    # Segment 1 (tilt up)
    angle1_final = math.radians(105)
    seg1_start = z_end + 1
    for f in range(seg1_start, seg1_start + max(1, seg1_frames)):
        t = (f - seg1_start) / max(1, (seg1_frames - 1))
        angle = t * angle1_final
        quat = Quaternion(right_vector, angle)
        new_offset = quat @ current_offset
        camera.location = scene_center + new_offset
        camera.keyframe_insert(data_path="location", frame=f)

    current_offset = camera.location - scene_center

    # Segment 2 (tilt to other side)
    angle2_final = math.radians(-160)
    seg2_start = seg1_start + seg1_frames
    for f in range(seg2_start, seg2_start + max(1, seg2_frames)):
        t = (f - seg2_start) / max(1, (seg2_frames - 1))
        angle = t * angle2_final
        quat = Quaternion(right_vector, angle)
        new_offset = quat @ current_offset
        camera.location = scene_center + new_offset
        camera.keyframe_insert(data_path="location", frame=f)

    # Hold final
    hold_start = seg2_start + seg2_frames
    final_offset = camera.location - scene_center
    for f in range(hold_start, hold_start + hold_frames):
        camera.location = scene_center + final_offset
        camera.keyframe_insert(data_path="location", frame=f)

# -----------------------------
# Lighting
# -----------------------------
def scatter_lights_around_object(center, radius, num_lights, target):
    lights = []
    for i in range(num_lights):
        theta = math.acos(1 - 2 * (i + 0.5) / num_lights)
        phi = math.pi * (1 + 5**0.5) * (i + 0.5)
        x = center.x + radius * math.sin(theta) * math.cos(phi)
        y = center.y + radius * math.sin(theta) * math.sin(phi)
        z = center.z + radius * math.cos(theta)
        bpy.ops.object.light_add(type='SUN', location=(x, y, z))
        light = bpy.context.object
        # point the light to target
        try:
            c = light.constraints.new('TRACK_TO')
            c.target = target
            c.track_axis = 'TRACK_NEGATIVE_Z'
            c.up_axis = 'UP_Y'
        except Exception:
            pass
        lights.append(light)
    return lights

def get_object_geometric_center(obj):
    coords = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_coord = Vector((min(v[i] for v in coords) for i in range(3)))
    max_coord = Vector((max(v[i] for v in coords) for i in range(3)))
    return (min_coord + max_coord) / 2

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # 1) Clean scene
    clean_scene()

    # 2) Import the mesh and catheter sequences
    groups = import_sequence(prefix="mesh", count=30)
    catheter_groups = import_sequence(prefix="catheter", count=30)

    # Ensure groups lengths match
    max_len = max(len(groups), len(catheter_groups))
    # pad with empty lists if one shorter
    while len(groups) < max_len:
        groups.append([])
    while len(catheter_groups) < max_len:
        catheter_groups.append([])

    # 3) Setup animation (visibility-based)
    setup_animation(groups, catheter_groups, total_frames, frame_rate, object_end_frame)

    # 4) Render settings
    setup_render_settings()

    # 5) Camera
    camera, target = setup_camera()

    # 6) Camera animation (start after object_end_frame)
    animate_camera(camera, target, object_end_frame + 1, z_orbit_frames, y_orbit_frames)

    # 7) Lighting: compute a center to scatter around
    mesh_obj = next((o for o in bpy.data.objects if o.name.startswith("mesh_000_rest")), None)
    if mesh_obj:
        mesh_center = get_object_geometric_center(mesh_obj)
    else:
        mesh_center = target.location

    lights = scatter_lights_around_object(mesh_center, light_radius, num_lights, target)
    for light in lights:
        try:
            light.data.energy = light_energy
            light.data.color = (1.0, 1.0, 1.0)
        except Exception:
            pass

    safe_print("Rendering animation...")
    try:
        bpy.ops.render.render(animation=True)
        safe_print(f"Animation saved to: {bpy.context.scene.render.filepath}")
    except Exception as e:
        safe_print(f"[Render] render failed: {e}")
