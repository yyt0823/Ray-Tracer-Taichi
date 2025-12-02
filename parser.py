import json5 as json
import os
from PIL import Image
import helperclasses as hc
from camera import Camera
import geometry as geom
import scene
import trimesh
import numpy as np
import taichi.math as tm
from pyglm import glm

geom_id = -1  # global geometry ID counter
meshes_total_nb_verts = 0  # global counter of total number of mesh vertices in the scene
meshes_total_nb_faces = 0  # global counter of total number of mesh faces in the scene
scene_meshes_verts = np.empty((0, 3), dtype=np.float32)
scene_meshes_faces = np.empty((0, 3), dtype=np.int32)

def load_scene(infile: str, image_scale_factor: float = 1.0) -> scene.Scene:
    ''' Load a scene from a json file 
    Args:
        infile (str): path to the scene json file
        image_scale_factor (float): scale factor for image resolution
    Returns:
        scene.Scene: the loaded scene object

        The json file can define a hierarchy of nodes, but they will be flattend into a list of geometries in the returned scene
        by accumulating the transformation matrices down the hierarchy.  Instances are provided as a convenience for
    '''
    print("Parsing file:", infile)
    f = open(infile)
    data = json.load(f)

    # Loading resolution
    default_resolution = [1280, 720]    
    width = int(image_scale_factor * data.get("resolution", default_resolution)[0])
    height = int(image_scale_factor * data.get("resolution", default_resolution)[1])
        
    # Loading camera
    cam_pos = glm.vec3(data["camera"]["position"])
    cam_lookat = glm.vec3(data["camera"]["lookAt"])
    cam_up = glm.vec3(data["camera"]["up"])
    cam_fovy = data["camera"]["fovy"]
    camera = Camera(width, height, cam_pos, cam_lookat, cam_up, cam_fovy)

    # Loading ambient light
    ambient = tm.vec3(data.get("ambient", [0.1, 0.1, 0.1])) # set a reasonable default ambient light

    # Loading environment map (image-based)
    env_conf = data.get("environment")
    use_environment = False
    env_pixels = np.zeros((1, 1, 3), dtype=np.float32)
    env_w = env_h = 0
    if env_conf:
        use_environment = env_conf.get("enabled", True)
        image_path = env_conf.get("image")
        if image_path:
            # Treat as path relative to the scene file (no extra magic)
            resolved = os.path.join(os.path.dirname(os.path.abspath(infile)), image_path)
            with Image.open(resolved) as img:
                img = img.convert("RGB")
                env_w, env_h = img.width, img.height
                env_pixels = np.asarray(img, dtype=np.float32) / 255.0


    # Loading Anti-Aliasing options    
    jitter = data.get( "AA_jitter", False ) # default to no jitter
    samples = data.get( "AA_samples", 1 ) # default to no supersampling
    
    # Loading scene lights
    lights_tmp = []
    for light in data.get("lights", []):
        l_type = light["type"]
        l_name = light["name"]
        l_colour = tm.vec3(light["colour"])
        l_power = light.get( "power", 1.0 ) # The power scales the specified light colour
        if l_type == "point":
            l_vector = tm.vec3(light["position"])
            l_attenuation = tm.vec3(0,0,1) if "attenuation" not in light else tm.vec3( light["attenuation"] )
            l_type = 1
        elif l_type == "directional":
            direction = np.array(light["direction"], dtype=np.float32)
            direction_normalized = direction / np.linalg.norm(direction)
            l_vector = tm.vec3(direction_normalized)
            l_attenuation = tm.vec3(0,0,0)
            l_type = 0
            if "attenuation" in light:
                print("Directional light", l_name, "has attenuation, ignoring")
        else:
            print("Unkown light type", l_type, ", skipping initialization")
            continue
        lights_tmp.append(hc.Light(l_type, len(lights_tmp), l_colour * l_power, l_vector, l_attenuation))

    # Use fixed field size for kernel caching (allow up to 10 lights)
    # Store actual count in a field to avoid affecting kernel hash
    nb_lights = len(lights_tmp)
    lights = hc.Light.field(shape=max(10, nb_lights))
    for i in range(len(lights_tmp)):
        lights[i] = lights_tmp[i]

    # Loading materials
    material_by_name = {} # materials dictionary
    mat_id = 0
    for material in data["materials"]:
        mat_name = material["name"]
        mat_diffuse = tm.vec3(material["diffuse"])
        mat_specular = tm.vec3(material["specular"])
        mat_shininess = 0 if "shininess" not in material else material["shininess"]
        mat_reflection = True if "reflection" in material and material["reflection"] == 1 else False
        mat_refraction = True if "refraction" in material and material["refraction"] == 1 else False
        mat_ior = 1.5 if "ior" not in material else material["ior"]  # Default IOR (glass-like)
        material_by_name[mat_name] = hc.Material(mat_id, mat_diffuse, mat_specular, mat_shininess, mat_reflection, mat_refraction, mat_ior)
        mat_id += 1

    # load geometires
    objects = {"sphere": [],
               "plane": [],
               "box": [],
               "mesh": [],
               "cone": [],
               "metaball": []}  # lists of loaded object geometries and hierarchy roots
    node_by_name = {}  # dictionary of geometries by name (for instances)

    M_parent = tm.mat4(np.eye(4))  # identity matrix as the initial parent transformation

    for geometry in data["objects"]:
        if geometry["type"] == "node":
            g = load_node(geometry, material_by_name, node_by_name, M_parent)
            for obj_type, obj in g:
                objects[obj_type].append(obj)
        elif geometry["type"] == "instance":
            g = load_instance(geometry, node_by_name)
            for obj_type, obj in g:
                objects[obj_type].append(obj)
        else:
            g = load_geometry(geometry, material_by_name, M_parent)
            if g is not None:
                obj_type = geometry["type"]
                # Bezier patches are tessellated into triangle meshes
                if obj_type == "bezier":
                    obj_type = "mesh"
                objects[obj_type].append(g)
                # check if "name" field exists
                if "name" in geometry:
                    node_by_name[geometry["name"]] = g

    print("Loaded", geom_id + 1, "geometric objects")

    # Use fixed field sizes for kernel caching
    # Store actual counts in fields to avoid affecting kernel hash
    nb_spheres = len(objects["sphere"])
    spheres = geom.Sphere.field(shape=max(10000, nb_spheres))
    for i in range(len(objects["sphere"])):
        spheres[i] = objects["sphere"][i]

    nb_planes = len(objects["plane"])
    planes = geom.Plane.field(shape=max(50, nb_planes))
    for i in range(len(objects["plane"])):
        planes[i] = objects["plane"][i]

    nb_boxes = len(objects["box"])
    boxes = geom.AABox.field(shape=max(50, nb_boxes))
    for i in range(len(objects["box"])):
        boxes[i] = objects["box"][i]

    nb_meshes = len(objects["mesh"])
    meshes = geom.Mesh.field(shape=max(50, nb_meshes))
    for i in range(len(objects["mesh"])):
        meshes[i] = objects["mesh"][i]

    nb_cones = len(objects["cone"])
    cones = geom.Cone.field(shape=max(50, nb_cones))
    for i in range(len(objects["cone"])):
        cones[i] = objects["cone"][i]

    nb_metaballs = len(objects["metaball"])
    metaballs = geom.Metaball.field(shape=max(20, nb_metaballs))
    for i in range(len(objects["metaball"])):
        metaballs[i] = objects["metaball"][i]

    return scene.Scene( jitter, samples,  # General settings
                camera,  # Camera settings
                ambient,
                use_environment, env_pixels, env_w, env_h,
                lights, nb_lights,  # Light settings
                spheres, nb_spheres,
                planes, nb_planes,
                boxes, nb_boxes,
                meshes, nb_meshes,
                cones, nb_cones,
                metaballs, nb_metaballs,
                scene_meshes_verts, scene_meshes_faces)  # Geometry settings

def mat4_glm_to_ti( M_glm: glm.mat4 ) -> tm.mat4:
    return tm.mat4( glm.transpose(M_glm).to_list() )

def load_geometry_transformation_matrix(geometry, M_parent: tm.mat4) -> (tm.mat4, tm.mat4):
    g_pos = glm.vec3(geometry.get("position", [0, 0, 0]))
    g_r = glm.vec3(geometry.get("rotation", [0, 0, 0]))  # not really useful for a sphere...
    g_s = geometry.get("scale", [1, 1, 1])
    if type(g_s) == float or type(g_s) == int:
        g_s = [g_s, g_s, g_s]
    g_s = glm.vec3(g_s)
    scale = glm.scale( g_s )
    rot_x = glm.rotate( glm.radians(g_r.x), glm.vec3(1,0,0) )
    rot_y = glm.rotate( glm.radians(g_r.y), glm.vec3(0,1,0) )
    rot_z = glm.rotate( glm.radians(g_r.z), glm.vec3(0,0,1) )
    translate = glm.translate( g_pos )
    M_parent_glm = glm.mat4( M_parent.to_numpy() )
    M = M_parent_glm * translate * rot_x * rot_y * rot_z * scale
    M_inv = glm.inverse(M)
    return mat4_glm_to_ti(M), mat4_glm_to_ti(M_inv)

def load_geometry(geometry, material_by_name, M_parent: tm.mat4 ):
    global geom_id, meshes_total_nb_verts, meshes_total_nb_faces, scene_meshes_verts, scene_meshes_faces

    # Elements common to all objects: name, type, and material(s)
    g_type = geometry["type"]
    g_materials = [ material_by_name[material_name] for material_name in geometry.get("materials",[]) ]

    geom_id += 1

    if g_type == "sphere":
        g_radius = geometry.get("radius",1)
        g_motion_dir = tm.vec3(geometry.get("motion_dir", [0, 0, 0]))
        M, M_inv = load_geometry_transformation_matrix(geometry, M_parent)
        return geom.Sphere(geom_id, g_materials[0], g_radius, M, M_inv, g_motion_dir)
    elif g_type == "plane":
        g_normal = tm.vec3(geometry.get("normal",[0,1,0]))
        M, M_inv = load_geometry_transformation_matrix(geometry, M_parent)
        two_materials = True if len(g_materials) > 1 else False
        mat1 = g_materials[0]
        mat2 = g_materials[1] if two_materials else g_materials[0]
        return geom.Plane(geom_id, two_materials, mat1, mat2, g_normal, M, M_inv)
    elif g_type == "box":
        minpos = tm.vec3(geometry.get("min",[-1,-1,-1]))
        maxpos = tm.vec3(geometry.get("max",[1,1,1]))
        M, M_inv = load_geometry_transformation_matrix(geometry, M_parent)
        return geom.AABox(geom_id, g_materials[0], minpos, maxpos, M, M_inv)
    elif g_type == "mesh":
        g_path = geometry["filepath"]
        M, M_inv = load_geometry_transformation_matrix(geometry, M_parent)
        mesh = trimesh.load_mesh(g_path)
        verts = mesh.vertices
        faces = mesh.faces

        # Compute local-space axis-aligned bounding box for acceleration
        if len(verts) > 0:
            vmin = verts.min(axis=0)
            vmax = verts.max(axis=0)
        else:
            # Fallback to a tiny box if mesh somehow has no vertices
            vmin = np.array((0.0, 0.0, 0.0), dtype=np.float32)
            vmax = np.array((0.0, 0.0, 0.0), dtype=np.float32)

        scene_meshes_verts = np.resize(scene_meshes_verts, (meshes_total_nb_verts + len(verts), 3))
        scene_meshes_faces = np.resize(scene_meshes_faces, (meshes_total_nb_faces + len(faces), 3))
        # NOTE: These should really be vectorized!
        for i in range(len(verts)):
            scene_meshes_verts[meshes_total_nb_verts + i] = np.array((verts[i, 0], verts[i, 1], verts[i, 2]))
        for i in range(len(faces)):
            scene_meshes_faces[meshes_total_nb_faces + i] = np.array((faces[i, 0] + meshes_total_nb_verts,
                                                                      faces[i, 1] + meshes_total_nb_verts,
                                                                      faces[i, 2] + meshes_total_nb_verts))
        # NOTE: an opportunity to transform the verts of the mesh rather than transforming the ray later
        mesh = geom.Mesh(
            geom_id,
            g_materials[0],
            meshes_total_nb_faces,
            len(faces),
            M,
            M_inv,
            tm.vec3(vmin.tolist()),
            tm.vec3(vmax.tolist()),
        )
        meshes_total_nb_verts += len(verts)
        meshes_total_nb_faces += len(faces)
        return mesh
    elif g_type == "cone":
        g_radius = geometry.get("radius", 1.0)
        g_height = geometry.get("height", 2.0)
        M, M_inv = load_geometry_transformation_matrix(geometry, M_parent)
        return geom.Cone(geom_id, g_materials[0], g_radius, g_height, M, M_inv)
    elif g_type == "metaball":
        g_threshold = geometry.get("threshold", 1.0)
        g_blobs = geometry.get("blobs", [])
        num_blobs = min(len(g_blobs), 3)  # Limit to 3 blobs max
        
        if num_blobs == 0:
            print("Metaball with no blobs, skipping")
            geom_id -= 1
            return None
        
        M, M_inv = load_geometry_transformation_matrix(geometry, M_parent)
        
        # Initialize blob positions and radii arrays
        blob_positions = [tm.vec3(0.0, 0.0, 0.0)] * 3
        blob_radii = [0.0] * 3
        
        # Load blob data using a loop
        for i in range(num_blobs):
            blob = g_blobs[i]
            blob_positions[i] = tm.vec3(blob.get("position", [0, 0, 0]))
            blob_radii[i] = blob.get("radius", 0.5)
        
        # Assign to individual fields
        blob0_pos, blob0_radius = blob_positions[0], blob_radii[0]
        blob1_pos, blob1_radius = blob_positions[1], blob_radii[1]
        blob2_pos, blob2_radius = blob_positions[2], blob_radii[2]
        
        return geom.Metaball(geom_id, num_blobs, g_threshold, g_materials[0], M, M_inv,
                          blob0_pos, blob0_radius,
                          blob1_pos, blob1_radius,
                          blob2_pos, blob2_radius)
    elif g_type == "bezier":
        # Bicubic Bezier surface patch tessellated into triangles (converted to a Mesh)
        control_points = geometry.get("controlPoints", None)
        if control_points is None or len(control_points) != 4 or any(len(row) != 4 for row in control_points):
            print("Bezier patch must have 4x4 'controlPoints'; skipping initialization")
            geom_id -= 1
            return None

        # Tessellation resolution (number of segments along each param axis)
        steps_u = geometry.get("tessellation_u", geometry.get("tessellation", 10))
        steps_v = geometry.get("tessellation_v", geometry.get("tessellation", 10))
        steps_u = max(1, int(steps_u))
        steps_v = max(1, int(steps_v))

        # Evaluate Bezier patch in Python using cubic Bernstein basis
        def bernstein3(t: float):
            it = 1.0 - t
            b0 = it * it * it
            b1 = 3.0 * t * it * it
            b2 = 3.0 * t * t * it
            b3 = t * t * t
            return [b0, b1, b2, b3]

        verts_local = []
        for iu in range(steps_u + 1):
            u = float(iu) / float(steps_u)
            bu = bernstein3(u)
            for iv in range(steps_v + 1):
                v = float(iv) / float(steps_v)
                bv = bernstein3(v)
                px = 0.0
                py = 0.0
                pz = 0.0
                for i in range(4):
                    for j in range(4):
                        w = bu[i] * bv[j]
                        cp = control_points[i][j]
                        px += w * cp[0]
                        py += w * cp[1]
                        pz += w * cp[2]
                verts_local.append(np.array((px, py, pz), dtype=np.float32))

        # Build triangle faces over the (steps_u+1) x (steps_v+1) grid
        faces_local = []
        verts_per_row = steps_v + 1
        for iu in range(steps_u):
            for iv in range(steps_v):
                i0 = iu * verts_per_row + iv
                i1 = (iu + 1) * verts_per_row + iv
                i2 = (iu + 1) * verts_per_row + (iv + 1)
                i3 = iu * verts_per_row + (iv + 1)
                # Two triangles per quad
                faces_local.append((i0, i1, i2))
                faces_local.append((i0, i2, i3))

        # Append to global mesh vertex/face arrays
        n_verts = len(verts_local)
        n_faces = len(faces_local)

        scene_meshes_verts = np.resize(scene_meshes_verts, (meshes_total_nb_verts + n_verts, 3))
        scene_meshes_faces = np.resize(scene_meshes_faces, (meshes_total_nb_faces + n_faces, 3))

        for i in range(n_verts):
            scene_meshes_verts[meshes_total_nb_verts + i] = verts_local[i]

        for i in range(n_faces):
            f = faces_local[i]
            scene_meshes_faces[meshes_total_nb_faces + i] = np.array(
                (f[0] + meshes_total_nb_verts,
                 f[1] + meshes_total_nb_verts,
                 f[2] + meshes_total_nb_verts),
                dtype=np.int32
            )

        # Compute local-space bounding box for the tessellated vertices
        if n_verts > 0:
            verts_np = np.array(verts_local, dtype=np.float32)
            vmin = verts_np.min(axis=0)
            vmax = verts_np.max(axis=0)
        else:
            vmin = np.array((0.0, 0.0, 0.0), dtype=np.float32)
            vmax = np.array((0.0, 0.0, 0.0), dtype=np.float32)

        M, M_inv = load_geometry_transformation_matrix(geometry, M_parent)
        mesh = geom.Mesh(
            geom_id,
            g_materials[0],
            meshes_total_nb_faces,
            n_faces,
            M,
            M_inv,
            tm.vec3(vmin.tolist()),
            tm.vec3(vmax.tolist()),
        )
        meshes_total_nb_verts += n_verts
        meshes_total_nb_faces += n_faces
        return mesh
    else:
        print("Unkown object type", g_type, ", skipping initialization")
        geom_id -= 1  # we cancel the increment of geom_id since we didn't create any geometry
        return None
    
def load_node(geometry, material_by_name, node_by_name, M_parent: tm.mat4 ):
    M, M_inv = load_geometry_transformation_matrix(geometry, M_parent)    
    # For this node, keep a list of all the childern objects
    objects = []
    node_by_name[geometry["name"]] = []
    
    for child in geometry["children"]:
        if child["type"] != "node" and child["type"] != "instance":
            g = load_geometry(child, material_by_name, M)
            objects.append((child["type"], g))  # each geometry is stored as a tuple (type, object)
            node_by_name[geometry["name"]] += objects
        elif child["type"] == "node":
            objects += load_node(child, material_by_name, node_by_name, M)
            node_by_name[geometry["name"]] += objects
        elif child["type"] == "instance":
            print("Instances are not allowed inside nodes, must be defined at root level")
        else:
            print("Unkown object type", child["type"], ", skipping initialization")

    return objects

def load_instance(geometry, node_by_name):
    # instances are loaded off the root, so will have an identiy matrix as parent
    M, M_inv = load_geometry_transformation_matrix(geometry, tm.mat4( np.eye(4) ) ) 
    objects = []
    node = node_by_name[geometry["ref"]]
    for obj_type, obj in node:
        if obj is not None:
            if obj_type == "sphere":
                new_obj = geom.Sphere(obj.id, obj.material, obj.radius, M @ obj.M, obj.M_inv @ M_inv, obj.motion_dir)                                      
            elif obj_type == "plane":
                new_obj = geom.Plane(obj.id, obj.two_materials, obj.material1, obj.material2, obj.position, obj.normal, M @ obj.M, obj.M_inv @ M_inv )            
            elif obj_type == "box":
                new_obj = geom.AABox(obj.id, obj.material, obj.minpos, obj.maxpos, M @ obj.M, obj.M_inv @ M_inv )  
            elif obj_type == "mesh":
                # For meshes, preserve face range and local AABB, update transforms
                new_obj = geom.Mesh(
                    obj.id,
                    obj.material,
                    obj.faces_ids_start,
                    obj.faces_ids_count,
                    M @ obj.M,
                    obj.M_inv @ M_inv,
                    obj.bbox_min,
                    obj.bbox_max,
                )
            elif obj_type == "cone":
                new_obj = geom.Cone(obj.id, obj.material, obj.radius, obj.height, M @ obj.M, obj.M_inv @ M_inv)
            objects.append((obj_type, new_obj))

    return objects