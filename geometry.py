from helperclasses import Ray, Intersection, Material, changeRayFrame, getRayPoint, changeIntersectFrame

import taichi as ti
import taichi.math as tm

EPSILON = 10 ** (-5)

@ti.dataclass
class Sphere:
    id: int
    material: Material
    radius: float
    M: tm.mat4
    M_inv: tm.mat4

@ti.func
def intersectSphere(sphere: Sphere, ray: Ray, t_min: float, t_max: float) -> Intersection:
    ''' Ray-sphere intersection  
    Args:
        sphere (Sphere): sphere to intersect with
        ray (Ray): ray in world space
        t_min (float): minimum t value for valid intersection
        t_max (float): maximum t value for valid intersection
    Returns:
        Intersection: intersection data (is_hit, t, normal, point, material)
        
        Note, nominally the intersection will be computed with the sphere
        at the origin with the specified radius.  For a sphere away from 
        the origin, the M and M_inv members of the Sphere class must be used to 
        transform the ray into the sphere's local frame for intersection, and
        likewise used to transform the resulting intersection data back to world space
        (TODO: See Objective ?).
    '''

    hit = Intersection() # default is no intersection (is_hit = False)

    local_ray = changeRayFrame(ray, sphere.M_inv)
    O_local = local_ray.origin
    D_local_norm = tm.normalize(local_ray.direction)
    center = tm.vec3(0.0)
    radius = sphere.radius
    a = tm.dot(D_local_norm, D_local_norm)
    b = 2 * tm.dot(D_local_norm, O_local.xyz)
    c = tm.dot(O_local.xyz, O_local.xyz) - radius * radius
    discriminant = b * b - 4.0 * a * c
    if discriminant >= 0.0:
        t1 = (-b - tm.sqrt(discriminant)) / (2.0 * a)
        t2 = (-b + tm.sqrt(discriminant)) / (2.0 * a)
        t = t_max
        if t1 > t_min and t1 < t_max:
            t = t1
            hit.is_hit = True
        if t2 > t_min and t2 < t_max and t2 < t:
            t = t2
            hit.is_hit = True
        
        if hit.is_hit:
            hit.t = t

            # Compute Local Point & Normal
            P_local = O_local.xyz + t * D_local_norm
            N_local = tm.normalize(P_local - center) # Normal at the surface
            
            # 1. Transform Point to World Space (using M)
            P_world_h = sphere.M @ tm.vec4(P_local, 1.0)
            hit.position = P_world_h.xyz

            # 2. Transform Normal to World Space (using Inverse Transpose of M)
            M_inv_T = sphere.M_inv.transpose()
            N_world_h = M_inv_T @ tm.vec4(N_local, 0.0)
            hit.normal = tm.normalize(N_world_h.xyz)
            
            hit.mat = sphere.material

    return hit


@ti.dataclass
class Plane:
    id: int
    two_materials: bool     # true if plane uses two materials for a checkerboard pattern
    material1: Material
    material2: Material
    normal: tm.vec3         # only plane normal (goes through origin in local space)
    M: tm.mat4              # transformation matrix to world space  
    M_inv: tm.mat4          # inverse transformation matrix to local space  

@ti.func
def intersectPlane(plane: Plane, ray: Ray, t_min: float, t_max: float) -> Intersection:
    ''' Ray-plane intersection
    Args:
        plane (Plane): plane to intersect with
        ray (Ray): ray in world space
        t_min (float): minimum t value for valid intersection
        t_max (float): maximum t value for valid intersection
    Returns:
        Intersection: intersection data (is_hit, t, normal, point, material)
        
        Note, nominally the intersection will be computed with the plane
        at the origin with the specified normal.  For a plane away from 
        the origin, the M and M_inv members of the Plane class must be used to 
        transform the ray into the plane's local frame for intersection, and
        likewise used to transform the resulting intersection data back to world space.

        Note, the plane can have two materials (when two_materials is true) which
        are applied in a checkerboard pattern in the plane's local XZ plane.
    '''

    hit = Intersection()  # default is no intersection (is_hit = False)

    # Use Objective 4 helper to transform ray into plane's local frame
    local_ray = changeRayFrame(ray, plane.M_inv)
    O_local = local_ray.origin
    D_local_norm = tm.normalize(local_ray.direction)

    # Plane normal in local space
    N_local = plane.normal

    # Ray-plane intersection in local space: N · (O + t D) = 0
    N_dot_D = tm.dot(N_local, D_local_norm)

    # Only do intersection work if not parallel
    if ti.abs(N_dot_D) >= EPSILON:
        # t = - (N · O) / (N · D)
        t = -tm.dot(N_local, O_local) / N_dot_D

        # Check valid t range
        if t >= t_min and t <= t_max:
            hit.is_hit = True
            hit.t = t

            # Local intersection point
            P_local = O_local + t * D_local_norm

            # Choose material (checkerboard in local XZ if two_materials)
            if plane.two_materials:
                x_int = ti.floor(P_local.x)
                z_int = ti.floor(P_local.z)
                x_parity = ti.cast(x_int, ti.i32) % 2
                z_parity = ti.cast(z_int, ti.i32) % 2
                if (x_parity + z_parity) % 2 == 0:
                    hit.mat = plane.material1
                else:
                    hit.mat = plane.material2
            else:
                hit.mat = plane.material1

            # Transform intersection data back to world space
            P_world_h = plane.M @ tm.vec4(P_local, 1.0)
            hit.position = P_world_h.xyz

            M_inv_T = plane.M_inv.transpose()
            N_world_h = M_inv_T @ tm.vec4(N_local, 0.0)
            hit.normal = tm.normalize(N_world_h.xyz)

    return hit


@ti.dataclass
class AABox:
    id: int
    material: Material
    minpos: tm.vec3     # lower left corner
    maxpos: tm.vec3     # upper right corner
    M: tm.mat4
    M_inv: tm.mat4

@ti.func
def intersectAABox(aabox: AABox, ray: Ray, t_min: float, t_max: float) -> Intersection:
    hit = Intersection()  # default is no intersection (is_hit = False)

    # Use Objective 4 helper to transform ray into the box's local frame
    # (where the AABox is axis-aligned between minpos and maxpos)
    local_ray = changeRayFrame(ray, aabox.M_inv)
    O_local = local_ray.origin
    D_local = local_ray.direction

    # Slab method for axis-aligned box intersection in local space
    t_enter = t_min
    t_exit = t_max
    hit_axis = -1  # 0=x,1=y,2=z to determine normal later

    for axis in ti.static(range(3)):
        origin_comp = O_local[axis]
        dir_comp = D_local[axis]
        min_comp = aabox.minpos[axis]
        max_comp = aabox.maxpos[axis]

        if ti.abs(dir_comp) < EPSILON:
            # Ray is parallel to slabs; must be within the slab range to possibly hit
            if origin_comp < min_comp or origin_comp > max_comp:
                # No intersection with this slab -> no hit at all
                t_enter = t_max + 1.0  # force miss
        else:
            inv_dir = 1.0 / dir_comp
            t0 = (min_comp - origin_comp) * inv_dir
            t1 = (max_comp - origin_comp) * inv_dir
            # Ensure t0 <= t1
            if t0 > t1:
                tmp = t0
                t0 = t1
                t1 = tmp

            # Update global enter and exit
            if t0 > t_enter:
                t_enter = t0
                hit_axis = axis
            if t1 < t_exit:
                t_exit = t1

    # Valid intersection if interval is non-empty and within [t_min, t_max]
    if t_enter <= t_exit and t_enter >= t_min and t_enter <= t_max:
        hit.is_hit = True
        hit.t = t_enter

        # Local hit point
        P_local = O_local + t_enter * D_local

        # Local normal based on which face we entered
        N_local = tm.vec3(0.0, 0.0, 0.0)
        if hit_axis == 0:
            # x-face
            if P_local.x <= (aabox.minpos.x + aabox.maxpos.x) * 0.5:
                N_local.x = -1.0
            else:
                N_local.x = 1.0
        elif hit_axis == 1:
            # y-face
            if P_local.y <= (aabox.minpos.y + aabox.maxpos.y) * 0.5:
                N_local.y = -1.0
            else:
                N_local.y = 1.0
        elif hit_axis == 2:
            # z-face
            if P_local.z <= (aabox.minpos.z + aabox.maxpos.z) * 0.5:
                N_local.z = -1.0
            else:
                N_local.z = 1.0

        # Assign material in local space
        hit.mat = aabox.material

        # Transform intersection point and normal back to world space
        P_world_h = aabox.M @ tm.vec4(P_local, 1.0)
        hit.position = P_world_h.xyz

        M_inv_T = aabox.M_inv.transpose()
        N_world_h = M_inv_T @ tm.vec4(N_local, 0.0)
        hit.normal = tm.normalize(N_world_h.xyz)

    return hit


@ti.dataclass
class Mesh:
    id: int
    material: Material
    faces_ids_start: ti.i32  # index of the first face in the global face array
    faces_ids_count: ti.i32  # number of faces in this mesh
    M: tm.mat4
    M_inv: tm.mat4

@ti.func
def intersectMesh(mesh: Mesh,                  # data for this mesh (start face and number of faces, etc.)
                  meshes_verts: ti.template(), # all vertices (for all meshes)
                  meshes_faces: ti.template(), # all faces (for all meshes)
                  ray: Ray,
                  t_min: float,
                  t_max: float
) -> Intersection:
    out_intersect = Intersection()  # default is no intersection (is_hit = False)

    # Transform ray into mesh's local frame (Objective 4)
    local_ray = changeRayFrame(ray, mesh.M_inv)
    O_local = local_ray.origin
    D_local = local_ray.direction

    best_t = t_max
    best_P_local = tm.vec3(0.0)
    best_N_local = tm.vec3(0.0)
    hit_any = False

    # Loop over all faces belonging to this mesh
    start = mesh.faces_ids_start
    count = mesh.faces_ids_count

    for fi in range(count):
        face_index = start + fi
        # Indices of the triangle vertices in the global vertex array
        f = meshes_faces[face_index]
        i0 = f[0]
        i1 = f[1]
        i2 = f[2]

        v0 = meshes_verts[i0]
        v1 = meshes_verts[i1]
        v2 = meshes_verts[i2]

        # Möller–Trumbore ray–triangle intersection in local space
        e1 = v1 - v0
        e2 = v2 - v0

        pvec = tm.cross(D_local, e2)
        det = tm.dot(e1, pvec)

        if ti.abs(det) > EPSILON:
            inv_det = 1.0 / det

            tvec = O_local - v0
            u = tm.dot(tvec, pvec) * inv_det
            if u < 0.0 or u > 1.0:
                continue

            qvec = tm.cross(tvec, e1)
            v = tm.dot(D_local, qvec) * inv_det
            if v < 0.0 or u + v > 1.0:
                continue

            t = tm.dot(e2, qvec) * inv_det

            if t >= t_min and t <= best_t:
                best_t = t
                hit_any = True

                best_P_local = O_local + t * D_local
                best_N_local = tm.normalize(tm.cross(e1, e2))

    if hit_any:
        out_intersect.is_hit = True
        out_intersect.t = best_t
        out_intersect.mat = mesh.material

        # Transform intersection point and normal back to world space
        P_world_h = mesh.M @ tm.vec4(best_P_local, 1.0)
        out_intersect.position = P_world_h.xyz

        M_inv_T = mesh.M_inv.transpose()
        N_world_h = M_inv_T @ tm.vec4(best_N_local, 0.0)
        out_intersect.normal = tm.normalize(N_world_h.xyz)

    return out_intersect
