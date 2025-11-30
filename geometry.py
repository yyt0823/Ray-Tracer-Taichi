from helperclasses import Ray, Intersection, Material, changeRayFrame, getRayPoint, changeIntersectFrame

import taichi as ti
import taichi.math as tm

EPSILON = 10 ** (-5)

@ti.func
def apply_motion_transform(M: tm.mat4, M_inv: tm.mat4, motion_dir: tm.vec3, time: float) -> (tm.mat4, tm.mat4):
    """Apply motion blur offset to transformation matrices.
    Args:
        M: original transformation matrix
        M_inv: original inverse transformation matrix
        motion_dir: motion direction vector per unit time
        time: time value (0.0 to 1.0)
    Returns:
        (M_motion, M_inv_motion): transformed matrices with motion offset
    """
    M_motion = M
    M_inv_motion = M_inv
    
    # Only apply motion if motion_dir is significant
    if tm.length(motion_dir) >= EPSILON:
        # Create translation matrix for motion offset
        offset = motion_dir * time
        translate = tm.mat4([
            [1.0, 0.0, 0.0, offset.x],
            [0.0, 1.0, 0.0, offset.y],
            [0.0, 0.0, 1.0, offset.z],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # Apply translation: M_motion = translate * M
        M_motion = translate @ M
        
        # Inverse: M_inv_motion = M_inv * translate_inv
        translate_inv = tm.mat4([
            [1.0, 0.0, 0.0, -offset.x],
            [0.0, 1.0, 0.0, -offset.y],
            [0.0, 0.0, 1.0, -offset.z],
            [0.0, 0.0, 0.0, 1.0]
        ])
        M_inv_motion = M_inv @ translate_inv
    
    return M_motion, M_inv_motion

@ti.func
def rayPlaneIntersection(ray_origin: tm.vec3, ray_dir: tm.vec3, plane_point: tm.vec3, plane_normal: tm.vec3) -> float:
    ''' Compute ray-plane intersection t value
    Args:
        ray_origin (tm.vec3): ray origin point
        ray_dir (tm.vec3): ray direction vector
        plane_point (tm.vec3): a point on the plane
        plane_normal (tm.vec3): normalized plane normal vector
    Returns:
        float: t value where ray intersects plane, or float('inf') if no intersection
    '''
    N_dot_D = tm.dot(plane_normal, ray_dir)
    
    # Check if ray is parallel to plane
    if ti.abs(N_dot_D) < EPSILON:
        return float('inf')
    
    # Ray-plane intersection: t = -N · (O - P) / (N · D)
    # where P is a point on the plane, O is ray origin, D is ray direction
    tvec = ray_origin - plane_point
    t = -tm.dot(plane_normal, tvec) / N_dot_D
    
    return t

@ti.dataclass
class Sphere:
    id: int
    material: Material
    radius: float
    M: tm.mat4
    M_inv: tm.mat4
    motion_dir: tm.vec3

@ti.func
def intersectSphere(sphere: Sphere, ray: Ray, t_min: float, t_max: float, motion_dir: tm.vec3) -> Intersection:
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
    
    # Check if motion_dir is valid (non-zero)
    has_motion = tm.length(motion_dir) >= EPSILON
    
    if has_motion:
        # Test 3 time frames: t=0, t=0.5, t=1.0
        num_samples = 3
        hit_count = 0  # Count how many time frames result in hits (1, 2, or 3)
        best_hit = Intersection()
        best_t = t_max
        
        for frame in range(num_samples):
            time = float(frame) / float(num_samples - 1)  # 0.0, 0.5, 1.0
            
            # Apply motion blur transformation
            M_motion, M_inv_motion = apply_motion_transform(sphere.M, sphere.M_inv, motion_dir, time)
            local_ray = changeRayFrame(ray, M_inv_motion)
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
                frame_hit = False
                if t1 > t_min and t1 < t_max:
                    t = t1
                    frame_hit = True
                if t2 > t_min and t2 < t_max and t2 < t:
                    t = t2
                    frame_hit = True
                
                if frame_hit:
                    hit_count += 1  # Increment hit count
                    # Keep the closest hit for the return value
                    if t < best_t:
                        best_t = t
                        # Compute Local Point & Normal
                        P_local = O_local.xyz + t * D_local_norm
                        N_local = tm.normalize(P_local - center)
                        
                        # Transform to world space
                        P_world_h = M_motion @ tm.vec4(P_local, 1.0)
                        best_hit.position = P_world_h.xyz
                        
                        M_inv_T = M_inv_motion.transpose()
                        N_world_h = M_inv_T @ tm.vec4(N_local, 0.0)
                        best_hit.normal = tm.normalize(N_world_h.xyz)
                        best_hit.t = t
                        best_hit.mat = sphere.material
                        best_hit.is_hit = True
        
        if best_hit.is_hit:
            hit = best_hit
            # Store hit count (1, 2, or 3) in hit_count field
            hit.hit_count = hit_count
    else:
        # No motion blur: single intersection test
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
                hit.hit_count = 0  # No motion blur, so 0 (single hit, not motion blur)

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
    D_local_norm = local_ray.direction

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
                x_parity = x_int % 2
                z_parity = z_int % 2
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

    local_ray = changeRayFrame(ray, aabox.M_inv)
    O_local = local_ray.origin
    D_local = local_ray.direction

    # Slab method for axis-aligned box intersection in local space
    t_enter = t_min
    t_exit = t_max
    hit_axis = -1  # 0=x,1=y,2=z to determine normal later

    # this is essentially calculation in 1 dimension for each axis
    for axis in ti.static(range(3)):
        # ray origin on that axis
        origin_comp = O_local[axis]
        # distance move on that axis each step
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
                  t_max: float) -> Intersection:

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

        # Step 1: Compute triangle normal and edges
        e1 = v1 - v0
        e2 = v2 - v0
        N_tri = tm.cross(e1, e2)
        N_normalized = tm.normalize(N_tri)
        
        # Step 2: Ray-plane intersection (same as plane intersection)
        N_dot_D = tm.dot(N_normalized, D_local)
        
        # Skip if ray is parallel to plane
        if ti.abs(N_dot_D) < EPSILON:
            continue
        
        # N · (R(t) - v0) = 0
        # N · (O_local + t*D_local - v0) = 0
        # Solve: t = -N · (O_local - v0) / (N · D_local)
        tvec = O_local - v0
        t = -tm.dot(N_normalized, tvec) / N_dot_D
        
        # Check if intersection is within valid t range
        if t < t_min or t > best_t:
            continue
        
        # Step 3: Compute intersection point
        P = O_local + t * D_local
        
        # Step 4: Use signed areas to check if P is inside triangle
        # Area of full triangle (v0, v1, v2)
        area_full_vec = tm.cross(e1, e2)
        area_full = tm.dot(area_full_vec, N_normalized)  # Project onto normal for signed area
        
        if ti.abs(area_full) < EPSILON:
            continue
        
        # Compute signed areas of three sub-triangles formed by point P
        # Area of triangle (P, v1, v2) - corresponds to barycentric coordinate for v0
        edge1 = v1 - P
        edge2 = v2 - P
        area1_vec = tm.cross(edge1, edge2)
        area1 = tm.dot(area1_vec, N_normalized)
        
        # Area of triangle (P, v2, v0) - corresponds to barycentric coordinate for v1
        edge3 = v2 - P
        edge4 = v0 - P
        area2_vec = tm.cross(edge3, edge4)
        area2 = tm.dot(area2_vec, N_normalized)
        
        # Area of triangle (P, v0, v1) - corresponds to barycentric coordinate for v2
        edge5 = v0 - P
        edge6 = v1 - P
        area3_vec = tm.cross(edge5, edge6)
        area3 = tm.dot(area3_vec, N_normalized)
        
        # Compute barycentric coordinates (normalized by full area)
        inv_area_full = 1.0 / area_full
        u = area1 * inv_area_full  # Weight for v0
        v = area2 * inv_area_full  # Weight for v1
        w = area3 * inv_area_full  # Weight for v2
        
        # Check if point is inside triangle: all barycentrics must be >= 0
        if u < 0.0 or v < 0.0 or w < 0.0:
            continue
        
        # Valid intersection found!
        best_t = t
        hit_any = True
        best_P_local = P
        best_N_local = N_normalized

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


"""
objective 9 Geometry: Cone

implicit form : f(x, y, z) = x² + z² - (r²/h²) * y² = 0

we will make k_sq = r²/h² to simplify the equation later



"""
@ti.dataclass
class Cone:
    id: int
    material: Material
    radius: float      # radius at the base
    height: float      # height of the cone
    M: tm.mat4
    M_inv: tm.mat4

@ti.func
def intersectCone(cone: Cone, ray: Ray, t_min: float, t_max: float) -> Intersection:
    ''' Ray-cone intersection
    Args:
        cone (Cone): cone to intersect with
        ray (Ray): ray in world space
        t_min (float): minimum t value for valid intersection
        t_max (float): maximum t value for valid intersection
    Returns:
        Intersection: intersection data (is_hit, t, normal, point, material)
        The cone equation in local space: x² + z² - (r²/h²) * y² = 0
        where r is the radius at height h, apex is at origin, extending along +Y axis.
    '''
    
    hit = Intersection()  # default is no intersection (is_hit = False)
    
    # Transform ray into cone's local frame
    local_ray = changeRayFrame(ray, cone.M_inv)
    O_local = local_ray.origin
    D_local = local_ray.direction
    
    # Cone parameters
    r = cone.radius
    h = cone.height
    k_sq = (r * r) / (h * h)  # k² = (r/h)²
    
    # Ray equation: R(t) = O + t*D
    # Cone equation: x² + z² - k² * y² = 0
    # Substitute: (O_x + t*D_x)² + (O_z + t*D_z)² - k² * (O_y + t*D_y)² = 0
    # Expand and group terms:
    # t²*(D_x² + D_z² - k²*D_y²) + 2*t*(O_x*D_x + O_z*D_z - k²*O_y*D_y) + (O_x² + O_z² - k²*O_y²) = 0
    
    # Quadratic coefficients: a*t² + b*t + c = 0
    a = D_local.x * D_local.x + D_local.z * D_local.z - k_sq * D_local.y * D_local.y
    b = 2.0 * (O_local.x * D_local.x + O_local.z * D_local.z - k_sq * O_local.y * D_local.y)
    c = O_local.x * O_local.x + O_local.z * O_local.z - k_sq * O_local.y * O_local.y
    
    discriminant = b * b - 4.0 * a * c
    
    if discriminant >= 0.0 and ti.abs(a) >= EPSILON:
        sqrt_disc = tm.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        
        # Check both solutions and find the closest valid one
        t = t_max
        for candidate_t in ti.static([t1, t2]):
            if candidate_t > t_min and candidate_t < t_max and candidate_t < t:
                # Check if intersection point is within cone bounds (0 <= y <= height)
                P_local = O_local + candidate_t * D_local
                if P_local.y >= 0.0 and P_local.y <= h:
                    t = candidate_t
                    hit.is_hit = True
        
        if hit.is_hit:
            hit.t = t
            P_local = O_local + t * D_local
            
            # Compute normal at intersection point
            # For cone: x² + z² - k² * y² = 0
            # Gradient: (2x, -2k²y, 2z)
            # Normal: normalize(2x, -2k²y, 2z) = normalize(x, -k²y, z)
            N_local = tm.vec3(P_local.x, -k_sq * P_local.y, P_local.z)
            N_local = tm.normalize(N_local)
            
            # Transform intersection point and normal back to world space
            P_world_h = cone.M @ tm.vec4(P_local, 1.0)
            hit.position = P_world_h.xyz
            
            M_inv_T = cone.M_inv.transpose()
            N_world_h = M_inv_T @ tm.vec4(N_local, 0.0)
            hit.normal = tm.normalize(N_world_h.xyz)
            
            hit.mat = cone.material
    
    return hit













"""
Metaball Ray Marching Implementation

Metaballs use a Signed Distance Function (SDF) that blends multiple blobs together.
The SDF formula: SDF(p) = threshold - sum(radius^2 / distance^2) for each blob
The surface is where SDF(p) = 0 (or crosses zero).
"""
@ti.dataclass
class Metaball:
    id: int
    count: int              
    threshold: float        
    material: Material
    M: tm.mat4
    M_inv: tm.mat4
    blob0_pos: tm.vec3
    blob0_radius: float
    blob1_pos: tm.vec3
    blob1_radius: float
    blob2_pos: tm.vec3
    blob2_radius: float




@ti.func
def metaball_sdf(metaball: Metaball, point: tm.vec3) -> float:
    """Compute the Signed Distance Function for a metaball at a given point.
    Args:
        metaball: The metaball object
        point: Point in local space to evaluate SDF
    Returns:
        float: SDF value (positive = inside, negative = outside, zero = on surface)
    """
    sdf_value = metaball.threshold
    
    # Sum contributions from all active blobs
    if metaball.count >= 1:
        dist0 = tm.length(point - metaball.blob0_pos)
        if dist0 > EPSILON:
            sdf_value -= metaball.blob0_radius * metaball.blob0_radius / (dist0 * dist0)
    
    if metaball.count >= 2:
        dist1 = tm.length(point - metaball.blob1_pos)
        if dist1 > EPSILON:
            sdf_value -= metaball.blob1_radius * metaball.blob1_radius / (dist1 * dist1)
    
    if metaball.count >= 3:
        dist2 = tm.length(point - metaball.blob2_pos)
        if dist2 > EPSILON:
            sdf_value -= metaball.blob2_radius * metaball.blob2_radius / (dist2 * dist2)
    
    return sdf_value

@ti.func
def compute_sdf_normal(metaball: Metaball, point: tm.vec3) -> tm.vec3:
    """Compute the normal vector at a point on the metaball surface using finite differences.
    Args:
        metaball: The metaball object
        point: Point in local space (should be near the surface)
    Returns:
        tm.vec3: Normalized normal vector
    """
    delta = 0.001  # Small offset for finite differences
    sdf_center = metaball_sdf(metaball, point)
    
    # Compute gradient using finite differences
    sdf_x = metaball_sdf(metaball, point + tm.vec3(delta, 0.0, 0.0))
    sdf_y = metaball_sdf(metaball, point + tm.vec3(0.0, delta, 0.0))
    sdf_z = metaball_sdf(metaball, point + tm.vec3(0.0, 0.0, delta))
    
    # Gradient = (df/dx, df/dy, df/dz)
    gradient = tm.vec3(
        (sdf_x - sdf_center) / delta,
        (sdf_y - sdf_center) / delta,
        (sdf_z - sdf_center) / delta
    )
    
    # Normalize and return (pointing outward, so negate if needed)
    grad_len = tm.length(gradient)
    result = tm.vec3(0.0, 0.0, 1.0)  # Default normal if gradient is zero
    if grad_len > EPSILON:
        result = tm.normalize(gradient)
    return result

@ti.func
def rayMarchMetaball(metaball: Metaball, ray: Ray, t_min: float, t_max: float) -> Intersection:
    """Ray march through a metaball to find intersection.
    Args:
        metaball: The metaball object
        ray: Ray in world space
        t_min: Minimum t value for valid intersection
        t_max: Maximum t value for valid intersection
    Returns:
        Intersection: intersection data (is_hit, t, normal, point, material)
    """
    hit = Intersection()  # default is no intersection
    
    # Transform ray to local space
    local_ray = changeRayFrame(ray, metaball.M_inv)
    O_local = local_ray.origin
    D_local = tm.normalize(local_ray.direction)
    
    # Ray marching parameters
    max_steps = 256
    min_step_size = 0.001
    max_step_size = 0.1
    surface_threshold = 0.001  # How close we need to get to surface
    
    t = t_min
    hit_found = False
    
    # Ray march along the ray
    for step in range(max_steps):
        if t > t_max:
            break
        
        # Current point along ray
        P_local = O_local + t * D_local
        
        # Evaluate SDF at current point
        sdf = metaball_sdf(metaball, P_local)
        
        # Check if we've hit the surface (SDF close to zero)
        if ti.abs(sdf) < surface_threshold:
            hit_found = True
            break
        
        # Step forward by the absolute value of SDF (distance to surface)
        # But clamp it to reasonable bounds
        step_size = ti.abs(sdf)
        if step_size < min_step_size:
            step_size = min_step_size
        if step_size > max_step_size:
            step_size = max_step_size
        
        t += step_size
    
    if hit_found:
        # Compute intersection point
        P_local = O_local + t * D_local
        
        # Compute normal at intersection point
        N_local = compute_sdf_normal(metaball, P_local)
        
        # Transform back to world space
        P_world_h = metaball.M @ tm.vec4(P_local, 1.0)
        hit.position = P_world_h.xyz
        
        M_inv_T = metaball.M_inv.transpose()
        N_world_h = M_inv_T @ tm.vec4(N_local, 0.0)
        hit.normal = tm.normalize(N_world_h.xyz)
        
        hit.t = t
        hit.mat = metaball.material
        hit.is_hit = True
        hit.hit_count = 0  # No motion blur for metaballs
    
    return hit

