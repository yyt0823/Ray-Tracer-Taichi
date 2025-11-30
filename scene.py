import geometry as geom
from helperclasses import Ray, Intersection
from camera import Camera

import numpy as np

import taichi as ti
import taichi.math as tm

shadow_epsilon = 10**(-2)

@ti.data_oriented
class Scene:
    def __init__(self,
                 jitter: bool,
                 samples: int,
                 camera: Camera,
                 ambient: tm.vec3,
                 lights: ti.template(),
                 nb_lights: int,
                 spheres: ti.template(),
                 nb_spheres: int,
                 planes: ti.template(),
                 nb_planes: int,
                 aaboxes: ti.template(),
                 nb_aaboxes: int,
                 meshes: ti.template(),
                 nb_meshes: int,
                 cones: ti.template(),
                 nb_cones: int,
                 meshes_verts: np.array,
                 meshes_faces: np.array,
                 ):
        self.jitter = jitter  # should rays be jittered
        self.samples = samples  # number of rays per pixel
        self.camera = camera
        self.ambient = ambient  # ambient lighting
        self.lights = lights  # all lights in the scene
        self.nb_lights = nb_lights
        self.spheres = spheres
        self.planes = planes
        self.aaboxes = aaboxes
        self.meshes = meshes
        self.cones = cones
        self.nb_spheres = nb_spheres
        self.nb_planes = nb_planes
        self.nb_aaboxes = nb_aaboxes
        self.nb_meshes = nb_meshes
        self.nb_cones = nb_cones

        self.meshes_verts = ti.Vector.field(3, shape=(max(1, meshes_verts.shape[0])), dtype=float)
        self.meshes_verts.from_numpy(meshes_verts)
        self.meshes_faces = ti.Vector.field(3, shape=(max(1, meshes_faces.shape[0])), dtype=int)
        self.meshes_faces.from_numpy(meshes_faces)

        self.image = ti.Vector.field( n=3, dtype=float, shape=(self.camera.width, self.camera.height) )

        self.offsets = ti.field(dtype=ti.f32, shape=((self.samples - 1) * (self.samples - 1) + 1, 2))


    @ti.kernel
    def render( self, iteration_count: int ):
        for x,y in ti.ndrange(self.camera.width, self.camera.height):
            if (y == x) and x%10 == 0: print(".",end='')
            ray = self.camera.create_ray( x, y, self.jitter )
            intersect = self.intersect_scene(ray, 0, float('inf'))
            sample_colour = tm.vec3(0, 0, 0) # background colour
            if intersect.is_hit:
                sample_colour = self.compute_shading(intersect, ray)
            self.image[x,y] += (sample_colour - self.image[x,y]) / iteration_count
        print() # end of line after one dot per 10 rows


    @ti.func
    def intersect_scene(self, ray: Ray, t_min: float, t_max: float) -> Intersection:
        best = Intersection() # default is no intersection (is_hit = False)        
        ti.loop_config(serialize=True) 
        for i in range(self.nb_spheres):
            hit = geom.intersectSphere(self.spheres[i], ray, t_min, t_max )
            if hit.is_hit: best = hit; t_max = hit.t # keep best hit only
        ti.loop_config(serialize=True) 
        for i in range(self.nb_planes):
            hit = geom.intersectPlane(self.planes[i], ray, t_min, t_max )
            if hit.is_hit: best = hit; t_max = hit.t                
        ti.loop_config(serialize=True) 
        for i in range(self.nb_aaboxes):
            hit = geom.intersectAABox(self.aaboxes[i], ray, t_min, t_max )
            if hit.is_hit: best = hit; t_max = hit.t
        ti.loop_config(serialize=True) 
        for i in range(self.nb_meshes):
            hit = geom.intersectMesh(self.meshes[i], self.meshes_verts, self.meshes_faces, ray, t_min, t_max)
            if hit.is_hit: best = hit; t_max = hit.t
        ti.loop_config(serialize=True) 
        for i in range(self.nb_cones):
            hit = geom.intersectCone(self.cones[i], ray, t_min, t_max)
            if hit.is_hit: best = hit; t_max = hit.t
        return best


    @ti.func
    def compute_local_shading(self, intersect: Intersection, ray: Ray) -> tm.vec3:
        """
        Compute ONLY local illumination (ambient + diffuse + specular).
        Does NOT include reflection - that's handled iteratively in compute_shading.
        """
        sample_colour = tm.vec3(0, 0, 0)
        
        # Ambient shading
        sample_colour += self.ambient * intersect.mat.diffuse

        ti.loop_config(serialize=True) 
        for l in range(self.nb_lights):
            current_light = self.lights[l]

            # Initialize variables
            L_norm = tm.vec3(0.0)
            t_max_shadow = float('inf') # Default for directional light
            light_distance = 0.0
            attenuation_factor = 1.0    

            # --- 1. Determine Light Vector (L), Distance, and Attenuation ---
            
            if current_light.ltype == 1: # Point Light
                # get the light direction
                L_vec = current_light.vector - intersect.position
                light_distance = tm.length(L_vec)
                
                # Normalize
                L_norm = L_vec / light_distance 
                
                # calculate the t_max
                t_max_shadow = light_distance - shadow_epsilon
                
                # Calculate Attenuation Factor
                # Attenuation = 1 / (C + L*d + Q*d^2)
                
                
                C = current_light.attenuation.z # constant
                L = current_light.attenuation.y # linear
                Q = current_light.attenuation.x # quadratic
                d = light_distance
                
                denominator = C + L * d + Q * d * d
                
                if denominator > 0.0:
                    attenuation_factor = 1.0 / denominator
            
            elif current_light.ltype == 0: # Directional Light
                # L_norm can get from the light field
                L_norm = current_light.vector 

                
            N_norm = intersect.normal 

            # Calculate N dot L and clamp it to 0
            N_dot_L = tm.max(0.0, tm.dot(N_norm, L_norm)) 
            
            # --- shadow check ---
            shadow_attenuation = 1.0
            
            if N_dot_L > 0.0: 
                
                shadow_ray = Ray(intersect.position, L_norm)
                shadow_hit = self.intersect_scene(shadow_ray, shadow_epsilon, t_max_shadow) 
                
                if shadow_hit.is_hit:
                    shadow_attenuation = 0.0
            
            # --- shading ---
            
            # Only proceed if the point is not in shadow
            if shadow_attenuation > 0.0:
                # Diffuse term
                diffuse_term = current_light.colour * intersect.mat.diffuse * N_dot_L * attenuation_factor

                # Specular set up
                V_norm = -ray.direction 
                # halfway vector
                H_norm = tm.normalize(L_norm + V_norm)
                N_dot_H = tm.max(0.0, tm.dot(N_norm, H_norm))
                specular_factor = tm.pow(N_dot_H, intersect.mat.shininess)
                
                # Specular term: apply light color, specular color, factor, AND attenuation
                specular_term = current_light.colour * intersect.mat.specular * specular_factor * attenuation_factor

                # Accumulation
                sample_colour += diffuse_term + specular_term

        return sample_colour

    @ti.func
    def compute_shading(self, intersect: Intersection, ray: Ray) -> tm.vec3:
        final_color = tm.vec3(1.0, 0.0, 0.0)
        if not intersect.mat.reflection:
            final_color =  self.compute_local_shading(intersect, ray)
        else:
            final_color = self.compute_reflection(intersect, ray)
        return final_color
    
    @ti.func
    def compute_reflection(self, intersect: Intersection, ray: Ray) -> tm.vec3:
        """
        Compute pure mirror reflection - single bounce only.
        For mirrors: shows ONLY what is reflected, no local color from the mirror itself.
        """
        # Reflection formula: R = I - 2 * (I · N) * N
        # I = incident direction (ray direction points toward surface)
        # N = surface normal (points away from surface)
        # R = reflected direction (points away from surface)
        
        I = ray.direction  # Incident direction (toward surface)
        N = intersect.normal  # Surface normal (already in world space, points away)
        print("N", N)
        print("I", I)
        
        # Ensure normal points away from surface (dot product should be negative for incoming ray)
        # If dot is positive, flip the normal
        dot_I_N = tm.dot(I, N)
        if dot_I_N > 0.0:
            N = -N
        
        # Compute reflected direction
        R_dir = I - 2.0 * tm.dot(I, N) * N
        
        # Start reflected ray slightly offset from surface to avoid self-intersection
        R_origin = intersect.position + N * shadow_epsilon
        reflected_ray = Ray(R_origin, tm.normalize(R_dir))
        
        # Cast the reflected ray into the scene
        reflected_hit = self.intersect_scene(reflected_ray, shadow_epsilon, 20.0)
        
        # If we hit something, return its local shading (what the mirror reflects)
        result_colour = tm.vec3(0.0, 0.0, 0.0)
        if reflected_hit.is_hit:
            result_colour = self.compute_local_shading(reflected_hit, reflected_ray)
        
        return result_colour
