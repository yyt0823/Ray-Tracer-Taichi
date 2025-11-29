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
        self.nb_spheres = nb_spheres
        self.nb_planes = nb_planes
        self.nb_aaboxes = nb_aaboxes
        self.nb_meshes = nb_meshes

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
        return best


    @ti.func
    def compute_shading(self, intersect: Intersection, ray: Ray) -> tm.vec3:
        sample_colour = tm.vec3(0, 0, 0)

        # Ambient shading
        sample_colour += self.ambient * intersect.mat.diffuse

        ti.loop_config(serialize=True) 
        for l in range(self.nb_lights):
            current_light = self.lights[l]

            # light vector (direction)
            L_vec = current_light.vector - intersect.position
            light_distance = tm.length(L_vec)
            
            # normalized light vector
            L_norm = L_vec / light_distance 
            N_norm = intersect.normal 

            N_dot_L = tm.max(0.0, tm.dot(N_norm, L_norm)) 
            
            shadow_attenuation = 1.0
            # if facing the light
            if N_dot_L > 0.0: 
                
                # Shadow Ray setup
                shadow_ray = Ray(intersect.position, L_norm)
                t_max_shadow = light_distance - shadow_epsilon
                
                # cast a ray from the intersection point to the light source and check if it hits an object
                shadow_hit = self.intersect_scene(shadow_ray, shadow_epsilon, t_max_shadow) 
                
                if shadow_hit.is_hit:
                    shadow_attenuation = 0.0
            
            # Only proceed if the point is not in shadow
            if shadow_attenuation > 0.0:
                
                # diffuse term
                diffuse_term = current_light.colour * intersect.mat.diffuse * N_dot_L

                # specular term
                V_norm = -ray.direction 
                
                # H_norm: Halfway Vector (H) = normalize(L + V)
                H_norm = tm.normalize(L_norm + V_norm)
                
                # Specular factor: max(0, N dot H)^shininess
                N_dot_H = tm.max(0.0, tm.dot(N_norm, H_norm))
                
                specular_factor = tm.pow(N_dot_H, intersect.mat.shininess)
                
                # Specular term: LightColor * SpecularColor * SpecularFactor
                specular_term = current_light.colour * intersect.mat.specular * specular_factor

                # --- 5. Accumulation ---
                sample_colour += diffuse_term + specular_term
            
            
            
            # TODO: Objective 6: Implement shadow rays

            continue

        return sample_colour