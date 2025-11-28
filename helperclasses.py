import taichi as ti
import taichi.math as tm

@ti.dataclass
class Ray:
    origin: tm.vec3
    direction: tm.vec3

@ti.func
def getRayDistance(ray: Ray, point: tm.vec3) -> float:
    return tm.length(point - ray.origin)

@ti.func
def getRayPoint(ray: Ray, t: float) -> tm.vec3:
    return ray.origin + ray.direction * t

@ti.func
def changeRayFrame(ray: Ray, M: tm.mat4) -> Ray:
    # TODO: Objective 4: Ray and Geometry Transformations


    return ray # change this placeholder to return a transformed ray

@ti.dataclass
class Material:
        id: int
        diffuse: tm.vec3    # kd diffuse coefficient
        specular: tm.vec3    # ks specular coefficient
        shininess: tm.vec3    # specular exponent

@ti.dataclass
class Light:
        ltype: int       # type is either 0 for "directional" or 1 for "point"
        id: int
        colour: tm.vec3   # colour and intensity of the light
        vector: tm.vec3    # position, or normalized direction towards light, depending on the light type
        attenuation: tm.vec3   # attenuation coeffs [quadratic, linear, constant] for point lights

@ti.dataclass 
class Intersection:
        # All fields will be set to zero on creation, otherwise specified in this order on construction
        is_hit: bool
        t: float
        normal: tm.vec3
        position: tm.vec3
        mat: Material

@ti.func
def changeIntersectFrame(intersect: Intersection, M: tm.mat4, M_inv: tm.mat4) -> Intersection:

    # TODO: Objective 4: Ray and Geometry Transformations


    return intersect # change this placeholder to return a transformed intersection