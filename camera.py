
from helperclasses import Ray
import taichi as ti
from pyglm import glm
import taichi.math as tm
import math

@ti.data_oriented
class Camera:
	def __init__(self, width, height, eye_position:glm.vec3, lookat:glm.vec3, up:glm.vec3, fovy) -> None:
		'''	Initialize the camera with given parameters.
		Args:
			width (int): width of the image in pixels
			height (int): height of the image in pixels
			eye_position (glm.vec3): position of the camera in world space
			lookat (glm.vec3): point the camera is looking at
			up (glm.vec3): up direction for the camera
			fovy (float): vertical field of view in degrees
		Returns:
			None

			Note that while glm types are provided to construct the class, the 
			member variables must be of taich tm.vec3 types so that they used 
			in the create_ray Taichi function.  
		'''
		# Store width and height in fields for kernel caching (even though we keep Python members for external access)
		self.width = width
		self.height = height
		self.width_field = ti.field(dtype=int, shape=())
		self.width_field[None] = width
		self.height_field = ti.field(dtype=int, shape=())
		self.height_field[None] = height
		
		# Store camera members in fields for kernel caching
		self.distance_to_plane = ti.field(dtype=float, shape=())
		self.distance_to_plane[None] = 1.0
		
		self.eye_position = ti.Vector.field(3, dtype=float, shape=())
		self.eye_position[None] = tm.vec3(eye_position.x, eye_position.y, eye_position.z)

		# set up for u v w 
		w_glm = glm.normalize(eye_position - lookat)
		self.w = ti.Vector.field(3, dtype=float, shape=())
		self.w[None] = tm.vec3(w_glm.x, w_glm.y, w_glm.z)

		u_glm = glm.normalize(glm.cross(up, w_glm))
		self.u = ti.Vector.field(3, dtype=float, shape=())
		self.u[None] = tm.vec3(u_glm.x, u_glm.y, u_glm.z)

		v_glm = glm.cross(w_glm, u_glm)
		self.v = ti.Vector.field(3, dtype=float, shape=())
		self.v[None] = tm.vec3(v_glm.x, v_glm.y, v_glm.z)

		# set up for top bottom left right
		fovy_rad = math.radians(fovy)
		half_height = self.distance_to_plane[None] * math.tan(fovy_rad / 2.0)

		self.top = ti.field(dtype=float, shape=())
		self.top[None] = half_height
		self.bottom = ti.field(dtype=float, shape=())
		self.bottom[None] = -half_height

		aspect_ratio = self.width / self.height
		half_width = half_height * aspect_ratio

		self.right = ti.field(dtype=float, shape=())
		self.right[None] = half_width
		self.left = ti.field(dtype=float, shape=())
		self.left[None] = -half_width
		
		
	




	@ti.func
	def create_ray(self, x, y, jitter=False) -> Ray:
		''' Create a ray going through pixel (x,y) in image space 
		Args:
			x (int): pixel x coordinate
			y (int): pixel y coordinate
			jitter (bool): whether to apply jittering within the pixel for anti-aliasing
		Returns:
			Ray: generated ray from camera through pixel

			If jitter is True, the ray should not go through the exact center of the
			pixel, but instead be through a random point (uniform) inthe pixel area.
		'''

		
		# TODO: Objective 1: Generate ray from camera through pixel (col, row) with jittering
		x_offset = 0.5
		y_offset = 0.5
		if jitter: 
			# Sub-pixel offset should be random between 0.0 and 1.0
			x_offset = ti.random(float)
			y_offset = ti.random(float)
		u_coord = self.left[None] + (x + x_offset) * (self.right[None] - self.left[None]) / self.width_field[None]
		v_coord = self.bottom[None] + (y + y_offset) * (self.top[None] - self.bottom[None]) / self.height_field[None]
		d = (
            - self.distance_to_plane[None] * self.w[None]
            + u_coord * self.u[None]
            + v_coord * self.v[None]
        )
		d = tm.normalize(d)
		return Ray(self.eye_position[None], d)

