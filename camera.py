
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
		self.width = width
		self.height = height		
		self.distance_to_plane = 1.0

		# TODO: Objective 1: Compute camera frame basis vectors, and top bottom left right for ray generation
		# NOTE: glm vectors are passed in to permit the work to be done here in python, but stored vectors must be tm.vec3
		self.eye_position = tm.vec3(eye_position.x, eye_position.y, eye_position.z)

		# set up for u v w 
		w_glm = glm.normalize(eye_position - lookat)
		self.w = tm.vec3(w_glm.x, w_glm.y, w_glm.z)

		u_glm = glm.normalize(glm.cross(up, w_glm))
		self.u = tm.vec3(u_glm.x, u_glm.y, u_glm.z)

		v_glm = glm.cross(w_glm, u_glm)
		self.v = tm.vec3(v_glm.x, v_glm.y, v_glm.z)

		# set up for top bottom left right
		fovy_rad = math.radians(fovy)
		half_height = self.distance_to_plane * math.tan(fovy_rad / 2.0)

		self.top = half_height
		self.bottom = -half_height

		aspect_ratio = self.width / self.height
		half_width = half_height * aspect_ratio

		self.right = half_width
		self.left = -half_width
		
		
	




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
		u_coord = self.left + (x + x_offset) * (self.right - self.left) / self.width
		v_coord = self.bottom + (y + y_offset) * (self.top - self.bottom) / self.height
		d = (
            - self.distance_to_plane * self.w
            + u_coord * self.u
            + v_coord * self.v
        )
		d = tm.normalize(d)
		return Ray(self.eye_position, d)

