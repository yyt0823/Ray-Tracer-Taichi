
from helperclasses import Ray
import taichi as ti
from pyglm import glm
import taichi.math as tm

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


		# Compute proper values below!
		self.top = 1
		self.bottom = -1
		self.left = -1
		self.right = 1
		self.u = tm.vec3(1,0,0)
		self.v = tm.vec3(0,0,0)
		self.w = tm.vec3(0,0,1)
		self.eye_position = tm.vec3(eye_position.x, eye_position.y, eye_position.z)



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


		return Ray( self.eye_position, tm.vec3(0,0,-1) )  # temporary ray
