"""
Random walk algorithm implementation for a mobile robot
equipped with 4 ranger sensors (front, back, left and right)
for obstacles detection

author: Ruslan Agishev (agishev_ruslan@mail.ru)
reference: https://ieeexplore.ieee.org/abstract/document/6850799/s
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

def plot_arrow(x, y, yaw, length=5, width=1):  # pragma: no cover
    plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)

def plot_robot(pose, params):
	r = int(100*params.sensor_range_m)
	plt.plot([pose[0]-r*np.cos(pose[2]), pose[0]+r*np.cos(pose[2])],
			 [pose[1]-r*np.sin(pose[2]), pose[1]+r*np.sin(pose[2])], '--', color='b')
	plt.plot([pose[0]-r*np.cos(pose[2]+np.pi/2), pose[0]+r*np.cos(pose[2]+np.pi/2)],
		     [pose[1]-r*np.sin(pose[2]+np.pi/2), pose[1]+r*np.sin(pose[2]+np.pi/2)], '--', color='b')
	plt.plot(pose[0], pose[1], 'ro', markersize=5)
	plot_arrow(pose[0], pose[1], pose[2])

def obstacle_check(pose, params):
	gmap = params.gmap

	r = int(100*params.sensor_range_m)
	back = [pose[0]-r*np.cos(pose[2]), pose[1]-r*np.sin(pose[2])]
	front = [pose[0]+r*np.cos(pose[2]), pose[1]+r*np.sin(pose[2])]
	right = [pose[0]+r*np.cos(pose[2]+np.pi/2), pose[1]+r*np.sin(pose[2]+np.pi/2)]
	left = [pose[0]-r*np.cos(pose[2]+np.pi/2), pose[1]-r*np.sin(pose[2]+np.pi/2)]

	pi = np.array(pose[:2], dtype=int)
	backi = np.array(back, dtype=int)
	fronti = np.array(front, dtype=int)
	lefti = np.array(left, dtype=int)
	righti = np.array(right, dtype=int)

	obstacle = {
		'front': 0,
		'back':  0,
		'right': 0,
		'left':  0,
	}

	for i in np.arange(min(pi[0], fronti[0]), max(pi[0], fronti[0])+1):
		for j in np.arange(min(pi[1], fronti[1]), max(pi[1], fronti[1])+1):
			m = min(j, gmap.shape[0]-1); n = min(i, gmap.shape[1]-1)
			if gmap[m,n]:
				# print('FRONT collision')
				obstacle['front'] = 1

	for i in np.arange(min(pi[0], backi[0]), max(pi[0], backi[0])+1):
		for j in np.arange(min(pi[1], backi[1]), max(pi[1], backi[1])+1):
			m = min(j, gmap.shape[0]-1); n = min(i, gmap.shape[1]-1)
			if gmap[m,n]:
				# print('BACK collision')
				obstacle['back'] = 1

	for i in np.arange(min(pi[0], lefti[0]), max(pi[0], lefti[0])+1):
		for j in np.arange(min(pi[1], lefti[1]), max(pi[1], lefti[1])+1):
			m = min(j, gmap.shape[0]-1); n = min(i, gmap.shape[1]-1)
			if gmap[m,n]:
				# print('LEFT collision')
				obstacle['left'] = 1

	for i in np.arange(min(pi[0], righti[0]), max(pi[0], righti[0])+1):
		for j in np.arange(min(pi[1], righti[1]), max(pi[1], righti[1])+1):
			m = min(j, gmap.shape[0]-1); n = min(i, gmap.shape[1]-1)
			if gmap[m,n]:
				# print('RIGHT collision')
				obstacle['right'] = 1

	return obstacle

def meters2grid(pose_m, params):
    # [0, 0](m) -> [100, 100]
    # [1, 0](m) -> [100+100, 100]
    # [0,-1](m) -> [100, 100-100]
    nrows = int(params.map_width_m / params.map_resolution_m)
    ncols = int(params.map_length_m / params.map_resolution_m)
    if np.isscalar(pose_m):
        pose_on_grid = int( pose_m/params.map_resolution_m + ncols/2 )
    else:
        pose_on_grid = np.array( np.array(pose_m)/params.map_resolution_m + np.array([ncols/2, nrows/2]), dtype=int )
    return pose_on_grid
def grid2meters(pose_grid, params):
    # [100, 100] -> [0, 0](m)
    # [100+100, 100] -> [1, 0](m)
    # [100, 100-100] -> [0,-1](m)
    nrows = int(params.map_width_m / params.map_resolution_m)
    ncols = int(params.map_length_m / params.map_resolution_m)
    if np.isscalar(pose_grid):
        pose_meters = (pose_grid - ncols/2) * params.map_resolution_m
    else:
        pose_meters = ( np.array(pose_grid) - np.array([ncols/2, nrows/2]) ) * params.map_resolution_m
    return pose_meters

def left_shift(pose, r):
	left = [pose[0]+r*np.cos(pose[2]+np.pi/2), pose[1]+r*np.sin(pose[2]+np.pi/2)]
	return left
def right_shift(pose, r):
	right = [pose[0]-r*np.cos(pose[2]+np.pi/2), pose[1]-r*np.sin(pose[2]+np.pi/2)]
	return right
def back_shift(pose, r):
	back = pose
	back[:2] = [pose[0]-r*np.cos(pose[2]), pose[1]-r*np.sin(pose[2])]
	return back

def draw_map(obstacles, params):
	ax = plt.gca()
	ax.set_xlim([-2.5, 2.5])
	ax.set_ylim([-2.5, 2.5])
	w = params.map_length_m; l = params.map_width_m; c = params.map_center
	boundaries = np.array([ c+[-w/2., -l/2.], c+[-w/2., +l/2.], c+[+w/2., +l/2.], c+[+w/2., -l/2.] ])
	ax.add_patch( Polygon(boundaries, linewidth=2, edgecolor='k',facecolor='none') )
	for k in range(len(obstacles)):
		ax.add_patch( Polygon(obstacles[k], color='k', zorder=10) )

def visualize(traj, pose, params):
	plt.plot(traj[:,0], traj[:,1], 'g')
	# plot_robot(pose, params)
	plt.legend()

def add_obstacles_to_grid_map(obstacles, params):
    """ Obstacles dicretized map """
    grid = params.gmap
    # rectangular obstacles
    for obstacle in obstacles:
        x1 = meters2grid(obstacle[0][1], params); x2 = meters2grid(obstacle[2][1], params)
        y1 = meters2grid(obstacle[0][0], params); y2 = meters2grid(obstacle[2][0], params)
        if x1 > x2: tmp = x2; x2 = x1; x1 = tmp
        if y1 > y2: tmp = y2; y2 = y1; y1 = tmp
        grid[x1:x2, y1:y2] = 1
    return grid

class Params:
	def __init__(self):
		self.map_center = np.array([0, 0])
		self.map_width_m = 2.0
		self.map_length_m = 2.0
		self.map_resolution_m = 0.01 # [m]
		self.sensor_range_m = 0.1
		self.wall_thickness_m = 1.0*self.sensor_range_m
		self.simulation_time = 10 # [sec]
		self.numiters = 1000
		self.animate = 0
		self.vel = 0.5 # [m/s]
		self.create_borders_grid_map()

	def create_borders_grid_map(self):
		WIDTH = int(100 * (self.map_width_m))
		LENGTH = int(100 * (self.map_length_m))
		border = int(100 * self.wall_thickness_m)
		gmap = np.zeros([WIDTH, LENGTH])
		# walls
		gmap[:border, :] = 1
		gmap[-border:, :] = 1
		gmap[:, :border] = 1
		gmap[:, -border:] = 1
		self.gmap = gmap

params = Params()

obstacles = [
	np.array([[0.7, -0.9], [1.3, -0.9], [1.3, -0.8], [0.7, -0.8]]) + np.array([-1.0, 0.0]),
	np.array([[0.7, -0.9], [1.3, -0.9], [1.3, -0.8], [0.7, -0.8]]) + np.array([-1.0, 0.5]),
	np.array([[0.7, -0.9], [0.8, -0.9], [0.8, -0.3], [0.7, -0.3]]) + np.array([-1.0, 0.0]),        
]
params.gmap = add_obstacles_to_grid_map(obstacles, params)


def main():
	#    x,    y,      yaw
	pose = [0.3, 0.6, -np.pi/3]
	traj = pose[:2]
	plt.figure(figsize=(10,10))
	draw_map(obstacles, params)
	plt.plot(pose[0], pose[1], 'ro', markersize=20, label='Initial position')
	# while True:
	for _ in range(params.numiters):
		dv = 0.1*params.vel
		pose[0] += dv*np.cos(pose[2])
		pose[1] += dv*np.sin(pose[2])

		pose_grid = meters2grid(pose[:2], params)
		boundary = obstacle_check([pose_grid[0], pose_grid[1], pose[2]], params)
		# print(boundary)

		if boundary['right'] or boundary['front']:
			pose = back_shift(pose, 0.03)
			pose[2] -= np.pi/2 * np.random.uniform(0.2, 0.6)
		elif boundary['left']:
			pose = back_shift(pose, 0.03)
			pose[2] += np.pi/2 * np.random.uniform(0.2, 0.6)

		traj = np.vstack([traj, pose[:2]])
		
		if params.animate:
			plt.cla()
			draw_map(obstacles, params)
			visualize(traj, pose, params)
			plt.pause(0.1)

	visualize(traj, pose, params)
	plt.show()

if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
	    pass
		