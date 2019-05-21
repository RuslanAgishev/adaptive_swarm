#!/usr/bin/env python

import numpy as np
from numpy.linalg import norm
from math import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import xlwt
import time


def draw_map(obstacles):
    # Obstacles. An obstacle is represented as a convex hull of a number of points. 
    # First row is x, second is y (position of vertices)

    # Bounds on world
    world_bounds_x = [-2.5, 2.5]
    world_bounds_y = [-2.5, 2.5]

    # Draw obstacles
    ax = plt.gca()
    ax.set_xlim(world_bounds_x)
    ax.set_ylim(world_bounds_y)
    for k in range(len(obstacles)):
        ax.add_patch( Polygon(obstacles[k], color='k', zorder=10) )


def draw_gradient(f, nrows=500, ncols=500):
    skip = 10
    [x_m, y_m] = np.meshgrid(np.linspace(-2.5, 2.5, ncols), np.linspace(-2.5, 2.5, nrows))
    [gy, gx] = np.gradient(-f);
    Q = plt.quiver(x_m[::skip, ::skip], y_m[::skip, ::skip], gx[::skip, ::skip], gy[::skip, ::skip])

def draw_sphere(pose, R=10):
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = R * np.outer(np.cos(u), np.sin(v)) + pose[0]
    y = R * np.outer(np.sin(u), np.sin(v)) + pose[1]
    z = R * np.outer(np.ones(np.size(u)), np.cos(v)) + pose[2]
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color='yellow')

def init_fonts(small=12, medium=16, big=26):
    SMALL_SIZE = small
    MEDIUM_SIZE = medium
    BIGGER_SIZE = big

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def waypts2setpts(P, params):
	"""
	construct a long array of setpoints, traj_global, with equal inter-distances, dx,
	from a set of via-waypoints, P = [[x0,y0], [x1,y1], ..., [xn,yn]]
	"""
	V = params.drone_vel * 1.3 # [m/s], the setpoint should travel a bit faster than the robot
	freq = params.ViconRate; dt = 1./freq
	dx = V * dt
	traj_global = np.array(P[-1])
	for i in range(len(P)-1, 0, -1):
		A = P[i]
		B = P[i-1]

		n = (B-A) / norm(B-A)
		delta = n * dx
		N = int( norm(B-A) / norm(delta) )
		sp = A
		traj_global = np.vstack([traj_global, sp])
		for i in range(N):
			sp += delta
			traj_global = np.vstack([traj_global, sp])
		sp = B
		traj_global = np.vstack([traj_global, sp])

	return traj_global


def formation(num_robots, leader_des, v, l):
    """
    geometry of the swarm: following robots desired locations
    relatively to the leader
    """
    u = np.array([-v[1], v[0]])
    """ followers positions """
    des2 = leader_des - v*l*sqrt(3)/2 + u*l/2
    des3 = leader_des - v*l*sqrt(3)/2 - u*l/2
    des4 = leader_des - v*l*sqrt(3)
    des5 = leader_des - v*l*sqrt(3)   + u*l
    des6 = leader_des - v*l*sqrt(3)   - u*l
    des7 = leader_des - v*l*sqrt(3)*3/2 - u*l/2
    des8 = leader_des - v*l*sqrt(3)*3/2 + u*l/2
    des9 = leader_des - v*l*sqrt(3)*2
    if num_robots<=1: return []
    if num_robots==2: return [des4]
    if num_robots==3: return [des2, des3]
    if num_robots==4: return [des2, des3, des4]
    if num_robots==5: return [des2, des3, des4, des5]
    if num_robots==6: return [des2, des3, des4, des5, des6]
    if num_robots==7: return [des2, des3, des4, des5, des6, des7]
    if num_robots==8: return [des2, des3, des4, des5, des6, des7, des8]
    if num_robots==9: return [des2, des3, des4, des5, des6, des7, des8, des9]
    
    return [des2, des3, des4]

def normalize(vector):
	if norm(vector)==0: return vector
	return vector / norm(vector)

def poses2polygons(poses, l=0.1):
    polygons = []
    for pose in poses:
        pose = np.array(pose)
        polygon = np.array([pose + [-l/2,-l/2], pose + [l/2,-l/2], pose + [l/2,l/2], pose + [-l/2,l/2]])
        polygons.append(polygon)
    return polygons

def poly_area(x,y):
    # https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    # https://en.wikipedia.org/wiki/Shoelace_formula
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def path_length(pose_array):
    length = 0
    for i in range( 1,len(pose_array) ):
        dl = np.linalg.norm(pose_array[i,:]-pose_array[i-1,:])
        # if dl > 0.01 and dl < 0.2: length += dl
        length += dl
    return length

def save_data(metrics):
    #style0 = xlwt.easyxf('font: name Times New Roman, color-index red, bold on', num_format_str='#,##0.00')
    #style1 = xlwt.easyxf(num_format_str='D-MMM-YY')

    wb = xlwt.Workbook()
    ws = wb.add_sheet('Metrics')

    ws.write(0, 0, 'T_reach goal'); ws.write(0, 1, metrics.t_reach_goal)
    ws.write(1, 0, 'Robots_path_lengths')
    for i in range(len(metrics.robots_path_lengths)):
        ws.write(1, i+1, metrics.robots_path_lengths[i])
    ws.write(2, 0, 'Centroid_path_length'); ws.write(2, 1, metrics.centroid_path_length)

    ws.write(3, 0, 'Average vels')
    for i in range(len(metrics.vels_mean)):
        ws.write(3, i+1, metrics.vels_mean[i])
    ws.write(4, 0, 'Max vels')
    for i in range(len(metrics.vels_max)):
        ws.write(4, i+1, metrics.vels_max[i])
    ws.write(5, 0, 'S_min'); ws.write(5, 1, metrics.S_min)
    ws.write(6, 0, 'S_mean'); ws.write(6, 1, metrics.S_mean)
    ws.write(7, 0, 'S_max'); ws.write(7, 1, metrics.S_max)

    ws.write(8,0, 'R_formation_mean'); ws.write(8,1, metrics.R_formation_mean)

    wb.save(metrics.folder_to_save+'output_%f.xls'%time.time())