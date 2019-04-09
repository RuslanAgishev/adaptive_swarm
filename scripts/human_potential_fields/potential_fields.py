#!/usr/bin/env python

import numpy as np
from numpy.linalg import norm
from scipy.ndimage.morphology import distance_transform_edt as bwdist


def meters2grid(pose_m, nrows=500, ncols=500):
    # [0, 0](m) -> [250, 250]
    # [1, 0](m) -> [250+100, 250]
    # [0,-1](m) -> [250, 250-100]
    pose_on_grid = np.array(pose_m)*100 + np.array([ncols/2, nrows/2])
    return np.array( pose_on_grid, dtype=int)
def grid2meters(pose_grid, nrows=500, ncols=500):
    # [250, 250] -> [0, 0](m)
    # [250+100, 250] -> [1, 0](m)
    # [250, 250-100] -> [0,-1](m)
    pose_meters = ( np.array(pose_grid) - np.array([ncols/2, nrows/2]) ) / 100.0
    return pose_meters

def grid_map(obstacles_poses, R_obstacles, borders_width=2, nrows=500, ncols=500):
    """ Obstacles map """
    obstacles_map = np.zeros((nrows, ncols));
    [x, y] = np.meshgrid(np.arange(ncols), np.arange(nrows))
    for pose in obstacles_poses:
        pose = meters2grid(pose)
        x0 = pose[0]; y0 = pose[1]
        # cylindrical obstacles
        t = ((x - x0)**2 + (y - y0)**2) < (100*R_obstacles)**2
        obstacles_map[t] = 1;
    # borders are obstacles
    w = borders_width
    obstacles_map[:,:int(w/2)] = 1; obstacles_map[:,-int(w/2):] = 1
    obstacles_map[:int(w/2),:] = 1; obstacles_map[-int(w/2):,:] = 1
    return obstacles_map

def gradient_planner(current_point, f, ncols=500, nrows=500, movement_speed=0.06):
    """
    GradientBasedPlanner : This function computes the next_point
    given current location, goal location and potential map, f.
    It also returns mean velocity, V, of the gradient map in current point.
    """
    # print meters2grid(current_point)
    [gy, gx] = np.gradient(-f);
    iy, ix = np.array( meters2grid(current_point), dtype=int )
    # w = 10 # smoothing window size for gradient-velocity
    # vx = np.mean(gx[ix-int(w/2) : ix+int(w/2), iy-int(w/2) : iy+int(w/2)])
    # vy = np.mean(gy[ix-int(w/2) : ix+int(w/2), iy-int(w/2) : iy+int(w/2)])
    vx = gx[ix, iy]
    vy = gy[ix, iy]
    V = np.array([vx, vy])
    dt = movement_speed / norm(V) if norm(V)!=0 else 0
    next_point = current_point + dt*V
    return next_point

def combined_potential(obstacles_poses, R_obstacles, goal, influence_radius=2, attractive_coef=1./700, repulsive_coef=200, nrows=500, ncols=500):
    """ Repulsive potential """
    obstacles_map = grid_map(obstacles_poses, R_obstacles)
    goal = meters2grid(goal)
    d = bwdist(obstacles_map==0)
    d2 = (d/100.) + 1 # Rescale and transform distances
    d0 = influence_radius
    nu = repulsive_coef
    repulsive = nu*((1./d2 - 1./d0)**2)
    repulsive [d2 > d0] = 0
    """ Attractive potential """
    [x, y] = np.meshgrid(np.arange(ncols), np.arange(nrows))
    xi = attractive_coef
    attractive = xi * ( (x - goal[0])**2 + (y - goal[1])**2 )
    """ Combine terms """
    f = attractive + repulsive
    return f