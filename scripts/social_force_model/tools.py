#!/usr/bin/env python

import numpy as np

def meters2grid(pose_m, nrows=1000, ncols=1000):
    # [0, 0](m) -> [250, 250]
    # [1, 0](m) -> [250+100, 250]
    # [0,-1](m) -> [250, 250-100]
    pose_on_grid = np.array(pose_m)*100 + np.array([ncols/2, nrows/2])
    return np.array( pose_on_grid, dtype=int)

def grid2meters(pose_grid, nrows=1000, ncols=1000):
    # [250, 250] -> [0, 0](m)
    # [250+100, 250] -> [1, 0](m)
    # [250, 250-100] -> [0,-1](m)
    pose_meters = ( np.array(pose_grid) - np.array([ncols/2, nrows/2]) ) / 100.0
    return pose_meters

def normalize(vector):
    if np.linalg.norm(vector)==0: return np.zeros_like(vector)
    return vector / np.linalg.norm(vector)

def draw_map(obstacles_poses, R_obstacles, nrows=1000, ncols=1000):
    plt.grid()
    plt.xlabel('X')
    plt.ylabel('Y')
    ax = plt.gca()
    for pose in obstacles_poses:
        circle = plt.Circle(pose, R_obstacles, color='black')
        ax.add_artist(circle)

def draw_robot(robot):
    plt.plot(robot.pose[0], robot.pose[1], 'ro', color='blue')

def g(x):
    return np.max(x, 0)   # Keep compatiable with numpy in 1.14.0 version

def distance_to_wall(point, wall):
    # Calculate the distance from the point to the segment
    # and the unit vector from the point to the intersection with the segment
    p0 = np.array([wall[0],wall[1]])
    p1 = np.array([wall[2],wall[3]])
    d = p1 - p0
    ymp0 = point-p0
    t = np.dot(d,ymp0) / np.dot(d,d)
    if t <= 0.0:
        dist = np.sqrt(np.dot(ymp0, ymp0))
        cross = p0 + t*d
    elif t >= 1.0:
        ymp1 = point - p1
        dist = np.sqrt(np.dot(ymp1, ymp1))
        cross = p0 + t*d
    else:
        cross = p0 + t*d
        dist = np.linalg.norm(cross - point)
    npw = normalize(cross - point)
    return dist, npw