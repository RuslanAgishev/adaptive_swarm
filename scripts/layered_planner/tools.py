#!/usr/bin/env python

import numpy as np
from numpy.linalg import norm
from math import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import xlwt
import time
import os
from potential_fields import meters2grid, grid2meters
from low_pass_filter import butter_lowpass_filter
import psutil

def memory_usage():
    # return the memory usage in MiB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20) # in mebibyte: 1 MiB = 1024^2 B
    return mem

def cpu_usage():
    # return the CPU usage in %
    # cpu_usage = float(os.popen('''grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage }' ''').readline())
    cpu_usage = psutil.cpu_percent()
    return cpu_usage # [%]

def draw_map(obstacles):
    # Obstacles. An obstacle is represented as a polygon of a number of points. 
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


def postprocessing(metrics, params, visualize=1):
    for robot in metrics.robots:
        robot.path_length = path_length(robot.route)
        print("Robot %d path length: %.2f [m]" %(robot.id, robot.path_length))
    metrics.t_reach_goal = metrics.t_array[-1]
    print("Time to reach goal: %.2f [s]" %metrics.t_reach_goal)
    print("\nCentroid path: %.2f [m]" %metrics.centroid_path_length)
    if visualize:
        plt.figure(figsize=(10,10))
        plt.title("Drones trajectories.")
        for robot in metrics.robots:
            plt.plot(robot.route[:,0], robot.route[:,1], '--', label='drone %d' %robot.id, linewidth=2)
        plt.plot(metrics.centroid_path[:,0], metrics.centroid_path[:,1], linewidth=3, label='centroid', color='k')
        plt.legend()
        plt.grid()

    print("\n")
    for robot in metrics.robots:
        print("Robot %d Average Velocity: %.2f [m/s]" %( robot.id, np.mean(np.array(robot.vel_array)) ))
        print("Robot %d Max Velocity: %.2f [m/s]" %( robot.id, np.max(np.array(robot.vel_array)) ))
        metrics.vels_mean.append( np.mean(np.array(robot.vel_array)) )
        metrics.vels_max.append( np.max(np.array(robot.vel_array)) )
    if visualize:
        plt.figure(figsize=(10,6))
        plt.title( "Robots Velocities" )
        plt.plot( metrics.t_array, metrics.robots[0].vel_array, '-', color='k', label='drone 1', linewidth=3)
        for r in range(1, len(metrics.robots)):
            plt.plot(metrics.t_array, metrics.robots[r].vel_array, '--', label='drone %d' %(r+1), linewidth=2)
        plt.xlabel('Time, [s]')
        plt.ylabel('Velocity, [m/s]')
        plt.legend()
        plt.grid()

    mean_vels = []; mean_accs = []; mean_jerks = []; mean_snaps = []
    for robot in metrics.robots:
        t = metrics.t_array
        vel = robot.vel_array
        vel = butter_lowpass_filter(vel, cutoff=2, fs=14)
        acc = np.diff(vel) / np.diff(t)
        # acc = butter_lowpass_filter(acc, cutoff=4, fs=14)
        jerk = np.diff(acc) / np.diff(t[:-1])
        # jerk = butter_lowpass_filter(jerk, cutoff=4, fs=14)
        snap = np.diff(jerk) / np.diff(t[:-2])
        mean_vels.append( np.mean(np.abs(vel)) )
        mean_accs.append( np.mean(np.abs(acc)) )
        mean_jerks.append( np.mean(np.abs(jerk)) )
        mean_snaps.append( np.mean(np.abs(snap)) )
        # print 'Mean vel robot %d:'%robot.id, np.mean(np.abs(vel))
        # print 'Mean acc robot %d:'%robot.id, np.mean(np.abs(acc))
        # print 'Mean jerk robot %d:'%robot.id,np.mean(np.abs(jerk))
        # print 'Mean snap robot %d:'%robot.id,np.mean(np.abs(snap))
        # plt.figure()
        # plt.plot(t, vel, label='Vel drone %d'%robot.id)
        # plt.plot(t[1:], acc, label='Acc drone %d'%robot.id)
        # plt.plot(t[2:], jerk, label='Jerk drone %d'%robot.id)
        # plt.plot(t[3:], snap, label='Snap drone %d'%robot.id)
        # plt.legend()
    metrics.vel_mean = np.mean( mean_vels )
    metrics.acc_mean = np.mean( mean_accs )
    metrics.jerk_mean = np.mean( mean_jerks )
    metrics.snap_mean = np.mean( mean_snaps )

    for i in range(len(metrics.robots[0].route)-1):
        X = np.array([]); Y = np.array([]); robots_poses = []
        for robot in metrics.robots: robots_poses.append(robot.route[i,:])
        robots_poses.sort(key=lambda p: atan2(p[1]-metrics.centroid_path[i,1], p[0]-metrics.centroid_path[i,0]))
        for pose in robots_poses:
            X = np.append( X, pose[0] )
            Y = np.append( Y, pose[1] )
        metrics.area_array.append(poly_area(X,Y))
    # default formation area
    if params.num_robots==3: S0 = 0.5*sqrt(3)/2.*params.interrobots_dist**2
    elif params.num_robots==4: S0 = sqrt(3)/2.*params.interrobots_dist**2
    else: S0 = metrics.area_array[1]
    metrics.area_array = metrics.area_array[1:]
    metrics.S_min = np.min( metrics.area_array )
    metrics.S_default = S0
    metrics.S_mean = np.mean( metrics.area_array )
    metrics.S_max = np.max( metrics.area_array )
    print("\nMin formation area: %.2f [m^2]" %metrics.S_min)
    print("Default formation area: %f [m^2]" %S0)
    print("Mean formation area: %.2f [m^2]" %metrics.S_mean)
    print("Max formation area: %.2f [m^2]" %metrics.S_max)
    if visualize:
        plt.figure(figsize=(10,6))
        plt.title("Area of robots' formation")
        plt.plot(metrics.t_array[:-1], metrics.area_array, 'k', label='Formation area', linewidth=2)
        plt.plot(metrics.t_array, S0*np.ones_like(metrics.t_array), '--', label='Default area', linewidth=2)
        plt.plot(metrics.t_array, metrics.S_mean*np.ones_like(metrics.t_array), '--', color='r', label='Mean area', linewidth=2)
        plt.xlabel('Time, [s]')
        plt.ylabel('Formation area, [m^2]')
        plt.grid()
        plt.legend()

    metrics.R_formation_mean = np.mean( metrics.max_dists_array )

    metrics.cpu_usage_mean = np.mean( metrics.cpu_usage_array )
    metrics.memory_usage_mean = np.mean( metrics.memory_usage_array )
    print("\nMean CPU usage: %.2f [percentage]" %metrics.cpu_usage_mean)
    print("Mean memory usage: %.2f [MiB]" %metrics.memory_usage_mean)

    if visualize:
        plt.figure(figsize=(10,6))
        l = params.interrobots_dist
        plt.plot(metrics.t_array, l*np.ones_like(metrics.t_array), '--', label='Default distance', linewidth=2)
        plt.plot(metrics.t_array, metrics.mean_dists_array, label='Mean inter-robots distance', color='r', linewidth=2)
        plt.plot(metrics.t_array, metrics.max_dists_array, label='Max inter-robots distance', color='k', linewidth=2)
        plt.grid()
        plt.xlabel('Time, [s]')
        plt.ylabel('Distance, [m]')
        plt.legend()

        fig = plt.figure(figsize=(10,10))
        ax = fig.gca(projection='3d')
        # plt.title ('Repulsive Potential')
        # start = meters2grid(xy_start); #ax.scatter3D(start[0], start[1], 100, color='r', s=100, zorder=10)
        # goal = meters2grid(xy_goal); #ax.scatter3D(goal[0], goal[1], 100, color='g', s=100, zorder=10)
        X, Y = np.meshgrid (np.arange(500), np.arange(500))
        Z = metrics.robots[1].U_r if params.num_robots>1 else metrics.robots[0].U_r
        surf = ax.plot_surface(X, Y, Z/np.max(Z)*200, cmap=plt.cm.jet, linewidth=0.01, vmin=0.0, vmax=200, zorder=1)
        fig.colorbar( surf, shrink=0.5, aspect=5)
        traj = meters2grid( metrics.centroid_path )
        ax.plot(traj[:,0], traj[:,1], 100*np.ones(traj.shape[0]), label='Centroid trajectory', linewidth=3, color='r', zorder=0)
        for robot in metrics.robots:
            traj = meters2grid( robot.route )
            ax.plot(traj[:,0], traj[:,1], 100*np.ones(traj.shape[0]), '--', label="Robots' trajectories", linewidth=1, color='y', zorder=1)
        ax.view_init(elev=90, azim=-90)
        ax.set_xlabel('X, [cm]')
        ax.set_ylabel('Y, [cm]')
        ax.set_zlabel('Z, [cm]')
        ax.set_zlim([0, 400])


def save_data(metrics, folder_name='output_%f'%time.time()):
    #style0 = xlwt.easyxf('font: name Times New Roman, color-index red, bold on', num_format_str='#,##0.00')
    #style1 = xlwt.easyxf(num_format_str='D-MMM-YY')

    wb = xlwt.Workbook()
    ws = wb.add_sheet('Metrics')

    ws.write(0, 0, 'T_reach goal'); ws.write(0, 1, metrics.t_reach_goal)
    ws.write(1, 0, 'Robots_path_lengths')
    for i in range(len(metrics.robots)):
        ws.write(1, i+1, metrics.robots[i].path_length)
    ws.write(2, 0, 'Centroid_path_length'); ws.write(2, 1, metrics.centroid_path_length)

    ws.write(3, 0, 'Average vels')
    for i in range(len(metrics.vels_mean)):
        ws.write(3, i+1, metrics.vels_mean[i])
    ws.write(4, 0, 'Max vels')
    for i in range(len(metrics.vels_max)):
        ws.write(4, i+1, metrics.vels_max[i])
    ws.write(5, 0, 'S_min'); ws.write(5, 1, metrics.S_min)
    ws.write(6, 0, 'S_default'); ws.write(6, 1, metrics.S_default)
    ws.write(7, 0, 'S_mean'); ws.write(7, 1, metrics.S_mean)
    ws.write(8, 0, 'S_max'); ws.write(8, 1, metrics.S_max)

    ws.write(9,0, 'R_formation_mean'); ws.write(9,1, metrics.R_formation_mean)

    ws.write(10, 0, 'Vel_mean'); ws.write(10, 1, metrics.vel_mean)
    ws.write(11, 0, 'Acc_mean'); ws.write(11, 1, metrics.acc_mean)
    ws.write(12, 0, 'Jerk_mean'); ws.write(12, 1, metrics.jerk_mean)
    ws.write(13, 0, 'Snap_mean'); ws.write(13, 1, metrics.snap_mean)

    ws.write(14, 0, 'CPU usage mean'); ws.write(14, 1, metrics.cpu_usage_mean)
    ws.write(15, 0, 'Memory usage mean'); ws.write(15, 1, metrics.memory_usage_mean)

    os.mkdir(metrics.folder_to_save+folder_name)
    wb.save(metrics.folder_to_save+folder_name+'/results.xls')

    r = 0
    for robot in metrics.robots:
        r+=1
        wb = xlwt.Workbook()
        ws = wb.add_sheet('Trajectories')

        ws.write(0,0, 'Time, [s]')
        for i in range(len(metrics.t_array)):
            ws.write(i+1,0, metrics.t_array[i])

        ws.write(0,1, 'X [m]'); ws.write(0,2, 'Y [m]')
        for i in range(robot.route.shape[0]):
            x, y = robot.route[i,:]
            ws.write(i+1,1, x); ws.write(i+1,2, y)

        ws.write(0,3, 'V [m/s]')
        for i in range(len(robot.vel_array)):
            v = robot.vel_array[i]
            ws.write(i+1, 3, v)

        wb.save( metrics.folder_to_save+folder_name+'/robot%d.xls'%r  )
