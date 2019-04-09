#!/usr/bin/env python

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib import collections
plt.rcParams.update({'font.size': 22})
from math import *
import random
from impedance.impedance_modeles import *
from potential_fields import gradient_planner, combined_potential
import time

from progress.bar import FillingCirclesBar
from tasks import *
from threading import Thread
from multiprocessing import Process
import os

""" ROS """
import rospy
from geometry_msgs.msg import TransformStamped


def poly_area(x,y):
    # https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    # https://en.wikipedia.org/wiki/Shoelace_formula
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def move_obstacles(obstacles_poses, obstacles_goal_poses):
    """ All of the obstacles tend to go to the origin, (0,0) - point """
    # for pose in obstacles_poses:
    #   dx = random.uniform(0, 0.03);        dy = random.uniform(0,0.03);
    #   pose[0] -= np.sign(pose[0])*dx;      pose[1] -= np.sign(pose[1])*dy;

    """ Each obstacles tends to go to its selected goal point with random speed """
    for p in range(len(obstacles_poses)):
        pose = obstacles_poses[p]; goal = obstacles_goal_poses[p]
        dx, dy = (goal - pose) / norm(goal-pose) * 0.05 #random.uniform(0,0.05)
        pose[0] += dx;      pose[1] += dy;

    return obstacles_poses


""" initialization """
animate              = 1   # show 1-each frame or 0-just final configuration
random_obstacles     = 0   # randomly distributed obstacles on the map
num_random_obstacles = 8   # number of random circular obstacles on the map
num_robots           = 1   # <=9, number of drones in formation
moving_obstacles     = 0   # 0-static or 1-dynamic obstacles
impedance            = 0   # impedance links between the leader and followers (leader's velocity)
impedance_mode       = 'overdamped'    # 'underdamped', 'overdamped', 'critically_damped', 'oscillations'
formation_gradient   = 1   # followers are attracting to their formation position and repelling from obstacles
draw_gradients       = 1   # 1-gradients plot, 0-grid
postprocessing       = 0   # show processed data figures after the flight
""" human guided swarm params """
influence_radius     = 2      # potential fields obstacles influence radius
human_name           = 'palm' # vicon mocap object
pos_coef             = 5.0    # scale of the leader's movement relatively to the human operator
initialized          = False  # is always inits with False: for relative position control
max_its              = 200 # max number of allowed iters for formation to reach the goal
# movie writer
progress_bar = FillingCirclesBar('Number of Iterations', max=max_its)
should_write_movie = 0; movie_file_name = os.getcwd()+'/videos/output.avi'
movie_writer = get_movie_writer(should_write_movie, 'Simulation Potential Fields', movie_fps=10., plot_pause_len=0.01)

R_obstacles = 0.10 # [m]
R_drones    = 0.05 # [m]
l           = 0.3 # [m]
repel_robots = 1
start = np.array([0,0]) #start = np.array([-1.7, 1.7]);
goal = np.array([1.7, -1.7])
V0 = (goal - start) / norm(goal-start)    # initial movement direction, |V0| = 1
U0 = np.array([-V0[1], V0[0]]) / norm(V0) # perpendicular to initial movement direction, |U0|=1
imp_pose_prev = np.array([0, 0])
imp_vel_prev  = np.array([0, 0])
imp_time_prev = time.time()

if random_obstacles:
    obstacles_poses      = np.random.uniform(low=-2.5, high=2.5, size=(num_random_obstacles,2)) # randomly located obstacles
    obstacles_goal_poses = np.random.uniform(low=-1.3, high=1.3, size=(num_random_obstacles,2)) # randomly located obstacles goal poses
else:
    obstacles_poses      = np.array([[-1, 1], [1.0, 0.5], [-1.0, 0.5], [0.1, 0.1], [1, -0.3], [-0.8, -0.8]]) # 2D - coordinates [m]
    obstacles_goal_poses = np.array([[-0, 0], [0.0, 0.0], [ 0.0, 0.0], [0.0, 0.0], [0,  0], [ 0.0,  0.0]])

def human_pos_callback(data):
    global human_pose
    global human_yaw
    human_pose = np.array( [data.transform.translation.x, data.transform.translation.y, data.transform.translation.z] )
    q = np.array( [data.transform.rotation.x, data.transform.rotation.y, data.transform.rotation.z, data.transform.rotation.w] )
    human_yaw = euler_from_quaternion(q)[2]

rospy.init_node('gradient_interactive', anonymous=True)
pos_sub = rospy.Subscriber('/vicon/' + human_name + '/' + human_name, TransformStamped, human_pos_callback)
time.sleep(1)


""" Main loop """

# drones polygonal formation
route1 = start # leader
robots_poses = [start] + formation(num_robots, start, V0, l)
routes = [route1] + robots_poses[1:]
centroid_route = [ sum([p[0] for p in robots_poses])/len(robots_poses), sum([p[1] for p in robots_poses])/len(robots_poses) ]
des_poses = robots_poses
vels = [];
for r in range(num_robots): vels.append([])
norm_vels = [];
for r in range(num_robots): norm_vels.append([])

# variables for postprocessing and performance estimation
area_array = []
start_time = time.time()

fig = plt.figure(figsize=(12, 12))
with movie_writer.saving(fig, movie_file_name, max_its) if should_write_movie else get_dummy_context_mgr():
    for i in range(max_its):
        if moving_obstacles: obstacles_poses = move_obstacles(obstacles_poses, obstacles_goal_poses)

        """ Leader's pose update """
        # human palm pose and velocity using Vicon motion capture
        if not initialized:
            human_pose_init = human_pose[:2]
            drone1_pose_init = start
            initialized = True
        dx, dy = human_pose[:2] - human_pose_init
        des_poses[0] = np.array([  drone1_pose_init[0] + pos_coef*dx, drone1_pose_init[1] + pos_coef*dy ])
        vels[0] = pos_coef*velocity(human_pose)
        f1 = combined_potential(obstacles_poses, R_obstacles, des_poses[0], influence_radius=influence_radius)
        des_poses[0], _ = gradient_planner(f1, des_poses[0])
        direction = np.array([cos(human_yaw), sin(human_yaw)]) # rotation of the swarm relatively to human orientation
        norm_vels[0].append(norm(vels[0]))

        # drones polygonal formation
        # direction = ( goal - des_poses[0] ) / norm(goal - des_poses[0])
        des_poses[1:] = formation(num_robots, des_poses[0], direction, l)
        v = direction; u = np.array([-v[1], v[0]])

        if impedance:
            # drones positions are corrected according to the impedance model
            # based on leader's velocity
            imp_pose, imp_vel, imp_time_prev = velocity_imp(vels[0], imp_pose_prev, imp_vel_prev, imp_time_prev, mode=impedance_mode)
            imp_pose_prev = imp_pose
            imp_vel_prev = imp_vel

            imp_scale = 0.1
            # des_poses[0] += 0.1*imp_scale * imp_pose
            du = imp_scale*np.dot(imp_pose, u)/norm(u) # u-vector direction
            dv = imp_scale*np.dot(imp_pose, v)/norm(v) # v-vector direction
            if num_robots>=2:
                des_poses[1] +=  du * u + dv * v # impedance correction term is projected in u,v-vectors directions
            if num_robots>=3:
                des_poses[2] += -du * u + dv * v
            if num_robots>=4:
                des_poses[3] += -du * u + dv * v

        if formation_gradient:
            # following drones are attracting to desired points - vertices of the polygonal formation
            for p in range(num_robots):
                """ including another robots in formation in obstacles array: """
                if repel_robots:
                    robots_obstacles = [x for i,x in enumerate(robots_poses) if i!=p]
                    obstacles_poses1 = np.array(robots_obstacles + obstacles_poses.tolist())
                    f = combined_potential(obstacles_poses1, R_obstacles, des_poses[p], influence_radius=1.8)
                else:
                    f = combined_potential(obstacles_poses, R_obstacles, des_poses[p], influence_radius=influence_radius)
                des_poses[p], vels[p] = gradient_planner(f, des_poses[p])
                norm_vels[p].append(norm(vels[p]))

        for r in range(num_robots):
            routes[r] = np.vstack([routes[r], des_poses[r]])

        pp = des_poses
        centroid = [ sum([p[0] for p in pp])/len(pp), sum([p[1] for p in pp])/len(pp) ]
        centroid_route = np.vstack([centroid_route, centroid])
        dist_to_goal = norm(centroid - goal)
        if dist_to_goal < 1.4*l:
            print('\nReached the goal')
            break

        progress_bar.next()
        plt.cla()

        draw_map(start, goal, obstacles_poses, R_obstacles, f, draw_gradients=draw_gradients)
        draw_robots(des_poses[0], 5*R_drones, routes, num_robots, robots_poses, centroid, vels[0], plot_routes=False)
        if animate:
            plt.draw()
            plt.pause(0.01)

        if should_write_movie:
            movie_writer.grab_frame()
        # print('Current simulation time: ', time.time()-start_time)
    print('\nDone')
    progress_bar.finish()
    end_time = time.time()
    print('Simulation execution time: ', round(end_time-start_time,2))
    plt.show()

""" Flight data postprocessing """
if postprocessing:
    plt.figure()
    plt.title("Drones trajectories")
    plt.plot(centroid_route[:,0], centroid_route[:,1], label='centroid')
    d = 0
    for route in routes:
        d += 1
        plt.plot(route[:,0], route[:,1], '--', label='drone %d' %d)
    plt.legend()
    plt.grid()

    plt.figure()
    plt.title("Average velocity, <V>= %s m/s" %round(np.mean(np.array(norm_vels[0])), 2))
    for r in range(num_robots):
        plt.plot(norm_vels[r], label='drone %d' %(r+1))
    plt.xlabel('time')
    plt.ylabel('velocity, [m/s]')
    plt.legend()
    plt.grid()

    for i in range(len(routes[0])):
        X = np.array([]); Y = np.array([])
        for r in range(num_robots):
            X = np.append( X, routes[r][i,0] )
            Y = np.append( Y, routes[r][i,1] )
        area_array.append(poly_area(X,Y))

    plt.figure()
    plt.title("Area of robots' formation")
    plt.plot(area_array)
    plt.xlabel('time')
    plt.ylabel('Formation area, [m^2]')
    plt.grid()
    # close windows if Enter-button is pressed
    plt.draw()
    plt.pause(1)
    raw_input('Hit Enter to close')
    plt.close('all')

# TODO:
# 1. local minimum problem (FM2 - algorithm: https://pythonhosted.org/scikit-fmm/)
# 2. impedance controlled shape of the formation: area(velocity)
# 3. postprocessing: trajectories smoothness, etc. compare imp modeles:
#     - oscillation, underdamped, critically damped, overdamped
#     - velocity plot for all drones, acc, jerk ?
# 4. another drones are obstacles for each individual drone (done, however attractive and repelling forces should be adjusted)
# 5. import swarmlib (OOP) and test flight
# 6. add borders: see image processing (mirrow or black)
