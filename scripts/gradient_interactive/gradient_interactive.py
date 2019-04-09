#!/usr/bin/env python

# TODO:
# 2. impedance controlled shape of the formation: area(velocity)
# 3. postprocessing: trajectories smoothness, etc. compare imp modeles:
#     - oscillation, underdamped, critically damped, overdamped
#     - velocity plot for all drones, acc, jerk ?
# 4. another drones are obstacles for each individual drone (done, however attractive and repelling forces should be adjusted)
# 6. add borders: see image processing (mirrow or black) for mean velocity calculation


import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib import collections
from scipy.ndimage.morphology import distance_transform_edt as bwdist
from math import *
import random
from potential_fields import gradient_planner, combined_potential
from impedance.impedance_modeles import *
import time

from progress.bar import FillingCirclesBar
from tasks import *
from threading import Thread
from multiprocessing import Process
import os

""" ROS """
import rospy
from geometry_msgs.msg import TransformStamped
import swarmlib
import crazyflie


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
        dx, dy = (goal - pose) / norm(goal-pose) * 0.01 #random.uniform(0,0.05)
        pose[0] += dx;      pose[1] += dy;

    return obstacles_poses

""" initialization """
random_obstacles     = 0   # randomly distributed obstacles on the map
num_random_obstacles = 4   # number of random circular obstacles on the map
moving_obstacles     = 0   # 0-static or 1-dynamic obstacles
impedance            = 0   # impedance links between the leader and followers (leader's velocity)
formation_gradient   = 1   # followers are attracting to their formation position and repelling from obstacles
draw_gradients       = 1

""" human guided swarm params """
human_name           = 'glove' # Vicon mocap object, e.g: 'palm', 'glove'
pos_coef             = 1.6    # scale of the leader's movement relatively to the human operator
initialized          = False  # is always inits with False: for relative position control
repel_robots         = 0      # robots repel from each other inside the formation
influence_radius     = 15     # repelling strength

R_obstacles = 0.2 # [m]
l           = 0.35  # [m]
start = np.array([-1.8, 1.8]); goal = np.array([1.2, -1.2])
V0 = (goal - start) / norm(goal-start)    # initial movement direction, |V0| = 1
U0 = np.array([-V0[1], V0[0]]) / norm(V0) # perpendicular to initial movement direction, |U0| = 1
imp_pose_prev = np.array([0, 0])
imp_vel_prev  = np.array([0, 0])
imp_time_prev = time.time()

# flight parameters
toFly                = 1   # 0-simulation, 1-real drones
TakeoffHeight        = 1.2
TimeToTakeoff        = 7.0
HUMAN_Z_TO_LAND      = 0.8
position_initialized = False # should be False at the begininng
limits           = np.array([ 1.7,  1.7,  2.5]) # limits desining safety flight area in the room
limits_negative  = np.array([-1.7, -1.5, -0.1])

cf_names = ['cf1', 'cf2', 'cf3', 'cf4']
# cf_names = ['cf1', 'cf2', 'cf3']
num_robots = len(cf_names)   # <=4, number of drones in formation
if num_robots > 4: num_robots = 4


obstacles_names = ['obstacle4', 'obstacle11', 'obstacle12'] # Vicon objects

if random_obstacles:
    obstacles_poses      = np.random.uniform(low=-2.5, high=2.5, size=(num_random_obstacles,2)) # randomly located obstacles
    obstacles_goal_poses = np.random.uniform(low=-1.3, high=1.3, size=(num_random_obstacles,2)) # randomly located obstacles goal poses: for moving obstacles
else:
    obstacles_poses      = np.array([[-1, 1], [1.0, 0.5], [-1.0, 0.3], [0.1, 0.1], [1, -0.3], [-0.8, -0.8]]) # 2D - coordinates [m]
    obstacles_goal_poses = np.array([[-0, 0], [0.0, 0.0], [ 0.0, 0.0], [0.0, 0.0], [0,  0], [ 0.0,  0.0]]) # for moving obstacles


if __name__ == '__main__':
    rospy.init_node('gradient_interactive', anonymous=True)
    # Objects inititalization
    human = swarmlib.Mocap_object(human_name)
    swarm = []
    for name in cf_names:
        swarm.append(swarmlib.Drone(name))
    swarm[0].leader=True

    # Obstacles init
    obstacles_poses = []
    for name in obstacles_names:
        obstacles_poses.append(swarmlib.Mocap_object(name).position()[:2])
    obstacles_poses = np.array(obstacles_poses)
    obstacles_array = []
    for ind in range(len(obstacles_poses)):
        obstacles_array.append( swarmlib.Obstacle('obstacle_%d' %ind) )
        obstacles_array[-1].pose = obstacles_poses[ind].tolist()+[TakeoffHeight]
        obstacles_array[-1].R = R_obstacles
        ind += 1

    if toFly:
        cf_list = []
        for name in cf_names:
            cf = crazyflie.Crazyflie(name, '/vicon/'+name+'/'+name)
            cf.setParam("commander/enHighLevel", 1)
            cf.setParam("stabilizer/estimator",  2) # Use EKF
            cf.setParam("stabilizer/controller", 2) # Use mellinger controller
            cf_list.append(cf)
        print "takeoff"
        for cf in cf_list:
            for t in range(3): cf.takeoff(targetHeight=TakeoffHeight, duration=TimeToTakeoff)
        time.sleep(TimeToTakeoff) # time to takeoff and select position for human

    """ Main loop """
    rate = rospy.Rate(200)
    while not rospy.is_shutdown():
        for drone in swarm: drone.pose = drone.position()
        human.pose = human.position(); human.orient = human.orientation()
        if moving_obstacles: obstacles_poses = move_obstacles(obstacles_poses, obstacles_goal_poses)
        for i in range(len(obstacles_array)):
            obstacles_array[i].pose[:2] = obstacles_poses[i]

        if not position_initialized:
            print "Human position is initialized"
            human_pose_init = human.position()
            drone1_pose_init = swarm[0].position()
            position_initialized = True
        dx, dy = (human.position() - human_pose_init)[:2]
        swarm[0].sp = np.array([  drone1_pose_init[0] + pos_coef*dx,
                                drone1_pose_init[1] + pos_coef*dy,
                                TakeoffHeight]) # from 2D to 3D, adding z=TakeoffHeight for the leader drone
        f1 = combined_potential(obstacles_poses, R_obstacles, swarm[0].sp[:2], influence_radius=influence_radius)
        swarm[0].sp[:2], swarm[0].vel[:2] = gradient_planner(f1, swarm[0].sp[:2])

        # limit the leaders position to remain within specified area
        if toFly:
            np.putmask(swarm[0].sp, swarm[0].sp >= limits, limits)
            np.putmask(swarm[0].sp, swarm[0].sp <= limits_negative, limits_negative)

        # drones polygonal formation
        direction = np.array([cos(human.orient[2]), sin(human.orient[2])])
        v = direction; u = np.array([-v[1], v[0]]) # u is prepedicular vector to v-human yaw direction
        followers_sp = formation(num_robots, swarm[0].sp[:2], direction, l)
        for i in range(len(followers_sp)):
            swarm[i+1].sp = followers_sp[i].tolist() + [TakeoffHeight] # from 2D to 3D, adding z=TakeoffHeight for each follower

        if impedance:
            # drones positions are corrected according to the impedance model
            # based on human velocity
            human.vel = velocity(human.pose)
            imp_pose, imp_vel, imp_time_prev = velocity_imp(human.vel, imp_pose_prev, imp_vel_prev, imp_time_prev, mode='critically_damped')
            imp_pose_prev = imp_pose
            imp_vel_prev = imp_vel

            imp_scale = 0.1
            swarm[0].sp[:2] += imp_scale * imp_pose
            du = imp_scale*np.dot(imp_pose, u)/norm(u) # u-vector direction
            dv = imp_scale*np.dot(imp_pose, v)/norm(v) # v-vector direction
            if num_robots>=2:
                swarm[0].sp[:2] +=  du * u + dv * v # impedance correction term is projected in u,v-vectors directions
            if num_robots>=3:
                swarm[0].sp[:2] += -du * u + dv * v
            if num_robots>=4:
                swarm[0].sp[:2] += -du * u + dv * v

        robots_poses = []
        for drone in swarm:
            robots_poses.append(drone.pose[:2])
        if formation_gradient:
            # following drones are attracting to desired points - vertices of the polygonal formation
            for p in range(num_robots):
                if repel_robots:
                    robots_obstacles = [x for i,x in enumerate(robots_poses) if i!=p]
                    obstacles_and_another_drones_poses = np.array(robots_obstacles + obstacles_poses.tolist())
                    f = combined_potential(obstacles_and_another_drones_poses, R_obstacles, swarm[p].sp[:2], influence_radius=3)
                else:
                    f = combined_potential(obstacles_poses, R_obstacles, swarm[p].sp[:2], influence_radius=influence_radius)
                swarm[p].sp[:2], swarm[p].vel[:2] = gradient_planner(f, swarm[p].sp[:2])


        # TO FLY
        if toFly:
            for drone in swarm: drone.fly()

        # TO VISUALIZE
        human.publish_position()
        human.publish_path()
        path_limit = 1000 # [frames]
        for drone in swarm:
            drone.publish_sp()
            drone.publish_path_sp(limit=path_limit) # set -1 for unlimited path
        for obstacle in obstacles_array:
            obstacle.publish_position()

        # Landing
        if toFly and human.position()[2]<HUMAN_Z_TO_LAND:
            print 'Landing!!!'
            for drone in swarm:
                drone.landing_pose = drone.position()
            while not rospy.is_shutdown():
                for drone in swarm:
                    drone.sp = drone.landing_pose
                    drone.landing_pose[2] -= 0.002
                    drone.fly()
                if swarm[0].sp[2]<-1:
                    time.sleep(1)
                    for cf in cf_list:
                        for t in range(3): cf.stop()
                    print 'reached the floor, shutdown'
                    rospy.signal_shutdown('landed')
                rate.sleep()

        rate.sleep()



