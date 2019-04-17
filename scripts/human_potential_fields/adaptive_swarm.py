#!/usr/bin/env python

import numpy as np
from numpy.linalg import norm
import time
import swarmlib
from swarmlib import Drone, Mocap_object
import rospy
import math

import crazyflie
from crazyflie_driver.msg import FullState
from crazyflie_driver.msg import Position

import matplotlib.pyplot as plt
from potential_fields import *
from tools import *

def publish_goal_pos(cf_goal_pos, cf_goal_yaw, cf_name):
    name = cf_name + "/cmd_position"
    msg = msg_def_crazyflie(cf_goal_pos, cf_goal_yaw)
    pub = rospy.Publisher(name, Position, queue_size=1)
    pub.publish(msg)

def msg_def_crazyflie(pose, yaw):
    worldFrame = rospy.get_param("~worldFrame", "/world")
    msg = Position()
    msg.header.seq = 0
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = worldFrame
    msg.x = pose[0]
    msg.y = pose[1]
    msg.z = pose[2]
    msg.yaw = yaw
    now = rospy.get_time()
    msg.header.seq = 0
    msg.header.stamp = rospy.Time.now()
    return msg

def hover(t=5):
    print "Hovering...\n"
    while not rospy.is_shutdown():
        for i in range(int(t*100 / len(drones))):
            for drone in drones:
                if toFly: drone.fly()
                drone.publish_sp()
                time.sleep(0.01)
        break

def landing():
    print 'Landing!!!'
    for drone in drones: drone.sp = drone.position()
    while(1):
        for drone in drones:
            drone.sp[2] -= 0.002
            drone.fly()
            time.sleep(0.01)

        if drones[0].sp[2]<-0.2:
            print 'reached the floor'
            time.sleep(0.1)
            if toFly:
                for cf in cf_list: cf.stop()
            break


class Robot(Drone):
    def __init__(self, name):
        Drone.__init__(self, name)
        self.initial_pose = self.position()
        self.route = np.array([self.initial_pose])
        self.f = 0
        self.leader = False

    def local_planner(self, obstacles_poses, params):
        self.f = combined_potential(obstacles_poses, params.R_obstacles, self.sp[:2], params.obstacles_influence_radius)
        self.sp[:2] = gradient_planner(self.sp[:2], self.f)
        # self.route = np.vstack( [self.route, self.sp] )

class Params:
    def __init__(self,):
        self.R_obstacles = 0.10 # [m]
        self.obstacles_influence_radius = 1.4
        self.l = 0.5 # [m], inter-drones distance

rospy.init_node('adaptive_swarm', anonymous=False)

toFly          = 0
TAKEOFFHEIGHT  = 0.8
TAKEOFFTIME    = 5.0
LANDTIME       = 2.0
initialized    = False
position_control = 1
pos_coef       = 3.0
velocity_control = not position_control
vel_coef       = 1.0
put_limits       = 1
limits           = np.array([ 2.0, 2.0, 2.5 ]) # limits desining safety flight area in the room
limits_negative  = np.array([ -2.0, -2.0, -0.1 ])
# limits           = np.array([ 1.7, 1.7, 2.5 ]) # limits desining safety flight area in the room
# limits_negative  = np.array([-1.7, -1.7, -0.1 ])
repel_robots   = 0
keep_formation = 1
collision_avoidance = 1

params = Params()

# joystick
human = Mocap_object("palm")

# drones-followers
cf_names = ['cf1', 'cf2', 'cf3']
# cf_names = ['cf1', 'cf2']
# cf_names = ['cf2']


drones = []
drones_poses = []
for name in cf_names:
    drone = Robot(name)
    drone.sp = drone.position()
    drones.append( drone )
    drones_poses.append(drone.position())

obstacles = []
obstacles_poses = []
# obstacles_names = ['obstacle4', 'obstacle10', 'obstacle12', 'obstacle25']
# obstacles_names = ['obstacle4', 'obstacle25']
obstacles_names = ['obstacle25']
for name in obstacles_names:
    obstacle = swarmlib.Obstacle(name)
    obstacles.append( obstacle )
    obstacles_poses.append(obstacle.position()[:2])
    

if __name__ == "__main__":
    if toFly:
        cf_list = []
        for cf_name in cf_names:
            # print "adding.. ", cf_name
            cf = crazyflie.Crazyflie(cf_name, '/vicon/'+cf_name+'/'+cf_name)
            cf.setParam("commander/enHighLevel", 1)
            cf.setParam("stabilizer/estimator",  2) # Use EKF
            cf.setParam("stabilizer/controller", 2) # Use mellinger controller
            cf_list.append(cf)
        for t in range(3):
            # print "takeoff.. ", cf.prefix
            for cf in cf_list:
                cf.takeoff(targetHeight = TAKEOFFHEIGHT, duration = TAKEOFFTIME)
        time.sleep(TAKEOFFTIME+2.0)
        # go to approximate human location
        # for t in range(3):
        # for cf in cf_list:
        #     cf.goTo(goal = [0.5, 0.5, 0], yaw=0.0, duration=3.0, relative = True)
        # time.sleep(3.5)
        # for t in range(3):
        #     for cf in cf_list: cf.land(targetHeight = -0.05, duration = 3.0)

    print 'start DroneStick \n'
    plt.figure(figsize=(8,8))
    while not rospy.is_shutdown():
        human.orient = human.orientation()
        human.position()
        for drone in drones: drone.pose = drone.position()

        # update obstacles poses
        for i in range(len(obstacles)):
            obstacles_poses[i] = obstacles[i].position()[:2]

        # update drones poses
        for i in range(len(drones)):
            drones_poses[i] = drones[i].sp[:2]

        human.vel = swarmlib.velocity(human.pose)
        # POSITION CONTROL
        if position_control:
        #     if not initialized:
        #         t_prev = time.time()
        #         human_pose_init = human.position()
        #         drone1_pose_init = drones[0].sp
        #         initialized = True
        #     t = time.time()
        #     if t - t_prev > 0.01: initialized = False
        #     dx, dy = human.position()[:2] - human_pose_init[:2]
        #     for drone in drones:
        #         drone.sp += np.array([  pos_coef*dx, pos_coef*dy, 0 ])
        #         drone.sp[2] = TAKEOFFHEIGHT
            for drone in drones:
                # print norm(human.vel)
                if norm(human.vel) > 0.0001: drone.sp += pos_coef * human.vel * 0.1
                drone.sp[2] = TAKEOFFHEIGHT

        # VELOCITY CONTROL
        elif velocity_control:
            if not initialized:
                for drone in drones: drone.sp = np.array( [drone.pose[0], drone.pose[1], TAKEOFFHEIGHT] )
                human_pose_init = human.position()
                time_prev = time.time()
                initialized = True
            dx, dy = human.position()[:2] - human_pose_init[:2]

            cmd_vel = vel_coef*(np.array([dx, dy, 0]))
            time_now = time.time()
            for drone in drones: drone.sp += cmd_vel*(time.time()-time_prev)
            time_prev = time_now

        # drones formation:
        if keep_formation and len(drones)>1:
            if len(drones) >= 2: drones[1].sp = drones[0].sp + (drones[1].initial_pose - drones[0].initial_pose)
            if len(drones) >= 3: drones[2].sp = drones[0].sp + (drones[2].initial_pose - drones[0].initial_pose)
            # drones[1].sp = drones[0].sp + np.array([-0.86*params.l , params.l/2., 0])
            # drones[2].sp = drones[0].sp + np.array([-0.86*params.l ,-params.l/2., 0])

        # correct point to follow with local planner
        if collision_avoidance:
            for p in range(len(drones)):
                if repel_robots:
                    robots_obstacles = [x for i,x in enumerate(drones_poses) if i!=p] # all poses except the robot[p]
                    obstacles_poses1 = obstacles_poses + robots_obstacles
                    drones[p].local_planner(obstacles_poses1, params)
                else:
                    drones[p].local_planner(obstacles_poses, params)

        if put_limits:
            for drone in drones:
                np.putmask(drone.sp, drone.sp >= limits, limits)
                np.putmask(drone.sp, drone.sp <= limits_negative, limits_negative)

        if toFly:
            for drone in drones: drone.fly()

        # visualization: RVIZ
        for drone in drones:
            drone.publish_sp()
            drone.publish_path_sp()
        for obstacle in obstacles: obstacle.publish_position()

        # visualization: matplotlib
        for drone in drones: drone.route = np.vstack( [drone.route, drone.sp] )
        plt.cla()
        plt.plot(human.pose[0], human.pose[1], 'ro', label='human pose')
        if collision_avoidance: draw_gradient(drones[0].f)
        draw_map(obstacles_poses, params.R_obstacles)
        for drone in drones:
            plt.plot(drone.sp[0], drone.sp[1], '^', markersize=10, label=drone.name)
            plt.plot(drone.route[:,0], drone.route[:,1], color='green')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.draw()
        plt.pause(0.01)


        # RETURN JOYSTICK TO THE SWARM
        Z = 0.8
        # if human.pose[2] > Z + 0.07:
        if (Z - human.pose[2]) > 0.10 and toFly:
            print 'Preparing to land...'
            # hover(t=1)

            landing()

            break


