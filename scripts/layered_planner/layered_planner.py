#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt

from tools import *
from rrt import *
from potential_fields import *

# for crazyflies
import crazyflie
import rospy
from swarmlib import Drone, Obstacle
import time


def pose2square(pose, l=0.2):
    pose  = np.array(pose)
    return np.array([pose[:2] + [l/2, l/2], pose[:2] + [-l/2, l/2], pose[:2] + [-l/2, -l/2], pose[:2] + [l/2, -l/2]])

def move_obstacles(obstacles):
    # obstacles[3] += np.array([0.004, 0.005])
    # small cubes movement
    obstacles[-3] += np.array([0.02, 0.0])
    obstacles[-2] += np.array([-0.006, 0.006])
    obstacles[-1] += np.array([0.0, 0.01])
    return obstacles

def landing():
    print 'Landing!!!'
    for robot in robots: robot.sp = robot.position()
    while(1):
        for robot in robots:
            robot.sp[2] -= 0.004
            if params.toFly: robot.fly()
            robot.publish_sp()
            robot.publish_path_sp()
            time.sleep(0.01)

        if robots[0].sp[2]<-0.2:
            print 'Reached the floor'
            time.sleep(0.1)
            if params.toFly:
                for cf in cf_list: cf.stop()
            break


class Params:
    def __init__(self):
        self.toFly = 0 # 1 - real drones flight and simulation, 0 - only simulation
        self.animate = 0 # show RRT construction, set 0 to reduce time of the RRT algorithm
        self.visualize = 1 # show constructed paths at the end of the RRT and path smoothing algorithms
        self.maxiters = 5000 # max number of samples to build the RRT
        self.goal_prob = 0.05 # with probability goal_prob, sample the goal
        self.minDistGoal = 0.25 # [m], min distance os samples from goal to add goal node to the RRT
        self.extension = 0.4 # [m], extension parameter: this controls how far the RRT extends in each step.
        self.world_bounds_x = [-2.5, 2.5] # [m], map size in X-direction
        self.world_bounds_y = [-2.5, 2.5] # [m], map size in Y-direction
        self.drone_vel = 3.0 # [m/s]
        self.ViconRate = 100 # [Hz]
        self.max_sp_dist = 0.15 * self.drone_vel # [m], maximum distance between current robot's pose and the sp from global planner
        self.influence_radius = 1.23 # potential fields radius, defining repulsive area size near the obstacle
        self.goal_tolerance = 0.05 # [m], maximum distance threshold to reach the goal
        self.cf_names = ['cf1', 'cf2', 'cf3']
        # self.cf_names = ['cf1', 'cf2']
        self.num_robots = len(self.cf_names)
        self.TakeoffHeight = 0.8 # [m]
        self.length_moving_obstacles = 0.2 # [m], size of Vicon objects: moving cubes, other drones
        self.reached_goal = 0
        self.l_drones = 0.3 # [m], distance between the drones in the formation

class Robot(Drone):
    def __init__(self, name):
        Drone.__init__(self, name)
        self.sp_global_planner = np.array([0,0,0])
        self.route = np.array([self.sp])
        self.f = 0
        self.leader = False

    def local_planner(self, obstacles, params):
        obstacles_grid = grid_map(obstacles)
        self.f = combined_potential(obstacles_grid, self.sp_global_planner[:2], params.influence_radius)
        self.sp[:2] = gradient_planner_next(self.sp[:2], self.f, params)
        self.route = np.vstack( [self.route, self.sp] )

# Initialization
rospy.init_node('adaptive_swarm', anonymous=False)

params = Params()

# Obstacles map construction
# obstacles = [
#               # bugtrap
#               np.array([[0.5, 0], [2.5, 0.], [2.5, 0.3], [0.5, 0.3]]),
#               np.array([[0.5, 0.3], [0.8, 0.3], [0.8, 1.5], [0.5, 1.5]]),
#               np.array([[0.5, 1.5], [1.5, 1.5], [1.5, 1.8], [0.5, 1.8]]),
#               # angle
#               np.array([[-2, -2], [-0.5, -2], [-0.5, -1.8], [-2, -1.8]]),
#               np.array([[-0.7, -1.8], [-0.5, -1.8], [-0.5, -0.8], [-0.7, -0.8]]),
#               # walls
#               np.array([[-2.5, -2.5], [2.5, -2.5], [2.5, -2.49], [-2.5, -2.49]]),
#               np.array([[-2.5, 2.49], [2.5, 2.49], [2.5, 2.5], [-2.5, 2.5]]),
#               np.array([[-2.5, -2.49], [-2.49, -2.49], [-2.49, 2.49], [-2.5, 2.49]]),
#               np.array([[2.49, -2.49], [2.5, -2.49], [2.5, 2.49], [2.49, 2.49]]),

#               # moving obstacle
#               np.array([[-2.3, 2.0], [-2.2, 2.0], [-2.2, 2.1], [-2.3, 2.1]]),
#               np.array([[2.3, -2.3], [2.4, -2.3], [2.4, -2.2], [2.3, -2.2]]),
#               np.array([[0.0, -2.3], [0.1, -2.3], [0.1, -2.2], [0.0, -2.2]]),
#             ]

""" Room environment """
# obstacles = [
              # # np.array([[-1.0, 2.0], [0.5, 2.0], [0.5, 2.5], [-1.0, 2.5]]), # my table
              # np.array([[-1.0, 2.0], [0.5, 2.0], [0.5, 2.5], [-1.0, 2.5]]) + np.array([2.0, 0]), # Evgeny's table
              # np.array([[-2.0, -0.5], [-2.0, 1.0], [-2.5, 1.0], [-2.5, -0.5]]), # Roman's table
              # np.array([[-1.2, -1.2], [-1.2, -2.5], [-2.5, -2.5], [-2.5, -1.2]]), # mats
              # np.array([[2.0, 0.8], [2.0, -0.8], [2.5, -0.8], [2.5, 0.8]]), # Mocap table
    
#               # bugtrap
#               # np.array([[0.7, -0.9], [1.3, -0.9], [1.3, -0.8], [0.7, -0.8]]),
#               # np.array([[0.7, -0.9], [1.3, -0.9], [1.3, -0.8], [0.7, -0.8]]) + np.array([0.0, 0.5]),
#               # np.array([[0.7, -0.9], [0.8, -0.9], [0.8, -0.3], [0.7, -0.3]]),        
#             ]

"""" Narrow passage """
passage_width = 0.3
passage_location = 0.0
obstacles = [
            # narrow passage
              np.array([[-2.5, -0.5], [-passage_location-passage_width/2., -0.5], [-passage_location-passage_width/2., 0.5], [-2.5, 0.5]]),
              np.array([[-passage_location+passage_width/2., -0.5], [2.5, -0.5], [2.5, 0.5], [-passage_location+passage_width/2., 0.5]]),
            ]

moving_obstacles_start_index = len(obstacles)
# moving_obstacles_names = ['obstacle25']
moving_obstacles_names = []

moving_obstacles = []
l = params.length_moving_obstacles
for name in moving_obstacles_names:
    obstacle = Obstacle(name)
    moving_obstacles.append(obstacle)
    obstacles.append( pose2square(obstacle.position(), l) )

# Robots initialization
robots = []
for name in params.cf_names:
    robot = Robot(name)
    robots.append( robot )
robot1 = robots[0]; robot1.leader=True

xy_start = robots[0].position()[:2] # np.array([1.2, 1.0])
# xy_goal =  np.array([-1.0, 1]) # np.array([1.5, -1.4])
xy_goal =  np.array([1.0, -1])

# Layered Motion Planning: RRT (global) + Potential Field (local)
if __name__ == '__main__':
    plt.figure(figsize=(10,10))
    draw_map(obstacles)
    plt.plot(xy_start[0], xy_start[1],'bo',color='red', markersize=20, label='start')
    plt.plot(xy_goal[0], xy_goal[1],'bo',color='green', markersize=20, label='goal')

    """ RRT path construction """
    P_long = rrt_path(obstacles, xy_start, xy_goal, params)
    P = ShortenPath(P_long, obstacles, smoothiters=30) # P = [[xN, yN], ..., [x1, y1], [x0, y0]]

    traj_global = waypts2setpts(P, params); P = np.vstack([P, xy_start])
    plt.plot(P[:,0], P[:,1], linewidth=3, color='orange', label='Global planner path')
    plt.pause(0.1)

    sp_ind = 0
    robot1.route = np.array([[traj_global[0,0], traj_global[0,1], params.TakeoffHeight]])
    robot1.sp = robot1.route[-1,:]

    followers_sp = formation(params.num_robots, leader_des=robot1.sp[:2], v=np.array([1, 0]), l=params.l_drones)
    for i in range(len(followers_sp)):
        robots[i+1].sp = followers_sp[i].tolist() + [params.TakeoffHeight]
        robots[i+1].route = np.array(robots[i+1].sp)

    """ Takeoff """
    if params.toFly:
        cf_list = []
        for cf_name in params.cf_names:
            # print "adding.. ", cf_name
            cf = crazyflie.Crazyflie(cf_name, '/vicon/'+cf_name+'/'+cf_name)
            cf.setParam("commander/enHighLevel", 1)
            cf.setParam("stabilizer/estimator",  2) # Use EKF
            cf.setParam("stabilizer/controller", 2) # Use mellinger controller
            cf_list.append(cf)
        for t in range(3):
            # print "takeoff.. ", cf.prefix
            for cf in cf_list:
                cf.takeoff(targetHeight = params.TakeoffHeight, duration = 4*params.TakeoffHeight)
        time.sleep(4*params.TakeoffHeight)

    t_goal = -1
    while not rospy.is_shutdown(): # loop through all the setpoint from global planner trajectory, traj_global
        dist_to_goal = norm(robot1.sp[:2] - xy_goal)
        if dist_to_goal < params.goal_tolerance and not params.reached_goal: # [m]
            print 'Goal is reached'
            t_goal = time.time()
            params.reached_goal = 1
        # wait for drones-followers to reach their predefined positions
        t_current = time.time()
        if t_current - t_goal > 3.0 and t_goal > 0:
            print 'Ready to land'
            break
        # obstacles = move_obstacles(obstacles) # change poses of some obstacles on the map
        for i in range( len(moving_obstacles) ):
            obstacles[moving_obstacles_start_index + i] = pose2square(moving_obstacles[i].position()[:2], l)

        # leader's setpoint from global planner
        robot1.sp_global_planner = np.array(traj_global[sp_ind,:].tolist() + [params.TakeoffHeight])
        # correct leader's pose with local planner

        robot1.local_planner(obstacles, params)

        """ adding following robots in the swarm """
        # formation poses from global planner
        direction = normalize(robot1.sp_global_planner[:2]-robot1.sp[:2])
        followers_sp_global_planner = formation(params.num_robots, robot1.sp_global_planner[:2], v=direction, l=params.l_drones)
        for i in range(len(followers_sp_global_planner)):
            robots[i+1].sp_global_planner = followers_sp_global_planner[i].tolist() + [params.TakeoffHeight]
        for p in range(len(followers_sp)): # formation poses correction with local planner
            # robots repel from each other inside the formation
            robots_obstacles_sp = [x for i,x in enumerate(followers_sp + [robot1.sp[:2]]) if i!=p] # all poses except the robot[p]
            robots_obstacles = poses2polygons( robots_obstacles_sp ) # each drone is defined as a small cube for inter-robots collision avoidance
            obstacles1 = obstacles + robots_obstacles # combine exisiting obstacles on the map with other robots[for each i: i!=p] in formation
            # follower robot's position correction with local planner
            robots[p+1].local_planner(obstacles1, params)
            followers_sp[p] = robots[p+1].sp[:2]

        if params.toFly:
            for robot in robots: robot.fly()

        # visualization: RVIZ
        for robot in robots:
            robot.publish_sp()
            robot.publish_path_sp()

        # visualization
        plt.cla()
        draw_map(obstacles)
        if params.num_robots == 1:
            draw_gradient(robots[0].f)
        else:
            draw_gradient(robots[1].f)
        for robot in robots: plt.plot(robot.sp[0], robot.sp[1], '^', color='blue', markersize=10, zorder=15) # robots poses
        plt.plot(robot1.route[:,0], robot1.route[:,1], linewidth=2, color='green', label="Robot's path, corrected with local planner", zorder=10)
        for robot in robots[1:]: plt.plot(robot.route[:,0], robot.route[:,1], '--', linewidth=2, color='green', zorder=10)
        plt.plot(P[:,0], P[:,1], linewidth=3, color='orange', label='Global planner path')
        plt.plot(traj_global[sp_ind,0], traj_global[sp_ind,1], 'ro', color='blue', markersize=7, label='Global planner setpoint')
        plt.plot(xy_start[0],xy_start[1],'bo',color='red', markersize=20, label='start')
        plt.plot(xy_goal[0], xy_goal[1],'bo',color='green', markersize=20, label='goal')
        plt.legend()
        plt.draw()
        plt.pause(0.01)

        # update loop variable
        if sp_ind < traj_global.shape[0]-1 and norm(robot1.sp_global_planner - robot1.sp) < params.max_sp_dist: sp_ind += 1

    landing()

    # close windows if Enter-button is pressed
    plt.draw()
    plt.pause(0.1)
    raw_input('Hit Enter to close')
    plt.close('all')