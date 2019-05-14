#!/usr/bin/env python

"""
Autonumous navigation of robots formation with Layered path-planner:
- global planner: RRT
- local planner: Artificial Potential Fields
"""

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from tools import *
from rrt import *
from potential_fields import *
from progress.bar import FillingCirclesBar
import time


def move_obstacles(obstacles, params):
    # obstacles[3] += np.array([0.004, 0.005])
    # small cubes movement
    obstacles[-3] += np.array([0.02, 0.0]) * params.drone_vel
    obstacles[-2] += np.array([-0.006, 0.006]) * params.drone_vel
    obstacles[-1] += np.array([0.0, 0.01]) * params.drone_vel
    return obstacles

class Params:
    def __init__(self):
        self.animate = 1 # show RRT construction, set 0 to reduce time of the RRT algorithm
        self.visualize = 0 # show constructed paths at the end of the RRT and path smoothing algorithms
        self.postprocessing = 1 # process and visualize the simulated experiment data after the simulation
        self.maxiters = 500 # max number of samples to build the RRT
        # self.progress_bar = FillingCirclesBar('Number of Iterations', max=self.maxiters)
        self.goal_prob = 0.05 # with probability goal_prob, sample the goal
        self.minDistGoal = 0.25 # [m], min distance os samples from goal to add goal node to the RRT
        self.extension = 0.8 # [m], extension parameter: this controls how far the RRT extends in each step.
        self.world_bounds_x = [-2.5, 2.5] # [m], map size in X-direction
        self.world_bounds_y = [-2.5, 2.5] # [m], map size in Y-direction
        self.drone_vel = 5.0 # [m/s]
        self.ViconRate = 100 # [Hz]
        self.max_sp_dist = 0.15 * self.drone_vel # [m], maximum distance between current robot's pose and the sp from global planner
        self.influence_radius = 1.3 # potential fields radius, defining repulsive area size near the obstacle
        self.goal_tolerance = 0.05 # [m], maximum distance threshold to reach the goal
        self.num_robots = 4

class Robot:
    def __init__(self):
        self.sp = [0,0]
        self.sp_global = [0,0]
        self.route = np.array([self.sp])
        self.vel_array = []
        self.f = 0
        self.leader = False

    def local_planner(self, obstacles, params):
        """
        This function computes the next_point
        given current location (self.sp) and potential filed function, f.
        It also computes mean velocity, V, of the gradient map in current point.
        """
        obstacles_grid = grid_map(obstacles)
        self.f = combined_potential(obstacles_grid, self.sp_global, params.influence_radius)
        [gy, gx] = np.gradient(-self.f)
        iy, ix = np.array( meters2grid(self.sp), dtype=int )
        w = 20 # smoothing window size for gradient-velocity
        vx = np.mean(gx[ix-int(w/2) : ix+int(w/2), iy-int(w/2) : iy+int(w/2)])
        vy = np.mean(gy[ix-int(w/2) : ix+int(w/2), iy-int(w/2) : iy+int(w/2)])
        self.V = params.drone_vel * np.array([vx, vy])
        self.vel_array.append(norm(self.V))
        dt = 0.01 * params.drone_vel / norm([vx, vy])
        # self.sp += dt**2/2.*np.array( [vx, vy] )
        self.sp += dt*np.array( [vx, vy] )
        self.route = np.vstack( [self.route, self.sp] )

# Initialization
params = Params()
xy_start = np.array([1.2, 1.0])
xy_goal =  np.array([1.5, -1.4])
# xy_goal =  np.array([1.5, 1.0])

# Obstacles map construction
obstacles = [
              # bugtrap
              np.array([[0.5, 0], [2.5, 0.], [2.5, 0.3], [0.5, 0.3]]),
              np.array([[0.5, 0.3], [0.8, 0.3], [0.8, 1.5], [0.5, 1.5]]),
              np.array([[0.5, 1.5], [1.5, 1.5], [1.5, 1.8], [0.5, 1.8]]),
              # angle
              np.array([[-2, -2], [-0.5, -2], [-0.5, -1.8], [-2, -1.8]]),
              np.array([[-0.7, -1.8], [-0.5, -1.8], [-0.5, -0.8], [-0.7, -0.8]]),
              # walls
              np.array([[-2.5, -2.5], [2.5, -2.5], [2.5, -2.49], [-2.5, -2.49]]),
              np.array([[-2.5, 2.49], [2.5, 2.49], [2.5, 2.5], [-2.5, 2.5]]),
              np.array([[-2.5, -2.49], [-2.49, -2.49], [-2.49, 2.49], [-2.5, 2.49]]),
              np.array([[2.49, -2.49], [2.5, -2.49], [2.5, 2.49], [2.49, 2.49]]),

              # moving obstacle
              np.array([[-2.3, 2.0], [-2.2, 2.0], [-2.2, 2.1], [-2.3, 2.1]]),
              np.array([[2.3, -2.3], [2.4, -2.3], [2.4, -2.2], [2.3, -2.2]]),
              np.array([[0.0, -2.3], [0.1, -2.3], [0.1, -2.2], [0.0, -2.2]]),
            ]
# obstacles = []

robots = []
for i in range(params.num_robots):
    robots.append(Robot())
robot1 = robots[0]; robot1.leader=True



# Layered Motion Planning: RRT (global) + Potential Field (local)
if __name__ == '__main__':
    plt.figure(figsize=(10,10))
    draw_map(obstacles)
    plt.plot(xy_start[0],xy_start[1],'bo',color='red', markersize=20, label='start')
    plt.plot(xy_goal[0], xy_goal[1],'bo',color='green', markersize=20, label='goal')

    P_long = rrt_path(obstacles, xy_start, xy_goal, params)
    print('Path Shortenning...')
    P = ShortenPath(P_long, obstacles, smoothiters=100) # P = [[xN, yN], ..., [x1, y1], [x0, y0]]

    traj_global = waypts2setpts(P, params)
    P = np.vstack([P, xy_start])
    plt.plot(P[:,0], P[:,1], linewidth=3, color='orange', label='Global planner path')
    plt.pause(0.1)

    sp_ind = 0
    robot1.route = np.array([traj_global[0,:]])
    robot1.sp = robot1.route[-1,:]

    followers_sp = formation(params.num_robots, leader_des=robot1.sp, v=np.array([0,-1]), l=0.3)
    for i in range(len(followers_sp)):
        robots[i+1].sp = followers_sp[i]
        robots[i+1].route = np.array([followers_sp[i]])
    print('Start movement...')
    t0 = time.time(); t_array = []
    while True: # loop through all the setpoint from global planner trajectory, traj_global
        t_array.append( time.time() - t0 )
        dist_to_goal = norm(robot1.sp - xy_goal)
        if dist_to_goal < params.goal_tolerance: # [m]
            print 'Goal is reached'
            break
        if len(obstacles)>2: obstacles = move_obstacles(obstacles, params) # change poses of some obstacles on the map

        # leader's setpoint from global planner
        robot1.sp_global = traj_global[sp_ind,:]
        # correct leader's pose with local planner
        robot1.local_planner(obstacles, params)

        """ adding following robots in the swarm """
        # formation poses from global planner
        followers_sp_global = formation(params.num_robots, robot1.sp_global, v=normalize(robot1.sp_global-robot1.sp), l=0.3)
        for i in range(len(followers_sp_global)): robots[i+1].sp_global = followers_sp_global[i]
        for p in range(len(followers_sp)): # formation poses correction with local planner
            # robots repel from each other inside the formation
            robots_obstacles_sp = [x for i,x in enumerate(followers_sp + [robot1.sp]) if i!=p] # all poses except the robot[p]
            robots_obstacles = poses2polygons( robots_obstacles_sp ) # each drone is defined as a small cube for inter-robots collision avoidance
            obstacles1 = np.array(obstacles + robots_obstacles) # combine exisiting obstacles on the map with other robots[for each i: i!=p] in formation
            # follower robot's position correction with local planner
            robots[p+1].local_planner(obstacles1, params)
            followers_sp[p] = robots[p+1].sp


        # visualization
        # params.progress_bar.next()
        plt.cla()
        draw_map(obstacles)
        draw_gradient(robots[2].f) if params.num_robots>1 else draw_gradient(robots[0].f)
        for robot in robots: plt.plot(robot.sp[0], robot.sp[1], '^', color='blue', markersize=10, zorder=15) # robots poses
        plt.plot(robot1.route[:,0], robot1.route[:,1], linewidth=2, color='green', label="Robot's path, corrected with local planner", zorder=10)
        for robot in robots[1:]: plt.plot(robot.route[:,0], robot.route[:,1], '--', linewidth=2, color='green', zorder=10)
        plt.plot(P[:,0], P[:,1], linewidth=3, color='orange', label='Global planner path')
        plt.plot(traj_global[sp_ind,0], traj_global[sp_ind,1], 'ro', color='blue', markersize=7, label='Global planner setpoint')
        plt.plot(xy_start[0],xy_start[1],'bo',color='red', markersize=20, label='start')
        plt.plot(xy_goal[0], xy_goal[1],'bo',color='green', markersize=20, label='goal')
        plt.legend()
        if params.animate:
            plt.draw()
            plt.pause(0.01)

        # update loop variable
        if sp_ind < traj_global.shape[0]-1 and norm(robot1.sp_global - robot1.sp) < params.max_sp_dist: sp_ind += 1


    # close windows if Enter-button is pressed
    # params.progress_bar.finish()
    plt.draw()
    plt.pause(0.1)
    raw_input('Hit Enter to continue')
    plt.close('all')


""" Flight data postprocessing """
if params.postprocessing:
    plt.figure(figsize=(10,10))
    plt.title("Drones trajectories")
    # plt.plot(centroid_route[:,0], centroid_route[:,1], label='centroid')
    d = 0
    for robot in robots:
        d += 1
        plt.plot(robot.route[:,0], robot.route[:,1], '--', label='drone %d' %d, linewidth=2)
    plt.legend()
    plt.grid()

    plt.figure(figsize=(10,6))
    plt.title( "Leader Average Velocity: %.2f" %np.mean(np.array(robot1.vel_array)) )
    plt.plot( t_array[1:], robot1.vel_array, '-', label='drone 1', linewidth=2)
    for r in range(1, params.num_robots):
        plt.plot(t_array[1:], robots[r].vel_array, '--', label='drone %d' %(r+1), linewidth=2)
    plt.xlabel('Time, [s]')
    plt.ylabel('velocity, [m/s]')
    plt.legend()
    plt.grid()

    area_array = []
    for i in range(len(robot1.route)):
        X = np.array([]); Y = np.array([])
        for robot in robots:
            X = np.append( X, robot.route[i,0] )
            Y = np.append( Y, robot.route[i,1] )
        area_array.append(poly_area(X,Y))
    area_array = area_array[1:]
    S0 = area_array[0] # default inter-robots distance
    plt.figure(figsize=(10,6))
    plt.title("Area of robots' formation")
    plt.plot(t_array[1:], area_array, 'k', label='Formation area', linewidth=2)
    plt.plot(t_array[1:], S0*np.ones_like(t_array[1:]), '--', label='Default area', linewidth=2)
    plt.xlabel('Time, [s]')
    plt.ylabel('Formation area, [m^2]')
    plt.grid()
    plt.legend()

    # close windows if Enter-button is pressed
    plt.draw()
    plt.pause(0.1)
    raw_input('Hit Enter to close')
    plt.close('all')