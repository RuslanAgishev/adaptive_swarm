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
import time

# for 3D plots
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def move_obstacles(obstacles, params):
    # small cubes movement
    obstacles[-3] += np.array([0.015, 0.0]) * params.drone_vel
    obstacles[-2] += np.array([-0.005, 0.005]) * params.drone_vel/2
    obstacles[-1] += np.array([0.0, 0.008]) * params.drone_vel/2
    return obstacles

class Params:
    def __init__(self):
        self.animate_rrt = 1 # show RRT construction, set 0 to reduce time of the RRT algorithm
        self.visualize = 1 # show robots movement
        self.postprocessing = 1 # process and visualize the simulated experiment data after the simulation
        self.savedata = 0 # save postprocessing metrics to the XLS-file
        self.maxiters = 500 # max number of samples to build the RRT
        self.goal_prob = 0.05 # with probability goal_prob, sample the goal
        self.minDistGoal = 0.25 # [m], min distance os samples from goal to add goal node to the RRT
        self.extension = 0.8 # [m], extension parameter: this controls how far the RRT extends in each step.
        self.world_bounds_x = [-2.5, 2.5] # [m], map size in X-direction
        self.world_bounds_y = [-2.5, 2.5] # [m], map size in Y-direction
        self.drone_vel = 4.0 # [m/s]
        self.ViconRate = 100 # [Hz]
        self.influence_radius = 0.15 # [m] potential fields radius, defining repulsive area size near the obstacle
        self.goal_tolerance = 0.05 # [m], maximum distance threshold to reach the goal
        self.num_robots = 4 # number of robots in the formation
        self.interrobots_dist = 0.3 # [m], distance between robots in default formation
        self.max_sp_dist = 0.2 * self.drone_vel# * np.sqrt(self.num_robots) # [m], maximum distance between current robot's pose and the sp from global planner

class Robot:
    def __init__(self, id):
        self.id = id
        self.sp = np.array([0, 0])
        self.sp_global = np.array([0,0])
        self.route = np.array([self.sp])
        self.vel_array = []
        self.U_a = 0 # attractive APF function
        self.U_r = 0 # repulsive APF function
        self.U = 0 # total APF function
        self.leader = False

    def local_planner(self, obstacles, params):
        """
        This function computes the next_point
        given current location (self.sp) and potential filed function, f.
        It also computes mean velocity, V, of the gradient map in current point.
        """
        obstacles_grid = grid_map(obstacles)
        self.U, self.U_a, self.U_r = combined_potential(obstacles_grid, self.sp_global, params.influence_radius)
        [gy, gx] = np.gradient(-self.U)
        iy, ix = np.array( meters2grid(self.sp), dtype=int )
        w = 20 # smoothing window size for gradient-velocity
        ax = np.mean(gx[ix-int(w/2) : ix+int(w/2), iy-int(w/2) : iy+int(w/2)])
        ay = np.mean(gy[ix-int(w/2) : ix+int(w/2), iy-int(w/2) : iy+int(w/2)])
        # ax = gx[ix, iy]; ay = gy[ix, iy]
        self.V = params.drone_vel * np.array([ax, ay])
        self.vel_array.append(norm(self.V))
        dt = 0.01 * params.drone_vel / norm([ax, ay]) if norm([ax, ay])!=0 else 0.01
        # self.sp += dt**2/2. * np.array( [ax, ay] )
        self.sp += dt*np.array( [ax, ay] ) #+ 0.1*dt**2/2. * np.array( [ax, ay] )
        self.route = np.vstack( [self.route, self.sp] )

def visualize2D():
    draw_map(obstacles)
    draw_gradient(robots[1].U) if params.num_robots>1 else draw_gradient(robots[0].U)
    for robot in robots: plt.plot(robot.sp[0], robot.sp[1], '^', color='blue', markersize=10, zorder=15) # robots poses
    robots_poses = []
    for robot in robots: robots_poses.append(robot.sp)
    robots_poses.sort(key=lambda p: atan2(p[1]-centroid[1],p[0]-centroid[0]))
    plt.gca().add_patch( Polygon(robots_poses, color='yellow') )
    plt.plot(centroid[0], centroid[1], '*', color='b', markersize=10, label='Centroid position')
    plt.plot(robot1.route[:,0], robot1.route[:,1], linewidth=2, color='green', label="Leader's path", zorder=10)
    # for robot in robots[1:]: plt.plot(robot.route[:,0], robot.route[:,1], '--', linewidth=2, color='green', zorder=10)
    plt.plot(P[:,0], P[:,1], linewidth=3, color='orange', label='Global planner path')
    plt.plot(traj_global[sp_ind,0], traj_global[sp_ind,1], 'ro', color='blue', markersize=7, label='Global planner setpoint')
    plt.plot(xy_start[0],xy_start[1],'bo',color='red', markersize=20, label='start')
    plt.plot(xy_goal[0], xy_goal[1],'bo',color='green', markersize=20, label='goal')
    plt.legend()

# Initialization
init_fonts(small=12, medium=16, big=26)
params = Params()
xy_start = np.array([1.2, 1.0])
xy_goal =  np.array([1.5, -1.4])
# xy_goal =  np.array([1.3, 1.0])

# Obstacles map construction
obstacles = [
              # bugtrap
              np.array([[0.5, 0], [2.5, 0.], [2.5, 0.3], [0.5, 0.3]]),
              np.array([[0.5, 0.3], [0.8, 0.3], [0.8, 1.5], [0.5, 1.5]]),
              # np.array([[0.5, 1.5], [1.5, 1.5], [1.5, 1.8], [0.5, 1.8]]),
              # angle
              np.array([[-2, -2], [-0.5, -2], [-0.5, -1.8], [-2, -1.8]]),
              np.array([[-0.7, -1.8], [-0.5, -1.8], [-0.5, -0.8], [-0.7, -0.8]]),
              # walls
              np.array([[-2.5, -2.5], [2.5, -2.5], [2.5, -2.47], [-2.5, -2.47]]), # comment this for better 3D visualization
              np.array([[-2.5, 2.47], [2.5, 2.47], [2.5, 2.5], [-2.5, 2.5]]),
              np.array([[-2.5, -2.47], [-2.47, -2.47], [-2.47, 2.47], [-2.5, 2.47]]),
              np.array([[2.47, -2.47], [2.5, -2.47], [2.5, 2.47], [2.47, 2.47]]), # comment this for better 3D visualization

              # moving obstacle
              np.array([[-2.3, 2.0], [-2.2, 2.0], [-2.2, 2.1], [-2.3, 2.1]]),
              np.array([[2.3, -2.3], [2.4, -2.3], [2.4, -2.2], [2.3, -2.2]]),
              np.array([[0.0, -2.3], [0.1, -2.3], [0.1, -2.2], [0.0, -2.2]]),
            ]
"""" Narrow passage """
# passage_width = 0.3
# passage_location = 0.0
# obstacles = [
#             # narrow passage
#               np.array([[-2.5, -0.5], [-passage_location-passage_width/2., -0.5], [-passage_location-passage_width/2., 0.5], [-2.5, 0.5]]),
#               np.array([[-passage_location+passage_width/2., -0.5], [2.5, -0.5], [2.5, 0.5], [-passage_location+passage_width/2., 0.5]]),
#             ]
# obstacles = []

robots = []
for i in range(params.num_robots):
    robots.append(Robot(i+1))
robot1 = robots[0]; robot1.leader=True


# Metrics to measure (for postprocessing)
class Metrics:
    def __init__(self):
        self.mean_dists_array = []
        self.max_dists_array = []
        self.centroid_path = [np.array([0,0])]
        self.centroid_path_length = 0
        self.robots = []
        self.vels_mean = []
        self.vels_max = []
        self.area_array = []
        self.cpu_usage_array = [] # [%]
        self.memory_usage_array = [] # [MiB]

        self.folder_to_save = '/home/rus/Desktop/'

metrics = Metrics()

# Layered Motion Planning: RRT (global) + Potential Field (local)
if __name__ == '__main__':
    fig2D = plt.figure(figsize=(10,10))
    draw_map(obstacles)
    plt.plot(xy_start[0],xy_start[1],'bo',color='red', markersize=20, label='start')
    plt.plot(xy_goal[0], xy_goal[1],'bo',color='green', markersize=20, label='goal')

    P_long = rrt_path(obstacles, xy_start, xy_goal, params)
    print('Path Shortenning...')
    P = ShortenPath(P_long, obstacles, smoothiters=50) # P = [[xN, yN], ..., [x1, y1], [x0, y0]]

    traj_global = waypts2setpts(P, params)
    P = np.vstack([P, xy_start])
    plt.plot(P[:,0], P[:,1], linewidth=3, color='orange', label='Global planner path')
    plt.pause(0.5)

    sp_ind = 0
    robot1.route = np.array([traj_global[0,:]])
    robot1.sp = robot1.route[-1,:]

    followers_sp = formation(params.num_robots, leader_des=robot1.sp, v=np.array([0,-1]), l=params.interrobots_dist)
    for i in range(len(followers_sp)):
        robots[i+1].sp = followers_sp[i]
        robots[i+1].route = np.array([followers_sp[i]])
    print('Start movement...')
    t0 = time.time(); t_array = []

    while True: # loop through all the setpoint from global planner trajectory, traj_global
        t_array.append( time.time() - t0 )
        # print("Current time [sec]: ", time.time() - t0)
        dist_to_goal = norm(robot1.sp - xy_goal)
        if dist_to_goal < params.goal_tolerance: # [m]
            print('Goal is reached')
            break
        if len(obstacles)>2: obstacles = move_obstacles(obstacles, params) # change poses of some obstacles on the map

        # leader's setpoint from global planner
        robot1.sp_global = traj_global[sp_ind,:]
        # correct leader's pose with local planner
        robot1.local_planner(obstacles, params)

        """ adding following robots in the swarm """
        # formation poses from global planner
        followers_sp_global = formation(params.num_robots, robot1.sp_global, v=normalize(robot1.sp_global-robot1.sp), l=params.interrobots_dist)
        for i in range(len(followers_sp_global)): robots[i+1].sp_global = followers_sp_global[i]
        for p in range(len(followers_sp)): # formation poses correction with local planner
            # robots repel from each other inside the formation
            robots_obstacles_sp = [x for i,x in enumerate(followers_sp + [robot1.sp]) if i!=p] # all poses except the robot[p]
            robots_obstacles = poses2polygons( robots_obstacles_sp ) # each drone is defined as a small cube for inter-robots collision avoidance
            obstacles1 = np.array(obstacles + robots_obstacles) # combine exisiting obstacles on the map with other robots[for each i: i!=p] in formation
            # follower robot's position correction with local planner
            robots[p+1].local_planner(obstacles1, params)
            followers_sp[p] = robots[p+1].sp

        # centroid pose:
        centroid = 0
        for robot in robots: centroid += robot.sp / len(robots)
        metrics.centroid_path = np.vstack([metrics.centroid_path, centroid])
        # dists to robots from the centroid:
        dists = []
        for robot in robots:
            dists.append( norm(centroid-robot.sp) )
        # Formation size estimation
        metrics.mean_dists_array.append(np.mean(dists)) # Formation mean Radius
        metrics.max_dists_array.append(np.max(dists)) # Formation max Radius

        # Algorithm performance (CPU and memory usage)
        metrics.cpu_usage_array.append( cpu_usage() )
        metrics.memory_usage_array.append( memory_usage() )
        # print("CPU: ", cpu_usage())
        # print("Memory: ", memory_usage())

        # visualization
        if params.visualize:
            plt.cla()
            visualize2D()        

            plt.draw()
            plt.pause(0.01)

        # update loop variable
        if sp_ind < traj_global.shape[0]-1 and norm(robot1.sp_global - centroid) < params.max_sp_dist: sp_ind += 1




""" Flight data postprocessing """
if params.postprocessing:
    t_array = t_array[1:]
    metrics.t_array = t_array
    metrics.centroid_path = metrics.centroid_path[1:,:]
    metrics.centroid_path_length = path_length(metrics.centroid_path)
    for robot in robots: metrics.robots.append( robot )

    postprocessing(metrics, params, visualize=1)
    if params.savedata: save_data(metrics)

# close windows if Enter-button is pressed
plt.draw()
plt.pause(0.1)
raw_input('Hit Enter to close')
plt.close('all')