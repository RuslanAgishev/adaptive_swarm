import contextlib
from matplotlib import animation as anim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from math import *
import time


def draw_map(start, goal, obstacles_poses, R_obstacles, f=None, draw_gradients=True, nrows=500, ncols=500):
    if draw_gradients and f is not None:
        skip = 10
        [x_m, y_m] = np.meshgrid(np.linspace(-2.5, 2.5, ncols), np.linspace(-2.5, 2.5, nrows))
        [gy, gx] = np.gradient(-f);
        Q = plt.quiver(x_m[::skip, ::skip], y_m[::skip, ::skip], gx[::skip, ::skip], gy[::skip, ::skip])#, scale=100, width=0.002)
    else:
        plt.grid()
    plt.plot(start[0], start[1], 'ro', color='yellow', markersize=10);
    plt.plot(goal[0], goal[1], 'ro', color='green', markersize=10);
    plt.xlabel('X, [m]')
    plt.ylabel('Y, [m]')
    ax = plt.gca()
    for pose in obstacles_poses:
        circle = plt.Circle(pose, R_obstacles, color='k')
        ax.add_artist(circle)
    # Create a Rectangle patch
    rect1 = patches.Rectangle((-2.5,-1.15),2.0,0.2,linewidth=1,color='k',fill='True')
    rect2 = patches.Rectangle((-1.2, 1.4), 0.2,1.0,linewidth=1,color='k',fill='True')
    rect3 = patches.Rectangle(( 0.4, 0.8), 2.0,0.5,linewidth=1,color='k',fill='True')
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    
def draw_robots(current_point1, R_drones, routes=None, num_robots=None, robots_poses=None, centroid=None, vel1=None, plot_routes=1, plot_arrow=0):
    if vel1 is not None and plot_arrow: plt.arrow(current_point1[0], current_point1[1], vel1[0], vel1[1], width=0.01, head_width=0.05, head_length=0.1, fc='k')
    if plot_routes:
        plt.plot(routes[0][:,0], routes[0][:,1], 'green', linewidth=2)
        for r in range(1, num_robots):
            plt.plot(routes[r][:,0], routes[r][:,1], '--', color='blue', linewidth=2)

    for pose in robots_poses[:-1]:
        plt.plot(pose[0], pose[1], '^', markersize=R_drones*100, color='blue')
    plt.plot(robots_poses[-1][0], robots_poses[-1][1], '^', markersize=R_drones*100, color='green')
    # plt.plot(robots_poses[-1][0], robots_poses[-1][1], '^', markersize=R_drones*100, color='blue')

    # compute centroid and sort poses by polar angle
    if num_robots<7:
        pp = robots_poses
        pp.sort(key=lambda p: atan2(p[1]-centroid[1],p[0]-centroid[0]))
        formation = patches.Polygon(pp, color='blue', fill=False, linewidth=2);
        plt.gca().add_patch(formation)
    plt.plot(centroid[0], centroid[1], '*', color='blue')

def get_movie_writer(should_write_movie, title, movie_fps, plot_pause_len):
    """
    :param should_write_movie: Indicates whether the animation of SLAM should be written to a movie file.
    :param title: The title of the movie with which the movie writer will be initialized.
    :param movie_fps: The frame rate of the movie to write.
    :param plot_pause_len: The pause durations between the frames when showing the plots.
    :return: A movie writer that enables writing MP4 movie with the animation from SLAM.
    """

    get_ff_mpeg_writer = anim.writers['ffmpeg']
    metadata = dict(title=title, artist='matplotlib', comment='Potential Fields Formation Navigation')
    movie_fps = min(movie_fps, float(1. / plot_pause_len))

    return get_ff_mpeg_writer(fps=movie_fps, metadata=metadata)

@contextlib.contextmanager
def get_dummy_context_mgr():
    """
    :return: A dummy context manager for conditionally writing to a movie file.
    """
    yield None


# OBJECT VELOCITY CALCULATION
time_array = np.ones(10)
pose_array = np.array([ np.ones(10), np.ones(10), np.ones(10) ])
def velocity(pose):
    for i in range(len(time_array)-1):
        time_array[i] = time_array[i+1]
    time_array[-1] = time.time()

    for i in range(len(pose_array[0])-1):
        pose_array[0][i] = pose_array[0][i+1]
        pose_array[1][i] = pose_array[1][i+1]
        pose_array[2][i] = pose_array[2][i+1]
    pose_array[0][-1] = pose[0]
    pose_array[1][-1] = pose[1]
    pose_array[2][-1] = pose[2]

    vel_x = (pose_array[0][-1]-pose_array[0][0])/(time_array[-1]-time_array[0])
    vel_y = (pose_array[1][-1]-pose_array[1][0])/(time_array[-1]-time_array[0])
    vel_z = (pose_array[2][-1]-pose_array[2][0])/(time_array[-1]-time_array[0])

    vel = np.array( [vel_x, vel_y, vel_z] )
    
    return vel


def euler_from_quaternion(q):
    """
    Intrinsic Tait-Bryan rotation of xyz-order.
    """
    q = q / np.linalg.norm(q)
    qx, qy, qz, qw = q
    roll = atan2(2.0*(qy*qz + qw*qx), qw*qw - qx*qx - qy*qy + qz*qz)
    pitch = asin(-2.0*(qx*qz - qw*qy))
    yaw = atan2(2.0*(qx*qy + qw*qz), qw*qw + qx*qx - qy*qy - qz*qz)
    return roll, pitch, yaw


def formation(num_robots, leader_des, v, l):
    """
    geometry of the swarm: following robots desired locations
    relatively to the leader
    """
    v = v / norm(v)
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

def init_fonts():
    SMALL_SIZE = 12
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 26

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title