#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from tools import *

class Robot():
    def __init__(self):
        self.pose = np.array([0, 0]) # [m]
        self.vel = np.array([0, 0]) # [m/sec]

        self.goal = np.array([0,0])
        self.direction = (self.goal - self.pose) / np.linalg.norm(self.goal - self.pose)
        
        self.speed_des = 1.0 # [m/sec]
        self.vel_des = self.speed_des*self.direction # array
        
        self.acclTime = 10.0
        self.drivenAcc = (self.vel_des - self.vel) / self.acclTime
               
        self.mass = 1.0 # [kg], random.uniform(40,90)
        self.radius = 0.35 # [m], 1.6 

    def adaptVel(self):
        deltaV = self.vel_des - self.vel
        if np.allclose(deltaV, np.zeros(2)): deltaV = np.zeros(2)
        return deltaV * self.mass / self.acclTime



# Generate some obstacles
nrows = 400
ncols = 600
obstacle = np.zeros((nrows, ncols))
[x, y] = np.meshgrid (np.arange(ncols), np.arange(nrows))
obstacle [300:, 100:250] = True
obstacle [150:200, 400:500] = True
t = ((x - 200)**2 + (y - 50)**2) < 50**2
obstacle[t] = True
t = ((x - 400)**2 + (y - 300)**2) < 100**2
obstacle[t] = True

# define start and goal locations:
goal = np.array([400, 50])
start = np.array([50, 350])

# Display map
def draw_map(obstacle, start, goal):
    plt.imshow(1-obstacle, 'gray');
    plt.plot (start[0], start[1], 'ro', markersize=10);
    plt.plot (goal[0], goal[1], 'ro', color='green', markersize=10);
    # plt.axis ([0, ncols, 0, nrows]);
    plt.xlabel ('x')
    plt.ylabel ('y')
    plt.title ('Obstacles Map')



robot = Robot()
robot.start = start; robot.goal = goal
i = 0
plt.figure()
while i<100:
    i+=1
    # interaction force calculation
    # initial velocity and position
    robot.direction = normalize(robot.goal - robot.pose)
    robot.vel_des = robot.speed_des * robot.direction

    # RESULTANT FORCE CALCULATION
    # 1. adapt - movement to the goal:
    adapt = robot.adaptVel()
    sumForce = adapt

    # resultant acceleration:
    accl = sumForce / robot.mass
    # resultant velocity:
    dt = 1 # timestep
    robot.vel = robot.vel + accl*dt
    # resultant displacement:
    robot.pose = robot.pose + robot.vel*dt

    plt.cla()
    draw_map(obstacle, robot.pose, goal)

    plt.draw()
    plt.pause(0.01)