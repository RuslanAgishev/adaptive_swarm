#!/usr/bin/env python

import numpy as np
from numpy.linalg import norm
import pygame
from tools import *
from agents import Robot


# drawing initialization
BACKGROUNDCOLOR = [255,255,255]
LINECOLOR       = [0,0,0]
AGENTCOLOR = [0,0,255]
AGENTSIZE = 9
AGENTSICKNESS = 3
nrows = 500; ncols = 500
GRIDSIZE = [nrows, ncols]
ROBOTSNUMBER = 10


pygame.init()
screen = pygame.display.set_mode(GRIDSIZE)
pygame.display.set_caption('Social Force Model')
clock = pygame.time.Clock()

walls = [[0.0, 0.0, 2.0, 0.0],
         [2.0, 0.0, 2.0, 2.0],
         [0.0, 0.0, 0.0,-2.0],
         [-3.0,-2.0, 0.0,-2.0]]


# define start and goal locations:
start_locations = [ np.array([-1.7, -1.7]), np.array([1.7, 1.7]) ] # [m, m]
goal = np.array([1, 1.0]) # [m, m]

robots = []
# for start in start_locations:
#     robot = Robot(start, goal)
#     robots.append( robot )
for n in range(ROBOTSNUMBER):
    robots.append(Robot())


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BACKGROUNDCOLOR)

    # draw walls
    for wall in walls:
        startPos = np.array([wall[0],wall[1]])
        endPos = np.array([wall[2],wall[3]])
        startPx = meters2grid(startPos, nrows, ncols)
        endPx = meters2grid(endPos, nrows, ncols)
        pygame.draw.line(screen, LINECOLOR, startPx, endPx)

    # desired velocity and direction of motion
    for robot in robots:
        robot.direction = normalize(robot.goal - robot.pose)
        robot.vel_des = robot.speed_des * robot.direction

        # RESULTANT FORCE CALCULATION
        # 1. attractive - movement to the goal:
        attractive = robot.goal_movement()

        # 2. peopleInter - force from people
        interrobots_force = 0.0
        for other_robot in robots:
            if other_robot != robot:
                interrobots_force += robot.robots_interaction(other_robot)

        # 3. wall_force - force from walls
        wall_force = 0.0
        for wall in walls:
            wall_force += robot.wall_interaction(wall)

        sumForce = attractive + interrobots_force + wall_force
        # resultant acceleration:
        accl = sumForce / robot.mass
        # resultant velocity:
        dt = 1 # timestep
        robot.vel = robot.vel + accl*dt
        # resultant displacement:
        robot.pose = robot.pose + robot.vel*dt

    

        # Visualization
        pygame.draw.circle(screen, AGENTCOLOR, meters2grid(robot.pose, nrows, ncols), AGENTSIZE, AGENTSICKNESS)
        pygame.draw.circle(screen, [0,255,0], meters2grid(robot.goal, nrows, ncols), AGENTSIZE, AGENTSICKNESS)
        pygame.draw.line(screen, AGENTCOLOR, meters2grid(robot.pose, nrows, ncols), meters2grid(robot.pose+robot.vel_des, nrows, ncols), 2)

    pygame.display.flip()
    clock.tick(20)











