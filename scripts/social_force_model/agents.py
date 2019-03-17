#!/usr/bin/env python

import numpy as np
import random
from tools import *

class Robot():
    def __init__(self, start=np.array([0,0]), goal=np.array([1,1])):
        self.start = np.array([random.uniform(-2.0, 2.0),random.uniform(-2.0, 2.0)])
        self.goal = goal
        
        self.pose = self.start # [m]
        self.vel = np.array([0, 0]) # [m/sec]

        self.direction = (self.goal - self.pose) / np.linalg.norm(self.goal - self.pose)
        self.speed_des = 0.1 # [m/sec]
        self.vel_des = self.speed_des*self.direction # array
        
        self.acclTime = 10.0
        self.drivenAcc = (self.vel_des - self.vel) / self.acclTime
               
        self.mass = 1.0 # [kg], random.uniform(40,90)
        self.radius = 0.1 # [m]
        self.p = 0.6
        
        self.A = 0.1
        self.B = 0.05 #random.uniform(0.8,1.6) #0.8 #0.08
        self.bodyFactor = 0.001

    def goal_movement(self):
        deltaV = self.vel_des - self.vel
        if np.allclose(deltaV, np.zeros(2)): deltaV = np.zeros(2)
        return deltaV * self.mass / self.acclTime

    def robots_interaction(self, other):
        rij = self.radius + other.radius
        dij = np.linalg.norm(self.pose - other.pose)
        nij = (self.pose - other.pose) / dij
        interrobot_repel = self.A * np.exp( (rij-dij) / self.B ) * nij
        return interrobot_repel

    def wall_interaction(self, wall):
        ri = self.radius
        diw, niw = distance_to_wall(self.pose, wall)
        first = - self.A * np.exp( (ri-diw) / self.B ) * niw + self.bodyFactor * g(ri-diw) * niw
        return first