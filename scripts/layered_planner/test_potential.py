#!/usr/bin/env python

import matplotlib.pyplot as plt
from scipy.ndimage.morphology import distance_transform_edt as bwdist
import numpy as np

def init_fonts(small=12, medium=16, big=26):
    SMALL_SIZE = small
    MEDIUM_SIZE = medium
    BIGGER_SIZE = big

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

obstacles_grid = np.zeros(500)
obstacles_grid[:235] = 1; obstacles_grid[265:] = 1


d = bwdist(obstacles_grid==0)
d2 = (d/100.) + 1

d0 = 0.15 + 1
nu = 200
repulsive = nu*((1./d2 - 1./d0)**2)
repulsive[d2 > d0] = 0
repulsive /= np.max(repulsive)

x = np.linspace(-2.5, 2.5, len(obstacles_grid))

init_fonts()
plt.plot(x, 2*repulsive, color='red', label='Repulsive potential, $U_y^r$', linewidth=2)
plt.plot(x, 2*obstacles_grid, color='k', label='Obstacles shape', linewidth=2)
plt.grid()
plt.xlim([-1,1])
plt.ylim([0, 3])
plt.xlabel('Y [m]')
plt.ylabel('Z [m]')
plt.legend()



# close windows if Enter-button is pressed
plt.draw()
plt.pause(0.1)
raw_input('Hit Enter to close')
plt.close('all')