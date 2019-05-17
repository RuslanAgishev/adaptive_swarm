#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from potential_fields import gradient_planner, combined_potential

# plt.rcParams.update({'font.size': 20})
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

def gradient_plot(x,y, gx,gy, skip=10):
    # plt.figure(figsize=(12,8))
    plt.figure()
    Q = plt.quiver(x[::skip, ::skip], y[::skip, ::skip], gx[::skip, ::skip], gy[::skip, ::skip],
                   pivot='mid', units='inches')
    qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                       coordinates='figure')
#     plt.scatter(x[::skip, ::skip], y[::skip, ::skip], color='r', s=5)


def GradientBasedPlanner (f, start_coords, end_coords, max_its):
    # GradientBasedPlanner : This function plans a path through a 2D
    # environment from a start to a destination based on the gradient of the
    # function f which is passed in as a 2D array. The two arguments
    # start_coords and end_coords denote the coordinates of the start and end
    # positions respectively in the array while max_its indicates an upper
    # bound on the number of iterations that the system can use before giving
    # up.
    # The output, route, is an array with 2 columns and n rows where the rows
    # correspond to the coordinates of the robot as it moves along the route.
    # The first column corresponds to the x coordinate and the second to the y coordinate

    [gy, gx] = np.gradient(-f);

    route = np.vstack( [np.array(start_coords), np.array(start_coords)] )
    for i in range(max_its):
        current_point = route[-1,:];
#         print(sum( abs(current_point-end_coords) ))
        if sum( abs(current_point-end_coords) ) < 5.0:
            print('Reached the goal !');
            break
        ix = int(round( current_point[1] ));
        iy = int(round( current_point[0] ));
        vx = gx[ix, iy]
        vy = gy[ix, iy]
        dt = 1 / np.linalg.norm([vx, vy]);
        next_point = current_point + dt*np.array( [vx, vy] );
        route = np.vstack( [route, next_point] );
    route = route[1:,:]
        
    return route

# Generate some points
nrows = 400;
ncols = 600;

obstacle = np.zeros((nrows, ncols));
[x, y] = np.meshgrid (np.arange(ncols), np.arange(nrows));

# Generate some obstacle
obstacle [300:, 100:250] = True;
obstacle [150:200, 400:500] = True;
t = ((x - 200)**2 + (y - 50)**2) < 50**2;
obstacle[t] = True;
t = ((x - 400)**2 + (y - 300)**2) < 80**2;
obstacle[t] = True;
plt.imshow(1 - obstacle, 'gray')

# Compute distance transform
from scipy.ndimage.morphology import distance_transform_edt as bwdist

d = bwdist(obstacle==0) # distance to closest obstacle function
d2 = (d/100.) + 1 # rescale and transform distances
d0 = 2 # radius of potential field influence
nu = 800
repulsive = nu*((1./d2 - 1./d0)**2)
repulsive [d2 > d0] = 0

# Display repulsive potential
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.title ('Repulsive Potential')
# Plot the surface.
surf = ax.plot_surface(x, y, repulsive, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# # Compute attractive force
goal = [400, 50];
start = [50, 350];

xi = 1./700

attractive = xi * ( (x - goal[0])**2 + (y - goal[1])**2 );

# Display attractive potential
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x, y, attractive, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.title ('Attractive Potential')

# # Combine terms
f = attractive + repulsive
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(y, x, f, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.title ('Total Potential')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel('X [cm]')
plt.ylabel('Y [cm]')
ax.set_zlabel('f(x,y)')

# # Display 2D configuration space
# plt.figure()
# plt.imshow(1-obstacle, 'gray')
# plt.plot (start[0], start[1], 'ro', markersize=10, label='start')
# plt.plot (goal[0], goal[1], 'ro', color='green', markersize=10, label='goal')
# # plt.axis ([0, ncols, 0, nrows]);
# plt.xlabel('X [cm]')
# plt.ylabel('Y [cm]')
# plt.legend()
# plt.title ('2D map with obstacles')

# Plan route
route = GradientBasedPlanner(f, start, goal, 700);

# Compute gradients for visualization
[gx, gy] = np.gradient(-f);


# plt.figure(figsize=(12,8))
# plt.imshow(gy, 'gray')
# plt.title('Gx=df/dx - gradient')


# plt.figure(figsize=(12,8))
# plt.imshow(gx, 'gray')
# plt.title('Gy=df/dy - gradient')


# Velocities plot
skip = 10;
xidx = np.arange(0,ncols,skip)
yidx = np.arange(0,nrows,skip)
gradient_plot(x,y, gy,gx, skip=10)
plt.plot(start[0], start[1], 'ro', color='green', markersize=10, label='Start');
plt.plot(goal[0], goal[1], 'ro', color='red', markersize=10, label='Goal');
plt.plot(route[:,0], route[:,1], linewidth=3, label='Path');
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='lower right')
# plt.title('Gradient plot')

plt.draw()
plt.pause(1)
raw_input('Hit Enter to close')
plt.close('all')