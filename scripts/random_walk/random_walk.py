"""
Random walk algorithm implementation for a mobile robot
equipped with 4 ranger sensors (front, back, left and right)
for obstacles detection

author: Ruslan Agishev (agishev_ruslan@mail.ru)
reference: https://ieeexplore.ieee.org/abstract/document/6850799/s
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_arrow(x, y, yaw, length=5, width=1):  # pragma: no cover
    plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)

def plot_robot(pose, r=10):
	plt.plot([pose[0]-r*np.cos(pose[2]), pose[0]+r*np.cos(pose[2])],
			 [pose[1]-r*np.sin(pose[2]), pose[1]+r*np.sin(pose[2])], '--', color='b')
	plt.plot([pose[0]-r*np.cos(pose[2]+np.pi/2), pose[0]+r*np.cos(pose[2]+np.pi/2)],
		     [pose[1]-r*np.sin(pose[2]+np.pi/2), pose[1]+r*np.sin(pose[2]+np.pi/2)], '--', color='b')
	plt.plot(pose[0], pose[1], 'ro', markersize=5)
	plot_arrow(pose[0], pose[1], pose[2])

def obstacle_check(pose, r, gmap):
	back = [pose[0]-r*np.cos(pose[2]), pose[1]-r*np.sin(pose[2])]
	front = [pose[0]+r*np.cos(pose[2]), pose[1]+r*np.sin(pose[2])]
	right = [pose[0]+r*np.cos(pose[2]+np.pi/2), pose[1]+r*np.sin(pose[2]+np.pi/2)]
	left = [pose[0]-r*np.cos(pose[2]+np.pi/2), pose[1]-r*np.sin(pose[2]+np.pi/2)]

	pi = np.array(pose[:2], dtype=int)
	backi = np.array(back, dtype=int)
	fronti = np.array(front, dtype=int)
	lefti = np.array(left, dtype=int)
	righti = np.array(right, dtype=int)

	obstacle = {
		'front': 0,
		'back':  0,
		'right': 0,
		'left':  0,
	}

	for i in np.arange(min(pi[0], fronti[0]), max(pi[0], fronti[0])+1):
		for j in np.arange(min(pi[1], fronti[1]), max(pi[1], fronti[1])+1):
			if gmap[j,i]:
				# print('FRONT collision')
				obstacle['front'] = 1

	for i in np.arange(min(pi[0], backi[0]), max(pi[0], backi[0])+1):
		for j in np.arange(min(pi[1], backi[1]), max(pi[1], backi[1])+1):
			if gmap[j,i]:
				# print('BACK collision')
				obstacle['back'] = 1

	for i in np.arange(min(pi[0], lefti[0]), max(pi[0], lefti[0])+1):
		for j in np.arange(min(pi[1], lefti[1]), max(pi[1], lefti[1])+1):
			if gmap[j,i]:
				# print('LEFT collision')
				obstacle['left'] = 1

	for i in np.arange(min(pi[0], righti[0]), max(pi[0], righti[0])+1):
		for j in np.arange(min(pi[1], righti[1]), max(pi[1], righti[1])+1):
			if gmap[j,i]:
				# print('RIGHT collision')
				obstacle['right'] = 1

	return obstacle


WIDTH = 100
HEIGHT = 100

#    x,    y,      yaw
pose = [10.0, 80.0, np.pi/3]
traj = np.array(pose)

gmap = np.zeros([WIDTH, HEIGHT])
# walls
border = 10
gmap[:border, :] = 1
gmap[-border:, :] = 1
gmap[:, :border] = 1
gmap[:, -border:] = 1
# obstacles
gmap[20:30, 60:80] = 1
gmap[40:50, 40:50] = 1
gmap[60:80, 50:70] = 1

plt.figure(figsize=(10,10))

if __name__ == '__main__':
	try:
		while True:
			vel = 3
			pose[0] += vel*np.cos(pose[2])
			pose[1] += vel*np.sin(pose[2])

			obstacle = obstacle_check(pose, 10, gmap)
			# print(obstacle)

			if obstacle['right'] or obstacle['front']:
				pose[2] -= np.pi/2 * np.random.uniform()
			elif obstacle['left']:
				pose[2] += np.pi/2 * np.random.uniform()

			traj = np.vstack([traj, pose])
			
			plt.cla()
			plt.imshow(1-gmap, cmap='gray')
			plt.plot(traj[:,0], traj[:,1], 'g')
			plot_robot(pose)

			plt.pause(0.1)
		plt.show()
	except KeyboardInterrupt:
	    pass
		