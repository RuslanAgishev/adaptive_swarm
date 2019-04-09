import numpy as np
import time
from scipy.integrate import odeint
from math import *
import matplotlib.pyplot as plt


def MassSpringDamper(state,t,F, mode='critically_damped'):
	x = state[0]
	xd = state[1]
	if mode=='oscillations':
		m = 1.0; k = 2; b = 0 # undapmped: oscillations
	elif mode=='underdamped':
		m = 1.0; k = 2; b = 2*sqrt(m*k)-2 # underdamped
	elif mode=='overdamped':
		m = 1.0; k = 2; b = 2*sqrt(m*k)+2 # overdamped
	else:
		m = 1.0; k = 2; b = 2*sqrt(m*k) # critically damped
	# print mode+", damping ratio="+str(b/2/sqrt(m*k))
	xdd = -(b/m)*xd - (k/m)*x + F/m
	return [xd, xdd]


y0 = [np.pi - 0.1, 0.0]
M = 10
t = np.linspace(0, 4*pi, 101)
modes = ['underdamped', 'overdamped', 'critically_damped', 'oscillations']


plt.rcParams.update({'font.size': 30})
plt.figure(figsize=(20,20))
plt.grid()
plt.title('Impedance modeles')
plt.xlabel('t')
plt.ylabel('x(t)')
for mode in modes:
	# differential equation solution
	sol = odeint(MassSpringDamper, y0, t, args=(M,mode,))
	plt.plot(t, sol[:, 0], label=mode, linewidth=4)
	# plt.plot(t, sol[:, 1], 'g', label='omega(t)')
plt.legend(loc='best')
	

# close windows if Enter-button is pressed
plt.draw()
plt.pause(1)
raw_input('Hit Enter to close')
plt.close('all')