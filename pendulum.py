import scipy as sc
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as anim


g = 9.81

dt = 0.001

fig, ax = plt.subplots()


class Pendulum:
	
	def __init__(self, m, l, theta0, v0):
		
		self.m, self.l = m, l

		self.theta0, self.v0 = theta0, v0

		self.T = 2 * np.pi * np.sqrt(l / g)

		self.theta, self.v = [theta0], [v0]

		old_theta = theta0

		t = 0

		while True:
			
			t += dt

			old_theta, old_v = self.theta[-1], self.v[-1]

			a = g * np.sin(old_theta)

			new_theta = old_theta - old_v * dt / l
			new_v = old_v + a * dt
			
			if (t > self.T and new_v * old_v <= 0) or t > 1.5 * self.T:
				break

			self.theta.append(new_theta)
			self.v.append(new_v)

	def PlotTime(self, plot_theta = True, plot_v = True, col = ((1,0,0), (0,0,1))):
		t = np.linspace(0, self.T, num = len(self.theta))

		if plot_theta == True:
			ax.plot(t, self.theta, color = col[0])

		if plot_v == True:
			ax.plot(t, self.v, color = col[1])

	def PlotPhase(self, col = (1,0,0)):
		ax.plot(self.theta, self.v, color = col)

	def PrintData(self):
		print(f'Mass: {self.m:.2f}, Length: {self.l:.2f}')
		print(f'Initial Angle: {self.theta0:.2f}, Initial Velocity: {self.v0:.2f}')
		print(f'Harmonic Time: {self.T:.2f}')

def VaryTheta(mass, length, v0, n):

	theta = np.linspace(0, np.pi, num = n)

	plasma = mpl.colormaps['plasma']

	colors = plasma(theta)

	for i in range(n):

		p = Pendulum(mass, length, theta[i], v0)

		p.PlotPhase(col = colors[i])

def VaryVel(mass, length, theta0, vmax, n):

	velocity = np.linspace(0, vmax, num = n)

	plasma = mpl.colormaps['plasma']

	colors = plasma(velocity)

	for i in range(n):

		p = Pendulum(mass, length, theta0, velocity[i])

		p.PlotPhase(col = colors[i])

#generate a pendulum, print its harmonic time, and plot its time evolution:

#p = Pendulum(1, 1, np.pi/6, 0)
#p.PrintData()
#p.PlotTime()

#plot 20 pendulums in phase space with varying initial angle

#VaryTheta(1, 1, 0, 20)

#plot 20 pendulums in phase space with varying initial velocity

#VaryVel(1, 1, np.pi / 6, 2, 20)

plt.show()