import scipy as sc
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as anim


g = 9.81

dt = 0.001


class Pendulum:
	
	def __init__(self, m1, m2, l1, l2, theta1, theta2, v1, v2, T = 0):
		
		self.m, self.l = np.array([m1, m2]), np.array([l1, l2])

		self.theta0, self.v0 = np.array([theta1, theta2]), np.array([v1, v2])

		self.T = T

		self.theta, self.v = [self.theta0], [self.v0]

		old_theta = self.theta0

		t = 0

		M = m1 + m2

		while True:
			
			t += dt

			old_theta, old_v = self.theta[-1], self.v[-1]

			c, s = np.cos(old_theta[0] - old_theta[1]), np.sin(old_theta[0] - old_theta[1])

			a = np.zeros(2)
			
			a[0] = (-g * (m1 + M) * np.sin(old_theta[0]) - g * m2 * np.sin(old_theta[0] - 2 * old_theta[1]) - 2 * m2 * s * (l2 * old_v[1]**2 + l1 * c * old_v[0]**2)) / l1 / (m1 + m2 * s**2) / 2
			a[1] = 2 * s *(l1 * M * old_v[0]**2 + g * M * np.cos(old_theta[0]) + 2 * m2 * c * old_v[1]**2) / l2 / (m1 + m2 * s**2) / 2

			new_theta = old_theta - dt * old_v / self.l
			new_v = old_v - a * dt

			if new_theta[0] > np.pi or new_theta[0] < -np.pi:
				self.fliptime = t

			if T == 0:
				if new_theta[0] > np.pi or new_theta[0] < -np.pi:
					break
			elif t >= T:
				break


			self.theta.append(new_theta)
			self.v.append(new_v)

	def PlotTime(self, plot_theta = True, plot_v = True, col = ('b', 'r', 'g', 'm', 'c')):

		fig, ax = plt.subplots()

		t = np.linspace(0, self.T, num = len(self.theta))

		theta1, theta2, v1, v2 = [], [], [], []


		for i in self.theta:

			theta1.append(i[0])
			theta2.append(i[1])

		for i in self.v:

			v1.append(i[0])
			v2.append(i[1])

		if plot_theta == True:

			ax[0].set_title("Angle")

			ax[0].plot(t, theta1, color = col[0])
			ax[0].plot(t, theta2, color = col[1])


		if plot_v == True:
		
			ax[1].set_title("Velocity")

			ax[1].plot(t, v1, color = col[3])
			ax[1].plot(t, v2, color = col[4])

	def Animate(self):

		fig, ax = plt.subplots()
		
		artists = []

		l = [0, self.l[0], self.l[1]]

		idx = np.round(np.linspace(0, len(self.theta) - 1, num = np.round(len(self.theta) / 60).astype(int))).astype(int)

		for i in idx:
			theta = [self.theta[i][0], self.theta[i][1]]
			x = l * np.array([0, np.sin(theta[0]), np.sin(theta[0]) + np.sin(theta[1])])
			y = l * np.array([0, -np.cos(theta[0]), - np.cos(theta[0]) - np.cos(theta[1])])

			plot = ax.plot(x, y, color = 'b')

			artists.append(plot)

		animation = anim.ArtistAnimation(fig = fig, artists = artists, interval = 80)

		plt.show()

def AnimatePendulums(pendulums, T):
		
	pcount = len(pendulums)
	
	cm = mpl.colormaps['plasma'].resampled(pcount)

	artists = []

	idx = np.round(np.linspace(0, T / dt, T * 60).astype(int)).astype(int)

	fig, ax = plt.subplots()

	for i in idx:

		plot = []

		j = 0

		for p in pendulums:

			if i >= len(p.theta):
				i = len(p.theta) - 1

			l = [0, p.l[0], p.l[1]]

			x = l * np.array([0, np.sin(p.theta[i][0]), l[1] / l[2] * np.sin(p.theta[i][0]) + np.sin(p.theta[i][1])])
			y = l * np.array([0, -np.cos(p.theta[i][0]), - l[1] / l[2] * np.cos(p.theta[i][0]) - np.cos(p.theta[i][1])])
			
			plot += ax.plot(x, y, color = cm(j / pcount))

			j += 1

		artists.append(plot)

	animation = anim.ArtistAnimation(fig = fig, artists = artists, interval = 60)

	plt.show()




#generate a pendulum, print its harmonic time, and plot its time evolution:

#p = Pendulum(1, 1, 1, 1, np.pi/2, np.pi/3, 0, 0, 10)
#p.PrintData()

AnimatePendulums([Pendulum(10, 10, 1, 1, np.pi , i,  0, 0, 10) for i in np.linspace(-np.pi/12, np.pi/12, num = 20)], 10)



plt.show()