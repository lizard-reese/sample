import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt

PI = 3.1415926535

class Function:
	def __init__(self, function, dimension, domain, plotCurve = False):
		self.plot = plotCurve
		
		self.f = function
		self.n = dimension
		self.domain = domain

		self.space = []
		self.length = []
		self.offset = []

		self.vol = 1

		for i in domain:
			self.space.append(np.linspace(i[0], i[1], num = 64))

			self.length.append(i[1] - i[0])
			self.offset.append(i[0])

			self.vol *= i[1] - i[0]

		if dimension == 1:
			self.z = function(self.space[0])
		else: 
			self.z = function(self.space)

		self.maximum = np.max(self.z)
		self.minimum = np.min(self.z)
		self.range = self.maximum - self.minimum

		self.vol *= self.range

		if self.plot:
			if dimension == 1 :
				self.fig, self.ax = plt.subplots()
				self.ax.plot(self.space[0], self.z)

				plt.show()
			if dimension == 2 :
				self.fig = plt.figure()
				self.ax = self.fig.add_subplot(projection='3d')
				self.ax.plot_surface(self.space[0],self.space[1],self.z)

				plt.show()

	def MCIntegrate(self, samples, runs, plotData = False):

		values = []

		while runs > 0:
			inSample = self.length * np.random.random_sample((samples, self.n)) + self.offset
			outSample = self.range * np.random.random_sample(samples) + self.minimum

			count = 0

			for i in range(0, samples - 1):
			
				if (outSample[i] < self.f(inSample[i])):

					count += 1

			values.append(self.vol * count / samples + self.minimum)

			runs -= 1

		if plotData:
			counts, bins = np.histogram(values, bins=15)
			plt.stairs(counts, bins)

			plt.show()

		return np.average(values)

#fn1 = Function(lambda x : np.sin(x[1]) + np.cos(x[0]), 2, [(-PI, PI) for i in range(2)])
#fn2 = Function(lambda x : (x[9] + x[8] +x[7] + x[6] + x[5] + x[4] + x[3] + x[2] + x[1] + x[0])**2, 10, [(0, 1) for i in range(10)])

#print("average: " + str(fn1.MCIntegrate(64, 64, plotData = True)))
#print("average: " + str(fn2.MCIntegrate(64, 64)))
	
plt.show()
