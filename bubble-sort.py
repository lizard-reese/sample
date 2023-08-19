import numpy as np
import numpy.random as rand

import time
import timeit

import matplotlib.pyplot as plt

N = 128
M = 16

def BubbleSort(n, printSorts = False):
	A = np.arange(n)
	rand.shuffle(A)

	st = time.time()

	while not np.all(A[:-1] <= A[1:]):

		for i in range(len(A) - 1):

			if A[i] > A[i+1]:
				A[i], A[i+1] = A[i+1], A[i]

		if printSorts:
			print(A)

	et = time.time()
	rt = et-st
	
	return(rt*1000)


fig, ax = plt.subplots()

domain = np.arange(N)
avg = []

plt.ylabel('Runtime (ms)')
plt.xlabel('Array Size')

for n in domain:
	data = []
	
	for m in range (M):
		data.append(BubbleSort(n))
		
	avg.append(np.average(data))

regression = np.poly1d(np.polyfit(domain, avg, 2))

plt.plot(domain, regression(domain), '-', color = (0.1, 0.3, 0.7))
plt.plot(domain, avg, 'b.', color = (0.1, 0.3, 0.7))

plt.show()



