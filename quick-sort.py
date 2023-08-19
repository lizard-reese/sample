import numpy as np
import numpy.random as rand

import time

import matplotlib.pyplot as plt

N = 256
M = 16

def quicksort(n, m):
	arr = np.arange(n)
	rand.shuffle(arr)

	rt = []

	for i in range(m):
		st = time.time()

		sort(arr, 0, n - 1)

		et = time.time()
		rt.append(1000*(et - st))

		rand.shuffle(arr)

	avg = np.average(rt)

	print(avg)
	return avg
	
def sort(A, r, s):
	if r >= s or r < 0:
		return

	p = partition(A, r, s)

	sort(A, r, p - 1)
	sort(A, p + 1, s)


def partition(A, r, s):
	pivot = A[s]
	
	i = r - 1
	for j in range(r, s):
		if A[j] <= pivot:
			i += 1

			A[i], A[j] = A[j], A[i]
	
	i += 1

	A[i], A[s] = A[s], A[i]

	return i

fig, ax = plt.subplots()

domain = np.arange(N)
data = []

plt.ylabel('Runtime (ms)')
plt.xlabel('Array Size')


for n in domain:
	data.append(quicksort(n, M))


plt.plot(np.arange(N), data, 'b.', color = (0.1, 0.3, 0.7))

plt.show()



