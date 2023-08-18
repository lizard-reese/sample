import numpy as np
import numpy.random as rand

import matplotlib.pyplot as plt

n = 50

sample = np.arange(0, n, 1)
rand.shuffle(sample)

print(sample)

def BubbleSort(sample, printSorts = False):

	while not np.all(sample[:-1] <= sample[1:]):

		for i in range(len(sample) - 1):

			if sample[i] > sample[i+1]:
				sample[i], sample[i+1] = sample[i+1], sample[i]

		if printSorts:
			print(sample)

	return(sample)

BubbleSort(sample, printSorts = True)



