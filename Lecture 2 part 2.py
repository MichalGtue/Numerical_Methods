import numpy as np


arr = np.array([1, 2, 3, 4, 5])

#print(arr[0])

#print(np.zeros(5))

#print (np.ones((3, 3)))

import random as random

#print(np.random.random((3,3))) # Graded assignment

#print(np.arange(0,12,2))

#print(np.linspace(0, np.pi, 10))

mat = np.linspace(1, 25, 25)

mat = np.linspace(1, 25, 25).reshape(5, 5)
mat[1,:] = -1

mat[mat>11] = -55

#print (mat)

import time

start = time.time()
x = np.linspace(0, 2*np.pi,1)
y = np.exp(-x) * (2+np.sin(2*np.pi*x))
total_time = time.time() - start

print(f'{total_time = }')


