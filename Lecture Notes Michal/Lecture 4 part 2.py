from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import time

M = np.array([[5,3,2],[0,9,1],[0,0,1]])
print(np.linalg.matrix_rank(M))
b = np.array([4,7,5])

Maug = np.c_[M,b]

#print(Maug)

#print(np.linalg.matrix_rank(Maug))

### Gaus elimination


A = np.array([[1,1,1],[2,1,3],[3,1,6]])

b = np.array([4,7,5])


d10 = A[1,0] / A[0,0]

A[1,0] = A[1,0] - (d10 * A[0,0])
A[1,1] = A[1,1] -( d10 * A[0,1])
A[1,2] = A[1,2] - (d10 * A[0,2])
b[1] = b[1] - (d10 *b[0])

d20 = A[2,0] / A[0,0]

A[2,0] = A[2,0] - d20 * A[0,0]
A[2,1] = A[2,1] - d20 * A[0,1]
A[2,2] = A[2,2] - d20 * A[0,2]
b[2] =   b[2] -   d20 *b[0]

d21 = A[2,1] / A[1,1]

A[2,1] = A[2,1] - d21 * A[1,1]
A[2,2] = A[2,2] -d21 * A[1,2]
b[2] = b[2] -d21 * b[1]
print(A, b)

from gaussjordan import gaussian_eliminate_draft as ge

Aprime, bprime = ge(A, b)

print(Aprime, bprime)


Abig = np.random.rand(500,500)
bbig = np.random.rand(500)

start = time.time()
#ge(Abig, bbig)

#print(time.time() - start)

from gaussjordan import gaussian_eliminate_v1 as ge1

Abig = np.random.rand(500,500)
bbig = np.random.rand(500)

start = time.time()
Aprime, bprime = ge1(A, b)

print(time.time() - start)
from gaussjordan import backsubstitution_draft as bs

x = bs(Aprime, bprime)

print(x)