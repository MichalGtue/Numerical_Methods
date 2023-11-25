import numpy as np

A=np.array([[1,1,1],[2,1,3],[3,1,6]])
print(A)

b=np.array([4,7,5])
print(b)

ainv = np.linalg.inv(A)
print(ainv)

ainv@b
x=np.linalg.solve(A,b)

#exercise slide 21: Create a script that generates matrices with random elements of various sizes N × N (e.g. values of N ∈ {10, 20, 50, 100, 200, . . . , 5000, 10000}). Compute the inverse of each matrix, and use tic and toc to see the computing time for each inversion. Plot the time as a function of the matrix size N.

#def generate_ra




plt.plot(sizes, total_time, 'r-o')
plt.xlabel('Matrix size')
plt.ylabel('')




M=np.array([[5,3,2], [0,9,1], [0,0,1]])
np.linalg.matrix_rank(M)
Maug=np.column_stack((M,b)
print(Maug))



#lecture 2 
A=np.array([[1,1,1],[2,1,3],[3,1,6]])
b=np.array([4,7,5])

d10=A[1,0]/A[0,0]
print(f'elimination factor (d10)')
A[1,0]=A[1,0]-d*A[0,0]
A[1,1]=A[1,1]-d*A[0,1]
A[1,2]=A[1,2]-d*A[0,2]
b[1]=b[1]-d*b[1]
print(A)
print(b)

A=np.array([[1,1,1],[],[3,1,6]])
b=np.array([4,7,5])

d20=A[2,0]/A[0,0]
print(f'elimination factor (d20)')
A[2,0]=A[2,0]-d20*A[0,0]
A[2,1]=A[2,1]-d20*A[2,0]
A[2,2]=A[2,2]-d20*A[2,1]
B[1]=B[1]-d20*B[0]
print(A)
print(B)

d21=A[2,1]/A[0,0]
print(f'elimination factor (d20)')
A[2,0]=A[2,0]-d20*A[0,0]
A[2,1]=A[2,1]-d20*A[2,0]
A[2,2]=A[2,2]-d20*A[2,1]
B[1]=B[1]-d20*B[0]
print(A)
print(B)

def gaussian_eliminate_draft(A,b):
"""Perform elimination to obtain an upper triangular matrix"""
A = np.array(A,dtype=np.float64)
b = np.array(b,dtype=np.float64)

assert A.shape[0] == A.shape[1], "Coefficient matrix should be square"

N = len(b)
for col in range(N-1): # Select pivot
for row in range(col+1,N): # Loop over rows below pivot
d = A[row,col] / A[col,col] # Choose elimination factor
for element in range(row,N): # Elements from diagonal to right
A[row,element] = A[row,element] - d * A[col,element]
b[row] = b[row] - d * b[col]
return A,b

from gaussjordan import gaussian_eliminate_draft as ge
A=np.array([[1,1,1], [2,1,3], [3,1,6]])
b=np.array([4,7,5])

Aprime, bprime=getattr(A,b)

