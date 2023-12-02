#Libraries
import numpy as np
import time
from scipy.linalg import lu 


# Functions taken from gaussjordan.py (Given in lecture 4)
def swap_rows(mat,i1,i2):
    """Swap two rows in a matrix/vector"""
    temp = mat[i1,...].copy()
    mat[i1,...] = mat[i2,...]
    mat[i2,...] = temp

def gaussian_eliminate_v2(A,b):
    """Perform elimination to obtain an upper triangular matrix
    
    Input:
    A: Coefficient matrix
    b: right hand side
    
    Returns:
    Aprime, bprime: row echelon form of matrix A and rhs vector b"""
    A = np.array(A,dtype=np.float64)
    b = np.array(b,dtype=np.float64)

    assert A.shape[0] == A.shape[1], "Coefficient matrix should be square"

    N = len(b)
    for col in range(N-1):
        index = np.argmax(np.abs(A[col:, col])) + col
        swap_rows(A,col,index)
        swap_rows(b,col,index)
        for row in range(col+1,N):
            d = A[row,col] / A[col,col]
            A[row,:] = A[row,:] - d * A[col,:]
            b[row] = b[row] - d * b[col]

    return A,b

def backsubstitution_v1(U,b):
    """Back substitutes an upper triangular matrix to find x in Ax=b"""
    x = np.empty_like(b)
    N = len(b)
    
    for row in range(N)[::-1]:
        x[row] = (b[row] - np.sum(U[row,row+1:] * x[row+1:])) / U[row,row]

    return x

def forwardsubstitution(L,d):
    N = len(L)
    y = np.empty_like(d)

    for row in range(N):
        y[row] = (d[row] - np.sum(L[row,:row] * y[:row])) / L[row,row]

#Define the values of the Q's (given in km^-3 / yr)
Q_SH = 72
Q_MH = 38
Q_HE = 160
Q_EO = 185
Q_OO = 215

#Define values for the different PCB source S_in's (given in kg/yr)

S_in   = 180
S_in_M = 810
S_in_H = 630
S_in_E = 2750
S_in_O = 3820

#Generating our coefficient matrix
M = np.array([[Q_SH,0,0,0,0], [0,Q_MH,0,0,0], [-Q_SH, -Q_MH, Q_HE, 0, 0], [0,0,-Q_HE, Q_EO, 0], [0,0,0,-Q_EO,Q_OO]])
#print(M) #Uncomment to see the coefficient matrix



#Solution Vector
sol_vec = np.array([S_in, S_in_M, S_in_H,S_in_E, S_in_O])
#print(sol_vec) #Uncomment to see the Solution Vector



#Using a normal the solver from numpy
start_time_solver = time.time() # Start time
np.linalg.solve(M,sol_vec)
Time_taken_solver = time.time() - start_time_solver # Difference in times to get time taken



#Gaussian Elimination
#Using code provided in Lecture 4 (gaussjordan.py)

start_time_gauss = time.time()
Gauss_M, Gauss_SolVec = gaussian_eliminate_v2(M,sol_vec)
backsubstitution_v1(Gauss_M, Gauss_SolVec)
Time_taken_gauss = time.time() - start_time_gauss

#LU decomposition

