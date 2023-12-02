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

    return y

#Taken from it_methods.py
def jacobi(A, b, tol=1e-2):
    
    # Set initial guess
    x = b + 1e-16
        
    # Initialize variables
    x_diff = 1
    N = A.shape[0]
    it_jac = 1
    
    # While not converged or max_it not reached
    while (x_diff > tol and it_jac < 1000):
        x_old = x.copy()
        for i in range(N):
            s = 0
            for j in range(N):
                if j != i:
                    # Sum off-diagonal*x_old
                    s += A[i,j] * x_old[j]
            # Compute new x value
            x[i] = (b[i] - s) / A[i,i]
            
        # Increase number of iterations
        it_jac += 1
        x_diff = np.linalg.norm(A@x - b)/np.linalg.norm(b)
        
    # Print number of iterations
    #print(it_jac)
    
    return x, it_jac

def jacobi_vec(A, b, tol=1e-2, itmax=1000): 
    # Set initial guess
    x = b + 1e-16
        
    # Initialize variables
    x_diff = 1
    N = A.shape[0]
    it_jac = 1
    
    # While not converged or max_it not reached
    while (x_diff > tol and it_jac < 1000):
        x_old = x.copy()
        for i in range(N):
            s = 0
            j_indices = np.concatenate((np.arange(0,i), np.arange(i+1, N)))
            s += A[i,j_indices] @  x_old[j_indices]
            # Compute new x value
            x[i] = (b[i] - s) / A[i,i]
            
        # Increase number of iterations
        it_jac += 1
        x_diff = np.linalg.norm(A@x - b)/np.linalg.norm(b)
        
    # Print number of iterations
    #print(it_jac)
    
    return x, it_jac

def gaussseidel(A, b, tol=1e-2):
    # Set initial guess
    x = b + 1e-16
        
    # Initialize variables
    x_diff = 1
    N = A.shape[0]
    it_gaussseidel = 1
    
    # While not converged or max_it not reached
    while (x_diff > tol and it_gaussseidel < 1000):
        x_old = x.copy()
        for i in range(N):
            s = 0
            s2 = 0
            for j in range(N):
                if j < i:
                    # Sum off-diagonal*x_old
                    s += A[i,j] * x[j]
                if j > i:
                    # Second summation
                    s2 += A[i,j] * x_old[j]
            # Compute new x value
            x[i] = (b[i] - s - s2) / A[i,i]
            
        # Increase number of iterations
        it_gaussseidel += 1
        x_diff = np.linalg.norm(A@x - b)/np.linalg.norm(b)
        
    # Print number of iterations
    #print(it_gaussseidel)
    
    return x, it_gaussseidel

def gaussseidel_vec(A, b, tol=1e-2):
    # Set initial guess
    x = b + 1e-16
        
    # Initialize variables
    x_diff = 1
    N = A.shape[0]
    it_gaussseidel = 1
    
    # While not converged or max_it not reached
    while (x_diff > tol and it_gaussseidel < 1000):
        x_old = x.copy()
        for i in range(N):
            s = 0
            s2 = 0
            j_indices_new = np.arange(0,i)
            j_indices_old = np.arange(i+1, N)
            s += A[i,j_indices_new] @  x[j_indices_new]
            s2 += A[i,j_indices_old] @  x_old[j_indices_old]
            # Compute new x value
            x[i] = (b[i] - s - s2) / A[i,i]
            
        # Increase number of iterations
        it_gaussseidel += 1
        x_diff = np.linalg.norm(A@x - b)/np.linalg.norm(b)
        
    # Print number of iterations
    #print(it_gaussseidel)
    
    return x, it_gaussseidel

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



start_time_lu = time.time()
P, L, U = lu(M) 
d = P @ sol_vec
y= forwardsubstitution(L, d)
x = backsubstitution_v1(U, y)
end_time_lu = time.time() - start_time_lu


#Jacobi (Function taken from lecture 5)
# First the non-vectorized
start_time_Jacobi = time.time()
jacobi_sol, jacobi_number_of_iterations = jacobi(M, sol_vec)
end_time_Jacobi = time.time() - start_time_Jacobi


#Then the vectorized
start_time_Jacobi_vec = time.time()
jacobi_sol_vec, jacobi_number_of_iterations_vec = jacobi_vec(M, sol_vec)
end_time_Jacobi_vec = time.time() - start_time_Jacobi_vec

#Gauss-Seidel
#First the non-vectorized

start_time_gaussseidel = time.time()
gaussseidel_sol, gaussseidel_number_of_iterations = gaussseidel(M, sol_vec)
end_time_gaussseidel = time.time() - start_time_gaussseidel




#Then the vectorized
start_time_gaussseidel_vec = time.time()
gaussseidel_sol_vec, gaussseidel_number_of_iterations_vec = gaussseidel_vec(M, sol_vec)
end_time_gaussseidel_vec = time.time() - start_time_gaussseidel_vec



#print(f"Time for various solvers given in seconds, np.solve:{Time_taken_solver}, Gaussian elimination:{Time_taken_gauss}, LU decomposition:{end_time_lu}, Jacobi (not vectorized):{end_time_Jacobi}, Jacobi (Vectorized):{end_time_gaussseidel_vec}, Gauss-Seidel (not vectorized):{end_time_gaussseidel}, Gauss-seidel (vectorized):{end_time_gaussseidel_vec}")