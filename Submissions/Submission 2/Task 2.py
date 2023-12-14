#Libraries
import numpy as np
import timeit
from scipy.linalg import lu 
import pandas as pd

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


#task 2 matrix
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

#Volumes (km^3)

V_S = 12000
V_M = 4900
V_H = 3500
V_E = 480
V_O = 1640



#Generating our coefficient matrix
M = np.array([[Q_SH,0,0,0,0], [0,Q_MH,0,0,0], [-Q_SH, -Q_MH, Q_HE, 0, 0], [0,0,-Q_HE, Q_EO , 0], [0,0,0,-Q_EO,Q_OO]])
#print(M) #Uncomment to see the coefficient matrix


#Solution Vector
sol_vec = np.array([S_in, S_in_M, S_in_H,S_in_E, S_in_O])
#print(sol_vec) #Uncomment to see the Solution Vector



#Using a normal the solver from numpy

solver_code = lambda: np.linalg.solve(M,sol_vec)
Time_taken_solver = timeit.timeit(solver_code, number=1)


#Gaussian Elimination
#Using code provided in Lecture 4 (gaussjordan.py)

Time_taken_gauss = timeit.timeit(lambda: backsubstitution_v1(*gaussian_eliminate_v2(M, sol_vec)), number=1) # * asterisk used to get the outputs of the gauss_eliminate_v2 function

#LU decomposition



def LU_function_for_timing():
    '''Function only written for task 2 for timing LU decomposition. This function only serves one purpose.'''
    P, L, U = lu(M) 
    d = P @ sol_vec
    y= forwardsubstitution(L, d)
    lu_function = lambda: backsubstitution_v1( U,y) # time.time() prints out 0.0 otherwise
end_time_lu = timeit.timeit(LU_function_for_timing ,number=1)

#Jacobi (Function taken from lecture 5)
# First the non-vectorized


jacobi_code = lambda:  jacobi(M, sol_vec)
end_time_Jacobi = timeit.timeit(jacobi_code, number=1)

jacobi_sol, jacobi_number_of_iterations = jacobi(M, sol_vec) #Yes code is ran twice but it runs so quickly that it doesnt matter. Plus again here using time.time() would print 0.0

#Then the vectorized

jacobi_sol_vec, jacobi_number_of_iterations_vec = jacobi_vec(M, sol_vec)

Jacobi_vec_code = lambda: jacobi_vec(M, sol_vec)
end_time_Jacobi_vec = timeit.timeit(Jacobi_vec_code, number=1)

#Gauss-Seidel
#First the non-vectorized


gaussseidel_sol, gaussseidel_number_of_iterations = gaussseidel(M, sol_vec)

gaussseidel_function = lambda:gaussseidel(M,sol_vec)
end_time_gaussseidel = timeit.timeit(gaussseidel_function, number=1)




#Then the vectorized

gaussseidel_sol_vec, gaussseidel_number_of_iterations_vec = gaussseidel_vec(M, sol_vec)

gaussseidel_function_vec = lambda:gaussseidel_vec(M, sol_vec)
end_time_gaussseidel_vec = timeit.timeit(gaussseidel_function_vec, number=1)



#Uncomment to see the output as as text
#print(f"Time for various solvers given in seconds, np.solve:{Time_taken_solver}, Gaussian elimination:{Time_taken_gauss}, LU decomposition:{end_time_lu}, Jacobi (not vectorized):{end_time_Jacobi} with {jacobi_number_of_iterations} iterations, Jacobi (Vectorized):{end_time_gaussseidel_vec} with {jacobi_number_of_iterations_vec} iterations, Gauss-Seidel (not vectorized):{end_time_gaussseidel} with {gaussseidel_number_of_iterations} iterations, Gauss-seidel (vectorized):{end_time_gaussseidel_vec} with {gaussseidel_number_of_iterations_vec} iterations")

#To generate table
#Columns

def accuracy_checker(A,B):
    equality = False
    if np.array_equal(A,B):
        equality = True
        return equality
    else:
        return equality





rows = [['np.linalg.solve', Time_taken_solver, 'N/A', 'N/A'], ['Gaussian Elimination', Time_taken_gauss, 'N/A', 'N/A'], ['LU Decomposition', end_time_lu, 'N/A', 'N/A'], ['Jacobi not Vectorized', end_time_Jacobi, jacobi_number_of_iterations, accuracy_checker(jacobi(M, sol_vec)[0], np.linalg.solve(M,sol_vec))], ['Jacobi Vectorized', end_time_Jacobi_vec, jacobi_number_of_iterations_vec, accuracy_checker(jacobi_vec(M, sol_vec)[0], np.linalg.solve(M,sol_vec))], ['Gauss-Seidel not Vectorized', end_time_gaussseidel, gaussseidel_number_of_iterations, accuracy_checker(gaussseidel(M, sol_vec)[0], np.linalg.solve(M,sol_vec))], ['Gauss-Seidel Vectorized', end_time_gaussseidel_vec, gaussseidel_number_of_iterations_vec, accuracy_checker(gaussseidel_vec(M, sol_vec)[0], np.linalg.solve(M,sol_vec))]]


df = pd.DataFrame(rows, columns = ['Solution method', 'Time taken (s)', 'Number of iterations', 'Accuracy'])
df['Time taken (s)'] = df['Time taken (s)'].apply(lambda x: f'{x:.11f}')

print(df)