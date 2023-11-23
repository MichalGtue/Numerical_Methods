import numpy as np
from scipy.linalg import lu 
#from functools import cache

#@cache
def gaussian_eliminate_draft(A,b):
    """Perform elimination to obtain an upper triangular matrix"""
    A = np.array(A,dtype=np.float64)
    b = np.array(b,dtype=np.float64)

    assert A.shape[0] == A.shape[1], "Coefficient matrix should be square"

    N = len(b)
    for col in range(N-1):                # Select pivot
        for row in range(col+1,N):        # Loop over rows below pivot
            d = A[row,col] / A[col,col]   # Choose elimination factor
            for element in range(col,N):  # Elements from diagonal to right
                A[row,element] = A[row,element] - d * A[col,element]
            b[row] = b[row] - d * b[col]

    return A,b

def gaussian_eliminate_v1(A,b):
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
        for row in range(col+1,N):
            d = A[row,col] / A[col,col]
            A[row,:] = A[row,:] - d * A[col,:]
            b[row] = b[row] - d * b[col]

    return A,b

def gaussian_eliminate_partial_pivot(A,b):
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
        temp = A[col,:]
        A[col,:] = A[index,:]
        A[index,:] = temp

        temp = b[col]
        b[col] = b[index]
        b[index] = temp
        for row in range(col+1,N):
            d = A[row,col] / A[col,col]
            A[row,:] = A[row,:] - d * A[col,:]
            b[row] = b[row] - d * b[col]

    return A,b

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

def backsubstitution_draft(A, b):
    """Back substitutes an upper triangular matrix to find x in Ax=b"""
    x = np.copy(b)
    N = len(b)

    for row in range(N-1, -1, -1):
        for i in range(row+1, N):
            x[row] = x[row] - A[row, i] * x[i]
        x[row] = x[row] / A[row, row]  
    
    return x

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

if __name__ == "__main__":

    A = np.array([[1, 1, 1], [2, 1, 3], [3, 1, 6]]) 
    b = np.array([4, 7, 5]) 

    Aprime,bprime = gaussian_eliminate_v2(A,b)
    x = backsubstitution_v1(Aprime,bprime)
  
    assert np.all(np.isclose(x, np.linalg.solve(A,b))), f"Test on simple matrix failed: x = {x}"
    

    # Example usage
    A = np.random.rand(5, 5) # Get random matrix
    A = np.array([[1, 1, 1], [2, 1, 3], [3, 1, 6]]) 

    P, L, U = lu(A)
    print(A,P,L,U,sep='\n')
    # Get L, U and P
    b = np.random.rand(5)
    b = np.array([4, 7, 5]) 

    # Random b vector
    d = P @ b
    # Permute b vector
    y = forwardsubstitution(L, d)
    # Can also do y=L\d
    x = backsubstitution_v1(U, y)
    # Can also do x=U\y
    rnorm = np.linalg.norm(A @ x - b) # Residual
    # for s in [10,50,500]:
    #     A = np.random.rand(s,s)
    #     b = np.random.rand(s,)
    #     Aprime,bprime = gaussian_eliminate_v2(A,b)
    #     x = backsubstitution_v1(Aprime,bprime)
    #     assert np.all(np.isclose(x, np.linalg.solve(A,b))), f"Test on random matrix failed: x = {np.column_stack((x,np.linalg.solve(A,b)))}"


