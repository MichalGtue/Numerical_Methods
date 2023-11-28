import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags,lil_matrix,csr_matrix,csc_matrix
from matplotlib import cm
import matplotlib.pyplot as plt

def create_laplace_coefficient_matrix(Nx=50,Ny=50):
    """Creates a sparse matrix for the 2D Laplacian equation
    
    Input: 
        Nx, Ny: Number of grid nodes in x, y direction"""
    
    Nc = Nx*Ny    # Total number of points

    e = np.ones(Nc)
    A = diags([e, e, -4*e, e, e], [-Nx, -1, 0, 1, Nx], shape=(Nc,Nc))
    b = np.zeros(Nc)

    return A,b

def set_boundary_conditions(A, b, Tb, Nx, Ny):
    """Set explicit boundary conditions in a coefficient matrix and right hand size, using dictionary Tb indicating temperatures for left, right, top and bottom."""
    
    # Convert to lil_matrix to facilitate setting of coefficients
    A = lil_matrix(A)

    # Select nodes that lie at the boundaries
    bnd_bottom = np.arange(Nx)
    bnd_left = np.arange(Ny) * Nx
    bnd_right = bnd_left + Nx - 1
    bnd_top = bnd_bottom + Nx*(Ny-1)

    # Gather all nodes in a single array
    bnd_all = np.unique(np.concatenate((bnd_bottom,bnd_left,bnd_right,bnd_top)))

    # For all equations that represent a boundary, reset the coefficient row to zero
    # consequently add a 1 only on the main diagonal
    A[bnd_all,:] = 0
    A[bnd_all,bnd_all] = 1

    b[bnd_bottom] = Tb['bottom']
    b[bnd_left] = Tb['left']
    b[bnd_right] = Tb['right']
    b[bnd_top] = Tb['top']

    return A.tocsr(), b

if __name__=="__main__":
    Nx = Ny = 20 # number of internal grid cells over x/y-direction
    # h = 1/(Nx+1) # grid spacing in both x- and y-direction
    # k = 0.1 * h # heat conduction coefficient
    # q = 100 # heat flux over x = 1

    T_boundary = {'bottom': 300, 'left': 1000, 'right': 1000, 'top': 500}
    
    A,b = create_laplace_coefficient_matrix(Nx,Ny)
    A,b = set_boundary_conditions(A, b, T_boundary, Nx, Ny)

    T = spsolve(A,b).reshape((Nx,Ny))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x,y = np.meshgrid(np.arange(Nx),np.arange(Ny))
    surf = ax.plot_surface(x,y,T,cmap=cm.inferno,edgecolor='black',linewidth=0.2)
    fig.colorbar(surf, shrink=0.5)
    plt.tight_layout()
    plt.show()

