import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import diags
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.cm as cm
from it_methods import jacobi, jacobi_vec
from laplace_demo import create_laplace_coefficient_matrix, set_boundary_conditions
import time



Nx,Ny = 20,20

Tb = {'bottom':10, 'top':20, 'left': 200, 'right':500}


A, b = create_laplace_coefficient_matrix(Nx, Ny)

A, b =  set_boundary_conditions(A,b, Tb, Nx, Ny)

if not isinstance(A, np.ndarray):
    A = A.toarray()

#T = spsolve(A,b).reshape(Nx,Ny)

start = time.time()
T = jacobi_vec(A,b)[0].reshape(Nx,Ny)
End = time.time()
print(f'Time to compute {End-start}')

x, y = np.meshgrid(np.linspace(0,1,Nx), np.linspace(0,1,Ny))


#T_ex = np.zeros_like(x)
#
#for n in range(1,120):
#    m = 2*n-1
#    term = np.sin(m*np.pi*x) * np.sinh(m*np.pi*y)/(m*np.sinh(m*np.pi))
#    T_ex += term
#
#T_ex *= (4/np.pi)

fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
surf = ax.plot_surface(x,y,T, cmap=cm.inferno,edgecolor='black')
fig.colorbar(surf, shrink=0.5)
plt.show()

