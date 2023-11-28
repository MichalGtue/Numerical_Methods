import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import diags
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.cm as cm

from laplace_demo import create_laplace_coefficient_matrix, set_boundary_conditions

Nx,Ny = 200,200

Tb = {'bottom':100, 'top':500, 'left': 600, 'right':-20}


A, b = create_laplace_coefficient_matrix(Nx, Ny)

A, b =  set_boundary_conditions(A,b, Tb, Nx, Ny)

T = spsolve(A,b).reshape(Nx,Ny)

x, y = np.meshgrid(np.linspace(0,1,Nx), np.linspace(0,1,Ny))
fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
surf = ax.plot_surface(x,y,T.reshape(Nx,Ny), cmap=cm.inferno,edgecolor='black')
fig.colorbar(surf, shrink=0.5)
plt.show()

