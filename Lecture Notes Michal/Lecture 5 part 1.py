import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import diags
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


A = np.eye(10000)



print(A.nbytes/1024/1024)

s = csr_matrix(A)

print(s.data.nbytes)

## Second derivative look at lecture slides slide 17


Nx, Ny = 5, 5

Nc = Nx * Ny 

e = np.ones(Nc)

print(e)

A = diags([e,e,-4*e,e,e], [-Nx, -1, 0, 1, Nx], shape=(Nc, Nc))

#plt.spy(A)

#plt.show()

print(A)

print(np.arange(Nc).reshape(5,5)[::-1])

T_bottom = np.arange(Nx)

print(T_bottom)

T_top = T_bottom + (Ny-1)*Nx

print(T_top)

T_left =  np.arange(Ny)*Nx

T_right = T_left + Nx-1

bnd_all = np.unique(np.concatenate((T_bottom, T_top, T_left, T_right)))

print(bnd_all)

A=lil_matrix(A)

A[bnd_all,:] = 0

A[bnd_all,bnd_all] = 1

print(A)

#plt.spy(A)

#plt.show()

b = np.zeros(Nc)
b[T_bottom] = 10
b[T_top] = 20
b[T_left] = 30
b[T_right] = 40

#plt.spy(b[:, None], marker='o')
#plt.show()

T=spsolve(A.tocsc(),b)

print(T.reshape(Nx,Ny)[::-1])

x, y = np.meshgrid(np.linspace(0,1,Nx), np.linspace(0,1,Ny))
fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
ax.plot_surface(x,y,T.reshape(Nx,Ny))

plt.show()