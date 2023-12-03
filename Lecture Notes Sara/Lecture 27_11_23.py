import numpy as np
from scipy.sparse import csr_matrix

A = np.eye(10000)
print(A.nbytes)

S = csr_matrix(A)
print(S.data.nbytes)

from scipy.sparse import diags
Nx, Ny = 5.5
Nc = Nx*Ny 
e=np.ones(Nc)
print(e)

import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as np

A= diags([e,e,-4*e,e,e], [-Nx,-1,0,1,Nx], shape=(Nc,Nc))

plt.spy(A, marker='o', markersize=6)
print(A)

print(np.arange(Nc), reshape(5,5)[::-1])

T_bottom = np_arrange(Nx)
print(T_bottom)

T_top=T_bottom+(Ny-1)*Nx
print(T_top)

T_left =np.arange(Ny)*(Nx)
print(T_left)

T_right = T_left+Nx-1
print(T_right)

bnd_all=np.unique(np.concatenate((T_bottom, T_top, T_left, T_right)))
print (bnd_all)
#np.unique removes repeated elements when combining functions
from scipy.sparse import lil_matrix
A=lil_matrix(A)
A[bnd_all, :]=0
A[bnd_all, bnd_all]=1
plt.spy(A,marker='o', markersize=5)

b=np.zeros(Nc)
b[T_bottom]=10
b[T_top]=20
b[T_left]=30
b[T_right]=40
plt.spy(b[:, None], marker='o')

from scipy.sparse.linalg import spsolve
T=spsolve(A.tocsc(),b)
print(T.reshape(Nx,Ny)[::-1])

x,y=np.meshgrid(np.linspace(0,1,Nx), np.linspace(0,1,Ny))
fig,ax=plt.subplots(subplot_kw={"projection":"3d"})
ax.plot_surface(x,y,T.reshape(Nx, Ny))