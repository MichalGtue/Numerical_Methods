import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.sparse.linalg as spss



# Getting all the variables
Nx = 100 ## Grid points
Nt = 10000 # Time steps
diam = 0.04 # Diameter
phiv = 6.3e-4 #volumetric flow rate
phivnit = 11.2e-4 #volumetric flow rate nitrogen
D = 1e-3
cL = 1 # Inlet conc
cR = 0 
t_end = 10 # 4 sec
x_end = 1 # Length
kR = 1.2 

u = phiv / (np.pi*(diam/2)**2) 
u_nit = phivnit / (np.pi*(diam/2)**2) 

dt = t_end / Nt
dx  = x_end / Nx

#Fourier Number
Fo = D * dt / dx**2
#print(Fo)
#Uncomment above line to verify that the Fourier number is less than 0.5

# Courant number
Co = u* dt /dx
Co_nit = u_nit*dt/dx
#print(Co)
#print(Co_nit)
#Uncomment to see that Co and Co_nit is less than 1

x = np.linspace(0,x_end, Nx+1)
c = np.zeros(Nx+1)

c[0] = cL
c[Nx] = cR
t = 0



plt.ion()
figure, ax = plt.subplots(figsize = (10, 5))
line = []
line += ax.plot(x, c)
plt.title(f"Time: {t:5.3f} s", fontsize = 16)
plt.xlabel('Axial position', fontsize = 14)
plt.ylabel('Concentration', fontsize=14)
plt.xlim(0, x_end)
plt.ylim(0, max(cL, cR))
plt.grid
iplot = 0


A = sps.diags([-(Co+Fo), (1+Co+2*Fo+kR*dt), -Fo], [-1, 0, 1], shape=(Nx+1, Nx+1))
A = sps.csr_matrix(A)
A[0,0]=1
A[0,1] = 0
A[Nx, Nx] = 1
A[Nx, Nx-1] = -1
for i in range(1, Nx):
    if dx*i < 0.1 or dx*i>0.9:
        A[i, i] = A[i,i] - kR*dt ## Taking away the reaction term  

A_new = sps.diags([-(Co_nit+Fo), (1+Co_nit+2*Fo+kR*dt), -Fo], [-1, 0, 1], shape=(Nx+1, Nx+1))
A_new = sps.csr_matrix(A_new)
A_new[0,0]=1
A_new[0,1] = 0
A_new[Nx, Nx] = 1
A_new[Nx, Nx-1] = -1
for i in range(1, Nx):
    if dx*i < 0.1 or dx*i>0.9:
        A_new[i, i] = A_new[i,i] - kR*dt
dense_A = A_new.toarray()
mirrored_A_not_sparse = dense_A[::-1, :] ## Flipping the matrix along the 'x' axis
mirrored_sparse_A = sps.csr_matrix(mirrored_A_not_sparse)


for n in range(Nt):
    t += dt
    c_old = np.copy(c) ## doing c_old = c then modification of c_old will also modify c
    c[0] = cL

    b = c_old
    b[Nx] = 0
    if n*dt<5:
        c = spss.spsolve(A, b)
    else:
        c = spss.spsolve(mirrored_sparse_A,b)
    iplot += 1
    if (iplot % 10 == 0):
        plt.title(f"Time = {t:5.3f} s")
        line[0].set_ydata(c)
        figure.canvas.draw()
        figure.canvas.flush_events()
        iplot = 0