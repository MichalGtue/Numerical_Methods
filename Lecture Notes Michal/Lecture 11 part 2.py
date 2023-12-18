## Solving PDEs 
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.sparse.linalg as spss
Nt = 40000
Nx = 100
D = 1e-8
cL = 1
cR = 0
t_end = 5000
x_end = 5e-3

dt = t_end/Nt
dx = x_end/Nx
Fo = (D*dt)/(dx**2)

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


A = sps.diags([-Fo, 1+2*Fo, -Fo], [-1, 0, 1], shape=(Nx+1, Nx+1))
A = sps.csr_matrix(A)
A[0,0]=1
A[0,1] = 0
A[Nx, Nx] = 1
A[Nx, Nx-1] = -1



for n in range(Nt):
    t += dt
    c_old = np.copy(c) ## doing c_old = c then modification of c_old will also modify c

    b = c_old
    b[Nx] = 0
    c = spss.spsolve(A, b)

    iplot += 1
    if (iplot % 100 == 0):
        plt.title(f"Time = {t:5.3f} s")
        line[0].set_ydata(c)
        figure.canvas.draw()
        figure.canvas.flush_events()
        iplot = 0


plt.pause()
plt.show()


