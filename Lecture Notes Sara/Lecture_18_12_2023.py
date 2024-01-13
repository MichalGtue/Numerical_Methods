## Solving PDEs 
import numpy as np
import matplotlib.pyplot as plt
Nx = 100 # Nx grid points
Nt = 40000 # Nt time steps
D = 1e-8 # m/s
cL = 1.0; c_R = 0 # mol/m3
cR=0
t_end = 5000.0 # s
x_end = 5e-3 # m

# Time step and grid size
dt = t_end / Nt
dx = x_end / Nx

Fo = (D*dt)/(dx**2)
print(f"dt: {dt} s, dx:{dx} m, Fo: {Fo}")
# Grid node and time step positions
x = np.linspace(0, x_end, Nx + 1)
# Initial matrices for solutions (Nx times Nt)
c = np.zeros((Nt + 1, Nx + 1)) # All concentrations are zero
c[:, 0] = c_L # Concentration at the left side
c[:, Nx] = c_R # Concentration at the right side
#t=np.linspace(0, t_end, Nt+1)
t=0

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

A=sps.diags([-Fo, 1+2*Fo, -Fo], [-1,0,1], shape=(Nx+1, Nx+1))
A[0,0]=1
A[0,1]=0
A[Nx, Nx]=1
A[Nx, Nx-1]=-1
A=sps.csr_matrix(4)

for n in range(Nt):
    t += dt
    c_old = np.copy(c) ## doing c_old = c then modification of c_old will also modify c
    b=c_old
    b[Nx]=0
    c[0] = cL
    for i in range(1, Nx):
        c[i] = Fo *c_old[i-1] + (1-2*Fo)*c_old[i] + Fo*c_old[i+1]
    c[Nx] = cR

    iplot += 1
    if (iplot % 1000 == 0):
        plt.title(f"Time = {t:5.3f} s")
        line[0].set_ydata(c)
        figure.canvas.draw()
        figure.canvas.flush_events()
        iplot = 0


#plt.pause(1)
plt.show()


#explicit convection

