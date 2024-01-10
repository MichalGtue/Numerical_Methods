import numpy as np
import matplotlib.pyplot as plt

# Getting all the variables
Nx = 100 ## Grid points
Nt = 1000 # Time steps
diam = 0.04 # Diameter
phiv = 6.3e-4 #volumetric flow rate
D = 1e-3
cL = 1 # Inlet conc
cR = 0 
t_end = 4 # 4 sec
x_end = 1 # Length
kR = 1.2 

u = phiv / (np.pi*(diam/2)**2) 

dt = t_end / Nt
dx  = x_end / Nx

#Fourier Number
Fo = D * dt / dx**2
#print(Fo)
#Uncomment above line to verify that the Fourier number is less than 0.5

# Courant number
Co = u* dt /dx
#print(Co)
#Uncomment to see that Co is less than 1

x = np.linspace(0,x_end, Nx+1)
c = np.zeros(Nx+1)

c[0] = cL
c[Nx] = cR
t = 0


def reaction(c):
    return -kR*c

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

for n in range(Nt):
    t += dt
    c_old = np.copy(c) ## doing c_old = c then modification of c_old will also modify c
    c[0] = cL

    rxn = reaction(c_old)
    for i in range(1, Nx):
        c[i] = c_old[i] + Fo *c_old[i-1] -2*Fo*c_old[i] + Fo*c_old[i+1] - Co*0.5*(c_old[i]-c_old[i-1]) + rxn[i]*dt
    c[Nx] = c[Nx-1]

    iplot += 1
    if (iplot % 10 == 0):
        plt.title(f"Time = {t:5.3f} s")
        line[0].set_ydata(c)
        figure.canvas.draw()
        figure.canvas.flush_events()
        iplot = 0


plt.show()


