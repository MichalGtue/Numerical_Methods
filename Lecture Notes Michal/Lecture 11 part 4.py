## Solving PDEs 
import numpy as np
import matplotlib.pyplot as plt



Nt = 100000
Nx = 1000
u = 0.001
c_in = 1
t_end = 100
x_end = 0.1

dt = t_end/Nt
dx = x_end/Nx
Co = u*dt/dx




x=np.linspace(0,x_end, Nx+1)
c = np.zeros(Nx+1)
c[0] = c_in
t= 0




plt.ion()
figure, ax = plt.subplots(figsize = (10, 5))
line = []
line += ax.plot(x, c)
plt.title(f"Time: {t:5.3f} s", fontsize = 16)
plt.xlabel('Axial position', fontsize = 14)
plt.ylabel('Concentration', fontsize=14)
plt.xlim(0, x_end)
plt.grid
iplot = 0


for n in range(Nt):
    t +=dt
    c_old = np.copy(c)

    c[0] = c_in
    for i in range(1,Nx):
        c[i] = c_old[i] - Co*0.5*(c_old[i]-c_old[i-1])
    c[Nx] = c[Nx-1]

    
    iplot += 1
    if (iplot % 100 == 0):
        plt.title(f"Time = {t:5.3f} s")
        line[0].set_ydata(c)
        figure.canvas.draw()
        figure.canvas.flush_events()
        iplot = 0


#plt.pause(0.0001)
plt.show()



