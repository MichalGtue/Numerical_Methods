import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.sparse.linalg as spss

# Getting all the variables

Nt = 100000 # Time steps 
## Yes its a high number of time steps but the implicit solution is very 'easy' to solve
diam = 0.04 # Diameter
phiv = 6.3e-4 #volumetric flow rate
cL = 1 # Inlet conc
cR = 0 
t_end = 4 # 4 sec
x_end = 1 # Length
kR = 1.2 

u = phiv / (np.pi*(diam/2)**2) 

dt = t_end / Nt

Nx = [10, 20, 30, 50, 100]## Grid points

for d in range(len(Nx)):
    dx  = x_end / Nx[d]
    # Courant number
    Co = u* dt /dx
    x = np.linspace(0,x_end, Nx[d]+1)
    c = np.zeros(Nx[d]+1)
    c[0] = cL
    c[Nx[d]] = cR
    t = 0
    plt.ion()
    figure, ax = plt.subplots(figsize = (10, 5))
    line = []
    line += ax.plot(x, c)
    plt.title(f"Time: {t:5.3f} s", fontsize = 16)
    plt.xlabel('Axial position', fontsize = 14)
    plt.ylabel('Concentration', fontsize=14)
    plt.xlim(0, x_end)
    plt.ylim(0, max(cL, cR)+0.1) # To show that the max conc is at 1
    plt.grid
    iplot = 0
    A = sps.diags([-(Co), (1+Co+kR*dt)], [-1, 0], shape=(Nx[d]+1, Nx[d]+1))
    A = sps.csr_matrix(A)
    A[0,0]=1
    A[0,1] = 0
    A[Nx[d], Nx[d]] = 1
    A[Nx[d], Nx[d]-1] = -1
    for i in range(1, Nx[d]):
        if dx*i < 0.1 or dx*i>0.9:
            A[i, i] = A[i,i] - kR*dt ## Taking away the reaction term  

    for n in range(Nt):
        t += dt
        c_old = np.copy(c) ## doing c_old = c then modification of c_old will also modify c
        c[0] = cL

        b = c_old
        b[Nx[d]] = 0
        c = spss.spsolve(A, b)

        iplot += 1
        if (iplot % 2000 == 0):
            plt.title(f"Time = {t:5.3f} s, Nx[d] = {Nx[d]}")
            line[0].set_ydata(c)
            figure.canvas.draw()
            figure.canvas.flush_events()
            iplot = 0
        if t == 3.999999999992335: #Max t value
            plt.title(f"Time = {t:5.3f} s, Nx = {Nx[d]}")
            line[0].set_ydata(c)
            plt.savefig(f'{Nx[d]}.png', format='png')
            plt.show()
