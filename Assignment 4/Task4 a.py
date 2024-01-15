import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.sparse.linalg as spss

def simpson_rule(xlist, ylist):
    '''Calculates the area of a function using the Simpson rule. \n
    Takes 3 arguments, first the function, then the lower integration bound, and finally the upper integration bound. \n
    Optional 4th argument to specify number of intervals (default set to 20)'''
    sum = 0
    dx = (xlist[1]-xlist[0])
    for i in range(len(ylist)):
        if i>0 and i<len(ylist)-1:
            sum += (dx/6) * (ylist[i-1]+4*ylist[i]+ylist[i+1])
    return sum

# Getting all the variables
Nx = 100 ## Grid points
Nt = 100000 # Time steps
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

A_new = sps.diags([(Fo), -(1+Co_nit+2*Fo+kR*dt), Fo+Co_nit], [-1, 0, 1], shape=(Nx+1, Nx+1))
A_new = sps.csr_matrix(A_new)
#A_new[0,0]= 1
#A_new[0,1] = 1
A_new[Nx, Nx] = 1
A_new[Nx, Nx-1] = 0
for i in range(1, Nx):
    if dx*i < 0.1 or dx*i>0.9:
        A_new[i, i] = A_new[i,i] + kR*dt

## Change the first and last elements of a

c_diff = []
t_list_plotting = []
for n in range(Nt):
    t += dt
    c_old = np.copy(c) ## doing c_old = c then modification of c_old will also modify c
    b = c_old
    b[Nx] = 0
    if n*dt<5: # Backflow turned on at 5 sec
        c[0] = cL
        c = spss.spsolve(A, b)
    else:
        c = spss.spsolve(A_new,b)
        t_list_plotting.append(t-5)
        c_diff.append(np.sum(np.abs(c_old-c)))
    if t > 5.05 and t < 5.051:
        c_505 = np.abs(np.copy(c))
    elif t > 5.1 and t < 5.101:
        c_51 = np.abs(np.copy(c))
    elif t > 5.15 and t < 5.151:
        c_515 = np.abs(np.copy(c))
    elif t > 5.2 and t < 5.201:
        c_52 = np.abs(np.copy(c))
    iplot += 1
    if (iplot % 500 == 0):
        plt.title(f"Time = {t:5.3f} s")
        line[0].set_ydata(c)
        figure.canvas.draw()
        figure.canvas.flush_events()
        iplot = 0

plt.ioff()


figure, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, c_51, label='t = 5.1 s')
ax.plot(x, c_505, label='t = 5.05 s')
ax.plot(x, c_515, label='t = 5.15 s')
ax.plot(x, c_52, label='t = 5.2 s')
plt.title("Concentration outflow", fontsize=14)
plt.xlabel('Time taken', fontsize=10)
plt.ylabel('Concentration', fontsize=10)
plt.xlim(0, x_end)
plt.ylim(-1, max(cL, cR))
plt.legend()
plt.grid()
plt.show()


figure, ax = plt.subplots(figsize=(10, 5))
ax.plot(t_list_plotting, c_diff)
plt.title("Concentration at various times", fontsize=16)
plt.xlabel('Axial position', fontsize=14)
plt.ylabel('Concentration', fontsize=14)
plt.grid()
plt.show()

print(simpson_rule(t_list_plotting, c_diff)* np.pi * (diam/2)**2 * x_end)