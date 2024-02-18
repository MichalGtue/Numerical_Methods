import matplotlib.pyplot as plt
import numpy as np

def func(x,y):
    return




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.arange(0, 1, 0.025)
y = np.arange(0, 1, 0.025)
x,y = np.meshgrid(x, y)

T = np.zeros_like(x)

for i in range(1,100):
    m = 2*i-1
    T = T + (np.sin(m*np.pi*x)*np.sinh(m*np.pi*y))/(m*np.sinh(m*np.pi))

T = T* 4/np.pi    

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(x,y,T)

plt.show()