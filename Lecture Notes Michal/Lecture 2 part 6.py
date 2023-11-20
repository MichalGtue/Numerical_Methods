import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0, 4*np.pi, 100)
y=np.sin(x)

fig, ax=plt.subplots()
line, = ax.plot(x, y, '-')

plt.show(block=False)

for i in range(len(x)):
    line.set_data(x[i-20:i+1], y[i-20:i+1])
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)