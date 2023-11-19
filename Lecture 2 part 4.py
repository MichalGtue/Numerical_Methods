import matplotlib.pyplot as plt
import numpy as np
import math

t= np.linspace(0, 2*np.pi, 1000)


y1, y2, y3 = np.sin(t), np.cos(t), np.tan(t)


fig = plt.figure(figsize=(8,7))

ax1 = plt.subplot(2, 2, 1)
ax1.plot(t, y1)
ax1.set_title('sin(x)')

ax2 = plt.subplot(2, 2, 3)
ax2.plot(t, y2)
ax2.set_title('cos(x)')

ax3 = plt.subplot(1, 2, 2)
ax3.plot(t, y3)
ax3.set_title('tan(x)')
ax3.set_ylim(-10, 10)
plt.tight_layout()
plt.show()
