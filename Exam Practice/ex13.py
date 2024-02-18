import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,3.6,1000,endpoint=False)

def func(t):
    if t >0 and t<=2.0:
        return -3.5*t**2 + 4.5*t +5
    if t<=3.0:
        return -8*t**2 + 40*t -48
    if t<=3.4:
        return -15*t**2 + 96*t-153
    if t<=3.6:
        return -10*t**2 +70*t-122.4
    


y = np.zeros_like(x)

for i in range(len(x)):
    y[i] = func(x[i])

y[0] = -3.5*x[0]**2 + 4.5*x[0] +5
plt.plot(x,y)
plt.xlim(0,3.6)
plt.show()