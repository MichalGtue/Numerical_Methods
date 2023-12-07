import numpy as np
import matplotlib.pyplot as plt

f= lambda x: 1/(x**2 + 1/25)
#x4 = np.linspace(-1,1, 5, endpoint=True)
#x10 = np.linspace(-1,1,5, endpoint=True)

x4, x11, xinf = [np.linspace(-1,1,n, endpoint= True) for n in [5,11, 1001]]

y4 = f(x4)
y11 = f(x11)
yinf = f(xinf)


p4 = np.polyfit(x4,y4,4)
p10 = np.polyfit(x11,y11, 4)
y4_int = np.polyval(p4, xinf)
y10_int = np.polyval(p10, xinf)

plt.plot(x4, y4,)
plt.plot(x11, y11)


#print(p4)