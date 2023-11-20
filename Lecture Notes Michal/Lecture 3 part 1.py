import numpy as np
import matplotlib.pyplot as plt

number = np.array([0.1])
for n in range (1, 30):
    number = np.vstack((number, number[-1]*10 - 0.9))




number_2 = 2
for n in range (1,30):
    number_2 = (number_2*10) - 18

#print('0.1 became', number, 'and 2 becomes', number_2)


v = np.logspace(0, 40, 41)
y = np.sin(v*np.pi)
plt.loglog(v, np.abs(y))
plt.show()

