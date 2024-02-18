import numpy as np
import matplotlib.pyplot as plt

def lehmer(z,a,m):
    return np.mod(a*z,m)



z_list=[342]


for i in range(100000):
    z_list.append(lehmer(z_list[-1],132,31657))

plt.hist(z_list, bins=15)



plt.show()