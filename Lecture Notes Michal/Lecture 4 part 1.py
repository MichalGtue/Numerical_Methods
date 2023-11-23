from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import time

A = np.array([[1,1,1],[2,1,3],[3,1,6]])

b = np.array([4,7,5])


Ainv = np.linalg.inv(A)


#print(Ainv @ b)
np.dot(Ainv, b)


x = np.linalg.solve(A,b)

#print(x)



array1 = np.zeros([1,2])
#print(array1)

total_time = []

sizes = [10,20,50,100,200,500,1000,2000,5000,10000]
for i in sizes:
    starttime = time.time()
    array = np.random.randint(i+1, size=(i,i))  
    arrayinv = np.linalg.inv(array)
    endtime = time.time() - starttime
    #array1 = np.add(array1, [i, endtime])
    total_time.append(endtime)

print(array1)

plt.loglog(sizes, total_time)
plt.show()    