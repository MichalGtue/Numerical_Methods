import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sum(a):
    #print(a)
    #number = a[0]
    yout = []
    print(yout)
    for j in range(len(a)):
        total = 0
        xval = a[j]
        for i in range(xval):
            total = total + (1/((i+1)**2)) # i starts at 0
        yout.append(total)
        print(yout)
    return yout

exact = (np.pi**2)/6

x = np.linspace(5, 2000, 50, dtype=int)
y= sum(x)

plt.plot(x,y, 'red',label='Sum')
plt.axhline(exact, label='Exact')
plt.legend()
plt.show()
print(np.floor(2.5))

curr_tot = 0

    

#for i in range(np.size(x)):
#    y[i] = sum([x[i]])


#rint(sum(30), exact)

