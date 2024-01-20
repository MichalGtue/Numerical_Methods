import numpy as np
import matplotlib.pyplot as plt

def inf_sum(n):
    sum = 0
    for i in range(n):
        sum = sum+1/((i+1)**2)

    return sum

exact = np.pi**2/6
x= np.linspace(1,1000, 100, dtype=int)

y=np.zeros_like(x)

y=[]

for i in range(len(x)):
    sol=inf_sum(x[i])
    y.append(sol)


plt.plot(x,y)
plt.axhline(exact)
plt.show()






























