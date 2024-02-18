import numpy as np
import matplotlib.pyplot as plt


def fun(x, n):
    total = 0

    for i in range(n):
        total = total + (3/(np.pi*(i+1))) * (1-(-1**(i+1))) * np.sin(np.pi*(i+1)*x/2)
    

    return total + 0.5


x = np.linspace(-2,2, 100)

iterations_list = [2,5,10,50,100,500,1000,10000]


for i in range(len(iterations_list)):
    plt.plot(x, fun(x,iterations_list[i]), label=f'Number of iterations = {iterations_list[i]}')


plt.legend()


plt.show()