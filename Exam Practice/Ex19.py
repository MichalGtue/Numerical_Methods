import numpy as np
import matplotlib.pyplot as plt

def func(x,sigma, mu=2*(2**0.5)):
    return 1/(sigma*(2*np.pi)**0.5) * np.exp(-0.5 * ((x-mu)/sigma)**2)



x_list = np.linspace(-15,15,1000)

sigma_list = [1,2,3,4,5,6]

for i in range(len(sigma_list)):
    plt.plot(x_list, func(x_list, sigma_list[i]), label=f'sigma = {sigma_list[i]}')

plt.legend()
plt.show()