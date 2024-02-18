import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
print(sp.constants.find('stefan'))


h = sp.constants.physical_constants['Planck constant'][0]
c = sp.constants.physical_constants['speed of light in vacuum'][0]
kb = sp.constants.physical_constants['Boltzmann constant'][0]
b = sp.constants.physical_constants['Wien wavelength displacement law constant'][0]
sigma = sp.constants.physical_constants['Stefan-Boltzmann constant'][0]


def func(l, T):
    frac = (2*np.pi * c**2 * h)/(l**5)
    return frac * (np.exp((h*c)/(l*kb*T))-1)**-1

def peak(T):
    return b/T

x = np.linspace(0,6e-6, 100, endpoint=False)

T_list = [1000,1250,1500,1750,2000,2250]

for i in range(len(T_list)):
    plt.plot(x,func(x,T_list[i]), label=f'T={T_list[i]}')
    plt.plot(peak(T_list[i]), func(peak(T_list[i]), T_list[i]), 'x')



plt.xlabel('wavelength(m)')
plt.ylabel('Emission (W*m^-3)')
plt.legend()


for i in range(len(T_list)):
    print(f'For T = {T_list[i]}, Approx sol = {scipy.integrate.quad(func, 0, 1e-2, args=T_list[i], epsrel=1e-12)[0]}, Exact sol = {sigma*T_list[i]**4}')