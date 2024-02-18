import numpy as np
import scipy.integrate

def func(x):
    y=np.sin((np.pi*x)/2)/(x) + 1
    return(y)

sol_exact = scipy.integrate.quad(func, 0, 10)
print(sol_exact[0])


def riemann(func, a, b, n):
    dx=(b-a)/n
    sum = 0
    for i in range(n):
        sum=sum+func(a+(i+1)*dx)*dx
    return(sum)

print(riemann(func,0,10,100))