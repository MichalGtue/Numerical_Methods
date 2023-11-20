import numpy as np
from sympy import simplify, symbols, integrate, diff, tan
from sympy.abc import x, y, z

def factorial(i):
    nfactorial = 1
    for numb in range(1, i+1):
        nfactorial = nfactorial * numb
    return(nfactorial)



def exp3(j):
    output = 1
    for n in  range(1,26):
        output = output + ((j)**(n))/(factorial(n)) 
    return(output)
print(exp3(-5), np.exp(-5))

print(round(exp3(-5),6))


f = (x-1)*(x+1)*(x**2 +1) +1
f2 = f.simplify()

print(f2)

g = 1/(x**3 +1)

G = integrate(g, x)

g_diff = diff(G, x)
g_og = g_diff.simplify()
print(g_og)

f_tan = 2*tan(x)/(1+tan(x)**2)

f_sim = f_tan.simplify()

print(f_sim)