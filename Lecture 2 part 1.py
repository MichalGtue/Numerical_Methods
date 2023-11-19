import numpy as np

def greet(name):
    print(f"Hello, {name}")


#greet('Peter')

def add(a,b):
    return a+b

print(add(3, 5))

def my_functio(*args):
    print(args)

my_functio(1,4,"hello")


def factorial(i):
    nfactorial = 1
    for numb in range(1, i+1):
        nfactorial = nfactorial * numb
    return(nfactorial)



def exp3(j):
    output = 1
    for n in  range(1,20):
        output = output + ((j)**(n))/(factorial(n)) 
    return(output)

print(exp3(4))

def mystery (a,b):
    assert b >= 1, "B must be >= 1" and isinstance(b, int) 
    if b ==1:
        return a
    else:
        return (a) + mystery(a, b-1)





import math


f = lambda x: x**2 + math.exp(x)

print(f(1))

xlist = [i/7*2*math.pi for i in range (0,8)]
ylist = [math.sin(x1) for x1 in xlist]

print(xlist, ylist)