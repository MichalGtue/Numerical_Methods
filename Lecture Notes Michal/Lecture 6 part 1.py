import numpy as np

## LU decomposition is usually best for big systems of linear equations. Need to read up on it
#Gaussian method is slow for bigger systems

#Method must be chosen that is correct for the case. No single winner


#for i in range(10):
#
#    x = (3**2 - 3*x**2 - 4)/3
#    print(f"it: {i:2}, x: {x:12.9}")


def fnc0(x):
    return (3*x**2 + 3*x + 4)**(1/3)

def fnc1(x):
    return (x**2 - 3*x**2 - 4)/3



def direct_iteration(func, x, itmax=20):
    for i in range (itmax):
        try:
            x = func(x)
        except OverflowError as e:
            print("Overflow error happened - method diverges!")
            break
        print(f"it: {i:2}, x: {x:12.9}")



direct_iteration(fnc1, 4, 15)