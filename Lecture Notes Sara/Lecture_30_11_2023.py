# exercise 1
def fnc0(x):
    x=2.5
    for i in range(20):
        x=(3*x**2+3*x+4)**(1/3)
print(f"it:{i:2}, x:{x:12.9f}")

def fnc2(x):
    return (3*x**2+3*x+4)**(1/3)
def fnc1(x):
    return(x**3-3*x**2-4)/4


def direct_itineration(func, x, itmax=20):
    for i in range(itmax):
        try:
            x=func(x)
        except OverflowError as e:
            print("Overflow error")

direct_itineration(fnc1,2.5,15)

