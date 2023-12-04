

#Newton method

def fnc(x):
    return x**2-4*x+2

def fncprime(x):
    return 2*x-4


def newton_method(func, funcprime, x1, nguess=100, tol_x=1e-6, tol_f=1e-6):
    converged = False
    counter = 0
    while (not converged) and (counter < nguess):
        x2 = x1 - func(x1)/funcprime(x1)
        x3 = x2
        x1 = x2
        counter +=1
        converged = (abs(x3 - x1) <= tol_x and abs(func(x1) <= tol_f))
        print(counter)
    return x1

def newton_class(func,grad,x,tol_x,tol_f):
    
    return x

print(newton_method(fnc,fncprime, 100))