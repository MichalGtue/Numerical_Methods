#Bracketing
#If a function is positive at some point and negative at some other then you know that 
#there exists a root (or roots) somewhere in between


def fnc(x):
    return x**2-4*x+2

def brac(func, x1, x2):
    found = False
    ntry  = 50
    factor = 1.6
    f1 = func(x1)
    f2 = func(x2)
    if x1<x2:
        f1 = func(x1)
        f2 = func(x2)
        for i in range(ntry):
            if f1*f2<0:
                found = True
                break
            if (i % 2) == 0 :
                x1 += factor*(x1-x2)
                f1 =func(x1)
            else:
                x2 += factor*(x2-x1)
                f2 = func(x2)
    else:
        print("Bad initial range!")
    if found:
        print(f"The bracketing interval is [{x1}, {x2}]")
    return found, x1, x2


def brak(func, x1, x2, n):
    nroot = 0
    dx = (x2 - x1)/n
    x_left = x1
    f_left = func(x_left)
    for i in range(n):
        x_right = x_left + dx
        f_right = func(x_right)
        if f_left*f_right <= 0:
            nroot+=1
            print(f"root {nroot} in interval[{x_left}, {x_right}]")
        x_left = x_right
        f_left = f_right
    if nroot == 0:
        print("No bracketing intervals")
    return func, x1, x2, n




#brak(lambda x: x**2 -4*x + 2, -5, 5, 5)

def bisection(func, x1, x2, nguess):
    iteration = 0
    y1 = func(x1)
    y2 = func(x2)
    for i in range(nguess):
        if y1>y2:
            print("Pick better guess")
            return
        midpoint = (x1 + x2)/2
        y3 = func(midpoint)
        if y3 > 0:
            x2 = midpoint
        else:
            x1 = midpoint
    print(f'Your root is close to {midpoint}')
    return(func, x1, x2, nguess)




def bisection_class(func, x1, x2, tol_x, tol_f):
    f1 = func(x1)
    f2 = func(x2)
    if f1*f2>0:
        print(f'Root should be bracketed')
        return
    else:
        converged = False
        it = 0
        while (not converged):
            it +=1
            p = 0.5*(x1+x2)
            fp = func(p)
            if f1*fp<0:
                x2=p
                f2=fp
            else:
                x1 = p
                f1 = fp
            converged = (abs(x2 - x1) <= tol_x and abs(fp) <=tol_f)
            print(f"it:{it:2}, x:{p:18.16}, f(x):{fp:15.8e}, New bracketing interval [{x1}, {x2}]")
        print(f"Bisection: Root found in {it} iterations at x = {p} with function value {fp:e}")
    return p

bisection_class(lambda x: x**2 -4*x + 2, 0, 1, 1e-14, 1e-13)

def false_position(func, x1, x2, tol_x, tol_f):
    f1 = func(x1)
    f2 = func(x2)
    if f1*f2>0:
        print(f'Root should be bracketed')
        return
    else:
        converged = False
        it = 0
        while (not converged):
            it +=1
            p = (x1*f2-x2*f1)/(f2-f1)
            fp = func(p)
            if f1*fp<0:
                x2=p
                f2=fp
            else:
                x1 = p
                f1 = fp
            converged = (abs(x2 - x1) <= tol_x and abs(fp) <=tol_f)
            print(f"it:{it:2}, x:{p:18.16}, f(x):{fp:15.8e}, New bracketing interval [{x1}, {x2}]")
        print(f"False position: Root found in {it} iterations at x = {p} with function value {fp:e}")
    return p

def secant(func, x1, x2, tol_x, tol_f):
    f1 = func(x1)
    f2 = func(x2)
    if f1*f2>0:
        print(f'Root should be bracketed')
        return
    else:
        converged = False
        it = 0
        while (not converged):
            it +=1
            p = (x1*f2-x2*f1)/(f2-f1)
            fp = func(p)
            x1, f1, x2, f2 = x2, f2, p, fp
            converged = (abs(x2 - x1) <= tol_x and abs(fp) <=tol_f)
            print(f"it:{it:2}, x:{p:18.16}, f(x):{fp:15.8e}, New bracketing interval [{x1}, {x2}]")
        print(f"False position: Root found in {it} iterations at x = {p} with function value {fp:e}")
    return p