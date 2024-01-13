import numpy as np
xdata=np.arange(0,6)
fun=lambda x: x**3/2-(10*x**2)/3/3+11*x/2+1
ydata=fun(xdata)
print(xdata,ydata)


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d 
xdata=np.arange(0,6)
fun=lambda x: x**3/2-(10*x**2)/3/3+11*x/2+1
ydata=fun(xdata)
fint=interp1d(xdata, ydata, kind='cubic')
print(fint(1.5))
xc=np.linspace(0,5,101)
yc=fint(xc)


import numpy as np
import matplotlib.pyplot as plt


plt.plot(xc,yc,'x-')
plt.plot(xdata,ydata, 's')
plt.plot(1.5,fint(1.5), 'r*')
plt.show()


#Exercise polynomical interpolation 
x = np.array ([0,1,2])
y = np.array([1.000,3.667,2.667])
V= np.vander (x, increasing = True)
print(V)
a= np.linalg.solve(V,y)
print(a)

#Exercise polynomical interpolation 
f = lambda x: 1/(x**2 + 1/25)
#x4= np.linspace(-1,1,5 endpoint = True)
#x411= np.linspace(-1,1,11 endpoint = True)
x4,x11, xinf= [np.linspace(-1,1,n,endpoint=True)for n in [5,11,1001]]
y4, y11, yinf = f(x4), f(x11), f(xinf)

p4 = np.polyfit(x4, y4, 4)
p10 = np.polyfit(x4, y4, 10)

y4_int = np.polyval(p4,xinf)
y10_int = np.polyval(p10,xinf)

plt.plot(x4, y4, 'o')
#plt.plot(x10, y10, 'x')
plt.plot(xinf, yinf, '-')
plt.plot(xinf, y4_int, '-')
plt.plot(xinf, y10_int, '-')


import numpy as np
from scipy.integrate import quad
def left_endpoint(func,a,b,npts=101):
    x=np.linspace(a,b,npts, endpoint=True)
    dx=x[1]-x[0]
    f=func(x)
    int_value= np.sum(f[0:-2]*dx)
    return int_value
def f(x):
    return x**2-4*x+6+np.sin(5*x)
I_exact = quad(f,0,10)
I= left_endpoint(f,0,10,21)
print(I_exact, I)

def right_endpoint(func,a,b,npts=101):
    x=np.linspace(a,b,npts, endpoint=True)
    dx=x[1]-x[0]
    f=func(x)
    int_value= np.sum(f[0:-2]*dx)
    return int_value
def f(x):
    return x**2-4*x+6+np.sin(5*x)
I_exact = quad(f,0,10)
I= right_endpoint(f,0,10,21)

print(I_exact, I)

def f(x):
    return x**2-4*x+6+np.sin(5*x)
def testint(func, method, a,b,npts=101):
    I_exact=quad(func,a,b)[0]
    I= method(function,a,b,npts)

print(f'quad yields: {I_exact}, {method.__name__}, ) yields: {I}, difference: {I_exact-I:1.3}')

if __name__ == '__main__':
    testint(f, left_endpoint,0,10,1000)
    testint(f, right_endpoint,0,10,1000)