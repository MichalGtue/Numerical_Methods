import scipy as sp
2 import numpy as np
3 import matplotlib.pyplot as plt
4
5 def system_ode(t,Xin,params):
6 # You don't have to use a dictionary, but for >2 additional variables I would recommend it.
Getting the values from the dictionary can be done in many different ways, the one
used here relies on the order of the dictionary (it could've been a list)
7 alpha,beta,gamma,delta = params.values()
8
9 # For convenience, we also unpack Xin, the incoming argument, to make the calculations
similar to the equations in the question. Again, this is not the only way it can be
done; we can also use Xin[0] for x and Xin[1] for y
10 x,y = Xin
11
12 # Compute the ODEs. Here, we give it the name dxdt since that most accurately describes
what it represents; dx/dt... But in the end, it is just a name and we could call it
banana, Jack or Mewtwo
13 dxdt = alpha * x - beta * x * y
14 dydt = delta*x*y - gamma*y
15
16 # Of course, we should return the complete system of solutions
17 return [dxdt,dydt]
18
19
20 # This statement is not strictly needed, and for the exam you might as well omit it. It makes
sure that upon importing a function from this file, the code below is not executed.
21 if __name__ == '__main__':
22 # Create a dictionary to hold all parameters
23 par = {'alpha': 2/3, 'beta': 4/3, 'gamma': 1.0, 'delta': 1.0}
24
25 # A time span was not defined in the question, in the exam it will generally be defined but
always check yourself if the range is reasonable (if you make a parameter typo, the
time to run towards a stable or pseudo-stable solution could be much longer or shorter)
26 tspan = [0, 50]
27 t_eval = np.linspace(tspan[0],tspan[1],1000)
28
29 # This part is quite verbose, we could just say init=[0.9,1.4]
30 x0 = 0.9
31 y0 = 1.4
32 init = [x0,y0]
33
34 # Call the ODE function; use as additional argument the dictionary
35 sol = sp.integrate.solve_ivp(system_ode,tspan,init,t_eval=t_eval,args=(par,))
36
37 # The sol structured variable contains much information. Try to print it! Here, we only
extract the time and result data for plotting
38 t = sol.t
39 x = sol.y
40 plt.plot(t,x[0,:],t,x[1,:])
41 plt.xlabel('Time [s]')
42 plt.ylabel('x,y [a.u.]')
43 plt.tight_layout()
44 plt.show()
9
Answer exercise 2
1 # ex02.py
2 import matplotlib.pyplot as plt
3 from math import pi, log
4 import numpy as np
5 # The current term has to be initialized at some value for the while loop to function
6 term = 0
7
8 # We are interested in the convergence, so we need access to subsequent values of the series -
hence we choose a list
9 the_sum = [0]
10 n = 1
11
12 # We compute the series for a reasonably long time, until the difference with the previous term
diminishes. Each term is added to the list
13 while term > 1.0e-12 or n==1:
14 term = 1/n**2
15 the_sum.append(the_sum[n-1] + term)
16 n += 1
17
18 # Note that n at this point exceeds the list size by 1, so when we refer to the end of the list
, we should do n-1
19
20 # Convert to np array to make it easier to compute the error
21 the_sum = np.array(the_sum)
22 err = the_sum - (pi**2/6)
23
24 # In the slides of ODEs, the rate of convergence equation is given
25 rate = log(err[n-1]/err[n//2])/log((n//2)/(n-1))
26 print(f'Numerical rate of convergence: {rate}')
27
28 # Also plot and observe how the error shrinks 1 order when n increases 1 order
29 plt.loglog(pi**2/6 - the_sum)
30 plt.show()
Answer exercise 3
1 # ex03.py
2 from math import exp, cos
3 import scipy.optimize as spop
4 # Note that the equations are given in x_1 through x_3, but Python starts counting at 0, so we
make the translation in-place: x_1 --> x[0] etc.
5 def system_eqns(x):
6 y1 = 3*x[0] - cos(x[1]*x[2]) - 0.5
7 y2 = x[0]**2 -x[1] - x[2]
8 y3 = x[0] + 0.5*x[1] - exp(x[2])
9 return [y1,y2,y3]
10
11 # Call fsolve to solve the system; if it doesn't resolve correctly, try with different initial
guess values
12 sol = spop.fsolve(system_eqns,[0,0,0])
13 print(f'Solution x={sol}, function values: f(x)={system_eqns(sol)}')
10
Answer exercise 4
1 # ex04.py
2 # Note: this exercise was written without too much detail; you should choose a suitable plot
type, colors, labels, legend (yes or no), axis labels, axis ranges, etc. The message is
that you need to think about the presentation before you submit/finalize.
3
4 import numpy as np
5 import matplotlib.pyplot as plt
6
7 set_T = [1,2,4,8]
8
9 # No range is given, so let's try one and see where we get
10 t = np.linspace(0,20,1001)
11
12 for T in set_T:
13 # Decaying wave
14 v = np.exp(-t/T)*np.sin(t)
15 plt.plot(t,v,label=f'T={T}')
16
17 plt.xlabel('Time [s]')
18 plt.xlabel('v(t) [a.u.]')
19 plt.legend()
20 plt.show()
11
Answer exercise 5
1 #ex05.py
2 from scipy.interpolate import interp1d
3 import numpy as np
4 import matplotlib.pyplot as plt
5
6 # Load t he data using numpy load:
7 data = np.load('answers/data.npy')
8
9 # After looking into the structure of data, we are able to tell that we have 2 arrays of size
98:
10 print(data.shape)
11 x = data[0,:]
12 y = data[1,:]
13
14 # First let's assess whether the data in x-direction is indeed not equidistant by computing the
differences. This also allows us to see if the data is monotonically increasing
15 d = np.diff(x)
16 print(f'Differences in x: {d}')
17
18 # It seems that all points in x are at an inceasing distance from each other, so we want to
interpolate to an equidistant grid. Define the grid, use the minimum and maximum values of
x to set the grid boundaries:
19 x_int = np.linspace(min(x),max(x),101)
20
21 # Compute the interpolation function. We can use any 'kind', as it has not been defined as a
requirement
22 f = interp1d(x,y,kind='cubic')
23
24 # Compute the interpolated values
25 y_int = f(x_int)
26
27 plt.plot(x_int,y_int,'x')
28 plt.plot(x,y,'o')
29 plt.show()
Answer exercise 6
1 # ex06.py
2
3 from scipy.integrate import quad
4 from math import sin,pi
5
6 # Define the function
7 def f(x):
8 y = sin(pi*x/2)/2 + 1
9 return y
10
11 # Compute the integral using a module routine
12 result = quad(f,0,10)
13
14 # Make sure to present your result in an appropriate way or to unpack it from the list that
quad creates:
15 print(f'The result is {result[0]} with an estimated error of {result[1]}')
16
17 # Alternatively, use a lambda function:
18
19 result1 = quad(lambda x: sin(pi*x/2)/2 + 1, 0, 10)
20 print(result1[0])
12
Answer exercise 7
1 # ex07.py
2 from scipy.integrate import solve_ivp
3 import numpy as np
4 import matplotlib.pyplot as plt
5
6 # Set up the Van Der Pol equation as given in the assignment
7 def vdpol(x,y,mu):
8 # Unpack y into y1 and y2 to create similarity with the assignment
9 y1,y2 = y
10 # Compute each ODE individually
11 y1prime = y2
12 y2prime = mu*(1-y1**2)*y2 - y1
13 # Do not forget to return a list of the individual components, otherwise your ODE solver
will run forever and get no result
14 return [y1prime,y2prime]
15
16 # We were asked to use different values of mu - so we can loop and use the variable mu into the
label. You can also simply copy the solve_ivp and plot statements.
17 for mu in [2,5,15]:
18 sol = solve_ivp(vdpol,[0,15],[2,0],args=(mu,),method='BDF')
19 plt.plot(sol.t,sol.y[0,:],label=f'$y_1$, mu={mu}')
20 plt.plot(sol.t,sol.y[1,:],label=f'$y_2$, mu={mu}')
21
22 # Finalize the plot.
23 plt.legend()
24 plt.xlabel('X [-]')
25 plt.ylabel('Y [-]')
26 plt.tight_layout()
27 plt.show()
13
Answer exercise 8
1 # ex08.py
2 from scipy.integrate import solve_ivp
3 import numpy as np
4 import matplotlib.pyplot as plt
5
6 def systemode(x,y,a):
7 # Define a matrix A with coefficients
8 A = np.array([[-2,1], [a-1,-a]])
9 # Defien a vector g
10 g = np.array([2*np.sin(x), a*(np.cos(x) - np.sin(x))])
11 # Compute yprime; the matrix A should be multiplied with the values of y, so we should use
@ or np.dot(A,y); a common multiplication A*y will attempt an element-wise operation,
which errors out because the shapes of A and y are different
12 dydx = A*y + g
13 return dydx
14
15 # Solve and plot the ODE solutions
16 sol = solve_ivp(systemode,[0,10],[2,3],args=(2,),method='Radau',atol=1e-18)
17 plt.plot(sol.t,sol.y[0,:],sol.t,sol.y[1,:])
18
19 # Solve and plot the exact solutions
20 x = np.linspace(0,10,1000)
21 # This y_exact computes the solution vector at once, but you may also split the y1 and y2 exact
calculations
22 y_exact = 2*np.exp(-x) + np.array([np.sin(x),np.cos(x)])
23 plt.plot(x,y_exact[0,:])
24 plt.plot(x,y_exact[1,:])
25 plt.show()
Answer exercise 9
1 # ex09.py
2 import numpy as np
3 import matplotlib.pyplot as plt
4 # Set up the points in p (first row contains x, second row contains y), then we transpose using
.T so each row contains the (x,y)
5 xp = [2,4,4.5,5.8]
6 yp = [4,4,9,-12]
7 p = np.array([xp,yp]).T
8 print(p)
9
10 # We create a unique polynomial by fitting a 3rd-degree polynomial
11 p3 = np.polyfit(p[:,0],p[:,1],3)
12
13 # Set up a high-resolution x-vector and compute y-values of the polynomial function
14 xh = np.linspace(1.5,6,1000)
15 yh = np.polyval(p3,xh)
16
17 # Plot the interpolated function along the data points. The title unpacks the different
coefficients to display the interpolated function.
18 plt.plot(xh,yh,'-',label='$p_3(x)$')
19 plt.plot(p[:,0],p[:,1], 'o',label='data')
20 plt.title(f'Interpolated $p_3(x) = {p3[0]:1.2f}x^3{p3[1]:+1.2f}x^2{p3[2]:+1.2f}x{p3[3]:+1.2f}$'
)
21 plt.legend()
22 plt.show()
14
Answer exercise 10
1 # ex10.py
2 import numpy as np
3 import matplotlib.pyplot as plt
4 import time
5 # Being an infinite summation series, we can choose to either loop until a large number, or use
a while statement to consider the value of an individual term. Here, I choose to sum until
N=large using a for-loop, we can see from the equation that if we choose N=1000, the
second term drops below 1e-6. The first term just adds or removes 1.
6 N = 10000
7
8 # We want to plot y as a function of t, so let's create a range 0<=t<=1. For every t value, we
want to create a y-value, so initialise a zeros array of the same size
9 t_range = np.linspace(0,1,10001)
10 y_final = np.zeros_like(t_range)
11 # We use enumerate, so that we also have an index-counter i that helps us to store the solution
at the right index in y_final
12 # for i,t in enumerate(t_range):
13 # y_sum = 0
14 # for k in range(1,N):
15 # y_sum += (-1)**(k+1) + 4/(k**2*np.pi**2) * np.cos(k*np.pi*t)
16 # y_final[i] = 2/3 + y_sum
17
18 # plt.plot(t_range,y_final)
19 # plt.xlabel('Time [s]')
20 # plt.ylabel('y [-]')
21 # plt.show()
22
23 # Alternative solution
24 # Loops make Python slow, especially nested ones; a more vectorized solution (but perhaps less
accessbible for new users) can be made like:
25
26 t_range = np.linspace(0,1,1001)
27 y_final = np.zeros_like(t_range)
28 k = np.arange(1,N)
29 # for i,t in enumerate(t_range):
30 # y_final[i] = 2/3 + np.sum((-1)**(k+1) + 4/(k**2*np.pi**2) * np.cos(k*np.pi*t))
31
32 # plt.plot(t_range,y_final)
33 # plt.xlabel('Time [s]')
34 # plt.ylabel('y [-]')
35 # plt.show()
36
37
38 """
39 Create a plot of the following function:
40 y(t) = 2/3 + \sum \limits_{k=1}^{\infty} (-1)^{k+1} + 4/(k^2\pi^2) \cos(k*\pi*t)
41
42 - Bas Oomen, 18 December 2023
43 """
44
45 y = np.zeros_like(t_range)
46 k = np.arange(1,N)
47
48 k_matrix, t_matrix = np.meshgrid(k, t_range, indexing='ij')
49 kt = k_matrix * t_matrix
50 start = time.time()
51 y = 2./3. + np.sum((-1)**(k_matrix+1) + 4./(k_matrix**2*np.pi**2) * np.cos(np.pi*kt), axis = 0)
52 print(f'Elapsed time: {start - time.time()}')
53
54 y_alt = np.zeros_like(t_range)
55 start = time.time()
56 for i,t in enumerate(t_range):
57 y_alt[i] = 2./3. + np.sum((-1)**(k+1) + 4./(k**2*np.pi**2) * np.cos(k*np.pi*t))
58 print(f'Elapsed time: {start - time.time()}')
59
60 plt.figure(dpi = 225)
61 plt.plot
62 plt.plot(t_range, y, label = 'NumPy matrices')
63 plt.plot(t_range, y_alt, label = 'Single for-loop')
64 plt.grid(linestyle = '--')
65 plt.xlabel('$t$')
66 plt.ylabel('$y$')
67 plt.legend()
68 plt.show()
15
Answer exercise 11
1 # ex11.py
2 import matplotlib.pyplot as plt
3 import numpy as np
4
5 # Essentially, when creating a plot of a 3-D function we need to make a surface plot. The bare
minimum requires us to set up a mesh grid to compute the function values on every grid node
(x,y):
6 space = np.linspace(-10,10,100)
7 x,y = np.meshgrid(space,space)
8 z = x**3 + 3*y - y**3 - 3*x
9
10 # Then the procedure is to create a 3d axis projection before you can call plot_surface:
11 fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
12 surf = ax.plot_surface(x, y, z)
13 plt.show()
14
15 # A somewhat more elaborate example defining the grid lines and a color map:
16 import matplotlib.cm as cm
17 fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
18 surf = ax1.plot_surface(x, y, z, cmap=cm.jet,linewidth=0.5,color='black',antialiased=True)
19 fig1.colorbar(surf, shrink=0.5, aspect=5)
20 plt.show()
Answer exercise 12
1 # ex12.py
2 import numpy as np
3 import matplotlib.pyplot as plt
4 from scipy.interpolate import CubicSpline
5
6 xdata = np.linspace(0,8*np.pi,25)
7 ydata = 50*np.sin(xdata)/(xdata+1)
8
9 # Create the spline interpolant
10 splfun = CubicSpline(xdata,ydata)
11
12 # Create the function plot, and plot alongside the data
13 x = np.linspace(0,8*np.pi,2000)
14 y = splfun(x)
15 plt.plot(x,y)
16 plt.plot(xdata,ydata,'o')
17 plt.show()
16
Answer exercise 13
1 # ex13.py
2 import numpy as np
3 import matplotlib.pyplot as plt
4
5 def yfunc(xin):
6 y = np.zeros_like(xin)
7 for i,x_el in enumerate(xin):
8 x = np.vander([x_el],3)
9 if 0<=x_el<=2:
10 y[i] = x@np.array([-3.5,4.5,5])
11 elif x_el<=3:
12 y[i] = x@np.array([-8,40,-48])
13 elif x_el<=3.4:
14 y[i] = x@np.array([-15,96,-153])
15 elif x_el<=3.6:
16 y[i] = x@np.array([-10,70,-122.4])
17
18 return y
19
20 x = np.linspace(0,3.6,10000)
21 y = yfunc(x)
22
23 plt.plot(x,y)
24 plt.show()
Answer exercise 14
1 # ex14.py
2 import matplotlib.pyplot as plt
3 import numpy as np
4
5 # One way to create the signal is to make a very high-res x-axis, and set all y values where
2<=x<=4 to 1
6 x = np.linspace(0,10,10001)
7 y = np.zeros_like(x)
8 y[(x>=2) & (x<=4)] = 1
9
10 # Alternatively, we can simply create a line defining all nodes where the signal changes;
11 x = [0,2,2,4,4,10]
12 y = [0,0,1,1,0,0]
13
14
15 # We should create a figure with a red line, a grid and axis labels. If you plot it just like
that, it will expand the axis limits, so we have to enforce limits to our desired values.
If you really want to approach the picture, add major and minor grid lines and add 'clip_on
=False' to the graph, so that it is not clipped by the box (alternatively you can work with
the z-order).
16 plt.figure(figsize=(10,4))
17 plt.plot(x,y,'r-',linewidth=2,clip_on=False)
18 plt.grid(which='major', color='#CCCCCC', linestyle='-')
19 plt.grid(which='minor', color='#DDDDDD', linestyle=':')
20 plt.xlim(0,10)
21 plt.ylim(0,1)
22 plt.minorticks_on()
23 plt.xlabel('Time [s]')
24 plt.ylabel('Signal [-]')
25 plt.show()
26 # plt.savefig('ex14.pdf')
17
Answer exercise 15
1 # ex15.py
2
3 def color_of_wavelength(l):
4 """Print the perceived color of light of a particular wavelength input l"""
5 if (380<=l<=450):
6 color = 'Violet'
7 elif (450<=l<=495):
8 color = 'Blue'
9 elif (495<=l<=570):
10 color = 'Green'
11 elif (570<=l<=590):
12 color = 'Yellow'
13 elif (590<=l<=620):
14 color = 'Orange'
15 elif (620<=l<=750):
16 color = 'Red'
17 else:
18 color = "Invisible"
19
20 # The question asks to display the color, so we prettify it by showing both the lambda and
the named color. You can also print the color from within the if-clauses, but I prefer
to not copy code as much.
21 print(f'Wavelength {l} shows as {color}')
22
23 # Just a test; not asked, but given nonetheless
24 for l in range(360,780,20):
25 color_of_wavelength(l)
Answer exercise 16
1 # ex16.py
2 import matplotlib.pyplot as plt
3 import numpy as np
4 # The point of a Lehmer random number generator is that it can simulate seemingly random
numbers, which are fully reproducible. We can create a single number generator (using the
previous number as input), and call it a number of times, or we can create a vectorised
number generator.
5 def lehmer_single(z_prev,a,m):
6 z_next = a*z_prev % m
7 return z_next
8
9 a = 142
10 m = 31657
11 # Create a list to contain all random numbers, and initialise it with the starting random
number
12 z = [342]
13
14 # We loop and append a new random number every time. We don't need the iterator, so call it _
15 for _ in range(10000):
16 z.append(lehmer_single(z[-1],a,m))
17
18 # We can see that it generates random numbers in a range [0,m]. From the histogram alone, we
cannot say much more than that all numbers in this range are generated - it remains the
question whether the pattern is unpredictable.
19 plt.hist(z)
20 plt.show()

Answer exercise 17
1 # ex17.py
2 import numpy as np
3 import matplotlib.pyplot as plt
4
5 N = 100
6 y = np.zeros(N)
7 y[0] = 1
8
9 # Compute range
10 for i in range(1,N):
11 y[i] = y[i-1] + (2/3)**(i-1)
12
13 # Compute error based on last outcome
14 err = np.abs(np.array(y) - y[-1])
15 print(err)
16
17 # Make a plot
18 plt.semilogy(err[:-1])
19 plt.show()
19
Answer exercise 18
1 # ex18.py
2 import scipy.constants as spc
3 from scipy.integrate import quad
4 import numpy as np
5 import matplotlib.pyplot as plt
6
7 def emission(l, T):
8 # We make lambda a np array if it isn't already
9 l = np.array(l)
10 # Get some constants from the scipy.constants module
11 h = spc.Planck
12 c = spc.speed_of_light
13 kB = spc.Boltzmann
14 phi = (2*np.pi*c**2*h / l**5) / (np.exp(h*c/(kB*l*T)) - 1)
15 return phi
16
17 ### Part 1
18 # Use scipy.constants for SI prefixes, in this case convert to micrometer
19 # Note: start just above zero, since a wave length of 0 does not exist
20 lam = np.linspace(0.1,6,1001)*spc.micro
21
22 # Compute and plot the profiles at the desired temperatures. Don't forget the labels!
23 for T in range(1000,2500,250):
24 phi = emission(lam,T)
25 plt.plot(lam,phi,label=f'T={T} K')
26
27 # Prettify the plot (compressed some lines together to fit on 1 page)
28 plt.grid(); plt.legend(); plt.xlabel('Wavelength [$\mu m$]'); plt.ylabel('Emission [$W/m^3$]')
29 # plt.show() # Enable this line for just part 1
30
31 ### Part 2
32 print(spc.find('wien')) # Find the right Wien constant, yields 2 possibilities:
33
34 # We need to select the wavelength displacement law constant, that is index '1'
35 b = spc.physical_constants[spc.find('wien')[1]]
36 print(b)
37
38 # This gives b as a tuple (value,unit,uncertainty), if we need the value we have to put b[0]
39 T = np.array(range(1000,2500,250))
40 l_peak = b[0]/T
41 phi_peak = emission(l_peak,T)
42 plt.plot(l_peak,phi_peak,'ro')
43 plt.show()
44
45 ### Part 3
46 # We integrate for each temperature separately using a list comprehension
47 # Note 1: quad returns the integral and uncertainty, we select the integral using [0]
48 # Note 2: if you integrate from 0 to infinity (np.inf), quad complains about divergence. Reduce
the range from very small (e.g. 1 nm) to large wavelengths (e.g. 1 cm):
49 Phi = [quad(emission,1e-9,1e-2,args=(temperature,))[0] for temperature in T]
50
51 # Also calculate the Phi using Stefan-Boltzmann's Law:
52 sigma = spc.Stefan_Boltzmann
53 Phi_sb = sigma * T**4
54
55 fig,axs = plt.subplots(2,1,sharex=True)
56 axs[0].plot(T,Phi,label=f'Integrated')
57 axs[0].plot(T,Phi_sb,label=f'Stefan-Boltzmann\'s law')
58 axs[0].legend()
59 plt.xlabel('Temperature [K]')
60 axs[0].set_ylabel('Emission [$W/m^2$]')
61 plt.grid()
62 axs[1].plot(T,Phi-Phi_sb)
63 axs[1].set_ylabel('$\Delta$ Emission [$W/m^2$]')
64 plt.grid()
65 plt.show() 20
Answer exercise 19
1 # ex19.py
2
3 import numpy as np
4 import matplotlib.pyplot as plt
5
6 def gaussian(x,mu,sigma):
7 return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-1/2 * ((x-mu)/sigma)**2)
8
9 x = np.linspace(-15,15,1001)
10 mu = 2*np.sqrt(2)
11 for s in range(1,7):
12 plt.plot(x,gaussian(x,mu,s),label=f'$\sigma = {s}$')
13
14 plt.legend()
15 plt.xlabel('x')
16 plt.ylabel('pdf [-]')
17 plt.show()
Answer exercise 20
1 # ex20.py
2
3 import numpy as np
4 import matplotlib.pyplot as plt
5
6 pos = np.zeros((200,2))
7 for i in range(1,200):
8 step = 1-2*np.random.rand(1,2)
9 pos[i] = pos[i-1] + step
10
11 plt.plot(pos[:,0],pos[:,1],'-o',markersize=2.5)
12 plt.plot(pos[0,0],pos[0,1],'ro')
13 plt.show()
Answer exercise 21
1 # ex21.py
2
3 import numpy as np
4 import matplotlib.pyplot as plt
5
6 x = np.linspace(-2,2,1001)
7 s = np.zeros_like(x)
8
9 for n in range(1,1000):
10 s += 3/(np.pi*n)*(1-(-1)**n)*np.sin((np.pi*n*x)/2)
11
12 g = 1/2 + s
13
14 plt.plot(x,g)
15 plt.show()
21
Answer exercise 22
1 # ex22.py
2 import numpy as np
3 from numpy import pi
4 import matplotlib.pyplot as plt
5
6 # This is a 2D function, so we should compute the solution on a 2D grid. We create x and y
vectors from 0 to 1 and use meshgrid to initialize the grid nodes. We also initialise a
zero-array T, same size as the grid, which we will also use for the summation and to
contain the final result.
7 space= np.linspace(0,1,101)
8 x,y = np.meshgrid(space,space)
9 T = np.zeros_like(x)
10
11 # Note: n should start at 1, not at 0!
12 for n in range(1,100):
13 m = 2*n-1
14 # The computation can be done in a vectorized fashion, i.e. we can compute the entire term
for all grid nodes (x,y) at the same time. Alternatively, you can loop over all grid
nodes, this is slower and less elegant, but will get you to the right solution as well.
15 T += np.sin(m*pi*x) * np.sinh(m*pi*y)/(m*np.sinh(m*pi))
16
17 T = T*4/pi
18
19 # A surface plot does most justice to this kind of result.
20 import matplotlib.cm as cm
21 fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
22 surf = ax1.plot_surface(x, y, T, cmap=cm.viridis,linewidth=0.5,color='black',antialiased=True)
23 fig1.colorbar(surf, shrink=0.5, aspect=5)
24 plt.show()
22
Answer exercise 23
1 # ex23.py
2 import numpy as np
3 import scipy as sp
4 import matplotlib.pyplot as plt
5
6 def duffing(t,x_in,par):
7 # Unpack x_in into velocity and position
8 v,x = x_in
9
10 # Unpack parameter struct
11 gamma = par['gamma']
12 alpha = par['alpha']
13 F = par['F']
14 beta = par['beta']
15 omega = par['omega']
16
17 # Compute first derivative, equal to velocity
18 dxdt = v
19
20 # Second derivative is the equation that was given, rewritten as a first order ODE:
21 dvdt = -2*gamma*v - alpha*x - beta*x**3 + F*np.cos(omega*t)
22
23 # The order here is crucial; we should take the same order as the input, and as the initial
conditions given to solve_ivp
24 return [dvdt,dxdt]
25
26 # Create a dictionary to contain the parameters used; It can also be done in a list, but in
that case the order of the parameters is crucial. Here, we can simply set the value of
alpha, beta at any position in the dictionary.
27 params = {'alpha': 0.01, 'beta': 1.0, 'gamma': 0.4, 'omega': 1.0, 'F': 0.2}
28 init = [0.2,0.009]
29 tspan = [0,20]
30 # The number of time steps can be increased a bit to facilitate a more smooth graph. It does
not add to the accuracy of the solution
31 timesteps = np.linspace(tspan[0],tspan[1], 1001)
32
33 # The actual solver
34 sol = sp.integrate.solve_ivp(duffing,tspan,init,args=(params,),t_eval=timesteps)
35
36 # Plot the results.
37 plt.plot(sol.t,sol.y[0,:],label='Velocity')
38 plt.plot(sol.t,sol.y[1,:],label='Position')
39
40 plt.legend()
41 plt.tight_layout()
42 plt.grid()
43 plt.show()
23
Answer exercise 24
1 # ex24.py
2 import numpy as np
3 import matplotlib.pyplot as plt
4
5 N = 60_000
6 pos = np.zeros((2,N))
7 print(pos)
8
9 for i in range(1,N):
10 r = np.random.uniform(0,1)
11 b = np.zeros((1,2))
12 if r <= 0.01:
13 A = np.array([[0,0],[0,0.16]])
14 elif r <= 0.86:
15 A = np.array([[0.85, 0.04],[-0.04, 0.85]])
16 b = np.array([0,1.6]).T
17 elif r <= 0.93:
18 A = np.array([[0.2, -0.26],[0.23, 0.22]])
19 b = np.array([0,1.6]).T
20 else:
21 A = np.array([[-0.15, 0.28],[0.26, 0.24]])
22 b = np.array([0,0.44]).T
23 pos[:,i] = A@pos[:,i-1] + b
24
25 plt.plot(pos[0,:],pos[1,:],'o',markersize=1,color='#00ff7fff')
26 plt.axis('equal')
27 plt.show()