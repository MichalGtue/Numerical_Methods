# The sole purpose of this file is to generate the table to compare methods from task 6 to the methods from task 7.
# As this is a continuation of task 7, the comments have been removed. They can be found in task 7.py
# Libraries Required
import numpy as np
import pandas as pd
import numpy as np
import scipy
from scipy import optimize
import pandas as pd


#Functions
# We know that we could use for example "import Task 6" to get the functions but we wanted to make sure that
# the code runs properly on all devices.
# Thus we simply copy-pasted the funtions into this file to make sure that the correctors can run the code perfectly
def f(t):
    return -0.1 * t**3 + 0.58 * t**2 * np.cosh((-1)/(t+1)) + np.exp(t/2.7) + 9.6

def riemann_sum_lower(func, a, b, intervals=20):
    '''Calculates the lower Riemann sum of a function. 
    Takes 3 arguments, first the function, then the lower integration bound, and finally the upper integration bound.
    Optional 4th argument to specify number of intervals (default set to 20)'''
    bin_size = (b-a)/intervals
    sum = 0
    for i in range(intervals):
        sum += func(a+i*bin_size)*bin_size
    return sum

def riemann_sum_upper(func, a, b, intervals=20):
    '''Calculates the upper Riemann sum of a function. 
    Takes 3 arguments, first the function, then the lower integration bound, and finally the upper integration bound.
    Optional 4th argument to specify number of intervals (default set to 20)'''
    bin_size = (b-a)/intervals
    sum = 0
    for i in range(intervals):
        sum += func(a+(i+1)*bin_size)*bin_size
    return sum

def riemann_sum_middle(func, a, b, intervals=20):
    '''Calculates the middle Riemann sum of a function. 
    Takes 3 arguments, first the function, then the lower integration bound, and finally the upper integration bound.
    Optional 4th argument to specify number of intervals (default set to 20)'''
    bin_size = (b-a)/intervals
    sum = 0
    for i in range(intervals):
        sum += func((a+i*bin_size + a+(i+1)*bin_size)/2)*bin_size
    return sum

def trapezoid_rule(func, a, b, intervals=20):
    '''Calculates the area of a function using the trapezoid rule. 
    Takes 3 arguments, first the function, then the lower integration bound, and finally the upper integration bound.
    Optional 4th argument to specify number of intervals (default set to 20)'''
    bin_size = (b-a)/intervals
    sum = 0
    for i in range(intervals):
        sum += (bin_size * (func(a+i*bin_size) + func(a+(i+1)*bin_size)))/2
    return sum

def simpson_rule(func, a, b, intervals=20):
    '''Calculates the area of a function using the Simpson rule. 
    Takes 3 arguments, first the function, then the lower integration bound, and finally the upper integration bound.
    Optional 4th argument to specify number of intervals (default set to 20)'''
    bin_size = (b-a)/intervals
    sum = 0
    for i in range(intervals):
        a_loop = a+i*bin_size
        b_loop = a+(i+1)*bin_size
        sum += (func(a_loop)+4*func((a_loop+b_loop)/(2))+func(b_loop)) * ((b_loop-a_loop)/(6))
    return sum

def find_max(f,a,b):
    '''Finds the maximum of a function on a given domain.
    First argument is your function, second is the lowerbound, and third is the upperbound.'''
    minimize_result = scipy.optimize.minimize_scalar(lambda x: -f((b-a)*x), bounds=[0,1], method='bounded') #Finds maximum of a function. According to information found online it has to be bound from 0 to 1 thus the function is shrunk.
    return -minimize_result.fun # Returns the maximum y value of a function.

def monte_carlo(func, a, b, N=1000):
    '''Returns the approximate area under a curve using the Monte Carlo method.
    First argument is the function you want to integrate.
    Second argument is the lower bound and third is the upperbound.
    Optional third argument to specify how many random points to choose with default set to 1000.'''
    approx_area_counter = 0
    y_max = find_max(func, a, b)
    for i in range(N):
        x_rand = np.random.uniform(a,b) ## random number on the given domain
        y_rand = np.random.uniform(0,y_max) ## Area of the shape lies between y = 0 and y = ymax
        if y_rand < f(x_rand):
            approx_area_counter += 1
    approx_area = (y_max*(b-a)*approx_area_counter) / N
    return approx_area

def percentage_error(a, b):
    '''Returns the percentage error of the function
    First argument is the observed value and second argument is the expected value'''
    return np.abs((a-b)/(b))*100

riemann_area_lower = riemann_sum_lower(f,0,12)

riemann_area_upper = riemann_sum_upper(f,0,12)

riemann_area_middle = riemann_sum_middle(f,0,12)

trapezoid_area = trapezoid_rule(f,0,12)

simpson_area = simpson_rule(f,0,12)

actual_result = 160.3582902978


rows = [['Lower Riemann Sum', riemann_area_lower, percentage_error(riemann_area_lower, actual_result)], ['Upper Riemann Sum', riemann_area_upper, percentage_error(riemann_area_upper, actual_result)], ['Middle Riemann Sum', riemann_area_middle,  percentage_error(riemann_area_middle, actual_result)], ['Trapezoid Rule', trapezoid_area,  percentage_error(trapezoid_area, actual_result)], ['Simpson Rule', simpson_area, percentage_error(simpson_area, actual_result)]]


big_numbers_for_table = [1000, 2000, 5000, 10000, 20000, 50000, 100000]

for i in big_numbers_for_table:
    area_loop = monte_carlo(f,0,12,i)
    error_loop = percentage_error(area_loop, actual_result)
    rows.append(['Monte-Carlo with N='+str(i),area_loop, error_loop])




df = pd.DataFrame(rows, columns = ['Method', 'Calculated Area', 'Percentage error from expected value'])

print(df)