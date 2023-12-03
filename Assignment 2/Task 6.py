import numpy as np
import timeit
import pandas as pd
import time
def f(t):
    return -0.1 * t**3 + 0.58 * t**2 * np.cosh((-1)/(t+1)) + np.exp(t/2.7) + 9.6

# t is given in months so we are looking for the integral from 0 to 12 (One full year)
# Since we want 20 intervals, every step will be 12/20

def riemann_sum(func, a, b, intervals=20):
    '''Calculates the lower Riemann sum of a function. 
    Takes 3 arguments, first the function, then the lower integration bound, and finally the upper integration bound.
    Optional 4th argument to specify number of intervals (default set to 20)'''
    bin_size = (b-a)/intervals
    sum = 0
    for i in range(intervals):
        sum += func(a+i*bin_size)*bin_size
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



riemann_area = riemann_sum(f,0,12)
riemann_code = lambda:riemann_sum(f,0,12)
end_time_riemann = timeit.timeit(riemann_code, number=1)


trapezoid_area = trapezoid_rule(f,0,12)
trapezoid_code = lambda:trapezoid_rule(f,0,12)
end_time_trapezoid = timeit.timeit(trapezoid_code, number=1)

simpson_area = simpson_rule(f,0,12)
simpson_code = lambda:simpson_rule(f,0,12)
end_time_simpson = timeit.timeit(simpson_code, number=1)

#"Actual" value obtained from graphical display calculator

actual_result = 160.3582902978

def percentage_error(a, b):
    '''Returns the percentage error of the function
    First argument is the observed value and second argument is the expected value'''
    return np.abs((a-b)/(b))*100

rows = [['Riemann Sum', riemann_area, end_time_riemann, percentage_error(riemann_area, actual_result)], ['Trapezoid Rule', trapezoid_area, end_time_trapezoid, percentage_error(trapezoid_area, actual_result)], ['Simpson Rule', simpson_area, end_time_simpson, percentage_error(simpson_area, actual_result)]]


df = pd.DataFrame(rows, columns = ['Solution method', 'Calculated Result', 'Time taken (s)', 'Percentage error from expected value'])


print(df)