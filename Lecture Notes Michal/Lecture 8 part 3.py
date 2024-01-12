import numpy as np
import timeit
import pandas as pd

def f(t):
    return -0.1 * t**3 + 0.58 * t**2 * np.cosh((-1)/(t+1)) + np.exp(t/2.7) + 9.6

# t is given in months so we are looking for the integral from 0 to 12 (One full year)
# Since we want 20 intervals, every step will be 12/20

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

