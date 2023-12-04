#Libraries
import numpy as np
import scipy
import pandas as pd

#Same function as in task 6
def f(t):
    return -0.1 * t**3 + 0.58 * t**2 * np.cosh((-1)/(t+1)) + np.exp(t/2.7) + 9.6

#We need to find its maximum
def find_max(f,a,b):
    '''Finds the maximum of a function on a given domain.
    First argument is your function, second is the lowerbound, and third is the upperbound.'''
    minimize_result = scipy.optimize.minimize_scalar(lambda x: -f((b-a)*x), bounds=[0,1], method='bounded') #Finds maximum of a function. According to information found online it has to be bound from 0 to 1 thus the function is shrunk.
    return -minimize_result.fun # Returns the maximum y value of a function.

def find_min(f,a,b):
    '''Finds the minimum of a function on a given domain.
    First argument is your function, second is the lowerbound, and third is the upperbound.'''
    minimize_result = scipy.optimize.fmin(lambda x: f((b-a)*x), bounds=[0,1], method='bounded') #Finds maximum of a function. According to information found online it has to be bound from 0 to 1 thus the function is shrunk.
    return minimize_result.fun # Returns the minimum y value of a function.

find_min(f,0,12)



#From gdc, y max ~= 18.394515656
#print(find_max(f,0,12)) #Uncomment in case you want to verify that the function works


def monte_carlo(func, a, b, N=1000):
    '''Returns the approximate area under a curve using the Monte Carlo method.
    First argument is the function you want to integrate.
    Second argument is the lower bound and third is the upperbound.
    Optional third argument to specify how many random points to choose with default set to 1000.'''
    approx_area_counter = 0
    y_max = find_max(func, a, b)
    for i in range(N):
        x_rand = np.random.uniform(a,b) ## random number on the given domain
        y_rand = np.random.uniform(0,y_max) ## Theoeretically any upperbound could be chosen but that may skew the results
        if y_rand < f(x_rand):
            approx_area_counter += 1
    approx_area = (y_max*(b-a)*approx_area_counter) / N
    return approx_area

#Taken from task 6, needed for the table
def percentage_error(a, b):
    '''Returns the percentage error of the function
    First argument is the observed value and second argument is the expected value'''
    return np.abs((a-b)/(b))*100





#Same as task 6, taken from graphical display calculator.
actual_result = 160.3582902978


rows = []
big_numbers_for_table = [1000, 2000, 5000, 10000, 20000, 50000, 100000]

for i in big_numbers_for_table:
    area_loop = monte_carlo(f,0,12,i)
    error_loop = percentage_error(area_loop, actual_result)
    rows.append([area_loop, i, error_loop])

df = pd.DataFrame(rows, columns = ['Total discharge into lake superior (kg)', 'Number of Iterations', 'Percentage error from expected value'])

print(df)