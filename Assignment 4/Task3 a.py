import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Simpson rule from assignment 2 task 6 slighly modified
def simpson_rule(xlist, ylist):
    '''Calculates the area of a function using the Simpson rule. \n
    Takes 3 arguments, first the function, then the lower integration bound, and finally the upper integration bound. \n
    Optional 4th argument to specify number of intervals (default set to 20)'''
    sum = 0
    dx = (xlist[1]-xlist[0])
    for i in range(len(ylist)):
        if i>0 and i<len(ylist)-1:
            sum += (dx/6) * (ylist[i-1]+4*ylist[i]+ylist[i+1])
    return sum



# Known that the integral from 0 to 1 should be 2



N = np.linspace(3, 100, 70, dtype=int)
solution_list = []
error_list = []
exact_sol = 2
rate_of_convergence = ['N/A']
for i in range(len(N)):
    x = np.linspace(0,np.pi,N[i])
    y = np.sin(x)
    approx_sol = simpson_rule(x,y)
    solution_list.append(approx_sol)
    rel_error = np.abs((approx_sol - exact_sol)/exact_sol)
    error_list.append(rel_error)
    if i > 0:  # Doesn't make sense to calculate for the first point
        r = (np.log(error_list[i]/error_list[i-1]))/(np.log(N[i-1] / N[i]))
        rate_of_convergence.append(r)

rows = []
for i in range(len(N)):
    rows.append([N[i], solution_list[i], error_list[i], rate_of_convergence[i]])

df = pd.DataFrame(rows, columns = ['N_t', 'Calculated solution', 'epsilon_rel', 'Rate of Convergence'])
print(df)



figure, ax = plt.subplots(figsize=(10, 5))
ax.plot(N[1:], rate_of_convergence[1:], label='Rate of Convergence')
ax.plot(N, error_list, label='Relative Error')
plt.title("Simpson rule", fontsize=16)
plt.xlabel('Number of steps', fontsize=14)
plt.ylabel('Output value', fontsize=14)
plt.xlim(0, max(N))
plt.ylim(0, 2.5)
plt.legend()
plt.grid()
plt.show()