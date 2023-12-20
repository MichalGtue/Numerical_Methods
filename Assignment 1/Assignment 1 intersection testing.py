import numpy as np
import sympy


x = sympy.symbols('x', positive=True)

boundry_condition = 12*sympy.cos(0.1*sympy.pi)

f1 = sympy.Piecewise((sympy.sqrt(12**2 - x**2), x >= boundry_condition), (0, True))


slope = (1.1-1)/(11.5-11.6)
y_intercept = 1.1 - (slope)*11.5

f2 = slope * x + y_intercept ## only intersects exit

f3= 2*x-20 ## Intersects x axis and exit

f4 = x


f5 = 2*x -15
solution = sympy.solve(f1 - f3, x)

#print(solution[0])

print(solution)
if len(solution)==1 and solution[0] < boundry_condition:
    print('Fail')
elif len(solution)==1 and solution[0] >= boundry_condition:
    print('Success')
elif len(solution)==0:
    print('fail')
elif len(solution) == 2 and solution[1] >= boundry_condition:
    print("success1")

print(len(solution))

print(solution)
#print(f2)
#sol2 = sympy.solve(f4-f3, x)

#print(sol2)