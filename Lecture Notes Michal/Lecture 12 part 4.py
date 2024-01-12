from scipy.optimize import linprog
c= [-5,-4,-6]
A=[[1,-1,1],[3,2,4],[3,2,0]]
b = [20,42,30]

bounds = [(0, None), (0, None), (0,None)]

res = linprog(c, A_ub=A, b_ub=b, bounds=bounds)

print(res)