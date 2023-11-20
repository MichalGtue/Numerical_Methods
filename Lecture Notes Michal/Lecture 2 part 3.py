import math
import numpy as np
import time
f = lambda x: x**2 + 2*x -4



xloop= [i/(10**6) for i in range(0,20*(10**6))]
startimeloop = time.time()
for i in xloop:
    y = f(i)
finaltimeloop = time.time() - startimeloop

print(f'{finaltimeloop = }')
x = np.linspace(0,21, 10**6)
starttimenp = time.time()
y = f(x)

totaltimenp = time.time() - starttimenp

print(f'{totaltimenp = }')
