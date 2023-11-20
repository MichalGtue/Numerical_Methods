#example 1
x=0.1
for _ in range (30):
    x= (10*x-0.9)
    print (x)

def compute_silly_range(N):
    x= [0.1]
    for _ in range(N):
        x.append((10*x[-1])-0.9)
    return x

my_list = compute_silly_range(40)
print(my_list)

#example 2
import numpy as np 
import matplotlib.pyplot as plt

v = np.logspace(0, 40, 41)
y = np.sin(v * np.pi)
plt.loglog(v, np.abs(y)) # Double log plot, values on y-axis must be positive.
plt.show()

i=1
while True:
    i+=1
print (i)

print(bin(214))
print(hex(214))
print(oct(214))


np.iinfo(np.int32).min
np.iinfo(np.int32).max
i = np.int16(np.iinfo(np.int32).max)
type(i)
i = i + 100
np.finfo(np.float64).max
f=0.1
type(f)
print(":.16f".format(f))
print(":.20f".format(f))

f=0.1
type(f)
print(f'{g:1.16e}') # 16 digits after deciman point

i=2.0**1000
for i in range(200):
    i *= 2
    print(i)
