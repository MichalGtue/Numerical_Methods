import numpy as np

def factorial(i):
    nfactorial = 1
    for numb in range(1, i+1):
        nfactorial = nfactorial * numb
    return(nfactorial)



def exp3(j):
    output = 1
    for n in  range(1,26):
        output = output + ((j)**(n))/(factorial(n)) 
    return(output)
print(exp3(-5), np.exp(-5))

print(round(exp3(-5),6))