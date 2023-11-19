import numpy as np

print(np.__version__)

a = list(range(2,21,2))
b = list(range(30,90,10))

x = a + b



s = "Python programming is fun!"

print(s.index('is'))
p = s.replace('fun', 'awesome')
print(p)
print(p.upper())

print(s)

t = (1, 2, 3, 4, 5, 6)
print(t[2])



t_new = t + (7, 8, 9)
print(t_new)

t1 = (1, )

d = {'Alice': 24, 'Bob': 27, 'Charlie': 22, 'Dave': 30}
d['Charlie'] = 36
d['Alice'] = 25
d['Eve'] = 29

print(d.keys())

#for i in range(5):
#    print(i)


#h = 0
#while h<3:
#    print(h)
#    h += 1
my_list = [1, 3, 7, 8, 9]

#for i in my_list:
#    print(i)

c=0
while (c<4):
    print(f"{my_list[c]=}")
    c +=1

#while not my_list[c] == 7:
#    c +=1
#    print(my_list[c])


e = [i for i in range(0,31) if i % 3 == 0]

print(e)

my_list_1 = [x**3 for x in range(1,25,2)]


for i in my_list_1:
    if i == 125:
        print('True')


print(my_list_1[3] % 3)


not_div = []
for p in my_list_1:
    if p%5 != 0:
        not_div.append(p)
print(not_div)
