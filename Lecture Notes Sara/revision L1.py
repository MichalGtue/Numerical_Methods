s = "Python programming is fun!"

# Find and print the index of the word "is"
index_of_is = s.find("is")
print("Index of 'is':", index_of_is)

# Create a new string where "fun" is replaced with "awesome"
new_string = s.replace("fun", "awesome")
print("String with 'fun' replaced:", new_string)

# Print the string in uppercase
uppercase_string = s.upper()
print("Uppercase string:", uppercase_string)

my_list=[1,3,7,8,9]

for i in my_list:
    print(i)

c=0
while (c<4):
    print(f"{my_list[c]=}")
    c +=1

while not my_list[c] == 7:
    c +=1
    print(my_list[c])


my_list_1 = [x**3 for x in range(1,25,2)]

for i in my_list_1:
    if i == 125:
        print('125 is an element of the list')

print(my_list_1[3]%3)

    def factorial(n):
        if n==0 or n==1:
            return 1
        else:
            return n*factorial(n-1)
        
    def compute_exp(x):
        result = 1.0
        term = 1.0
        n = 1

        while abs(term) > 1e-6:
            term *= x / n
            result += term
            n += 1

        return result

import random
from collections import Counter
N=1000
def simulate_dice_throws(N):
        # Simulate N dice throws
    throws = [random.randint(1, 6) for _ in range(N)]

        # Count the occurrences of each value
    value_counts = Counter(throws)

        # Print the results
    print(f"Results of {N} dice throws:")
    for value, count in value_counts.items():
        print(f"Value {value}: {count} times")