"""
step 1 - Find the smallest element in array and exchange it with the element in the first
position.
step 1 - Find the smallest element in the left array without prevouse value and
exchange it with the element in the second position"""

from random import randint

array_int = [randint(0, 100) for i in range(10)]
print(array_int)

for current_pos in range(len(array_int)):
    min_i = current_pos
    for i in range(current_pos + 1, len(array_int)):
        if array_int[i] < array_int[min_i]:
            min_i = i
    array_int[min_i], array_int[current_pos] = array_int[current_pos], array_int[min_i]
print(all(array_int[i] <= array_int[i + 1] for i in range(len(array_int) - 1)))
