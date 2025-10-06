"""
Find the smallest element in the array and exchange it with the elemnt in
the first position
find the second smallest element in the array and exchange it  with
the element in the second position.
"""

from random import randint


def find_min(data):
    min_index = 0
    for i in range(len(data)):
        if data[min_index] > data[i]:
            min_index = i
    return data[min_index]


def selection_sort(data):
    for current_index in range(len(data)-1):
        min_index = current_index
        for i in range(current_index+1, len(data)):
            if data[min_index] > data[i]:
                min_index = i

        data[current_index], data[min_index] = data[min_index], data[current_index]
    # return data


data_array = [randint(0, 10) for i in range(10)]
print(data_array)
selection_sort(data_array)
print(data_array)
# print(find_min(data_array))
# assert all(data_array[i] <= data_array[i + 1] for i in range(len(data_array) - 1))
