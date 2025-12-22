import random


def binary_search(data, target):
    l_d = len(data)
    left, right = 0, l_d-1
    while left <= right:
        middle = (left + right) // 2

        # if data[left]==target:
        #     return left
        # if data[right]==target:
        #     return right
        if data[middle] < target:
            left = middle + 1
        elif data[middle] > target:
            right = middle - 1
        elif data[middle] == target:
            return middle
    return -1


n = 10
max_val = 100
data = [random.randint(1, max_val) for i in range(n)]
data.sort()
print("Data:", data)
target = int(input("Enter target value: "))
target_pos = binary_search(data, target)
if target_pos == -1:
    print("Your target value is not in the list.")
else:
    print("You target value has been found at index", target_pos)