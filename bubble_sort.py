l = [5,48, 6, 1,2,5,4,3,99,23]
"""
time complexity is O(n**2)
Space Complexity O(1)
"""
for i in range(len(l)-1, 0, -1):
    for j in range(i):
        if l[j] > l[j+1]:
            l[j], l[j+1] = l[j+1], l[j]

def opt_bubble_sort(arr):
    l_a = len(arr)
    for i in range(l_a-1, 0, -1):
        swapped = False
        for j in range(i):
            if l[j] > l[j+1]:
                l[j], l[j+1] = l[j+1], l[j]
                swapped = True
            if not swapped:
                break
            



