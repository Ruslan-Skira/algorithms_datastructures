import heapq

# Implementation of priority queue
"""Its left child is at 2i + 1

Its right child is at 2i + 2

Its parent is at (i - 1) / 2 (integer division)"""
"""
T1 - 5
T2 - 4
T3 -7
T4 -9
T5 - 2
"""
"""min heap the min value the bigerr priority"""

data = [10, 4, 3, 95, 29, 3, 5]
heapq.heapify(data)
print(data) #[3, 4, 3, 95, 29, 10, 5]
heapq.heappush(data, 2)
print(data) # 2, 3, 3, 4, 29, 10, 5, 95]

heapq.heappush(data, 19)
print(data) # [2, 3, 3, 4, 29, 10, 5, 95, 19]

l1 = [5,3,2,5]
l2 = [5,3,2,5]
l3 = heapq.merge(l1, l2)
print([f for f in l3])

