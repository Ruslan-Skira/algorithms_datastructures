"""Given an integer input array heights representing the heights of vertical lines,
write a function that returns the maximum area of water that can be contained
by two of the lines (and the x-axis). The function should take in an array
of integers and return an integer.
Hint: Instead of summing the elements at each pointer, compare their values instead.
Which containers can you eliminate?"""

from typing import List


class Solution:
    def max_area(self, heigts: List[int]) -> int:
        left = 0
        right = len(heigts) - 1
        max_area = 0
        while left < right:
            width = right - left
            area = width * (min(heigts[left], heigts[right]))
            max_area = max(area, max_area)
            # You should always move pointers
            if heigts[left] < heigts[right]:
                left += 1
            else:
                right -= 1

        return max_area


s = Solution()

print(answer := s.max_area([3, 4, 1, 2, 2, 4, 1, 3, 2]))
assert answer == 21
