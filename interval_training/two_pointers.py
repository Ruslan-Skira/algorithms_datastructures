"""
TODO: 2 Pointers Algorithm - DSA Course in Python Lecture 12
Given an integer array nums sorted in not-descreasing order,
return an array of the
squares of each number sorted in non-decreasing order
"""

from typing import List


class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        left = 0
        right = len(nums) - 1
        result = []

        while left <= right:
            if abs(nums[left]) > abs(nums[right]):
                result.append(nums[left] ** 2)
                lef += 1
            else:
                result.append(nums[right] ** 2)
                right -= 1
        result.reverse()
        return result

    # Time: O(n)
    # Space: O(1)



