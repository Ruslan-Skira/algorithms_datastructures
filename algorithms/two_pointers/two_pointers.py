"""Squares of a Sorted Array
Given an integer array nums sorted in non-decreasing(encreasing with duplicates) order,
return an array of the suqares of each number sorted in non decreadsing order
Example 1:
Input: nums = [-4, -1, 0, 3, 10]
Output: [0, 1, 9, 16, 100]
Exlanation: After squaring, the array becomes [16, 1, 0, 9, 100].
After sorting, it becomes [0, 1, 9, 16, 100].
Example 2:
Input: nums = [-7, -3, 2, 3, 11]
Output: [4, 9, 9, 49, 121]
Constraints:
1 <= nums.length<=10^4
-10^4 <= nums[i] <= 10^4
nums is sorted in non-descreasing order
"""
from typing import List
class Solution:
    def sorted_squares(self, nums:List[int]) -> List[int]:
        left = 0
        right = len(nums)-1
        result = []
        while left <= right:
            if abs(nums[left]) > abs(nums[right]):
                result.append(nums[left]**2)
                left +=1
            else:
                result.append(nums[right]**2)
                right -= 1
        result.reverse()
        return result

s = Solution()
nums = [-4, -1, 0, 3, 10]
answer =s.sorted_squares(nums)
print(answer)
assert answer == [0, 1, 9, 16, 100], 'Not right sorting or answer.'

"""two-pointers algorithm for sorting squares:

Time Complexity: O(n)
The while loop runs exactly n times (where n is the length of the array)
Each iteration processes one element from either the left or right pointer
result.reverse() takes O(n) time
Overall: O(n) + O(n) = O(n)
Space Complexity: O(n)
The result array stores n elements
The algorithm uses constant extra space O(1) for variables (left, right)
Overall: O(n) for the output array
This is an optimal solution for this problem. The two-pointers approach is more efficient than the naive approach of squaring all elements and then sorting, which would be O(n log n) time complexity.

The algorithm works efficiently because it leverages the fact that the input array is already sorted, allowing you to compare absolute values from both ends and build the result in descending order of squares, then reverse it at the end."""


