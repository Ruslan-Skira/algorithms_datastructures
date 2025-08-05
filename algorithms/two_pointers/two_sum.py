"""Two Sum (easy)
Starting with a sorted array of integers, find a pair of numbers that sum to the given target.
https://www.linkedin.com/pulse/two-pointer-technique-guide-visual-learners-jimmy-zhang-toxie/
"""

from typing import List


class Solution:
    def two_sum(self, l: List, target: int) -> List:
        left = 0
        right = len(l) - 1
        while left < right:
            current_sum = l[left] + l[right]
            if current_sum == target:
                return [l[left], l[right]]
            elif current_sum > target:
                right -= 1
            elif current_sum < target:
                left += 1

        return []


s = Solution()
l = [1, 3, 4, 6, 8, 10, 13]
target = 13
answer = s.two_sum(l, target)

assert answer == [3, 110], f"{answer=},\n {target=}"
