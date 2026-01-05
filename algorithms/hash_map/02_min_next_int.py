"""Create algorithm that return one integer. It will be the smallest integer that is not present in the given list of integers.
Given a list of integers, find the smallest positive integer that is not present in the list.
For example, given the list [1, 2, 0], the smallest missing positive integer is 3.
if the list is [-1, -3], the smallest missing positive integer is 1.
integers could be positive, negative or zero.
List could be unsorted and contain duplicates.
List could be empty.
List could contain large integers.
List could contain only negative integers.
List could contain only positive integers.
List could contain a mix of positive and negative integers.
Example:
Input: [3, 4, -1, 1]
Output: 2
Input: [1, 2, 0]
Output: 3
"""

"""What this REALLY is:
The optimal solution uses hash set or cyclic sort, not greedy or two-pointers:

Hash Set approach (most common): Convert to set, iterate from 1 upward until you find missing number

Time: O(n), Space: O(n)
Cyclic Sort approach (optimal): Place each number in its "correct" position (e.g., 1 at index 0, 2 at index 1), then find the first missing

Time: O(n), Space: O(1)"""


# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")


def solution(A):
    """
      Example test:   [1, 3, 6, 4, 1, 2]
    OK

    Example test:   [1, 2, 3]
    OK

    Example test:   [-1, -3]
    """
    # Implement your solution here
    A.sort()
    array_int = list(set(A))
    if not list(filter(lambda a: a > 0, array_int)):
        return 1
    current = array_int[0]
    for i in range(len(array_int)):
        if array_int[i] != current:
            return array_int[i - 1] + 1
        else:
            current += 1
    return current

def smallest_missing_positive_integer(nums):
    """
    Hash Set
    Convert the list to a set for o(1) lookups.
    Start checking from 1 upwards to find the smallest missing positive integer.
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    num_set = set(nums)
    smallest_missing = 1

    while smallest_missing in num_set:
        smallest_missing += 1

    return smallest_missing

assert smallest_missing_positive_integer([3, 4, -1, 1]) == 2
assert smallest_missing_positive_integer([1, 2, 0]) == 3
assert smallest_missing_positive_integer([1, 3, 6, 4, 1, 2]) == 5
assert smallest_missing_positive_integer([1, 2, 3]) == 4
assert smallest_missing_positive_integer([-1, -3]) == 1
assert smallest_missing_positive_integer([]) == 1
assert smallest_missing_positive_integer([-5, -2, -1]) == 1
# V2

def cyclic_sort(nums): # TODO: not working properly need to understand how it works.
    """
    Cyclic Sort

    """
    n = len(nums)
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            correct_i = nums[i] - 1
            nums[i], nums[correct_i] = nums[correct_i], nums[i]
    return nums

print(cyclic_sort([3, 4, -1, 1]))  # Expected: [1, -1, 3, 4]
print(cyclic_sort([1, 2, 0]))      # Expected: [1, 2, 0]
print(cyclic_sort([1, 3, 6, 4, 1, 2]))  # Expected: [1, 1, 2, 3, 4, 6]
assert cyclic_sort([3, 1, 5, 4, 2]) == [1, 2, 3, 4, 5]



def cyclic_sort_smallest_int(nums):
    """
    Cyclic Sort
    Place each number in its correct position (1 at index 0, 2 at index 1, etc.).
    Then find the first index where the number is not correct.
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    n = len(nums)
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            correct_index = nums[i] - 1
            nums[i], nums[correct_index] = nums[correct_index], nums[i]

    for i in range(n):
        if nums[i] != i + 1:
            return i + 1

    return n + 1
