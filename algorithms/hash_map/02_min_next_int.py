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
