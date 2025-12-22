"""
Have the function ArrayChallenge(arr) take the array of numbers
stored in arr and return the number that occurs most frequently (the mode).
For example: if arr is [10, 4, 5, 2, 4] then your program should return 4.
If there is more than one mode return the one that appeared in the array first
(ie. [5, 10, 10,6, 5] should return 5 because it appeared first).
The array will not be empty. The numbers will be in the range 1-1000.
If there is no mode return -1.
Examples
Input: [3, 3, 1, 3, 2, 1]
Output: 3
Input: [3, 1, 4, 4, 5, 5, 6, 6]
Output: 4
Input: [3, 1, 4, 5, 6]
Output: -1

Notes:
the term 'mode' is used in statistics to describe the value that
appears most frequently in a data set. The most frequent value.
"""
from typing import List
from collections import Counter

def frequency_counting(arr: List[int]) -> int:
    """
    Finds the mode of an array of numbers.

    Args:
        arr (List[int]): List of integers.

    Returns:
        int:  The number that appears most frequently (mode).
            If there is no mode, returns -1.
            If multiple modes, returns the one that appeared first.
    """
    if not arr:
        return -1
    counts =  Counter(arr)
    max_count = max(counts.values())
    if max_count == 1:
        return -1
    for n in counts:
        if counts[n] == max_count:
            return n


input1 = [5,5,2,2,2,1]
print(frequency_counting(input1))

"""Time complexity O(n)
Space complexity O(n)"""
answer = lambda input1: {k:input1.count(k) for k in set(input1)} # if don't want to sue Counter.
print(answer(input1))


