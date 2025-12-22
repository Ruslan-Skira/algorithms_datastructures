"""
have the function that take the array of number
stored in arr and determine the total number of duplicate entries.
For example if the input is [1, 2, 2,  2, 3]
the programm should output 2 because there are two duplicates of one of the elements.
Examples:
Input: [1, 2, 2, 2, 3]
Output: 2
Input: [1, 1, 1, 1, 1]
Output: 4
Input: [0, -2, -2, 5, 5, 5]
Output: 3
"""
import asyncio

async def duplicates(arr):
    """
    Counts the total number of duplicates """
    if not arr :
        return
    counter = {}
    for i in arr:
        counter[i]=counter.get(i, 0) + 1

    answer = 0
    for dupl_qt in counter.values():
        if dupl_qt > 1:
            answer += dupl_qt - 1
    return answer


input1 = [1, 2, 2, 2, 3,3]
print(asyncio.run(duplicates(input1)))
# Space complexity O(n)
#  time  complexity O(n)

##########

# async def duplicates2(arr):
#     """
#     Counts the total number of duplicates """
#     if not arr :
#         return
#     counter = {k: arr.count(k) for k in set(arr)}

#     answer = 0
#     for dupl_qt in counter.values():
#         if dupl_qt > 1:
#             answer += dupl_qt - 1
#     return answer


# input1 = [1, 2, 2, 2, 3,3]
# print(asyncio.run(duplicates2(input1)))




