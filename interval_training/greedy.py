from typing import List

def task1_basic_merge_v1(intervals: List[List[int]]) -> List[List[int]]:
    """
    Basic approach: Merge overlapping intervals
    Input: [[1,3],[2,6],[8,10],[15,18]]
    Output: [[1,6],[8,10],[15,18]]

    Approach: Basic iteration with sorting
    """
    answer = []
    intervals.sort(key=lambda a: a[0])

    for interval in intervals:
        if not answer or interval[0] > answer[-1][1]:
            answer.append(interval)
        else:
            answer[-1][1] = max(interval[1], answer[-1][1])
    return answer
intervals =[[1,3],[2,6],[8,10],[15,18]]
answer_intevals = [[1,6],[8,10],[15,18]]
assert task1_basic_merge_v1(intervals) == answer_intevals
print('success')




