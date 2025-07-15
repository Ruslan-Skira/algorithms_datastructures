intervals = [[1, 3], [2, 6], [1, 5], [8, 10], [9, 12]]

#TODO: finish this logic.
"""
Compare first intex and then update second index."""
# res: [[1, 6], [8, 12]]
intervals = sorted(intervals, key=lambda i:i[0] )

# print(srt_intervals)
"""
intersections of intervals need to combine.
[1,2] union [2,4] => [1,4]
[1,5] union [2,6] => []

;
"""
#TODO: v1
def union_intervals_v1(intervals):
    answer = []
    for interval in intervals:
        c_start, c_end = interval
        current_interval=list(range(c_start, c_end+1))
        for i in range(1, len(intervals)):
            s, e = intervals[i]
            if s in current_interval:

                current_interval = range(c_start, max(e, c_end)+1)
        answer.append([current_interval[0], current_interval[-1]])
    return answer


#Solving of v1
def union_intervals_v1(intervals):
    answer = []
    for interval in intervals:
        c_start, c_end = interval
        if not answer or answer[-1][1] < c_start:
            answer.append([c_start, c_end])
        else:
            answer[-1][1] = max(answer[-1][1], c_end)
    return answer




print(union_intervals_v1(intervals))
#TODO: v2
# for f, s in intervals:
#     o = range(f, s)






