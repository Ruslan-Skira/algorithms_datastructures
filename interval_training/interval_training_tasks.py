"""
Interval Training Tasks - From Easy to Hard
=========================================

This file contains 10 interval-related problems with multiple solution approaches,
progressing from basic to advanced techniques using different Python libraries.
"""

import bisect
import heapq
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple

# =============================================================================
# TASK 1: BASIC INTERVAL MERGE (EASY)
# =============================================================================


def task1_basic_merge_v1(intervals: List[List[int]]) -> List[List[int]]:
    """
    Basic approach: Merge overlapping intervals
    Input: [[1,3],[2,6],[8,10],[15,18]]
    Output: [[1,6],[8,10],[15,18]]

    Approach: Basic iteration with sorting
    """
    if not intervals:
        return []

    intervals.sort()
    result = [intervals[0]]

    for current in intervals[1:]:
        if result[-1][1] >= current[0]:  # Overlap
            result[-1][1] = max(result[-1][1], current[1])
        else:
            result.append(current)

    return result


def task1_basic_merge_v2(intervals: List[List[int]]) -> List[List[int]]:
    """
    Using collections.deque for efficient append/pop operations just learn deque!
    """
    if not intervals:
        return []

    intervals.sort()
    result = deque([intervals[0]])

    for current in intervals[1:]:
        last = result[-1]
        if last[1] >= current[0]:
            result[-1] = [last[0], max(last[1], current[1])]
        else:
            result.append(current)

    return list(result)   # O(n) - EXTRA OVERHEAD!


# =============================================================================
# TASK 2: INSERT INTERVAL (EASY-MEDIUM)
# =============================================================================


def task2_insert_interval_v1(
    intervals: List[List[int]], newInterval: List[int]
) -> List[List[int]]:
    """
    Insert new interval and merge if necessary
    Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
    Output: [[1,5],[6,9]]

    Approach: Basic three-part logic
    """
    result = []
    i = 0

    # Add intervals before newInterval
    while i < len(intervals) and intervals[i][1] < newInterval[0]:
        result.append(intervals[i])
        i += 1

    # Merge overlapping intervals
    while i < len(intervals) and intervals[i][0] <= newInterval[1]:
        newInterval[0] = min(newInterval[0], intervals[i][0])
        newInterval[1] = max(newInterval[1], intervals[i][1])
        i += 1

    result.append(newInterval)

    # Add remaining intervals
    while i < len(intervals):
        result.append(intervals[i])
        i += 1

    return result


def task2_insert_interval_v2(
    intervals: List[List[int]], newInterval: List[int]
) -> List[List[int]]:
    """
    Using bisect module for efficient insertion point finding
    """
    if not intervals:
        return [newInterval]

    # Find insertion points
    left = bisect.bisect_left(intervals, [newInterval[0]])
    right = bisect.bisect_right(intervals, [newInterval[1]])

    # Check for overlaps and merge
    start = newInterval[0]
    end = newInterval[1]

    # Check left boundary
    if left > 0 and intervals[left - 1][1] >= newInterval[0]:
        left -= 1
        start = intervals[left][0]

    # Check right boundary
    if right < len(intervals) and intervals[right][0] <= newInterval[1]:
        right += 1
        end = intervals[right - 1][1]

    # Merge all overlapping intervals
    for i in range(left, right):
        start = min(start, intervals[i][0])
        end = max(end, intervals[i][1])

    return intervals[:left] + [[start, end]] + intervals[right:]


# =============================================================================
# TASK 3: NON-OVERLAPPING INTERVALS (MEDIUM)
# =============================================================================


def task3_remove_intervals_v1(intervals: List[List[int]]) -> int:
    """
    Find minimum number of intervals to remove to make rest non-overlapping
    Input: [[1,2],[2,3],[3,4],[1,3]]
    Output: 1 (remove [1,3])

    Approach: Greedy - sort by end time
    """
    if not intervals:
        return 0

    intervals.sort(key=lambda x: x[1])  # Sort by end time
    count = 0
    end = intervals[0][1]

    for i in range(1, len(intervals)):
        if intervals[i][0] < end:  # Overlap
            count += 1
        else:
            end = intervals[i][1]

    return count


def task3_remove_intervals_v2(intervals: List[List[int]]) -> int:
    """
    Using heap to track active intervals
    """
    if not intervals:
        return 0

    intervals.sort()
    heap = []
    removed = 0

    for start, end in intervals:
        # Remove intervals that end before current starts
        while heap and heap[0] <= start:
            heapq.heappop(heap)

        # If there's an overlap, remove the one with later end time
        if heap:
            if heap[0] > start:  # Overlap
                if end < heap[0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap, end)
                removed += 1
                continue

        heapq.heappush(heap, end)

    return removed


# =============================================================================
# TASK 4: MEETING ROOMS (MEDIUM)
# =============================================================================


def task4_can_attend_meetings_v1(intervals: List[List[int]]) -> bool:
    """
    Check if person can attend all meetings
    Input: [[0,30],[5,10],[15,20]]
    Output: False (conflicts)

    Approach: Sort and check consecutive intervals
    """
    if not intervals:
        return True

    intervals.sort()

    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i - 1][1]:  # Overlap
            return False

    return True


def task4_min_meeting_rooms_v1(intervals: List[List[int]]) -> int:
    """
    Find minimum number of meeting rooms needed
    Input: [[0,30],[5,10],[15,20]]
    Output: 2

    Approach: Sweep line algorithm
    """
    if not intervals:
        return 0

    events = []
    for start, end in intervals:
        events.append((start, 1))  # Meeting starts
        events.append((end, -1))  # Meeting ends

    events.sort()

    active_meetings = 0
    max_rooms = 0

    for time, event_type in events:
        active_meetings += event_type
        max_rooms = max(max_rooms, active_meetings)

    return max_rooms


def task4_min_meeting_rooms_v2(intervals: List[List[int]]) -> int:
    """
    Using heap to track meeting end times
    """
    if not intervals:
        return 0

    intervals.sort()
    heap = []  # Min heap to track meeting end times

    for start, end in intervals:
        # Remove meetings that have ended
        while heap and heap[0] <= start:
            heapq.heappop(heap)

        # Add current meeting's end time
        heapq.heappush(heap, end)

    return len(heap)


# =============================================================================
# TASK 5: INTERVAL INTERSECTION (MEDIUM)
# =============================================================================


def task5_interval_intersection_v1(
    firstList: List[List[int]], secondList: List[List[int]]
) -> List[List[int]]:
    """
    Find intersection of two interval lists
    Input: firstList = [[0,2],[5,10],[13,23],[24,25]], secondList = [[1,5],[8,12],[15,24],[25,26]]
    Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]

    Approach: Two pointers
    """
    result = []
    i = j = 0

    while i < len(firstList) and j < len(secondList):
        # Find intersection
        start = max(firstList[i][0], secondList[j][0])
        end = min(firstList[i][1], secondList[j][1])

        if start <= end:  # Valid intersection
            result.append([start, end])

        # Move pointer of interval that ends first
        if firstList[i][1] < secondList[j][1]:
            i += 1
        else:
            j += 1

    return result


def task5_interval_intersection_v2(
    firstList: List[List[int]], secondList: List[List[int]]
) -> List[List[int]]:
    """
    Using itertools-style approach with generators
    """

    def intersect_intervals():
        i = j = 0
        while i < len(firstList) and j < len(secondList):
            start = max(firstList[i][0], secondList[j][0])
            end = min(firstList[i][1], secondList[j][1])

            if start <= end:
                yield [start, end]

            if firstList[i][1] < secondList[j][1]:
                i += 1
            else:
                j += 1

    return list(intersect_intervals())


# =============================================================================
# TASK 6: EMPLOYEE FREE TIME (HARD)
# =============================================================================


@dataclass
class Interval:
    start: int
    end: int


def task6_employee_free_time_v1(schedule: List[List[Interval]]) -> List[Interval]:
    """
    Find free time for all employees
    Input: [[[1,3],[6,7]],[[2,4]],[[2,5],[9,12]]]
    Output: [[5,6],[7,9]]

    Approach: Merge all intervals, find gaps
    """
    all_intervals = []
    for employee in schedule:
        all_intervals.extend(employee)

    # Sort by start time
    all_intervals.sort(key=lambda x: x.start)

    # Merge overlapping intervals
    merged = [all_intervals[0]]
    for current in all_intervals[1:]:
        if merged[-1].end >= current.start:
            merged[-1].end = max(merged[-1].end, current.end)
        else:
            merged.append(current)

    # Find gaps
    free_time = []
    for i in range(1, len(merged)):
        if merged[i - 1].end < merged[i].start:
            free_time.append(Interval(merged[i - 1].end, merged[i].start))

    return free_time


def task6_employee_free_time_v2(schedule: List[List[Interval]]) -> List[Interval]:
    """
    Using heap to process intervals in chronological order
    """
    # Create events for all intervals
    events = []
    for employee in schedule:
        for interval in employee:
            events.append((interval.start, "start"))
            events.append((interval.end, "end"))

    events.sort()

    active_count = 0
    free_time = []
    last_end = None

    for time, event_type in events:
        if event_type == "start":
            if active_count == 0 and last_end is not None:
                free_time.append(Interval(last_end, time))
            active_count += 1
        else:  # end
            active_count -= 1
            if active_count == 0:
                last_end = time

    return free_time


# =============================================================================
# TASK 7: MAXIMUM OVERLAPPING INTERVALS (HARD)
# =============================================================================


def task7_max_overlapping_v1(intervals: List[List[int]]) -> Tuple[int, List[int]]:
    """
    Find maximum number of overlapping intervals and the time point
    Input: [[1,4],[2,6],[3,5],[7,8]]
    Output: (3, [3,4])

    Approach: Sweep line with coordinate compression
    """
    if not intervals:
        return (0, [])

    events = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))

    events.sort()

    max_overlap = 0
    current_overlap = 0
    max_time = None

    for time, delta in events:
        current_overlap += delta
        if current_overlap > max_overlap:
            max_overlap = current_overlap
            max_time = time

    return (max_overlap, [max_time, max_time])


def task7_max_overlapping_v2(intervals: List[List[int]]) -> Tuple[int, List[int]]:
    """
    Using defaultdict to count overlaps at each point
    """
    if not intervals:
        return (0, [])

    point_count = defaultdict(int)

    for start, end in intervals:
        point_count[start] += 1
        point_count[end] -= 1

    sorted_points = sorted(point_count.keys())

    max_overlap = 0
    current_overlap = 0
    max_intervals = []

    for point in sorted_points:
        current_overlap += point_count[point]
        if current_overlap > max_overlap:
            max_overlap = current_overlap
            max_intervals = [point]
        elif current_overlap == max_overlap:
            max_intervals.append(point)

    return (max_overlap, max_intervals)


# =============================================================================
# TASK 8: INTERVAL TREE OPERATIONS (HARD)
# =============================================================================


class IntervalTreeNode:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
        self.max_end = end
        self.left = None
        self.right = None


class IntervalTree:
    def __init__(self):
        self.root = None

    def insert(self, start: int, end: int):
        """Insert interval into tree"""
        if not self.root:
            self.root = IntervalTreeNode(start, end)
        else:
            self._insert(self.root, start, end)

    def _insert(self, node: IntervalTreeNode, start: int, end: int):
        # Update max_end
        node.max_end = max(node.max_end, end)

        if start < node.start:
            if not node.left:
                node.left = IntervalTreeNode(start, end)
            else:
                self._insert(node.left, start, end)
        else:
            if not node.right:
                node.right = IntervalTreeNode(start, end)
            else:
                self._insert(node.right, start, end)

    def search_overlapping(self, start: int, end: int) -> List[List[int]]:
        """Find all intervals overlapping with given interval"""
        result = []
        self._search_overlapping(self.root, start, end, result)
        return result

    def _search_overlapping(
        self, node: IntervalTreeNode, start: int, end: int, result: List[List[int]]
    ):
        if not node:
            return

        # Check if current interval overlaps
        if node.start <= end and start <= node.end:
            result.append([node.start, node.end])

        # Search left subtree if it might contain overlapping intervals
        if node.left and node.left.max_end >= start:
            self._search_overlapping(node.left, start, end, result)

        # Search right subtree
        if node.right and node.start <= end:
            self._search_overlapping(node.right, start, end, result)


def task8_interval_tree_demo():
    """
    Demonstrate interval tree operations
    """
    tree = IntervalTree()
    intervals = [[15, 20], [10, 30], [17, 19], [5, 20], [12, 15], [30, 40]]

    # Insert intervals
    for start, end in intervals:
        tree.insert(start, end)

    # Search for overlapping intervals
    query = [14, 16]
    overlapping = tree.search_overlapping(query[0], query[1])

    return overlapping


# =============================================================================
# TASK 9: CALENDAR SCHEDULING (HARD)
# =============================================================================


class MyCalendar:
    def __init__(self):
        self.bookings = []

    def book_v1(self, start: int, end: int) -> bool:
        """
        Book event if no conflict
        Approach: Linear search
        """
        for s, e in self.bookings:
            if start < e and end > s:  # Overlap
                return False

        self.bookings.append((start, end))
        return True

    def book_v2(self, start: int, end: int) -> bool:
        """
        Book event using binary search
        """
        # Find insertion point
        left, right = 0, len(self.bookings)

        while left < right:
            mid = (left + right) // 2
            if self.bookings[mid][0] < start:
                left = mid + 1
            else:
                right = mid

        # Check conflicts
        if left > 0 and self.bookings[left - 1][1] > start:
            return False
        if left < len(self.bookings) and self.bookings[left][0] < end:
            return False

        # Insert at correct position
        self.bookings.insert(left, (start, end))
        return True


class MyCalendarTwo:
    def __init__(self):
        self.bookings = []
        self.overlaps = []

    def book(self, start: int, end: int) -> bool:
        """
        Allow at most 2 overlapping events
        """
        # Check if triple booking would occur
        for s, e in self.overlaps:
            if start < e and end > s:
                return False

        # Add overlaps with existing bookings
        for s, e in self.bookings:
            if start < e and end > s:
                self.overlaps.append((max(start, s), min(end, e)))

        self.bookings.append((start, end))
        return True


# =============================================================================
# TASK 10: SKYLINE PROBLEM (VERY HARD)
# =============================================================================


def task10_skyline_v1(buildings: List[List[int]]) -> List[List[int]]:
    """
    The Skyline Problem - find key points where height changes
    Input: [[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]
    Output: [[2,10],[3,15],[7,12],[12,0],[15,10],[20,8],[24,0]]

    Approach: Sweep line with heap
    """
    events = []

    # Create events for building start and end
    for left, right, height in buildings:
        events.append((left, -height))  # Start event (negative for max heap)
        events.append((right, height))  # End event

    events.sort()

    result = []
    heights = [0]  # Ground level

    i = 0
    while i < len(events):
        current_x = events[i][0]

        # Process all events at same x-coordinate
        while i < len(events) and events[i][0] == current_x:
            height = events[i][1]
            if height < 0:  # Start event
                heights.append(-height)
            else:  # End event
                heights.remove(height)
            i += 1

        # Get current max height
        max_height = max(heights)

        # Add to result if height changed
        if not result or result[-1][1] != max_height:
            result.append([current_x, max_height])

    return result


def task10_skyline_v2(buildings: List[List[int]]) -> List[List[int]]:
    """
    Using heap with lazy deletion
    """
    import heapq
    from collections import defaultdict

    events = []
    for left, right, height in buildings:
        events.append((left, -height, "s"))  # Start
        events.append((right, height, "e"))  # End

    events.sort()

    result = []
    max_heap = [0]  # Max heap (use negative values)
    height_count = defaultdict(int)
    height_count[0] = 1

    for x, h, event_type in events:
        if event_type == "s":
            h = -h
            heapq.heappush(max_heap, -h)
            height_count[h] += 1
        else:  # End event
            height_count[h] -= 1
            if height_count[h] == 0:
                del height_count[h]

        # Clean up heap - remove heights that are no longer active
        while max_heap and (-max_heap[0]) not in height_count:
            heapq.heappop(max_heap)

        current_max = -max_heap[0] if max_heap else 0

        if not result or result[-1][1] != current_max:
            result.append([x, current_max])

    return result


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def run_all_tests():
    """Run all test cases"""
    print("=" * 50)
    print("INTERVAL TRAINING TASKS - TEST RESULTS")
    print("=" * 50)

    # Task 1: Basic Merge
    print("\n1. Basic Interval Merge:")
    intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
    result1 = task1_basic_merge_v1(intervals.copy())
    result2 = task1_basic_merge_v2(intervals.copy())
    print(f"Input: {intervals}")
    print(f"Output v1: {result1}")
    print(f"Output v2: {result2}")

    # Task 2: Insert Interval
    print("\n2. Insert Interval:")
    intervals = [[1, 3], [6, 9]]
    new_interval = [2, 5]
    result1 = task2_insert_interval_v1(intervals.copy(), new_interval)
    result2 = task2_insert_interval_v2(intervals.copy(), new_interval)
    print(f"Input: {intervals}, New: {new_interval}")
    print(f"Output v1: {result1}")
    print(f"Output v2: {result2}")

    # Task 3: Remove Overlapping
    print("\n3. Remove Overlapping Intervals:")
    intervals = [[1, 2], [2, 3], [3, 4], [1, 3]]
    result1 = task3_remove_intervals_v1(intervals.copy())
    result2 = task3_remove_intervals_v2(intervals.copy())
    print(f"Input: {intervals}")
    print(f"Minimum to remove v1: {result1}")
    print(f"Minimum to remove v2: {result2}")

    # Task 4: Meeting Rooms
    print("\n4. Meeting Rooms:")
    intervals = [[0, 30], [5, 10], [15, 20]]
    can_attend = task4_can_attend_meetings_v1(intervals.copy())
    min_rooms1 = task4_min_meeting_rooms_v1(intervals.copy())
    min_rooms2 = task4_min_meeting_rooms_v2(intervals.copy())
    print(f"Input: {intervals}")
    print(f"Can attend all: {can_attend}")
    print(f"Min rooms v1: {min_rooms1}")
    print(f"Min rooms v2: {min_rooms2}")

    # Task 5: Interval Intersection
    print("\n5. Interval Intersection:")
    list1 = [[0, 2], [5, 10], [13, 23], [24, 25]]
    list2 = [[1, 5], [8, 12], [15, 24], [25, 26]]
    result1 = task5_interval_intersection_v1(list1, list2)
    result2 = task5_interval_intersection_v2(list1, list2)
    print(f"List1: {list1}")
    print(f"List2: {list2}")
    print(f"Intersection v1: {result1}")
    print(f"Intersection v2: {result2}")

    # Task 6: Employee Free Time
    print("\n6. Employee Free Time:")
    schedule = [
        [Interval(1, 3), Interval(6, 7)],
        [Interval(2, 4)],
        [Interval(2, 5), Interval(9, 12)],
    ]
    result1 = task6_employee_free_time_v1(schedule)
    result2 = task6_employee_free_time_v2(schedule)
    print(f"Schedule: {[[f'[{i.start},{i.end}]' for i in emp] for emp in schedule]}")
    print(f"Free time v1: {[[f'[{i.start},{i.end}]'] for i in result1]}")
    print(f"Free time v2: {[[f'[{i.start},{i.end}]'] for i in result2]}")

    # Task 7: Maximum Overlapping
    print("\n7. Maximum Overlapping:")
    intervals = [[1, 4], [2, 6], [3, 5], [7, 8]]
    result1 = task7_max_overlapping_v1(intervals)
    result2 = task7_max_overlapping_v2(intervals)
    print(f"Input: {intervals}")
    print(f"Max overlapping v1: {result1}")
    print(f"Max overlapping v2: {result2}")

    # Task 8: Interval Tree
    print("\n8. Interval Tree:")
    overlapping = task8_interval_tree_demo()
    print(f"Overlapping with [14,16]: {overlapping}")

    # Task 9: Calendar
    print("\n9. Calendar Scheduling:")
    calendar = MyCalendar()
    bookings = [(10, 20), (15, 25), (20, 30)]
    results = []
    for start, end in bookings:
        result = calendar.book_v1(start, end)
        results.append(result)
    print(f"Bookings: {bookings}")
    print(f"Results: {results}")

    # Task 10: Skyline
    print("\n10. Skyline Problem:")
    buildings = [[2, 9, 10], [3, 7, 15], [5, 12, 12], [15, 20, 10], [19, 24, 8]]
    result1 = task10_skyline_v1(buildings)
    result2 = task10_skyline_v2(buildings)
    print(f"Buildings: {buildings}")
    print(f"Skyline v1: {result1}")
    print(f"Skyline v2: {result2}")

    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETED!")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
