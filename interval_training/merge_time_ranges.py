import pprint
from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass
class TimeRange:
    start: datetime
    end: datetime


def merge_time_ranges_old(time_ranges: List[TimeRange]) -> List[TimeRange]:
    """
    Merge overlapping or adjacent time ranges.

    Rules:
        - Ranges that overlap or touch should be merged
        - Consider ranges as inclusive of start time, exclusive of end time
        - Adjacent ranges (end of one = start of next) should be merged
        - Return merged ranges sorted by start time

    Example:
        ranges = [
            TimeRange(datetime(2024, 1, 15, 9, 0), datetime(2024, 1, 15, 11, 0)),   # 9-11 AM
            TimeRange(datetime(2024, 1, 15, 10, 30), datetime(2024, 1, 15, 12, 0)), # 10:30-12 PM
            TimeRange(datetime(2024, 1, 15, 14, 0), datetime(2024, 1, 15, 16, 0))   # 2-4 PM
        ]
        # Should merge first two ranges into 9 AM - 12 PM
        # Result: 2 ranges total
    """

    # TODO: Add your implementation here
    answer = []
    time_ranges.sort(key=lambda t_r: t_r.start)

    for t in time_ranges:
        if not answer:
            answer.append(t)

        if answer[-1].start < t.start and answer[-1].end < t.end:
            answer[-1].end = t.end
        else:
            answer.append(t)
    import pdb

    pdb.set_trace()

    return answer

    # if answer[-1].start.year ==  t.start.year and answer[-1].start.month < t.start.month and answer[-1].start.day < t.start.day and answer[-1].start.second < t.start.second:
    # answer.append(TimeRange(datetime(answer[-1].start.year, t.start.month, t.start.day, second=t.start.second)))


def merge_time_ranges(time_ranges: List[TimeRange]) -> List[TimeRange]:
    """
    Merge overlapping or adjacent time ranges.

    Rules:
        - Ranges that overlap or touch should be merged
        - Consider ranges as inclusive of start time, exclusive of end time
        - Adjacent ranges (end of one = start of next) should be merged
        - Return merged ranges sorted by start time

    Example:
        ranges = [
            TimeRange(datetime(2024, 1, 15, 9, 0), datetime(2024, 1, 15, 11, 0)),   # 9-11 AM
            TimeRange(datetime(2024, 1, 15, 10, 30), datetime(2024, 1, 15, 12, 0)), # 10:30-12 PM
            TimeRange(datetime(2024, 1, 15, 14, 0), datetime(2024, 1, 15, 16, 0))   # 2-4 PM
        ]
        # Should merge first two ranges into 9 AM - 12 PM
        # Result: 2 ranges total
    """
    if not time_ranges:
        return []

    answer = []
    time_ranges.sort(key=lambda t_r: t_r.start)

    for t in time_ranges:
        # If answer is empty or the current range doesn't overlap/touch with last merged range
        if not answer or answer[-1].end < t.start:
            answer.append(t)
        else:
            # Merge: extend the end time of the last range.
            answer[-1].end = max(answer[-1].end, t.end)


    return answer

    # if answer[-1].start.year ==  t.start.year and answer[-1].start.month < t.start.month and answer[-1].start.day < t.start.day and answer[-1].start.second < t.start.second:
    # answer.append(TimeRange(datetime(answer[-1].start.year, t.start.month, t.start.day, second=t.start.second)))


def main():
    result = merge_time_ranges(
        [
            TimeRange(
                datetime(2024, 1, 15, 10, 30), datetime(2024, 1, 15, 12, 0)
            ),  # 10:30AM - 12PM
            TimeRange(
                datetime(2024, 1, 15, 9, 0), datetime(2024, 1, 15, 11, 0)
            ),  # 9AM - 11AM
            TimeRange(
                datetime(2024, 1, 15, 9, 30), datetime(2024, 1, 15, 10, 30)
            ),  # 9:30AM - 10:30AM
            TimeRange(
                datetime(2024, 1, 15, 14, 0), datetime(2024, 1, 15, 16, 0)
            ),  # 2PM - 4PM
            TimeRange(
                datetime(2024, 1, 15, 16, 0), datetime(2024, 1, 15, 17, 0)
            ),  # 4PM - 5PM (adjacent)
        ]
    )
    expected = [
        TimeRange(
            datetime(2024, 1, 15, 9, 0), datetime(2024, 1, 15, 12, 0)
        ),  # 9AM - 12PM
        TimeRange(
            datetime(2024, 1, 15, 14, 0), datetime(2024, 1, 15, 17, 0)
        ),  # 2PM - 5PM
    ]
    pprint.pprint(result)
    assert result == expected, f"Result: {result}, Expected: {expected}"

    print("\nAll tasks completed")


# [TimeRange(start=datetime.datetime(2024, 1, 15, 10, 30), end=datetime.datetime(2024, 1, 15, 12, 0)), TimeRange(start=datetime.datetime(2024, 1, 15, 10, 30), end=datetime.datetime(2024, 1, 15, 12, 0)), TimeRange(start=datetime.datetime(2024, 1, 15, 9, 0), end=datetime.datetime(2024, 1, 15, 17, 0))]
# [TimeRange(start=datetime.datetime(2024, 1, 15, 9, 0), end=datetime.datetime(2024, 1, 15, 17, 0)), TimeRange(start=datetime.datetime(2024, 1, 15, 9, 0), end=datetime.datetime(2024, 1, 15, 17, 0))]

if __name__ == "__main__":
    main()
