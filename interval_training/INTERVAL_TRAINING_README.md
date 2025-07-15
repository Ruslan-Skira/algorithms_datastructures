# Interval Training Tasks - Interview Preparation

This repository contains 10 carefully crafted interval-related problems, progressing from **Easy** to **Very Hard** difficulty levels. Each problem includes multiple solution approaches using different Python libraries and techniques.

## 🎯 Problem Difficulty Progression

### Easy (Tasks 1-2)

- **Task 1**: Basic Interval Merge
- **Task 2**: Insert Interval

### Medium (Tasks 3-5)

- **Task 3**: Non-overlapping Intervals
- **Task 4**: Meeting Rooms I & II
- **Task 5**: Interval Intersection

### Hard (Tasks 6-8)

- **Task 6**: Employee Free Time
- **Task 7**: Maximum Overlapping Intervals
- **Task 8**: Interval Tree Operations

### Very Hard (Tasks 9-10)

- **Task 9**: Calendar Scheduling
- **Task 10**: Skyline Problem

## 📚 Learning Objectives

By completing these tasks, you'll master:

- **Sorting and Greedy Algorithms**
- **Two Pointers Technique**
- **Sweep Line Algorithm**
- **Heap/Priority Queue Operations**
- **Binary Search Applications**
- **Advanced Data Structures (Interval Trees)**
- **Event-driven Programming**

## 🛠️ Libraries and Techniques Used

### Built-in Libraries

- `bisect` - Binary search operations
- `heapq` - Priority queue/heap operations
- `collections.defaultdict` - Enhanced dictionaries
- `collections.deque` - Double-ended queue for efficient operations
- `dataclasses` - Clean data structure definitions
- `datetime` - Time-based interval handling

### Advanced Techniques

- **Coordinate Compression**
- **Lazy Deletion in Heaps**
- **Event-driven Processing**
- **Tree-based Interval Management**

## 📖 Detailed Problem Explanations

### Task 1: Basic Interval Merge (Easy)

**Problem**: Given overlapping intervals, merge them into non-overlapping intervals.
**Example**: `[[1,3],[2,6],[8,10]]` → `[[1,6],[8,10]]`

**Approaches**:

- **v1**: Basic sorting + iteration
- **v2**: Using `collections.deque` for efficient operations

**Key Concepts**: Sorting, basic merging logic

---

### Task 2: Insert Interval (Easy-Medium)

**Problem**: Insert a new interval into a sorted list of non-overlapping intervals.
**Example**: `intervals=[[1,3],[6,9]], new=[2,5]` → `[[1,5],[6,9]]`

**Approaches**:

- **v1**: Three-phase approach (before, merge, after)
- **v2**: Using `bisect` module for efficient insertion point finding

**Key Concepts**: Binary search, efficient insertion

---

### Task 3: Non-overlapping Intervals (Medium)

**Problem**: Find minimum number of intervals to remove to make rest non-overlapping.
**Example**: `[[1,2],[2,3],[3,4],[1,3]]` → `1` (remove `[1,3]`)

**Approaches**:

- **v1**: Greedy algorithm (sort by end time)
- **v2**: Heap-based approach for tracking active intervals

**Key Concepts**: Greedy algorithms, activity selection problem

---

### Task 4: Meeting Rooms (Medium)

**Problem A**: Check if person can attend all meetings.
**Problem B**: Find minimum meeting rooms needed.
**Example**: `[[0,30],[5,10],[15,20]]` → `False`, `2 rooms`

**Approaches**:

- **v1**: Sweep line algorithm with events
- **v2**: Heap-based room allocation

**Key Concepts**: Sweep line, event processing, resource allocation

---

### Task 5: Interval Intersection (Medium)

**Problem**: Find intersection of two sorted interval lists.
**Example**: `[[0,2],[5,10]]` ∩ `[[1,5],[8,12]]` → `[[1,2],[5,5],[8,10]]`

**Approaches**:

- **v1**: Two pointers technique
- **v2**: Generator-based approach for memory efficiency

**Key Concepts**: Two pointers, set intersection

---

### Task 6: Employee Free Time (Hard)

**Problem**: Find common free time for all employees.
**Example**: `[[[1,3],[6,7]],[[2,4]],[[2,5],[9,12]]]` → `[[5,6],[7,9]]`

**Approaches**:

- **v1**: Merge all intervals, find gaps
- **v2**: Event-driven processing with heap

**Key Concepts**: Interval merging, gap finding, event processing

---

### Task 7: Maximum Overlapping Intervals (Hard)

**Problem**: Find maximum number of overlapping intervals at any point.
**Example**: `[[1,4],[2,6],[3,5],[7,8]]` → `3` overlaps at point `3`

**Approaches**:

- **v1**: Sweep line with coordinate compression
- **v2**: Point counting with `defaultdict`

**Key Concepts**: Coordinate compression, sweep line, overlap counting

---

### Task 8: Interval Tree Operations (Hard)

**Problem**: Build interval tree for efficient overlap queries.
**Example**: Store intervals and quickly find all overlapping with query `[14,16]`

**Approaches**:

- Custom interval tree implementation
- Tree-based range queries

**Key Concepts**: Binary search trees, range queries, tree augmentation

---

### Task 9: Calendar Scheduling (Very Hard)

**Problem**: Implement calendar that prevents double/triple bookings.
**Example**: Allow at most 1 or 2 overlapping events

**Approaches**:

- **v1**: Linear search for conflicts
- **v2**: Binary search with sorted insertion
- **Calendar II**: Allow double booking but prevent triple

**Key Concepts**: Conflict detection, efficient booking management

---

### Task 10: Skyline Problem (Very Hard)

**Problem**: Find key points where building heights change in city skyline.
**Example**: `[[2,9,10],[3,7,15]]` → `[[2,10],[3,15],[7,10],[9,0]]`

**Approaches**:

- **v1**: Sweep line with heap
- **v2**: Lazy deletion technique

**Key Concepts**: Sweep line, priority queues, lazy deletion

## 🚀 How to Use This Repository

1. **Start with Easy Problems**: Begin with Tasks 1-2 to understand basic concepts
2. **Progress Gradually**: Move through medium and hard problems systematically
3. **Compare Approaches**: Study different solution methods for each problem
4. **Practice Implementation**: Implement solutions from scratch
5. **Time Yourself**: Practice under interview conditions

## 🎯 Interview Tips

### Common Patterns to Remember

1. **Sort first** - Most interval problems benefit from sorting
2. **Two pointers** - Efficient for intersection/merging problems
3. **Sweep line** - Great for event-based problems
4. **Heap/Priority Queue** - Essential for scheduling problems
5. **Binary search** - Speeds up insertion/search operations

### Time Complexities

- **Basic merge**: O(n log n)
- **Insert interval**: O(n)
- **Meeting rooms**: O(n log n)
- **Intersection**: O(n + m)
- **Skyline**: O(n log n)

### Space Complexities

- **In-place merge**: O(1)
- **Heap-based**: O(n)
- **Tree-based**: O(n)

## 🧪 Running the Tests

```bash
python interval_training_tasks.py
```

This will run all test cases and display results for each problem.

## 📈 Progression Strategy

1. **Week 1**: Master Tasks 1-3 (Easy to Medium)
2. **Week 2**: Tackle Tasks 4-6 (Medium to Hard)
3. **Week 3**: Challenge yourself with Tasks 7-8 (Hard)
4. **Week 4**: Conquer Tasks 9-10 (Very Hard)

## 🎓 Additional Practice

After mastering these problems, consider:

- **Leetcode**: Problems 56, 57, 435, 252, 253, 986, 759, 436, 729, 218
- **System Design**: Meeting scheduler, calendar systems
- **Advanced Topics**: Segment trees, lazy propagation

## 📝 Notes

- Each problem includes detailed comments explaining the approach
- Multiple solutions show different trade-offs (time vs space, simplicity vs efficiency)
- Test cases verify correctness of all implementations
- Progressive difficulty helps build confidence gradually

Good luck with your interview preparation! 🚀
