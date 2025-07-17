# Linked List Training Tasks - Interview Preparation

This repository contains 10 carefully crafted linked list problems, progressing from **Easy** to **Very Hard** difficulty levels. Each problem includes multiple solution approaches using different optimization techniques and data structures.

## 🎯 Problem Difficulty Progression

### Easy (Tasks 1-2)

- **Task 1**: Basic Operations (Reverse, Find Middle)
- **Task 2**: Cycle Detection (Floyd's Algorithm)

### Medium (Tasks 3-5)

- **Task 3**: Merge Sorted Lists
- **Task 4**: Remove Duplicates
- **Task 5**: Linked List Intersection

### Hard (Tasks 6-8)

- **Task 6**: Reorder List
- **Task 7**: Rotate List
- **Task 8**: Palindrome Linked List

### Very Hard (Tasks 9-10)

- **Task 9**: Copy List with Random Pointer
- **Task 10**: LRU Cache Implementation

## 📚 Learning Objectives

By completing these tasks, you'll master:

- **Basic Linked List Operations**
- **Two-Pointer Techniques** (Tortoise and Hare)
- **Cycle Detection and Finding**
- **List Reversal and Rotation**
- **Merging and Sorting**
- **Duplicate Removal**
- **Advanced Data Structures** (LRU Cache)
- **Memory Management** (Deep Copy)

## 🛠️ Libraries and Techniques Used

### Built-in Libraries

- `collections.deque` - Double-ended queue for efficient operations
- `collections.OrderedDict` - Alternative LRU cache implementation
- `dataclasses` - Clean node definitions
- `heapq` - Priority queue for K-way merging
- `typing` - Type hints for better code clarity

### Advanced Techniques

- **Two-Pointer Technique** - Fast and slow pointers
- **Floyd's Cycle Detection** - Tortoise and hare algorithm
- **Dummy Node Pattern** - Simplify edge cases
- **In-place Reversal** - Constant space operations
- **Recursive Solutions** - Elegant problem solving
- **Hash Map Optimization** - O(1) lookup operations

## 📖 Detailed Problem Explanations

### Task 1: Basic Operations (Easy)

**Problem A**: Reverse a linked list
**Example**: `1 -> 2 -> 3 -> 4 -> 5` → `5 -> 4 -> 3 -> 2 -> 1`

**Problem B**: Find middle node
**Example**: `1 -> 2 -> 3 -> 4 -> 5` → `Node(3)`

**Approaches**:

- **Reverse v1**: Iterative with three pointers
- **Reverse v2**: Recursive approach
- **Middle v1**: Two-pass (count then find)
- **Middle v2**: Two-pointer (tortoise and hare)

**Key Concepts**: Basic pointer manipulation, two-pointer technique

---

### Task 2: Cycle Detection (Easy-Medium)

**Problem A**: Detect if linked list has cycle
**Problem B**: Find start of cycle

**Approaches**:

- **v1**: Hash set for visited nodes
- **v2**: Floyd's cycle detection algorithm

**Key Concepts**: Cycle detection, Floyd's algorithm, space optimization

---

### Task 3: Merge Sorted Lists (Medium)

**Problem A**: Merge two sorted linked lists
**Example**: `1->2->4` + `1->3->4` → `1->1->2->3->4->4`

**Problem B**: Merge k sorted linked lists
**Example**: `[[1,4,5],[1,3,4],[2,6]]` → `1->1->2->3->4->4->5->6`

**Approaches**:

- **Two lists v1**: Iterative with dummy node
- **Two lists v2**: Recursive approach
- **K lists v1**: Divide and conquer
- **K lists v2**: Min heap approach

**Key Concepts**: Merging algorithms, divide and conquer, heap operations

---

### Task 4: Remove Duplicates (Medium)

**Problem A**: Remove duplicates from sorted list
**Example**: `1->1->2->3->3` → `1->2->3`

**Problem B**: Remove all duplicates (keep only unique)
**Example**: `1->2->3->3->4->4->5` → `1->2->5`

**Problem C**: Remove duplicates from unsorted list
**Example**: `1->3->2->3->1` → `1->3->2`

**Approaches**:

- **Sorted v1**: Single pass with current pointer
- **All duplicates**: Dummy node with skip logic
- **Unsorted v1**: Hash set tracking
- **Unsorted v2**: O(n²) without extra space

**Key Concepts**: Duplicate detection, hash set optimization, space trade-offs

---

### Task 5: Linked List Intersection (Medium-Hard)

**Problem**: Find intersection point of two linked lists
**Example**: `A = 4->1->8->4->5`, `B = 5->6->1->8->4->5` → `Node(8)`

**Approaches**:

- **v1**: Hash set to track nodes from first list
- **v2**: Length calculation and alignment
- **v3**: Elegant two-pointer approach

**Key Concepts**: Node comparison, pointer alignment, optimal solutions

---

### Task 6: Reorder List (Hard)

**Problem**: Reorder list in specific pattern
**Example**: `1->2->3->4->5` → `1->5->2->4->3`

**Approaches**:

- **v1**: Find middle, reverse second half, merge
- **v2**: Using deque for simpler implementation

**Key Concepts**: List manipulation, reversal, merging patterns

---

### Task 7: Rotate List (Hard)

**Problem**: Rotate list to the right by k places
**Example**: `1->2->3->4->5`, k=2 → `4->5->1->2->3`

**Approaches**:

- **v1**: Create circular list, break at right point
- **v2**: Two-pass approach without circular list

**Key Concepts**: Circular lists, rotation algorithms, modular arithmetic

---

### Task 8: Palindrome Linked List (Hard)

**Problem**: Check if linked list is palindrome
**Example**: `1->2->2->1` → `True`

**Approaches**:

- **v1**: Convert to array and check
- **v2**: O(1) space - reverse second half
- **v3**: Recursive approach with helper

**Key Concepts**: Palindrome checking, space optimization, recursion

---

### Task 9: Copy List with Random Pointer (Very Hard)

**Problem**: Deep copy linked list with random pointers
**Example**: `[[7,null],[13,0],[11,4],[10,2],[1,0]]` → Deep copy

**Approaches**:

- **v1**: HashMap for original -> copy mapping
- **v2**: O(1) space - interweave original and copy nodes

**Key Concepts**: Deep copying, complex pointer management, space optimization

---

### Task 10: LRU Cache (Very Hard)

**Problem**: Implement LRU Cache with O(1) operations
**Operations**: `get(key)`, `put(key, value)`

**Approaches**:

- **v1**: Doubly linked list + hash map
- **v2**: Using collections.OrderedDict

**Key Concepts**: Cache implementation, doubly linked lists, hash map integration

## 🚀 Algorithm Complexity Analysis

### Time Complexities

- **Basic Operations**: O(n) for traversal-based
- **Two-Pointer**: O(n) with single pass
- **Cycle Detection**: O(n) with Floyd's algorithm
- **Merge Operations**: O(n + m) for two lists, O(nk log k) for k lists
- **LRU Cache**: O(1) for all operations

### Space Complexities

- **In-place Operations**: O(1) space
- **Hash Set Approaches**: O(n) space
- **Recursive Solutions**: O(n) stack space
- **LRU Cache**: O(capacity) space

## 🎯 Interview Patterns to Master

### Core Linked List Patterns

1. **Two-Pointer Technique** - Fast/slow pointers
2. **Dummy Node Pattern** - Simplify edge cases
3. **Reversal Pattern** - In-place list reversal
4. **Merge Pattern** - Combining sorted lists
5. **Cycle Detection** - Floyd's algorithm

### Advanced Patterns

1. **Deep Copy with Complex Pointers**
2. **Cache Implementation with Doubly Linked List**
3. **Multi-step Algorithms** (find middle, reverse, merge)
4. **Space-Time Trade-offs**
5. **Recursive vs Iterative Solutions**

### Common Edge Cases

- **Empty list** - null/None input
- **Single node** - list with one element
- **Cycle detection** - lists with cycles
- **Memory management** - avoiding memory leaks
- **Pointer updates** - maintaining list integrity

## 🧪 Running the Tests

```bash
cd /home/ruslan/Projects/algorithms_datastructures
python linked_list_training_tasks.py
```

This will run all test cases and display results for each problem.

## 📈 Study Progression Strategy

### Week 1: Foundations (Tasks 1-2)

- Master basic linked list operations
- Learn two-pointer technique
- Understand cycle detection

### Week 2: Intermediate Skills (Tasks 3-5)

- Practice merging algorithms
- Learn duplicate removal techniques
- Master intersection finding

### Week 3: Advanced Manipulation (Tasks 6-8)

- Handle complex reordering
- Learn rotation algorithms
- Master palindrome checking

### Week 4: Expert Level (Tasks 9-10)

- Deep copy with complex pointers
- Implement advanced data structures
- Master LRU cache design

## 🎓 LeetCode Problem Mapping

### Easy to Medium

- **206. Reverse Linked List** → Task 1
- **141. Linked List Cycle** → Task 2
- **21. Merge Two Sorted Lists** → Task 3
- **83. Remove Duplicates from Sorted List** → Task 4

### Medium to Hard

- **160. Intersection of Two Linked Lists** → Task 5
- **143. Reorder List** → Task 6
- **61. Rotate List** → Task 7
- **234. Palindrome Linked List** → Task 8

### Hard to Very Hard

- **138. Copy List with Random Pointer** → Task 9
- **146. LRU Cache** → Task 10

## 💡 Interview Tips

### Common Mistakes to Avoid

1. **Null pointer exceptions** - always check for null
2. **Memory leaks** - proper cleanup in languages like C++
3. **Infinite loops** - especially with cycle detection
4. **Edge cases** - empty lists, single nodes
5. **Pointer updates** - maintaining list integrity

### Optimization Strategies

1. **Use dummy nodes** for easier edge case handling
2. **Two-pointer technique** for O(1) space solutions
3. **In-place operations** when possible
4. **Hash maps** for O(1) lookup when space allows
5. **Recursive solutions** for cleaner code (when stack space allows)

### Time Management

- **5 minutes**: Problem understanding and approach selection
- **15 minutes**: Implementation with basic test cases
- **5 minutes**: Edge case handling and optimization
- **5 minutes**: Complexity analysis and alternative approaches

## 🔧 Additional Practice Resources

### System Design Applications

- **Memory Management** - Understanding linked list internals
- **Database Systems** - B+ tree implementations
- **Operating Systems** - Process scheduling queues
- **Compiler Design** - Symbol table implementations

### Advanced Topics

- **Skip Lists** - Probabilistic data structures
- **Memory Pools** - Efficient memory allocation
- **Persistent Data Structures** - Functional programming
- **Lock-free Data Structures** - Concurrent programming

### Related Data Structures

- **Doubly Linked Lists** - Bidirectional traversal
- **Circular Linked Lists** - Ring buffers
- **XOR Linked Lists** - Memory-efficient implementation
- **Unrolled Linked Lists** - Cache-friendly variants

## 🌟 Advanced Challenges

After mastering the basic problems, try these advanced variations:

1. **Flatten a Multilevel Doubly Linked List**
2. **Add Two Numbers Represented by Linked Lists**
3. **Merge k Sorted Lists with Memory Constraints**
4. **Implement a Thread-Safe Linked List**
5. **Design a Memory-Efficient Linked List**

Good luck with your linked list mastery! 🚀

_This training package covers all essential linked list patterns commonly tested in technical interviews at top tech companies._
