# Dijkstra's Algorithm Training Tasks - Interview Preparation

This repository contains 10 carefully crafted Dijkstra's algorithm problems, progressing from **Easy** to **Very Hard** difficulty levels. Each problem includes multiple solution approaches using different optimization techniques and data structures.

## 🎯 Problem Difficulty Progression

### Easy (Tasks 1-2)

- **Task 1**: Basic Dijkstra Implementation
- **Task 2**: Shortest Path with Path Reconstruction

### Medium (Tasks 3-5)

- **Task 3**: Network Delay Time
- **Task 4**: Cheapest Flights with K Stops
- **Task 5**: Path with Minimum Effort

### Hard (Tasks 6-8)

- **Task 6**: Swim in Rising Water
- **Task 7**: Shortest Path in Binary Matrix
- **Task 8**: Minimum Spanning Tree (Prim's Algorithm)

### Very Hard (Tasks 9-10)

- **Task 9**: Dijkstra with Modifications
- **Task 10**: Advanced Applications (Multi-state Dijkstra)

## 📚 Learning Objectives

By completing these tasks, you'll master:

- **Classic Dijkstra's Algorithm**
- **Path Reconstruction Techniques**
- **Multi-state Dijkstra (3D/4D)**
- **Early Termination Optimizations**
- **A\* Algorithm**
- **Binary Search + Graph Algorithms**
- **Union-Find Integration**
- **Prim's Algorithm (MST)**
- **Floyd-Warshall All-Pairs Shortest Path**

## 🛠️ Libraries and Techniques Used

### Built-in Libraries

- `heapq` - Priority queue for efficient shortest path computation
- `collections.defaultdict` - Graph representation
- `collections.deque` - BFS implementation
- `dataclasses` - Clean data structure definitions
- `enum` - State management in complex scenarios
- `bisect` - Binary search optimizations

### Advanced Techniques

- **Early Termination** - Stop when target is reached
- **Multi-state Tracking** - Handle complex constraints
- **Lazy Deletion** - Efficient heap management
- **Coordinate Compression** - Optimize space usage
- **Bidirectional Search** - Reduce search space
- **A\* Heuristics** - Guide search with admissible heuristics

## 📖 Detailed Problem Explanations

### Task 1: Basic Dijkstra Implementation (Easy)

**Problem**: Find shortest distances from source to all nodes in weighted graph.
**Example**: `graph={0:[(1,4),(2,1)], 1:[(3,1)], 2:[(1,2),(3,5)], 3:[]}` from node 0
**Output**: `{0:0, 1:3, 2:1, 3:4}`

**Approaches**:

- **v1**: Standard implementation with defaultdict
- **v2**: Visited set optimization

**Key Concepts**: Priority queue, relaxation, optimal substructure

---

### Task 2: Shortest Path with Path Reconstruction (Easy-Medium)

**Problem**: Find shortest path and actual route between two nodes.
**Example**: From node 0 to node 3
**Output**: `(4, [0, 2, 1, 3])`

**Approaches**:

- **v1**: Track predecessors during search
- **v2**: Early termination when target reached

**Key Concepts**: Predecessor tracking, path reconstruction, early stopping

---

### Task 3: Network Delay Time (Medium)

**Problem**: Find minimum time for signal to reach all nodes in network.
**Example**: `times=[[2,1,1],[2,3,1],[3,4,1]], n=4, k=2` → `2`

**Approaches**:

- **v1**: Dijkstra with reachability check
- **v2**: Array-based distance tracking

**Key Concepts**: Single-source shortest path, network propagation

---

### Task 4: Cheapest Flights with K Stops (Medium)

**Problem**: Find cheapest flight path with at most K intermediate stops.
**Example**: `flights=[[0,1,100],[1,2,100],[0,2,500]], k=1` → `200`

**Approaches**:

- **v1**: Modified Dijkstra with stop tracking
- **v2**: Bellman-Ford approach

**Key Concepts**: Constrained shortest path, state expansion

---

### Task 5: Path with Minimum Effort (Medium-Hard)

**Problem**: Find path in 2D grid minimizing maximum height difference.
**Example**: `heights=[[1,2,2],[3,8,2],[5,3,5]]` → `2`

**Approaches**:

- **v1**: Dijkstra on 2D grid
- **v2**: Binary search + BFS optimization

**Key Concepts**: 2D Dijkstra, binary search on answer, BFS validation

---

### Task 6: Swim in Rising Water (Hard)

**Problem**: Find minimum time to swim through rising water levels.
**Example**: `grid=[[0,2],[1,3]]` → `3`

**Approaches**:

- **v1**: Dijkstra with time as weight
- **v2**: Union-Find with binary search

**Key Concepts**: Time-based constraints, Union-Find optimization

---

### Task 7: Shortest Path in Binary Matrix (Hard)

**Problem**: Find shortest path in binary matrix (8-directional movement).
**Example**: `grid=[[0,0,0],[1,1,0],[1,1,0]]` → `4`

**Approaches**:

- **v1**: BFS (since unweighted)
- **v2**: A\* with Manhattan distance heuristic

**Key Concepts**: 8-directional movement, BFS vs A\*, heuristic functions

---

### Task 8: Minimum Spanning Tree (Hard)

**Problem**: Find MST using Prim's algorithm (Dijkstra-like approach).
**Example**: `edges=[[0,1,10],[0,2,6],[0,3,5],[1,3,15],[2,3,4]]` → `19`

**Approaches**:

- **v1**: Prim's algorithm with weight tracking
- **v2**: Return actual MST edges

**Key Concepts**: MST, Prim's algorithm, greedy selection

---

### Task 9: Dijkstra with Modifications (Very Hard)

**Problem**: Find city with smallest number of reachable cities within threshold.
**Example**: `edges=[[0,1,3],[1,2,1],[1,3,4],[2,3,1]], threshold=4` → `3`

**Approaches**:

- **v1**: Run Dijkstra from each city
- **v2**: Floyd-Warshall all-pairs shortest path

**Key Concepts**: Multiple source Dijkstra, Floyd-Warshall, optimization strategies

---

### Task 10: Advanced Applications (Very Hard)

**Problem**: Shortest path with special movement rules (knight moves available once).
**Example**: Grid with obstacles, can use horse for L-shaped moves once

**Approaches**:

- **v1**: Multi-state Dijkstra with state tracking
- **v2**: 3D distance array optimization

**Key Concepts**: Multi-state search, complex state management, 3D/4D Dijkstra

## 🚀 Algorithm Complexity Analysis

### Time Complexities

- **Basic Dijkstra**: O((V + E) log V)
- **Path Reconstruction**: O(V) additional
- **2D Grid**: O(MN log(MN))
- **Multi-state**: O(S × V log(S × V)) where S = number of states
- **All-pairs (Floyd-Warshall)**: O(V³)

### Space Complexities

- **Standard**: O(V + E) for graph storage
- **Multi-state**: O(S × V) for state tracking
- **Path reconstruction**: O(V) for predecessors
- **2D Grid**: O(MN) for distance array

## 🎯 Interview Patterns to Master

### Core Dijkstra Patterns

1. **Single Source Shortest Path** - Classic application
2. **Path Reconstruction** - Track predecessors
3. **Early Termination** - Stop when target found
4. **Multi-state Dijkstra** - Handle complex constraints
5. **2D/3D Grid Dijkstra** - Extend to multiple dimensions

### Optimization Techniques

1. **Visited Set** - Avoid reprocessing nodes
2. **Lazy Deletion** - Handle duplicate entries in heap
3. **Binary Search Integration** - Combine with other algorithms
4. **A\* Heuristics** - Guide search direction
5. **Bidirectional Search** - Meet in the middle

### Common Variations

- **Constrained Shortest Path** - Additional constraints (K stops, effort)
- **Time-dependent Weights** - Weights change over time
- **Multi-criteria Optimization** - Multiple objectives
- **Network Flow Integration** - Combine with max flow

## 🧪 Running the Tests

```bash
cd /home/ruslan/Projects/algorithms_datastructures
python dijkstra_training_tasks.py
```

This will run all test cases and display results for each problem.

## 📈 Study Progression Strategy

### Week 1: Foundations (Tasks 1-3)

- Master basic Dijkstra implementation
- Learn path reconstruction
- Understand single-source shortest path

### Week 2: Constraints (Tasks 4-5)

- Handle additional constraints (K stops, effort)
- Learn binary search integration
- Practice 2D grid problems

### Week 3: Advanced Applications (Tasks 6-7)

- Union-Find integration
- A\* algorithm implementation
- Complex state management

### Week 4: Expert Level (Tasks 8-10)

- MST algorithms (Prim's)
- All-pairs shortest path
- Multi-state Dijkstra mastery

## 🎓 LeetCode Problem Mapping

### Easy to Medium

- **56. Merge Intervals** → Graph representation
- **743. Network Delay Time** → Task 3
- **1631. Path with Minimum Effort** → Task 5

### Medium to Hard

- **787. Cheapest Flights Within K Stops** → Task 4
- **778. Swim in Rising Water** → Task 6
- **1091. Shortest Path in Binary Matrix** → Task 7

### Hard to Very Hard

- **1334. Find the City** → Task 9
- **1293. Shortest Path with Obstacles** → Task 10 variation
- **864. Shortest Path to Get All Keys** → Multi-state pattern

## 💡 Interview Tips

### Common Mistakes to Avoid

1. **Forgetting to handle unreachable nodes**
2. **Not using lazy deletion in heap**
3. **Incorrect early termination logic**
4. **Missing edge cases (empty graph, single node)**
5. **Inefficient path reconstruction**

### Optimization Strategies

1. **Use visited set for better performance**
2. **Consider A\* for grid problems**
3. **Binary search for optimization problems**
4. **Early termination when possible**
5. **Choose right data structure for graph representation**

### Time Management

- **5 minutes**: Problem understanding and approach
- **15 minutes**: Basic implementation
- **5 minutes**: Testing and edge cases
- **5 minutes**: Optimization discussion

## 🔧 Additional Practice Resources

### System Design Applications

- **Route Planning Systems** (GPS, Maps)
- **Network Routing Protocols** (OSPF, BGP)
- **Social Network Analysis** (shortest path between users)
- **Game AI** (pathfinding in games)

### Advanced Topics

- **Johnson's Algorithm** - All-pairs with negative weights
- **Bidirectional Dijkstra** - Meet in the middle
- **Parallel Dijkstra** - Multi-threaded implementations
- **Dynamic Dijkstra** - Handle graph updates

Good luck with your Dijkstra's algorithm mastery! 🚀
