"""
Dijkstra's Algorithm Training Tasks - From Easy to Hard
======================================================

This file contains 10 Dijkstra-related problems with multiple solution approaches,
progressing from basic to advanced techniques using different Python libraries.
"""

import bisect
import heapq
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

# =============================================================================
# TASK 1: BASIC DIJKSTRA IMPLEMENTATION (EASY)
# =============================================================================


def task1_basic_dijkstra_v1(
    graph: Dict[int, List[Tuple[int, int]]], start: int
) -> Dict[int, int]:
    """
    Basic Dijkstra's algorithm - find shortest distances from start to all nodes
    Input: graph = {0: [(1, 4), (2, 1)], 1: [(3, 1)], 2: [(1, 2), (3, 5)], 3: []}
           start = 0
    Output: {0: 0, 1: 3, 2: 1, 3: 4}

    Approach: Using heapq with distance tracking
    """
    distances = defaultdict(lambda: float("inf"))
    distances[start] = 0
    pq = [(0, start)]  # (distance, node)

    while pq:
        current_dist, node = heapq.heappop(pq)

        # Skip if we've already found a shorter path
        if current_dist > distances[node]:
            continue

        # Check all neighbors
        for neighbor, weight in graph[node]:
            new_dist = current_dist + weight

            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))

    return dict(distances)


def task1_basic_dijkstra_v2(
    graph: Dict[int, List[Tuple[int, int]]], start: int
) -> Dict[int, int]:
    """
    Using set for visited nodes optimization
    """
    distances = {node: float("inf") for node in graph}
    distances[start] = 0
    visited = set()
    pq = [(0, start)]

    while pq:
        current_dist, node = heapq.heappop(pq)

        if node in visited:
            continue

        visited.add(node)

        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                new_dist = current_dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))

    return distances


# =============================================================================
# TASK 2: SHORTEST PATH WITH PATH RECONSTRUCTION (EASY-MEDIUM)
# =============================================================================


def task2_shortest_path_v1(
    graph: Dict[int, List[Tuple[int, int]]], start: int, end: int
) -> Tuple[int, List[int]]:
    """
    Find shortest path and reconstruct the actual path
    Input: graph, start=0, end=3
    Output: (4, [0, 2, 1, 3])

    Approach: Track predecessors during Dijkstra
    """
    distances = defaultdict(lambda: float("inf"))
    distances[start] = 0
    predecessors = {}
    pq = [(0, start)]

    while pq:
        current_dist, node = heapq.heappop(pq)

        if node == end:
            break

        if current_dist > distances[node]:
            continue

        for neighbor, weight in graph[node]:
            new_dist = current_dist + weight

            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                predecessors[neighbor] = node
                heapq.heappush(pq, (new_dist, neighbor))

    # Reconstruct path
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = predecessors.get(current)

    path.reverse()
    return distances[end], path


def task2_shortest_path_v2(
    graph: Dict[int, List[Tuple[int, int]]], start: int, end: int
) -> Tuple[int, List[int]]:
    """
    Early termination when target is reached
    """
    if start == end:
        return 0, [start]

    distances = {start: 0}
    predecessors = {start: None}
    pq = [(0, start)]
    visited = set()

    while pq:
        current_dist, node = heapq.heappop(pq)

        if node in visited:
            continue

        visited.add(node)

        if node == end:
            # Reconstruct path
            path = []
            current = end
            while current is not None:
                path.append(current)
                current = predecessors[current]
            path.reverse()
            return distances[end], path

        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                new_dist = current_dist + weight
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = node
                    heapq.heappush(pq, (new_dist, neighbor))

    return float("inf"), []


# =============================================================================
# TASK 3: NETWORK DELAY TIME (MEDIUM)
# =============================================================================


def task3_network_delay_v1(times: List[List[int]], n: int, k: int) -> int:
    """
    Network delay time - minimum time for signal to reach all nodes
    Input: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
    Output: 2

    Approach: Dijkstra to find max of all shortest paths
    """
    # Build adjacency list
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))

    # Dijkstra from source k
    distances = {}
    pq = [(0, k)]

    while pq:
        time, node = heapq.heappop(pq)

        if node in distances:
            continue

        distances[node] = time

        for neighbor, weight in graph[node]:
            if neighbor not in distances:
                heapq.heappush(pq, (time + weight, neighbor))

    # Check if all nodes are reachable
    if len(distances) != n:
        return -1

    return max(distances.values())


def task3_network_delay_v2(times: List[List[int]], n: int, k: int) -> int:
    """
    Using array-based distance tracking for better performance
    """
    # Build adjacency list
    graph = [[] for _ in range(n + 1)]
    for u, v, w in times:
        graph[u].append((v, w))

    # Dijkstra with array-based distances
    distances = [float("inf")] * (n + 1)
    distances[k] = 0
    pq = [(0, k)]

    while pq:
        time, node = heapq.heappop(pq)

        if time > distances[node]:
            continue

        for neighbor, weight in graph[node]:
            new_time = time + weight
            if new_time < distances[neighbor]:
                distances[neighbor] = new_time
                heapq.heappush(pq, (new_time, neighbor))

    # Find maximum time (excluding index 0)
    max_time = max(distances[1:])
    return max_time if max_time != float("inf") else -1


# =============================================================================
# TASK 4: CHEAPEST FLIGHTS WITH K STOPS (MEDIUM)
# =============================================================================


def task4_cheapest_flights_v1(
    n: int, flights: List[List[int]], src: int, dst: int, k: int
) -> int:
    """
    Find cheapest flight with at most k stops
    Input: n=3, flights=[[0,1,100],[1,2,100],[0,2,500]], src=0, dst=2, k=1
    Output: 200

    Approach: Modified Dijkstra with stop count tracking
    """
    # Build adjacency list
    graph = defaultdict(list)
    for u, v, price in flights:
        graph[u].append((v, price))

    # (cost, node, stops_used)
    pq = [(0, src, 0)]
    # best_cost[node] = min cost to reach node
    best_cost = {}

    while pq:
        cost, node, stops = heapq.heappop(pq)

        if node == dst:
            return cost

        if stops > k:
            continue

        # Skip if we've seen this node with fewer stops and same/lower cost
        if node in best_cost and best_cost[node] <= cost:
            continue

        best_cost[node] = cost

        for neighbor, price in graph[node]:
            new_cost = cost + price
            heapq.heappush(pq, (new_cost, neighbor, stops + 1))

    return -1


def task4_cheapest_flights_v2(
    n: int, flights: List[List[int]], src: int, dst: int, k: int
) -> int:
    """
    Using Bellman-Ford approach for k-stops constraint
    """
    # Initialize distances
    distances = [float("inf")] * n
    distances[src] = 0

    # Relax edges k+1 times
    for _ in range(k + 1):
        temp_distances = distances[:]
        for u, v, price in flights:
            if distances[u] != float("inf"):
                temp_distances[v] = min(temp_distances[v], distances[u] + price)
        distances = temp_distances

    return distances[dst] if distances[dst] != float("inf") else -1


# =============================================================================
# TASK 5: PATH WITH MINIMUM EFFORT (MEDIUM-HARD)
# =============================================================================


def task5_minimum_effort_v1(heights: List[List[int]]) -> int:
    """
    Find path with minimum effort (max absolute difference in path)
    Input: heights = [[1,2,2],[3,8,2],[5,3,5]]
    Output: 2

    Approach: Dijkstra on 2D grid with effort as weight
    """
    rows, cols = len(heights), len(heights[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # efforts[i][j] = minimum effort to reach (i,j)
    efforts = [[float("inf")] * cols for _ in range(rows)]
    efforts[0][0] = 0

    pq = [(0, 0, 0)]  # (effort, row, col)

    while pq:
        effort, row, col = heapq.heappop(pq)

        if row == rows - 1 and col == cols - 1:
            return effort

        if effort > efforts[row][col]:
            continue

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            if 0 <= new_row < rows and 0 <= new_col < cols:
                new_effort = max(
                    effort, abs(heights[new_row][new_col] - heights[row][col])
                )

                if new_effort < efforts[new_row][new_col]:
                    efforts[new_row][new_col] = new_effort
                    heapq.heappush(pq, (new_effort, new_row, new_col))

    return efforts[rows - 1][cols - 1]


def task5_minimum_effort_v2(heights: List[List[int]]) -> int:
    """
    Using binary search + BFS for optimization
    """
    rows, cols = len(heights), len(heights[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def can_reach_with_effort(max_effort):
        """Check if we can reach destination with given max effort"""
        if max_effort < 0:
            return False

        visited = [[False] * cols for _ in range(rows)]
        queue = deque([(0, 0)])
        visited[0][0] = True

        while queue:
            row, col = queue.popleft()

            if row == rows - 1 and col == cols - 1:
                return True

            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc

                if (
                    0 <= new_row < rows
                    and 0 <= new_col < cols
                    and not visited[new_row][new_col]
                ):

                    effort = abs(heights[new_row][new_col] - heights[row][col])
                    if effort <= max_effort:
                        visited[new_row][new_col] = True
                        queue.append((new_row, new_col))

        return False

    # Binary search on effort
    left, right = 0, max(max(row) for row in heights)

    while left < right:
        mid = (left + right) // 2
        if can_reach_with_effort(mid):
            right = mid
        else:
            left = mid + 1

    return left


# =============================================================================
# TASK 6: SWIM IN RISING WATER (HARD)
# =============================================================================


def task6_swim_in_water_v1(grid: List[List[int]]) -> int:
    """
    Find minimum time to swim from top-left to bottom-right
    Input: grid = [[0,2],[1,3]]
    Output: 3

    Approach: Dijkstra with time as weight
    """
    n = len(grid)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # times[i][j] = minimum time to reach (i,j)
    times = [[float("inf")] * n for _ in range(n)]
    times[0][0] = grid[0][0]

    pq = [(grid[0][0], 0, 0)]  # (time, row, col)

    while pq:
        time, row, col = heapq.heappop(pq)

        if row == n - 1 and col == n - 1:
            return time

        if time > times[row][col]:
            continue

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            if 0 <= new_row < n and 0 <= new_col < n:
                new_time = max(time, grid[new_row][new_col])

                if new_time < times[new_row][new_col]:
                    times[new_row][new_col] = new_time
                    heapq.heappush(pq, (new_time, new_row, new_col))

    return times[n - 1][n - 1]


def task6_swim_in_water_v2(grid: List[List[int]]) -> int:
    """
    Using Union-Find with binary search
    """
    n = len(grid)

    class UnionFind:
        def __init__(self, size):
            self.parent = list(range(size))
            self.rank = [0] * size

        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        def union(self, x, y):
            px, py = self.find(x), self.find(y)
            if px == py:
                return
            if self.rank[px] < self.rank[py]:
                px, py = py, px
            self.parent[py] = px
            if self.rank[px] == self.rank[py]:
                self.rank[px] += 1

    def can_swim_at_time(max_time):
        """Check if we can swim at given time"""
        if grid[0][0] > max_time or grid[n - 1][n - 1] > max_time:
            return False

        uf = UnionFind(n * n)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        for i in range(n):
            for j in range(n):
                if grid[i][j] <= max_time:
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n and 0 <= nj < n and grid[ni][nj] <= max_time:
                            uf.union(i * n + j, ni * n + nj)

        return uf.find(0) == uf.find(n * n - 1)

    left, right = 0, n * n - 1
    while left < right:
        mid = (left + right) // 2
        if can_swim_at_time(mid):
            right = mid
        else:
            left = mid + 1

    return left


# =============================================================================
# TASK 7: SHORTEST PATH IN BINARY MATRIX (HARD)
# =============================================================================


def task7_shortest_path_binary_v1(grid: List[List[int]]) -> int:
    """
    Find shortest path in binary matrix (8-directional movement)
    Input: grid = [[0,0,0],[1,1,0],[1,1,0]]
    Output: 4

    Approach: BFS (unweighted) or Dijkstra
    """
    n = len(grid)
    if grid[0][0] == 1 or grid[n - 1][n - 1] == 1:
        return -1

    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # BFS since all edges have weight 1
    queue = deque([(0, 0, 1)])  # (row, col, distance)
    visited = set()
    visited.add((0, 0))

    while queue:
        row, col, dist = queue.popleft()

        if row == n - 1 and col == n - 1:
            return dist

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            if (
                0 <= new_row < n
                and 0 <= new_col < n
                and grid[new_row][new_col] == 0
                and (new_row, new_col) not in visited
            ):

                visited.add((new_row, new_col))
                queue.append((new_row, new_col, dist + 1))

    return -1


def task7_shortest_path_binary_v2(grid: List[List[int]]) -> int:
    """
    Using A* algorithm with Manhattan distance heuristic
    """
    n = len(grid)
    if grid[0][0] == 1 or grid[n - 1][n - 1] == 1:
        return -1

    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def heuristic(row, col):
        return max(abs(row - (n - 1)), abs(col - (n - 1)))

    # A* algorithm
    pq = [(1 + heuristic(0, 0), 1, 0, 0)]  # (f_score, g_score, row, col)
    visited = set()

    while pq:
        f_score, g_score, row, col = heapq.heappop(pq)

        if (row, col) in visited:
            continue

        visited.add((row, col))

        if row == n - 1 and col == n - 1:
            return g_score

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            if (
                0 <= new_row < n
                and 0 <= new_col < n
                and grid[new_row][new_col] == 0
                and (new_row, new_col) not in visited
            ):

                new_g = g_score + 1
                new_f = new_g + heuristic(new_row, new_col)
                heapq.heappush(pq, (new_f, new_g, new_row, new_col))

    return -1


# =============================================================================
# TASK 8: MINIMUM SPANNING TREE WITH DIJKSTRA (HARD)
# =============================================================================


def task8_mst_dijkstra_v1(n: int, edges: List[List[int]]) -> int:
    """
    Find Minimum Spanning Tree using Dijkstra-like approach (Prim's algorithm)
    Input: n=4, edges=[[0,1,10],[0,2,6],[0,3,5],[1,3,15],[2,3,4]]
    Output: 19

    Approach: Prim's algorithm with priority queue
    """
    # Build adjacency list
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    # Prim's algorithm
    visited = set()
    pq = [(0, 0)]  # (weight, node)
    mst_weight = 0

    while pq and len(visited) < n:
        weight, node = heapq.heappop(pq)

        if node in visited:
            continue

        visited.add(node)
        mst_weight += weight

        for neighbor, edge_weight in graph[node]:
            if neighbor not in visited:
                heapq.heappush(pq, (edge_weight, neighbor))

    return mst_weight if len(visited) == n else -1


def task8_mst_dijkstra_v2(n: int, edges: List[List[int]]) -> List[Tuple[int, int, int]]:
    """
    Return actual MST edges using Prim's algorithm
    """
    # Build adjacency list
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    # Prim's algorithm with edge tracking
    visited = set()
    pq = [(0, 0, -1)]  # (weight, node, parent)
    mst_edges = []

    while pq and len(visited) < n:
        weight, node, parent = heapq.heappop(pq)

        if node in visited:
            continue

        visited.add(node)
        if parent != -1:
            mst_edges.append((parent, node, weight))

        for neighbor, edge_weight in graph[node]:
            if neighbor not in visited:
                heapq.heappush(pq, (edge_weight, neighbor, node))

    return mst_edges


# =============================================================================
# TASK 9: DIJKSTRA WITH MODIFICATIONS (VERY HARD)
# =============================================================================


def task9_modified_dijkstra_v1(
    n: int, edges: List[List[int]], distanceThreshold: int
) -> int:
    """
    Find city with smallest number of reachable cities within distance threshold
    Input: n=4, edges=[[0,1,3],[1,2,1],[1,3,4],[2,3,1]], distanceThreshold=4
    Output: 3

    Approach: Run Dijkstra from each city
    """
    # Build adjacency list
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    def dijkstra_from(start):
        """Return count of reachable cities within threshold"""
        distances = [float("inf")] * n
        distances[start] = 0
        pq = [(0, start)]

        while pq:
            dist, node = heapq.heappop(pq)

            if dist > distances[node]:
                continue

            for neighbor, weight in graph[node]:
                new_dist = dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))

        return sum(1 for d in distances if d <= distanceThreshold) - 1  # Exclude self

    min_count = float("inf")
    result_city = -1

    for city in range(n):
        count = dijkstra_from(city)
        if count <= min_count:
            min_count = count
            result_city = city

    return result_city


def task9_modified_dijkstra_v2(
    n: int, edges: List[List[int]], distanceThreshold: int
) -> int:
    """
    Using Floyd-Warshall for all-pairs shortest paths
    """
    # Initialize distance matrix
    dist = [[float("inf")] * n for _ in range(n)]

    # Distance from node to itself is 0
    for i in range(n):
        dist[i][i] = 0

    # Fill in direct edges
    for u, v, w in edges:
        dist[u][v] = w
        dist[v][u] = w

    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    # Count reachable cities for each city
    min_count = float("inf")
    result_city = -1

    for i in range(n):
        count = sum(1 for j in range(n) if i != j and dist[i][j] <= distanceThreshold)
        if count <= min_count:
            min_count = count
            result_city = i

    return result_city


# =============================================================================
# TASK 10: ADVANCED DIJKSTRA APPLICATIONS (VERY HARD)
# =============================================================================


class State(Enum):
    NORMAL = 0
    HORSE = 1


def task10_shortest_path_horse_v1(grid: List[List[int]], horse: List[int]) -> int:
    """
    Shortest path in grid where you can use a horse for faster movement
    Horse can move in L-shape (knight moves) but only once
    Input: grid (0=empty, 1=obstacle), horse=[1,1] (horse position)
    Output: minimum steps to reach bottom-right

    Approach: Modified Dijkstra with state tracking
    """
    rows, cols = len(grid), len(grid[0])
    if grid[0][0] == 1 or grid[rows - 1][cols - 1] == 1:
        return -1

    # Knight moves
    knight_moves = [
        (-2, -1),
        (-2, 1),
        (-1, -2),
        (-1, 2),
        (1, -2),
        (1, 2),
        (2, -1),
        (2, 1),
    ]
    # Regular moves
    regular_moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # State: (steps, row, col, has_horse)
    pq = [(0, 0, 0, True)]  # Start with horse
    visited = set()

    while pq:
        steps, row, col, has_horse = heapq.heappop(pq)

        if row == rows - 1 and col == cols - 1:
            return steps

        state = (row, col, has_horse)
        if state in visited:
            continue
        visited.add(state)

        # Regular moves
        for dr, dc in regular_moves:
            new_row, new_col = row + dr, col + dc
            if (
                0 <= new_row < rows
                and 0 <= new_col < cols
                and grid[new_row][new_col] == 0
            ):
                new_state = (new_row, new_col, has_horse)
                if new_state not in visited:
                    heapq.heappush(pq, (steps + 1, new_row, new_col, has_horse))

        # Horse moves (if we have the horse)
        if has_horse:
            for dr, dc in knight_moves:
                new_row, new_col = row + dr, col + dc
                if (
                    0 <= new_row < rows
                    and 0 <= new_col < cols
                    and grid[new_row][new_col] == 0
                ):
                    new_state = (new_row, new_col, False)  # Use up the horse
                    if new_state not in visited:
                        heapq.heappush(pq, (steps + 1, new_row, new_col, False))

    return -1


def task10_shortest_path_horse_v2(grid: List[List[int]], horse: List[int]) -> int:
    """
    Optimized version using 3D distance array
    """
    rows, cols = len(grid), len(grid[0])
    if grid[0][0] == 1 or grid[rows - 1][cols - 1] == 1:
        return -1

    # distances[r][c][horse_available] = min steps to reach (r,c) with horse state
    distances = [[[float("inf")] * 2 for _ in range(cols)] for _ in range(rows)]
    distances[0][0][1] = 0  # Start with horse available

    knight_moves = [
        (-2, -1),
        (-2, 1),
        (-1, -2),
        (-1, 2),
        (1, -2),
        (1, 2),
        (2, -1),
        (2, 1),
    ]
    regular_moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    pq = [(0, 0, 0, 1)]  # (steps, row, col, has_horse)

    while pq:
        steps, row, col, has_horse = heapq.heappop(pq)

        if steps > distances[row][col][has_horse]:
            continue

        # Regular moves
        for dr, dc in regular_moves:
            new_row, new_col = row + dr, col + dc
            if (
                0 <= new_row < rows
                and 0 <= new_col < cols
                and grid[new_row][new_col] == 0
            ):
                new_steps = steps + 1
                if new_steps < distances[new_row][new_col][has_horse]:
                    distances[new_row][new_col][has_horse] = new_steps
                    heapq.heappush(pq, (new_steps, new_row, new_col, has_horse))

        # Horse moves
        if has_horse:
            for dr, dc in knight_moves:
                new_row, new_col = row + dr, col + dc
                if (
                    0 <= new_row < rows
                    and 0 <= new_col < cols
                    and grid[new_row][new_col] == 0
                ):
                    new_steps = steps + 1
                    if new_steps < distances[new_row][new_col][0]:
                        distances[new_row][new_col][0] = new_steps
                        heapq.heappush(pq, (new_steps, new_row, new_col, 0))

    result = min(distances[rows - 1][cols - 1][0], distances[rows - 1][cols - 1][1])
    return result if result != float("inf") else -1


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def run_all_tests():
    """Run all test cases"""
    print("=" * 60)
    print("DIJKSTRA'S ALGORITHM TRAINING TASKS - TEST RESULTS")
    print("=" * 60)

    # Task 1: Basic Dijkstra
    print("\n1. Basic Dijkstra Implementation:")
    graph = {0: [(1, 4), (2, 1)], 1: [(3, 1)], 2: [(1, 2), (3, 5)], 3: []}
    result1 = task1_basic_dijkstra_v1(graph, 0)
    result2 = task1_basic_dijkstra_v2(graph, 0)
    print(f"Graph: {graph}")
    print(f"From node 0 - v1: {result1}")
    print(f"From node 0 - v2: {result2}")

    # Task 2: Shortest Path with Reconstruction
    print("\n2. Shortest Path with Path Reconstruction:")
    distance, path1 = task2_shortest_path_v1(graph, 0, 3)
    distance2, path2 = task2_shortest_path_v2(graph, 0, 3)
    print(f"From 0 to 3 - v1: distance={distance}, path={path1}")
    print(f"From 0 to 3 - v2: distance={distance2}, path={path2}")

    # Task 3: Network Delay Time
    print("\n3. Network Delay Time:")
    times = [[2, 1, 1], [2, 3, 1], [3, 4, 1]]
    result1 = task3_network_delay_v1(times, 4, 2)
    result2 = task3_network_delay_v2(times, 4, 2)
    print(f"Times: {times}, n=4, k=2")
    print(f"Network delay v1: {result1}")
    print(f"Network delay v2: {result2}")

    # Task 4: Cheapest Flights
    print("\n4. Cheapest Flights with K Stops:")
    flights = [[0, 1, 100], [1, 2, 100], [0, 2, 500]]
    result1 = task4_cheapest_flights_v1(3, flights, 0, 2, 1)
    result2 = task4_cheapest_flights_v2(3, flights, 0, 2, 1)
    print(f"Flights: {flights}, from 0 to 2 with ≤1 stops")
    print(f"Cheapest price v1: {result1}")
    print(f"Cheapest price v2: {result2}")

    # Task 5: Path with Minimum Effort
    print("\n5. Path with Minimum Effort:")
    heights = [[1, 2, 2], [3, 8, 2], [5, 3, 5]]
    result1 = task5_minimum_effort_v1(heights)
    result2 = task5_minimum_effort_v2(heights)
    print(f"Heights: {heights}")
    print(f"Minimum effort v1: {result1}")
    print(f"Minimum effort v2: {result2}")

    # Task 6: Swim in Rising Water
    print("\n6. Swim in Rising Water:")
    grid = [[0, 2], [1, 3]]
    result1 = task6_swim_in_water_v1(grid)
    result2 = task6_swim_in_water_v2(grid)
    print(f"Grid: {grid}")
    print(f"Minimum time v1: {result1}")
    print(f"Minimum time v2: {result2}")

    # Task 7: Shortest Path in Binary Matrix
    print("\n7. Shortest Path in Binary Matrix:")
    grid = [[0, 0, 0], [1, 1, 0], [1, 1, 0]]
    result1 = task7_shortest_path_binary_v1(grid)
    result2 = task7_shortest_path_binary_v2(grid)
    print(f"Grid: {grid}")
    print(f"Shortest path v1: {result1}")
    print(f"Shortest path v2: {result2}")

    # Task 8: MST with Dijkstra
    print("\n8. Minimum Spanning Tree (Prim's Algorithm):")
    edges = [[0, 1, 10], [0, 2, 6], [0, 3, 5], [1, 3, 15], [2, 3, 4]]
    result1 = task8_mst_dijkstra_v1(4, edges)
    result2 = task8_mst_dijkstra_v2(4, edges)
    print(f"Edges: {edges}")
    print(f"MST weight: {result1}")
    print(f"MST edges: {result2}")

    # Task 9: Modified Dijkstra
    print("\n9. City with Smallest Number of Reachable Cities:")
    edges = [[0, 1, 3], [1, 2, 1], [1, 3, 4], [2, 3, 1]]
    result1 = task9_modified_dijkstra_v1(4, edges, 4)
    result2 = task9_modified_dijkstra_v2(4, edges, 4)
    print(f"Edges: {edges}, threshold=4")
    print(f"Result city v1: {result1}")
    print(f"Result city v2: {result2}")

    # Task 10: Advanced Application
    print("\n10. Shortest Path with Horse (Knight Moves):")
    grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    result1 = task10_shortest_path_horse_v1(grid, [1, 1])
    result2 = task10_shortest_path_horse_v2(grid, [1, 1])
    print(f"Grid: {grid}")
    print(f"Shortest path with horse v1: {result1}")
    print(f"Shortest path with horse v2: {result2}")

    print("\n" + "=" * 60)
    print("ALL DIJKSTRA TESTS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
