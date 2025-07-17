Implement Dijkstra's shortest path algorithm.
Given a weighted, directed graph, and a starting vertex, return the shortest
distance from the starting vertex to every vertex in the graph.

Input:
* `n` - the number of vertices in the graph (`2 <= n <= 100`).
* `edges` - a list of tuples, each representing a directed edge in the form `(u, v, w)`, where `u` is the source vertex, `v` is the destination vertex, and `w` (`1 <= w <= 10`) is the weight of the edge. Each vertex is labeled from `0` to `n-1`.
* `src` (`0 <= src < n`) - the source vertex from which to start the algorithm.

Note: If a vertex is unreachable from the source vertex, the shortest path distance for that vertex should be set to infinity (or a large constant).

### Examples

| n  | edges                                   | src | Output                | Explanation                                      |
|----|-----------------------------------------|-----|-----------------------|--------------------------------------------------|
| 4  | `[(0, 1, 1), (0, 2, 4), (1, 2, 2), (2, 3, 1)]` | 0   | `[0, 1, 3, 4]`        | Shortest paths from 0 to all vertices            |
| 3  | `[(0, 1, 5), (1, 2, 2)]`               | 0   | `[0, 5, 7]`           | Path: 0→1→2                                      |
| 3  | `[(0, 1, 2)]`                          | 0   | `[0, 2, inf]`         | Vertex 2 is unreachable from 0                   |
| 5  | `[(0, 1, 3), (0, 2, 8), (1, 3, 1), (3, 4, 2)]` | 0   | `[0, 3, 8, 4, 6]`     | Shortest paths from 0 to all vertices            |
