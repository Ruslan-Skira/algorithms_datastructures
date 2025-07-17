import heapq
from typing import Dict, List


class Solution:
    """Dijkstra's shortest path algorithm implementation"""

    def shortestPath(self, n: int, edges: list[list[int]], src: int) -> Dict[int, int]:
        # Build adjacency list: vertex -> [(neighbor, weight), ...]
        graph = {}
        for vertex in range(n):
            graph[vertex] = []

        for source_vertex, destination_vertex, edge_weight in edges:
            graph[source_vertex].append([destination_vertex, edge_weight])

        # Store final shortest distances from source to each vertex
        shortest_distances = {}

        # Priority queue: [distance_from_source, vertex]
        # Start with source vertex at distance 0
        priority_queue = [[0, src]]

        while priority_queue:  # While we still have vertices to explore
            current_distance, current_vertex = heapq.heappop(
                priority_queue
            )  # Get closest unvisited vertex

            if current_vertex in shortest_distances:  # Already processed this vertex
                continue  # Skip it - we found a better path earlier

            shortest_distances[current_vertex] = (
                current_distance  # Record shortest distance to this vertex
            )

            # Explore all neighbors of current vertex
            for neighbor_vertex, edge_weight in graph[current_vertex]:
                new_distance = current_distance + edge_weight
                heapq.heappush(
                    priority_queue, [new_distance, neighbor_vertex]
                )  # Add neighbor to queue

        # Mark unreachable vertices with -1
        for vertex in range(n):
            if vertex not in shortest_distances:
                shortest_distances[vertex] = -1

        return shortest_distances


# Test the implementation with examples from the task
if __name__ == "__main__":
    sol = Solution()

    print("🔗 Testing Dijkstra's Algorithm with Clear Variable Names")
    print("=" * 60)

    # Test case 1: Basic graph
    print("\n📍 Test 1: Basic connected graph")
    result1 = sol.shortestPath(4, [(0, 1, 1), (0, 2, 4), (1, 2, 2), (2, 3, 1)], 0)
    print(f"Edges: [(0,1,1), (0,2,4), (1,2,2), (2,3,1)], Source: 0")
    print(f"Expected: {{0: 0, 1: 1, 2: 3, 3: 4}}")
    print(f"Got:      {result1}")
    print(f"✅ {'PASS' if result1 == {0: 0, 1: 1, 2: 3, 3: 4} else 'FAIL'}")

    # Test case 2: Linear path
    print("\n📍 Test 2: Linear path")
    result2 = sol.shortestPath(3, [(0, 1, 5), (1, 2, 2)], 0)
    print(f"Edges: [(0,1,5), (1,2,2)], Source: 0")
    print(f"Expected: {{0: 0, 1: 5, 2: 7}}")
    print(f"Got:      {result2}")
    print(f"✅ {'PASS' if result2 == {0: 0, 1: 5, 2: 7} else 'FAIL'}")

    # Test case 3: Unreachable vertex
    print("\n📍 Test 3: Unreachable vertex")
    result3 = sol.shortestPath(3, [(0, 1, 2)], 0)
    print(f"Edges: [(0,1,2)], Source: 0")
    print(f"Expected: {{0: 0, 1: 2, 2: -1}}")
    print(f"Got:      {result3}")
    print(f"✅ {'PASS' if result3 == {0: 0, 1: 2, 2: -1} else 'FAIL'}")

    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
