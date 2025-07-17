"""
STEP 5: Complete Dijkstra Implementation
Now put everything together in the final algorithm
"""

import heapq
from typing import Dict


class CompleteDijkstra:
    def shortestPath(self, n: int, edges: list[list[int]], src: int) -> Dict[int, int]:
        """
        Complete Dijkstra's algorithm - now you understand each piece!
        """
        print(f"🎯 Complete Dijkstra's Algorithm")
        print(f"Vertices: {n}, Edges: {edges}, Source: {src}")
        print("=" * 50)

        # STEP 1: Build graph (you practiced this!)
        print("Step 1: Building adjacency list...")
        graph = {}
        for vertex in range(n):
            graph[vertex] = []

        for source_vertex, destination_vertex, edge_weight in edges:
            graph[source_vertex].append([destination_vertex, edge_weight])

        print(f"Graph: {graph}")

        # STEP 2: Initialize data structures (you practiced this!)
        print("\nStep 2: Initializing...")
        shortest_distances = {}
        priority_queue = [[0, src]]
        print(f"Starting with vertex {src} at distance 0")

        # STEP 3: Main algorithm loop (you practiced this!)
        print("\nStep 3: Main algorithm...")
        step_num = 1

        while priority_queue:
            print(f"\n  Iteration {step_num}:")
            current_distance, current_vertex = heapq.heappop(priority_queue)

            if current_vertex in shortest_distances:
                print(f"    Skipping vertex {current_vertex} (already processed)")
                continue

            shortest_distances[current_vertex] = current_distance
            print(
                f"    Processing vertex {current_vertex}, distance: {current_distance}"
            )

            # Explore neighbors
            for neighbor_vertex, edge_weight in graph[current_vertex]:
                new_distance = current_distance + edge_weight
                heapq.heappush(priority_queue, [new_distance, neighbor_vertex])
                print(
                    f"      Adding neighbor {neighbor_vertex} with distance {new_distance}"
                )

            step_num += 1

        # STEP 4: Handle unreachable vertices
        print("\nStep 4: Handling unreachable vertices...")
        for vertex in range(n):
            if vertex not in shortest_distances:
                shortest_distances[vertex] = -1
                print(f"  Vertex {vertex} is unreachable")

        print(f"\nFinal result: {shortest_distances}")
        return shortest_distances


# Test the complete implementation
def test_complete_dijkstra():
    dijkstra = CompleteDijkstra()

    # Test case from your original problem
    result = dijkstra.shortestPath(4, [(0, 1, 1), (0, 2, 4), (1, 2, 2), (2, 3, 1)], 0)

    expected = {0: 0, 1: 1, 2: 3, 3: 4}
    print(f"\nExpected: {expected}")
    print(f"Got:      {result}")
    print(f"✅ {'PASS' if result == expected else 'FAIL'}")


if __name__ == "__main__":
    test_complete_dijkstra()
