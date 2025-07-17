"""
STEP 4: Simple Dijkstra Implementation
Put it all together with a very simple example
"""

import heapq


def simple_dijkstra_with_debug(graph: dict, start: int):
    """
    Dijkstra's algorithm with detailed step-by-step output
    """
    print(f"🔧 STEP 4: Simple Dijkstra from vertex {start}")
    print("=" * 50)
    print(f"Graph: {graph}")
    print()

    # Initialize
    shortest_distances = {}
    priority_queue = [[0, start]]
    step = 1

    while priority_queue:
        print(f"Step {step}:")
        print(f"  Priority queue: {priority_queue}")

        # Get closest unvisited vertex
        current_distance, current_vertex = heapq.heappop(priority_queue)
        print(f"  Processing vertex {current_vertex} (distance {current_distance})")

        # Skip if already processed
        if current_vertex in shortest_distances:
            print(f"  ❌ Already processed vertex {current_vertex}, skipping")
            print()
            continue

        # Record shortest distance
        shortest_distances[current_vertex] = current_distance
        print(f"  ✅ Shortest distance to vertex {current_vertex}: {current_distance}")

        # Add neighbors to queue
        neighbors = graph.get(current_vertex, [])
        print(f"  Neighbors of vertex {current_vertex}: {neighbors}")

        for neighbor_vertex, edge_weight in neighbors:
            new_distance = current_distance + edge_weight
            heapq.heappush(priority_queue, [new_distance, neighbor_vertex])
            print(f"    Added vertex {neighbor_vertex} with distance {new_distance}")

        print(f"  Current shortest distances: {shortest_distances}")
        print()
        step += 1

    return shortest_distances


# Practice with simple examples
def practice_simple_examples():
    print("🎯 Practice Examples")
    print("=" * 30)

    # Example 1: Linear path
    print("\nExample 1: Linear path 0→1→2")
    graph1 = {0: [[1, 2]], 1: [[2, 3]], 2: []}
    result1 = simple_dijkstra_with_debug(graph1, 0)
    print(f"Final result: {result1}")
    print("\n" + "=" * 50)

    # Example 2: Two paths to same destination
    print("\nExample 2: Two paths - direct vs indirect")
    graph2 = {
        0: [[1, 1], [2, 4]],  # Can go to 1 (cost 1) or 2 (cost 4)
        1: [[2, 2]],  # From 1, can go to 2 (cost 2)
        2: [],
    }
    result2 = simple_dijkstra_with_debug(graph2, 0)
    print(f"Final result: {result2}")


if __name__ == "__main__":
    practice_simple_examples()
