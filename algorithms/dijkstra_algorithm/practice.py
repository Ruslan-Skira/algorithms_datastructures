INF = float("infinity")
graph = {
    "U": {"V": 6, "W": 7},
    "V": {"U": 6, "X": 10},
    "W": {"U": 7, "X": 1},
    "X": {"W": 1, "V": 10},
}


def dijkstra_simple(graph, start):
    distances = {vertex: float("inf") for vertex in graph}
    distances[start] = 0
    unvisited = set(graph.keys())

    print(f"Starting from {start}")
    print(f"Initial distances: {distances}")

    while unvisited:
        current = min(unvisited, key=lambda vertex: distances[vertex])
        print(f"\n--- Processing vertex {current} ---")
        print(f"Current distance to {current}: {distances[current]}")

        unvisited.remove(current)

        for neighbor, weight in graph[current].items():
            if neighbor in unvisited:
                old_distance = distances[neighbor]
                new_distance = distances[current] + weight

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    print(f"  Updated {neighbor}: {old_distance} → {new_distance}")
                else:
                    print(
                        f"  No update for {neighbor}: {new_distance} >= {old_distance}"
                    )

        print(f"Distances after processing {current}: {distances}")

    return distances


if __name__ == "__main__":
    start_vertex = "U"
    shortest_path = dijkstra_simple(graph=graph, start=start_vertex)
    print(f"Shortest paths from vertex {start_vertex}: {shortest_path}")
# Dijkstra's Algorithm Implementation
# Your implementation: O(V²) due to sorting unvisited vertices
