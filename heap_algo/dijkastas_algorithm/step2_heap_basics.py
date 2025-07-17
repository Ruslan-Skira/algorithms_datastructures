"""
STEP 2: Understanding Priority Queue (Min Heap)
Practice using heapq for getting minimum elements
"""

import heapq


def practice_heap_operations():
    """
    Learn how heapq works - always gives you the smallest element
    """
    print("🔧 STEP 2: Heap Operations Practice")
    print("=" * 40)

    # Create a min heap
    heap = []

    # Add elements [distance, vertex]
    elements_to_add = [[5, "A"], [2, "B"], [8, "C"], [1, "D"]]

    print("Adding elements to heap:")
    for distance, vertex in elements_to_add:
        heapq.heappush(heap, [distance, vertex])
        print(f"  Added [{distance}, '{vertex}'] -> Heap: {heap}")

    print("\nRemoving elements (always gets minimum):")
    while heap:
        min_element = heapq.heappop(heap)
        print(f"  Removed: {min_element} -> Remaining: {heap}")


def practice_dijkstra_heap():
    """
    Practice with actual distances like in Dijkstra's
    """
    print("\n🎯 Dijkstra-style Heap Practice")
    print("=" * 40)

    # Simulate finding shortest paths
    priority_queue = [[0, 0]]  # [distance, vertex] - start at vertex 0
    processed = set()

    # Simulate algorithm steps
    steps = [
        ([1, 1], [4, 2]),  # From vertex 0, can reach 1 (cost 1) or 2 (cost 4)
        ([3, 2]),  # From vertex 1, can reach 2 (cost 1+2=3)
        ([4, 3]),  # From vertex 2, can reach 3 (cost 3+1=4)
    ]

    step_num = 0
    while priority_queue and step_num < len(steps):
        distance, vertex = heapq.heappop(priority_queue)

        if vertex in processed:
            print(f"  Vertex {vertex} already processed, skipping")
            continue

        processed.add(vertex)
        print(f"Step {step_num + 1}: Process vertex {vertex} (distance {distance})")

        # Add neighbors
        if step_num < len(steps):
            for new_dist, new_vertex in steps[step_num]:
                heapq.heappush(priority_queue, [new_dist, new_vertex])
                print(f"  Added [{new_dist}, {new_vertex}] to queue")

        print(f"  Queue now: {priority_queue}")
        print(f"  Processed: {processed}")
        print()
        step_num += 1


if __name__ == "__main__":
    practice_heap_operations()
    practice_dijkstra_heap()
