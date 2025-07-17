"""
STEP 3: Understanding Distance Tracking
Practice maintaining shortest distances and avoiding reprocessing
"""


def practice_distance_tracking():
    """
    Learn why we track shortest distances and skip processed vertices
    """
    print("🔧 STEP 3: Distance Tracking Practice")
    print("=" * 40)

    # Simulate discovering multiple paths to same vertex
    shortest_distances = {}
    paths_discovered = [
        (0, 0, "Start"),
        (1, 1, "Direct path 0→1"),
        (4, 2, "Direct path 0→2"),
        (3, 2, "Better path 0→1→2"),  # This should be ignored!
        (4, 3, "Path 0→1→2→3"),
    ]

    for distance, vertex, description in paths_discovered:
        print(f"\nDiscovered: {description} (distance {distance} to vertex {vertex})")

        if vertex in shortest_distances:
            print(
                f"  ❌ Vertex {vertex} already processed with distance {shortest_distances[vertex]}"
            )
            print(f"  ❌ Ignoring this path (distance {distance})")
        else:
            shortest_distances[vertex] = distance
            print(f"  ✅ Recording shortest distance to vertex {vertex}: {distance}")

        print(f"  Current shortest distances: {shortest_distances}")


def practice_why_skip_processed():
    """
    Understand WHY we skip already processed vertices
    """
    print("\n🎯 Why Skip Processed Vertices?")
    print("=" * 40)

    print("Scenario: We have paths 0→1→2 (cost 3) and 0→2 (cost 4)")
    print()

    # Simulate processing order
    print("Processing order (heap gives us minimum first):")
    print("1. Process vertex 0 (distance 0)")
    print("   - Add vertex 1 to queue (distance 1)")
    print("   - Add vertex 2 to queue (distance 4)")
    print()

    print("2. Process vertex 1 (distance 1) - comes first because 1 < 4")
    print("   - Add vertex 2 to queue (distance 1+2=3)")
    print("   - Queue now has: [3,2] and [4,2]")
    print()

    print("3. Process vertex 2 (distance 3) - comes first because 3 < 4")
    print("   - Record shortest distance to vertex 2 as 3")
    print()

    print("4. Try to process vertex 2 again (distance 4)")
    print("   - ❌ SKIP! We already found a better path (distance 3)")
    print("   - This is why we check 'if vertex in shortest_distances'")


if __name__ == "__main__":
    practice_distance_tracking()
    practice_why_skip_processed()
