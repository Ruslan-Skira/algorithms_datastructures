"""
STEP 1: Understanding Graph Representation
Practice building adjacency lists from edge lists
"""


def build_graph_from_edges(n: int, edges: list) -> dict:
    """
    Convert edge list to adjacency list representation

    Input: n=4, edges=[(0,1,5), (0,2,3), (1,3,1)]
    Output: {0: [[1,5], [2,3]], 1: [[3,1]], 2: [], 3: []}
    """
    # TODO: Practice this step
    graph = {}

    # Initialize empty lists for all vertices
    for vertex in range(n):
        graph[vertex] = []

    # Add edges to adjacency list
    for source, destination, weight in edges:
        graph[source].append([destination, weight])

    return graph


# Practice exercises
def practice_graph_building():
    print("🔧 STEP 1: Graph Building Practice")
    print("=" * 40)

    # Exercise 1
    edges1 = [(0, 1, 5), (0, 2, 3), (1, 3, 1)]
    graph1 = build_graph_from_edges(4, edges1)
    print(f"Edges: {edges1}")
    print(f"Graph: {graph1}")
    print()

    # Exercise 2 - Try this yourself!
    edges2 = [(0, 1, 2), (1, 2, 4), (0, 3, 7)]
    graph2 = build_graph_from_edges(4, edges2)
    print(f"Edges: {edges2}")
    print(f"Graph: {graph2}")


if __name__ == "__main__":
    practice_graph_building()
