import numpy as np


def follow_path(ob, graph, shortest_path):
    """
    TODO: Make the robot follow the path.
    """
    for i in range(len(shortest_path) - 1):
        u, v = shortest_path[i], shortest_path[i + 1]
        u_config, v_config = graph.nodes[u]['config'], graph.nodes[v]['config']

    return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
