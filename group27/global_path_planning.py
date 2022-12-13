import time

import numpy as np
import networkx as nx


def sample_config(size=2, scale=10.0):
    """
    In final version this should be extended to whole observation / configuration space points.
    Currently, it is only a 2D point in the x-y plane. Add a 0 for z (height).
    """
    return np.pad(np.random.uniform(-1, 1, size) * scale, (0, 1))


def is_collision_free(config1, obs_list, robot_radius, config2=None):
    """
    Check if sampled point is collision free with obstacles.
    linalg.norm is the euclidean distance between two points.
    """
    for obs in obs_list:
        if config2 is None and np.linalg.norm(config1 - obs[0]) < obs[1] + robot_radius:
            # Check if config1 is within an obstacle. (Only when config2 is not defined)
            return False
        if config2 is not None and np.linalg.norm(np.cross(config2 - config1, config1 - obs[0])) / np.linalg.norm(config2 - config1) < obs[1] + robot_radius:
            # Check if the line between config1 and config2 intersects an obstacle.
            return False
    return True


def find_nearest_node(graph, sampled_config):
    """
    Find the nearest node in graph to sampled point.
    """
    nearest_node = None
    min_distance = float('inf')
    for node in graph.nodes:
        distance = np.linalg.norm(sampled_config - graph.nodes[node]['config'])  # Might have to be updated to weighted distance
        if distance < min_distance:
            nearest_node = node
            min_distance = distance
    return nearest_node


def steer_towards(from_point, to_point, step_size=0.1):
    """
    Implement motion primitives here.
    To find closest to the to_point.
    """
    return to_point


def find_near_nodes(graph, new_config, rrt_radius):
    """
    Find nodes in graph that are within rrt_radius of new_config.
    """
    near_nodes = []
    for node in graph.nodes:
        if np.linalg.norm(new_config - graph.nodes[node]['config']) < rrt_radius:
            near_nodes.append(node)
    return near_nodes


def calc_cost(config1, graph=None, config2=None):
    """
    Calculate the cost of a node or edge.
    The cost of a node is equal to the length of the path from node 0 to the node.
    The cost of an edge is equal to the length.
    """
    try:
        if graph is None:
            return np.linalg.norm(config1 - config2)
        else:
            return nx.shortest_path_length(graph, 0, config1, weight='weight', method='dijkstra')
    except nx.NetworkXNoPath:
        # If there is no path from 0 to config1 return infinity.
        return float('inf')


def extend(graph, sampled_config, obstacle_configs, robot_radius, rrt_radius=20.0, force_connect=False):
    """
    Extend the graph towards the sampled point.
    Implemented so far: RRT*
    Source: DOI: 10.1109/ACCESS.2020.2969316
    Source paper name: Informed RRT*-Connect: An Asymptotically Optimal Single-Query Path Planning Method
    """
    nearest_node = find_nearest_node(graph, sampled_config)  # This is a node id (so not a config)
    nearest_node_config = graph.nodes[nearest_node]['config']
    new_config = steer_towards(nearest_node_config, sampled_config)
    if is_collision_free(new_config, obstacle_configs, robot_radius, config2=nearest_node_config) or force_connect:
        new_node = len(graph.nodes)
        if force_connect:
            new_node = -1

        graph.add_node(new_node, config=new_config)
        min_node = nearest_node
        near_nodes = find_near_nodes(graph, new_config, rrt_radius)
        min_cost = calc_cost(nearest_node, graph=graph) + calc_cost(new_config, config2=nearest_node_config)

        near_nodes_without_nearest = [node for node in near_nodes if node != nearest_node]
        for near_node in near_nodes_without_nearest:
            near_node_config = graph.nodes[near_node]['config']
            if is_collision_free(near_node_config, obstacle_configs, robot_radius, config2=new_config):
                cost = calc_cost(near_node, graph=graph) + calc_cost(new_config, config2=near_node_config)
                if cost < min_cost:
                    min_cost = cost
                    min_node = near_node

        graph.add_edge(min_node, new_node, weight=calc_cost(new_config, config2=graph.nodes[min_node]['config']))

        near_nodes_without_min = [node for node in near_nodes if node != min_node]
        for near_node in near_nodes_without_min:
            near_node_config = graph.nodes[near_node]['config']
            if is_collision_free(near_node_config, obstacle_configs, robot_radius, config2=new_config):
                cost = calc_cost(new_node, graph=graph) + calc_cost(new_config, config2=near_node_config)
                if cost < calc_cost(near_node, graph=graph):
                    parent_node = list(graph.predecessors(near_node))[0]
                    graph.remove_edge(parent_node, near_node)
                    graph.add_edge(new_node, near_node, weight=calc_cost(new_config, config2=near_node_config))

        if (new_config == sampled_config).all():
            return 'reached'
        else:
            return 'advanced'
    else:
        return 'trapped'


def rrt_path(graph, robot_config, goal_config, obstacle_configs, seconds=0.1, rrt_radius=20.0):
    """
    Extend the graph towards the sampled point.
    Implemented so far: RRT*
    Source: DOI: 10.1109/ACCESS.2020.2969316
    Source paper name: Informed RRT*-Connect: An Asymptotically Optimal Single-Query Path Planning Method
    """
    if seconds < 0.005:
        return graph

    robot_pos_config = np.pad(robot_config[0][0:2], (0, 1))  # (x, y, 0)
    graph.add_node(0, config=robot_pos_config, cost=0)

    start_time = time.time()
    # while time.time() - start_time < seconds:
    while len(graph.nodes) < 150:
        sampled_config = sample_config()
        status = extend(graph, sampled_config, obstacle_configs, robot_config[1], rrt_radius=rrt_radius)

    status = extend(graph, goal_config, obstacle_configs, robot_config[1], rrt_radius=rrt_radius, force_connect=True)
    return graph
