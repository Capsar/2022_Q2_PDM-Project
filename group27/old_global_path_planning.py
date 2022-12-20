import math
import time

import numpy as np
import networkx as nx


def sample_config(size=2, scale=10.0):
    """
    In final version this should be extended to whole observation / configuration space points.
    Currently, it is only a 2D point in the x-y plane. Add a 0 for z (height).
    """
    return np.pad(np.random.uniform(-1, 1, size) * scale, (0, 1))


def distance(diff_config):
    # return np.sqrt(np.sum(diff_config**2))
    return np.linalg.norm(diff_config)


def sample_points_in_ellipse(center_config, a, b, ellipse_angle=0.4):
    center_config_x = center_config[0]
    center_config_y = center_config[1]

    angle = np.random.uniform(0.0, 2.0 * np.pi, 1)[0]
    random_radius = np.sqrt(np.random.rand())
    x_e = random_radius * a * math.cos(angle)
    y_e = random_radius * b * math.sin(angle)

    x = x_e * math.cos(ellipse_angle) - y_e * math.sin(ellipse_angle) + center_config_x
    y = x_e * math.sin(ellipse_angle) + y_e * math.cos(ellipse_angle) + center_config_y

    return np.asarray([x, y, 0])


def is_in_obstacle(config, obstacle_configs, robot_radius):
    """
    Check if sampled config is collision free with obstacles.
    linalg.norm is the euclidean distance between two points.
    """
    for obs in obstacle_configs:
        diff_config_robot = config - obs[0]
        if distance(diff_config_robot) < obs[1] + robot_radius:
            return True
    return False


def is_in_line_of_sight(config1, config2, obstacle_configs, robot_radius):
    """
    Check if there is a line of sight between two points.
    This means that there are no obstacles in between the 2 configs.
    """
    diff_config = config2 - config1
    diff_norm = distance(diff_config)
    for obs in obstacle_configs:
        diff_config1_robot = config1 - obs[0]
        dist2 = distance(np.cross(diff_config, diff_config1_robot)) / diff_norm
        if dist2 < obs[1] + robot_radius:
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
        if node == -1:
            continue
        node_config = graph.nodes[node]['config']
        temp_distance = distance(sampled_config - node_config)  # Might have to be updated to weighted distance
        if temp_distance < min_distance:
            nearest_node = node
            min_distance = temp_distance
    return nearest_node


def steer(from_point, to_point, step_size=0.1):
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
        if distance(new_config - graph.nodes[node]['config']) < rrt_radius:
            near_nodes.append(node)
    return near_nodes


def choose_parent(graph, near_nodes, nearest_node, new_config, obstacle_configs, robot_radius):
    min_node = nearest_node
    min_cost = graph.nodes[nearest_node]['cost'] + distance(new_config - graph.nodes[nearest_node]['config'])
    # Find the node with the lowest cost.
    for near_node in near_nodes:
        if near_node == nearest_node:
            continue

        near_node_config = graph.nodes[near_node]['config']
        if is_in_line_of_sight(near_node_config, new_config, obstacle_configs, robot_radius):
            cost = graph.nodes[near_node]['cost'] + distance(new_config - near_node_config)
            if cost < min_cost:
                min_cost = cost
                min_node = near_node
    return min_node


def rewire(graph, near_nodes, new_config, new_node, obstacle_configs, robot_radius):
    for near_node in near_nodes:
        near_node_config = graph.nodes[near_node]['config']
        if is_in_line_of_sight(near_node_config, new_config, obstacle_configs, robot_radius):
            cost = graph.nodes[new_node]['cost'] + distance(new_config-near_node_config)
            if cost < graph.nodes[near_node]['cost']:
                parent_node = list(graph.predecessors(near_node))[0]
                graph.remove_edge(parent_node, near_node)
                graph.add_edge(new_node, near_node, weight=distance(new_config-near_node_config))
                graph.add_node(near_node, config=near_node_config, cost=cost)


def extend(graph, sampled_config, obstacle_configs, robot_radius, rrt_radius):
    """
    Extend the graph towards the sampled point.
    Implemented so far: RRT*
    Source: DOI: 10.1109/ACCESS.2020.2969316
    Source paper name: Informed RRT*-Connect: An Asymptotically Optimal Single-Query Path Planning Method
    """
    nearest_node = find_nearest_node(graph, sampled_config)  # This is a node id (so not a config)
    new_config = steer(graph.nodes[nearest_node]['config'], sampled_config)

    if is_in_obstacle(new_config, obstacle_configs, robot_radius):
        return 'trapped'

    if is_in_line_of_sight(new_config, graph.nodes[nearest_node]['config'], obstacle_configs, robot_radius):
        new_node = len(graph.nodes)

        near_nodes = find_near_nodes(graph, new_config, rrt_radius)
        min_node = choose_parent(graph, near_nodes, nearest_node, new_config, obstacle_configs, robot_radius)

        graph.add_node(new_node, config=new_config, cost=graph.nodes[min_node]['cost'] + distance(new_config-graph.nodes[min_node]['config']))
        graph.add_edge(min_node, new_node, weight=distance(new_config-graph.nodes[min_node]['config']))

        # Rewire
        rewire(graph, near_nodes, new_config, new_node, obstacle_configs, robot_radius)

        if (new_config == sampled_config).all():
            return 'reached'
        else:
            return 'advanced'
    else:
        return 'trapped'


def rrt_path(graph, robot_config, goal_config, obstacle_configs, seconds, rrt_radius):
    """
    Extend the graph towards the sampled point.
    Implemented so far: RRT*
    Source: DOI: 10.1109/ACCESS.2020.2969316
    Source paper name: Informed RRT*-Connect: An Asymptotically Optimal Single-Query Path Planning Method
    """
    if seconds < 0.005:
        return graph

    robot_pos_config = np.pad(robot_config[0][0:2], (0, 1))  # (x, y, 0)
    center_config = robot_pos_config + np.subtract(goal_config, robot_pos_config) / 2
    angle = -np.arctan2(goal_config[0] - robot_pos_config[0], goal_config[1] - robot_pos_config[1])

    graph.add_node(0, config=robot_pos_config, cost=0)
    graph.add_node(-1, config=goal_config, cost=1e3)
    graph.add_edge(0, -1, weight=1e3)

    start_time = time.time()
    while time.time() - start_time < seconds:
        # rrt_radius = 200*(np.log(len(graph.nodes))/len(graph.nodes))**0.5
        sampled_config = sample_config()
        # sampled_config = sample_points_in_ellipse(center_config, 8, np.linalg.norm(goal_config-robot_pos_config)/1.5, angle)
        status = extend(graph, sampled_config, obstacle_configs, robot_config[1], rrt_radius=rrt_radius)
    return graph
