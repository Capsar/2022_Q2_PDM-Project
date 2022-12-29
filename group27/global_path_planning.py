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


def distance_to_start(graph, node, start=0):
    return nx.shortest_path_length(graph, source=start, target=node, weight='weight')


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
    direction = config2 - config1
    for obs in obstacle_configs:
        dot_product = np.dot(obs[0]-config1, direction)
        t = dot_product / np.dot(direction, direction)
        if 0 <= t <= 1:
            closest_point = config1 + t * direction
            if distance(closest_point - obs[0]) < obs[1] + robot_radius:
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


def steer(from_point, to_point, rrt_radius, step_size=0.1):
    """
    Implement motion primitives here.
    To find closest to the to_point.
    """
    direction_vector = to_point - from_point
    direction_length = distance(direction_vector)
    if direction_length > rrt_radius:
        return (direction_vector / direction_length) * rrt_radius + to_point

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
    min_cost = distance_to_start(graph, nearest_node) + distance(new_config - graph.nodes[nearest_node]['config'])
    # Find the node with the lowest cost.
    for near_node in near_nodes:
        if near_node == nearest_node:
            continue

        near_node_config = graph.nodes[near_node]['config']
        if is_in_line_of_sight(near_node_config, new_config, obstacle_configs, robot_radius):
            cost = distance_to_start(graph, near_node) + distance(new_config - near_node_config)
            if cost < min_cost:
                min_cost = cost
                min_node = near_node
    return min_node


def rewire(graph, near_nodes, new_config, new_node, obstacle_configs, robot_radius):
    found = False
    for near_node in near_nodes:
        near_node_config = graph.nodes[near_node]['config']
        if is_in_line_of_sight(near_node_config, new_config, obstacle_configs, robot_radius):
            cost = distance_to_start(graph, new_node) + distance(new_config - near_node_config)
            if cost < distance_to_start(graph, near_node):
                parent_node = list(graph.predecessors(near_node))[0]
                graph.remove_edge(parent_node, near_node)
                graph.add_edge(new_node, near_node, weight=distance(new_config - near_node_config))
                if near_node == -1:
                    found = True
    return found


def extend(graph, sampled_config, obstacle_configs, robot_radius, rrt_radius):
    """
    Extend the graph towards the sampled point.
    Implemented so far: RRT*
    Source: DOI: 10.1109/ACCESS.2020.2969316
    Source paper name: Informed RRT*-Connect: An Asymptotically Optimal Single-Query Path Planning Method
    """
    nearest_node = find_nearest_node(graph, sampled_config)  # This is a node id (so not a config)
    new_config = steer(graph.nodes[nearest_node]['config'], sampled_config, rrt_radius)

    if is_in_obstacle(new_config, obstacle_configs, robot_radius):
        return 'trapped'

    if is_in_line_of_sight(new_config, graph.nodes[nearest_node]['config'], obstacle_configs, robot_radius):

        new_node = len(graph.nodes)

        near_nodes = find_near_nodes(graph, new_config, rrt_radius)
        min_node = choose_parent(graph, near_nodes, nearest_node, new_config, obstacle_configs, robot_radius)

        graph.add_node(new_node, config=new_config, cost=distance_to_start(graph, min_node) + distance(new_config - graph.nodes[min_node]['config']))
        graph.add_edge(min_node, new_node, weight=distance(new_config - graph.nodes[min_node]['config']))

        # Rewire
        rewired_goal = rewire(graph, near_nodes, new_config, new_node, obstacle_configs, robot_radius)

        if (new_config == sampled_config).all():
            if rewired_goal:
                return 'goal_found'
            return 'reached'
        else:
            return 'advanced'
    else:
        return 'trapped'


def optimize_path(graph, obstacle_configs, robot_radius):
    shortest_path = nx.shortest_path(graph, 0, -1, weight='weight')
    print('Optimizing path!')
    print('Shortest path: ', shortest_path)
    beacon_nodes = shortest_path.copy()
    while True and len(beacon_nodes) > 2:
        found = False
        for i in range(len(beacon_nodes) - 2):
            node_0, node_1, node_2 = beacon_nodes[i], beacon_nodes[i + 1], beacon_nodes[i + 2]
            config_0, config_1, config_2 = graph.nodes[node_0]['config'], graph.nodes[node_1]['config'],graph.nodes[node_2]['config']
            if is_in_line_of_sight(config_0, config_2, obstacle_configs, robot_radius):
                print('Found a short cut from', node_0, 'to', node_2, 'skipping', node_1)
                beacon_nodes.pop(i + 1)
                found = True
                break

        if not found:
            break
    print('Beacon nodes: ', beacon_nodes, '\n')
    return beacon_nodes


def sample_biased_config(graph, beacon_nodes, obstacle_configs, robot_radius):
    random_node = np.random.choice(beacon_nodes)
    beacon_config = graph.nodes[random_node]['config']
    random_config = sample_config(size=2, scale=robot_radius)
    return beacon_config + random_config


def rrt_path(graph, robot_config, goal_config, obstacle_configs, seconds, rrt_factor, smart_radius=1, smart_frequency=1000):
    """
    Extend the graph towards the sampled point.
    Implemented so far: RRT*
    Source: DOI: 10.1109/ACCESS.2020.2969316
    Source paper name: Informed RRT*-Connect: An Asymptotically Optimal Single-Query Path Planning Method
    """
    if seconds < 0.005:
        return graph

    robot_pos_config = np.pad(robot_config[0][0:2], (0, 1))  # (x, y, 0)

    graph.add_node(0, config=robot_pos_config)
    graph.add_node(-1, config=goal_config)
    graph.add_edge(0, -1, weight=1e3)

    start_time = time.time()
    initial_time = seconds / 4
    start_biased_sampling = False
    beacon_nodes = None
    while time.time() - start_time < seconds:
        n = len(graph.nodes)
        rrt_radius = rrt_factor*np.sqrt(np.log(n) / n)
        if start_biased_sampling and time.time() - start_time > initial_time:
            sampled_config = sample_biased_config(graph, beacon_nodes, obstacle_configs, smart_radius)
            initial_time += seconds / smart_frequency
        else:
            sampled_config = sample_config()

        status = extend(graph, sampled_config, obstacle_configs, robot_config[1], rrt_radius=rrt_radius)
        if status == 'goal_found':
            beacon_nodes = optimize_path(graph, obstacle_configs, robot_config[1])
            start_biased_sampling = True

    return graph
