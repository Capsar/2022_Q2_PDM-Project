import time

import gym
import numpy as np
import pybullet as p

from networkx import DiGraph
from urdfenvs.robots.albert import AlbertRobot
import warnings
import random
import networkx as nx
from urdfenvs.sensors.full_sensor import FullSensor
from urdfenvs.urdf_common.urdf_env import filter_shape_dim


def add_obstacles(env, seed=28, number=20, scale=10.0):
    from MotionPlanningEnv.sphereObstacle import SphereObstacle
    random.seed(seed)
    for i in range(number):
        random_x = random.uniform(-1, 1) * scale
        random_z = random.uniform(-1, 1) * scale
        sphere_obst_dict = {
            "type": "sphere",
            'movable': False,
            "geometry": {"position": [random_x, random_z, 0.0], "radius": 0.5},
        }
        sphere_obst = SphereObstacle(name=f'obstacle_{i}', content_dict=sphere_obst_dict)
        env.add_obstacle(sphere_obst)
    # adding a table from which to grap the goal
    table_height = 1
    table_length = 2
    table_width = 1
    table_size = [table_width, table_length, table_height]
    table_position = [[1, 1, 0]]
    env.add_shapes(shape_type="GEOM_BOX", dim=table_size, mass=100000000, poses_2d=table_position)
    # adding the box that the robot arm has to pick up
    box_dim = 0.1
    box_size = [box_dim for n in range(3)]
    env.add_shapes(shape_type="GEOM_BOX", dim=box_size, mass=10, poses_2d=table_position, place_height=table_height + 0.5 * box_dim)


def add_goal(env):
    """
    Add the goal to the environment!
    TODO: extend for picking up the block. (Now it is just a position)
    :param env:
    :return:
    """
    from MotionPlanningGoal.staticSubGoal import StaticSubGoal

    goal_dict = {
        "weight": 1.0, "is_primary_goal": True, 'indices': [0, 1, 2], 'parent_link': 0, 'child_link': 3,
        'desired_position': [-5, 5, 0.0], 'epsilon': 0.02, 'type': "staticSubGoal",
    }

    goal = StaticSubGoal(name="goal1", content_dict=goal_dict)
    env.add_goal(goal)


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
            return nx.shortest_path_length(graph, 0, config1, weight='weight')
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

    robot_pos_config = np.pad(robot_config[0][0:2], (0, 1)) # (x, y, 0)
    graph.add_node(0, config=robot_pos_config)

    start_time = time.time()
    while time.time() - start_time < seconds:
        sampled_config = sample_config()
        status = extend(graph, sampled_config, obstacle_configs, robot_config[1], rrt_radius=rrt_radius)
    status = extend(graph, goal_config, obstacle_configs, robot_config[1], rrt_radius=rrt_radius, force_connect=True)
    return graph


def add_graph_to_env(graph, shortest_path, point_size=5, place_height=0.2):
    """ Add the graph to the environment as objects. """
    p.removeAllUserDebugItems()
    # Draw edges
    for edge in graph.edges:
        line_color = [0.2, 0.2, 0.2]
        line_width = 1
        if edge[0] in shortest_path and edge[1] in shortest_path:  # If both nodes are in the shortest path make color green.
            line_color = [0, 1, 0]
            line_width = 3

        p.addUserDebugLine(  # Got from pybullet documentation
            lineFromXYZ=[graph.nodes[edge[0]]['config'][0], graph.nodes[edge[0]]['config'][1], place_height],
            lineToXYZ=[graph.nodes[edge[1]]['config'][0], graph.nodes[edge[1]]['config'][1], place_height],
            lineColorRGB=line_color,
            lineWidth=line_width
        )

    # # Draw nodes.
    # for node in graph.nodes:
    #     node_color = [0, 0, 1]
    #     _point_size = point_size
    #     if node <= 0: # If the node is either the start or end node make it green.
    #         node_color = [0, 1, 0]
    #         _point_size = point_size * 2
    #
    #     p.addUserDebugPoints(  # Got from pybullet documentation
    #         pointPositions=[[graph.nodes[node]['config'][0], graph.nodes[node]['config'][1], place_height]],
    #         pointColorsRGB=[node_color],
    #         pointSize=_point_size
    #     )


def follow_path(ob, graph, shortest_path):
    """
    TODO: Make the robot follow the path.
    """
    for i in range(len(shortest_path)-1):
        u, v = shortest_path[i], shortest_path[i+1]
        u_config, v_config = graph.nodes[u]['config'], graph.nodes[v]['config']

    return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])


def run_albert(n_steps=500000, render=True, goal=True, obstacles=True):
    robots = [
        AlbertRobot(mode="vel"),
    ]
    sensor = FullSensor(goal_mask=['position', 'radius'], obstacle_mask=['position', 'radius'])
    robots[0].add_sensor(sensor)

    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )

    # Init environment (robot position, obstacles, goals)
    pos0 = np.array([-10.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0])  # might change later
    env.reset(pos=pos0)
    if obstacles:
        add_obstacles(env)
    if goal:
        add_goal(env)

    p.resetDebugVisualizerCamera(cameraDistance=16, cameraYaw=0, cameraPitch=-89.99, cameraTargetPosition=[0, 0, 0])

    # Perform 1 random action to get the initial observation (containing obstacles & goal)
    action = np.random.random(env.n())
    ob, _, _, _ = env.step(action)
    print(f"Initial observation : {ob['robot_0']}")  # This now contains the obstacles and goal (env.reset(pos=pos0) did not)

    # Calculate path
    robot_config = [ob['robot_0']['joint_state']['position'], 0.2]
    goal_config = ob['robot_0']['goals'][0][0]
    obstacle_configs = [obstacle_config for obstacle_config in ob['robot_0']['obstacles']]


    print('obstacle_configs:', obstacle_configs)
    print('goal_config:', goal_config)
    print('robot_config:', robot_config)

    graph = DiGraph()  # Graph should be directed to figure out parent nodes.
    start_time = time.time()
    graph = rrt_path(graph, robot_config, goal_config, obstacle_configs, seconds=10, rrt_radius=10.0)
    shortest_path = nx.shortest_path(graph, 0, -1, weight='weight')
    add_graph_to_env(graph, shortest_path)
    print(f'Sampled a total of {len(graph.nodes)} nodes in the graph in {round(time.time()-start_time, 1)} seconds.')
    print(f'Shortest path length: {calc_cost(-1, graph=graph)}')

    history = []
    for step in range(n_steps):
        action = follow_path(ob, graph, shortest_path)  # Action space is 9 dimensional
        ob, _, _, _ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert()
