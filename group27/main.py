import time

import gym
import numpy as np
import pybullet as p

from networkx import Graph
from urdfenvs.robots.albert import AlbertRobot
import warnings
import random
import networkx as nx
from urdfenvs.sensors.full_sensor import FullSensor
from urdfenvs.urdf_common.urdf_env import filter_shape_dim


def add_obstacles(env, seed=28, scale=10.0):
    from MotionPlanningEnv.sphereObstacle import SphereObstacle
    random.seed(seed)
    for i in range(30):
        random_x = random.uniform(-1, 1) * scale
        random_z = random.uniform(-1, 1) * scale
        sphere_obst_dict = {
            "type": "sphere",
            'movable': False,
            "geometry": {"position": [random_x, random_z, 0.0], "radius": 0.5},
        }
        sphere_obst = SphereObstacle(name=f'obstacle_{i}', content_dict=sphere_obst_dict)
        env.add_obstacle(sphere_obst)


def add_goal(env):
    from MotionPlanningGoal.staticSubGoal import StaticSubGoal

    goal_dict = {
        "weight": 1.0, "is_primary_goal": True, 'indices': [0, 1, 2], 'parent_link': 0, 'child_link': 3,
        'desired_position': [10, 10, 0.0], 'epsilon': 0.02, 'type': "staticSubGoal",
    }

    goal = StaticSubGoal(name="goal1", content_dict=goal_dict)
    env.add_goal(goal)


def sample_config(size=2, scale=12.0):
    """
    In final version this should be extended to whole observation / configuration space points.
    Currently, it is only a 2D point in the x-y plane. Add a 0 for z (height).
    """
    return np.pad(np.random.uniform(-1, 1, size) * scale, (0, 1))


def is_collision_free(sampled_config, obs_list, robot_radius):
    """
    Check if sampled point is collision free with obstacles.
    linalg.norm is the euclidean distance between two points.
    """
    for obs in obs_list:
        if np.linalg.norm(sampled_config - obs[0]) < obs[1] + robot_radius:
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


def extend(graph, sampled_config, obstacle_configs, robot_radius):
    nearest_node = find_nearest_node(graph, sampled_config)
    new_config = steer_towards(graph.nodes[nearest_node]['config'], sampled_config)
    if is_collision_free(new_config, obstacle_configs, robot_radius):
        graph_node_index = len(graph.nodes)
        graph.add_node(graph_node_index, config=new_config)
        graph.add_edge(nearest_node, graph_node_index)
        if (new_config == sampled_config).all():
            return 'reached'
        else:
            return 'advanced'
    else:
        return 'trapped'


def rrt_path(robot_config, goal_config, obstacle_configs, seconds=1):
    graph = Graph()
    graph.add_node(0, config=robot_config[0])
    graph.add_node(1, config=goal_config)
    start_time = time.time()
    while time.time() - start_time < seconds:
        sampled_config = sample_config()
        status = extend(graph, sampled_config, obstacle_configs, robot_config[1])
    return graph
    # return nx.shortest_path(graph, 0, 1)


def add_graph_to_env(env, graph, radius=0.1, place_height=0):
    """ Add the graph to the environment as objects. """

    positions = [graph.nodes[node]['config'][0:2] for node in graph.nodes]  # x, y, orientation
    shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[0, 0, 1, 1])
    for pose in positions:
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=shape_id,
            basePosition=[pose[0], pose[1], place_height]
        )
    pass


def run_albert(n_steps=1000, render=True, goal=True, obstacles=True):
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
    obstacle_configs = [obstacle_config for obstacle_config in ob['robot_0']['obstacles']]
    goal_config = ob['robot_0']['goals'][0][0]
    robot_config = [np.pad(ob['robot_0']['joint_state']['position'][0:2], (0, 1)), 0.4]  # The 0.8 is the radius of the robot.

    print('obstacle_configs:', obstacle_configs)
    print('goal_config:', goal_config)
    print('robot_config:', robot_config)

    graph = rrt_path(robot_config, goal_config, obstacle_configs)
    print(f'Sampled a total of {len(graph.nodes)} nodes in the graph.')
    add_graph_to_env(env, graph)

    history = []
    for step in range(n_steps):
        action = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])  # Action space is 9 dimensional
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
