import time

import gym
import numpy as np
from networkx import Graph
from urdfenvs.robots.albert import AlbertRobot
import warnings
import random
import networkx as nx


def add_obstacles(env, seed=28, scale=10.0):
    from MotionPlanningEnv.sphereObstacle import SphereObstacle
    random.seed(seed)
    for i in range(50):
        random_x = random.uniform(-1, 1) * scale
        random_z = random.uniform(-1, 1) * scale
        sphere_obst_dict = {
            "type": "sphere",
            'movable': False,
            "geometry": {"position": [random_x, random_z, 0.0], "radius": 0.5},
        }
        sphere_obst = SphereObstacle(name="simpleSphere", content_dict=sphere_obst_dict)
        env.add_obstacle(sphere_obst)
    # adding a table from which to grap the goal
    table_height = 1
    table_length = 2
    table_width = 1
    table_size = [table_width, table_length, table_height]
    table_position = [[1,1,0]]
    env.add_shapes(shape_type="GEOM_BOX", dim=table_size, mass=100000000, poses_2d=table_position)
    # adding the box that the robot arm has to pick up
    box_dim = 0.1
    box_size = [box_dim for n in range(3)]
    env.add_shapes(shape_type="GEOM_BOX", dim=box_size, mass=10, poses_2d=table_position, place_height=table_height+0.5*box_dim)




def add_goal(env):
    from MotionPlanningGoal.staticSubGoal import StaticSubGoal

    goal_dict = {
        "weight": 1.0, "is_primary_goal": True, 'indices': [0, 1, 2], 'parent_link': 0, 'child_link': 3,
        'desired_position': [10, 10, 0.0], 'epsilon': 0.02, 'type': "staticSubGoal",
    }

    goal = StaticSubGoal(name="goal1", content_dict=goal_dict)
    env.add_goal(goal)


def sample_point(robot_pos, scale=12.0):
    """
    In final version this should be extended to whole observation / configuration space points.
    Currently it is only a 2D point in the x-z plane.
    """
    return np.random.uniform(-1, 1, len(robot_pos)) * scale


def is_collision_free(sampled_point, obs_list):
    """
    Check if sampled point is collision free with obstacles.
    linalg.norm is the euclidean distance between two points.
    """
    for obs in obs_list:
        if np.linalg.norm(sampled_point - obs[0]) < obs[1]:
            return False
    return True

def find_nearest_node(graph, sampled_point):
    """
    Find the nearest node in graph to sampled point.
    """
    nearest_node = None
    min_distance = float('inf')
    for node in graph.nodes:
        distance = np.linalg.norm(sampled_point - graph.nodes[node]['pos'])
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


def extend(graph, sampled_point, obs_list):
    nearest_node = find_nearest_node(graph, sampled_point)
    new_point = steer_towards(graph.nodes[nearest_node]['pos'], sampled_point)
    if is_collision_free(sampled_point, obs_list):
        graph_node_index = len(graph.nodes)
        graph.add_node(graph_node_index, pos=sampled_point)
        graph.add_edge(nearest_node, graph_node_index)
        if new_point == sampled_point:
            return 'reached'
        else:
            return 'advanced'
    else:
        return 'trapped'


def rrt_path(env, ob, start_time=time.time()):
    obs_list = [(np.asarray([obstacle.position()[0], obstacle.position()[1]]), obstacle.radius()) for obstacle in env.get_obstacles().values()]
    robot_pos = [ob['robot_0']['joint_state']['position'][0], ob['robot_0']['joint_state']['position'][1]]
    goal_pos = [env.get_goals()['goal1'].desired_position[0], env.get_goals()['goal1'].desired_position[1]]

    graph = Graph()
    graph.add_node(0, pos=robot_pos)
    graph.add_node(1, pos=goal_pos)

    while time.time() - start_time < 1:
        sampled_point = sample_point(robot_pos)
        status = extend(graph, sampled_point, obs_list)
    return nx.shortest_path(graph, 0, 1)


def run_albert(n_steps=1000, render=False, goal=True, obstacles=True):
    robots = [
        AlbertRobot(mode="vel"),
    ]
    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )

    action = np.ones(9) * 2  # Action space is 9 dimensional
    pos0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0, 0])
    ob = env.reset(pos=pos0)
    print(f"Initial observation : {ob}")

    if obstacles:
        add_obstacles(env)

    if goal:
        add_goal(env)

    ## Calculate path
    # path = rrt_path(env, ob)

    history = []
    for step in range(n_steps):
        ob, _, _, _ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True)
    
