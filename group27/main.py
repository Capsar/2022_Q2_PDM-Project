import time
import warnings

from networkx import DiGraph
from urdfenvs.robots.albert import AlbertRobot
from urdfenvs.sensors.full_sensor import FullSensor
import numpy as np
import pybullet as p
import gym
import networkx as nx

from group27.global_path_planning import rrt_path, calc_cost
from group27.local_path_planning import follow_path
from group27.urdf_env_helpers import add_obstacles, add_goal, add_graph_to_env


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
    print(f'Sampled a total of {len(graph.nodes)} nodes in the graph in {round(time.time() - start_time, 1)} seconds.')
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
