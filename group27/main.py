import random
import time
import warnings
import math
from copy import deepcopy

from networkx import DiGraph
from urdfenvs.robots.albert import AlbertRobot
from urdfenvs.sensors.full_sensor import FullSensor
import numpy as np
import pybullet as p
import gym
import networkx as nx
from matplotlib import pyplot as plt

from global_path_planning import CollisionManager, RRTStarSmart
from local_path_planning import follow_path, path_smoother,interpolate_path, PID_Base
from urdf_env_helpers import add_obstacles, add_goal, add_graph_to_env, draw_path, transform_to_arm, add_obstacles_3D
# from robot_arm_kinematics import Direct_Kinematics
from arm_kinematics import RobotArmKinematics


def run_albert(n_steps=500000, render=True, goal=True, obstacles=True, seed=42, albert_radius=0.3):
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
    pos0 = np.array([-10.0, -10.0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0])  # might change later

    env.reset(pos=pos0)
    random.seed(seed)
    np.random.seed(seed)

    if obstacles:
        add_obstacles(env)
    if goal:
        add_goal(env, table_position=[0, 1, 0], albert_radius=albert_radius)

    p.resetDebugVisualizerCamera(cameraDistance=16, cameraYaw=0, cameraPitch=-89.99, cameraTargetPosition=[0, 0, 0])

    # Perform 1 random action to get the initial observation (containing obstacles & goal)
    action = np.random.random(env.n())
    ob, _, _, _ = env.step(action)
    print(f"Initial observation : {ob['robot_0']}")  # This now contains the obstacles and goal (env.reset(pos=pos0) did not)

    # Calculate path
    robot_config = [ob['robot_0']['joint_state']['position'], albert_radius]
    goal_config = ob['robot_0']['goals'][0][0]
    obstacle_configs = [obstacle_config for obstacle_config in ob['robot_0']['obstacles']]
    robot_pos_config = np.pad(robot_config[0][0:2], (0, 1))  # (x, y, 0)

    print('obstacle_configs:', obstacle_configs)
    print('goal_config:', goal_config)
    print('robot_config:', robot_config)

    collision_manager = CollisionManager(obstacle_configs, albert_radius)

    start_time = time.time()
    total_nodes = 0
    for _ in range(n := 1):
        temp_time = time.time()
        domain = {'xmin': -10, 'xmax': 10, 'ymin': -10, 'ymax': 10, 'zmin': 0, 'zmax': 0}
        rrt_star_smart = RRTStarSmart(robot_pos_config, goal_config, collision_manager, domain)
        rrt_star_smart.run(8, rrt_factor=40, init_rrt_star_frac=4, smart_radius=1, smart_frequency=1000)
        found_graph = deepcopy(rrt_star_smart.graph)

        print(f'Sampled a total of {len(found_graph.nodes)} nodes in the graph in {round(time.time() - temp_time, 1)} seconds.')
        total_nodes += len(found_graph.nodes)
    print(f'Average nodes sampled: {round(total_nodes / n, 1)}, in average {round((time.time() - start_time) / n, 1)} seconds.')
    print(f'Shortest path length: {nx.shortest_path_length(found_graph, 0, -1, weight="weight")}')

    shortest_path = nx.shortest_path(found_graph, 0, -1, weight='weight')
    add_graph_to_env(found_graph, shortest_path)

    shortest_path_configs = [found_graph.nodes[node]['config'] for node in shortest_path]
    # shortest_path_configs = [robot_pos_config, [-10, -5, 0], [-5, -5, 0]]
    # shortest_path_configs = [robot_pos_config, [-9, -10, 0], [-8, -10, 0]]
    draw_path(shortest_path_configs)

    print("shortest_path_configs", shortest_path_configs)

    interpolated_path_configs = interpolate_path(shortest_path_configs, max_dist=2.5)
    smooth_path_configs = path_smoother(interpolated_path_configs)
    draw_path(smooth_path_configs)

    # print("Initial endpoint position: ", endpoint_xyz)
    # endpoint_xyz = kinematics.FK(robot_config[0][2:], xyz=True)
    # kinematics = RobotArmKinematics()

    base = PID_Base(ob, smooth_path_configs)

    history = []
    for step in range(n_steps):
        if not history:
            action = follow_path(ob, smooth_path_configs)  # Action space is 9 dimensional
        else:
            action = base.pid_follow_path(ob)
            if action == "DONE":
                break
        ob, _, done, _ = env.step(action)
        history.append(ob)
        if done:
            print("DONE")

    transform_to_arm(ob)
    
    # below for robot arm
    # button_position = base.return_position() + np.array([.63, .63, 1])
    # claw_position = base.return_position() + np.array([0.05, 0, 1.4])   # change later!
    # p.addUserDebugPoints(
    #     pointPositions=[claw_position],
    #     pointColorsRGB=[[1, 0, 0]],
    #     pointSize=5
    #     )
    #
    # add_obstacles_3D(env, location=base.return_position())
    #
    # add_goal(env, table_position=button_position.tolist(), albert_radius=.05)

    # joint_config = ob['robot_0']['joint_state']['position'][2:]
    # print("position:", joint_config, len(joint_config))
    # print(center_config)
    #
    # p.addUserDebugPoints(  # Got from pybullet documentation
    #             pointPositions=[joint_config],
    #             pointColorsRGB=[[0, 1, 0]],
    #             pointSize=5
    #         )

    # start_time = time.time()
    # total_nodes = 0
    #
    # for _ in range(n := 1):
    #     temp_time = time.time()
    #     graph = DiGraph()  # Graph should be directed to figure out parent nodes.
    #     graph = rrt_path(graph, robot_config, goal_config, obstacle_configs, seconds=5, rrt_radius=10)
    #     print(f'Sampled a total of {len(graph.nodes)} nodes in the graph in {round(time.time() - temp_time, 1)} seconds.')
    #     total_nodes += len(graph.nodes)
    # print(f'Average nodes sampled: {round(total_nodes / n, 1)}, in average {round((time.time() - start_time) / n, 1)} seconds.')
    # print(f'Dynamic shortest path length: {graph.nodes[-1]["cost"]} vs Exact shortest path length: {nx.shortest_path_length(graph, 0, -1, weight="weight")}')
    #
    # shortest_path = nx.shortest_path(graph, 0, -1, weight='weight')
    # add_graph_to_env(graph, shortest_path)
    #
    # shortest_path_configs = [graph.nodes[node]['config'] for node in shortest_path]

    time.sleep(300)
    env.close()

    return history


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert()


