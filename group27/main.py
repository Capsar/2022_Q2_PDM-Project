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

from global_path_planning import CollisionManager, RRTStarSmart, path_length
from local_path_planning import follow_path, path_smoother, interpolate_path, PID_Base
from urdf_env_helpers import add_obstacles, add_goal, add_graph_to_env, draw_node_configs, draw_path, transform_to_arm, add_obstacles_3D
# from robot_arm_kinematics import Direct_Kinematics
from arm_kinematics import RobotArmKinematics

def run_albert(n_steps=500000, render=True, goal=True, obstacles=False, seed=42, albert_radius=0.3):
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
    # pos0 = np.array([-10.0, -10.0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0])  # might change later
    pos0 = np.array([-0.02934186, 1.09177044, -1.01096624, -0.286753, -0.40583808, 0.35806924, -0.82694153, 0.15506737,  0.24741826, -0.08700108])
    pos0 = np.array([-0.02934186, 1.09177044, -1.01096624, 0, 0, 0, 0, 0, 0, 0])

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

    # collision_manager = CollisionManager(obstacle_configs, albert_radius)
    # domain = {'xmin': -10, 'xmax': 10, 'ymin': -10, 'ymax': 10, 'zmin': 0, 'zmax': 0}
    #
    # rrt_star_smart = RRTStarSmart(robot_pos_config, goal_config, collision_manager, domain, seed=seed)
    # rrt_star_smart.smart_run(total_duration=10, rrt_factor=30, smart_sample_ratio=0.5, smart_radius=1)
    # shortest_path = nx.shortest_path(rrt_star_smart.graph, 0, -1, weight='weight')
    # shortest_path_configs = [rrt_star_smart.graph.nodes[node]['config'] for node in shortest_path]
    #
    # add_graph_to_env(rrt_star_smart.graph, 0.005)
    # draw_node_configs(rrt_star_smart.biased_sampled_configs, 0.005)
    # draw_path(shortest_path_configs, place_height=0.007)
    # print("Shortest path length: ", path_length(shortest_path_configs))
    #
    # interpolated_path_configs = interpolate_path(shortest_path_configs, max_dist=2.5)
    # smooth_path_configs = path_smoother(interpolated_path_configs)
    # draw_path(smooth_path_configs, line_color=[1, 0, 0], place_height=0.009)
    # print("Smooth path length: ", path_length(smooth_path_configs))

    # print("Initial endpoint position: ", endpoint_xyz)
    # endpoint_xyz = kinematics.FK(robot_config[0][2:], xyz=True)
    # kinematics = RobotArmKinematics()

    # base = PID_Base(ob, smooth_path_configs)
    # history = []
    # for step in range(n_steps):
    #     if not history:
    #         action = follow_path(ob, smooth_path_configs)  # Action space is 9 dimensional
    #     else:
    #         action = base.pid_follow_path(ob)
    #         if action == "DONE":
    #             break
    #     ob, _, done, _ = env.step(action)
    #     history.append(ob)
    #     if done:
    #         print("DONE")

    transform_to_arm(ob)

    robot_config = [ob['robot_0']['joint_state']['position'], albert_radius]
    robot_pos_config = np.pad(robot_config[0][0:2], (0, 1))  # (x, y, 0)
    arm_base_position = robot_pos_config + np.array([0, 0, 0.8])
    claw_position = robot_pos_config + np.array([0.05, 0, 1.4])   # change later!
    p.addUserDebugPoints(
        pointPositions=[arm_base_position, claw_position],
        pointColorsRGB=[[1, 0, 0], [0, 1, 0]],
        pointSize=5
        )
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

    time.sleep(300)
    env.close()

    return history


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert()
