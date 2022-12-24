import time
import warnings

from networkx import DiGraph
from urdfenvs.robots.albert import AlbertRobot
from urdfenvs.sensors.full_sensor import FullSensor
import numpy as np
import pybullet as p
import gym
import networkx as nx
from matplotlib import pyplot as plt

from global_path_planning import rrt_path, sample_points_in_ellipse
from local_path_planning import follow_path, path_smoother,interpolate_path, PID_Base
from urdf_env_helpers import add_obstacles, add_goal, add_graph_to_env, draw_path, transform_camera
# from robot_arm_kinematics import Direct_Kinematics
# from arm_kinematics import RobotArmKinematics


def run_albert(n_steps=500000, render=True, goal=True, obstacles=False, albert_radius=0.3):
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
        add_goal(env, albert_radius=albert_radius)

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
    center_config = robot_pos_config + np.subtract(goal_config, robot_pos_config) / 2
    print(center_config)
    # p.addUserDebugPoints(  # Got from pybullet documentation
    #             pointPositions=[center_config],
    #             pointColorsRGB=[[1, 0, 0]],
    #             pointSize=10
    #         )
    #
    # angle = -np.arctan2(goal_config[0] - robot_pos_config[0], goal_config[1] - robot_pos_config[1])
    # # for _ in range(1000):
    # #     sampled_config = sample_points_in_ellipse(center_config, 5, 20, angle)
    # #     p.addUserDebugPoints(  # Got from pybullet documentation
    # #         pointPositions=[sampled_config],
    # #         pointColorsRGB=[[0, 1, 0]],
    # #         pointSize=5
    # #     )
    #
    # print('obstacle_configs:', obstacle_configs)
    # print('goal_config:', goal_config)
    # print('robot_config:', robot_config)
    #
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
    # shortest_path_configs = [robot_pos_config, [-10, -5, 0], [-5, -5, 0]]
    shortest_path_configs = [robot_pos_config, [-8, -8, 0], [-7, -7, 0]]
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
            action = follow_path(ob, shortest_path_configs)  # Action space is 9 dimensional
        else:
            action = base.pid_follow_path(ob)
            if action == "DONE":
                break
        ob, _, done, _ = env.step(action)
        history.append(ob)
        if done:
            print("DONE")
    # env.close()

    # p.resetDebugVisualizerCamera(cameraDistance=16, cameraYaw=0, cameraPitch=-89.99, cameraTargetPosition=base.return_position())
    transform_camera(4, 45, -45, base.return_position())
    base.plot_results(False)

    return history


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert()


