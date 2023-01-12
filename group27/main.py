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
from transforms import * 
from local_path_planning import follow_path, path_smoother, interpolate_path, PID_Base, PID_arm
from urdf_env_helpers import add_obstacles, add_goal, add_graph_to_env, draw_node_configs, draw_path, transform_to_arm, add_obstacles_3D, draw_domain, \
    add_sphere
from arm_kinematics import RobotArmKinematics

# change to 1, 2, 3 or "random" for obstacle setup in environment
obstacle_setup = 1


def run_albert(n_steps=500000, render=True, goal=True, obstacles=True, at_end=True, seed=42, albert_radius=0.3):
    robots = [
        AlbertRobot(mode="vel"),
    ]
    sensor = FullSensor(goal_mask=['position', 'radius'], obstacle_mask=['position', 'radius'])
    robots[0].add_sensor(sensor)

    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )

    kinematics = RobotArmKinematics()

    # Init environment (robot position, obstacles, goals)
    pos0 = np.array([-10.0, -10.0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0])  # might change later
    # pos0 = np.array([-0.02934186, 1.09177044, -1.01096624, -0.286753, -0.40583808, 0.35806924, -0.82694153, 0.15506737,  0.24741826, -0.08700108])
    if at_end:
        pos0 = np.hstack(([0.0, 0.0, np.pi], kinematics.inital_pose))

    env.reset(pos=pos0)
    random.seed(seed)
    np.random.seed(seed)

    if not at_end:
        if obstacles:
            add_obstacles(env, obstacle_setup)
        if goal:
            add_goal(env, table_position=[0, 1, 0], albert_radius=albert_radius)

        p.resetDebugVisualizerCamera(cameraDistance=16, cameraYaw=0, cameraPitch=-89.99, cameraTargetPosition=[0, 0, 0])

    # Perform 1 random action to get the initial observation (containing obstacles & goal)
    for step in range(100):
        action = np.zeros(9)
        ob, _, _, _ = env.step(action)
    action = np.zeros(env.n())
    ob, _, _, _ = env.step(action)

    print(f"Initial observation : {ob['robot_0']}")  # This now contains the obstacles and goal (env.reset(pos=pos0) did not)

    if not at_end:
        # Calculate path
        robot_config = [ob['robot_0']['joint_state']['position'], albert_radius]
        goal_config = ob['robot_0']['goals'][0][0]
        obstacle_configs = [obstacle_config for obstacle_config in ob['robot_0']['obstacles']]
        robot_pos_config = np.pad(robot_config[0][0:2], (0, 1))  # (x, y, 0)

        print('obstacle_configs:', obstacle_configs)
        print('goal_config:', goal_config)
        print('robot_config:', robot_config)

        collision_manager = CollisionManager(obstacle_configs, albert_radius)
        domain = {'xmin': -10, 'xmax': 10, 'ymin': -10, 'ymax': 10, 'zmin': 0, 'zmax': 0}
        draw_domain(domain, place_height=0.007)

        rrt_star_smart = RRTStarSmart(robot_pos_config, goal_config, collision_manager, domain, seed=seed)
        rrt_star_smart.smart_run(total_duration=10, rrt_factor=30, smart_sample_ratio=0.5, smart_radius=1)
        shortest_path = nx.shortest_path(rrt_star_smart.graph, 0, -1, weight='weight')
        shortest_path_configs = [rrt_star_smart.graph.nodes[node]['config'] for node in shortest_path]

        add_graph_to_env(rrt_star_smart.graph, 0.005)
        draw_node_configs(rrt_star_smart.biased_sampled_configs, 0.005)
        draw_path(shortest_path_configs, place_height=0.007)
        print("Shortest path length: ", path_length(shortest_path_configs))
        #
        interpolated_path_configs = interpolate_path(shortest_path_configs, max_dist=2.5)
        smooth_path_configs = path_smoother(interpolated_path_configs)
        draw_path(smooth_path_configs, line_color=[1, 0, 0], place_height=0.009)
        print("Smooth path length: ", path_length(smooth_path_configs))

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

    # transform_to_arm(ob)

    robot_config = [ob['robot_0']['joint_state']['position'], albert_radius]
    robot_pos_config = np.pad(robot_config[0][0:2], (0, 1))  # (x, y, 0)
    claw_end_position = T_world_arm(kinematics.FK(robot_config[0][3:], xyz=True), robot_config)

    arm_radius = 0.855
    arm_height = 1.119
    albert_height = 0.8
    claw_goal_positions = [T_world_arm([0.3, 0.6, 0.3], robot_config), T_world_arm([-.5, 0.3, 0.3], robot_config), T_world_arm([-.5, -0.3, 0.3], robot_config)]

    p.addUserDebugPoints(
        pointPositions=[claw_end_position, *claw_goal_positions],
        pointColorsRGB=[[1, 0, 0], *[[1, 0, 1] for _ in claw_goal_positions]],
        pointSize=5
    )
    arm_domain = {'xmin': robot_pos_config[0] - arm_radius, 'xmax': robot_pos_config[0] + arm_radius,
                  'ymin': robot_pos_config[1] - arm_radius, 'ymax': robot_pos_config[1] + arm_radius,
                  'zmin': albert_height, 'zmax': albert_height + arm_height}
    draw_domain(arm_domain)

    for x in range(10):
        x *= 0.08
        for y in range(15):
            y *= 0.08
            obstacle_pos = T_world_arm([-1.2+x, 0, 0.1+y], robot_config)
            add_sphere(env, pos=obstacle_pos.tolist(), radius=0.08)

    action = np.random.random(env.n())
    ob, _, _, _ = env.step(action)
    print(f"Initial observation : {ob['robot_0']}")  # This now contains the obstacles and goal (env.reset(pos=pos0) did not)
    obstacle_configs = [obstacle_config for obstacle_config in ob['robot_0']['obstacles']]
    arm_collision_manager = CollisionManager(obstacle_configs, 0.15)

    for claw_goal_position in claw_goal_positions:
        arm_rrt_star_smart = RRTStarSmart(claw_end_position, claw_goal_position, arm_collision_manager, arm_domain, seed=seed)
        arm_rrt_star_smart.smart_run(total_duration=2, rrt_factor=1.5, smart_sample_ratio=0.1, smart_radius=0.1)
        arm_shortest_path = nx.shortest_path(arm_rrt_star_smart.graph, 0, -1, weight='weight')
        arm_shortest_path_configs = [arm_rrt_star_smart.graph.nodes[node]['config'] for node in arm_shortest_path]
        draw_path(arm_shortest_path_configs, line_color=[0.3, 0.5, 0.1], place_height=0)

        arm_controller = PID_arm(kinematics)
        for step in range(n_steps):
            joint_positions = ob['robot_0']['joint_state']['position'][3:]
            robot_config = [ob['robot_0']['joint_state']['position'], albert_radius]
            arm_end_pos = T_world_arm(kinematics.FK(robot_config[0][3:], xyz=True), robot_config)

            if np.linalg.norm(arm_end_pos - arm_shortest_path_configs[0]) < 0.05:
                claw_end_position = arm_shortest_path_configs.pop(0)
                if not arm_shortest_path_configs:
                    break

            arm_goal_robot_frame = T_arm_world(arm_shortest_path_configs[0], robot_config)
            joint_vel = arm_controller.PID(arm_goal_robot_frame, joint_positions, endpoint_orientation=True)
            action = np.hstack((np.zeros(2), joint_vel))  # Action space is 9 dimensional
            ob, _, _, _ = env.step(action)

    time.sleep(300)
    env.close()
    return history


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert()
