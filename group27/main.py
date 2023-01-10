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
from local_path_planning import follow_path, path_smoother, interpolate_path, PID_Base, PID_arm
from urdf_env_helpers import add_obstacles, add_goal, add_graph_to_env, draw_node_configs, draw_path, transform_to_arm, add_obstacles_3D, draw_domain, \
    add_sphere
from arm_kinematics import RobotArmKinematics

def run_albert(n_steps=500000, render=True, goal=True, obstacles=True, at_end=False, seed=42, albert_radius=0.3):
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
        # pos0 = np.array([-0.02934186, 1.09177044, -1.01096624, 0.4, 0, .5, 0.4, np.pi, np.pi, 0])
        pos0 = kinematics.inital_pose

    env.reset(pos=pos0)
    random.seed(seed)
    np.random.seed(seed)

    if not at_end:
        if obstacles:
            add_obstacles(env)
        if goal:
            add_goal(env, table_position=[0, 1, 0], albert_radius=albert_radius)

        p.resetDebugVisualizerCamera(cameraDistance=16, cameraYaw=0, cameraPitch=-89.99, cameraTargetPosition=[0, 0, 0])

    # Perform 1 random action to get the initial observation (containing obstacles & goal)
    for step in range(100):
        action = np.zeros(env.n())
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
    claw_position, arm_base_position = kinematics.Endpoint_world_frame(robot_config)

    arm_radius = 0.855
    arm_height = 1.119
    albert_height = 0.8
    claw_goal_position = claw_position + np.array([0.8, -0.4, -0.9])
    p.addUserDebugPoints(
        pointPositions=[arm_base_position, claw_position, claw_goal_position],
        pointColorsRGB=[[1, 0, 0], [0, 0, 1], [0, 1, 0]],
        pointSize=5
        )
    arm_domain = {'xmin': robot_pos_config[0]-arm_radius, 'xmax': robot_pos_config[0]+arm_radius,
              'ymin': robot_pos_config[1]-arm_radius, 'ymax': robot_pos_config[1]+arm_radius,
              'zmin': albert_height, 'zmax': albert_height+arm_height}

    draw_domain(arm_domain)

    # for x in range(10):
    #     x *= 0.1
    #     for y in range(10):
    #         y*= 0.1
    #         obstacle_pos = claw_goal_position + np.array([-0.3+x, -0.1+y, 0.55-y])
    #         add_sphere(env, pos=obstacle_pos.tolist(), radius=0.1)
    #
    #         obstacle_pos = claw_goal_position + np.array([-0.3 + x, -0.1-y, 0.55])
    #         add_sphere(env, pos=obstacle_pos.tolist(), radius=0.1)

    action = np.random.random(env.n())
    ob, _, _, _ = env.step(action)
    print(f"Initial observation : {ob['robot_0']}")  # This now contains the obstacles and goal (env.reset(pos=pos0) did not)
    obstacle_configs = [obstacle_config for obstacle_config in ob['robot_0']['obstacles']]

    arm_collision_manager = CollisionManager(obstacle_configs, 0.01)
    arm_rrt_star_smart = RRTStarSmart(claw_position, claw_goal_position, arm_collision_manager, arm_domain, seed=seed)
    arm_rrt_star_smart.smart_run(total_duration=2, rrt_factor=3, smart_sample_ratio=0.0, smart_radius=0.0)
    add_graph_to_env(arm_rrt_star_smart.graph, 0)
    arm_shortest_path = nx.shortest_path(arm_rrt_star_smart.graph, 0, -1, weight='weight')
    arm_shortest_path_configs = [arm_rrt_star_smart.graph.nodes[node]['config'] for node in arm_shortest_path]
    draw_path(arm_shortest_path_configs, line_color=[0.3, 0.5, 0.1], place_height=0)

    # pid_arm = PID_arm(kinematics)
    # for step in range(n_steps):
    #     action = pid_arm.PID(arm_shortest_path_configs, ob['robot_0']['joint_state']['position'])
    #     ob, _, done, _ = env.step(np.hstack((np.zeros(2), action)))

    time.sleep(300)
    env.close()
    return history


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert()
