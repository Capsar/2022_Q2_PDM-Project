import os
import random
import warnings

import gym
import networkx as nx
from urdfenvs.robots.albert import AlbertRobot
from urdfenvs.sensors.full_sensor import FullSensor
import numpy as np

from group27.global_path_planning import CollisionManager, RRTStarSmart, RRTStar
from group27.urdf_env_helpers import add_obstacles, add_goal, draw_domain

import csv
import pybullet as p


def test_run(rrt_instance, run_function, params):
    run_function(*params)
    shortest_path = nx.shortest_path(rrt_instance.graph, 0, -1, weight='weight')
    shortest_path_length = nx.shortest_path_length(rrt_instance.graph, 0, -1, weight='weight')
    if shortest_path_length != 101010:
        return len(rrt_instance.graph.nodes), len(shortest_path), shortest_path_length
    else:
        return len(rrt_instance.graph.nodes), -1, -1


def test_rrt_star(seed, albert_radius, seed_factor):
    rrt_star_settings = [
        {
            'n': 10,
            'rrt_factor': [15, 20, 25, 30, 35, 40],
            'duration': [2.5, 5, 7.5, 10, 12.5, 15, 17.5]
        }, {
            'n': 10,
            'rrt_factor': [15, 20, 25, 30, 35, 40],
            'duration': [5, 8, 11, 14, 17, 20, 23]
        }, {
            'n': 10,
            'rrt_factor': [15, 20, 25, 30, 35, 40],
            'duration': [1, 2, 3, 4, 5, 6, 7]
        }
    ]

    for i, rrt_star_setting in enumerate(rrt_star_settings):
        obstacle_setup = i + 1
        robot_config, goal_config, obstacle_configs, robot_pos_config = init_env(obstacle_setup, seed, albert_radius)
        collision_manager = CollisionManager(obstacle_configs, albert_radius)
        domain = {'xmin': -10, 'xmax': 10, 'ymin': -10, 'ymax': 10, 'zmin': 0, 'zmax': 0}
        draw_domain(domain, place_height=0.007)
        for total_duration in rrt_star_setting['duration']:
            for rrt_factor in rrt_star_setting['rrt_factor']:
                for seed in range(rrt_star_setting['n']):
                    rrt_star = RRTStar(robot_pos_config, goal_config, collision_manager, domain, seed=seed * seed_factor)
                    sampled_n_nodes, path_n_nodes, path_length = test_run(rrt_star, rrt_star.run, [total_duration, rrt_factor])
                    print(f"RRT*, {obstacle_setup}, {total_duration}, {rrt_factor}, {seed * seed_factor}, {sampled_n_nodes}, {path_n_nodes}, {path_length}")
                    with open("../results_rrt_star/RRT-Star.csv", 'a') as file:
                        writer = csv.writer(file)
                        writer.writerow(["RRT*", obstacle_setup, total_duration, rrt_factor, seed * seed_factor, sampled_n_nodes, path_n_nodes, path_length])

def test_rrt_star_smart(seed, albert_radius, seed_factor):
    rrt_star_smart_settings = [
        {
            'setup': 1,
            'n': 10,
            'rrt_factor': 25,
            'smart_ratio': [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
            'smart_radius': [0.5, 1.0],
            'duration': [2.5, 5, 7.5, 10, 12.5, 15, 17.5]
        }, {
            'setup': 2,
            'n': 10,
            'rrt_factor': 25,
            'smart_ratio': [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
            'smart_radius': [0.5, 1.0],
            'duration': [5, 8, 11, 14, 17, 20, 23]
        }, {
            'setup': 3,
            'n': 10,
            'rrt_factor': 20,
            'smart_ratio': [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
            'smart_radius': [0.5, 1.0],
            'duration': [1, 2, 3, 4, 5, 6, 7]
        }
    ]

    for rrt_star_smart_setting in rrt_star_smart_settings:
        obstacle_setup = rrt_star_smart_setting['setup']
        robot_config, goal_config, obstacle_configs, robot_pos_config = init_env(obstacle_setup, seed, albert_radius)
        collision_manager = CollisionManager(obstacle_configs, albert_radius)
        domain = {'xmin': -10, 'xmax': 10, 'ymin': -10, 'ymax': 10, 'zmin': 0, 'zmax': 0}
        draw_domain(domain, place_height=0.007)
        rrt_factor = rrt_star_smart_setting['rrt_factor']
        for total_duration in rrt_star_smart_setting['duration']:
            for smart_ratio in rrt_star_smart_setting['smart_ratio']:
                for smart_radius in rrt_star_smart_setting['smart_radius']:
                    for seed in range(rrt_star_smart_setting['n']):
                        rrt_star_smart = RRTStarSmart(robot_pos_config, goal_config, collision_manager, domain, seed=seed)
                        sampled_n_nodes, path_n_nodes, path_length = test_run(rrt_star_smart, rrt_star_smart.smart_run,
                                                                              [total_duration, rrt_factor, smart_ratio, smart_radius])
                        print(
                            f"RRT*-Smart, {obstacle_setup}, {total_duration}, {rrt_factor}, {smart_ratio}, {smart_radius}, {seed * seed_factor}, {sampled_n_nodes}, {path_n_nodes}, {path_length}")
                        with open("../results_rrt_star/RRT-Star-Smart.csv", 'a') as file:
                            writer = csv.writer(file)
                            writer.writerow(
                                ["RRT*-Smart", obstacle_setup, total_duration, rrt_factor, smart_ratio, smart_radius, seed * seed_factor, sampled_n_nodes,
                                 path_n_nodes, path_length])



def main(seed=42, albert_radius=0.3, seed_factor=13):
    test_rrt_star(seed, albert_radius, seed_factor)
    test_rrt_star_smart(seed, albert_radius, seed_factor)


def init_env(obstacle_setup, seed, albert_radius):
    robots = [
        AlbertRobot(mode="vel")
    ]
    sensor = FullSensor(goal_mask=['position', 'radius'], obstacle_mask=['position', 'radius'])
    robots[0].add_sensor(sensor)

    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=False
    )

    # Init environment (robot position, obstacles, goals)
    pos0 = np.array([-10.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0])  # might change later

    env.reset(pos=pos0)
    random.seed(seed)
    np.random.seed(seed)

    add_obstacles(env, obstacle_setup)
    add_goal(env, table_position=[0, 1, 0], albert_radius=albert_radius)

    # p.resetDebugVisualizerCamera(cameraDistance=16, cameraYaw=0, cameraPitch=-89.99, cameraTargetPosition=[0, 0, 0])

    # Perform 1 random action to get the initial observation (containing obstacles & goal)
    action = np.zeros(env.n())
    for _ in range(50):
        ob, _, _, _ = env.step(action)
    ob, _, _, _ = env.step(action)

    robot_config = [ob['robot_0']['joint_state']['position'], albert_radius]
    goal_config = ob['robot_0']['goals'][0][0]
    obstacle_configs = [obstacle_config for obstacle_config in ob['robot_0']['obstacles']]
    robot_pos_config = np.pad(robot_config[0][0:2], (0, 1))  # (x, y, 0)
    return robot_config, goal_config, obstacle_configs, robot_pos_config


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        main()
