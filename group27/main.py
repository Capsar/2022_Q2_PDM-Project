import random
import time
import warnings
import argparse
from copy import deepcopy

import gym
import networkx as nx
import pybullet as p
from urdfenvs.robots.albert import AlbertRobot
from urdfenvs.sensors.full_sensor import FullSensor

from arm_kinematics import RobotArmKinematics
from global_path_planning import CollisionManager, RRTStarSmart, path_length
from local_path_planning import follow_path, path_smoother, interpolate_path, PDBaseController, PDArmController
from transforms import *
from urdf_env_helpers import add_obstacles, add_goal, add_graph_to_env, draw_node_configs, draw_path, transform_to_arm, draw_domain, \
    add_sphere

def check_env_type(value):
    if value == "random":
        return value
    elif value in {"1", "2", "3"}:
        return int(value)
    else:
        raise argparse.ArgumentTypeError("%s is an invalid environmnet.  Please select 1,2,3 or 'random'" % value)

parser = argparse.ArgumentParser(description='This file runs the main simulation of the albert robot')
parser.add_argument("--arm_only", help="set arm_only to skip the mobile base navigation part of the simulation, and see only the robot arm path following.", action=argparse.BooleanOptionalAction)
parser.add_argument("--environment", help="select the simulation environment (1,2,3 or 'random')", default=2, type=check_env_type)

args = vars(parser.parse_args())

obstacle_setup = args["environment"]
arm_only = args["arm_only"]
rrt_star_settings = [
    {
        'rrt_factor': 25,
        'duration': 15,
        's_ratio': 1.2,
        's_radius': 0.5
    }, {
        'rrt_factor': 25,
        'duration': 20,
        's_ratio': 0.2,
        's_radius': 1.0
    }, {
        'rrt_factor': 20,
        'duration': 10,
        's_ratio': 0.8,
        's_radius': 1.0
    }
]


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

    # Init the robot arm kinematics.
    kinematics = RobotArmKinematics()

    # Init environment (robot position, obstacles, goals)
    pos0 = np.hstack(([-10.0, -10.0, 0.0], kinematics.inital_pose))
    if at_end:
        pos0 = np.hstack(([0.0, 0.0, np.pi], kinematics.inital_pose))

    env.reset(pos=pos0)
    random.seed(seed)
    np.random.seed(seed)

    # If not to only run the 3D environment, add obstacles and goals to entire environment
    if not at_end:
        if obstacles:
            add_obstacles(env, obstacle_setup)
        if goal:
            add_goal(env, table_position=[0, 1, 0], albert_radius=albert_radius)
        # Set the camera to be in top view of the environment
        p.resetDebugVisualizerCamera(cameraDistance=12, cameraYaw=0, cameraPitch=-89.99, cameraTargetPosition=[0, 0, 0])

    # Perform 1 random action to get the initial observation (containing obstacles & goal)
    action = np.zeros(env.n())
    for step in range(30):
        ob, _, _, _ = env.step(action)
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

        # Initialize RRT*-Smart and run the algorithm with optimized settings.
        rrt_star_smart = RRTStarSmart(robot_pos_config, goal_config, collision_manager, domain, seed=seed)
        rrt_star_smart_settings = rrt_star_settings[obstacle_setup - 1]
        rrt_star_smart.smart_run(total_duration=rrt_star_smart_settings['duration'], rrt_factor=rrt_star_smart_settings['rrt_factor'],
                                 smart_sample_ratio=rrt_star_smart_settings['s_ratio'], smart_radius=rrt_star_smart_settings['s_radius'])
        shortest_path = nx.shortest_path(rrt_star_smart.graph, 0, -1, weight='weight')
        shortest_path_configs = [rrt_star_smart.graph.nodes[node]['config'] for node in shortest_path]
        add_graph_to_env(rrt_star_smart.graph, 0.005)
        draw_node_configs(rrt_star_smart.biased_sampled_configs, 0.005)
        draw_path(shortest_path_configs, place_height=0.007)
        print("Shortest path length: ", path_length(shortest_path_configs))

        # Smooth the path by interpolating and smoothing.
        interpolated_path_configs = interpolate_path(shortest_path_configs, max_dist=2.0)
        smooth_path_configs = path_smoother(interpolated_path_configs)
        draw_path(smooth_path_configs, line_color=[1, 0, 0], place_height=0.009)
        print("Smooth path length: ", path_length(smooth_path_configs))

        # Init the controller for the base and start following the path.
        base = PDBaseController(ob, smooth_path_configs)
        for step in range(n_steps):
            if step == 0:
                action = follow_path(ob, smooth_path_configs)  # Action space is 9 dimensional
            else:
                action = base.pid_follow_path(ob)
                if action == "DONE":
                    break
            ob, _, _, _ = env.step(action)

    # Transform the camera to be close to the robot.
    transform_to_arm(ob)

    # Add obstacles to the arm domain
    robot_config = [ob['robot_0']['joint_state']['position'], albert_radius]
    for x in range(10):
        x *= 0.08
        for y in range(15):
            y *= 0.08
            obstacle_pos = T_world_arm([-1.2 + x, 0, 0.1 + y], robot_config)
            add_sphere(env, pos=obstacle_pos.tolist(), radius=0.08)

    robot_config = [ob['robot_0']['joint_state']['position'], albert_radius]
    robot_pos_config = np.pad(robot_config[0][0:2], (0, 1))  # (x, y, 0)
    claw_end_position = T_world_arm(kinematics.FK(robot_config[0][3:], xyz=True), robot_config)

    arm_radius = 0.855
    arm_height = 1.119
    albert_height = 0.8
    claw_goal_positions = [T_world_arm([0.3, 0.6, 0.3], robot_config), T_world_arm([-.5, 0.3, 0.3], robot_config), T_world_arm([-.5, -0.3, 0.3], robot_config)]

    p.addUserDebugPoints(
        pointPositions=[claw_end_position, *claw_goal_positions],
        pointColorsRGB=[[1, 0, 0], *[[0, 1, 1] for _ in claw_goal_positions]],
        pointSize=5
    )
    arm_domain = {'xmin': robot_pos_config[0] - arm_radius, 'xmax': robot_pos_config[0] + arm_radius,
                  'ymin': robot_pos_config[1] - arm_radius, 'ymax': robot_pos_config[1] + arm_radius,
                  'zmin': albert_height, 'zmax': albert_height + arm_height}
    draw_domain(arm_domain)

    action = np.random.random(env.n())
    ob, _, _, _ = env.step(action)
    print(f"Initial observation : {ob['robot_0']}")  # This now contains the obstacles and goal (env.reset(pos=pos0) did not)
    obstacle_configs = [obstacle_config for obstacle_config in ob['robot_0']['obstacles']]
    arm_collision_manager = CollisionManager(obstacle_configs, 0.14)

    for claw_goal_position in claw_goal_positions:
        arm_rrt_star_smart = RRTStarSmart(claw_end_position, claw_goal_position, arm_collision_manager, arm_domain, seed=seed)
        arm_rrt_star_smart.smart_run(total_duration=4, rrt_factor=3, smart_sample_ratio=0.2, smart_radius=0.1)
        arm_shortest_path = nx.shortest_path(arm_rrt_star_smart.graph, 0, -1, weight='weight')
        arm_shortest_path_configs = [arm_rrt_star_smart.graph.nodes[node]['config'] for node in arm_shortest_path]
        draw_path(arm_shortest_path_configs, line_color=[0, 1, 0], place_height=0)
        arm_interpolated_path_configs = interpolate_path(arm_shortest_path_configs, max_dist=0.06)
        arm_smooth_path_configs = path_smoother(arm_interpolated_path_configs)
        draw_path(arm_smooth_path_configs, line_color=[1, 0, 0], place_height=0.009)

        arm_controller = PDArmController(kinematics)
        for step in range(n_steps):
            joint_positions = ob['robot_0']['joint_state']['position'][3:]
            robot_config = [ob['robot_0']['joint_state']['position'], albert_radius]
            arm_end_pos = T_world_arm(kinematics.FK(robot_config[0][3:], xyz=True), robot_config)

            if np.linalg.norm(arm_end_pos - arm_smooth_path_configs[0]) < 0.05:
                claw_end_position = arm_smooth_path_configs.pop(0)
                if not arm_smooth_path_configs:
                    break

            arm_goal_robot_frame = T_arm_world(arm_smooth_path_configs[0], robot_config)
            joint_vel = arm_controller.PID(arm_goal_robot_frame, joint_positions, endpoint_orientation=False)
            action = np.hstack((np.zeros(2), joint_vel))  # Action space is 9 dimensional
            ob, _, _, _ = env.step(action)

    time.sleep(300)
    env.close()


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(at_end=arm_only)

