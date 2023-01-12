import time
import warnings
from xml.etree.ElementTree import PI

from urdfenvs.robots.albert import AlbertRobot
from urdfenvs.sensors.full_sensor import FullSensor
import numpy as np
import pybullet as p
import gym
import networkx as nx

from transforms import *
from local_path_planning import follow_path, path_smoother,interpolate_path, PID_arm
from urdf_env_helpers import add_obstacles, add_goal, add_graph_to_env, draw_path
from arm_kinematics import RobotArmKinematics


def run_albert(n_steps=500000, render=True, goal=True, obstacles=True, albert_radius=0.3, base_angle = 0):
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
    pos0 = np.hstack((np.array([0, 0, base_angle]), kinematics.inital_pose))
    env.reset(pos=pos0)
    if goal:
        add_goal(env, albert_radius=albert_radius)

    # p.resetDebugVisualizerCamera(cameraDistance=16, cameraYaw=0, cameraPitch=-89.99, cameraTargetPosition=[0, 0, 0])

    # Perform 1 random action to get the initial observation (containing obstacles & goal)
    action = np.random.random(env.n())
    ob, _, _, _ = env.step(action)
    print(f"Initial observation : {ob['robot_0']}")  # This now contains the obstacles and goal (env.reset(pos=pos0) did not)

    # Calculate path
    robot_config = [ob['robot_0']['joint_state']['position'], albert_radius]
    goal_config = ob['robot_0']['goals'][0][0]
    obstacle_configs = [obstacle_config for obstacle_config in ob['robot_0']['obstacles']]
    
    print('goal_config:', goal_config)
    print('robot_config:', robot_config)


    endpoint_xyz = kinematics.FK(robot_config[0][3:], xyz=False)
    print("Initial endpoint position: ", endpoint_xyz)
    jacobian = kinematics.jacobian(robot_config[0][3:])
    print("Initial jacobian: ", jacobian) 


    arm_controller = PID_arm(kinematics)
    arm_goal = np.array([-0.5, -0.4, 1.0
                         ])





    
    history = []
    for step in range(n_steps):
        joint_positions = ob['robot_0']['joint_state']['position'][3:]
        robot_config = [ob['robot_0']['joint_state']['position'], albert_radius]
        arm_goal_robot_frame = T_arm_world(arm_goal, robot_config)

        robot_pos_config = np.pad(robot_config[0][0:2], (0, 1))  # (x, y, 0)

        # claw_end_position = T_world_arm(kinematics.FK(robot_config[0][3:], xyz=True), robot_config)
        # p.addUserDebugPoints(
        #     pointPositions=[claw_end_position, robot_pos_config],
        #     pointColorsRGB=[[1, 0, 0], [0, 1, 0]],
        #     pointSize=5
        # )

        p.addUserDebugPoints(pointPositions=[arm_goal, T_world_arm(kinematics.FK(robot_config[0][3:], xyz=True), robot_config)],
                        pointColorsRGB = [[1,0,1], [1, 0, 0]],
                        pointSize = 10)

        joint_vel = arm_controller.PID(arm_goal_robot_frame, joint_positions, endpoint_orientation=False)
        action = np.hstack((np.zeros(2), joint_vel)) #Action space is 9 dimensional
        ob, _, _, _ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(base_angle=2)

