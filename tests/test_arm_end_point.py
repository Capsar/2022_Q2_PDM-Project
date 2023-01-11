import random
import warnings

import gym
from urdfenvs.robots.albert import AlbertRobot
from urdfenvs.sensors.full_sensor import FullSensor

from group27.arm_kinematics import RobotArmKinematics
from group27.transforms import T_world_robot
import numpy as np

from group27.urdf_env_helpers import add_obstacles, add_goal
import pybullet as p

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
        pos0 = np.hstack(([0.0, 0.0, 0], kinematics.inital_pose))

    env.reset(pos=pos0)
    random.seed(seed)
    np.random.seed(seed)

    if not at_end:
        if obstacles:
            add_obstacles(env, 1)
        if goal:
            add_goal(env, table_position=[0, 1, 0], albert_radius=albert_radius)

        p.resetDebugVisualizerCamera(cameraDistance=16, cameraYaw=0, cameraPitch=-89.99, cameraTargetPosition=[0, 0, 0])

    p.addUserDebugPoints(
        pointPositions=[[0, 0, 0]],
        pointColorsRGB=[[1, 1, 0]],
        pointSize=30
    )
    # Perform 1 random action to get the initial observation (containing obstacles & goal)
    for step in range(10000):
        action = [0, 1, 0, 0, 0, 0, 0, 0, 0]
        ob, _, _, _ = env.step(action)
        robot_config = [ob['robot_0']['joint_state']['position'], albert_radius]
        robot_pos_config = np.pad(robot_config[0][0:2], (0, 1))  # (x, y, 0)
        claw_end_position = T_world_robot(kinematics.FK(robot_config[0][3:], xyz=True), robot_config)
        p.addUserDebugPoints(
            pointPositions=[claw_end_position, robot_pos_config],
            pointColorsRGB=[[1, 0, 0], [0, 1, 0]],
            pointSize=5
        )


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert()
