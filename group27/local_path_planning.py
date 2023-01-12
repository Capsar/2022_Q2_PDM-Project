import numpy as np
from scipy import interpolate
import copy
from matplotlib import pyplot as plt
import math

import matplotlib
from transforms import rotate_X, rotate_Y, rotate_Z


def get_robot_config(ob):
    return ob['robot_0']['joint_state']['position']


def get_robot_velocity(ob):
    return ob['robot_0']['joint_state']['forward_velocity'][0]


def interpolate_path(path, max_dist=5.0):
    """
    This function interpolates points bet between nodes in case nodes are far apart
    """
    interpolated_path = []
    for i in range(len(path)-1):
        x1 = path[i][0]
        y1 = path[i][1]
        x2 = path[i+1][0]
        y2 = path[i+1][1]
        x_dist = x2-x1
        y_dist = y2-y1
        edge_length = np.sqrt(x_dist**2 + y_dist**2)
        interpolated_path.append(path[i])

        if edge_length > max_dist:
            n_nodes = (edge_length // max_dist).astype(int)
            # print("interpolated nodes: ", n_nodes 
            for i in range(n_nodes):
                interpolated_path.append(np.array([x1+(i+1)*x_dist/(n_nodes+1), y1+(i+1)*y_dist/(n_nodes +1), 0]))

    interpolated_path.append(path[-1])
    return interpolated_path


def path_smoother(shortest_path_configs):
    x = []
    y = []

    for point in shortest_path_configs:
        x.append(point[0])
        y.append(point[1])
    tck, *rest = interpolate.splprep([x,y], s=0.1)
    u = np.linspace(0,1,num=100)
    x_smooth, y_smooth = interpolate.splev(u, tck) 

    smooth_configs = [np.array([x_smooth[i], y_smooth[i], 0]) for i in range(len(x_smooth))]
    return smooth_configs


def follow_path(ob, shortest_path_configs):
    """
    TODO: Make the robot follow the path.
    """

    first_node = shortest_path_configs[0]
    robot_config = get_robot_config(ob)
    if np.linalg.norm(np.pad(robot_config[0:2], (0, 1)) - first_node) < 0.1:
        shortest_path_configs.pop(0)

    angle_between = np.arctan2(first_node[1] - robot_config[1], first_node[0] - robot_config[0])
    angle_diff = angle_between - robot_config[2]
    if np.abs(angle_diff) > 0.1:
        return np.array([0.5, 2*angle_diff, 0, 0, 0, 0, 0, 0, 0])
    return np.array([0.5, 0, 0, 0, 0, 0, 0, 0, 0])


class PID_Base:
    """"
    PID controller class for our base
    """

    def __init__(self, ob, path, kp=(1, 3), ki=(0.0, 0.00), kd=(0, 5)):
        # initialize the object and path
        self.ob = ob
        self.path = path

        # controller values (velocity gain, angle gain)
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral_error = [0., 0.]
        self.derivative_error = [0., 0.]

        self.threshold = 25
        self.max_velocity = 1.5

        # values for printing result
        self.velocities = {
            "velocity": [],
            "control": []
        }
        self.angles = {
            "diff": []
        }
        self.vertical_lines = []
        self.angle_control = []

    def pid_follow_path(self, ob_current):
        if len(self.path) == 0:
            return "DONE"

        first_node = self.path[0]
        robot_config = get_robot_config(ob_current)
        prev_robot_config = get_robot_config(self.ob)

        distance_diff = np.linalg.norm(np.pad(robot_config[0:2], (0, 1)) - first_node)

        # pop nearest node if the robot is close enough
        if distance_diff < 0.1:
            self.path.pop(0)
            self.vertical_lines.append(len(self.angles["diff"]))

        # calculate the angle between goal and robot
        angle_between = np.arctan2(first_node[1] - robot_config[1], first_node[0] - robot_config[0])
        angle_diff = angle_between - robot_config[2]

        # calculate previous differences
        previous_angle_diff = angle_between - prev_robot_config[2]

        # update errors for angle
        self.integral_error[1] += angle_diff
        self.derivative_error[1] = angle_diff - previous_angle_diff

        # calculate angle action
        control_angle = self.kp[1] * angle_diff + self.ki[1] * self.integral_error[1] + self.kd[1] * self.derivative_error[1]

        # calculate desired velocity based on the angle control
        error = np.exp(abs(control_angle))**-1

        self.integral_error[0] += error
        if len(self.angles["diff"]) > 1:
            # self.derivative_error[0] += error - min(1, 1 - self.angles["diff"][-1])
            self.derivative_error[0] += error - np.exp(abs(self.angles["diff"][-1]))**-1
        else:
            self.derivative_error[0] = 0

        control_velocity = error * self.kp[0] + self.integral_error[0] * self.ki[0] + self.derivative_error[0] * self.kd[0]

        control_velocity = np.clip(control_velocity*self.max_velocity, .0, self.max_velocity)

        self.velocities["control"].append(control_velocity)
        self.velocities["velocity"].append(get_robot_velocity(ob_current))

        self.angles["diff"].append(abs(angle_diff))
        self.angle_control.append(control_angle)

        # update self object
        self.ob = ob_current

        return np.array([control_velocity, control_angle, 0, 0, 0, 0, 0, 0, 0])

    def return_position(self):
        robot_config = get_robot_config(self.ob)
        return np.pad(robot_config[0:2], (0, 1))  # (x, y, 0)

    def plot_results(self, save=False):
        # plotting
        fig, axs = plt.subplots(3, 1, figsize=(15, 10))
        i = 0
        title = f"kp={self.kp}, ki={self.ki}, kd={self.kd}"
        fig.suptitle(title)
        x = range(len(self.angles["diff"]))

        axs[0].plot(x, self.velocities["velocity"], label="velocity")
        axs[0].plot(x, self.velocities["control"], label="control")
        for vertical_line in self.vertical_lines:
            axs[0].axvline(x=vertical_line, color="black")
        axs[0].legend()
        axs[0].set_title(f"Velocities for kp:{self.kp[0]}, ki:{self.ki[0]}, kd:{self.kd[0]}")

        axs[1].plot(x, self.angle_control, label="Angle control")
        for vertical_line in self.vertical_lines:
            axs[1].axvline(x=vertical_line, color="black")
        axs[1].legend()
        axs[1].set_title(f"Angle control for: kp={self.kp[1]}, ki={self.ki[1]}, kd={self.kd[1]}")

        axs[2].plot(x, self.angles["diff"], label="Angle difference")
        axs[2].set_title(f"Angle difference for: kp={self.kp[1]}, ki={self.ki[1]}, kd={self.kd[1]}")
        for vertical_line in self.vertical_lines:
            axs[2].axvline(x=vertical_line, color="black")
        axs[2].legend()

        plt.show()

        if save:
            fig.savefig(f"plots/full run = {title}.png")


class PID_arm:
    """
    PID for arm to follow path
    """
    def __init__(self, arm_model, kp = 0.40, ki = 0.0, kd = 0.00):
        self.arm_model = arm_model
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.errors = [0]
        self.integral_error = 0.0

    def PID(self, goal, joint_positions, endpoint_orientation=False):

        state = self.arm_model.FK(joint_positions)

        if endpoint_orientation:
            goal_state = np.vstack([state[:9], goal.reshape(-1,1)])
        else:
            orientation = np.array([[1, 0, 0],
                                    [0, -1, 0], 
                                    [0, 0, -1]])
            goal_state = np.vstack([orientation.reshape(-1, 1), goal.reshape(-1, 1)])

        error = goal_state - state
        derivative_error = error - self.errors[-1]
        self.integral_error += error

        J = self.arm_model.jacobian(joint_positions)

        endpoint_vel = self.kp * error + self.ki * self.integral_error + self.kd * derivative_error



        joint_vel = np.linalg.pinv(J) @ endpoint_vel 
        joint_vel = np.clip(joint_vel, -self.arm_model.max_joint_speed, self.arm_model.max_joint_speed)

        return joint_vel.flatten()







        






