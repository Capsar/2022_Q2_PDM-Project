import numpy as np
from scipy import interpolate
import copy


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


def PID_follow_path(ob, ob_prev, shortest_path_configs):
    """
    PID following path.
    """
    threshold_diff = 0.1  # threshold for lateral distance, if > 1 Albert can increase speed
    threshold_angle = 0.2  # threshold for angle difference, if > then increase speed
    max_speed = 2
    max_acceleration = 0.01

    first_node = shortest_path_configs[0]
    second_node = shortest_path_configs[1]   # used to calculate lateral distance
    robot_config = get_robot_config(ob)
    prev_robot_config = get_robot_config(ob_prev)

    distance_diff = np.linalg.norm(np.pad(robot_config[0:2], (0, 1)) - first_node)

    # calculate lateral distance to the line, used to decide whether Albert should accelerate or not
    distance_lat = (abs((second_node[0] - first_node[0]) * (first_node[1] - robot_config[1]) -
                        (first_node[0] - robot_config[0]) * (second_node[1] - first_node[1]))
                    / np.sqrt((second_node[0] - first_node[0])**2 + (second_node[1] - first_node[1])**2))

    # pop nearest node if the robot is close enough
    if distance_diff < 0.1:
        shortest_path_configs.pop(0)

    # calculate the angle between goal and robot
    angle_between = np.arctan2(first_node[1] - robot_config[1], first_node[0] - robot_config[0])
    angle_diff = angle_between - robot_config[2]

    # calculate previous differences
    previous_distance_diff = np.linalg.norm(np.pad(prev_robot_config[0:2], (0, 1)) - first_node)
    previous_angle_diff = angle_between - prev_robot_config[2]

    # define PID gains, [velocity gain, angle gain]
    kp = [.1, 1.0]
    ki = [0.01, 0.1]
    kd = [0.001, 0.1]

    # initialize errors
    integral_error = [0.0, 0.0]
    derivative_error = [0.0, 0.0]

    # update integral error
    integral_error[0] += distance_diff
    integral_error[1] += angle_diff

    # update derivative error
    derivative_error[0] = distance_diff - previous_distance_diff
    derivative_error[1] = angle_diff - previous_angle_diff

    # calculate angle action
    control_angle = kp[1] * angle_diff + ki[1] * integral_error[1] + kd[1] * derivative_error[1]

    control_vel = 0.5
    if abs(angle_diff) <= threshold_angle:
        if get_robot_velocity(ob) < max_speed:
            control_vel = get_robot_velocity(ob) + max_acceleration
            # print(f"Accelerating...from {get_robot_velocity(ob)} to {control_vel}")
    else:
        control_vel = get_robot_velocity(ob) - max_acceleration
        # print(f"Slowing down...from {get_robot_velocity(ob)} to {control_vel}")

    # print(f"Angle difference: {angle_diff}, Angle control: {control_angle}")

    # return action
    return np.array([max(control_vel, 0.5), control_angle, 0, 0, 0, 0, 0, 0, 0])

class PID_arm:
    """
    PID for arm to follow path
    """
    def __init__(self, arm_model, kp = 1.0, ki = 0.1, kd = 0.01):
        self.arm_model = arm_model
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.errors = [0]
        self.integral_error = 0.0



    def PID(self, goal, joint_positions, endpoint_orientation=False):
        q = joint_positions
        state = self.arm_model.FK(joint_positions)
        goal_state = np.vstack((state[:,:9], goal.reshape(-1,1) ))

        error = goal_state - state
        derivative_error = error - self.errors[-1]
        self.integral_error += error

        J = self.arm_model.Jacobian(joint_positions)

        endpoint_vel = self.kp * error + self.ki * self.integral_error + self.kd * derivative_error

        joint_vel = np.linalg.pinv(J) @ endpoint_vel 

        return joint_vel







        






