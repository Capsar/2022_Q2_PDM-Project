import numpy as np
from scipy import interpolate
import copy

import matplotlib


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


# lists for plotting
velocities = {
    "velocity": [],
    "control": [],
    "goal": [],
    "diff": []
}
angles = {
    "diff": []
}
nodes = []



def pid_follow_path(ob, ob_prev, shortest_path_configs):
    """
    PID following path.
    """
    threshold_angle = 25  # threshold for angle difference, if > then increase speed
    max_velocity = 2

    if len(shortest_path_configs) == 0:
        return "DONE"

    first_node = shortest_path_configs[0]
    robot_config = get_robot_config(ob)
    prev_robot_config = get_robot_config(ob_prev)

    distance_diff = np.linalg.norm(np.pad(robot_config[0:2], (0, 1)) - first_node)

    # pop nearest node if the robot is close enough
    if distance_diff < 0.1:
        shortest_path_configs.pop(0)
        nodes.append(len(velocities["velocity"]))

    # calculate the angle between goal and robot
    angle_between = np.arctan2(first_node[1] - robot_config[1], first_node[0] - robot_config[0])
    angle_diff = angle_between - robot_config[2]

    # calculate previous differences
    previous_angle_diff = angle_between - prev_robot_config[2]

    # define PID gains, [velocity gain, angle gain]
    kp = [1, 4]
    ki = [0.1, 1]
    kd = [0.01, 50]

    # initialize errors
    integral_error = [0.0, 0.0]
    derivative_error = [0.0, 0.0]

    # update errors for angle
    integral_error[1] += angle_diff
    derivative_error[1] = angle_diff - previous_angle_diff

    # calculate angle action
    control_angle = kp[1] * angle_diff + ki[1] * integral_error[1] + kd[1] * derivative_error[1]

    # calculate desired velocity based on angle difference
    angle_diff_degree = abs(angle_diff * 180 / np.pi)
    velocity_goal = max(max_velocity - (max_velocity/threshold_angle) * angle_diff_degree, 0.5)
    velocity_diff = velocity_goal - get_robot_velocity(ob)

    # update errors for velocity
    integral_error[0] += velocity_diff
    derivative_error[0] = velocity_diff - (get_robot_velocity(ob) - get_robot_velocity(ob_prev))

    # print(f'Velocity goal: {velocity_goal}')

    # calculate velocity action
    velocity_control = kp[0] * get_robot_velocity(ob) + ki[0] * integral_error[0] + kd[0] * derivative_error[0]

    # print(f'Velocity control : {velocity_control}')
    # print(f'Difference: {velocity_control-velocity_goal}')

    velocities["velocity"].append(get_robot_velocity(ob))
    velocities["control"].append(velocity_control)
    velocities["goal"].append(velocity_goal)
    velocities["diff"].append(velocity_control-velocity_goal)

    angles["diff"].append(angle_diff)

    return np.array([max(velocity_control, 0.5), control_angle, 0, 0, 0, 0, 0, 0, 0])





