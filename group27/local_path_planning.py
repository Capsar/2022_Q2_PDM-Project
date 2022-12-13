import numpy as np
from scipy import interpolate
import copy


def get_robot_config(ob):
    return ob['robot_0']['joint_state']['position']

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
    tck, *rest = interpolate.splprep([x,y], s=0.001)
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

    first_node = shortest_path_configs[0]
    robot_config = get_robot_config(ob)
    prev_robot_config = get_robot_config(ob_prev)

    # pop nearest node if the robot is close enough
    if np.linalg.norm(np.pad(robot_config[0:2], (0, 1)) - first_node) < 0.1:
        shortest_path_configs.pop(0)

    # calculate the angle between goal and robot
    angle_between = np.arctan2(first_node[1] - robot_config[1], first_node[0] - robot_config[0])
    angle_diff = angle_between - robot_config[2]

    previous_angle_diff = angle_between - prev_robot_config[2]

    # define PID gains
    kp = 1.0
    ki = 0.1
    kd = 0.01

    # initialize errors
    integral_error = 0.0
    derivative_error = 0.0

    # update integral error
    integral_error += angle_diff

    # update derivative error
    derivative_error = angle_diff - previous_angle_diff

    # calculate angle action
    control_angle = kp * angle_diff + ki * integral_error + kd * derivative_error

    # return action
    return np.array([0.5, control_angle, 0, 0, 0, 0, 0, 0, 0])





