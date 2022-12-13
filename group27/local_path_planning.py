import numpy as np
from scipy import interpolate


def get_robot_config(ob):
    return ob['robot_0']['joint_state']['position']

def path_smoother(shortest_path_configs):
    x = []
    y = []

    for point in shortest_path_configs:
        x.append(point[0])
        y.append(point[1])
    tck, *rest = interpolate.splprep([x,y])
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
