import numpy as np


def get_robot_config(ob):
    return ob['robot_0']['joint_state']['position']


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
