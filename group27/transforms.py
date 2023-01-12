import numpy as np

def get_T_world_robot(robot_config, base_link_position=False):
    base_rotation = robot_config[0][2]
    base_x, base_y = robot_config[0][:2]
    base_z = 0.13
    base_pos = np.array([base_x, base_y, base_z])
    arm_mount_base_link_frame = np.array([-0.15, 0.0, 0.48])
    base_rotation_matrix = rotate_Z(-base_rotation)
    translation = base_pos + rotate_Z(base_rotation) @ arm_mount_base_link_frame

    T_matrix = np.hstack((base_rotation_matrix, -translation[:,np.newaxis]))
    T_matrix = np.vstack((T_matrix, np.array([0,0,0,1])))
    if base_link_position:
        return translation
    else:
        return T_matrix


def T_robot_world(xyz_world, robot_config):  # from world to robot
    xyz_world_homogeneous = np.hstack((xyz_world, np.array([1])))
    xyz_robot_homogeneous = get_T_world_robot(robot_config) @ xyz_world_homogeneous
    return xyz_robot_homogeneous[:3]


def T_world_robot(xyz_robot, robot_config):  # from robot to world
    xyz_robot_homogeneous = np.hstack((xyz_robot, np.array([1])))
    xyz_world_homogeneous = np.linalg.inv(get_T_world_robot(robot_config)) @ xyz_robot_homogeneous
    return xyz_world_homogeneous[:3]


def rotate_X(angle):
    return np.array([[1, 0, 0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])


def rotate_Y(angle):
    return np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])


def rotate_Z(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])
