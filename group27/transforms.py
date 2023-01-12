import numpy as np

def get_T_world_rbase(robot_config, base_link_position=False): # get the transformation matrix from robot base frame to world frame
    base_rotation = robot_config[0][2]
    base_x, base_y = robot_config[0][:2]
    base_z = 0.13
    base_pos = np.array([base_x, base_y, base_z])
    base_rotation_matrix = rotate_Z(base_rotation)
    translation = base_pos 

    T_matrix = np.hstack((base_rotation_matrix, translation[:,np.newaxis]))
    T_matrix = np.vstack((T_matrix, np.array([0,0,0,1])))
    if base_link_position:
        return translation
    else:
        return T_matrix



def T_rbase_arm(xyz_arm): # transform robot arm frame -> robot base link frame
    arm_mount_base_link_frame = np.array([-0.15, 0.0, 0.48])
    return xyz_arm + arm_mount_base_link_frame

def T_arm_rbase(xyz_arm): # transform robot base link frame -> robot arm frame
    arm_mount_base_link_frame = np.array([-0.15, 0.0, 0.48])
    return xyz_arm - arm_mount_base_link_frame



def T_world_arm(xyz_arm, robot_config): # transform robot arm frame -> world frame
    xyz_rbase = T_rbase_arm(xyz_arm)
    xyz_world = get_T_world_rbase(robot_config) @ homogeneous(xyz_rbase)
    return xyz_world[:3]

def T_arm_world(xyz_world, robot_config): # transform world frame -> robot arm frame
    xyz_rbase = np.linalg.pinv(get_T_world_rbase(robot_config)) @ homogeneous(xyz_world)
    xyz_arm = T_arm_rbase(xyz_rbase[:3])
    return xyz_arm




############  helper funcitons ##########

def homogeneous(xyz): # convert xyz to homogeneous coordinates
    return np.hstack((xyz ,np.array([1])))


def rotate_X(angle): # rotation matrix around X axix
    return np.array([[1, 0, 0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])


def rotate_Y(angle): # rotation matrix around Y axis
    return np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])


def rotate_Z(angle): # rotation matrix arounc Z axis
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])
