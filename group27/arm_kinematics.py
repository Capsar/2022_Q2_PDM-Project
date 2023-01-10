from sympy import symbols, init_printing, Matrix, eye, sin, cos, pi
init_printing(use_unicode=True)
import numpy as np
from sympy import lambdify
from numba import jit


class RobotArmKinematics:
    def __init__(self):
            
        # create joint angles as symbols
        q1, q2, q3, q4, q5, q6, q7 = symbols('theta_1 theta_2 theta_3 theta_4 theta_5 theta_6 theta_7')
        joint_angles = [q1, q2, q3, q4, q5, q6, q7]

        # construct symbolic direct kinematics  from Craig's DH parameters
        # see https://frankaemika.github.io/docs/control_parameters.html
        dh_craig = [
            {'a':  0,      'd': 0.333, 'alpha':  0,  },
            {'a':  0,      'd': 0,     'alpha': -pi/2},
            {'a':  0,      'd': 0.316, 'alpha':  pi/2},
            {'a':  0.0825, 'd': 0,     'alpha':  pi/2},
            {'a': -0.0825, 'd': 0.384, 'alpha': -pi/2},
            {'a':  0,      'd': 0,     'alpha':  pi/2},
            {'a':  0.088,  'd': 0.107, 'alpha':  pi/2},
        ]

        DK = eye(4)
        for i, (p, q) in enumerate(zip(reversed(dh_craig), reversed(joint_angles))):
            d = p['d']
            a = p['a']
            alpha = p['alpha']
            ca = cos(alpha)
            sa = sin(alpha)
            cq = cos(q)
            sq = sin(q)
            transform = Matrix(
                [
                    [cq, -sq, 0, a],
                    [ca * sq, ca * cq, -sa, -d * sa],
                    [sa * sq, cq * sa, ca, d * ca],
                    [0, 0, 0, 1],
                ]
            )

            DK = transform @ DK
        A = DK[0:3, 0:4]  # crop last row
        A = A.transpose().reshape(12,1)  # reshape to column vector A = [a11, a21, a31, ..., a34]

        Q = Matrix(joint_angles)
        J = A.jacobian(Q)  # compute Jacobian symbolically

        self.A_lamb = (lambdify((q1, q2, q3, q4, q5, q6, q7), A, 'numpy'))
        self.J_lamb = (lambdify((q1, q2, q3, q4, q5, q6, q7), J, 'numpy'))

        self.joint_limits = [
            (-2.8973, 2.8973),
            (-1.7628, 1.7628),
            (-2.8973, 2.8973),
            (-3.0718, -0.0698),
            (-2.8973, 2.8973),
            (-0.0175, 3.7525),
            (-2.8973, 2.8973)
        ]
        self.max_joint_speed = np.array([
            [2.1750],
            [2.1750],
            [2.1750],
            [2.1750],
            [2.6100],
            [2.6100],
            [2.6100]
            ])
        self.inital_pose = np.array([l+(u-l)/2 for l, u in self.joint_limits], dtype=np.float64)
        

    def FK(self, joint_positions, xyz=False):
        q = joint_positions
        A = self.A_lamb(q[0], q[1], q[2], q[3], q[4], q[5], q[6])
        if xyz:
            return A.flatten()[-3:]
        else:
            return A

    def Endpoint_world_frame(self, robot_config):
        joint_positions = robot_config[0][3:]
        endpoint_arm_frame = self.FK(joint_positions, xyz=True)
        arm_mount_base_link_frame = np.array([-0.15, 0.0, 0.487])
        enpoint_base_link_frame = endpoint_arm_frame + arm_mount_base_link_frame
        base_position = [0, 0, 0.13]
        base_position[:2] = robot_config[0][:2]
        base_rotation = robot_config[0][2]
        base_rotation_matrix = np.array([[np.cos(base_rotation), -np.sin(base_rotation), 0],
                                         [np.sin(base_rotation), np.cos(base_rotation), 0],
                                         [0, 0, 1]])
        endpoint_world_frame = base_position + base_rotation_matrix @ enpoint_base_link_frame
        arm_mount_world_frame = base_position + base_rotation_matrix @ arm_mount_base_link_frame
        return endpoint_world_frame, arm_mount_world_frame

    def jacobian(self, joint_positions):
        q = joint_positions
        J = self.J_lamb(q[0], q[1], q[2], q[3], q[4], q[5], q[6])
        J = J/np.linalg.norm(J)
        return J
    



    
