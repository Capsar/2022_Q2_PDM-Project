import random
import pybullet as p
import math
import numpy as np
import time


def add_obstacles(env, number=20, scale=10.0):
    from MotionPlanningEnv.sphereObstacle import SphereObstacle
    for i in range(number):
        random_x = random.uniform(-1, 1) * scale
        random_z = random.uniform(-1, 1) * scale
        sphere_obst_dict = {
            "type": "sphere",
            'movable': False,
            "geometry": {"position": [random_x, random_z, 0.0], "radius": 0.5},
        }
        sphere_obst = SphereObstacle(name=f'obstacle_{i}', content_dict=sphere_obst_dict)
        env.add_obstacle(sphere_obst)

    for i in range(8):
        sphere_obst_dict = {
            "type": "sphere",
            'movable': False,
            "geometry": {"position": [-10 + i, -6, 0.0], "radius": 0.5},
        }
        sphere_obst = SphereObstacle(name=f'obstacle_{-10 + i}_{-6}', content_dict=sphere_obst_dict)
        env.add_obstacle(sphere_obst)

    for i in range(15):
        sphere_obst_dict = {
            "type": "sphere",
            'movable': False,
            "geometry": {"position": [-5 + i, -2.5, 0.0], "radius": 0.5},
        }
        sphere_obst = SphereObstacle(name=f'obstacle_{-5 + i}_{-2.5}', content_dict=sphere_obst_dict)
        env.add_obstacle(sphere_obst)

        for i in range(7):
            sphere_obst_dict = {
                "type": "sphere",
                'movable': False,
                "geometry": {"position": [-5, -2.5 + i, 0.0], "radius": 0.5},
            }
            sphere_obst = SphereObstacle(name=f'obstacle_{-5}_{-2.5 + i}', content_dict=sphere_obst_dict)
            env.add_obstacle(sphere_obst)

            # Adding walls as obstacles.
    # wall_length = int(scale*2.5)
    # for x in range(wall_length):
    #     for y in range(wall_length):
    #         if x != 0 and x != wall_length-1 and y != 0 and y != wall_length-1:
    #             continue
    #         sphere_obst_dict = {
    #             "type": "sphere",
    #             'movable': False,
    #             "geometry": {"position": [x-int(wall_length/2), y-int(wall_length/2), 0.0], "radius": 0.5},
    #         }
    #         sphere_obst = SphereObstacle(name=f'obstacle_{x}_{y}', content_dict=sphere_obst_dict)
    #         env.add_obstacle(sphere_obst)

    # adding a table from which to grab the goal
    table_height = 1
    table_length = 2
    table_width = 1
    table_size = [table_width, table_length, table_height]
    table_position = [[1, 1, 0]]
    env.add_shapes(shape_type="GEOM_BOX", dim=table_size, mass=100000000, poses_2d=table_position)
    # adding the box that the robot arm has to pick up
    box_dim = 0.1
    box_size = [box_dim for n in range(3)]
    env.add_shapes(shape_type="GEOM_BOX", dim=box_size, mass=10, poses_2d=table_position,
                   place_height=table_height + 0.5 * box_dim)


def add_obstacles_3D(env, location=None, seed=63, number=8, scale=1.0):
    """
    Add obstacles in the air for the arm to avoid.
    """
    from MotionPlanningEnv.sphereObstacle import SphereObstacle

    x, y, z = 0, 0, 0
    if location[0]:
        x, y, z = location

    random.seed(seed)
    for i in range(number):
        random_x = float(x + random.uniform(-1, 1) * scale)
        random_y = float(1 + random.uniform(0, 1) * scale)
        random_z = float(y + random.uniform(-1, 1) * scale)

        sphere_obst_dict = {
            "type": "sphere",
            'movable': False,
            "geometry": {"position": [random_x, random_z, random_y], "radius": 0.1},
        }
        sphere_obst = SphereObstacle(name=f'obstacle_{i}', content_dict=sphere_obst_dict)
        env.add_obstacle(sphere_obst)


def add_goal(env, table_position=[-5, 5, 0], albert_radius=1.0):
    """
    Add the goal to the environment!
    TODO: extend for picking up the block. (Now it is just a position)
    """
    from MotionPlanningGoal.staticSubGoal import StaticSubGoal

    goal_dict = {
        "weight": 1.0, "is_primary_goal": True, 'indices': [0, 1, 2], 'parent_link': 0, 'child_link': 3,
        'desired_position': table_position, 'epsilon': albert_radius, 'type': "staticSubGoal",
    }

    goal = StaticSubGoal(name="goal1", content_dict=goal_dict)
    env.add_goal(goal)


def add_graph_to_env(graph, place_height=0.005):
    """ Add the graph to the environment as objects. """
    p.removeAllUserDebugItems()
    # Draw edges
    for edge in graph.edges:
        line_color = [0.2, 0.2, 0.2]
        line_width = 1

        u_config = graph.nodes[edge[0]]['config']
        v_config = graph.nodes[edge[1]]['config']

        p.addUserDebugLine(  # Got from pybullet documentation
            lineFromXYZ=[u_config[0], u_config[1], u_config[2] + place_height],
            lineToXYZ=[v_config[0], v_config[1], v_config[2] + place_height],
            lineColorRGB=line_color,
            lineWidth=line_width
        )

def draw_domain(domain, line_color=[0,0,0], line_width=1):
    path = [[domain['xmin'], domain['ymin'], domain['zmin']], [domain['xmin'], domain['ymax'], domain['zmin']],
            [domain['xmax'], domain['ymax'], domain['zmin']], [domain['xmax'], domain['ymin'], domain['zmin']],
            [domain['xmin'], domain['ymin'], domain['zmin']], [domain['xmin'], domain['ymin'], domain['zmax']],
            [domain['xmin'], domain['ymax'], domain['zmax']], [domain['xmin'], domain['ymax'], domain['zmin']],
            [domain['xmin'], domain['ymax'], domain['zmax']], [domain['xmax'], domain['ymax'], domain['zmax']],
            [domain['xmax'], domain['ymax'], domain['zmin']], [domain['xmax'], domain['ymax'], domain['zmax']],
            [domain['xmax'], domain['ymin'], domain['zmax']], [domain['xmax'], domain['ymin'], domain['zmin']],
            [domain['xmax'], domain['ymin'], domain['zmax']], [domain['xmin'], domain['ymin'], domain['zmax']],
            [domain['xmin'], domain['ymin'], domain['zmax']]]
    draw_path(path, line_color=line_color, line_width=line_width, place_height=0)


def draw_node_configs(node_configs, place_height=0.005, point_size=5):
    """ Draw the nodes in the graph. """
    for node_config in node_configs:
        p.addUserDebugPoints(  # Got from pybullet documentation
                pointPositions=[[node_config[0], node_config[1], node_config[2]+place_height]],
                pointColorsRGB=[[1, 0, 1]],
                pointSize=point_size
            )


def draw_path(node_configs, line_color=[0, 1, 0], line_width=3, place_height=0.005):
    for i in range(len(node_configs) - 1):
        p.addUserDebugLine(
            lineFromXYZ=[node_configs[i][0], node_configs[i][1], node_configs[i][2] + place_height],
            lineToXYZ=[node_configs[i + 1][0], node_configs[i + 1][1], node_configs[i + 1][2] + place_height],
            lineColorRGB=line_color,
            lineWidth=line_width
        )


def goal_radius(goal_config, albert_radius=1.0, n=50):
    pi = math.pi
    r = albert_radius
    points = []
    for x in range(1, n + 1):
        point = [goal_config[0] + math.cos(2 * pi / n * x) * r, goal_config[1] + math.sin(2 * pi / n * x) * r,
                 goal_config[2]]
        points.append(point)
    points = np.array(points)
    return points


def transform_camera(dist, yaw, pitch, target, t=1, dt=0.01):
    """"
    Transforms the camera from its old position to a given new one
    within timestamp t and using dt steps
    """
    steps = t / dt
    _, _, _, _, _, _, _, _, old_yaw, old_pitch, old_dist, old_target = p.getDebugVisualizerCamera()

    diff_dist = (dist - old_dist) / steps
    diff_yaw = (yaw - old_yaw) / steps
    diff_pitch = (pitch - old_pitch) / steps
    diff_target = [(target[i] - old_target[i]) / steps for i in range(len(old_target))]

    for i in range(int(steps)):
        old_dist += diff_dist
        old_yaw += diff_yaw
        old_pitch += diff_pitch
        old_target = [old_target[j] + diff_target[j] for j in range(len(old_target))]

        p.resetDebugVisualizerCamera(cameraDistance=old_dist, cameraYaw=old_yaw, cameraPitch=old_pitch,
                                     cameraTargetPosition=old_target)
        time.sleep(dt)


def transform_to_arm(ob, dist=6, pitch=-60):
    """"
    Specific transformation function that transforms the camera
    Takes ob as input
    """

    goal_target = np.pad(ob['robot_0']['joint_state']['position'][0:2], (0, 1))
    orientation = math.degrees(ob['robot_0']['joint_state']['position'][2])

    transform_camera(dist, orientation, pitch, goal_target)
