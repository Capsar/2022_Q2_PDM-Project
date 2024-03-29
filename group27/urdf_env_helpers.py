import random
import pybullet as p
import math
import numpy as np
import time


def add_sphere(env, pos, radius):
    sphere_obst_dict = {
        "type": "sphere",
        'movable': False,
        "geometry": {"position": pos, "radius": radius},
    }
    from MotionPlanningEnv.sphereObstacle import SphereObstacle
    sphere_obst = SphereObstacle(name=f'obstacle_{pos[0]}_{pos[1]}_{pos[2]}', content_dict=sphere_obst_dict)
    env.add_obstacle(sphere_obst)

def add_wall(env, begin_pos, end_pos, horizontal=True, radius=0.5):
    if horizontal:
        assert begin_pos[1] == end_pos[1]
    else:
        assert begin_pos[0] == end_pos[0]

    if horizontal:
        n_spheres = abs(np.round((end_pos[0] - begin_pos[0]) / (radius * 2)).astype(int))
    else:
        n_spheres = abs(np.round((end_pos[1] - begin_pos[1]) / (radius * 2)).astype(int))

    # add obstacles
    for i in range(n_spheres):
        if horizontal:
            add_sphere(env, [begin_pos[0] + i, begin_pos[1], 0.0], radius)
        else:
            add_sphere(env, [begin_pos[0], begin_pos[1] + i, 0.0], radius)

    # add covering wall
    height = radius
    if horizontal:
        width = n_spheres - (radius * 2)
        length = radius * 2
        pos = [[(begin_pos[0] + end_pos[0]) / 2 - radius, (begin_pos[1] + end_pos[1]) / 2, 0]]
    else:
        width = radius * 2
        length = n_spheres - (radius * 2)
        pos = [[(begin_pos[0] + end_pos[0]) / 2, (begin_pos[1] + end_pos[1]) / 2 - radius, 0]]

    size = [width, length, height]
    env.add_shapes(shape_type="GEOM_BOX", dim=size, mass=0, poses_2d=pos)


def add_obstacles(env, obstacle_setup, number=20, scale=10.0):

    if obstacle_setup not in {1, 2, 3, "random"}:
        print(f"Obstacle setup: {obstacle_setup} is not in the given options [1, 2, 3, 'random']\n"
              f"No obstacles added. ")
        return

    if obstacle_setup == 1:
        add_wall(env, [-10, -6], [-2, -6])
        add_wall(env, [-5, -2.5], [10, -2.5])
        add_wall(env, [-5, -2.5], [-5, 8.5], False)
        add_wall(env, [9, -8.5], [9, -1.5], False)
        add_wall(env, [-2, 5.5], [-2, 12.5], False)
    elif obstacle_setup == 2:
        add_wall(env, [-7, -10], [-7, 8], False)
        add_wall(env, [-7, 7], [5, 7])
        add_wall(env, [-2, 3], [5, 3])
        add_wall(env, [4, -3], [4, 4], False)
        add_wall(env, [-2, 1], [-2, 4], False)
    elif obstacle_setup == 3:
        add_wall(env, [-5, -10], [-5, -7], False)
        add_wall(env, [-5, -5], [-5, 11], False)
        add_wall(env, [-5, -5], [-1, -5])
        add_wall(env, [-2, -7], [-2, -4], False)
        add_wall(env, [-2, -2], [11, -2])
    else:
        for i in range(number):
            random_x = random.uniform(-1, 1) * scale
            random_z = random.uniform(-1, 1) * scale
            add_sphere(env, [random_x, random_z, 0], 0.5)

    # adding a table from which to grab the goal
    # table_height = 1
    # table_length = 2
    # table_width = 1
    # table_size = [table_width, table_length, table_height]
    # table_position = [[1, 1, 0]]
    # env.add_shapes(shape_type="GEOM_BOX", dim=table_size, mass=100000000, poses_2d=table_position)
    # # adding the box that the robot arm has to pick up
    # box_dim = 0.1
    # box_size = [box_dim for n in range(3)]
    # env.add_shapes(shape_type="GEOM_BOX", dim=box_size, mass=10, poses_2d=table_position,
    #                place_height=table_height + 0.5 * box_dim)


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
        time.sleep(0.001)


def draw_domain(domain, line_color=[0, 0, 0], line_width=1, place_height=0):
    path = [[domain['xmin'], domain['ymin'], domain['zmin']], [domain['xmin'], domain['ymax'], domain['zmin']],
            [domain['xmax'], domain['ymax'], domain['zmin']], [domain['xmax'], domain['ymin'], domain['zmin']],
            [domain['xmin'], domain['ymin'], domain['zmin']], [domain['xmin'], domain['ymin'], domain['zmax']],
            [domain['xmin'], domain['ymax'], domain['zmax']], [domain['xmin'], domain['ymax'], domain['zmin']],
            [domain['xmin'], domain['ymax'], domain['zmax']], [domain['xmax'], domain['ymax'], domain['zmax']],
            [domain['xmax'], domain['ymax'], domain['zmin']], [domain['xmax'], domain['ymax'], domain['zmax']],
            [domain['xmax'], domain['ymin'], domain['zmax']], [domain['xmax'], domain['ymin'], domain['zmin']],
            [domain['xmax'], domain['ymin'], domain['zmax']], [domain['xmin'], domain['ymin'], domain['zmax']],
            [domain['xmin'], domain['ymin'], domain['zmax']]]
    draw_path(path, line_color=line_color, line_width=line_width, place_height=place_height)


def draw_node_configs(node_configs, place_height=0.005, point_size=5):
    """ Draw the nodes in the graph. """
    for node_config in node_configs:
        p.addUserDebugPoints(  # Got from pybullet documentation
            pointPositions=[[node_config[0], node_config[1], node_config[2] + place_height]],
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


def transform_to_arm(ob, dist=3.5, orientation_offset=50, pitch=-64):
    """"
    Specific transformation function that transforms the camera
    Takes ob as input
    """

    goal_target = np.pad(ob['robot_0']['joint_state']['position'][0:2], (0, 1))
    orientation = math.degrees(ob['robot_0']['joint_state']['position'][2]) + orientation_offset

    transform_camera(dist, orientation, pitch, goal_target)
