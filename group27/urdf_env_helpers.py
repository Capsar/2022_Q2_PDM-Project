import random
import pybullet as p
import math 
import numpy as np
import time


def add_obstacles(env, seed=28, number=20, scale=10.0):
    from MotionPlanningEnv.sphereObstacle import SphereObstacle
    random.seed(seed)
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
    # adding a table from which to grap the goal
    table_height = 1
    table_length = 2
    table_width = 1
    table_size = [table_width, table_length, table_height]
    table_position = [[1, 1, 0]]
    env.add_shapes(shape_type="GEOM_BOX", dim=table_size, mass=100000000, poses_2d=table_position)
    # adding the box that the robot arm has to pick up
    box_dim = 0.1
    box_size = [box_dim for n in range(3)]
    env.add_shapes(shape_type="GEOM_BOX", dim=box_size, mass=10, poses_2d=table_position, place_height=table_height + 0.5 * box_dim)


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


def add_graph_to_env(graph, shortest_path, point_size=5, place_height=0.2):
    """ Add the graph to the environment as objects. """
    p.removeAllUserDebugItems()
    # Draw edges
    for edge in graph.edges:
        line_color = [0.2, 0.2, 0.2]
        line_width = 1
        if edge[0] in shortest_path and edge[1] in shortest_path:  # If both nodes are in the shortest path make color green.
            line_color = [0, 1, 0]
            line_width = 3

        p.addUserDebugLine(  # Got from pybullet documentation
            lineFromXYZ=[graph.nodes[edge[0]]['config'][0], graph.nodes[edge[0]]['config'][1], place_height],
            lineToXYZ=[graph.nodes[edge[1]]['config'][0], graph.nodes[edge[1]]['config'][1], place_height],
            lineColorRGB=line_color,
            lineWidth=line_width
        )
        

def goal_radius(goal_config,albert_radius=1.0,n=50):
	pi = math.pi
	r = albert_radius
	points =[]
	for x in range (1,n+1):
		point =[goal_config[0]+math.cos(2*pi/n*x)*r,goal_config[1]+math.sin(2*pi/n*x)*r,goal_config[2]]
		points.append(point)
	points = np.array(points)
	return points
	
	
	
	

def draw_path(path, place_height=0.2, line_width=1):
    line_color = [1,0,0]
    for i in range(len(path) -1):
        p.addUserDebugLine(
            lineFromXYZ=[path[i][0], path[i][1], place_height],
            lineToXYZ=[path[i+1][0], path[i+1][1], place_height],
            lineColorRGB=line_color,
            lineWidth=line_width
        )
    # # Draw nodes.
    # for node in graph.nodes:
    #     node_color = [0, 0, 1]
    #     _point_size = point_size
    #     if node <= 0: # If the node is either the start or end node make it green.
    #         node_color = [0, 1, 0]
    #         _point_size = point_size * 2
    #
    #     p.addUserDebugPoints(  # Got from pybullet documentation
    #         pointPositions=[[graph.nodes[node]['config'][0], graph.nodes[node]['config'][1], place_height]],
    #         pointColorsRGB=[node_color],
    #         pointSize=_point_size
    #     )


def transform_camera(dist, yaw, pitch, target, t=3, dt=0.01):
    """"
    Transforms the camera from its old position to a given new one
    within timestamp t and using dt steps
    """
    steps = t/dt
    _, _, _, _, _, _, _,_, old_yaw, old_pitch, old_dist, old_target = p.getDebugVisualizerCamera()

    diff_dist = (dist - old_dist) / steps
    diff_yaw = (yaw - old_yaw) / steps
    diff_pitch = (pitch - old_pitch) / steps
    diff_target = [(target[i] - old_target[i]) / steps for i in range(len(old_target))]

    for i in range(int(steps)):
        old_dist += diff_dist
        old_yaw += diff_yaw
        old_pitch += diff_pitch
        old_target = [old_target[j] + diff_target[j] for j in range(len(old_target))]

        p.resetDebugVisualizerCamera(cameraDistance=old_dist, cameraYaw=old_yaw, cameraPitch=old_pitch, cameraTargetPosition=old_target)
        time.sleep(dt)
