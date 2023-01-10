import gym
import networkx as nx
from urdfenvs.robots.albert import AlbertRobot
from urdfenvs.sensors.full_sensor import FullSensor
import numpy as np

from group27.global_path_planning import CollisionManager, RRTStarSmart, RRTStar
from group27.urdf_env_helpers import add_obstacles, add_goal


def test_run(rrt_instance, run_function, params):
    run_function(*params)
    shortest_path = nx.shortest_path(rrt_instance.graph, 0, -1, weight='weight')
    shortest_path_length = nx.shortest_path_length(rrt_instance.graph, 0, -1, weight='weight')
    if shortest_path_length != 101010:
        return len(rrt_instance.graph.nodes), len(shortest_path), shortest_path_length
    else:
        return len(rrt_instance.graph.nodes), -1, -1


def bruteforce_test_rrt_star_smart(robot_pos_config, goal_config, collision_manager, domain, seed):
    n = 3
    total_durations = [3]
    rrt_factors = [20, 30, 40, 50, 60]
    smart_ratio = 0.5
    smart_radius = 0.2

    for total_duration in total_durations:
        for rrt_factor in rrt_factors:
            for _ in range(n):
                rrt_star = RRTStar(robot_pos_config, goal_config, collision_manager, domain, seed=seed)
                sampled_n_nodes, path_n_nodes, path_length = test_run(rrt_star, rrt_star.run, [total_duration, rrt_factor])
                print(f"RRT*, {total_duration}, {rrt_factor}, {sampled_n_nodes}, {path_n_nodes}, {path_length}")

                rrt_star_smart = RRTStarSmart(robot_pos_config, goal_config, collision_manager, domain, seed=seed)
                sampled_n_nodes, path_n_nodes, path_length = test_run(rrt_star_smart, rrt_star_smart.smart_run,
                                                                      [total_duration, rrt_factor, smart_ratio, smart_radius])
                print(f"RRT*-Smart, {total_duration}, {rrt_factor}, {sampled_n_nodes}, {path_n_nodes}, {path_length}")


def main(seed=28, albert_radius=0.3):
    robots = [
        AlbertRobot(mode="vel"),
    ]
    sensor = FullSensor(goal_mask=['position', 'radius'], obstacle_mask=['position', 'radius'])
    robots[0].add_sensor(sensor)

    env = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=False
    )

    # Init environment (robot position, obstacles, goals)
    pos0 = np.array([-10.0, -10.0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0])  # might change later

    env.reset(pos=pos0)
    np.random.seed(seed)

    add_obstacles(env)
    add_goal(env, table_position=[0, 1, 0], albert_radius=albert_radius)

    # Perform 1 random action to get the initial observation (containing obstacles & goal)
    action = np.random.random(env.n())
    ob, _, _, _ = env.step(action)
    print(f"Initial observation : {ob['robot_0']}")  # This now contains the obstacles and goal (env.reset(pos=pos0) did not)
    # Calculate path
    robot_config = [ob['robot_0']['joint_state']['position'], albert_radius]
    goal_config = ob['robot_0']['goals'][0][0]
    obstacle_configs = [obstacle_config for obstacle_config in ob['robot_0']['obstacles']]
    robot_pos_config = np.pad(robot_config[0][0:2], (0, 1))  # (x, y, 0)

    print('obstacle_configs:', obstacle_configs)
    print('goal_config:', goal_config)
    print('robot_config:', robot_config)

    collision_manager = CollisionManager(obstacle_configs, albert_radius)
    domain = {'xmin': -10, 'xmax': 10, 'ymin': -10, 'ymax': 10, 'zmin': 0, 'zmax': 0}
    bruteforce_test_rrt_star_smart(robot_pos_config, goal_config, collision_manager, domain, seed)



if __name__ == "__main__":
    main()