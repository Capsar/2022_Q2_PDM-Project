import math
import time

import numpy as np
import networkx as nx


def path_length(configs):
    return sum([distance(configs[i] - configs[i + 1]) for i in range(len(configs) - 1)])


def distance(diff_config):
    # return np.sqrt(np.sum(diff_config**2))
    return np.linalg.norm(diff_config)


def steer(from_point, to_point, rrt_radius, step_size=0.1):
    """
    Implement motion primitives here.
    To find closest to the to_point.
    """
    direction_vector = to_point - from_point
    direction_length = distance(direction_vector)
    if direction_length > rrt_radius:
        return (direction_vector / direction_length) * rrt_radius + from_point
    return to_point


class CollisionManager:

    def __init__(self, obstacle_configs, epsilon):
        self.obstacle_configs = obstacle_configs
        self.epsilon = epsilon

    def is_in_obstacle(self, config):
        """
        Check if sampled config is collision free with obstacles.
        linalg.norm is the euclidean distance between two points.
        """
        for obs in self.obstacle_configs:
            diff_config_robot = config - obs[0]
            if distance(diff_config_robot) < obs[1] + self.epsilon:
                return True
        return False

    def is_in_line_of_sight(self, config1, config2):
        """
        Check if there is a line of sight between two points.
        This means that there are no obstacles in between the 2 configs.
        """
        direction = config2 - config1
        for obs in self.obstacle_configs:
            dot_product = np.dot(obs[0] - config1, direction)
            t = dot_product / np.dot(direction, direction)
            if 0 <= t <= 1:
                closest_point = config1 + t * direction
                if distance(closest_point - obs[0]) < obs[1] + self.epsilon:
                    return False
        return True


class RRTStar:

    def __init__(self, start_config, goal_config, collision_manager: CollisionManager, domain, seed=-1):
        self.start_config = start_config
        self.goal_config = goal_config
        self.collision_manager = collision_manager
        self.domain = domain
        self.graph = None
        self.beacon_nodes = None
        self.start_time = None
        if seed != -1:
            np.random.seed(seed)

    def sample_config(self, domain, scale=1):
        for _ in range(100): # Try 100 times to sample a valid config.
            sampled_config = np.random.uniform([domain['xmin'], domain['ymin'], domain['zmin']],
                                               [domain['xmax'], domain['ymax'], domain['zmax']], size=3) * scale
            if not self.collision_manager.is_in_obstacle(sampled_config):
                return sampled_config

    def parent(self, node_id):
        """
        Get the parent of a node.
        """
        return list(self.graph.predecessors(node_id))[0]

    def node_config(self, node_id):
        return self.graph.nodes[node_id]['config']

    def distance_to_start(self, node, start=0):
        return nx.shortest_path_length(self.graph, source=start, target=node, weight='weight')

    def reset(self):
        self.graph = nx.DiGraph()
        self.start_time = time.time()

    def find_nearest_node(self, sampled_config):
        """
        Find the nearest node in graph to sampled point.
        """
        nearest_node = None
        min_distance = float('inf')
        for node in self.graph.nodes:
            if node == -1:
                continue
            node_config = self.node_config(node)
            temp_distance = distance(sampled_config - node_config)  # Might have to be updated to weighted distance
            if temp_distance < min_distance:
                nearest_node = node
                min_distance = temp_distance
        return nearest_node

    def find_near_nodes(self, new_config, rrt_radius):
        """
        Find nodes in graph that are within rrt_radius of new_config.
        """
        near_nodes = []
        for node in self.graph.nodes:
            if distance(new_config - self.node_config(node)) < rrt_radius:
                near_nodes.append(node)
        return near_nodes

    def choose_parent(self, near_nodes, nearest_node, new_config):
        min_node = nearest_node
        min_cost = self.distance_to_start(nearest_node) + distance(new_config - self.node_config(nearest_node))
        # Find the node with the lowest cost.
        for near_node in near_nodes:
            if near_node == nearest_node:
                continue

            near_node_config = self.node_config(near_node)
            if self.collision_manager.is_in_line_of_sight(near_node_config, new_config):
                cost = self.distance_to_start(near_node) + distance(new_config - near_node_config)
                if cost < min_cost:
                    min_cost = cost
                    min_node = near_node
        return min_node

    def rewire(self, near_nodes, new_config, new_node):
        rewired_goal = False
        for near_node in near_nodes:
            near_node_config = self.node_config(near_node)
            if self.collision_manager.is_in_line_of_sight(near_node_config, new_config):
                cost = self.distance_to_start(new_node) + distance(new_config - near_node_config)
                if cost < self.distance_to_start(near_node):
                    parent_node = self.parent(near_node)
                    self.graph.remove_edge(parent_node, near_node)
                    self.graph.add_edge(new_node, near_node, weight=distance(new_config - near_node_config))
                    if near_node == -1:
                        rewired_goal = True
        return rewired_goal

    def step(self, sampled_config, rrt_radius):
        nearest_node = self.find_nearest_node(sampled_config)  # This is a node id (so not a config)
        new_config = steer(self.node_config(nearest_node), sampled_config, rrt_radius)
        if self.collision_manager.is_in_obstacle(new_config):
            return 'collision'

        if not self.collision_manager.is_in_line_of_sight(new_config, self.node_config(nearest_node)):
            return 'collision'

        new_node = len(self.graph.nodes)
        near_nodes = self.find_near_nodes(new_config, rrt_radius)
        min_node = self.choose_parent(near_nodes, nearest_node, new_config)

        self.graph.add_node(new_node, config=new_config)
        self.graph.add_edge(min_node, new_node, weight=distance(new_config - self.node_config(min_node)))

        # Rewire
        rewired_goal = self.rewire(near_nodes, new_config, new_node)

        if rewired_goal:
            return 'goal_found'
        if (new_config == sampled_config).all():
            return 'reached'
        else:
            return 'advanced'

    def run(self, total_duration, rrt_factor):
        self.reset()

        self.graph.add_node(0, config=self.start_config)
        self.graph.add_node(-1, config=self.goal_config)
        self.graph.add_edge(0, -1, weight=101010)
        self.start_time = time.time()

        while time.time() - self.start_time < total_duration:
            n = len(self.graph.nodes)
            rrt_radius = rrt_factor * np.sqrt(np.log(n) / n)
            sampled_config = self.sample_config(self.domain)
            self.step(sampled_config, rrt_radius=rrt_radius)


class RRTStarSmart(RRTStar):

    def __init__(self, robot_pos_config, goal_config, collision_manager, domain, seed):
        super().__init__(robot_pos_config, goal_config, collision_manager, domain, seed)
        self.biased_sampled_configs = []

    def sample_biased_config(self, beacon_nodes, smart_radius):
        if len(beacon_nodes) <= 2:
            return self.sample_config(self.domain)

        random_node = np.random.choice(beacon_nodes[1:-1])
        beacon_config = self.node_config(random_node)
        domain = {
            'xmin': max(beacon_config[0] - smart_radius, self.domain['xmin']),
            'xmax': min(beacon_config[0] + smart_radius, self.domain['xmax']),
            'ymin': max(beacon_config[1] - smart_radius, self.domain['ymin']),
            'ymax': min(beacon_config[1] + smart_radius, self.domain['ymax']),
            'zmin': max(beacon_config[2] - smart_radius, self.domain['zmin']),
            'zmax': min(beacon_config[2] + smart_radius, self.domain['zmax'])
        }
        return self.sample_config(domain)

    def optimize_path(self):
        shortest_path = nx.shortest_path(self.graph, 0, -1, weight='weight')
        beacon_nodes = shortest_path.copy()
        while True and len(beacon_nodes) > 2:
            found = False
            for i in range(len(beacon_nodes) - 2):
                node_0, node_1, node_2 = beacon_nodes[i], beacon_nodes[i + 1], beacon_nodes[i + 2]
                config_0, config_1, config_2 = self.node_config(node_0), self.node_config(node_1), self.node_config(node_2)
                if self.collision_manager.is_in_line_of_sight(config_0, config_2):
                    beacon_nodes.pop(i + 1)
                    found = True
                    break

            if not found:
                break
        return beacon_nodes

    def optimize_graph(self):
        shortest_path = nx.shortest_path(self.graph, 0, -1, weight='weight')
        beacon_nodes = shortest_path.copy()
        while True and len(beacon_nodes) > 2:
            found = False
            for i in range(len(beacon_nodes) - 2):
                node_0, node_1, node_2 = beacon_nodes[i], beacon_nodes[i + 1], beacon_nodes[i + 2]
                config_0, config_1, config_2 = self.node_config(node_0), self.node_config(node_1), self.node_config(node_2)
                if self.collision_manager.is_in_line_of_sight(config_0, config_2):
                    self.graph.remove_edge(node_1, node_2)
                    self.graph.add_edge(node_0, node_2, weight=distance(config_0 - config_2))
                    beacon_nodes.pop(i + 1)
                    found = True
                    break

            if not found:
                break
        return beacon_nodes

    def smart_run(self, total_duration, rrt_factor, smart_sample_ratio, smart_radius, smart_switch_time=0):
        self.reset()

        self.graph.add_node(0, config=self.start_config)
        self.graph.add_node(-1, config=self.goal_config)
        self.graph.add_edge(0, -1, weight=101010)
        self.start_time = time.time()

        start_biased_sampling = False
        beacon_nodes, direct_cost = None, 101010
        normal_sample_counter, smart_sample_counter = 1, 0
        while time.time() - self.start_time < total_duration:
            n = len(self.graph.nodes)
            rrt_radius = rrt_factor * np.sqrt(np.log(n) / n)
            if start_biased_sampling and time.time() - self.start_time > smart_switch_time \
                    and float(smart_sample_counter / normal_sample_counter) < smart_sample_ratio:
                sampled_config = self.sample_biased_config(beacon_nodes, smart_radius)
                self.biased_sampled_configs.append(sampled_config)
                smart_sample_counter += 1
            else:
                sampled_config = self.sample_config(self.domain)
                if start_biased_sampling:
                    normal_sample_counter += 1

            status = self.step(sampled_config, rrt_radius=rrt_radius)
            if status == 'collision':
                continue

            if status == 'goal_found':
                start_biased_sampling = True

            if start_biased_sampling:
                temp_beacon_nodes = self.optimize_graph()
                direct_cost_new = path_length([self.node_config(node) for node in temp_beacon_nodes])
                if direct_cost_new <= direct_cost:
                    beacon_nodes = temp_beacon_nodes
                    direct_cost = direct_cost_new
                else:
                    print('new direct cost not lower! {} vs {}'.format(direct_cost_new, direct_cost))