#!/usr/bin/env python3
import sys
import math
import time
import argparse
import rclpy
import yaml
import random
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from threading import Lock

# Use existing modules from your project - only import the class we need
from mission_stage1_new import FormationSwarmConductor

# Formation definitions defined directly in this file
def formation_line_5():
    """
    A horizontal line formation with the leader at center and others on both sides.
    Adjust d to change spacing.
    """
    d = 0.4
    # Leader at (0,0), others to the left and right
    return [(0.0, 0.0), (-2*d, 0.0), (-d, 0.0), (d, 0.0), (2*d, 0.0)]

def formation_column_5():
    """
    A column formation with the leader at the front and the others trailing.
    Adjust d to change spacing.
    """
    d = 0.4
    # Leader at (0,0), followers are directly behind
    return [(0.0, 0.0), (0.0, -d), (0.0, -2*d), (0.0, -3*d), (0.0, -4*d)]

class RRTPlanner:
    """Rapidly-exploring Random Tree planner for dynamic environments"""
    
    def __init__(self, start, goal, obstacles, bounds, step_size=0.3, goal_sample_rate=0.1, max_iterations=1000):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles  # List of obstacle positions and radii [(x, y, radius), ...]
        self.bounds = bounds  # [min_x, max_x, min_y, max_y]
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iterations = max_iterations
        self.min_dist_to_obstacles = 0.7  # Minimum distance to obstacles
        
    def plan(self):
        """Generate an RRT path from start to goal"""
        # Tree structure: (x, y) -> parent node
        tree = {tuple(self.start): None}
        
        for i in range(self.max_iterations):
            # Sample random point (with bias toward goal)
            if random.random() < self.goal_sample_rate:
                random_point = self.goal
            else:
                random_point = np.array([
                    random.uniform(self.bounds[0], self.bounds[1]),
                    random.uniform(self.bounds[2], self.bounds[3])
                ])
            
            # Find nearest node in tree
            nearest_node = self._find_nearest(random_point, tree)
            
            # Steer toward the random point
            new_node = self._steer(nearest_node, random_point)
            
            # Check if the new node is valid
            if self._is_valid(nearest_node, new_node):
                # Add to tree
                tree[tuple(new_node)] = nearest_node
                
                # Check if we can reach the goal from here
                dist_to_goal = np.linalg.norm(new_node - self.goal)
                if dist_to_goal < self.step_size:
                    if self._is_valid(new_node, self.goal):
                        tree[tuple(self.goal)] = tuple(new_node)
                        return self._extract_path(tree)
        
        # If we couldn't reach the goal, find the node closest to the goal
        nodes = np.array(list(tree.keys()))
        if len(nodes) > 0:
            distances = np.linalg.norm(nodes - self.goal, axis=1)
            closest_idx = np.argmin(distances)
            closest_node = tuple(nodes[closest_idx])
            
            return self._extract_path(tree, closest_node)
        
        # Fallback: return a direct path to goal (might be invalid)
        return [self.start, self.goal]
    
    def _find_nearest(self, point, tree):
        """Find the nearest node in the tree to the given point"""
        nodes = np.array(list(tree.keys()))
        distances = np.linalg.norm(nodes - point, axis=1)
        nearest_idx = np.argmin(distances)
        return tuple(nodes[nearest_idx])
    
    def _steer(self, from_node, to_point):
        """Steer from from_node toward to_point with limited step size"""
        from_node = np.array(from_node)
        direction = to_point - from_node
        distance = np.linalg.norm(direction)
        
        if distance < self.step_size:
            return to_point
        else:
            normalized_direction = direction / distance
            return from_node + normalized_direction * self.step_size
    
    def _is_valid(self, from_node, to_node):
        """Check if the path between nodes is collision-free"""
        from_node = np.array(from_node)
        to_node = np.array(to_node)
        
        # Check multiple points along the segment
        for t in np.linspace(0, 1, 10):
            point = from_node + t * (to_node - from_node)
            
            # Check against all obstacles
            for obs in self.obstacles:
                if len(obs) == 3:  # Static obstacle with radius
                    obs_pos = np.array(obs[:2])
                    obs_radius = obs[2]
                    distance = np.linalg.norm(point - obs_pos)
                    if distance < obs_radius + self.min_dist_to_obstacles:
                        return False
                else:  # Dynamic obstacle (just position)
                    obs_pos = np.array(obs)
                    distance = np.linalg.norm(point - obs_pos)
                    if distance < self.min_dist_to_obstacles:
                        return False
                        
        return True
    
    def _extract_path(self, tree, end_node=None):
        """Extract the path from start to end_node (or goal if none provided)"""
        if end_node is None:
            end_node = tuple(self.goal)
            
        path = [end_node]
        current = end_node
        
        while current in tree and tree[current] is not None:
            current = tree[current]
            path.append(current)
            
        return list(reversed(path))

class Stage4MissionConductor(Node):
    def __init__(self, namespaces, yaml_file=None, verbose=False, use_sim_time=True):
        super().__init__('stage4_mission_conductor')
        
        # Set up ROS parameters
        if use_sim_time:
            param = rclpy.parameter.Parameter(
                'use_sim_time', 
                value=True
            )
            self.set_parameters([param])
        
        self.verbose = verbose
        self.swarm = FormationSwarmConductor(namespaces, verbose=verbose, use_sim_time=use_sim_time)
        
        # Load mission parameters from YAML
        self.stage_center = [6.0, -6.0]  # Default
        self.start_point_rel = [0.0, 4.0]
        self.end_point_rel = [0.0, -4.0]
        self.obstacles_rel = []
        self.altitude = 2.5
        
        if yaml_file:
            self.load_yaml_config(yaml_file)
        
        # Initialize dynamic obstacles tracking
        self.dynamic_obstacles = []
        self.obstacle_lock = Lock()
        
        # Subscribe to dynamic obstacle positions
        self.obstacle_subscription = self.create_subscription(
            PoseArray,
            '/dynamic_obstacles/locations',
            self.obstacle_callback,
            10
        )
        
        # Convert relative positions to absolute
        self.start_wp = [self.stage_center[0] + self.start_point_rel[0], 
                         self.stage_center[1] + self.start_point_rel[1]]
        self.end_wp = [self.stage_center[0] + self.end_point_rel[0], 
                       self.stage_center[1] + self.end_point_rel[1]]
        
        # Convert static obstacles to absolute coordinates
        self.static_obstacles = []
        for obs in self.obstacles_rel:
            abs_obs = [self.stage_center[0] + obs[0], self.stage_center[1] + obs[1], 0.5]  # x, y, radius
            self.static_obstacles.append(abs_obs)
        
        # Initial leader position
        self.leader_pos = list(self.start_wp)
        
        # Path planning parameters
        self.path = []
        self.current_path_index = 0
        self.planning_bounds = [
            self.stage_center[0] - 5.0, self.stage_center[0] + 5.0,
            self.stage_center[1] - 5.0, self.stage_center[1] + 5.0
        ]
        
        # Mission status
        self.mission_complete = False
    
    def load_yaml_config(self, yaml_file):
        """Load mission configuration from YAML file"""
        try:
            with open(yaml_file, 'r') as f:
                config = yaml.safe_load(f)
                
            if 'stage_center' in config:
                self.stage_center = config['stage_center']
            if 'start_point_rel' in config:
                self.start_point_rel = config['start_point_rel']
            if 'end_point_rel' in config:
                self.end_point_rel = config['end_point_rel']
            if 'obstacles' in config:
                self.obstacles_rel = config['obstacles']
            if 'altitude' in config:
                self.altitude = config['altitude']
                
            if self.verbose:
                print(f"Loaded configuration from {yaml_file}")
                
        except Exception as e:
            self.get_logger().error(f"Failed to load YAML configuration: {e}")
    
    def obstacle_callback(self, msg):
        """Callback for receiving dynamic obstacle positions"""
        with self.obstacle_lock:
            self.dynamic_obstacles = []
            for pose in msg.poses:
                # Extract position x, y from the pose message
                self.dynamic_obstacles.append([pose.position.x, pose.position.y])
            
            if self.verbose and self.dynamic_obstacles:
                self.get_logger().info(f"Received {len(self.dynamic_obstacles)} dynamic obstacles")
    
    def get_all_obstacles(self):
        """Combine static and dynamic obstacles"""
        with self.obstacle_lock:
            all_obstacles = list(self.static_obstacles)  # Clone the static obstacles
            all_obstacles.extend(self.dynamic_obstacles)  # Add dynamic obstacles
            return all_obstacles
    
    def select_formation(self, leader_pos, obstacles):
        """
        Determine which formation to use based on obstacle proximity:
        - If two or more obstacles are within the sensing radius, use a column formation
        - Otherwise, use the default horizontal line formation
        """
        sensing_radius_front = 1.8
        sensing_radius_back = 1.2
        count = 0
        x, y = leader_pos
        
        for obs in obstacles:
            if len(obs) == 3:  # Static obstacle with radius
                ox, oy = obs[0], obs[1]
            else:  # Dynamic obstacle (just position)
                ox, oy = obs[0], obs[1]
            
            if y - oy > 0.3:  # Obstacle in front
                if math.hypot(x - ox, y - oy) < sensing_radius_front:
                    count += 1
            elif y - oy < 0:  # Obstacle behind
                if math.hypot(x - ox, y - oy) < sensing_radius_back:
                    count += 1
        
        if count >= 2:
            return "column", formation_column_5()
        else:
            return "default", formation_line_5()
    
    def plan_path(self):
        """Plan a path from current leader position to goal using RRT"""
        all_obstacles = self.get_all_obstacles()
        
        planner = RRTPlanner(
            start=self.leader_pos,
            goal=self.end_wp,
            obstacles=all_obstacles,
            bounds=self.planning_bounds,
            step_size=0.3,
            goal_sample_rate=0.1,
            max_iterations=1000
        )
        
        path = planner.plan()
        if path and len(path) > 1:
            # Convert path tuples to lists for easier handling
            self.path = [list(p) for p in path]
            self.current_path_index = 0
            if self.verbose:
                self.get_logger().info(f"Planned path with {len(self.path)} waypoints")
        else:
            self.get_logger().warn("Failed to find a valid path")
            # Fallback to direct path
            self.path = [self.leader_pos, self.end_wp]
            self.current_path_index = 0
    
    def execute_mission(self):
        """Execute the full mission plan"""
        # Initialize swarm
        print("Arming and taking off...")
        if not self.swarm.get_ready():
            print("Failed to arm/offboard!")
            return False
        
        self.swarm.takeoff(height=self.altitude, speed=0.7)
        
        # Move to initial formation around start point
        initial_offsets = formation_line_5()
        init_positions = [(self.leader_pos[0] + dx, self.leader_pos[1] + dy) 
                         for (dx, dy) in initial_offsets]
        self.swarm.move_swarm(init_positions, altitude=self.altitude, speed=1.0)
        
        # Initial path planning
        self.plan_path()
        
        dt = 0.15  # time step [seconds]
        replan_interval = 10  # steps before replanning
        step = 0
        
        while rclpy.ok() and not self.mission_complete:
            # Process ROS callbacks
            rclpy.spin_once(self, timeout_sec=0.01)
            
            print("=" * 60)
            print(f"[Step {step}]")
            
            # Check if mission is complete
            dx_end = self.end_wp[0] - self.leader_pos[0]
            dy_end = self.end_wp[1] - self.leader_pos[1]
            dist_to_end = math.hypot(dx_end, dy_end)
            if dist_to_end < 0.5:
                print("Leader reached final waypoint.")
                self.mission_complete = True
                break
            
            # Replan periodically to account for dynamic obstacles
            if step % replan_interval == 0 or self.current_path_index >= len(self.path) - 1:
                self.plan_path()
            
            # Follow the planned path
            target = self.path[self.current_path_index]
            dx_wp = target[0] - self.leader_pos[0]
            dy_wp = target[1] - self.leader_pos[1]
            dist_to_wp = math.hypot(dx_wp, dy_wp)
            
            # If close to current waypoint, advance to next
            if dist_to_wp < 0.25 and self.current_path_index < len(self.path) - 1:
                self.current_path_index += 1
                target = self.path[self.current_path_index]
                dx_wp = target[0] - self.leader_pos[0]
                dy_wp = target[1] - self.leader_pos[1]
                dist_to_wp = math.hypot(dx_wp, dy_wp)
            
            # Calculate velocity toward waypoint
            max_speed = 0.8
            speed = min(max_speed, dist_to_wp / dt)
            if dist_to_wp > 0.001:
                vx = (dx_wp / dist_to_wp) * speed
                vy = (dy_wp / dist_to_wp) * speed
            else:
                vx, vy = 0.0, 0.0
            
            # Update leader position
            self.leader_pos[0] += vx * dt
            self.leader_pos[1] += vy * dt
            
            # Get all obstacles for formation selection
            all_obstacles = self.get_all_obstacles()
            
            # Select formation based on obstacle density
            formation_name, formation_offsets = self.select_formation(self.leader_pos, all_obstacles)
            
            # Compute absolute target positions for each drone (leader + offsets)
            positions = [(self.leader_pos[0] + off_x, self.leader_pos[1] + off_y) 
                         for (off_x, off_y) in formation_offsets]
            
            # Debug output
            print(f"Leader: ({self.leader_pos[0]:.2f}, {self.leader_pos[1]:.2f}), " +
                  f"Dist to goal: {dist_to_end:.2f}, Formation: {formation_name}")
            print(f" → Velocity: vx = {vx:.3f}, vy = {vy:.3f}, Speed = {speed:.3f}")
            print(f" → Current waypoint: {self.current_path_index+1}/{len(self.path)}")
            print(" → Drone target positions:")
            for i, p in enumerate(positions):
                print(f"    Drone {i}: ({p[0]:.2f}, {p[1]:.2f})")
            
            # Report nearby obstacles
            for i, obs in enumerate(all_obstacles):
                if len(obs) == 3:  # Static obstacle
                    ox, oy = obs[0], obs[1]
                    obs_type = "Static"
                else:  # Dynamic obstacle
                    ox, oy = obs[0], obs[1]
                    obs_type = "Dynamic"
                
                dx_obs = self.leader_pos[0] - ox
                dy_obs = self.leader_pos[1] - oy
                d_obs = math.hypot(dx_obs, dy_obs)
                
                if d_obs < 2.0:
                    print(f"    ⚠️ {obs_type} Obstacle at ({ox:.2f}, {oy:.2f}) is CLOSE: dist = {d_obs:.2f}")
                    if d_obs < 0.5:
                        print("       ‼️  WARNING: Obstacle is VERY CLOSE! Possible collision!")
            
            # Command the swarm to move
            self.swarm.move_swarm(positions, altitude=self.altitude, speed=1.0)
            time.sleep(0.05)
            step += 1
        
        print("Mission complete. Landing swarm.")
        self.swarm.land()
        return True

def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: Centralized Leader-Follower with RRT Planning and Dynamic Obstacle Avoidance")
    parser.add_argument('-n', '--namespaces', nargs='+',
                        default=['drone0', 'drone1', 'drone2', 'drone3', 'drone4'], 
                        help='Drone namespaces')
    parser.add_argument('-y', '--yaml', type=str, default=None,
                        help='Path to YAML configuration file')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('-s', '--use_sim_time', action='store_true', default=True)
    args = parser.parse_args()

    rclpy.init()
    
    mission = Stage4MissionConductor(
        namespaces=args.namespaces,
        yaml_file=args.yaml,
        verbose=args.verbose,
        use_sim_time=args.use_sim_time
    )
    
    success = mission.execute_mission()
    
    # Cleanup
    mission.destroy_node()
    rclpy.shutdown()
    
    if not success:
        sys.exit(1)
    sys.exit(0)

if __name__ == '__main__':
    main() 