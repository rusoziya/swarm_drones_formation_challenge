#!/usr/bin/env python3
import sys
import math
import time
import random
import argparse
import rclpy
import yaml
import numpy as np
import heapq
import threading
from rclpy.node import Node
from threading import Lock
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from typing import List, Tuple, Dict, Set

# Additional imports needed for drone interface functionality
from rclpy.executors import MultiThreadedExecutor
from as2_msgs.msg import YawMode, BehaviorStatus
from as2_python_api.drone_interface import DroneInterface
from as2_python_api.behavior_actions.behavior_handler import BehaviorHandler
from std_msgs.msg import ColorRGBA

#############################
# Configuration Constants
#############################
ALTITUDE = 2.5  # Default flight altitude (m)
DRONE_SPEED = 1.0  # Default drone speed (m/s)
FORMATION_SPACING = 0.5  # Spacing between drones in formation (m)
SAFETY_RADIUS = 0.3  # Safety distance from obstacles (0.5m diameter + 0.1m buffer)
OBSTACLE_PREDICTION_HORIZON = 1.0  # How far ahead to predict obstacle trajectories (seconds)
REPLAN_INTERVAL = 0.3  # How often to replan paths (seconds)
GRID_RESOLUTION = 0.2  # Grid cell size for A* planning (m)
MAX_PLANNING_ITERATIONS = 1000  # Maximum A* iterations before giving up
LEADER_DRONE = "drone2"  # Designate the leader drone

#############################
# Formation Functions
#############################
def formation_v_5():
    """
    V formation for 5 drones with specified spacing.
    Returns offsets relative to the leader drone position.
    Leader drone (drone2) is placed at the center of the formation.
    """
    d = FORMATION_SPACING
    return [
        (-2*d, -2*d),        # drone0 at far left back (more pronounced V)
        (-d, -d),          # drone1 at left back
        (0, 0),            # drone2 (leader) at center
        (d, -d),           # drone3 at right back
        (2*d, -2*d)          # drone4 at far right back (more pronounced V)
    ]

#############################
# Dynamic Obstacle Prediction
#############################
class DynamicObstacle:
    """
    Represents a dynamic obstacle with position, velocity, and prediction capabilities.
    """
    def __init__(self, id, position, velocity, radius=0.25):
        self.id = id
        self.position = np.array(position)  # [x, y, z]
        self.velocity = np.array(velocity)  # [vx, vy, vz]
        self.radius = radius
        self.last_update_time = time.time()
        
    def update(self, position, timestamp=None):
        """Update obstacle position and estimate velocity"""
        if timestamp is None:
            timestamp = time.time()
            
        dt = timestamp - self.last_update_time
        if dt > 0:
            # Calculate velocity based on position change
            new_position = np.array(position)
            self.velocity = (new_position - self.position) / dt
            self.position = new_position
            self.last_update_time = timestamp
    
    def predict_position(self, future_time):
        """Predict future position with reflections at stage boundaries"""
        dt = future_time - self.last_update_time
        
        # Start with simple linear prediction
        predicted_pos = self.position + self.velocity * dt
        
        # TODO: Implement bounce prediction if needed based on stage bounds
        
        return predicted_pos
    
    def predicted_path(self, time_horizon, step_count=10):
        """Return a series of predicted positions over the time horizon"""
        path = []
        for i in range(step_count):
            t = self.last_update_time + (i / (step_count-1)) * time_horizon
            path.append(self.predict_position(t))
        return path

#############################
# A* Path Planning
#############################
class AStarNode:
    """Node representation for A* path planning"""
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic (estimated cost to goal)
        self.f = 0  # Total cost (g + h)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __lt__(self, other):
        # For priority queue
        return self.f < other.f or (self.f == other.f and self.h < other.h)

class AStarPlanner:
    """A* path planner with dynamic obstacle avoidance"""
    def __init__(self, grid_resolution=GRID_RESOLUTION):
        self.grid_resolution = grid_resolution
        self.directions = [
            (1, 0), (0, 1), (-1, 0), (0, -1),  # 4-connected
            (1, 1), (-1, 1), (-1, -1), (1, -1)  # Diagonals
        ]
        # Default bounds (will be overridden by actual stage bounds)
        self.bounds = [-10.0, 10.0, -10.0, 10.0]
    
    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        return (round(x / self.grid_resolution), round(y / self.grid_resolution))
    
    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates"""
        return (grid_x * self.grid_resolution, grid_y * self.grid_resolution)
    
    def is_collision(self, x, y, obstacles, radius=SAFETY_RADIUS):
        """Check if position (x,y) collides with any obstacle, including predicted positions"""
        for obs in obstacles:
            if isinstance(obs, DynamicObstacle):
                # Check current position
                obs_pos = obs.position[:2]  # Use only x,y
                distance = math.hypot(x - obs_pos[0], y - obs_pos[1])
                if distance < (radius + obs.radius):
                    return True
                
                # Also check predicted position over prediction horizon
                predicted_positions = obs.predicted_path(OBSTACLE_PREDICTION_HORIZON, step_count=5)
                for pred_pos in predicted_positions:
                    pred_x, pred_y = pred_pos[:2]
                    distance = math.hypot(x - pred_x, y - pred_y)
                    if distance < (radius + obs.radius):
                        return True
            else:  # Simple position tuple
                obs_pos = obs[:2]
                distance = math.hypot(x - obs_pos[0], y - obs_pos[1])
                if distance < radius:
                    return True
        return False

    def heuristic(self, node, goal):
        """Diagonal distance heuristic"""
        dx = abs(node.x - goal.x)
        dy = abs(node.y - goal.y)
        # Diagonal shortcut cost (assuming cost 1.4 for diagonal, 1 for orthogonal)
        return 1.0 * (dx + dy) + (1.414 - 2.0) * min(dx, dy)
    
    def plan_path(self, start, goal, obstacles, max_iterations=MAX_PLANNING_ITERATIONS, bounds=None):
        """
        Plan a path from start to goal avoiding obstacles
        Args:
            start: (x, y) start position in world coordinates
            goal: (x, y) goal position in world coordinates
            obstacles: list of obstacle objects or positions
            max_iterations: maximum number of A* iterations
            bounds: [min_x, max_x, min_y, max_y] planning bounds
        Returns:
            List of waypoints [(x, y), ...] in world coordinates
        """
        try:
            # Update bounds if provided
            if bounds:
                self.bounds = bounds
                print(f"DEBUG: A* planner using bounds: {self.bounds}")
            
            print(f"DEBUG: A* planning from {start} to {goal} with {len(obstacles)} obstacles")
            
            # Check if start or goal is out of bounds
            if not self._is_within_bounds(start[0], start[1]) or not self._is_within_bounds(goal[0], goal[1]):
                print(f"DEBUG: Warning: Start {start} or goal {goal} is outside planning bounds {self.bounds}")
                # Adjust to be within bounds if needed
                start_adj = [
                    max(min(start[0], self.bounds[1]), self.bounds[0]),
                    max(min(start[1], self.bounds[3]), self.bounds[2])
                ]
                goal_adj = [
                    max(min(goal[0], self.bounds[1]), self.bounds[0]),
                    max(min(goal[1], self.bounds[3]), self.bounds[2])
                ]
                if start != start_adj or goal != goal_adj:
                    print(f"DEBUG: Adjusted start to {start_adj} and goal to {goal_adj}")
                    start = start_adj
                    goal = goal_adj
            
            # First try direct path - if it's obstacle-free, just use that
            direct_path_viable = True
            test_points = 20
            for t in np.linspace(0, 1, test_points):
                x = start[0] + t * (goal[0] - start[0])
                y = start[1] + t * (goal[1] - start[1])
                if self.is_collision(x, y, obstacles):
                    direct_path_viable = False
                    print(f"DEBUG: Collision detected at {t*100:.0f}% along direct path")
                    break
            
            if direct_path_viable:
                print("DEBUG: Direct path to goal is viable! Returning simple path.")
                # Return a simple direct path
                return [start, goal]
            
            # Otherwise, try A* planning
            print("DEBUG: Direct path has collisions, performing A* planning")
            
            # Convert to grid coordinates
            start_grid = self.world_to_grid(start[0], start[1])
            goal_grid = self.world_to_grid(goal[0], goal[1])
            
            print(f"DEBUG: Grid coordinates - start: {start_grid}, goal: {goal_grid}")
            
            # Create start and goal nodes
            start_node = AStarNode(start_grid[0], start_grid[1])
            start_node.g = start_node.h = start_node.f = 0
            goal_node = AStarNode(goal_grid[0], goal_grid[1])
            goal_node.g = goal_node.h = goal_node.f = 0
            
            # Initialize open and closed sets
            open_set = []
            closed_set = set()
            
            # Add start node to open set
            heapq.heappush(open_set, start_node)
            
            # Main A* loop
            iterations = 0
            closest_node = start_node
            min_distance_to_goal = float('inf')
            
            while open_set and iterations < max_iterations:
                iterations += 1
                
                # Get node with lowest f score
                current = heapq.heappop(open_set)
                
                # Check if goal reached
                if current.x == goal_node.x and current.y == goal_node.y:
                    path = self._reconstruct_path(current)
                    print(f"DEBUG: A* found path with {len(path)} points in {iterations} iterations")
                    return path
                
                # Track closest node to goal for fallback path
                dist_to_goal = abs(current.x - goal_node.x) + abs(current.y - goal_node.y)
                if dist_to_goal < min_distance_to_goal:
                    min_distance_to_goal = dist_to_goal
                    closest_node = current
                
                # Add to closed set
                closed_set.add((current.x, current.y))
                
                # Generate successors
                for dx, dy in self.directions:
                    successor = AStarNode(current.x + dx, current.y + dy, current)
                    
                    # Skip if already evaluated
                    if (successor.x, successor.y) in closed_set:
                        continue
                    
                    # Check collision in world coordinates
                    world_x, world_y = self.grid_to_world(successor.x, successor.y)
                    if self.is_collision(world_x, world_y, obstacles):
                        continue
                    
                    # Calculate costs
                    if abs(dx) + abs(dy) == 1:  # Cardinal directions
                        successor.g = current.g + 1.0
                    else:  # Diagonal movement
                        successor.g = current.g + 1.414
                    
                    successor.h = self.heuristic(successor, goal_node)
                    successor.f = successor.g + successor.h
                    
                    # Check if this node is already in open set with a better path
                    in_open_set = False
                    for i, open_node in enumerate(open_set):
                        if (open_node.x, open_node.y) == (successor.x, successor.y):
                            in_open_set = True
                            # If we found a better path, update it
                            if successor.g < open_node.g:
                                open_set[i] = successor
                                heapq.heapify(open_set)
                            break
                    
                    # Add to open set if not there
                    if not in_open_set:
                        heapq.heappush(open_set, successor)
            
            # If no path found but we have a closest node, return path to that
            if closest_node != start_node:
                partial_path = self._reconstruct_path(closest_node)
                print(f"DEBUG: No complete path found. Using partial path with {len(partial_path)} points to closest possible position.")
                
                # If we have a partial path, let's try to extend it with a direct line to the goal
                if partial_path:
                    # Get the end of the partial path
                    last_point = partial_path[-1]
                    # Add a direct connection to the goal (even if there might be obstacles)
                    partial_path.append((goal[0], goal[1]))
                    print(f"DEBUG: Extended partial path with direct line to goal from {last_point}")
                    return partial_path
            
            # If we still have no path, just return a direct path
            print(f"DEBUG: A* failed after {iterations} iterations. Returning direct path as fallback.")
            return [start, goal]
        
        except Exception as e:
            print(f"ERROR in A* planner: {str(e)}")
            # Return a simple direct path
            return [start, goal]

    def _reconstruct_path(self, node):
        """Reconstruct path from A* result and convert to world coordinates"""
        path = []
        current = node
        
        while current:
            world_x, world_y = self.grid_to_world(current.x, current.y)
            path.append((world_x, world_y))
            current = current.parent
            
        # Reverse to get path from start to goal
        return path[::-1]

    def add_intermediate_waypoints(self, path, interval=0.5):
        """Add intermediate waypoints to the path for smoother movement"""
        if not path or len(path) < 2:
            return path
            
        detailed_path = []
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            detailed_path.append(p1)
            
            # Calculate distance
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            
            # Add intermediate points if distance is large enough
            if dist > interval:
                num_points = max(1, int(dist / interval))
                for j in range(1, num_points):
                    ratio = j / num_points
                    x = p1[0] + (p2[0] - p1[0]) * ratio
                    y = p1[1] + (p2[1] - p1[1]) * ratio
                    detailed_path.append((x, y))
                    
        # Add last point
        detailed_path.append(path[-1])
        return detailed_path

    def _is_within_bounds(self, x, y):
        """Check if position is within planning bounds"""
        return (self.bounds[0] <= x <= self.bounds[1] and 
                self.bounds[2] <= y <= self.bounds[3])

#############################
# Drone Interface
#############################
class FormationDrone(DroneInterface):
    """
    A modified drone interface for formation flight with dynamic obstacle avoidance.
    """
    def __init__(self, namespace: str, verbose: bool = False, use_sim_time: bool = False):
        super().__init__(namespace, verbose=verbose, use_sim_time=use_sim_time)
        self.verbose = verbose
        self._speed = DRONE_SPEED
        self._yaw_mode = YawMode.PATH_FACING
        self._yaw_angle = None
        self._frame_id = "earth"
        self.current_behavior: BehaviorHandler = None
        self.led_pub = self.create_publisher(ColorRGBA, f"/{namespace}/leds/control", 10)

    def change_led_colour(self, colour):
        msg = ColorRGBA()
        msg.r = colour[0] / 255.0
        msg.g = colour[1] / 255.0
        msg.b = colour[2] / 255.0
        self.led_pub.publish(msg)

    def change_leds_random_colour(self):
        self.change_led_colour([random.randint(0, 255) for _ in range(3)])

    def do_behavior(self, beh, *args) -> None:
        """Start behavior and store it to later check completion."""
        if self.verbose:
            print(f"[{self.namespace}] do_behavior: {beh} with args {args}")
        self.current_behavior = getattr(self, beh)
        self.current_behavior(*args)

    def go_to_position(self, x, y, z, speed=DRONE_SPEED) -> None:
        """Command the drone to move to a specific position."""
        if self.verbose:
            print(f"[{self.namespace}] go_to_position called with x={x}, y={y}, z={z}, speed={speed}")
        self.do_behavior("go_to",
                         x, y, z,
                         speed,
                         self._yaw_mode,
                         self._yaw_angle,
                         self._frame_id,
                         False)
        self.change_leds_random_colour()

    def goal_reached(self) -> bool:
        """Check if the current behavior has finished (IDLE)."""
        if not self.current_behavior:
            return False
        return self.current_behavior.status == BehaviorStatus.IDLE

class FormationSwarmConductor:
    """
    Coordinates a swarm of drones in formation flight.
    """
    def __init__(self, drones_ns: list, verbose: bool = False, use_sim_time: bool = False):
        self.drones = []
        for ns in drones_ns:
            self.drones.append(FormationDrone(ns, verbose, use_sim_time))
        self.verbose = verbose

    def shutdown(self):
        for d in self.drones:
            d.shutdown()

    def wait_all(self):
        # Poll the goal status to check if all drones have finished their behavior.
        all_done = False
        while not all_done:
            time.sleep(0.01)
            all_done = all(d.goal_reached() for d in self.drones)
        if self.verbose:
            print("All drones reached goal.")
    
    def get_ready(self) -> bool:
        """Arm and set offboard mode for all drones."""
        success = True
        for d in self.drones:
            success_arm = d.arm()
            success_offboard = d.offboard()
            success = success and success_arm and success_offboard
        return success

    def takeoff(self, height=ALTITUDE, speed=0.7):
        for d in self.drones:
            d.do_behavior("takeoff", height, speed, False)
            d.change_led_colour((0, 255, 0))
        self.wait_all()

    def land(self, speed=0.4):
        for d in self.drones:
            d.do_behavior("land", speed, False)
        self.wait_all()

    def move_swarm(self, positions, altitude=ALTITUDE, speed=DRONE_SPEED):
        """
        positions: list of (x, y) tuples for each drone.
        Commands each drone to move to its corresponding (x, y, altitude) point.
        """
        for d, (px, py) in zip(self.drones, positions):
            if self.verbose:
                print(f"Moving {d.namespace} to ({px:.2f}, {py:.2f}, {altitude})")
            d.go_to_position(px, py, altitude, speed=speed)
        if self.verbose:
            print("Waiting for all drones to reach goal...")
        self.wait_all()

    def move_swarm_with_altitudes(self, positions, altitudes, speed=DRONE_SPEED):
        """
        positions: list of (x, y) tuples for each drone.
        altitudes: list of altitudes for each drone.
        Commands each drone to move to its corresponding (x, y, altitude) point.
        """
        for d, (px, py), alt in zip(self.drones, positions, altitudes):
            if self.verbose:
                print(f"Moving {d.namespace} to ({px:.2f}, {py:.2f}, {alt})")
            d.go_to_position(px, py, alt, speed=speed)
        if self.verbose:
            print("Waiting for all drones to reach goal...")
        self.wait_all()

#############################
# Formation Controller
#############################
class FormationController:
    """Manages drone formation configuration, deformation, and reformation"""
    
    def __init__(self, base_formation=None):
        if base_formation is None:
            self.base_formation = formation_v_5()
        else:
            self.base_formation = base_formation
        self.deformed_formation = self.base_formation.copy()
        
    def get_formation_positions(self, leader_pos, deformed=False):
        """
        Calculate positions of all drones in formation
        
        Args:
            leader_pos: (x, y) position of leader drone
            deformed: whether to use deformed formation or base formation
        
        Returns:
            List of (x, y) positions for all drones
        """
        formation = self.deformed_formation if deformed else self.base_formation
        return [(leader_pos[0] + dx, leader_pos[1] + dy) for (dx, dy) in formation]
    
    def check_collision_with_obstacles(self, positions, obstacles, safety_radius=SAFETY_RADIUS):
        """
        Check if any drone position collides with obstacles
        
        Returns:
            List of booleans indicating whether each drone collides
        """
        collisions = []
        for pos in positions:
            collides = False
            for obs in obstacles:
                if isinstance(obs, DynamicObstacle):
                    distance = math.hypot(pos[0] - obs.position[0], pos[1] - obs.position[1])
                    if distance < safety_radius + obs.radius:
                        collides = True
                        break
                else:  # Simple position tuple
                    distance = math.hypot(pos[0] - obs[0], pos[1] - obs[1])
                    if distance < safety_radius:
                        collides = True
                        break
            collisions.append(collides)
        return collisions
    
    def deform_formation(self, obstacles, leader_pos, safety_radius=SAFETY_RADIUS):
        """
        Deform formation to avoid obstacles
        
        Args:
            obstacles: List of obstacle objects
            leader_pos: Current leader position (x, y)
            safety_radius: Safety distance from obstacles
            
        Returns:
            List of new (x, y) positions
        """
        # Get base formation positions
        base_positions = self.get_formation_positions(leader_pos, deformed=False)
        
        # Check which drones in base formation would collide
        collisions = self.check_collision_with_obstacles(base_positions, obstacles, safety_radius)
        
        # If no collisions, use base formation
        if not any(collisions):
            self.deformed_formation = self.base_formation.copy()
            return base_positions
        
        print(f"Formation deformation needed: {collisions.count(True)} drones at risk of collision")
        
        # Initialize deformed formation from base formation
        self.deformed_formation = self.base_formation.copy()
        
        # For each drone that would collide, generate a new position
        for i, collides in enumerate(collisions):
            if collides:
                print(f"Finding safe position for drone {i}")
                # Try to find a safe position with increasing search radius
                found_safe_pos = False
                for radius in [0.5, 1.0, 1.5, 2.0]:
                    # Try multiple random directions
                    for _ in range(10):
                        angle = random.uniform(0, 2 * math.pi)
                        offset_x = radius * math.cos(angle)
                        offset_y = radius * math.sin(angle)
                        
                        # Calculate potential new position (relative to leader)
                        new_pos_rel = (offset_x, offset_y)
                        new_pos_abs = (leader_pos[0] + offset_x, leader_pos[1] + offset_y)
                        
                        # Check if this position is safe
                        is_safe = True
                        for obs in obstacles:
                            if isinstance(obs, DynamicObstacle):
                                distance = math.hypot(new_pos_abs[0] - obs.position[0], 
                                                      new_pos_abs[1] - obs.position[1])
                                if distance < safety_radius + obs.radius:
                                    is_safe = False
                                    break
                            else:
                                distance = math.hypot(new_pos_abs[0] - obs[0], 
                                                      new_pos_abs[1] - obs[1])
                                if distance < safety_radius:
                                    is_safe = False
                                    break
                        
                        # Also check for collisions with other drones
                        for j, other_rel_pos in enumerate(self.deformed_formation):
                            if j != i:  # Don't check against self
                                other_abs_pos = (leader_pos[0] + other_rel_pos[0], 
                                                leader_pos[1] + other_rel_pos[1])
                                distance = math.hypot(new_pos_abs[0] - other_abs_pos[0], 
                                                      new_pos_abs[1] - other_abs_pos[1])
                                if distance < 2 * safety_radius:  # Double the safety radius for drone-drone
                                    is_safe = False
                                    break
                        
                        if is_safe:
                            self.deformed_formation[i] = new_pos_rel
                            found_safe_pos = True
                            print(f"Safe position found for drone {i} at offset {new_pos_rel}")
                            break
                    
                    if found_safe_pos:
                        break
                
                if not found_safe_pos:
                    print(f"Warning: Could not find safe position for drone {i}")
        
        # Get positions in the deformed formation
        return self.get_formation_positions(leader_pos, deformed=True)
    
    def try_reform(self, obstacles, leader_pos, safety_radius=SAFETY_RADIUS):
        """
        Check if formation can be reformed and do it if possible
        
        Returns:
            (reformed, positions) - Whether reformation happened and new positions
        """
        # If already in base formation, nothing to do
        if self.deformed_formation == self.base_formation:
            return False, self.get_formation_positions(leader_pos, deformed=True)
        
        # Get base formation positions
        base_positions = self.get_formation_positions(leader_pos, deformed=False)
        
        # Check which drones in base formation would collide
        collisions = self.check_collision_with_obstacles(base_positions, obstacles, safety_radius)
        
        # If any would collide, keep the deformed formation
        if any(collisions):
            print(f"Cannot reform: {collisions.count(True)} drones would collide")
            return False, self.get_formation_positions(leader_pos, deformed=True)
        
        # If none would collide, reform to base formation
        print("Reforming to base formation")
        self.deformed_formation = self.base_formation.copy()
        return True, base_positions

#############################
# Dynamic Obstacles Handler
#############################
class DynamicObstacleHandler:
    """Maintains dynamic obstacle state and performs predictions"""
    
    def __init__(self):
        self.obstacles = {}  # Map from ID to DynamicObstacle objects
        self.last_update_time = time.time()
    
    # NOTE: The update_obstacles method for PoseArray is now obsolete 
    # since we directly update obstacles in the stage4MissionController's obstacle_callback
    
    def predict_obstacle_positions(self, time_horizon=OBSTACLE_PREDICTION_HORIZON):
        """Predict positions of all obstacles over the time horizon"""
        predictions = {}
        current_time = time.time()
        
        for obs_id, obstacle in self.obstacles.items():
            predictions[obs_id] = obstacle.predicted_path(time_horizon)
            
        return predictions
    
    def get_current_obstacles(self):
        """Get all current obstacles"""
        return list(self.obstacles.values())

#############################
# Main Mission Controller
#############################
class Stage4MissionController(Node):
    """
    Controls the mission for 5 drones in formation navigating through dynamic obstacles
    """
    def __init__(self, namespaces, scenario_file=None, verbose=False, use_sim_time=True):
        super().__init__('stage4_mission_controller')
        
        # ROS parameters
        if use_sim_time:
            param = rclpy.parameter.Parameter('use_sim_time', value=True)
            self.set_parameters([param])
        
        self.verbose = verbose
        self.namespaces = namespaces
        self.swarm = FormationSwarmConductor(namespaces, verbose=verbose, use_sim_time=use_sim_time)
        
        # Load mission parameters from scenario file
        self.altitude = ALTITUDE
        self.stage_center = [0.0, 6.0]
        self.start_point_rel = [0.0, -4.0]
        self.end_point_rel = [0.0, 6.0]
        self.num_obstacles = 5
        self.obstacle_velocity = 0.5
        self.obstacle_height = 5.0
        self.obstacle_diameter = 0.5
        self.stage_size = [10.0, 10.0]  # Default stage size
        
        if scenario_file:
            self.load_scenario(scenario_file)
        
        # Convert relative positions to absolute
        self.start_point = [
            self.stage_center[0] + self.start_point_rel[0],
            self.stage_center[1] + self.start_point_rel[1]
        ]
        self.end_point = [
            self.stage_center[0] + self.end_point_rel[0],
            self.stage_center[1] + self.end_point_rel[1]
        ]
        
        # Stage bounds with +2 safety margin
        min_x = self.stage_center[0] - self.stage_size[0]/2.0 - 2.0
        max_x = self.stage_center[0] + self.stage_size[0]/2.0 + 2.0
        min_y = self.stage_center[1] - self.stage_size[1]/2.0 - 2.0
        max_y = self.stage_center[1] + self.stage_size[1]/2.0 + 2.0
        
        self.stage_bounds = [
            min_x, max_x,
            min_y, max_y
        ]
        
        print(f"Planning bounds set to: {self.stage_bounds}")
        
        # Initialize formations
        self.formation_controller = FormationController(formation_v_5())
        
        # Initialize path planning
        self.path_planner = AStarPlanner(grid_resolution=GRID_RESOLUTION)
        self.current_path = []
        self.current_path_index = 0
        
        # Initialize dynamic obstacles tracking
        self.obstacle_handler = DynamicObstacleHandler()
        self.obstacle_lock = Lock()
        
        # Subscribe to dynamic obstacle positions - corrected based on runner.py
        print(f"Subscribing to obstacle topic: '/dynamic_obstacles/locations' using PoseStamped")
        self.obstacle_subscription = self.create_subscription(
            PoseStamped,  # Changed from PoseArray to PoseStamped
            '/dynamic_obstacles/locations',
            self.obstacle_callback,
            10
        )
        
        # Path planning timer
        self.planning_timer = self.create_timer(
            REPLAN_INTERVAL,
            self.replan_path_callback
        )
        
        # Add timer to print obstacle locations every 5 seconds
        self.obstacle_print_timer = self.create_timer(
            5.0,
            self.print_obstacles_callback
        )
        
        # Leader position (used as reference for formation)
        self.leader_position = self.start_point.copy()
        
        # Mission status
        self.mission_complete = False
        
    def load_scenario(self, file_path):
        """Load mission parameters from scenario file"""
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Extract relevant data
            if 'stage4' in data:
                stage4 = data['stage4']
                if 'stage_center' in stage4:
                    self.stage_center = stage4['stage_center']
                if 'start_point' in stage4:
                    self.start_point_rel = stage4['start_point']
                if 'end_point' in stage4:
                    self.end_point_rel = stage4['end_point']
                if 'num_obstacles' in stage4:
                    self.num_obstacles = stage4['num_obstacles']
                if 'obstacle_velocity' in stage4:
                    self.obstacle_velocity = stage4['obstacle_velocity']
                if 'obstacle_height' in stage4:
                    self.obstacle_height = stage4['obstacle_height']
                if 'obstacle_diameter' in stage4:
                    self.obstacle_diameter = stage4['obstacle_diameter']
            
            # Get stage size from yaml
            if 'stage_size' in data:
                self.stage_size = data['stage_size']
            else:
                # Default stage size if not specified
                self.stage_size = [10.0, 10.0]  
            
            print(f"Loaded stage with center {self.stage_center} and size {self.stage_size}")
            
            if self.verbose:
                print(f"Loaded scenario from {file_path}")
                print(f"Stage center: {self.stage_center}")
                print(f"Start point (rel): {self.start_point_rel}")
                print(f"End point (rel): {self.end_point_rel}")
                print(f"Num obstacles: {self.num_obstacles}")
                
        except Exception as e:
            self.get_logger().error(f"Failed to load scenario: {e}")
            
    def obstacle_callback(self, msg):
        """Callback for receiving dynamic obstacle positions (as PoseStamped)"""
        # Extract obstacle ID from frame_id
        obstacle_id = msg.header.frame_id
        
        with self.obstacle_lock:
            # Create a single-item dictionary to update obstacles with PoseStamped
            obstacle_position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
            current_time = time.time()
            
            # Update existing obstacle or create new one
            if obstacle_id in self.obstacle_handler.obstacles:
                self.obstacle_handler.obstacles[obstacle_id].update(obstacle_position, current_time)
            else:
                # For new obstacles, assume zero initial velocity
                self.obstacle_handler.obstacles[obstacle_id] = DynamicObstacle(
                    obstacle_id, obstacle_position, [0, 0, 0], radius=0.25
                )

    def replan_path_callback(self):
        """Timer callback to replan path based on updated obstacle positions"""
        try:
            print("\n===== DEBUG: Starting replan_path_callback =====")
            
            if self.mission_complete:
                print("DEBUG: Mission complete, skipping replanning")
                return
            
            with self.obstacle_lock:
                # Get current obstacles
                obstacles = self.obstacle_handler.get_current_obstacles()
                
                # Print some diagnostics
                print(f"DEBUG: Leader position: {self.leader_position}")
                print(f"DEBUG: Goal: {self.end_point}")
                print(f"DEBUG: Obstacles count: {len(obstacles)}")
                
                # Plan path from current leader position to goal
                try:
                    print(f"DEBUG: Calling A* planner...")
                    path = self.path_planner.plan_path(
                        self.leader_position,
                        self.end_point,
                        obstacles,
                        bounds=self.stage_bounds
                    )
                    print(f"DEBUG: A* planner returned path of length {len(path)}")
                except Exception as e:
                    print(f"ERROR in A* planner: {str(e)}")
                    path = []
                
                # If we have a path, process it
                if path and len(path) > 0:
                    # Add intermediate waypoints for smoother movement
                    detailed_path = self.path_planner.add_intermediate_waypoints(path)
                    self.current_path = detailed_path
                    self.current_path_index = 0
                    
                    print(f"DEBUG: Generated path with {len(self.current_path)} waypoints")
                    if len(self.current_path) > 0:
                        print(f"DEBUG: First few waypoints: {self.current_path[:min(3, len(self.current_path))]}")
                else:
                    print("DEBUG: A* planner failed to generate a path!")
                    
                    # ALWAYS create a fallback direct path regardless of obstacles
                    print("DEBUG: Creating fallback direct path to goal")
                    direct_path = []
                    start = self.leader_position
                    goal = self.end_point
                    
                    # Add 5 intermediate waypoints
                    for i in range(6):
                        t = i / 5.0
                        x = start[0] + t * (goal[0] - start[0])
                        y = start[1] + t * (goal[1] - start[1])
                        direct_path.append((x, y))
                        
                    self.current_path = direct_path
                    self.current_path_index = 0
                    print(f"DEBUG: Created fallback direct path with {len(direct_path)} waypoints")
                    print(f"DEBUG: Fallback path: {direct_path}")
            
            print("===== DEBUG: Finished replan_path_callback =====\n")
        except Exception as e:
            print(f"ERROR in replan_path_callback: {str(e)}")
            # Always ensure we have some path
            self.current_path = [(self.leader_position[0], self.leader_position[1]), 
                                (self.end_point[0], self.end_point[1])]
            self.current_path_index = 0
            print("Emergency direct path created due to error")
    
    def print_obstacles_callback(self):
        """Print obstacle locations and IDs every 5 seconds - now disabled"""
        # Debug output disabled as requested
        pass
    
    def check_and_adjust_altitudes(self, positions, default_altitude=ALTITUDE, 
                                  separation=0.25, collision_threshold=0.4):
        """
        Adjust altitudes if any drones are too close to each other in xy-plane
        """
        n = len(positions)
        altitudes = [default_altitude for _ in range(n)]
        
        # For each unique pair, check distance
        for i in range(n):
            for j in range(i+1, n):
                d = math.hypot(positions[i][0]-positions[j][0], positions[i][1]-positions[j][1])
                if d < collision_threshold:
                    if self.verbose:
                        print(f"Collision risk detected between drones {i} and {j} at distance {d:.2f}. Adjusting altitudes.")
                    # Simple strategy: one drone gets raised, one lowered.
                    altitudes[i] = default_altitude + separation
                    altitudes[j] = default_altitude - separation
        return altitudes
    
    def execute_mission(self):
        """Execute the multi-drone formation mission using leader-follower approach"""
        print("\n\n========== STARTING MISSION EXECUTION ==========")
        print(f"Start point: {self.start_point}")
        print(f"End point: {self.end_point}")
        print(f"Using {len(self.namespaces)} drones: {self.namespaces}")
        print(f"Leader drone: {LEADER_DRONE}")
        print("=================================================\n")
        
        if self.verbose:
            print("Starting Stage 4 mission with leader-follower formation")
        
        # Find the leader drone's index
        leader_index = -1
        for i, ns in enumerate(self.namespaces):
            if ns == LEADER_DRONE:
                leader_index = i
                break
        
        if leader_index == -1:
            print(f"Leader drone '{LEADER_DRONE}' not found in namespaces. Using drone0 as leader.")
            leader_index = 0
        
        print(f"Leader drone index is {leader_index}")
        
        # Arm and take off
        print("Arming drones and preparing for takeoff...")
        if not self.swarm.get_ready():
            print("Failed to arm/offboard!")
            return False
        
        print(f"Taking off to altitude: {self.altitude}m")
        self.swarm.takeoff(height=self.altitude, speed=0.7)
        print("Takeoff complete!")
        
        # Initial formation setup at start position
        print("Moving to initial formation at start position...")
        formation_positions = self.formation_controller.get_formation_positions(self.start_point)
        print(f"Initial formation positions: {formation_positions}")
        self.swarm.move_swarm(formation_positions, altitude=self.altitude)
        print("Initial formation established")
        
        # Get the leader's position (which should be at the formation position corresponding to leader_index)
        leader_position = formation_positions[leader_index]
        self.leader_position = list(leader_position)
        print(f"Leader drone starting at position: {self.leader_position}")
        
        # Wait a bit for dynamic obstacles to appear and be tracked
        print("Waiting for obstacles to be detected...")
        time.sleep(1.0)
        
        # Forcibly print obstacle information at the start
        with self.obstacle_lock:
            obstacles = self.obstacle_handler.obstacles
            if obstacles:
                print(f"Found {len(obstacles)} obstacles initially")
            else:
                print("No obstacles detected yet. Continuing anyway.")
        
        # Keep running until goal reached
        goal_reached = False
        waypoint_count = 0
        
        print("Beginning path execution loop...")
        loop_count = 0
        
        while rclpy.ok() and not goal_reached:
            loop_count += 1
            print(f"\nDEBUG: Loop iteration {loop_count}")
            
            # Get latest path and obstacles (synchronized access)
            with self.obstacle_lock:
                current_path = self.current_path.copy() if hasattr(self, 'current_path') and self.current_path else []
                current_idx = self.current_path_index if hasattr(self, 'current_path_index') else 0
                obstacles = self.obstacle_handler.get_current_obstacles()
                print(f"DEBUG: Path length={len(current_path)}, current_idx={current_idx}, obstacles={len(obstacles)}")
            
            # Check if we have a valid path
            if not current_path:
                print("DEBUG: No valid path! Waiting...")
                time.sleep(0.1)
                continue
            
            # Get next waypoint for the leader
            if current_idx < len(current_path):
                next_waypoint = current_path[current_idx]
                self.current_path_index = current_idx + 1
                waypoint_count += 1
                print(f"Leader moving to waypoint {waypoint_count}: {next_waypoint}")
            else:
                # End of path reached, check if we're at the goal
                dist_to_goal = math.hypot(
                    self.leader_position[0] - self.end_point[0],
                    self.leader_position[1] - self.end_point[1]
                )
                print(f"DEBUG: End of path reached. Distance to goal: {dist_to_goal:.2f}m")
                if dist_to_goal < 0.5:  # Close enough to goal
                    goal_reached = True
                    print(f"Goal proximity reached: {dist_to_goal:.2f}m from target")
                    break
            
            # Otherwise, use last waypoint
            print("DEBUG: Using last waypoint as next target")
            next_waypoint = current_path[-1]
            
            # Update leader position
            prev_position = self.leader_position.copy()
            self.leader_position = list(next_waypoint)
            movement_dist = math.hypot(
                self.leader_position[0] - prev_position[0],
                self.leader_position[1] - prev_position[1]
            )
            print(f"Leader moved {movement_dist:.2f}m to {self.leader_position}")
            
            # Calculate desired formation positions based on leader's position
            desired_formation = self.formation_controller.get_formation_positions(self.leader_position)
            
            # Adjust leader's position back to its correct formation slot
            # (The leader is not necessarily at slot 0 in the formation)
            leader_offset = desired_formation[leader_index]
            for i in range(len(desired_formation)):
                if i != leader_index:
                    desired_formation[i] = (
                        self.leader_position[0] + (desired_formation[i][0] - leader_offset[0]),
                        self.leader_position[1] + (desired_formation[i][1] - leader_offset[1])
                    )
            
            # Set leader's position to the actual waypoint
            desired_formation[leader_index] = self.leader_position
                
            # Check for collisions and deform formation for follower drones if needed
            deformed_positions = desired_formation.copy()
            
            # Only deform positions for non-leader drones
            for i in range(len(deformed_positions)):
                if i != leader_index:
                    pos = deformed_positions[i]
                    # Check if this position collides with any obstacle
                    collision_detected = False
                    for obs in obstacles:
                        if isinstance(obs, DynamicObstacle):
                            distance = math.hypot(pos[0] - obs.position[0], pos[1] - obs.position[1])
                            if distance < SAFETY_RADIUS + obs.radius:
                                collision_detected = True
                                break
                    
                    # If collision detected, find safe position
                    if collision_detected:
                        print(f"Collision risk for drone {i}, finding safe position...")
                        safe_pos_found = False
                        # Try positions at increasing distances from desired position
                        for radius in [0.5, 1.0, 1.5, 2.0]:
                            if safe_pos_found:
                                break
                            # Try various angles
                            for angle in range(0, 360, 30):
                                angle_rad = math.radians(angle)
                                test_pos = (
                                    pos[0] + radius * math.cos(angle_rad),
                                    pos[1] + radius * math.sin(angle_rad)
                                )
                                
                                # Check if this position is safe
                                safe = True
                                for obs in obstacles:
                                    if isinstance(obs, DynamicObstacle):
                                        distance = math.hypot(test_pos[0] - obs.position[0], 
                                                              test_pos[1] - obs.position[1])
                                        if distance < SAFETY_RADIUS + obs.radius:
                                            safe = False
                                            break
                                
                                if safe:
                                    deformed_positions[i] = test_pos
                                    safe_pos_found = True
                                    print(f"Safe position found for drone {i} at {test_pos}")
                                    break
            
            # Adjust altitudes if needed to avoid inter-drone collisions
            adjusted_altitudes = self.check_and_adjust_altitudes(
                deformed_positions,
                default_altitude=self.altitude
            )
            
            if adjusted_altitudes.count(self.altitude) < len(adjusted_altitudes):
                print(f"Altitude adjustments needed: {adjusted_altitudes}")
            
            # Move swarm to new positions
            print(f"Moving swarm to new positions with {len(deformed_positions)} drones")
            self.swarm.move_swarm_with_altitudes(
                deformed_positions,
                adjusted_altitudes,
                speed=DRONE_SPEED
            )
            
        # Final approach to goal in formation
        if goal_reached:
            print("Goal reached! Moving to final formation.")
            
            # Calculate final formation positions based on goal
            final_formation = self.formation_controller.get_formation_positions(self.end_point)
            
            # Adjust for leader position in formation
            leader_offset = final_formation[leader_index]
            for i in range(len(final_formation)):
                if i != leader_index:
                    final_formation[i] = (
                        self.end_point[0] + (final_formation[i][0] - leader_offset[0]),
                        self.end_point[1] + (final_formation[i][1] - leader_offset[1])
                    )
            
            # Set leader position to exact goal
            final_formation[leader_index] = self.end_point
            
            # Avoid obstacles in final formation
            with self.obstacle_lock:
                obstacles = self.obstacle_handler.get_current_obstacles()
                
            # Deform final formation if needed to avoid obstacles
            for i in range(len(final_formation)):
                if i != leader_index:  # Don't move the leader
                    pos = final_formation[i]
                    # Check if this position collides with any obstacle
                    collision_detected = False
                    for obs in obstacles:
                        if isinstance(obs, DynamicObstacle):
                            distance = math.hypot(pos[0] - obs.position[0], pos[1] - obs.position[1])
                            if distance < SAFETY_RADIUS + obs.radius:
                                collision_detected = True
                                break
                    
                    # If collision detected, find safe position
                    if collision_detected:
                        print(f"Collision risk in final formation for drone {i}, finding safe position...")
                        # Find a safe position using the same logic as during flight
                        # [Insert code similar to the in-flight collision avoidance]
            
            # Adjust altitudes for final formation
            final_altitudes = self.check_and_adjust_altitudes(
                final_formation,
                default_altitude=self.altitude
            )
            
            # Move to final position
            self.swarm.move_swarm_with_altitudes(
                final_formation,
                final_altitudes,
                speed=DRONE_SPEED
            )
        
        # Land drones
        print("Mission complete. Landing drones.")
        self.swarm.land()
        
        self.mission_complete = True
        return True

def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: Multi-drone formation with dynamic obstacle avoidance")
    parser.add_argument('-n', '--namespaces', nargs='+',
                        default=['drone0', 'drone1', 'drone2', 'drone3', 'drone4'],
                        help='List of drone namespaces')
    parser.add_argument('-s', '--scenario', type=str,
                        default='scenarios/scenario1_stage4.yaml',
                        help='Path to scenario configuration file')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Enable verbose output')
    parser.add_argument('-t', '--use_sim_time', action='store_true', default=True,
                        help='Use simulation time')
    args = parser.parse_args()

    # Initialize ROS
    rclpy.init()
    
    # Debug: List all available topics to help identify obstacle topic
    print("\n\nChecking for available ROS topics...")
    # Create a temporary node to query for topics
    topic_check_node = rclpy.create_node("topic_checker")
    
    # Give it a brief moment to discover topics
    time.sleep(2.0)
    
    # Get topics
    topic_names_and_types = topic_check_node.get_topic_names_and_types()
    print("Available ROS topics:")
    for topic_name, topic_types in topic_names_and_types:
        print(f"  * {topic_name} ({', '.join(topic_types)})")
    print("End of topic list\n")
    
    # Clean up the topic check node
    topic_check_node.destroy_node()
    
    # Create mission controller
    mission = Stage4MissionController(
        args.namespaces,
        scenario_file=args.scenario,
        verbose=args.verbose,
        use_sim_time=args.use_sim_time
    )
    
    # Create thread for spinning ROS node
    executor = MultiThreadedExecutor()
    executor.add_node(mission)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    try:
        # Run mission
        mission.execute_mission()
    except KeyboardInterrupt:
        print("Mission aborted by user.")
    except Exception as e:
        print(f"Mission failed with error: {e}")
    finally:
        # Clean up
        mission.swarm.shutdown()
        mission.destroy_node()
        rclpy.shutdown()
        executor_thread.join()

if __name__ == '__main__':
    main() 