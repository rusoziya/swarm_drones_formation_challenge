#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Multi-Drone Mission Script for Dynamic Obstacle Avoidance (Stage 4)
#
# This script enables 5 drones to navigate from a starting position to a goal
# position while avoiding dynamic obstacles in real-time. The obstacles' positions
# are obtained from the ROS topic /dynamic_obstacles/locations/posestamped.
#
# Each drone uses an A* algorithm to plan and continuously replan paths as obstacles move.
# ------------------------------------------------------------------------------

# ------------------------
# Configuration (Modifiable Parameters)
# ------------------------

# Drone motion parameters
TAKE_OFF_HEIGHT = 1.0      # Height in meters at takeoff 
TAKE_OFF_SPEED = 1.0       # m/s for takeoff 
SLEEP_TIME = 0.05          # Minimal delay between commands (seconds) 
SPEED = 5.0                # m/s during flight 
LAND_SPEED = 1.0          # m/s for landing 

# Altitude adjustment parameters
MIN_ALTITUDE = 0.5         # Minimum permitted flying altitude (m)
MAX_ALTITUDE = 1.5         # Maximum permitted flying altitude (m)

# Obstacle avoidance parameters 
SAFETY_MARGIN = 0.1        # Additional margin (in meters) added around each obstacle 

# Replanning parameters
REPLAN_FREQUENCY = 10.0     # How often to replan (in Hz)
LOOKAHEAD_TIME = 1.0       # How far ahead to predict obstacle positions (in seconds)

# Path planning parameters 
PLANNING_TIME_LIMIT = 1.0        # Time limit for planning each segment (in seconds) 
ASTAR_RESOLUTION = 0.5         # Default grid resolution for A* (in meters)

# RRT planning parameters
STEP_SIZE = 0.5            # Maximum step size for RRT edge expansion (meters)
MAX_NODES = 5000           # Maximum number of nodes in the RRT tree
GOAL_BIAS = 0.1            # Probability of sampling goal directly
GOAL_THRESHOLD = 0.5       # Distance threshold to consider goal reached (meters)

# Parallel planning parameters
CBS_MAX_WORKERS = 18              # Maximum number of worker threads for parallel planning

# CBS optimization parameters
CBS_MAX_ITERATIONS = 100           # Maximum number of CBS iterations before returning best solution
CBS_CONSTRAINT_HORIZON = 20       # Maximum number of timesteps ahead to consider conflicts
CBS_SPATIAL_THRESHOLD = 2.0       # Maximum distance from current position to consider conflicts
CBS_LEARNING_RATE = 0.3           # Learning rate for branch selection based on planning duration

# Number of drones
NUM_DRONES = 5

# ------------------------
# Imports and Setup
# ------------------------
import argparse
import time
import math
import yaml
import logging
import numpy as np
import rclpy
import os
import threading
import json
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import random
import heapq
from copy import deepcopy

from as2_python_api.drone_interface import DroneInterface

# Remove plotting libraries - we don't need visualization
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.patches as patches
# from matplotlib.lines import Line2D

# ------------------------
# Helper Functions
# ------------------------
def load_scenario(scenario_file):
    """Load scenario from a YAML file."""
    with open(scenario_file, 'r') as f:
        scenario = yaml.safe_load(f)
    return scenario

def compute_euclidean_distance(p1, p2):
    """Compute Euclidean distance between two 3D points."""
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def interpolate_angle(a, b, t):
    """
    Interpolate between angles a and b (in radians) by fraction t.
    Handles angle wrap-around.
    """
    diff = (b - a + math.pi) % (2*math.pi) - math.pi
    return a + diff * t

def predict_obstacle_position(obstacle_data, prediction_time):
    """
    Predict the position of an obstacle after a certain time,
    assuming constant velocity.
    """
    pos = obstacle_data["position"].copy()
    vel = obstacle_data.get("velocity", [0, 0, 0])
    
    # Simple linear prediction
    pos[0] += vel[0] * prediction_time
    pos[1] += vel[1] * prediction_time
    
    return pos

def predict_obstacle_trajectory(obstacle_data, lookahead_time, steps=10):
    """
    Generate a trajectory of positions for an obstacle during the entire lookahead time period.
    Returns a list of predicted positions at different time points.
    """
    pos = obstacle_data["position"].copy()
    vel = obstacle_data.get("velocity", [0, 0, 0])
    diameter = obstacle_data.get("diameter", 0.5)
    
    # Generate trajectory points at evenly spaced intervals during the lookahead period
    trajectory = []
    for t in np.linspace(0, lookahead_time, steps):
        pred_pos = [
            pos[0] + vel[0] * t,
            pos[1] + vel[1] * t,
            pos[2] + vel[2] * t
        ]
        trajectory.append(pred_pos)
    
    return trajectory

def is_state_valid(state, dynamic_obstacles, lookahead_time=0.0):
    """
    Checks whether a given (x, y, z) state is free of collisions with dynamic obstacles.
    Each obstacle is treated as a cylinder with the specified diameter and height.
    If lookahead_time > 0, checks against all possible positions during that time period.
    """
    x, y, z = state[0], state[1], state[2]
    
    # Check if within altitude bounds
    if z < MIN_ALTITUDE or z > MAX_ALTITUDE:
        return False
    
    # If no obstacles data yet, assume it's valid
    if not dynamic_obstacles:
        return True
    
    for obs_id, obs_data in dynamic_obstacles.items():
        # Get obstacle parameters
        diameter = obs_data.get("diameter", 0.5)
        effective_radius = (diameter / 2.0) + SAFETY_MARGIN
        height = obs_data.get("height", 5.0)
        
        if lookahead_time > 0:
            # Check against the entire trajectory
            obstacle_trajectory = predict_obstacle_trajectory(obs_data, lookahead_time)
            
            for pred_pos in obstacle_trajectory:
                # Check horizontal distance (cylinder radius)
                dx = x - pred_pos[0]
                dy = y - pred_pos[1]
                horizontal_dist = math.sqrt(dx*dx + dy*dy)
                
                if horizontal_dist <= effective_radius:
                    # Check vertical distance (cylinder height)
                    if 0 <= z <= height:
                        return False
        else:
            # Just check current position (traditional approach)
            pos = obs_data["position"]
            
            # Check horizontal distance (cylinder radius)
            dx = x - pos[0]
            dy = y - pos[1]
            horizontal_dist = math.sqrt(dx*dx + dy*dy)
            
            if horizontal_dist <= effective_radius:
                # Check vertical distance (cylinder height)
                if 0 <= z <= height:
                    return False
    
    return True

# --------------------------
# A* Path Planning
# --------------------------
def plan_rrt(start, goal, dynamic_obstacles, bounds, 
              step_size=STEP_SIZE, max_nodes=MAX_NODES, 
              planning_time_limit=PLANNING_TIME_LIMIT, prediction_time=LOOKAHEAD_TIME):
    """
    Plan a collision-free path between start and goal using a Rapidly-Exploring Random Tree (RRT).
    Takes into account predicted positions of dynamic obstacles.
    
    Args:
        start: Start position as [x, y, z]
        goal: Goal position as [x, y, z]
        dynamic_obstacles: Dictionary of obstacles with position, velocity, etc.
        bounds: Dictionary with 'low' and 'high' 3D bounds
        step_size: Maximum distance for extending the tree
        max_nodes: Maximum number of nodes in the tree
        planning_time_limit: Maximum time allowed for planning
        prediction_time: How far ahead to predict obstacle positions
        
    Returns:
        path: List of waypoints from start to goal, or None if no path found
        planning_time: Time taken for planning
    """
    # Debug info
    print(f"Planning RRT from {start} to {goal}")
    print(f"Bounds: {bounds}")
    print(f"Number of obstacles: {len(dynamic_obstacles)}")
    print(f"Using lookahead time: {prediction_time}s to predict obstacle positions")
    
    start_time = time.time()
    
    # For prediction, create a deep copy of obstacles
    predicted_obstacles = {}
    for obs_id, obs_data in dynamic_obstacles.items():
        predicted_obstacles[obs_id] = obs_data.copy()
    
    # Create bounds ensuring they respect altitude limits
    low_x, low_y, low_z = bounds["low"][0], bounds["low"][1], bounds["low"][2]
    high_x, high_y, high_z = bounds["high"][0], bounds["high"][1], bounds["high"][2]
    
    # Ensure z-bounds respect the altitude limits
    low_z = max(low_z, MIN_ALTITUDE)
    high_z = min(high_z, MAX_ALTITUDE)
    
    # Define a Node class for the RRT tree
    class Node:
        def __init__(self, position):
            self.position = position  # [x, y, z]
            self.parent = None
            self.cost = 0.0  # Cost from start to this node
    
    # Initialize tree with start node
    start_node = Node(start)
    nodes = [start_node]
    
    # Check if direct path is possible
    if is_path_clear(start, goal, predicted_obstacles, check_points=10):
        print("Direct path is clear, returning straight line")
        return [start, goal], time.time() - start_time
    
    # Create helper functions for the RRT algorithm
    def random_sample():
        """Generate a random sample in the 3D workspace"""
        if random.random() < GOAL_BIAS:
            # With some probability, sample the goal directly
            return goal.copy()
        else:
            # Sample randomly within bounds
            x = random.uniform(low_x, high_x)
            y = random.uniform(low_y, high_y)
            z = random.uniform(low_z, high_z)
            return [x, y, z]
    
    def nearest_node(position):
        """Find the nearest node in the tree to the given position"""
        min_dist = float('inf')
        nearest = None
        for node in nodes:
            dist = compute_euclidean_distance(node.position, position)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        return nearest
    
    def steer(from_pos, to_pos, step):
        """Steer from one position toward another with maximum step size"""
        dist = compute_euclidean_distance(from_pos, to_pos)
        if dist <= step:
            return to_pos.copy()
        else:
            # Interpolate with maximum step
            ratio = step / dist
            new_pos = [
                from_pos[0] + ratio * (to_pos[0] - from_pos[0]),
                from_pos[1] + ratio * (to_pos[1] - from_pos[1]),
                from_pos[2] + ratio * (to_pos[2] - from_pos[2])
            ]
            return new_pos
    
    def is_edge_valid(pos1, pos2):
        """Check if the edge between two positions is collision-free"""
        # Check multiple points along the edge
        check_points = max(3, int(compute_euclidean_distance(pos1, pos2) / (step_size/2)))
        for i in range(check_points + 1):
            t = i / check_points
            interp = [
                pos1[0] + t * (pos2[0] - pos1[0]),
                pos1[1] + t * (pos2[1] - pos1[1]),
                pos1[2] + t * (pos2[2] - pos1[2])
            ]
            if not is_state_valid(interp, predicted_obstacles, prediction_time):
                return False
        return True
    
    # Main RRT loop
    goal_node = None
    iterations = 0
    
    while iterations < max_nodes and time.time() - start_time < planning_time_limit:
        iterations += 1
        
        # Sample a random point
        random_point = random_sample()
        
        # Find nearest node in the tree
        nearest = nearest_node(random_point)
        
        # Steer toward the random point with maximum step size
        new_pos = steer(nearest.position, random_point, step_size)
        
        # Check if the new position and edge to it are valid
        if is_state_valid(new_pos, predicted_obstacles, prediction_time) and is_edge_valid(nearest.position, new_pos):
            # Create and add new node
            new_node = Node(new_pos)
            new_node.parent = nearest
            new_node.cost = nearest.cost + compute_euclidean_distance(nearest.position, new_pos)
            nodes.append(new_node)
            
            # Check if we've reached the goal
            dist_to_goal = compute_euclidean_distance(new_pos, goal)
            if dist_to_goal <= GOAL_THRESHOLD:
                goal_node = new_node
                print(f"Goal reached after {iterations} iterations")
                break
        
        # Occasionally print progress
        if iterations % 500 == 0:
            print(f"RRT iteration {iterations}, nodes: {len(nodes)}, elapsed: {time.time()-start_time:.2f}s")
    
    planning_time = time.time() - start_time
    
    # Check if we found a path
    if goal_node is None:
        # If we didn't reach goal but time is up, find the closest node to goal
        if time.time() - start_time >= planning_time_limit:
            print(f"RRT planning time limit exceeded after {iterations} iterations")
            # Find node closest to goal
            min_dist = float('inf')
            for node in nodes:
                dist = compute_euclidean_distance(node.position, goal)
                if dist < min_dist:
                    min_dist = dist
                    goal_node = node
            
            if min_dist <= GOAL_THRESHOLD * 2:  # Accept if reasonably close
                print(f"Found path to near-goal position, distance: {min_dist:.2f}")
            else:
                print(f"No path found within time limit, closest distance: {min_dist:.2f}")
                return None, planning_time
        else:
            print(f"RRT failed to find path after {iterations} iterations")
            return None, planning_time
    
    # Reconstruct path from goal to start
    path = []
    current = goal_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    
    # Reverse to get path from start to goal
    path.reverse()
    
    # Add actual goal to the path if not already there
    if compute_euclidean_distance(path[-1], goal) > GOAL_THRESHOLD/2:
        path.append(goal)
    
    # Path smoothing - remove redundant waypoints while maintaining safety
    smoothed_path = [path[0]]  # Start with the first waypoint
    i = 0
    while i < len(path) - 1:
        # Try to connect to farthest safe waypoint
        for j in range(len(path) - 1, i, -1):
            if is_edge_valid(path[i], path[j]):
                smoothed_path.append(path[j])
                i = j
                break
        else:
            # If no skip possible, just add the next waypoint
            i += 1
            if i < len(path):
                smoothed_path.append(path[i])
    
    print(f"RRT planning completed in {planning_time:.3f}s, generated {len(nodes)} nodes, path length: {len(smoothed_path)}")
    return smoothed_path, planning_time

def is_path_clear(start, goal, obstacles, check_points=5):
    """Check if there's a clear straight line from start to goal"""
    for i in range(check_points + 1):
        t = i / check_points
        interp = [
            start[0] + t * (goal[0] - start[0]),
            start[1] + t * (goal[1] - start[1]),
            start[2] + t * (goal[2] - start[2])
        ]
        # Also use the trajectory-based collision checking for path validation
        if not is_state_valid(interp, obstacles, LOOKAHEAD_TIME):
            return False
    return True

# --------------------------
# Dynamic Obstacle Tracking
# --------------------------
class DynamicObstacleTracker(Node):
    """Tracks dynamic obstacles from PoseStamped messages"""
    
    def __init__(self):
        super().__init__('dynamic_obstacle_tracker')
        
        # Dictionary to track obstacles {obstacle_id: (position, velocity, last_update_time)}
        self.obstacles = {}
        self.obstacle_lock = threading.Lock()  # Lock for thread safety
        self.update_count = 0
        
        # Parameters from the scenario
        self.obstacle_height = 5.0
        self.obstacle_diameter = 0.5
        
        # Subscribe to dynamic obstacle positions
        self.obstacle_sub = self.create_subscription(
            PoseStamped,
            '/dynamic_obstacles/locations',
            self.obstacle_callback,
            10
        )
        self.get_logger().info("DynamicObstacleTracker initialized and subscribed to /dynamic_obstacles/locations")
    
    def set_obstacle_parameters(self, height, diameter):
        """Set parameters for obstacles from the scenario file"""
        self.obstacle_height = height
        self.obstacle_diameter = diameter
    
    def obstacle_callback(self, msg):
        """Process incoming obstacle position updates"""
        # Extract obstacle ID from frame_id
        obstacle_id = msg.header.frame_id
        
        with self.obstacle_lock:
            self.update_count += 1
            current_time = time.time()
            current_position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
            
            # Log the received obstacle data
            # DEBUG: Uncomment the following line to see all obstacle updates
            # self.get_logger().info(f"Received obstacle update for {obstacle_id} at position [{current_position[0]:.2f}, {current_position[1]:.2f}]")
            
            # Update existing obstacle or create new one
            if obstacle_id in self.obstacles:
                # Get previous information to calculate velocity
                prev_position, prev_velocity, prev_time = self.obstacles[obstacle_id]
                
                # Calculate time difference
                time_diff = current_time - prev_time
                if time_diff > 0:
                    # Calculate new velocity (position difference / time)
                    new_velocity = [
                        (current_position[0] - prev_position[0]) / time_diff,
                        (current_position[1] - prev_position[1]) / time_diff,
                        (current_position[2] - prev_position[2]) / time_diff
                    ]
                    
                    # Smooth velocity with previous estimate (simple low-pass filter)
                    velocity = [
                        0.7 * new_velocity[0] + 0.3 * prev_velocity[0],
                        0.7 * new_velocity[1] + 0.3 * prev_velocity[1],
                        0.7 * new_velocity[2] + 0.3 * prev_velocity[2]
                    ]
                    
                    # Update obstacle information
                    self.obstacles[obstacle_id] = (current_position, velocity, current_time)
                    # DEBUG: Uncomment for detailed velocity updates
                    # self.get_logger().debug(f"Updated obstacle {obstacle_id}: pos={current_position[:2]}, vel={velocity[:2]}")
                else:
                    # Just update position if time difference is too small
                    self.obstacles[obstacle_id] = (current_position, prev_velocity, current_time)
            else:
                # For new obstacles, assume zero initial velocity
                self.obstacles[obstacle_id] = (current_position, [0.0, 0.0, 0.0], current_time)
                # Keep this log for new obstacles only
                self.get_logger().info(f"Added new obstacle {obstacle_id} at position {current_position[:2]}")
            
            # Only log every 10 updates to reduce console spam
            if self.update_count % 1000 == 0:
                self.get_logger().info(f"Currently tracking {len(self.obstacles)} obstacles")
    
    def get_obstacles(self):
        """Return the current obstacles dictionary in the format expected by the rest of the code"""
        with self.obstacle_lock:
            # Convert our tuple format to the dictionary format expected by the existing code
            result = {}
            for obstacle_id, (position, velocity, last_update) in self.obstacles.items():
                result[obstacle_id] = {
                    "position": position,
                    "velocity": velocity,
                    "last_update": last_update,
                    "height": self.obstacle_height,
                    "diameter": self.obstacle_diameter
                }
            return result
    
    def get_obstacle_predictions(self, prediction_time=None, horizon=5.0, timestep=0.5):
        """Get predicted obstacle positions for planning"""
        if prediction_time is None:
            prediction_time = time.time()
        
        with self.obstacle_lock:
            predictions = {}
            
            # Skip if no obstacles are being tracked
            if not self.obstacles:
                return predictions
            
            # Generate predictions for each obstacle
            for obstacle_id, (position, velocity, last_update_time) in self.obstacles.items():
                # First, project to current time
                time_diff = prediction_time - last_update_time
                current_pos = [
                    position[0] + velocity[0] * time_diff,
                    position[1] + velocity[1] * time_diff,
                    position[2] + velocity[2] * time_diff
                ]
                
                # Generate trajectory for future time steps
                obstacle_traj = []
                for t in np.arange(0.0, horizon, timestep):
                    future_pos = [
                        current_pos[0] + velocity[0] * t,
                        current_pos[1] + velocity[1] * t,
                        current_pos[2] + velocity[2] * t
                    ]
                    obstacle_traj.append(future_pos)
                
                predictions[obstacle_id] = obstacle_traj
            
            return predictions

# --------------------------
# Conflict-Based Search (CBS) for Multi-Drone Path Planning
# --------------------------
class CBSConstraint:
    """
    Constraint for CBS algorithm with position, timestep, and radius
    """
    def __init__(self, drone_id, position, timestep, radius=0.5):
        self.drone_id = drone_id
        self.position = position
        self.timestep = timestep
        self.radius = radius  # Radius around position to avoid
        
    def __str__(self):
        return f"Constraint(drone={self.drone_id}, pos={self.position}, t={self.timestep}, r={self.radius})"

class CBSNode:
    """
    Node in the CBS constraint tree.
    Contains constraints, paths, conflicts, and cost information.
    """
    def __init__(self):
        self.constraints = {}  # {drone_id: [constraints]}
        self.paths = {}  # {drone_id: path}
        self.conflicts = []  # List of conflicts
        self.cost = 0  # Total cost of all paths
        
    def __lt__(self, other):
        """
        Comparison operator for priority queue.
        Sort by number of conflicts first, then by cost.
        """
        if len(self.conflicts) != len(other.conflicts):
            return len(self.conflicts) < len(other.conflicts)
        return self.cost < other.cost

def detect_conflicts(paths):
    """
    Detect conflicts between paths.
    
    Args:
        paths: Dictionary mapping drone_id to path
        
    Returns:
        List of conflicts, each being a tuple (drone_i, drone_j, position, timestep)
    """
    conflicts = []
    drone_ids = list(paths.keys())
    
    # Safety distance between drones
    SAFETY_DISTANCE = 1.0
    
    # Check each pair of drones
    for i in range(len(drone_ids)):
        for j in range(i + 1, len(drone_ids)):
            drone_i, drone_j = drone_ids[i], drone_ids[j]
            path_i, path_j = paths[drone_i], paths[drone_j]
            
            # Determine the maximum timestep to check
            max_timestep = max(len(path_i), len(path_j))
            
            for t in range(max_timestep):
                # Get positions at timestep t, using the last position if the path is shorter
                pos_i = path_i[min(t, len(path_i) - 1)]
                pos_j = path_j[min(t, len(path_j) - 1)]
                
                # Calculate distance between drones
                distance = ((pos_i[0] - pos_j[0])**2 + 
                           (pos_i[1] - pos_j[1])**2 + 
                           (pos_i[2] - pos_j[2])**2)**0.5
                
                # Check if drones are too close
                if distance < SAFETY_DISTANCE:
                    # Record the conflict with the midpoint position
                    conflict_pos = ((pos_i[0] + pos_j[0]) / 2,
                                   (pos_i[1] + pos_j[1]) / 2,
                                   (pos_i[2] + pos_j[2]) / 2)
                    conflicts.append((drone_i, drone_j, conflict_pos, t))
                    break  # Only record the first conflict between this pair
    
    return conflicts

def astar_plan_path(start, goal, drone_id, constraints, dynamic_obstacles=None, bounds=None):
    """
    A* path planning with constraints.
    
    Args:
        start: Start position (x, y, z)
        goal: Goal position (x, y, z)
        drone_id: ID of the drone
        constraints: List of CBSConstraint objects
        dynamic_obstacles: List of dynamic obstacles
        bounds: Planning bounds
        
    Returns:
        Planned path as a list of positions
    """
    import heapq
    import numpy as np
    
    # Extract constraints relevant to this drone
    drone_constraints = [c for c in constraints if c.drone_id == drone_id]
    
    # If there are no constraints and direct path is clear, return direct path
    if not drone_constraints and dynamic_obstacles:
        if is_path_clear(start, goal, dynamic_obstacles, check_points=10):
            print(f"Direct path is clear for drone {drone_id}, returning straight line")
            return [start, goal]
    
    # Default resolution for A* grid
    resolution = ASTAR_RESOLUTION
    
    # Define grid bounds with some padding
    if bounds:
        x_min, y_min, z_min = bounds["low"]
        x_max, y_max, z_max = bounds["high"]
    else:
        # Default bounds if none provided
        x_min, y_min, z_min = -10, -10, MIN_ALTITUDE
        x_max, y_max, z_max = 10, 10, MAX_ALTITUDE
    
    # Ensure z bounds respect altitude limits
    z_min = max(z_min, MIN_ALTITUDE)
    z_max = min(z_max, MAX_ALTITUDE)
    
    # A* grid dimensions
    grid_dims = [
        int((x_max - x_min) / resolution) + 1,
        int((y_max - y_min) / resolution) + 1,
        int((z_max - z_min) / resolution) + 1
    ]
    
    # Define node class for A*
    class AStarNode:
        def __init__(self, position, g=float('inf'), h=0, parent=None, timestep=0):
            self.position = position  # (x, y, z)
            self.g = g  # Cost from start
            self.h = h  # Heuristic to goal
            self.f = g + h  # Total cost
            self.parent = parent  # Parent node
            self.timestep = timestep  # Time step (for constraints)
        
        def __lt__(self, other):
            return self.f < other.f
    
    # Convert between grid indices and world coordinates
    def grid_to_world(indices):
        return (
            x_min + indices[0] * resolution,
            y_min + indices[1] * resolution,
            z_min + indices[2] * resolution
        )
    
    def world_to_grid(coords):
        return (
            int((coords[0] - x_min) / resolution),
            int((coords[1] - y_min) / resolution),
            int((coords[2] - z_min) / resolution)
        )
    
    # Heuristic function (Euclidean distance)
    def heuristic(a, b):
        return compute_euclidean_distance(a, b)
    
    # Check if a position is valid (within bounds and collision-free)
    def is_position_valid(position, timestep):
        # Check bounds
        if (position[0] < x_min or position[0] > x_max or
            position[1] < y_min or position[1] > y_max or
            position[2] < z_min or position[2] > z_max):
            return False
        
        # Check collision with dynamic obstacles
        if dynamic_obstacles and not is_state_valid(position, dynamic_obstacles, LOOKAHEAD_TIME):
            return False
        
        # Check constraints
        for constraint in drone_constraints:
            if constraint.timestep == timestep:
                dist = compute_euclidean_distance(position, constraint.position)
                if dist < constraint.radius:
                    return False
        
        return True
    
    # Get neighbors for a grid cell
    def get_neighbors(position, timestep):
        neighbors = []
        
        # 6-connected grid (up, down, left, right, forward, backward)
        directions = [
            (resolution, 0, 0),   # Right
            (-resolution, 0, 0),  # Left
            (0, resolution, 0),   # Forward
            (0, -resolution, 0),  # Backward
            (0, 0, resolution),   # Up
            (0, 0, -resolution),  # Down
            
            # Add diagonal moves in xy-plane for smoother paths
            (resolution, resolution, 0),    # Right-Forward
            (resolution, -resolution, 0),   # Right-Backward
            (-resolution, resolution, 0),   # Left-Forward
            (-resolution, -resolution, 0),  # Left-Backward
        ]
        
        for dx, dy, dz in directions:
            new_pos = (position[0] + dx, position[1] + dy, position[2] + dz)
            
            # Check if the new position is valid
            if is_position_valid(new_pos, timestep + 1):
                neighbors.append(new_pos)
        
        # Always add "wait" action (stay in place) as an option
        if is_position_valid(position, timestep + 1):
            neighbors.append(position)
        
        return neighbors
    
    # Convert grid positions to start/goal
    start_grid = world_to_grid(start)
    goal_grid = world_to_grid(goal)
    
    # Convert back to world coordinates to ensure we're using grid-aligned positions
    start_world = grid_to_world(start_grid)
    goal_world = grid_to_world(goal_grid)
    
    # Initialize open and closed sets
    open_set = []
    closed_set = set()
    
    # Create start node
    start_node = AStarNode(
        position=start_world,
        g=0,
        h=heuristic(start_world, goal_world),
        timestep=0
    )
    
    # Add start node to open set
    heapq.heappush(open_set, start_node)
    
    # Track visited positions with their timesteps
    visited = {}  # (position, timestep) -> node
    visited[(start_world, 0)] = start_node
    
    # Main A* loop
    start_time = time.time()
    iterations = 0
    max_iterations = 10000  # Prevent infinite loops
    
    while open_set and iterations < max_iterations and time.time() - start_time < PLANNING_TIME_LIMIT:
        iterations += 1
        
        # Get node with lowest f-score
        current = heapq.heappop(open_set)
        
        # Check if we've reached the goal
        dist_to_goal = compute_euclidean_distance(current.position, goal_world)
        if dist_to_goal <= resolution * 1.5:
            # Goal reached, reconstruct path
            path = []
            node = current
            while node:
                path.append(node.position)
                node = node.parent
            
            path.reverse()
            
            # Make sure the exact goal is included
            if compute_euclidean_distance(path[-1], goal) > 0.1:
                path.append(goal)
                
            print(f"A* found path for drone {drone_id} with {len(path)} waypoints after {iterations} iterations")
            return path
        
        # Add to closed set
        pos_tuple = tuple(map(lambda x: round(x, 3), current.position))
        closed_set.add((pos_tuple, current.timestep))
        
        # Get neighbors
        neighbors = get_neighbors(current.position, current.timestep)
        
        for neighbor_pos in neighbors:
            # Calculate tentative g score
            move_cost = compute_euclidean_distance(current.position, neighbor_pos)
            if neighbor_pos == current.position:  # "wait" action
                move_cost = 0.5  # Waiting has a small cost
                
            tentative_g = current.g + move_cost
            
            # New timestep
            new_timestep = current.timestep + 1
            
            # Check if we've seen this position at this timestep
            neighbor_tuple = tuple(map(lambda x: round(x, 3), neighbor_pos))
            
            if (neighbor_tuple, new_timestep) in closed_set:
                continue
            
            # Create or update neighbor node
            new_node = AStarNode(
                position=neighbor_pos,
                g=tentative_g,
                h=heuristic(neighbor_pos, goal_world),
                parent=current,
                timestep=new_timestep
            )
            
            # If we found a better path to this neighbor
            if (neighbor_pos, new_timestep) not in visited or tentative_g < visited[(neighbor_pos, new_timestep)].g:
                visited[(neighbor_pos, new_timestep)] = new_node
                heapq.heappush(open_set, new_node)
    
    # If we exit the loop without finding a path
    print(f"A* failed to find path for drone {drone_id} after {iterations} iterations")
    
    # Try RRT as a fallback if A* failed
    if dynamic_obstacles:
        print(f"Trying RRT as fallback for drone {drone_id}")
        rrt_path, _ = plan_rrt(start, goal, dynamic_obstacles, bounds, prediction_time=LOOKAHEAD_TIME)
        return rrt_path
    
    # Return direct path as a last resort
    return [start, goal]

def cbs_plan_multi_drone_paths(drone_data, dynamic_obstacles=None, bounds=None):
    """
    Conflict-Based Search (CBS) for multi-drone path planning with improved handling of time-zero conflicts.
    
    Args:
        drone_data: Dictionary mapping drone_id to (start_position, goal_position)
        dynamic_obstacles: Optional list of dynamic obstacles with positions over time
        bounds: Optional tuple (min_bounds, max_bounds) for the planning space
        
    Returns:
        Dictionary mapping drone_id to planned path
    """
    import heapq
    import time
    from copy import deepcopy
    
    # Initialize statistics for tracking conflicts
    conflict_stats = {
        'total_conflicts': 0,
        'time_zero_conflicts': 0,
        'conflict_pairs': set()
    }
    
    # Vertical separation parameters
    MIN_VERTICAL_SEPARATION = 1.0  # Minimum vertical separation in meters
    MAX_ATTEMPTS = 5  # Maximum number of attempts to resolve initial conflicts
    
    # Create initial CBS node
    root = CBSNode()
    
    # Initialize empty constraints for each drone
    for drone_id in drone_data:
        root.constraints[drone_id] = []
    
    # Initial path planning for each drone
    for drone_id, (start, goal) in drone_data.items():
        # Plan the initial path without constraints
        path = astar_plan_path(start, goal, drone_id, [], dynamic_obstacles, bounds)
        if path is None:
            print(f"Failed to find initial path for drone {drone_id}")
            return None
        root.paths[drone_id] = path
        root.cost += len(path)
    
    # Check for conflicts
    root.conflicts = detect_conflicts(root.paths)
    conflict_stats['total_conflicts'] += len(root.conflicts)
    
    # Handle time-zero conflicts by adjusting starting positions
    time_zero_conflicts = [c for c in root.conflicts if c[3] == 0]  # [drone_i, drone_j, pos, time]
    
    if time_zero_conflicts:
        conflict_stats['time_zero_conflicts'] += len(time_zero_conflicts)
        for drone_i, drone_j, _, _ in time_zero_conflicts:
            conflict_stats['conflict_pairs'].add((min(drone_i, drone_j), max(drone_i, drone_j)))
        
        # Attempt to resolve initial conflicts by staggering starting altitudes
        attempt = 0
        while time_zero_conflicts and attempt < MAX_ATTEMPTS:
            attempt += 1
            print(f"Attempting to resolve {len(time_zero_conflicts)} time-zero conflicts (attempt {attempt})")
            
            # Adjust starting positions with increasing vertical separation
            modified_drone_data = deepcopy(drone_data)
            for drone_i, drone_j, _, _ in time_zero_conflicts:
                # Add vertical separation
                start_i, goal_i = modified_drone_data[drone_i]
                start_j, goal_j = modified_drone_data[drone_j]
                
                # Adjust starting heights
                start_i = (start_i[0], start_i[1], start_i[2] + attempt * MIN_VERTICAL_SEPARATION)
                start_j = (start_j[0], start_j[1], start_j[2] - attempt * MIN_VERTICAL_SEPARATION)
                
                modified_drone_data[drone_i] = (start_i, goal_i)
                modified_drone_data[drone_j] = (start_j, goal_j)
            
            # Replan with adjusted starting positions
            root = CBSNode()
            for drone_id in modified_drone_data:
                root.constraints[drone_id] = []
            
            # Replan paths with new starting positions
            for drone_id, (start, goal) in modified_drone_data.items():
                path = astar_plan_path(start, goal, drone_id, [], dynamic_obstacles, bounds)
                if path is None:
                    print(f"Failed to find path for drone {drone_id} with adjusted starting position")
                    continue
                root.paths[drone_id] = path
                root.cost += len(path)
            
            # Check if we've resolved the time-zero conflicts
            root.conflicts = detect_conflicts(root.paths)
            time_zero_conflicts = [c for c in root.conflicts if c[3] == 0]
    
    # Initialize the open list with the root node
    open_list = [root]
    
    # Main CBS loop
    start_time = time.time()
    iterations = 0
    max_iterations = 1000  # Prevent infinite loops
    
    while open_list and iterations < max_iterations:
        iterations += 1
        
        # Get the node with lowest cost
        current = heapq.heappop(open_list)
        
        # If there are no conflicts, we have found a solution
        if not current.conflicts:
            elapsed = time.time() - start_time
            print(f"CBS found solution in {iterations} iterations, {elapsed:.2f} seconds")
            print(f"Total conflicts encountered: {conflict_stats['total_conflicts']}")
            print(f"Time-zero conflicts: {conflict_stats['time_zero_conflicts']}")
            print(f"Conflict pairs: {conflict_stats['conflict_pairs']}")
            return current.paths
        
        # Choose the first conflict to resolve
        drone_i, drone_j, conflict_pos, conflict_time = current.conflicts[0]
        
        # Create two child nodes to resolve the conflict
        for drone_id in [drone_i, drone_j]:
            # Create a new node
            child = CBSNode()
            
            # Copy constraints and paths from parent
            for d_id in current.paths:
                child.constraints[d_id] = current.constraints.get(d_id, [])[:]
            child.paths = deepcopy(current.paths)
            
            # Add new constraint
            constraint = CBSConstraint(drone_id, conflict_pos, conflict_time)
            if drone_id not in child.constraints:
                child.constraints[drone_id] = []
            child.constraints[drone_id].append(constraint)
            
            # Replan path for the constrained drone
            new_path = astar_plan_path(
                drone_data[drone_id][0], 
                drone_data[drone_id][1],
                drone_id, 
                child.constraints[drone_id],
                dynamic_obstacles,
                bounds
            )
            
            # If a valid path is found, update the child node
            if new_path:
                # Update the path and cost
                old_path_cost = len(child.paths[drone_id])
                child.paths[drone_id] = new_path
                child.cost = current.cost - old_path_cost + len(new_path)
                
                # Check for new conflicts
                child.conflicts = detect_conflicts(child.paths)
                conflict_stats['total_conflicts'] += len(child.conflicts)
                
                # Add the child to the open list
                heapq.heappush(open_list, child)
    
    # If we've exhausted all nodes or reached iteration limit without finding a solution
    print(f"CBS failed to find a solution after {iterations} iterations")
    print(f"Total conflicts encountered: {conflict_stats['total_conflicts']}")
    
    # Return the best solution found so far (may have conflicts)
    if open_list:
        best_node = min(open_list, key=lambda n: len(n.conflicts))
        return best_node.paths
    return None

def cbs_parallel_planning_task(start, goal, drone_id, constraints, dynamic_obstacles, bounds, node, result_queue):
    """
    Task function for parallel planning in CBS.
    Plans a path for a drone with given constraints and adds the result to the queue.
    Now incorporating altitude variations to resolve conflicts and tracking planning duration.
    """
    try:
        planning_start_time = time.time()
        success = False
        
        # First, try planning with slight altitude adjustment to the goal
        # This biases the planner to adjust altitude to resolve conflicts
        altitude_options = [0, 0.2, -0.2, 0.4, -0.4]  # Try original altitude first, then +/-20cm, then +/-40cm
        
        for alt_adjustment in altitude_options:
            adjusted_goal = goal.copy()
            adjusted_goal[2] += alt_adjustment
            
            # Ensure adjusted goal is within altitude limits
            adjusted_goal[2] = max(MIN_ALTITUDE, min(MAX_ALTITUDE, adjusted_goal[2]))
            
            # Debug message for altitude adjustment attempt
            if alt_adjustment != 0:
                print(f"DEBUG: Attempting altitude variation of {alt_adjustment:.2f}m for drone {drone_id} " 
                      f"(Z={adjusted_goal[2]:.2f}m)")
            
            # Plan a new path with the constraints and adjusted goal
            new_path, planning_time = time_indexed_rrt(
                start, adjusted_goal, drone_id, constraints, 
                dynamic_obstacles, bounds
            )
            
            # If planning succeeded, update the path and add the node to the result queue
            if new_path:
                node.paths[drone_id] = new_path
                node.cost = max(len(path) for path in node.paths.values())
                
                # Debug message for successful planning with altitude adjustment
                if alt_adjustment != 0:
                    print(f"DEBUG: Successfully planned path for drone {drone_id} with altitude " 
                          f"variation of {alt_adjustment:.2f}m (goal Z={adjusted_goal[2]:.2f}m)")
                
                success = True
                break
            elif alt_adjustment != 0:
                print(f"DEBUG: Planning failed with altitude adjustment of {alt_adjustment:.2f}m for drone {drone_id}")
        
        # If all altitude adjustments fail, try with the original goal one more time
        # (this might be redundant with the first try, but serves as a fallback)
        if not success:
            new_path, planning_time = time_indexed_rrt(
                start, goal, drone_id, constraints, 
                dynamic_obstacles, bounds
            )
            
            if new_path:
                node.paths[drone_id] = new_path
                node.cost = max(len(path) for path in node.paths.values())
                success = True
            else:
                print(f"Replanning failed for drone {drone_id} with constraints after trying all altitude adjustments")
        
        # Calculate total planning duration
        planning_duration = time.time() - planning_start_time
        
        # Store planning duration in the node for branch selection heuristics
        node.planning_duration = planning_duration
        
        # Print planning duration for profiling
        constraint_count = len([c for c in constraints if c.drone_id == drone_id])
        print(f"CBS planning for drone {drone_id} with {constraint_count} constraints took {planning_duration:.3f}s")
        
        result_queue.put((node, success, planning_duration))
    except Exception as e:
        print(f"Error in parallel planning task for drone {drone_id}: {e}")
        result_queue.put((node, False, float('inf')))

# --------------------------
# Multi-Drone Mission Class
# --------------------------
class MultiDroneMission:
    def __init__(self, drone_ids, scenario_file, use_sim_time=True, verbose=False):
        """Initialize the mission with multiple drones"""
        self.drone_ids = drone_ids
        self.drones = {}
        self.drone_positions = {}  # Store latest positions
        
        # Load scenario
        self.scenario = load_scenario(scenario_file)
        self.setup_from_scenario(self.scenario)
        
        # Initialize ROS
        rclpy.init()
        
        # Create a simple node for position subscriptions
        self.position_node = rclpy.create_node('position_subscriber_node')
        
        # Create QoS profile that matches the publisher
        position_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Initialize dynamic obstacle tracker
        self.obstacle_tracker = DynamicObstacleTracker()
        
        # Get starting positions in a zigzag pattern
        start_positions = self.get_distributed_positions(self.start_position, len(drone_ids))
        
        # Initialize drones and position subscribers
        for idx, drone_id in enumerate(drone_ids):
            self.drones[drone_id] = DroneInterface(drone_id=drone_id, use_sim_time=use_sim_time, verbose=verbose)
            
            # Initialize position with the calculated start position
            self.drone_positions[drone_id] = start_positions[idx].copy()
            
            # Subscribe to position topic with correct QoS
            topic = f'/{drone_id}/self_localization/pose'
            self.position_node.create_subscription(
                PoseStamped,
                topic,
                lambda msg, d_id=drone_id: self.position_callback(msg, d_id),
                position_qos
            )
            print(f"Subscribed to position topic: {topic} with BEST_EFFORT reliability")
        
        # Create separate executors for each node to avoid threading issues
        self.position_executor = rclpy.executors.SingleThreadedExecutor()
        self.position_executor.add_node(self.position_node)
        
        self.obstacle_executor = rclpy.executors.SingleThreadedExecutor()
        self.obstacle_executor.add_node(self.obstacle_tracker)
        
        # Start position node in a separate thread
        self.position_thread = threading.Thread(target=self.spin_position_node, daemon=True)
        self.position_thread.start()
        
        # Start obstacle tracker in a separate thread
        self.obstacle_thread = threading.Thread(target=self.spin_obstacle_node, daemon=True)
        self.obstacle_thread.start()
        
        # Check if we need to start the scenario runner
        self.check_and_start_scenario_runner(scenario_file)
        
        # Allow more time for obstacle data to be received
        print("Waiting for obstacle data to be published (this may take 10 seconds)...")
        obstacle_wait_time = 10.0  # Increase wait time to 10 seconds
        start_wait = time.time()
        while time.time() - start_wait < obstacle_wait_time:
            obstacles = self.obstacle_tracker.get_obstacles()
            if obstacles:
                print(f"Detected {len(obstacles)} obstacles after {time.time() - start_wait:.1f} seconds!")
                break
            time.sleep(0.5)
            print(".", end="", flush=True)
        print()  # Newline after progress dots
        
        # Set obstacle parameters from scenario
        self.obstacle_tracker.set_obstacle_parameters(
            self.obstacle_height,
            self.obstacle_diameter
        )
        
        print(f"Multi-drone mission initialized with {len(drone_ids)} drones in a zigzag formation")
    
    def check_and_start_scenario_runner(self, scenario_file):
        """Check if the scenario runner is active and start it if needed"""
        # Create a temporary node to check for active publishers on the obstacle topic
        check_node = rclpy.create_node('topic_check_node')
        
        # Wait a bit for discovery
        time.sleep(2.0)
        
        # Get publishers on the obstacle topic
        topic_name = '/dynamic_obstacles/locations'
        publishers_info = check_node.get_publishers_info_by_topic(topic_name)
        
        if not publishers_info:
            print(f"No publishers found on {topic_name}. Starting scenario runner...")
            
            # Start the scenario runner in a separate process
            try:
                import subprocess
                
                # Make sure the scenario file exists
                if not os.path.exists(scenario_file):
                    print(f"Error: Scenario file {scenario_file} not found!")
                    scenario_file = "scenarios/scenario1_stage4.yaml"
                    print(f"Falling back to default: {scenario_file}")
                
                # Try to create an absolute path to the scenario file
                current_dir = os.getcwd()
                scenario_path = os.path.join(current_dir, scenario_file)
                print(f"Starting scenario runner with file: {scenario_path}")
                
                # Run the command
                cmd = f"ros2 run config_sim ros2 scenario_runner --ros-args -p scenario_file:={scenario_path}"
                print(f"Running command: {cmd}")
                process = subprocess.Popen(cmd, shell=True)
                
                # Wait for scenario runner to start
                print("Waiting for scenario runner to start publishing obstacles...")
                wait_time = 0
                max_wait = 15  # seconds
                while wait_time < max_wait:
                    time.sleep(1.0)
                    wait_time += 1
                    print(".", end="", flush=True)
                    # Check if publishers exist now
                    publishers = check_node.get_publishers_info_by_topic(topic_name)
                    if publishers:
                        print(f"\nScenario runner started successfully after {wait_time}s!")
                        break
                
                if wait_time >= max_wait:
                    print("\nWarning: Scenario runner may not have started properly.")
            except Exception as e:
                print(f"Failed to start scenario runner: {e}")
        else:
            print(f"Found {len(publishers_info)} publishers on {topic_name}")
        
        # Clean up the temporary node
        check_node.destroy_node()
    
    def spin_position_node(self):
        """Spin the position subscriber node in a separate thread"""
        try:
            self.position_executor.spin()
        except Exception as e:
            print(f"Error in position node: {e}")
    
    def spin_obstacle_node(self):
        """Spin the obstacle tracker node in a separate thread"""
        try:
            self.obstacle_executor.spin()
        except Exception as e:
            print(f"Error in obstacle tracker node: {e}")
            
    def position_callback(self, msg, drone_id):
        """Callback for position updates"""
        position = [
            msg.pose.position.x,
            msg.pose.position.y, 
            msg.pose.position.z
        ]
        self.drone_positions[drone_id] = position
        # Uncomment for debugging
        # print(f"Position update for {drone_id}: {position}")
    
    def get_drone_position(self, drone):
        """Get the current position of the drone from stored positions"""
        return self.drone_positions[drone.drone_id].copy()

    def setup_from_scenario(self, scenario):
        """Extract important information from the scenario file"""
        # Stage 4 data
        self.stage4 = scenario.get("stage4", {})
        self.stage_center = self.stage4.get("stage_center", [0.0, 6.0])
        self.start_point = self.stage4.get("start_point", [0.0, -6.0])
        self.end_point = self.stage4.get("end_point", [0.0, 6.0])
        
        # Convert relative points to absolute coordinates
        self.start_position = [
            self.stage_center[0] + self.start_point[0],
            self.stage_center[1] + self.start_point[1],
            TAKE_OFF_HEIGHT
        ]
        
        self.end_position = [
            self.stage_center[0] + self.end_point[0],
            self.stage_center[1] + self.end_point[1],
            TAKE_OFF_HEIGHT
        ]
        
        # Stage bounds from scenario
        stage_size = scenario.get("stage_size", [10.0, 10.0])
        
        # Calculate exact bounds according to stage4.yaml with +2.0 safety margin
        stage_min_x = self.stage_center[0] - stage_size[0]/2.0
        stage_max_x = self.stage_center[0] + stage_size[0]/2.0
        stage_min_y = self.stage_center[1] - stage_size[1]/2.0
        stage_max_y = self.stage_center[1] + stage_size[1]/2.0
        
        self.bounds = {
            "low": [
                stage_min_x - 2.0,  # Add safety margin
                stage_min_y - 2.0,  # Add safety margin
                MIN_ALTITUDE      # Use the allowed minimum altitude
            ],
            "high": [
                stage_max_x + 2.0,  # Add safety margin
                stage_max_y + 2.0,  # Add safety margin
                MAX_ALTITUDE      # Use the allowed maximum altitude
            ]
        }
        
        # Get obstacle parameters
        self.obstacle_diameter = self.stage4.get("obstacle_diameter", 0.5)
        self.obstacle_height = self.stage4.get("obstacle_height", 5.0)
        
        print(f"Stage 4 setup: Start={self.start_position}, End={self.end_position}")
        print(f"Stage center: {self.stage_center}, Stage size: {stage_size}")
        print(f"Stage boundaries: X=[{stage_min_x:.1f}, {stage_max_x:.1f}], Y=[{stage_min_y:.1f}, {stage_max_y:.1f}]")
        print(f"A* bounds with vertical range: X=[{self.bounds['low'][0]:.1f}, {self.bounds['high'][0]:.1f}], "
              f"Y=[{self.bounds['low'][1]:.1f}, {self.bounds['high'][1]:.1f}], "
              f"Z=[{self.bounds['low'][2]:.1f}, {self.bounds['high'][2]:.1f}]")
    
    def start_mission(self):
        """Start the mission: all drones arm, offboard, and takeoff simultaneously"""
        print("Arming all drones...")
        
        # Use threads for arming all drones simultaneously
        arm_threads = []
        for drone_id, drone in self.drones.items():
            print(f"Arming {drone_id}")
            thread = threading.Thread(target=drone.arm, daemon=True)
            thread.start()
            arm_threads.append(thread)
        
        # Wait for all arming commands to complete
        for thread in arm_threads:
            thread.join()
        
        # Small delay after arming
        time.sleep(0.5)
        
        # Use threads for offboard mode
        print("Setting all drones to offboard mode...")
        offboard_threads = []
        for drone_id, drone in self.drones.items():
            thread = threading.Thread(target=drone.offboard, daemon=True)
            thread.start()
            offboard_threads.append(thread)
        
        # Wait for all offboard commands to complete
        for thread in offboard_threads:
            thread.join()
        
        # Small delay after offboard
        time.sleep(0.5)
        
        # Command all drones to takeoff simultaneously using threads
        print("Commanding all drones to takeoff simultaneously...")
        takeoff_threads = []
        for drone_id, drone in self.drones.items():
            thread = threading.Thread(
                target=drone.takeoff,
                kwargs={'height': TAKE_OFF_HEIGHT, 'speed': TAKE_OFF_SPEED},
                daemon=True
            )
            thread.start()
            takeoff_threads.append(thread)
        
        # Wait for all takeoff commands to be sent
        for thread in takeoff_threads:
            thread.join()
        
        # Give time for all drones to reach takeoff height
        print("Waiting for all drones to reach takeoff height...")
        time.sleep(5.0)
        
        # Get starting positions in a zigzag pattern
        print("Moving drones to their zigzag formation...")
        start_positions = self.get_distributed_positions(self.start_position, len(self.drones))
        
        # Command drones to move to their starting positions in the zigzag pattern
        threads = []
        for idx, (drone_id, drone) in enumerate(self.drones.items()):
            target_pos = start_positions[idx]
            # Calculate yaw to face towards the goal
            yaw = math.atan2(
                self.end_position[1] - target_pos[1],
                self.end_position[0] - target_pos[0]
            )
            
            print(f"Moving {drone_id} to position: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}] with yaw {yaw:.2f}")
            
            # Start moving in a separate thread
            thread = threading.Thread(
                target=drone.go_to.go_to_point_with_yaw,
                args=(target_pos,),
                kwargs={'angle': yaw, 'speed': SPEED/2},  # Use lower speed for formation
                daemon=True
            )
            thread.start()
            threads.append(thread)
        
        # Wait for all drones to reach their starting positions
        print("Waiting for all drones to reach their starting positions...")
        for thread in threads:
            thread.join()
        
        # Give drones time to stabilize in the formation
        time.sleep(2.0)
        
        # Print out current obstacle locations
        print("\n==== DYNAMIC OBSTACLE INFORMATION ====")
        obstacles = self.obstacle_tracker.get_obstacles()
        if not obstacles:
            print("No obstacles detected yet!")
        else:
            print(f"Number of dynamic obstacles: {len(obstacles)}")
            for i, (obs_id, obs_data) in enumerate(obstacles.items()):
                pos = obs_data["position"]
                vel = obs_data.get("velocity", [0, 0, 0])
                diameter = obs_data.get("diameter", 0.5)
                height = obs_data.get("height", 5.0)
                print(f"Obstacle {i+1} (ID: {obs_id}):")
                print(f"  Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
                print(f"  Velocity: [{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}]")
                print(f"  Size: diameter={diameter}m, height={height}m")
        print("======================================\n")
        
        print("All drones ready in zigzag formation")
        return True
    
    def get_distributed_positions(self, center_position, num_drones):
        """
        Get distributed positions around a center point.
        For multiple drones: arrange them in a zigzag pattern.
        """
        positions = []
        
        # Spacing between drones
        x_spacing = 1.5
        y_spacing = 0.25
        
        # Calculate the leftmost position (to center the formation)
        left_offset = -x_spacing * (num_drones - 1) / 2
        
        for i in range(num_drones):
            # Place drones in a zigzag pattern
            x_offset = left_offset + (i * x_spacing)
            y_offset = y_spacing if i % 2 == 0 else -y_spacing
            
            # Alternate altitude for better separation
            alt_variation = 0.0
            if i > 0:  # First drone at standard height, others with small variations
                alt_variation = 0.1 if i % 2 == 1 else -0.1  # Opposite of Y pattern for better separation
            
            pos = [
                center_position[0] + x_offset,      # X: horizontal spacing
                center_position[1] + y_offset,      # Y: zigzag offset
                center_position[2] + alt_variation  # Z: altitude with small variation
            ]
            
            # Ensure altitude is within the allowed range
            pos[2] = max(MIN_ALTITUDE, min(MAX_ALTITUDE, pos[2]))
            positions.append(pos)
            
            # Add debug message for position
            print(f"DEBUG: Position {i} in zigzag: X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f}")
            
            # Add debug message for altitude variation
            alt_diff = pos[2] - TAKE_OFF_HEIGHT
            if abs(alt_diff) > 0.01:  # If there's a meaningful difference
                print(f"DEBUG: Position {i} has altitude variation: {alt_diff:.2f}m (Z={pos[2]:.2f}m)")
        
        return positions
    
    def run_mission(self):
        """
        Main mission execution:
        1. Each drone plans coordinated paths using CBS
        2. Drones follow their paths while continuously checking for dynamic obstacles
        3. If an obstacle is detected on the path, the drones replan using CBS
        """
        print("Starting mission execution with CBS coordination")
        
        # Print current obstacle count
        obstacles = self.obstacle_tracker.get_obstacles()
        print(f"Current obstacle count before starting: {len(obstacles)}")
        
        # Get distributed end positions for the drones in a zigzag pattern
        end_positions = self.get_distributed_positions(self.end_position, len(self.drones))
        print("Created zigzag formation for end positions")
        
        # Get current positions of all drones
        start_positions = {}
        for drone_id, drone in self.drones.items():
            start_positions[drone_id] = self.get_drone_position(drone)
        
        # Prepare drone_data dictionary for CBS (mapping drone_id to (start, goal) tuple)
        drone_data = {}
        for i, drone_id in enumerate(self.drones.keys()):
            drone_data[drone_id] = (start_positions[drone_id], end_positions[i])
        
        # Use CBS to plan coordinated, conflict-free paths
        print("Planning initial conflict-free paths using CBS...")
        drone_paths = cbs_plan_multi_drone_paths(
            drone_data,
            self.obstacle_tracker.get_obstacles(),
            self.bounds
        )
        
        # Check if CBS planning succeeded
        if not drone_paths:
            print("CBS planning failed. Falling back to individual planning.")
            # Fall back to individual planning
            drone_paths = {}
            for i, (drone_id, drone) in enumerate(self.drones.items()):
                try:
                    path, _ = plan_rrt(
                        start_positions[drone_id],
                        end_positions[i],
                        self.obstacle_tracker.get_obstacles(),
                        self.bounds,
                        prediction_time=LOOKAHEAD_TIME
                    )
                    
                    if path:
                        drone_paths[drone_id] = path
                        print(f"{drone_id}: Initial path planned with {len(path)} waypoints")
                    else:
                        print(f"{drone_id}: Initial path planning failed!")
                except Exception as e:
                    print(f"{drone_id}: Error during path planning: {e}")
        else:
            print("CBS planning successful!")
            for drone_id, path in drone_paths.items():
                print(f"{drone_id}: Path planned with {len(path)} waypoints")
        
        # Track the current waypoint index for each drone
        drone_current_waypoints = {drone_id: 0 for drone_id in self.drones}
        drone_goal_reached = {drone_id: False for drone_id in self.drones}
        drone_active = {drone_id: drone_id in drone_paths for drone_id in self.drones}
        
        # Last replanning time
        last_replan_time = time.time()
        
        # Main execution loop
        try:
            while not all(drone_goal_reached.values()) and any(drone_active.values()):
                # Get current obstacle data once per iteration
                current_obstacles = self.obstacle_tracker.get_obstacles()
                current_time = time.time()
                
                # Check if we need to replan (based on REPLAN_FREQUENCY)
                need_global_replan = False
                if current_time - last_replan_time > (1.0 / REPLAN_FREQUENCY):
                    print("Checking if replanning is needed...")
                    last_replan_time = current_time
                    
                    # Dictionary to store command parameters for each active drone
                    drone_commands = {}
                    
                    # First collect all command parameters and check for replanning needs
                    for drone_id, drone in self.drones.items():
                        if not drone_active[drone_id] or drone_goal_reached[drone_id]:
                            continue
                        
                        if drone_id not in drone_paths:
                            continue
                        
                        try:
                            current_position = self.get_drone_position(drone)
                            path = drone_paths[drone_id]
                            waypoint_idx = drone_current_waypoints[drone_id]
                            
                            # If we've reached the final waypoint
                            if waypoint_idx >= len(path) - 1:
                                print(f"{drone_id}: Goal reached!")
                                drone_goal_reached[drone_id] = True
                                continue
                            
                            # Check current target waypoint
                            target_waypoint = path[waypoint_idx + 1]
                            
                            # Check if we've reached the current waypoint
                            distance_to_waypoint = compute_euclidean_distance(current_position, target_waypoint)
                            if distance_to_waypoint < 0.3:
                                # Move to next waypoint
                                drone_current_waypoints[drone_id] += 1
                                if drone_current_waypoints[drone_id] >= len(path) - 1:
                                    print(f"{drone_id}: Goal reached!")
                                    drone_goal_reached[drone_id] = True
                                    continue
                                target_waypoint = path[drone_current_waypoints[drone_id] + 1]
                            
                            # Check if the path to the next waypoint is collision-free
                            path_obstructed = False
                            for t in np.linspace(0, 1, 5):
                                interp = [
                                    current_position[0] + t * (target_waypoint[0] - current_position[0]),
                                    current_position[1] + t * (target_waypoint[1] - current_position[1]),
                                    current_position[2] + t * (target_waypoint[2] - current_position[2])
                                ]
                                if not is_state_valid(interp, current_obstacles, LOOKAHEAD_TIME):
                                    path_obstructed = True
                                    need_global_replan = True
                                    print(f"{drone_id}: Path obstructed, need to replan")
                                    break
                            
                            if not path_obstructed:
                                # Calculate yaw based on the target waypoint
                                yaw = math.atan2(
                                    target_waypoint[1] - current_position[1],
                                    target_waypoint[0] - current_position[0]
                                )
                                
                                # Store command parameters for this drone
                                drone_commands[drone_id] = {
                                    'drone': drone,
                                    'target_waypoint': target_waypoint,
                                    'yaw': yaw
                                }
                            
                        except Exception as e:
                            print(f"{drone_id}: Error during mission execution: {e}")
                            drone_active[drone_id] = False
                
                    # If we need to replan, use CBS
                    if need_global_replan:
                        print("Replanning all paths using CBS...")
                        
                        # Update start positions to current positions
                        for drone_id, drone in self.drones.items():
                            if drone_active[drone_id] and not drone_goal_reached[drone_id]:
                                start_positions[drone_id] = self.get_drone_position(drone)
                        
                        # Use CBS to replan all paths
                        active_drones = {drone_id: drone for drone_id, drone in self.drones.items() 
                                        if drone_active[drone_id] and not drone_goal_reached[drone_id]}
                        
                        if active_drones:
                            # Only consider active drones for replanning
                            # Prepare drone_data dictionary for active drones
                            active_drone_data = {}
                            for drone_id in active_drones:
                                active_drone_data[drone_id] = (
                                    start_positions[drone_id],
                                    end_positions[list(self.drones.keys()).index(drone_id)]
                                )
                            
                            new_paths = cbs_plan_multi_drone_paths(
                                active_drone_data,
                                current_obstacles,
                                self.bounds
                            )
                            
                            if new_paths:
                                print("CBS replanning successful")
                                # Update paths and reset waypoint indices
                                for drone_id, path in new_paths.items():
                                    drone_paths[drone_id] = path
                                    drone_current_waypoints[drone_id] = 0
                            else:
                                print("CBS replanning failed, continuing with current paths")
                                
                            # Skip command execution this cycle to give time for replanning
                            drone_commands = {}
                    
                    # Execute all drone commands concurrently using threads
                    if drone_commands:
                        threads = []
                        for drone_id, cmd in drone_commands.items():
                            thread = threading.Thread(
                                target=cmd['drone'].go_to.go_to_point_with_yaw,
                                args=(cmd['target_waypoint'],),
                                kwargs={'angle': cmd['yaw'], 'speed': SPEED},
                                daemon=True
                            )
                            thread.start()
                            threads.append(thread)
                        
                        # Optionally wait a brief moment to ensure all commands are dispatched
                        for t in threads:
                            t.join(timeout=0.01)
                
                # Sleep to avoid busy waiting
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("Mission interrupted!")
            return False
        
        # Check how many drones reached their goals
        successful_drones = sum(1 for reached in drone_goal_reached.values() if reached)
        print(f"{successful_drones} out of {len(self.drones)} drones reached their goals")
        
        return successful_drones > 0  # Consider mission successful if at least one drone reached its goal
    
    def end_mission(self):
        """End the mission: land all drones simultaneously and clean up"""
        # Command all drones to land simultaneously using threads
        print("Commanding all drones to land simultaneously...")
        land_threads = []
        for drone_id, drone in self.drones.items():
            thread = threading.Thread(
                target=drone.land,
                kwargs={'speed': LAND_SPEED},
                daemon=True
            )
            thread.start()
            land_threads.append(thread)
        
        # Wait for all land commands to be sent
        for thread in land_threads:
            thread.join()
        
        # Wait for landing to complete
        print("Waiting for all drones to land...")
        time.sleep(5.0)
        
        # Set to manual mode using threads
        manual_threads = []
        for drone_id, drone in self.drones.items():
            thread = threading.Thread(target=drone.manual, daemon=True)
            thread.start()
            manual_threads.append(thread)
        
        # Wait for all manual mode commands to complete
        for thread in manual_threads:
            thread.join()
        
        # Shutdown ROS nodes
        self.obstacle_tracker.destroy_node()
        rclpy.shutdown()
        
        print("Mission completed successfully")
        return True
    
    def shutdown(self):
        """Clean shutdown of all interfaces"""
        for drone_id, drone in self.drones.items():
            drone.shutdown()
        
        if hasattr(self, 'obstacle_tracker'):
            self.obstacle_tracker.destroy_node()
        
        if hasattr(self, 'position_node'):
            self.position_node.destroy_node()
        
        try:
            rclpy.shutdown()
        except:
            pass

# --------------------------
# Main
# --------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Multi-drone mission with dynamic obstacle avoidance for Stage 4'
    )
    parser.add_argument('-s', '--scenario', type=str, default='scenarios/scenario1_stage4.yaml',
                        help='Path to scenario YAML file')
    parser.add_argument('-n', '--namespace', type=str, nargs='+', 
                        help='Namespace/ID for the drones (e.g., -n drone0 drone1 drone2 drone3 drone4)')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Enable verbose output')
    parser.add_argument('-t', '--use_sim_time', action='store_true', default=True,
                        help='Use simulation time')
    
    args = parser.parse_args()
    scenario_file = args.scenario
    verbosity = args.verbose
    use_sim_time = args.use_sim_time
    
    # Create drone IDs either from command line or default
    if args.namespace:
        drone_ids = args.namespace
        NUM_DRONES = len(drone_ids)
    else:
        drone_ids = [f'drone{i}' for i in range(NUM_DRONES)]
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    print(f'Running Stage 4 mission for {NUM_DRONES} drones using scenario {scenario_file}')
    print(f'Drone namespaces: {drone_ids}')
    
    # Initialize and run mission
    mission = MultiDroneMission(drone_ids, scenario_file, use_sim_time, verbosity)
    
    try:
        # Start all drones
        success = mission.start_mission()
        
        if success:
            # Run the main mission
            success = mission.run_mission()
        
        if success:
            # End mission and land drones
            mission.end_mission()
        
    except Exception as e:
        print(f"Mission failed with error: {e}")
        success = False
    finally:
        # Ensure clean shutdown
        mission.shutdown()
    
    print("Mission script completed")
    exit(0 if success else 1) 