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
SPEED = 3.0                # m/s during flight 
LAND_SPEED = 0.5           # m/s for landing 

# Obstacle avoidance parameters 
SAFETY_MARGIN = 0.2        # Additional margin (in meters) added around each obstacle 

# Replanning parameters
REPLAN_FREQUENCY = 10.0     # How often to replan (in Hz)
LOOKAHEAD_TIME = 2.0       # How far ahead to predict obstacle positions (in seconds)

# A* planner parameters 
PLANNING_TIME_LIMIT = 1.0         # Time limit for planning each segment (in seconds) 
ASTAR_RESOLUTION = 0.5           # Default grid resolution for A* (in meters)

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
def plan_astar(start, goal, dynamic_obstacles, bounds, resolution=ASTAR_RESOLUTION, 
              planning_time_limit=PLANNING_TIME_LIMIT, prediction_time=0.0):
    """
    Plan a collision-free path between start and goal using a grid-based A* search,
    taking into account the predicted positions of dynamic obstacles.
    """
    # Debug info
    print(f"Planning from {start} to {goal}")
    print(f"Bounds: {bounds}")
    print(f"Number of obstacles: {len(dynamic_obstacles)}")
    print(f"Using lookahead time: {prediction_time}s to predict obstacle positions")
    
    start_time = time.time()
    
    # For prediction, we create a deep copy of obstacles and update positions
    # but we'll use the trajectory-based collision checking in our search
    predicted_obstacles = {}
    for obs_id, obs_data in dynamic_obstacles.items():
        predicted_obstacles[obs_id] = obs_data.copy()
    
    # COMMENTED OUT: Always use A* planning with 0.5m resolution, even for clear paths
    # if not predicted_obstacles or is_path_clear(start, goal, predicted_obstacles):
    #     print("Direct path is clear, returning straight line")
    #     return [start, goal], time.time() - start_time
    
    low_x, low_y, low_z = bounds["low"][0], bounds["low"][1], bounds["low"][2]
    high_x, high_y, high_z = bounds["high"][0], bounds["high"][1], bounds["high"][2]

    def world_to_grid(point):
        i = int(round((point[0] - low_x) / resolution))
        j = int(round((point[1] - low_y) / resolution))
        k = int(round((point[2] - low_z) / resolution))
        return (i, j, k)
    
    def grid_to_world(cell):
        x = low_x + cell[0] * resolution
        y = low_y + cell[1] * resolution
        z = low_z + cell[2] * resolution
        return (x, y, z)
    
    try:
        start_cell = world_to_grid(start)
        goal_cell = world_to_grid(goal)
    except Exception as e:
        print(f"Error converting coordinates: {e}")
        return None, time.time() - start_time

    # Calculate grid dimensions
    try:
        max_i = int(round((high_x - low_x) / resolution))
        max_j = int(round((high_y - low_y) / resolution))
        max_k = int(round((high_z - low_z) / resolution))
        print(f"Grid dimensions: {max_i}x{max_j}x{max_k}")
    except Exception as e:
        print(f"Error calculating grid dimensions: {e}")
        return None, time.time() - start_time
    
    from heapq import heappush, heappop
    open_set = []
    heappush(open_set, (0, start_cell))
    came_from = {}
    cost_so_far = {start_cell: 0}
    
    max_iterations = 10000  # Prevent infinite loops
    iterations = 0
    
    found = False
    while open_set and iterations < max_iterations:
        iterations += 1
        
        if time.time() - start_time > planning_time_limit:
            print("A* planning time limit exceeded")
            break
        
        current_priority, current = heappop(open_set)
        
        # Check if current cell is goal
        if current == goal_cell:
            found = True
            break
        
        # Regular A* expansion - only using 6 directions (not diagonals)
        directions = [
            (1, 0, 0), (-1, 0, 0),  # x directions
            (0, 1, 0), (0, -1, 0),  # y directions
            (0, 0, 1), (0, 0, -1)   # z directions
        ]
        
        for dx, dy, dz in directions:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
            
            # Check bounds
            if (neighbor[0] < 0 or neighbor[0] > max_i or
                neighbor[1] < 0 or neighbor[1] > max_j or
                neighbor[2] < 0 or neighbor[2] > max_k):
                continue
            
            # Check collision - using the trajectory-based collision checking
            neighbor_world = grid_to_world(neighbor)
            if not is_state_valid(neighbor_world, predicted_obstacles, prediction_time):
                continue
            
            # Calculate cost
            move_cost = 1.0  # Use uniform cost for cardinal directions
            new_cost = cost_so_far[current] + move_cost
            
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                # Use Manhattan distance for heuristic (faster than Euclidean)
                heuristic = (abs(neighbor[0] - goal_cell[0]) + 
                            abs(neighbor[1] - goal_cell[1]) + 
                            abs(neighbor[2] - goal_cell[2]))
                priority = new_cost + heuristic
                heappush(open_set, (priority, neighbor))
                came_from[neighbor] = current
    
    planning_time = time.time() - start_time
    print(f"A* search completed in {planning_time:.3f}s after {iterations} iterations, found={found}")
    
    if not found:
        return None, planning_time
    
    # Reconstruct path
    path_cells = []
    current = goal_cell
    while current != start_cell:
        path_cells.append(current)
        current = came_from[current]
    path_cells.append(start_cell)
    path_cells.reverse()
    
    path_states = [grid_to_world(cell) for cell in path_cells]
    
    return path_states, planning_time

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
class Constraint:
    """Represents a spatio-temporal constraint for CBS"""
    def __init__(self, drone_id, position, timestep):
        self.drone_id = drone_id  # The drone this constraint applies to
        self.position = position  # The position (x,y,z) that is forbidden
        self.timestep = timestep  # The timestep when the position is forbidden
    
    def __str__(self):
        return f"Drone {self.drone_id} cannot be at {self.position} at timestep {self.timestep}"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        if not isinstance(other, Constraint):
            return False
        return (self.drone_id == other.drone_id and 
                self.position == other.position and 
                self.timestep == other.timestep)
    
    def __hash__(self):
        return hash((self.drone_id, tuple(self.position), self.timestep))

class CBSNode:
    """A node in the CBS constraint tree"""
    def __init__(self, constraints=None, paths=None, cost=0, parent=None):
        self.constraints = constraints or []  # List of constraints
        self.paths = paths or {}  # Dictionary mapping drone_id to path
        self.cost = cost  # Cost of the solution (e.g., max path length)
        self.parent = parent  # Parent node in the constraint tree
    
    def __lt__(self, other):
        """For priority queue comparison, nodes with lower cost are prioritized"""
        return self.cost < other.cost

def detect_conflicts(paths, safety_distance=0.5):
    """
    Detect spatio-temporal conflicts between drone paths.
    Returns a list of conflicts, each represented as a tuple (drone1_id, drone2_id, position, timestep).
    """
    conflicts = []
    
    # Convert paths to time-indexed format {drone_id: {timestep: position}}
    time_indexed_paths = {}
    max_path_length = 0
    
    for drone_id, path in paths.items():
        time_indexed_paths[drone_id] = {}
        for i, pos in enumerate(path):
            time_indexed_paths[drone_id][i] = pos
        max_path_length = max(max_path_length, len(path))
    
    # Check for conflicts at each timestep
    drone_ids = list(paths.keys())
    for t in range(max_path_length):
        for i in range(len(drone_ids)):
            for j in range(i + 1, len(drone_ids)):
                drone1_id = drone_ids[i]
                drone2_id = drone_ids[j]
                
                # Skip if either drone doesn't have a position at this timestep
                if t not in time_indexed_paths[drone1_id] or t not in time_indexed_paths[drone2_id]:
                    continue
                
                pos1 = time_indexed_paths[drone1_id][t]
                pos2 = time_indexed_paths[drone2_id][t]
                
                # Check for collision (distance less than safety threshold)
                distance = compute_euclidean_distance(pos1, pos2)
                if distance < safety_distance:
                    midpoint = [(pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2, (pos1[2] + pos2[2]) / 2]
                    conflicts.append((drone1_id, drone2_id, midpoint, t))
    
    return conflicts

def time_indexed_astar(start, goal, drone_id, constraints, dynamic_obstacles, bounds, 
                      resolution=ASTAR_RESOLUTION, planning_time_limit=PLANNING_TIME_LIMIT, 
                      prediction_time=LOOKAHEAD_TIME):
    """
    A* path planning with time-indexed constraints.
    Similar to the original A* but respects time-indexed constraints from CBS.
    """
    print(f"Planning for drone {drone_id} with {len(constraints)} constraints")
    
    start_time = time.time()
    
    # Filter constraints for this drone
    drone_constraints = {(tuple(c.position), c.timestep) for c in constraints if c.drone_id == drone_id}
    
    # Create deep copy of obstacles
    predicted_obstacles = {}
    for obs_id, obs_data in dynamic_obstacles.items():
        predicted_obstacles[obs_id] = obs_data.copy()
    
    # Grid conversion functions
    low_x, low_y, low_z = bounds["low"][0], bounds["low"][1], bounds["low"][2]
    high_x, high_y, high_z = bounds["high"][0], bounds["high"][1], bounds["high"][2]
    
    def world_to_grid(point):
        i = int(round((point[0] - low_x) / resolution))
        j = int(round((point[1] - low_y) / resolution))
        k = int(round((point[2] - low_z) / resolution))
        return (i, j, k)
    
    def grid_to_world(cell):
        x = low_x + cell[0] * resolution
        y = low_y + cell[1] * resolution
        z = low_z + cell[2] * resolution
        return (x, y, z)
    
    # Convert start and goal to grid coordinates
    try:
        start_cell = world_to_grid(start)
        goal_cell = world_to_grid(goal)
    except Exception as e:
        print(f"Error converting coordinates: {e}")
        return None, time.time() - start_time
    
    # Calculate grid dimensions
    try:
        max_i = int(round((high_x - low_x) / resolution))
        max_j = int(round((high_y - low_y) / resolution))
        max_k = int(round((high_z - low_z) / resolution))
    except Exception as e:
        print(f"Error calculating grid dimensions: {e}")
        return None, time.time() - start_time
    
    # A* search with time index
    from heapq import heappush, heappop
    
    # State now includes timestep: (i, j, k, t)
    open_set = []
    # Priority, state, timestep
    heappush(open_set, (0, start_cell, 0))
    
    # Track states visited at each timestep
    came_from = {}  # (cell, timestep) -> (previous_cell, previous_timestep)
    cost_so_far = {(start_cell, 0): 0}
    
    max_iterations = 10000  # Prevent infinite loops
    iterations = 0
    
    max_timestep = 100  # Limit search to a reasonable number of timesteps
    
    found = False
    while open_set and iterations < max_iterations:
        iterations += 1
        
        if time.time() - start_time > planning_time_limit:
            print("Time-indexed A* planning time limit exceeded")
            break
        
        current_priority, current_cell, current_time = heappop(open_set)
        
        # Check if current cell is goal
        if current_cell == goal_cell:
            found = True
            break
        
        # Regular A* expansion with 6 directions (and staying in place)
        directions = [
            (1, 0, 0), (-1, 0, 0),  # x directions
            (0, 1, 0), (0, -1, 0),  # y directions
            (0, 0, 1), (0, 0, -1),  # z directions
            (0, 0, 0)  # Stay in place (wait)
        ]
        
        for dx, dy, dz in directions:
            next_cell = (current_cell[0] + dx, current_cell[1] + dy, current_cell[2] + dz)
            next_time = current_time + 1
            
            # Skip if exceeding max timestep
            if next_time > max_timestep:
                continue
            
            # Check bounds
            if (next_cell[0] < 0 or next_cell[0] > max_i or
                next_cell[1] < 0 or next_cell[1] > max_j or
                next_cell[2] < 0 or next_cell[2] > max_k):
                continue
            
            # Convert to world coordinates for collision checking
            next_world = grid_to_world(next_cell)
            
            # Check if this violates a constraint
            if (tuple(next_world), next_time) in drone_constraints:
                continue
            
            # Check collision with obstacles - adjusting prediction time based on current timestep
            adjusted_pred_time = min(prediction_time, next_time * 0.2)  # Scale prediction with timestep
            if not is_state_valid(next_world, predicted_obstacles, adjusted_pred_time):
                continue
            
            # Calculate cost
            move_cost = 1.0  # Use uniform cost for all moves
            new_cost = cost_so_far[(current_cell, current_time)] + move_cost
            
            if (next_cell, next_time) not in cost_so_far or new_cost < cost_so_far[(next_cell, next_time)]:
                cost_so_far[(next_cell, next_time)] = new_cost
                
                # Use Manhattan distance plus time difference as heuristic
                heuristic = (abs(next_cell[0] - goal_cell[0]) + 
                            abs(next_cell[1] - goal_cell[1]) + 
                            abs(next_cell[2] - goal_cell[2]))
                
                priority = new_cost + heuristic
                heappush(open_set, (priority, next_cell, next_time))
                came_from[(next_cell, next_time)] = (current_cell, current_time)
    
    planning_time = time.time() - start_time
    print(f"Time-indexed A* completed in {planning_time:.3f}s after {iterations} iterations, found={found}")
    
    if not found:
        return None, planning_time
    
    # Reconstruct path (including timesteps)
    path_cells = []
    timesteps = []
    current = (current_cell, current_time)
    
    while current != (start_cell, 0):
        path_cells.append(current[0])
        timesteps.append(current[1])
        current = came_from[current]
    
    path_cells.append(start_cell)
    timesteps.append(0)
    
    # Reverse to get start-to-goal order
    path_cells.reverse()
    timesteps.reverse()
    
    # Convert back to world coordinates
    path_states = [grid_to_world(cell) for cell in path_cells]
    
    # Return path with timestep information
    return path_states, planning_time

def cbs_plan_multi_drone_paths(drones, start_positions, goal_positions, dynamic_obstacles, bounds):
    """
    Plan paths for multiple drones using Conflict-Based Search (CBS).
    Returns a dictionary mapping drone IDs to conflict-free paths.
    """
    print("Planning multi-drone paths using Conflict-Based Search (CBS)")
    start_time = time.time()
    
    # Initialize the root node of the constraint tree
    root = CBSNode(constraints=[], paths={}, cost=0)
    
    # Initial individual planning for each drone without constraints
    all_paths_valid = True
    for drone_id, drone in drones.items():
        idx = list(drones.keys()).index(drone_id)
        start = start_positions[drone_id]
        goal = goal_positions[idx]
        
        # Plan the initial path using time-indexed A*
        path, _ = time_indexed_astar(
            start, goal, drone_id, [], dynamic_obstacles, bounds
        )
        
        if path:
            root.paths[drone_id] = path
        else:
            print(f"Failed to find initial path for drone {drone_id}")
            all_paths_valid = False
    
    if not all_paths_valid:
        print("CBS failed: Could not find initial paths for all drones")
        return {}
    
    # Calculate the initial node cost (max path length)
    root.cost = max(len(path) for path in root.paths.values())
    
    # Use a priority queue for the CBS constraint tree
    from heapq import heappush, heappop
    open_list = [root]
    
    max_iterations = 100  # Limit the number of iterations
    iterations = 0
    
    while open_list and iterations < max_iterations:
        iterations += 1
        
        # Get the node with the lowest cost
        current = heappop(open_list)
        
        # Check for conflicts in the current solution
        conflicts = detect_conflicts(current.paths)
        
        if not conflicts:
            # Solution found!
            planning_time = time.time() - start_time
            print(f"CBS found conflict-free paths in {planning_time:.3f}s after {iterations} iterations")
            return current.paths
        
        # Pick the first conflict to resolve
        conflict = conflicts[0]
        drone1_id, drone2_id, conflict_pos, conflict_time = conflict
        
        # Create two new constraint nodes, one for each drone involved in the conflict
        for constrained_drone in [drone1_id, drone2_id]:
            # Create a new constraint
            new_constraint = Constraint(
                drone_id=constrained_drone,
                position=conflict_pos,
                timestep=conflict_time
            )
            
            # Create a new node with all constraints from the parent plus the new one
            new_constraints = current.constraints.copy()
            new_constraints.append(new_constraint)
            
            new_node = CBSNode(
                constraints=new_constraints,
                paths=current.paths.copy(),
                parent=current
            )
            
            # Replan for the constrained drone
            idx = list(drones.keys()).index(constrained_drone)
            start = start_positions[constrained_drone]
            goal = goal_positions[idx]
            
            new_path, _ = time_indexed_astar(
                start, goal, constrained_drone, new_constraints, 
                dynamic_obstacles, bounds
            )
            
            # If replanning succeeded, update the path and add the node to the open list
            if new_path:
                new_node.paths[constrained_drone] = new_path
                new_node.cost = max(len(path) for path in new_node.paths.values())
                heappush(open_list, new_node)
            else:
                print(f"Replanning failed for drone {constrained_drone} with new constraint")
    
    # If we reach here, CBS failed to find a solution
    print(f"CBS failed to find conflict-free paths after {iterations} iterations")
    
    # Return the best solution found so far (might still have conflicts)
    if open_list:
        best_node = min(open_list, key=lambda node: node.cost)
        return best_node.paths
    return {}

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
        
        # Initialize drones and position subscribers
        for drone_id in drone_ids:
            self.drones[drone_id] = DroneInterface(drone_id=drone_id, use_sim_time=use_sim_time, verbose=verbose)
            
            # Initialize position with default values based on start position
            idx = drone_ids.index(drone_id)
            if idx == 0:
                # First drone at center
                self.drone_positions[drone_id] = self.start_position.copy()
            else:
                # Other drones in a square formation
                offset = 1.0
                offsets = [
                    [offset, 0, 0],   # right
                    [0, offset, 0],   # forward
                    [-offset, 0, 0],  # left
                    [0, -offset, 0],  # back
                ]
                offset_idx = (idx - 1) % 4
                self.drone_positions[drone_id] = [
                    self.start_position[0] + offsets[offset_idx][0],
                    self.start_position[1] + offsets[offset_idx][1],
                    self.start_position[2]
                ]
            
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
        
        print(f"Multi-drone mission initialized with {len(drone_ids)} drones")
    
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
                0.0
            ],
            "high": [
                stage_max_x + 2.0,  # Add safety margin
                stage_max_y + 2.0,  # Add safety margin
                TAKE_OFF_HEIGHT * 2
            ]
        }
        
        # Get obstacle parameters
        self.obstacle_diameter = self.stage4.get("obstacle_diameter", 0.5)
        self.obstacle_height = self.stage4.get("obstacle_height", 5.0)
        
        print(f"Stage 4 setup: Start={self.start_position}, End={self.end_position}")
        print(f"Stage center: {self.stage_center}, Stage size: {stage_size}")
        print(f"Stage boundaries: X=[{stage_min_x:.1f}, {stage_max_x:.1f}], Y=[{stage_min_y:.1f}, {stage_max_y:.1f}]")
        print(f"A* bounds with safety margin: X=[{self.bounds['low'][0]:.1f}, {self.bounds['high'][0]:.1f}], Y=[{self.bounds['low'][1]:.1f}, {self.bounds['high'][1]:.1f}]")
    
    def start_mission(self):
        """Start the mission: all drones arm, offboard, and takeoff simultaneously"""
        print("Arming all drones...")
        # Arm all drones first
        for drone_id, drone in self.drones.items():
            print(f"Arming {drone_id}")
            drone.arm()
        
        # Small delay after arming
        time.sleep(0.5)
        
        # Set all drones to offboard mode
        print("Setting all drones to offboard mode...")
        for drone_id, drone in self.drones.items():
            drone.offboard()
        
        # Small delay after offboard
        time.sleep(0.5)
        
        # Command all drones to takeoff simultaneously
        print("Commanding all drones to takeoff simultaneously...")
        for drone_id, drone in self.drones.items():
            drone.takeoff(height=TAKE_OFF_HEIGHT, speed=TAKE_OFF_SPEED)
        
        # Give time for all drones to reach takeoff height
        print("Waiting for all drones to reach takeoff height...")
        time.sleep(5.0)
        
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
        
        print("All drones ready")
        return True
    
    def get_distributed_positions(self, center_position, num_drones):
        """
        Get distributed positions around a center point.
        For 5 drones: one at center, four in a small square around it.
        """
        positions = [center_position.copy()]
        
        # Offset distance for distributed positions
        offset = 1.0
        
        if num_drones > 1:
            # Add positions in a small square pattern
            offsets = [
                [offset, 0, 0],   # right
                [0, offset, 0],   # forward
                [-offset, 0, 0],  # left
                [0, -offset, 0],  # back
            ]
            
            for i in range(1, min(num_drones, 5)):
                pos = [
                    center_position[0] + offsets[i-1][0],
                    center_position[1] + offsets[i-1][1],
                    center_position[2] + offsets[i-1][2]
                ]
                positions.append(pos)
        
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
        
        # Get distributed end positions for the drones
        end_positions = self.get_distributed_positions(self.end_position, len(self.drones))
        
        # Get current positions of all drones
        start_positions = {}
        for drone_id, drone in self.drones.items():
            start_positions[drone_id] = self.get_drone_position(drone)
        
        # Use CBS to plan coordinated, conflict-free paths
        print("Planning initial conflict-free paths using CBS...")
        drone_paths = cbs_plan_multi_drone_paths(
            self.drones,
            start_positions,
            end_positions,
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
                    path, _ = plan_astar(
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
                            new_paths = cbs_plan_multi_drone_paths(
                                active_drones,
                                {d_id: start_positions[d_id] for d_id in active_drones},
                                [end_positions[list(self.drones.keys()).index(d_id)] for d_id in active_drones],
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
        # Command all drones to land simultaneously
        print("Commanding all drones to land simultaneously...")
        for drone_id, drone in self.drones.items():
            drone.land(speed=LAND_SPEED)
        
        # Wait for landing to complete
        print("Waiting for all drones to land...")
        time.sleep(5.0)
        
        # Set to manual mode
        for drone_id, drone in self.drones.items():
            drone.manual()
        
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