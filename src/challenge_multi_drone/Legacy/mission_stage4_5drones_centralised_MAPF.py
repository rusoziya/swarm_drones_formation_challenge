#!/usr/bin/env python3
import argparse
import math
import sys
import random
import time
import threading
import rclpy
import yaml
import os
import numpy as np
import heapq
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from as2_msgs.msg import YawMode, BehaviorStatus
from as2_python_api.drone_interface import DroneInterface
from as2_python_api.behavior_actions.behavior_handler import BehaviorHandler
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import PoseStamped, Point, Vector3
from typing import List, Tuple, Dict, Set, Optional
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))

# Grid cell size for path planning (in meters)
GRID_RESOLUTION = 0.2
# Safety radius for collision detection (in meters)
SAFETY_RADIUS = 0.7
# Time between path replanning (in seconds)
REPLAN_INTERVAL = 0.3
# Maximum planning horizon (in seconds)
PLANNING_HORIZON = 3.0
# How far into the future to predict obstacle positions (in seconds)
PREDICTION_HORIZON = 2.0
# Maximum iterations for path planning
MAX_ITERATIONS = 300
# Default drone speed (in m/s)
DEFAULT_SPEED = 0.6

class GridMapState:
    """Represents a grid map state for pathfinding algorithms."""
    def __init__(self, x: int, y: int, time_step: int = 0):
        self.x = x
        self.y = y
        self.time_step = time_step
        
    def __eq__(self, other):
        if not isinstance(other, GridMapState):
            return False
        return self.x == other.x and self.y == other.y and self.time_step == other.time_step
    
    def __hash__(self):
        return hash((self.x, self.y, self.time_step))
    
    def __lt__(self, other):
        # For priority queue
        return (self.x, self.y, self.time_step) < (other.x, other.y, other.time_step)
    
    def __str__(self):
        return f"({self.x}, {self.y}, t={self.time_step})"

class DynamicObstacle:
    """Represents a dynamic obstacle with position and velocity."""
    def __init__(self, obstacle_id: str, position: Point, timestamp: float = None):
        self.id = obstacle_id
        self.position = position
        self.velocity = Vector3(x=0.0, y=0.0, z=0.0)
        self.radius = 0.5  # Default radius in meters
        self.prev_position = None
        self.prev_timestamp = None
        self.timestamp = timestamp or time.time()
        
    def update_position(self, position: Point, timestamp: float = None):
        """Update position and calculate velocity."""
        timestamp = timestamp or time.time()
        self.prev_position = self.position
        self.prev_timestamp = self.timestamp
        self.position = position
        self.timestamp = timestamp
        
        # Calculate velocity if we have previous position
        if self.prev_position and (self.timestamp - self.prev_timestamp) > 0.001:
            dt = self.timestamp - self.prev_timestamp
            self.velocity.x = (self.position.x - self.prev_position.x) / dt
            self.velocity.y = (self.position.y - self.prev_position.y) / dt
            self.velocity.z = (self.position.z - self.prev_position.z) / dt
    
    def predict_position(self, future_time: float) -> Point:
        """Predict position at a future time based on current velocity."""
        dt = future_time - self.timestamp
        return Point(
            x=self.position.x + self.velocity.x * dt,
            y=self.position.y + self.velocity.y * dt,
            z=self.position.z + self.velocity.z * dt
        )

class FormationController:
    """Controls formation flying for multiple drones."""
    def __init__(self, formation_type="V", spacing=1.0, leader_id="drone1"):
        self.formation_type = formation_type.lower()
        self.spacing = spacing
        self.leader_id = leader_id
        self.drone_positions = {}  # Current positions of drones
        self.formation_positions = {}  # Desired positions in formation
        self.avoiding_obstacles = set()  # Drones currently avoiding obstacles
        self.in_formation = False
        self.formation_heading = [0, 1]  # Default heading (north)
        self.all_in_formation = False
        self.ready_drones = set()  # Drones ready to move as a formation
        self.move_as_formation = False  # Flag to indicate when to start moving
        self.intermediate_goals = {}  # Intermediate goals for each drone
        self.current_formation_waypoint_index = 0  # Current waypoint index for formation movement
        self.formation_path = []  # Path for the entire formation to follow
        
    def register_drone(self, namespace, initial_position, formation_position):
        """Register a drone with the formation controller."""
        self.drone_positions[namespace] = initial_position
        self.formation_positions[namespace] = formation_position
        print(f"Registered drone {namespace} with formation position {formation_position}")
        
    def update_drone_position(self, drone_id, position):
        """Update the position of a drone in the formation."""
        self.drone_positions[drone_id] = position
        
    def is_avoiding_obstacles(self, drone_id):
        """Check if a drone is currently avoiding obstacles."""
        return drone_id in self.avoiding_obstacles
        
    def set_obstacle_avoidance(self, drone_id, is_avoiding):
        """Set whether a drone is currently avoiding obstacles."""
        if is_avoiding:
            self.avoiding_obstacles.add(drone_id)
        elif drone_id in self.avoiding_obstacles:
            self.avoiding_obstacles.remove(drone_id)

    def mark_drone_ready(self, drone_id):
        """Mark a drone as ready to move in formation."""
        self.ready_drones.add(drone_id)
        # Check if all drones are ready
        if len(self.ready_drones) == len(self.drone_positions):
            self.move_as_formation = True
            print(f"All drones are in formation and ready to move together!")
            
    def reset_ready_state(self):
        """Reset the ready state of all drones."""
        self.ready_drones.clear()
        self.move_as_formation = False
    
    def normalize_vector(self, vector):
        """Normalize a vector to unit length."""
        magnitude = math.sqrt(vector[0]**2 + vector[1]**2)
        if magnitude < 0.0001:  # Avoid division by zero
            return [0, 1]
        return [vector[0]/magnitude, vector[1]/magnitude]
    
    def get_perpendicular_vector(self, vector):
        """Get a vector perpendicular to the given vector."""
        return [-vector[1], vector[0]]
    
    def generate_formation_path(self, leader_position, goal_position, num_waypoints=5):
        """Generate a path for the entire formation to follow."""
        # Clear existing path
        self.formation_path = []
        
        # Calculate vector from leader to goal
        dx = goal_position[0] - leader_position[0]
        dy = goal_position[1] - leader_position[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        # If too close to goal, don't generate intermediate waypoints
        if distance < 3.0:
            self.formation_path = [leader_position, goal_position]
            return self.formation_path
            
        # Generate intermediate waypoints
        for i in range(num_waypoints + 1):
            ratio = i / num_waypoints
            x = leader_position[0] + dx * ratio
            y = leader_position[1] + dy * ratio
            self.formation_path.append([x, y])
            
        return self.formation_path
    
    def get_next_formation_waypoint(self):
        """Get the next waypoint for the formation to move to."""
        if not self.formation_path or self.current_formation_waypoint_index >= len(self.formation_path):
            return None
            
        return self.formation_path[self.current_formation_waypoint_index]
    
    def advance_formation_waypoint(self):
        """Move to the next formation waypoint."""
        if self.current_formation_waypoint_index < len(self.formation_path) - 1:
            self.current_formation_waypoint_index += 1
            return True
        return False
        
    def calculate_formation_positions(self, leader_position, goal_position):
        """Calculate the desired positions for each drone in the formation."""
        # Calculate formation heading (from leader toward goal)
        dx = goal_position[0] - leader_position[0]
        dy = goal_position[1] - leader_position[1]
        if abs(dx) < 0.01 and abs(dy) < 0.01:
            # If leader is at the goal, use default heading
            heading = self.formation_heading
        else:
            # Normalize the heading vector
            heading = self.normalize_vector([dx, dy])
            self.formation_heading = heading
            
        # Get perpendicular vector for V formation (90 degrees to the heading)
        perp = self.get_perpendicular_vector(heading)
        
        # Calculate formation positions based on formation type
        result = {}
        
        if self.formation_type == "v":
            # Leader at the front
            result[self.leader_id] = leader_position
            
            # Calculate positions for other drones based on their defined formation positions
            for drone_id, offset in self.formation_positions.items():
                if drone_id == self.leader_id:
                    continue
                    
                # Apply formation offsets in the direction of travel
                offset_x = -heading[0] * offset[1] + perp[0] * offset[0]
                offset_y = -heading[1] * offset[1] + perp[1] * offset[0]
                
                result[drone_id] = [
                    leader_position[0] + offset_x * self.spacing,
                    leader_position[1] + offset_y * self.spacing
                ]
        
        return result
    
    def get_formation_position(self, drone_id, leader_position, goal_position):
        """Get the desired position for a drone in the formation."""
        formation_positions = self.calculate_formation_positions(leader_position, goal_position)
        if drone_id in formation_positions:
            return formation_positions[drone_id]
        return leader_position  # Default to leader position if unknown drone
    
    def should_maintain_formation(self):
        """Determine if drones should maintain formation or go directly to goals."""
        # If any drone is avoiding obstacles, break formation
        if self.avoiding_obstacles:
            return False
        
        # Not enough drones reporting positions
        if len(self.drone_positions) < 3:
            return False
            
        return self.all_in_formation and self.move_as_formation
        
    def check_formation_status(self):
        """Check if all drones are in their formation positions."""
        if not self.drone_positions or not self.formation_positions:
            return False
            
        # Get leader position
        if self.leader_id not in self.drone_positions:
            return False
            
        leader_pos = self.drone_positions[self.leader_id]
        
        # Count drones in formation
        drones_in_formation = 0
        total_drones = len(self.drone_positions)
        
        for drone_id, position in self.drone_positions.items():
            if drone_id in self.avoiding_obstacles:
                continue
                
            # Get desired formation position
            formation_pos = self.get_formation_position(drone_id, leader_pos, [0, 0])
            
            # Calculate distance to formation position
            dx = position[0] - formation_pos[0]
            dy = position[1] - formation_pos[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Check if drone is in position (within tolerance)
            if distance < 0.5:  # 50cm tolerance
                drones_in_formation += 1
                
        # Check if all drones are in formation
        formation_percentage = drones_in_formation / total_drones
        self.all_in_formation = formation_percentage > 0.8  # 80% of drones in formation
        
        return self.all_in_formation
        
    def set_intermediate_goal(self, drone_id, goal):
        """Set an intermediate goal for a specific drone."""
        self.intermediate_goals[drone_id] = goal
        
    def get_intermediate_goal(self, drone_id):
        """Get the current intermediate goal for a drone."""
        return self.intermediate_goals.get(drone_id, None)
        
    def is_drone_in_formation(self, drone_id):
        """Check if a specific drone is in formation."""
        if drone_id not in self.drone_positions or drone_id not in self.formation_positions:
            return False
            
        # Get the leader position
        if self.leader_id not in self.drone_positions:
            return False
            
        leader_position = self.drone_positions[self.leader_id]
        drone_position = self.drone_positions[drone_id]
        
        # Get desired formation position
        formation_position = self.get_formation_position(drone_id, leader_position, [0, 0])
        
        # Calculate distance to formation position
        dx = drone_position[0] - formation_position[0]
        dy = drone_position[1] - formation_position[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Check if drone is in position (within tolerance)
        return distance < 0.5  # 50cm tolerance

class MultiAgentPathFinder:
    """Implements a Spatio-Temporal Multi-Agent Path Finding algorithm."""
    def __init__(self, world_size=(40, 40), grid_resolution=GRID_RESOLUTION, safety_radius=SAFETY_RADIUS, world_origin=None):
        self.grid_resolution = grid_resolution
        self.safety_radius = safety_radius
        self.world_size = world_size  # Size in meters
        
        # Calculate grid dimensions
        self.grid_size = (
            int(world_size[0] / grid_resolution),
            int(world_size[1] / grid_resolution)
        )
        
        # Define world origin offset to handle negative coordinates
        # This centers the grid so (0,0) world coordinates are in the middle of the grid
        if world_origin:
            self.world_origin = world_origin
        else:
            self.world_origin = (-world_size[0]/2, -world_size[1]/2)
        
        # Store obstacles and agents
        self.dynamic_obstacles: Dict[str, DynamicObstacle] = {}
        self.agent_positions = {}
        self.agent_goals = {}
        self.agent_paths = {}
        self.agent_velocities = {}
        
        # Directions for 8-connected grid
        self.directions = [
            (0, 0),  # Wait
            (1, 0),  # Right
            (-1, 0), # Left
            (0, 1),  # Up
            (0, -1), # Down
            (1, 1),  # Up-Right
            (-1, 1), # Up-Left
            (1, -1), # Down-Right
            (-1, -1) # Down-Left
        ]
        
        print(f"MAPF Grid initialized: World size={world_size}m, Grid size={self.grid_size} cells, Resolution={grid_resolution}m")
        print(f"World origin at {self.world_origin}, grid center at cell {(self.grid_size[0]//2, self.grid_size[1]//2)}")
        
    def update_obstacle(self, obstacle_id: str, position: Point):
        """Update a dynamic obstacle's position."""
        if obstacle_id in self.dynamic_obstacles:
            self.dynamic_obstacles[obstacle_id].update_position(position)
        else:
            self.dynamic_obstacles[obstacle_id] = DynamicObstacle(obstacle_id, position)
    
    def update_agent(self, agent_id: str, position: Point, goal: Point, velocity: Vector3 = None):
        """Update an agent's position and goal."""
        self.agent_positions[agent_id] = position
        self.agent_goals[agent_id] = goal
        if velocity:
            self.agent_velocities[agent_id] = velocity
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        # Offset by world origin to handle negative coordinates
        offset_x = x - self.world_origin[0]
        offset_y = y - self.world_origin[1]
        
        grid_x = int(offset_x / self.grid_resolution)
        grid_y = int(offset_y / self.grid_resolution)
        
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates."""
        # Convert to world coordinates and apply origin offset
        world_x = (grid_x + 0.5) * self.grid_resolution + self.world_origin[0]
        world_y = (grid_y + 0.5) * self.grid_resolution + self.world_origin[1]
        
        return world_x, world_y
    
    def is_collision(self, x: int, y: int, time_step: int) -> bool:
        """Check if there's a collision with any predicted obstacle at the given time step."""
        # Check if out of bounds
        if x < 0 or x >= self.grid_size[0] or y < 0 or y >= self.grid_size[1]:
            return True
        
        # Convert grid coordinates to world
        world_x, world_y = self.grid_to_world(x, y)
        
        # Check collision with predicted obstacle positions
        future_time = time.time() + (time_step * REPLAN_INTERVAL)
        for obstacle in self.dynamic_obstacles.values():
            predicted_pos = obstacle.predict_position(future_time)
            
            # Calculate distance to obstacle
            dx = predicted_pos.x - world_x
            dy = predicted_pos.y - world_y
            distance = math.sqrt(dx**2 + dy**2)
            
            # Check if within safety radius
            if distance < (self.safety_radius + obstacle.radius):
                return True
        
        return False
    
    def heuristic(self, a: GridMapState, b: GridMapState) -> float:
        """Manhattan distance heuristic with time penalty."""
        # Manhattan distance
        distance = abs(a.x - b.x) + abs(a.y - b.y)
        # Add small time penalty to encourage earlier arrival
        time_penalty = a.time_step * 0.1
        return distance + time_penalty
    
    def find_path(self, agent_id: str) -> List[Tuple[float, float]]:
        """Find a path for a single agent using A* considering time-dependent obstacles."""
        if agent_id not in self.agent_positions or agent_id not in self.agent_goals:
            return []
        
        position = self.agent_positions[agent_id]
        goal = self.agent_goals[agent_id]
        
        # Convert to grid coordinates
        start_grid = self.world_to_grid(position.x, position.y)
        goal_grid = self.world_to_grid(goal.x, goal.y)
        
        start_state = GridMapState(start_grid[0], start_grid[1], 0)
        goal_state = GridMapState(goal_grid[0], goal_grid[1])
        
        # Open and closed sets for A*
        open_set = []
        closed_set = set()
        
        # For path reconstruction
        came_from = {}
        
        # Cost from start to current node
        g_score = {start_state: 0}
        
        # Estimated total cost
        f_score = {start_state: self.heuristic(start_state, goal_state)}
        
        # Push start node to priority queue
        heapq.heappush(open_set, (f_score[start_state], start_state))
        
        iterations = 0
        while open_set and iterations < MAX_ITERATIONS:
            iterations += 1
            
            # Get node with lowest f_score
            _, current = heapq.heappop(open_set)
            
            # If goal reached (only checking x,y not time)
            if current.x == goal_state.x and current.y == goal_state.y:
                path = self.reconstruct_path(came_from, current)
                return path
            
            closed_set.add(current)
            
            # Generate successors
            for dx, dy in self.directions:
                successor = GridMapState(
                    current.x + dx, 
                    current.y + dy, 
                    current.time_step + 1
                )
                
                # Skip if already evaluated or collision
                if successor in closed_set or self.is_collision(successor.x, successor.y, successor.time_step):
                    continue
                
                # Calculate tentative g_score
                if dx == 0 and dy == 0:  # Wait action
                    tentative_g_score = g_score[current] + 0.5  # Small penalty for waiting
                elif abs(dx) + abs(dy) == 1:  # Cardinal directions
                    tentative_g_score = g_score[current] + 1.0
                else:  # Diagonal movement
                    tentative_g_score = g_score[current] + 1.414  # sqrt(2)
                
                # If new path is better, or node not visited yet
                if successor not in g_score or tentative_g_score < g_score[successor]:
                    came_from[successor] = current
                    g_score[successor] = tentative_g_score
                    f_score[successor] = tentative_g_score + self.heuristic(successor, goal_state)
                    
                    # Add to open set if not there
                    if not any(successor == item[1] for item in open_set):
                        heapq.heappush(open_set, (f_score[successor], successor))
        
        # No path found
        return []
    
    def reconstruct_path(self, came_from: Dict, current: GridMapState) -> List[Tuple[float, float]]:
        """Reconstruct path from A* search result and add interpolated points for smoother movement."""
        path = []
        # Build the raw path from A* result
        raw_path = []
        while current in came_from:
            world_x, world_y = self.grid_to_world(current.x, current.y)
            raw_path.append((world_x, world_y))
            current = came_from[current]
            
        # Add start position
        world_x, world_y = self.grid_to_world(current.x, current.y)
        raw_path.append((world_x, world_y))
        
        # Reverse path (from start to goal)
        raw_path.reverse()
        
        # Now interpolate between waypoints to create smaller steps
        # Use very small distance between waypoints for ultra-fine control
        INTERPOLATION_STEP = 0.1  # 10cm between waypoints for very precise movement
        
        for i in range(len(raw_path) - 1):
            x1, y1 = raw_path[i]
            x2, y2 = raw_path[i + 1]
            
            # Calculate distance between these points
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Add the first point
            path.append((x1, y1))
            
            # If distance is large enough, add intermediate points
            if distance > INTERPOLATION_STEP:
                num_steps = max(int(distance / INTERPOLATION_STEP), 1)
                for step in range(1, num_steps):
                    ratio = step / num_steps
                    x = x1 + (x2 - x1) * ratio
                    y = y1 + (y2 - y1) * ratio
                    path.append((x, y))
        
        # Add the final point
        if raw_path:
            path.append(raw_path[-1])
            
        print(f"Path interpolated from {len(raw_path)} raw points to {len(path)} detailed waypoints")
        return path
    
    def find_paths_for_all_agents(self):
        """Find paths for all agents while considering previously planned paths."""
        # Sort agents by priority (can be customized)
        agent_ids = list(self.agent_positions.keys())
        random.shuffle(agent_ids)  # Randomize planning order to avoid bias
        
        all_paths_found = True
        for agent_id in agent_ids:
            path = self.find_path(agent_id)
            if path:
                self.agent_paths[agent_id] = path
            else:
                all_paths_found = False
                print(f"No path found for agent {agent_id}!")
        
        return all_paths_found
    
    def find_path_from_tuples(self, start: Tuple, goal: Tuple, agent_id: str, obstacles=None):
        """Convenience method to find path using tuple coordinates instead of Point objects."""
        # Create Point objects from tuples
        start_point = Point(x=start[0], y=start[1], z=0.0)
        goal_point = Point(x=goal[0], y=goal[1], z=0.0)
        
        # Update agent position and goal
        self.update_agent(agent_id, start_point, goal_point)
        
        # Update obstacles if provided
        if obstacles:
            for obs_id, obs_pos in obstacles:
                if isinstance(obs_pos, tuple) and len(obs_pos) >= 2:
                    pos = Point(x=obs_pos[0], y=obs_pos[1], z=0.0)
                    self.update_obstacle(obs_id, pos)
        
        # Find the path
        return self.find_path(agent_id)

def load_scenario_config(file_path):
    """Load scenario configuration from YAML file."""
    if not os.path.exists(file_path):
        print(f"Config file not found: {file_path}")
        return None
        
    with open(file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return None

class MAPFDroneConfig:
    """Configuration class for MAPF drones"""
    def __init__(self, namespace, start_position, formation_position):
        self.namespace = namespace
        self.start_position = start_position
        self.formation_position = formation_position

class MAPFDrone:
    """Drone class for multiagent path finding mission"""
    def __init__(self, namespace, start_position, end_point, stage_center, 
                 planner, formation_controller=None, formation_position=None,
                 replan_path_interval=1.0, speed=0.6):
        # Basic drone settings
        self.namespace = namespace
        self.start_position = start_position
        self.end_point = end_point
        self.stage_center = stage_center
        self.altitude = start_position[2]  # Flight altitude
        
        # Path planning
        self.planner = planner
        self.replan_path_interval = replan_path_interval  # How often to replan in seconds
        self.path = []  # Current path from A* planner
        self.waypoints = []  # Waypoints for following
        
        # Formation flying
        self.formation_controller = formation_controller
        self.formation_position = formation_position  # Position in formation
        self.is_in_formation = False  # Whether we are currently in formation
        self.speed = speed  # Default movement speed
        
        # Status tracking
        self.is_running = False
        self.mission_complete = False
        self.direct_to_goal = False
        self.obstacle_detected = False
        self.current_position = start_position
        self.dynamic_obstacles = []  # List of dynamic obstacles
        
        # Threading
        self.path_thread = None
        self.lock = threading.Lock()
        
    def position(self):
        """Return the current drone position"""
        return self.current_position
        
    def run(self):
        """Main drone execution loop"""
        try:
            print(f"[{self.namespace}] Starting mission execution")
            
            # Initialize position to start position
            self.current_position = self.start_position
            
            # Plan initial path
            self.plan_path()
            
            # Start path following thread
            self.is_running = True
            self.path_thread = threading.Thread(target=self.follow_path_thread)
            self.path_thread.daemon = True
            self.path_thread.start()
            
            # Keep running until mission is complete
            while self.is_running and not self.mission_complete:
                time.sleep(0.1)
                
            print(f"[{self.namespace}] Mission completed")
                
        except Exception as e:
            print(f"[{self.namespace}] Error: {e}")
            import traceback
            traceback.print_exc()
    
    def plan_path(self):
        """Plan path using the MAPF planner"""
        with self.lock:
            try:
                # Get start and goal positions
                current_x, current_y = self.current_position[0], self.current_position[1]
                goal_x = self.stage_center[0] + self.end_point[0]
                goal_y = self.stage_center[1] + self.end_point[1]
                
                # Get obstacles from the map and other drones
                obstacles = self.dynamic_obstacles
                
                # Call the MAPF planner to plan a path
                print(f"[{self.namespace}] Planning path from ({current_x}, {current_y}) to ({goal_x}, {goal_y})")
                path = self.planner.find_path_from_tuples(
                    start=(current_x, current_y),
                    goal=(goal_x, goal_y),
                    agent_id=self.namespace,
                    obstacles=obstacles
                )
                
                if path:
                    self.path = path
                    # Extract waypoints from the path - path already includes current position
                    self.waypoints = path[1:]  # Skip first point (current position)
                    print(f"[{self.namespace}] Path planned with {len(self.waypoints)} waypoints")
                else:
                    print(f"[{self.namespace}] Failed to plan path!")
            
        except Exception as e:
                print(f"[{self.namespace}] Error planning path: {e}")
                import traceback
                traceback.print_exc()
    
    def update_dynamic_obstacles(self, obstacles):
        """Update the list of dynamic obstacles (other drones)"""
        with self.lock:
            self.dynamic_obstacles = obstacles
    
    def terminate(self):
        """Terminate drone operations"""
        self.is_running = False
        if self.path_thread:
            self.path_thread.join(timeout=2.0)
        print(f"[{self.namespace}] Terminated")
    
    def follow_path_thread(self):
        """Thread for following the planned path with formation constraints"""
        try:
            # Parameters for waypoint following
            WAYPOINT_REACH_THRESHOLD = 0.3  # Distance threshold to consider a waypoint reached
            STUCK_TIME_THRESHOLD = 5.0  # Time in seconds to consider drone as stuck
            MAX_WAYPOINTS_AT_ONCE = 3  # How many waypoints to process at once for smoother movement
            
            # Formation phase management
            formation_phase = True  # Start with formation phase
            formation_start_time = time.time()
            formation_time_required = 15.0  # 15 seconds to establish formation
            formation_duration = 0.0
            formation_established = False
            last_movement_time = time.time()
            last_replan_time = time.time()
            last_formation_check_time = time.time()
            
            # Main loop for path following
            while self.is_running and not self.mission_complete:
            current_time = time.time()
                
                # Current position
                current_x, current_y, current_z = self.current_position
                
                # Check if we need to replan path
                if current_time - last_replan_time > self.replan_path_interval:
                    self.plan_path()
                    last_replan_time = current_time
                
                # Check formation status
                if current_time - last_formation_check_time > 0.5:  # Check every 0.5 seconds
                    if self.formation_controller:
                        self.is_in_formation = self.formation_controller.is_drone_in_formation(self.namespace)
                    last_formation_check_time = current_time
                
                # If we're in formation phase
                if formation_phase:
                    # Check if we've spent enough time in formation phase
                    elapsed_time = current_time - formation_start_time
                    
                    # If we're in formation, count the time
                    if self.is_in_formation and not formation_established:
                        formation_established = True
                        print(f"[{self.namespace}] Formation established")
                    
                    # Count how long we've been in formation
                    if formation_established:
                        formation_duration = current_time - formation_start_time - formation_time_required
                        if formation_duration > 120.0:  # After 2 minutes in formation
                            formation_phase = False
                            self.direct_to_goal = True
                            print(f"[{self.namespace}] Moving to goal after maintaining formation")
                    
                    # If we've been in formation phase for too long without establishing it
                    if elapsed_time > 60.0 and not formation_established:
                        formation_phase = False
                        self.direct_to_goal = True
                        print(f"[{self.namespace}] Failed to establish formation, moving to goal")
                
                # Process current waypoint (if available)
                with self.lock:
                    if not self.waypoints:
                        # No waypoints - check if we're at the goal
                goal_x = self.stage_center[0] + self.end_point[0]
                goal_y = self.stage_center[1] + self.end_point[1]
                        
                        dx = goal_x - current_x
                        dy = goal_y - current_y
                distance_to_goal = math.sqrt(dx*dx + dy*dy)
                
                        if distance_to_goal < WAYPOINT_REACH_THRESHOLD:
                            # We've reached the goal
                            if not self.mission_complete:
                                print(f"[{self.namespace}] Reached goal position!")
                    self.mission_complete = True
                    break
                        else:
                            # Add final goal as waypoint if not there
                            self.waypoints.append((goal_x, goal_y))
                    
                    # Get the current next waypoints to process (up to MAX_WAYPOINTS_AT_ONCE)
                    waypoints_to_process = self.waypoints[:MAX_WAYPOINTS_AT_ONCE]
                    
                    if waypoints_to_process:
                        # Process the next waypoint
                        next_waypoint = waypoints_to_process[0]
                        next_x, next_y = next_waypoint
                    
                    # Calculate distance to waypoint
                        dx = next_x - current_x
                        dy = next_y - current_y
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                        # Check if we've reached the waypoint
                        if distance < WAYPOINT_REACH_THRESHOLD:
                            # Remove the waypoint we've reached
                            self.waypoints.pop(0)
                            print(f"[{self.namespace}] Reached waypoint, {len(self.waypoints)} remaining")
                            last_movement_time = current_time
                            continue
                        
                        # Check if we're stuck
                        if current_time - last_movement_time > STUCK_TIME_THRESHOLD:
                            print(f"[{self.namespace}] Stuck detected, replanning...")
                            self.plan_path()
                            last_movement_time = current_time
                            continue
                        
                        # Calculate movement direction
                        if distance > 0:
                            # Normalize direction
                            direction_x = dx / distance
                            direction_y = dy / distance
                            
                            # Adjust speed based on formation status
                            move_speed = self.speed
                            if formation_phase and not self.is_in_formation:
                                # Move faster if not in formation during formation phase
                                move_speed = self.speed * 1.2
                            elif formation_phase and self.is_in_formation:
                                # Move slower if in formation to maintain it
                                move_speed = self.speed * 0.8
                            elif not formation_phase and self.direct_to_goal:
                                # Move at full speed to goal
                                move_speed = self.speed
                            
                            # Calculate movement step
                            step_size = min(move_speed * 0.1, distance)  # Don't overshoot
                            step_x = direction_x * step_size
                            step_y = direction_y * step_size
                            
                            # Update position
                            new_x = current_x + step_x
                            new_y = current_y + step_y
                            self.current_position = (new_x, new_y, current_z)
                            last_movement_time = current_time
                
                # Sleep to prevent tight loop
                time.sleep(0.05)
            
            print(f"[{self.namespace}] Path following complete")
                
            except Exception as e:
            print(f"[{self.namespace}] Error in path following: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"[{self.namespace}] Stopped path following")

def run_drones(config_file=None):
    """Run the drones in formation with multi-agent path planning."""
    try:
        print("Starting mission with formation flying and multi-agent path planning")
        
        # Load scenario configuration
        # Try multiple possible paths, starting with the provided config file
        config_paths = []
        if config_file:
            config_paths.append(config_file)
        
        # Add relative paths as fallbacks using the script directory
        script_dir = SCRIPT_DIR
        config_paths.extend([
            "scenarios/scenario1_stage4.yaml",  # Relative to current directory
            os.path.join(script_dir, "scenarios/scenario1_stage4.yaml"), # Using script directory
            os.path.join(script_dir, "../scenarios/scenario1_stage4.yaml"),  # Parent directory
            os.path.join(script_dir.parent, "scenarios/scenario1_stage4.yaml"),  # Parent directory using Path
        ])
            
        # Try each path until one works
        config = None
        for path in config_paths:
            print(f"Trying to load config from: {path}")
            config = load_scenario_config(path)
            if config:
                print(f"Successfully loaded configuration from {path}")
                break
                
        if not config:
            print(f"Failed to load configuration from any of the tried paths")
            return
        
        # Extract world parameters with fallback defaults
        stage4_config = config.get('stage4', {})
        
        # Use defaults if keys are missing
        stage_center = stage4_config.get('stage_center', [0.0, 0.0, 0.0])
        world_dimensions = stage4_config.get('world_dimensions', [40.0, 40.0])  # Default 40x40m world
        world_origin = stage4_config.get('world_origin', [-20.0, -20.0])  # Default origin at -20,-20
        
        print(f"Using stage_center: {stage_center}")
        print(f"Using world_dimensions: {world_dimensions}")
        print(f"Using world_origin: {world_origin}")
        
        # Add padding to world dimensions for path planning
        world_min_x = world_origin[0] - 1.0
        world_min_y = world_origin[1] - 1.0
        world_max_x = world_origin[0] + world_dimensions[0] + 1.0
        world_max_y = world_origin[1] + world_dimensions[1] + 1.0
        
        print(f"World bounds: ({world_min_x}, {world_min_y}) to ({world_max_x}, {world_max_y})")
        
        # Create multi-agent path planner
        grid_resolution = 0.2  # 20cm grid cells
        planner = MultiAgentPathFinder(
            world_size=(world_max_x - world_min_x, world_max_y - world_min_y),
            grid_resolution=grid_resolution,
            safety_radius=SAFETY_RADIUS,
            world_origin=(world_min_x, world_min_y)
        )
        
        # Initialize formation controller
        formation_controller = FormationController(
            formation_type="V",  # V formation
            spacing=1.0,        # 1m between drones
            leader_id="drone1"  # drone1 is the leader
        )
        
        # Define end positions for drones
        end_positions = {
            "drone1": (5.0, 5.0, 2.0),   # Leader
            "drone2": (5.0, 5.0, 2.0),   # These positions are in formation relative to leader
            "drone3": (5.0, 5.0, 2.0),
            "drone4": (5.0, 5.0, 2.0),
            "drone5": (5.0, 5.0, 2.0)
        }
        
        # Define start positions (with slight offsets to avoid collisions)
        start_positions = {
            "drone1": (-5.0, -5.0, 2.0),
            "drone2": (-5.0, -4.5, 2.0),
            "drone3": (-5.0, -4.0, 2.0),
            "drone4": (-5.0, -3.5, 2.0),
            "drone5": (-5.0, -3.0, 2.0)
        }
        
        # Define formation positions (V formation offsets relative to leader)
        formation_positions = {
            "drone1": (0.0, 0.0),      # Leader at center
            "drone2": (-1.0, -1.0),    # Behind and left
            "drone3": (-2.0, -2.0),    # Further behind and left
            "drone4": (1.0, -1.0),     # Behind and right
            "drone5": (2.0, -2.0)      # Further behind and right
        }
        
        # Create drone configurations
        drone_configs = {
            namespace: MAPFDroneConfig(
                namespace=namespace,
                start_position=start_positions[namespace],
                formation_position=formation_positions[namespace]
            ) for namespace in ["drone1", "drone2", "drone3", "drone4", "drone5"]
        }
        
        # Create drones
        drones = {}
        for namespace, config in drone_configs.items():
            drone = MAPFDrone(
                namespace=namespace,
                start_position=config.start_position,
                end_point=end_positions[namespace],
                stage_center=stage_center,
                planner=planner,
                formation_controller=formation_controller,
                formation_position=config.formation_position
            )
            drones[namespace] = drone
            
            # Register drone with formation controller
            formation_controller.register_drone(
                namespace=namespace, 
                initial_position=[config.start_position[0], config.start_position[1]],
                formation_position=config.formation_position
            )
        
        # Start thread for each drone
        threads = {}
        for namespace, drone in drones.items():
            print(f"Starting drone: {namespace}")
            thread = threading.Thread(target=drone.run)
            thread.daemon = True
            thread.start()
            threads[namespace] = thread
        
        # Main control loop
        try:
            print("All drones started. Press Ctrl+C to stop.")
            
            # Run until all drones complete their missions or interrupted
            running = True
            update_interval = 0.1  # seconds
            last_obstacle_update = time.time()
            
            while running:
                time.sleep(update_interval)
                
                # Check if all drones have completed their missions
            all_complete = True
                for namespace, drone in drones.items():
                if not drone.mission_complete:
                    all_complete = False
                    break
            
                if all_complete:
                    print("All drones have completed their missions!")
                    running = False
                    break
                
                # Update dynamic obstacles every 0.5 seconds
                current_time = time.time()
                if current_time - last_obstacle_update > 0.5:
                    # Collect current positions of all drones to use as dynamic obstacles
                    obstacles = []
                    for name, drone in drones.items():
                        position = drone.position()
                        obstacles.append((name, position))
                    
                    # Update obstacles for each drone (excluding itself)
                    for name, drone in drones.items():
                        drone_obstacles = [obs for obs in obstacles if obs[0] != name]
                        drone.update_dynamic_obstacles(drone_obstacles)
                    
                    # Update formation controller with current positions
                    for name, drone in drones.items():
                        position = drone.position()
                        formation_controller.update_drone_position(name, [position[0], position[1]])
                    
                    # Check formation status
                    in_formation_count = 0
                    for name, drone in drones.items():
                        if drone.is_in_formation:
                            in_formation_count += 1
                    
                    # Print formation status occasionally
                    if random.random() < 0.1:  # 10% chance
                        print(f"Formation status: {in_formation_count}/5 drones in formation")
                    
                    last_obstacle_update = current_time
    
    except KeyboardInterrupt:
            print("Mission interrupted by user")
        
        # Clean up and wait for threads to terminate
        for namespace, drone in drones.items():
            print(f"Terminating drone: {namespace}")
            drone.terminate()
        
        for namespace, thread in threads.items():
            thread.join(timeout=2.0)
            
        print("Mission completed successfully")
        
    except Exception as e:
        print(f"Error in run_drones: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Stage 4: MAPF-based Dynamic Obstacle Avoidance")
    parser.add_argument('-n', '--namespaces', nargs='+',
                        default=['drone0', 'drone1', 'drone2', 'drone3', 'drone4'],
                        help='List of drone namespaces')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Enable verbose output')
    parser.add_argument('-s', '--use_sim_time', action='store_true', default=True,
                        help='Use simulation time')
    parser.add_argument('-c', '--config', type=str, 
                        default=None,
                        help='Path to scenario configuration file')
    
    args = parser.parse_args()
    
    # Initialize ROS
    rclpy.init()
    
    # Run the drones
    run_drones(args.config)
    
    # Shutdown ROS
    rclpy.shutdown()

if __name__ == "__main__":
    main() 