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
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from as2_msgs.msg import YawMode, BehaviorStatus
from as2_python_api.drone_interface import DroneInterface
from as2_python_api.behavior_actions.behavior_handler import BehaviorHandler
from std_msgs.msg import ColorRGBA, Float32MultiArray, String
from geometry_msgs.msg import PoseStamped, Point, Vector3, PoseArray
from typing import List, Tuple, Optional

# --- RVO Implementation with Formation Bias ---
class RVO:
    """Reciprocal Velocity Obstacles with formation bias"""
    def __init__(self, radius=0.3, time_horizon=1.0, max_speed=2.0, 
                 formation_weight=0.2, neighbor_weight=2.0, obstacle_weight=4.0):
        self.radius = radius  # Collision radius
        self.time_horizon = time_horizon  # Time horizon for collision detection
        self.max_speed = max_speed  # Maximum speed
        
        # Weights for different constraints
        self.formation_weight = formation_weight  # Weight for formation bias - reduced from 0.5 to 0.2
        self.neighbor_weight = neighbor_weight  # Weight for neighbor avoidance - staying at 2.0
        self.obstacle_weight = obstacle_weight  # Weight for obstacle avoidance - increased from 2.0 to 4.0
        
        # Number of samples for RVO
        self.num_samples = 200  # Increased from 150 to 200 for better sampling
        
    def compute_velocity(self, position: Point, velocity: Vector3, 
                        desired_velocity: Vector3, neighbors: List[Tuple[Point, Vector3]],
                        formation_position: Optional[Point] = None):
        """
        Compute a new velocity based on RVO
        
        Args:
            position: Current position
            velocity: Current velocity
            desired_velocity: Velocity towards goal
            neighbors: List of (position, velocity) tuples for other agents
            formation_position: Optional target position in formation
            
        Returns:
            New velocity as Vector3
        """
        # Create velocity candidates (sample velocities)
        candidates = self._generate_candidates(velocity, desired_velocity)
        
        # Score each candidate
        best_velocity = None
        best_score = float('-inf')
        
        for candidate in candidates:
            # Compute score based on similarity to desired velocity and collision avoidance
            score = self._score_velocity(candidate, desired_velocity, position, 
                                        neighbors, formation_position)
            
            if score > best_score:
                best_score = score
                best_velocity = candidate
                
        # If no valid velocity was found, use a scaled-down version of the desired velocity
        if best_velocity is None:
            best_velocity = Vector3(
                x=desired_velocity.x * 0.2,  # Increased from 0.1 to 0.2
                y=desired_velocity.y * 0.2,  # Increased from 0.1 to 0.2
                z=desired_velocity.z
            )
            
        return best_velocity
    
    def _generate_candidates(self, current_velocity: Vector3, desired_velocity: Vector3):
        """Generate candidate velocities to evaluate"""
        candidates = []
        
        # Always include the desired velocity
        candidates.append(desired_velocity)
        
        # Add current velocity
        candidates.append(current_velocity)
        
        # Generate random samples within max_speed
        for _ in range(self.num_samples):
            # Biased sampling: 70% close to desired velocity, 30% uniformly
            if random.random() < 0.7:
                # Sample close to desired velocity
                speed = min(math.sqrt(desired_velocity.x**2 + desired_velocity.y**2), self.max_speed)
                angle_desired = math.atan2(desired_velocity.y, desired_velocity.x)
                
                # Add some noise to the angle
                angle = angle_desired + random.uniform(-math.pi/4, math.pi/4)
                
                # Random speed between 0.5 and 1.2 times the desired speed
                rand_speed = speed * random.uniform(0.5, 1.2)
                rand_speed = min(rand_speed, self.max_speed)
                
                candidate = Vector3(
                    x=rand_speed * math.cos(angle),
                    y=rand_speed * math.sin(angle),
                    z=desired_velocity.z
                )
            else:
                # Uniform sampling
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(0, self.max_speed)
                
                candidate = Vector3(
                    x=speed * math.cos(angle),
                    y=speed * math.sin(angle),
                    z=desired_velocity.z
                )
                
            candidates.append(candidate)
            
        return candidates
    
    def _score_velocity(self, velocity: Vector3, desired_velocity: Vector3, 
                       position: Point, neighbors: List[Tuple[Point, Vector3]],
                       formation_position: Optional[Point] = None):
        """Score a velocity based on goal directness and collision avoidance"""
        # Start with a base score based on similarity to desired velocity
        desired_speed = math.sqrt(desired_velocity.x**2 + desired_velocity.y**2)
        if desired_speed < 0.001:  # Avoid division by zero
            similarity = 1.0
        else:
            dot_product = velocity.x * desired_velocity.x + velocity.y * desired_velocity.y
            vel_speed = math.sqrt(velocity.x**2 + velocity.y**2)
            similarity = dot_product / (desired_speed * vel_speed) if vel_speed > 0.001 else 0
        
        # Base score on similarity to desired velocity
        score = similarity
        
        # Penalty for collision with neighbors
        for neighbor_pos, neighbor_vel in neighbors:
            # Relative position and velocity
            rel_pos_x = neighbor_pos.x - position.x
            rel_pos_y = neighbor_pos.y - position.y
            rel_vel_x = neighbor_vel.x - velocity.x
            rel_vel_y = neighbor_vel.y - velocity.y
            
            # Distance squared
            dist_sq = rel_pos_x**2 + rel_pos_y**2
            
            # Skip if too far away
            if dist_sq > (self.radius * 10)**2:
                continue
                
            # Time to collision (if heading towards collision)
            if dist_sq <= (2 * self.radius)**2:
                # Already too close
                ttc = 0.0
            else:
                # Project relative velocity onto relative position
                proj = (rel_vel_x * rel_pos_x + rel_vel_y * rel_pos_y) / dist_sq
                
                if proj >= 0:
                    # Moving away, no collision
                    continue
                    
                # Compute time to collision
                c = dist_sq - (2 * self.radius)**2
                a = rel_vel_x**2 + rel_vel_y**2
                b = 2 * (rel_vel_x * rel_pos_x + rel_vel_y * rel_pos_y)
                
                if abs(a) < 0.001:
                    # Relative velocity is almost zero
                    continue
                    
                discriminant = b*b - 4*a*c
                if discriminant < 0:
                    # No collision
                    continue
                    
                ttc = (-b - math.sqrt(discriminant)) / (2*a)
                if ttc <= 0 or ttc > self.time_horizon:
                    # Collision too soon or too far in future
                    continue
                    
            # Penalty decreases as ttc approaches time_horizon
            penalty = self.neighbor_weight * (1.0 - ttc / self.time_horizon)**2
            score -= penalty
            
        # Formation bias (if formation position is provided)
        if formation_position is not None:
            # Direction to formation position
            form_dir_x = formation_position.x - position.x
            form_dir_y = formation_position.y - position.y
            form_dist = math.sqrt(form_dir_x**2 + form_dir_y**2)
            
            if form_dist > 0.001:
                # Normalize
                form_dir_x /= form_dist
                form_dir_y /= form_dist
                
                # Dot product with velocity direction
                vel_dist = math.sqrt(velocity.x**2 + velocity.y**2)
                if vel_dist > 0.001:
                    vel_dir_x = velocity.x / vel_dist
                    vel_dir_y = velocity.y / vel_dist
                    
                    formation_alignment = form_dir_x * vel_dir_x + form_dir_y * vel_dir_y
                    
                    # Add formation bias (higher for further distances, max at 2 meters)
                    bias = self.formation_weight * min(form_dist, 2.0) / 2.0 * formation_alignment
                    score += bias
        
        return score

class ORCA:
    def __init__(self, radius=0.8, time_horizon=2.0, max_speed=2.0):
        """Initialize ORCA parameters
        
        Args:
            radius: Agent radius for collision avoidance (meters)
            time_horizon: Time window for collision avoidance (seconds)
            max_speed: Maximum speed of the agent (m/s)
        """
        self.radius = radius  # Increased from 0.6 to 0.8 for safer avoidance
        self.time_horizon = time_horizon  # Increased from 1.5 to 2.0 for earlier detection
        self.max_speed = max_speed
    
    def compute_velocity(self, current_pos, current_vel, desired_vel, neighbors, formation_pos=None):
        """Compute new velocity using ORCA algorithm
        
        Args:
            current_pos: Current position (Point)
            current_vel: Current velocity (Vector3)
            desired_vel: Desired velocity (Vector3)
            neighbors: List of (position, velocity) tuples for other agents
            formation_pos: Target formation position (Point or None)
            
        Returns:
            Vector3: New velocity that avoids collisions
        """
        import numpy as np
        
        # Convert inputs to numpy arrays for easier computation
        position = np.array([current_pos.x, current_pos.y])
        velocity = np.array([current_vel.x, current_vel.y])
        preferred_velocity = np.array([desired_vel.x, desired_vel.y])
        
        # ORCA lines represented as a line with point and normal vector
        orca_lines = []
        
        # Process each neighbor
        for neighbor_pos, neighbor_vel in neighbors:
            neighbor_position = np.array([neighbor_pos.x, neighbor_pos.y])
            neighbor_velocity = np.array([neighbor_vel.x, neighbor_vel.y])
            
            # Vector from this agent to neighbor
            relative_position = neighbor_position - position
            distance = np.linalg.norm(relative_position)
            
            # Skip if too far away to matter
            if distance > 5.0:
                continue
                
            # Combined radius for collision
            combined_radius = self.radius * 2.0
            
            # Skip if collision isn't possible within time horizon
            if distance > combined_radius + self.max_speed * self.time_horizon:
                continue
            
            # Relative velocity
            relative_velocity = velocity - neighbor_velocity
            
            # Time to collision
            time_to_collision = self.compute_time_to_collision(relative_position, relative_velocity, combined_radius)
            
            if time_to_collision > self.time_horizon:
                continue
                
            # If there will be a collision, add an ORCA line
            if time_to_collision > 0:
                # Compute point on boundary after time_to_collision
                w = relative_velocity * time_to_collision - relative_position
                w_norm = np.linalg.norm(w)
                
                if w_norm < 0.001:
                    # Avoid division by zero
                    continue
                
                # Normal vector of the ORCA line
                n = (w / w_norm)
                
                # ORCA line: u' - u is on the side containing n where u is current velocity
                # and u' is the new velocity
                u = velocity
                point = u + (combined_radius / time_to_collision - np.dot(relative_velocity, n)) * n / 2.0
                orca_lines.append((point, n))
            else:
                # Immediate collision, use vector away from neighbor
                if np.linalg.norm(relative_position) < 0.001:
                    # Agents are too close, use random direction
                    angle = np.random.uniform(0, 2 * np.pi)
                    n = np.array([np.cos(angle), np.sin(angle)])
                else:
                    n = relative_position / np.linalg.norm(relative_position)
                
                point = velocity + n * self.max_speed
                orca_lines.append((point, n))
        
        # Add formation constraint as a soft constraint
        if formation_pos is not None:
            formation_pos_array = np.array([formation_pos.x, formation_pos.y])
            to_formation = formation_pos_array - position
            dist_to_formation = np.linalg.norm(to_formation)
            
            if dist_to_formation > 0.1:  # Only if we're not already at the formation position
                # Add a soft constraint towards formation position
                to_formation_norm = to_formation / dist_to_formation
                formation_weight = min(1.0, dist_to_formation / 3.0)  # Weight increases with distance
                preferred_velocity = preferred_velocity + formation_weight * to_formation_norm * self.max_speed
        
        # Solve the linear program to find new velocity
        new_velocity = self.linear_program2(orca_lines, self.max_speed, preferred_velocity, velocity)
        
        # Convert back to Vector3
        return Vector3(x=float(new_velocity[0]), y=float(new_velocity[1]), z=0.0)
    
    def compute_time_to_collision(self, relative_position, relative_velocity, combined_radius):
        """Compute time to collision between two agents"""
        import numpy as np
        
        # Quadratic equation coefficients: a*t^2 + b*t + c = 0
        a = np.dot(relative_velocity, relative_velocity)
        if a < 0.001:
            return float('inf')  # Not moving relative to each other
            
        b = 2 * np.dot(relative_position, relative_velocity)
        c = np.dot(relative_position, relative_position) - combined_radius * combined_radius
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return float('inf')  # No collision
            
        t = (-b - np.sqrt(discriminant)) / (2 * a)
        
        if t < 0:
            return 0  # Collision already happened
        
        return t
    
    def linear_program2(self, orca_lines, max_speed, preferred_velocity, current_velocity):
        """Find a new velocity that satisfies all ORCA constraints"""
        import numpy as np
        
        if not orca_lines:
            # No constraints, use preferred velocity limited by max_speed
            if np.linalg.norm(preferred_velocity) > max_speed:
                return preferred_velocity * (max_speed / np.linalg.norm(preferred_velocity))
            return preferred_velocity
        
        # Try preferred velocity first
        new_velocity = preferred_velocity
        if np.linalg.norm(new_velocity) > max_speed:
            new_velocity = new_velocity * (max_speed / np.linalg.norm(new_velocity))
            
        # Check if preferred velocity satisfies all constraints
        satisfied = True
        for point, normal in orca_lines:
            if np.dot(normal, new_velocity - point) < 0:
                satisfied = False
                break
                
        if satisfied:
            return new_velocity
            
        # Otherwise solve the linear program
        # Project preferred velocity onto each ORCA line and pick the one
        # that is closest to the preferred velocity
        min_dist = float('inf')
        min_velocity = None
        
        for i, (point, normal) in enumerate(orca_lines):
            # Project preferred velocity onto ORCA line
            projection = preferred_velocity - np.dot(preferred_velocity - point, normal) * normal
            
            # Ensure it's within max speed
            if np.linalg.norm(projection) > max_speed:
                projection = projection * (max_speed / np.linalg.norm(projection))
                
            # Check if this projection satisfies all other constraints
            valid = True
            for j, (other_point, other_normal) in enumerate(orca_lines):
                if i != j and np.dot(other_normal, projection - other_point) < -0.001:
                    valid = False
                    break
                    
            if valid:
                dist = np.linalg.norm(projection - preferred_velocity)
                if dist < min_dist:
                    min_dist = dist
                    min_velocity = projection
        
        if min_velocity is not None:
            return min_velocity
            
        # If no valid projection found, use current velocity as a fallback
        return current_velocity

def formation_staggered_5():
    """
    Return offsets for a staggered formation for 5 drones.
    Creates a pattern with leader (drone2) in the middle.
    """
    d = 0.4  # Spacing parameter
    return [(-2*d, 0.0), (-d, d/2), (0.0, 0.0), (d, d/2), (2*d, 0.0)]

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

# --- Drone class ---
class Drone(DroneInterface):
    """
    Drone that navigates to end point while avoiding obstacles.
    """
    def __init__(self, namespace: str, config_path: str, verbose: bool = False, use_sim_time: bool = False):
        super().__init__(namespace, verbose=verbose, use_sim_time=use_sim_time)
        self._speed = 1.0  # Base speed reduced from 2.0 to 1.0 as requested
        self._yaw_mode = YawMode.PATH_FACING
        self._yaw_angle = None
        self._frame_id = "earth"
        self.current_behavior: BehaviorHandler = None
        self.led_pub = self.create_publisher(ColorRGBA, f"/{namespace}/leds/control", 10)
        
        # Load configuration
        self.config = load_scenario_config(config_path)
        if not self.config:
            raise ValueError(f"Failed to load configuration from {config_path}")
            
        # Stage 4 parameters
        stage4_config = self.config['stage4']
        self.stage_center = stage4_config['stage_center']
        self.start_point = stage4_config['start_point']
        self.end_point = stage4_config['end_point']
        self.altitude = 2.0  # Default flight altitude
        
        # Debug print of scenario configuration
        print(f"[{namespace}] Config loaded - stage center: {self.stage_center}, end point: {self.end_point}")
        
        # Set different start positions for each drone to avoid collisions
        # Extract drone number from namespace (assumes format 'droneX' where X is a number)
        try:
            drone_number = int(namespace[5:])
            # Position all drones at the start point (with small offsets to avoid exact overlap)
            offset = 0.3 * drone_number  # Small offset between drones to avoid collisions
            self.takeoff_x = self.stage_center[0] + self.start_point[0]
            self.takeoff_y = self.stage_center[1] + self.start_point[1] + offset
            self.drone_number = drone_number  # Store drone number for formation calculations
            print(f"[{namespace}] Using start point ({self.takeoff_x}, {self.takeoff_y})")
        except (ValueError, IndexError):
            # Default to start point if parsing fails
            self.takeoff_x = self.stage_center[0] + self.start_point[0]
            self.takeoff_y = self.stage_center[1] + self.start_point[1]
            self.drone_number = 0  # Default to leader
        
        # Formation parameters
        self.formation_spacing = 2.0  # Increased from 1.2 to 2.0 as requested
        self.all_drone_namespaces = []  # Will be set in run_drone function
        
        # RVO collision avoidance - updated parameters for higher obstacle priority
        self.rvo = RVO(radius=0.8, time_horizon=2.0, max_speed=2.0, 
                        formation_weight=0.2, neighbor_weight=2.0, obstacle_weight=4.0)
        
        # Track dynamic obstacles
        self.dynamic_obstacles = {}  # Dictionary to store obstacle positions
        
        # Subscribe to dynamic obstacle topic
        print(f"[{namespace}] Subscribing to dynamic obstacles...")
        self.obstacle_sub = self.create_subscription(
            PoseStamped,
            '/dynamic_obstacles/locations',
            self.obstacle_callback,
            10
        )
        
        # Track neighbor states for RVO
        self.neighbor_states = {}  # Dictionary to store neighbor positions and velocities
        
        # Mission state
        self.running = False
        self.mission_complete = False
        
        # Waypoints for navigation
        self.waypoints = []
        self.current_waypoint_index = 0
        self.waypoint_reached_threshold = 1.0  # Distance in meters to consider a waypoint reached
        
        # Generate waypoints
        self._generate_waypoints()
        
        # Position update thread
        self.update_thread = None

    def _generate_waypoints(self):
        """Generate a series of waypoints from start to end position."""
        # Calculate absolute coordinates of start and end
        start_x = self.stage_center[0] + self.start_point[0]
        start_y = self.stage_center[1] + self.start_point[1]
        end_x = self.stage_center[0] + self.end_point[0]
        end_y = self.stage_center[1] + self.end_point[1]
        
        # Calculate total distance
        dx = end_x - start_x
        dy = end_y - start_y
        total_distance = math.sqrt(dx*dx + dy*dy)
        
        # Normalize direction vector
        if total_distance > 0.001:
            nx = dx / total_distance
            ny = dy / total_distance
        else:
            nx, ny = 0.0, -1.0  # Default direction if start and end are too close
        
        # Create waypoints at regular intervals (every 2.0 meters)
        waypoint_spacing = 2.0  # Distance between waypoints
        num_waypoints = max(2, int(total_distance / waypoint_spacing))
        
        # Create waypoints
        self.waypoints = []
        for i in range(num_waypoints + 1):  # +1 to include end point
            progress = i / num_waypoints
            waypoint_x = start_x + dx * progress
            waypoint_y = start_y + dy * progress
            self.waypoints.append(Point(x=waypoint_x, y=waypoint_y, z=self.altitude))
        
        # Add the exact end point as final waypoint
        if len(self.waypoints) > 0 and (
            abs(self.waypoints[-1].x - end_x) > 0.1 or
            abs(self.waypoints[-1].y - end_y) > 0.1
        ):
            self.waypoints.append(Point(x=end_x, y=end_y, z=self.altitude))
        
        # Print waypoints for debugging
        print(f"[{self.namespace}] Generated {len(self.waypoints)} waypoints:")
        for i, wp in enumerate(self.waypoints):
            print(f"  Waypoint {i}: ({wp.x:.2f}, {wp.y:.2f})")

    def get_current_waypoint(self):
        """Get the current target waypoint."""
        if not self.waypoints or self.current_waypoint_index >= len(self.waypoints):
            # If no waypoints or all waypoints reached, use end point
            return Point(
                x=self.stage_center[0] + self.end_point[0],
                y=self.stage_center[1] + self.end_point[1],
                z=self.altitude
            )
        return self.waypoints[self.current_waypoint_index]
    
    def update_waypoint_progress(self):
        """Check if current waypoint is reached and update index if needed."""
        if not self.waypoints or self.current_waypoint_index >= len(self.waypoints):
            return False  # No more waypoints
            
        # Get current position and waypoint
        x, y, z = self.position()
        current_pos = Point(x=x, y=y, z=z)
        waypoint = self.waypoints[self.current_waypoint_index]
        
        # Calculate distance to waypoint
        dx = waypoint.x - current_pos.x
        dy = waypoint.y - current_pos.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Check if waypoint is reached
        if distance < self.waypoint_reached_threshold:
            self.current_waypoint_index += 1
            if self.current_waypoint_index < len(self.waypoints):
                next_waypoint = self.waypoints[self.current_waypoint_index]
                print(f"[{self.namespace}] Reached waypoint {self.current_waypoint_index-1}, "
                      f"moving to next waypoint: ({next_waypoint.x:.2f}, {next_waypoint.y:.2f})")
                return True  # Waypoint advanced
            else:
                # All waypoints reached
                self.mission_complete = True
                print(f"[{self.namespace}] Reached final waypoint! Mission complete.")
                return False
        return False  # Still heading to current waypoint

    def obstacle_callback(self, msg):
        """Process incoming dynamic obstacle positions."""
        try:
            obstacle_id = msg.header.frame_id
            position = msg.pose.position
            self.dynamic_obstacles[obstacle_id] = position
            
            # Print debug info about obstacle positions periodically
            if random.random() < 0.02:  # Approximately 2% chance to print (to avoid flooding)
                print(f"[{self.namespace}] Detected obstacle {obstacle_id} at position: ({position.x:.2f}, {position.y:.2f}, {position.z:.2f})")
                
                # Also print our current goal
                end_x = self.stage_center[0] + self.end_point[0]
                end_y = self.stage_center[1] + self.end_point[1]
                print(f"[{self.namespace}] Current goal: ({end_x:.2f}, {end_y:.2f})")
                
        except Exception as e:
            print(f"[{self.namespace}] Error in obstacle_callback: {e}")

    def calculate_formation_positions(self, lead_position):
        """Calculate formation positions for all drones based on leader position and progress vector.
        
        Args:
            lead_position: Position of the formation leader
            
        Returns:
            List of formation positions (Point objects) for each drone
        """
        # Calculate direction vector toward goal
        goal_x = self.stage_center[0] + self.end_point[0]
        goal_y = self.stage_center[1] + self.end_point[1]
        
        # Vector from leader to goal (direction of travel)
        dx = goal_x - lead_position.x
        dy = goal_y - lead_position.y
        dist = math.sqrt(dx*dx + dy*dy)
        
        # Print goal direction information
        print(f"[{self.namespace}] Goal direction: dx={dx:.2f}, dy={dy:.2f}, dist={dist:.2f}")
        
        # Normalize direction vector
        if dist > 0.001:
            dx /= dist
            dy /= dist
        else:
            # Default direction if at goal (moving left toward -4,6)
            dx, dy = -1.0, 0.0
            
        # Perpendicular vector for formation (rotate 90 degrees)
        perp_x, perp_y = -dy, dx
            
        # Calculate formation positions
        formation_positions = []
        
        # Number of drones in the swarm
        num_drones = len(self.all_drone_namespaces)
        
        # Create a formation heading toward the goal
        # Lead drone is at the front, others in a V-formation behind it
        lead_pos = Point(
            x=lead_position.x + dx * 1.0,  # Move lead drone slightly ahead toward goal
            y=lead_position.y + dy * 1.0,
            z=lead_position.z
        )
        
        for i in range(num_drones):
            if i == 0:  # Leader position
                formation_positions.append(lead_pos)
            else:
                # Staggered V-formation: spread drones behind leader
                # Odd drones on left, even drones on right
                side = -1 if i % 2 == 1 else 1
                offset = (i + 1) // 2  # 1,1,2,2,3,3,...
                
                # Calculate position in formation
                # Perpendicular offset to create V-formation
                form_x = lead_pos.x + side * perp_x * self.formation_spacing * offset
                form_y = lead_pos.y + side * perp_y * self.formation_spacing * offset
                
                # Back-staggering - further back for drones that are further out
                form_x -= dx * offset * 0.7
                form_y -= dy * offset * 0.7
                
                # Create point for formation position
                form_pos = Point(x=form_x, y=form_y, z=lead_pos.z)
                formation_positions.append(form_pos)
        
        # Print formation positions for debugging
        if random.random() < 0.01:  # 1% chance to print
            print("Formation positions:")
            for i, pos in enumerate(formation_positions):
                print(f"  Drone {i}: ({pos.x:.2f}, {pos.y:.2f})")
                
        return formation_positions
        
    def get_formation_position(self):
        """Get this drone's position in the formation."""
        # Get current positions of all drones
        positions = []
        for ns in self.all_drone_namespaces:
            if ns in self.neighbor_states and self.neighbor_states[ns] is not None:
                pos = self.neighbor_states[ns]['position']
                positions.append((ns, pos))
            elif ns == self.namespace:
                # My own position
                x, y, z = self.position()
                pos = Point(x=x, y=y, z=z)
                positions.append((ns, pos))
        
        # Find the leader (drone0) position
        leader_pos = None
        for ns, pos in positions:
            if ns == 'drone0' and pos is not None:
                leader_pos = pos
                break
                
        # If no leader found, use my position as reference
        if leader_pos is None:
            x, y, z = self.position()
            leader_pos = Point(x=x, y=y, z=z)
            
        # Calculate all formation positions based on leader
        formation_positions = self.calculate_formation_positions(leader_pos)
        
        # Find my position in the formation
        try:
            my_index = self.all_drone_namespaces.index(self.namespace)
            return formation_positions[my_index]
        except (ValueError, IndexError):
            # Fallback: return leader position
            return leader_pos

    def calculate_position(self):
        """Calculate next position using RVO for collision avoidance and formation constraints."""
        try:
            # Get current position and velocity
            x, y, z = self.position()
            current_pos = Point(x=x, y=y, z=z)
            
            # Get current velocity (default to zero if not available)
            current_vel = Vector3(x=0.0, y=0.0, z=0.0)
            
            # Get current waypoint (intermediate goal)
            waypoint = self.get_current_waypoint()
            
            # Check if current waypoint is reached and update if needed
            self.update_waypoint_progress()
            
            # Calculate direction and distance to waypoint
            dx = waypoint.x - current_pos.x
            dy = waypoint.y - current_pos.y
            distance_to_waypoint = math.sqrt(dx*dx + dy*dy)
            
            # Print debugging info occasionally
            if random.random() < 0.01:  # 1% chance to print
                print(f"[{self.namespace}] Current position: ({current_pos.x:.2f}, {current_pos.y:.2f})")
                print(f"[{self.namespace}] Current waypoint ({self.current_waypoint_index}): "
                      f"({waypoint.x:.2f}, {waypoint.y:.2f})")
                print(f"[{self.namespace}] Distance to waypoint: {distance_to_waypoint:.2f}m")
                
                # Also display final goal
                end_x = self.stage_center[0] + self.end_point[0]
                end_y = self.stage_center[1] + self.end_point[1]
                goal_dx = end_x - current_pos.x
                goal_dy = end_y - current_pos.y
                goal_dist = math.sqrt(goal_dx*goal_dx + goal_dy*goal_dy)
                print(f"[{self.namespace}] Distance to final goal: {goal_dist:.2f}m")
            
            # Calculate desired velocity (towards waypoint)
            # Reduced direct-to-goal factor to give more room for avoidance
            direct_to_goal_factor = 0.9  
            
            # Direct velocity toward waypoint with minimum speed to ensure movement
            min_speed = 0.8  
            if distance_to_waypoint > 0.5:  # Only apply minimum if not close to waypoint
                if abs(dx) < 0.001 and abs(dy) < 0.001:
                    # If direction vector is too small, use default direction
                    goal_vel = Vector3(
                        x=-1.0,  # Default movement toward goal (left in this case)
                        y=0.0,
                        z=0.0
                    )
                else:
                    # Normalize direction vector
                    dir_mag = math.sqrt(dx*dx + dy*dy)
                    norm_dx = dx / dir_mag
                    norm_dy = dy / dir_mag
                    
                    # Apply full speed - always move aggressively toward waypoint
                    goal_speed = max(self._speed * 1.5, min_speed)  # Increased speed multiplier
                    goal_vel = Vector3(
                        x=norm_dx * goal_speed,
                        y=norm_dy * goal_speed,
                        z=0.0
                    )
            else:
                goal_vel = Vector3(
                    x=dx * self._speed,
                    y=dy * self._speed,
                    z=0.0
                )
            
            # Get this drone's desired formation position
            formation_pos = self.get_formation_position()
            
            # Calculate velocity component toward formation position
            form_dx = formation_pos.x - current_pos.x
            form_dy = formation_pos.y - current_pos.y
            form_dist = math.sqrt(form_dx*form_dx + form_dy*form_dy)
            
            # Combine direct-to-goal with formation velocity
            # Significantly reduce formation influence
            formation_factor = min(0.2, form_dist / 5.0) * (1.0 - direct_to_goal_factor)  # Reduced from 0.5 to 0.2
            
            # Combined desired velocity - heavily prioritizing goal
            desired_vel = Vector3(
                x=goal_vel.x + form_dx * self._speed * formation_factor * 0.5,  # Reduced formation influence
                y=goal_vel.y + form_dy * self._speed * formation_factor * 0.5,  # Reduced formation influence
                z=0.0
            )
            
            # Check if we're close enough to the final goal
            end_x = self.stage_center[0] + self.end_point[0]
            end_y = self.stage_center[1] + self.end_point[1]
            dx_final = end_x - current_pos.x
            dy_final = end_y - current_pos.y
            distance_to_goal = math.sqrt(dx_final*dx_final + dy_final*dy_final)
            
            if distance_to_goal < 0.5:  # Within 0.5m of end point
                self.mission_complete = True
                print(f"[{self.namespace}] Reached end point! Distance: {distance_to_goal:.2f}m")
                return current_pos  # Just stay in place
                        
            # Get neighbor information for RVO
            neighbors = []
            for ns, state in self.neighbor_states.items():
                if state is not None and 'position' in state:
                    neighbors.append((state['position'], state['velocity']))
            
            # Add dynamic obstacles as neighbors with higher priority (add them multiple times to increase influence)
            for obs_id, obs_pos in self.dynamic_obstacles.items():
                if obs_pos is not None:
                    # Add the obstacle three times to increase its influence on avoidance
                    for _ in range(3):
                        neighbors.append((obs_pos, Vector3(x=0.0, y=0.0, z=0.0)))
            
            # Compute new velocity using RVO with formation constraints
            new_vel = self.rvo.compute_velocity(
                current_pos,
                current_vel,
                desired_vel,
                neighbors,
                formation_pos  # Include formation position for potential wells
            )
            
            # Add momentum - don't let velocity drop to zero
            if math.sqrt(new_vel.x**2 + new_vel.y**2) < 0.1:
                # If velocity is too low, inject momentum in waypoint direction
                new_vel.x += dx / distance_to_waypoint * 0.2 if distance_to_waypoint > 0.001 else 0.0
                new_vel.y += dy / distance_to_waypoint * 0.2 if distance_to_waypoint > 0.001 else 0.0
            
            # Calculate new position based on new velocity
            dt = 0.2  # Increased time step for larger movements
            new_pos = Point(
                x=current_pos.x + new_vel.x * dt,
                y=current_pos.y + new_vel.y * dt,
                z=current_pos.z  # Maintain altitude
            )
            
            return new_pos
            
        except Exception as e:
            print(f"[{self.namespace}] Error in calculate_position: {e}")
            # Return current position as fallback
            x, y, z = self.position()
            return Point(x=x, y=y, z=z)
            
    def position(self):
        """Get the drone's current position."""
        try:
            # Try to get position from DroneInterface API
            try:
                # Different DroneInterface implementations may use different methods
                # Attempt various options in order of likely accuracy
                
                # Try get_position_as_list if available
                if hasattr(self, 'get_position_as_list') and callable(getattr(self, 'get_position_as_list')):
                    pos = self.get_position_as_list()
                    if pos and len(pos) >= 3:
                        return pos[0], pos[1], pos[2]
                
                # Try get_position if available
                if hasattr(self, 'get_position') and callable(getattr(self, 'get_position')):
                    pos = self.get_position()
                    if pos:
                        return pos[0], pos[1], pos[2]
                        
                # Try get_pose if available
                if hasattr(self, 'get_pose') and callable(getattr(self, 'get_pose')):
                    pose = self.get_pose()
                    if pose and hasattr(pose, 'pose') and hasattr(pose.pose, 'position'):
                        pos = pose.pose.position
                        return pos.x, pos.y, pos.z
                        
                # Try state_pose if available
                if hasattr(self, 'state_pose') and self.state_pose:
                    pos = self.state_pose.pose.position
                    return pos.x, pos.y, pos.z
            except Exception as e:
                print(f"[{self.namespace}] Error accessing position methods: {e}")
            
            # If we have a current behavior, try to use it to get position
            if self.current_behavior and hasattr(self.current_behavior, 'status'):
                # If behavior is still active, return the target position
                # as our current position approximation
                if hasattr(self.current_behavior, '_goal_pose'):
                    pose = self.current_behavior._goal_pose
                    if pose and hasattr(pose, 'position'):
                        return pose.position.x, pose.position.y, pose.position.z
            
            # Fallback: return last known or default position
            return self.takeoff_x, self.takeoff_y, self.altitude
            
        except Exception as e:
            print(f"[{self.namespace}] Error getting position: {e}")
            # Return default position as fallback
            return self.takeoff_x, self.takeoff_y, self.altitude

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
        print(f"[{self.namespace}] do_behavior: {beh} with args {args}")
        self.current_behavior = getattr(self, beh)
        self.current_behavior(*args)

    def go_to_position(self, x, y, z, speed=1.0) -> None:
        """Command the drone to move to a specific position."""
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
    
    def update_position(self):
        """Continuously update drone position while avoiding obstacles."""
        self.running = True
        last_position = None
        last_update_time = 0
        min_update_interval = 0.03  # Reduced from 0.05 to 0.03 for more frequent updates
        position_threshold = 0.005  # Reduced threshold to 0.5cm for more responsive updates
        
        # Force an initial movement command directly toward goal
        try:
            # Calculate the end point (goal) coordinates
            end_x = self.stage_center[0] + self.end_point[0]
            end_y = self.stage_center[1] + self.end_point[1]
            
            # Calculate direction to goal
            x, y, z = self.position()
            current_pos = Point(x=x, y=y, z=z)
            
            # Print initial goal information
            print(f"[{self.namespace}] Starting mission toward goal: ({end_x:.2f}, {end_y:.2f})")
            print(f"[{self.namespace}] Current position: ({current_pos.x:.2f}, {current_pos.y:.2f})")
            
            # Calculate direct vector to goal
            dx = end_x - x
            dy = end_y - y
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist > 0.001:
                # Normalize and scale for direct initial movement (more aggressive)
                dx = dx / dist * 1.5  # Move 1.5 meters (reduced from 2.0) in goal direction
                dy = dy / dist * 1.5
                
                # Send initial direct movement command
                next_pos = Point(x=x + dx, y=y + dy, z=z)
                print(f"[{self.namespace}] Initial move directly toward goal: ({next_pos.x:.2f}, {next_pos.y:.2f})")
                self.go_to_position(next_pos.x, next_pos.y, next_pos.z, speed=2.5)  # Reduced speed for initial movement
                last_position = next_pos
                last_update_time = time.time()
                
                # Short delay to start movement
                time.sleep(0.05)
            
        except Exception as e:
            print(f"[{self.namespace}] Error in initial position update: {e}")
        
        # Counter for forcing movement
        updates_without_movement = 0
        max_updates_without_movement = 5  # Reduced from 10 to 5 to force updates more often
        
        # Shorter look-ahead for intermediate waypoints
        look_ahead_distance = 1.0  # Maximum distance for next waypoint (reduced from default)
        
        while self.running and not self.mission_complete:
            try:
                # Calculate next position using RVO
                target_position = self.calculate_position()
                current_time = time.time()
                
                if target_position:
                    # Get current position
                    x, y, z = self.position()
                    current_pos = Point(x=x, y=y, z=z)
                    
                    # Calculate vector to target
                    dx = target_position.x - current_pos.x
                    dy = target_position.y - current_pos.y
                    dist_to_target = math.sqrt(dx*dx + dy*dy)
                    
                    # Limit distance for more intermediate steps
                    if dist_to_target > look_ahead_distance:
                        # Normalize and scale
                        dx = dx / dist_to_target * look_ahead_distance
                        dy = dy / dist_to_target * look_ahead_distance
                        # Create intermediate waypoint
                        target_position = Point(
                            x=current_pos.x + dx,
                            y=current_pos.y + dy,
                            z=current_pos.z
                        )
                    
                    # Check if we need to update position based on time and distance
                    update_position = False
                    
                    # Update if it's the first position, if minimum time has passed, or periodically force updates
                    if (last_position is None or 
                        (current_time - last_update_time >= min_update_interval) or
                        updates_without_movement >= max_updates_without_movement):
                        
                        # Calculate distance to previous position
                        if last_position:
                            dx = target_position.x - last_position.x
                            dy = target_position.y - last_position.y
                            distance = math.sqrt(dx*dx + dy*dy)
                            
                            # Update if moved more than threshold, it's been a while, or force update
                            if (distance > position_threshold or 
                                (current_time - last_update_time >= 0.15) or  # Reduced from 0.2
                                updates_without_movement >= max_updates_without_movement):
                                update_position = True
                                updates_without_movement = 0
                            else:
                                updates_without_movement += 1
                        else:
                            # First position
                            update_position = True
                    
                    if update_position:
                        # Adjust speed based on distance to goal
                        # Calculate distance to goal
                        end_x = self.stage_center[0] + self.end_point[0]
                        end_y = self.stage_center[1] + self.end_point[1]
                        dx_goal = end_x - x
                        dy_goal = end_y - y
                        dist_to_goal = math.sqrt(dx_goal*dx_goal + dy_goal*dy_goal)
                        
                        # Adjust speed to be lower for closer obstacles
                        obstacle_nearby = False
                        min_obstacle_dist = float('inf')
                        
                        # Check distance to nearest obstacles
                        for obs_id, obs_pos in self.dynamic_obstacles.items():
                            if obs_pos is not None:
                                dx_obs = obs_pos.x - x
                                dy_obs = obs_pos.y - y
                                dist_obs = math.sqrt(dx_obs*dx_obs + dy_obs*dy_obs)
                                min_obstacle_dist = min(min_obstacle_dist, dist_obs)
                                if dist_obs < 1.5:  # If obstacle within 1.5m
                                    obstacle_nearby = True
                        
                        # Set speed based on obstacles and goal
                        if obstacle_nearby:
                            # Slower speed when obstacles are nearby
                            # Scale speed inversely with obstacle proximity
                            speed_factor = min(1.0, max(0.4, min_obstacle_dist / 3.0))
                            speed = 1.0 * speed_factor
                        else:
                            # Full speed when no obstacles nearby
                            speed = 2.0
                        
                        # Move to the new position
                        print(f"[{self.namespace}] Moving to ({target_position.x:.2f}, {target_position.y:.2f}), "
                              f"distance to goal: {dist_to_goal:.2f}m, speed: {speed:.2f}")
                        self.go_to_position(target_position.x, target_position.y, target_position.z, speed=speed)
                        last_position = target_position
                        last_update_time = current_time
                    
                # Very short wait time for position updates
                time.sleep(0.01)
                
            except Exception as e:
                print(f"[{self.namespace}] Error in update_position: {e}")
                time.sleep(0.05)  # Short wait time before retrying
            
            time.sleep(0.01)  # Higher frequency update checks
    
    def start(self):
        """Start the drone's position update thread."""
        if self.update_thread is None or not self.update_thread.is_alive():
            self.update_thread = threading.Thread(target=self.update_position)
            self.update_thread.daemon = True
            self.update_thread.start()
            print(f"[{self.namespace}] Started position updates")
    
    def stop(self):
        """Stop position updates."""
        self.running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join()
            print(f"[{self.namespace}] Stopped position updates")

    def get_current_pose(self):
        """Get the drone's current pose."""
        try:
            # Check if we have a behavior with pose
            if hasattr(self, 'state'):
                return self.state.pose
            
            # Create a default pose at the drone's assigned position
            pose = PoseStamped()
            pose.header.frame_id = "earth"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.position = Point(x=self.takeoff_x, y=self.takeoff_y, z=self.altitude)
            
            return pose
        except Exception as e:
            print(f"[{self.namespace}] Error getting pose: {e}")
            # Return default pose at takeoff position
            pose = PoseStamped()
            pose.position = Point(x=self.takeoff_x, y=self.takeoff_y, z=self.altitude)
            return pose

class FollowerDrone(DroneInterface):
    def __init__(self, node_name='follower_drone', fixed_frame='world'):
        super().__init__(node_name, fixed_frame)
        self.leader_state = None
        self.neighbor_states = {}
        self.dynamic_obstacles = {}  # Dictionary to store positions of dynamic obstacles
        self.base_speed = 1.0        # Base drone movement speed
        self.formation_spacing = 2.0  # Spacing between drones in formation

        # ORCA parameters
        self.orca = ORCA(radius=0.6, time_horizon=1.5, max_speed=2.0)  
        self.obstacle_weight = 2.0    # Weight for avoiding obstacles
        self.neighbor_weight = 2.0    # Weight for avoiding neighbors
        self.collision_radius = 0.6   # Radius for collision detection

        # Subscribe to dynamic obstacle positions
        self.obstacle_subscription = self.create_subscription(
            PoseArray,
            '/dynamic_obstacles/locations',
            self.obstacle_callback,
            10
        )

        self.get_logger().info(f'{node_name} initialized')

    def obstacle_callback(self, msg):
        """Callback for dynamic obstacle positions"""
        obstacle_id = msg.header.frame_id
        position = msg.pose.position
        
        # Calculate velocity if we have previous position
        velocity = Vector3()
        if obstacle_id in self.dynamic_obstacles:
            prev_pos = self.dynamic_obstacles[obstacle_id]['position']
            prev_time = self.dynamic_obstacles[obstacle_id]['time']
            current_time = time.time()
            dt = current_time - prev_time
            
            # Only compute velocity if time difference is reasonable
            if dt > 0.01:
                velocity.x = (position.x - prev_pos.x) / dt
                velocity.y = (position.y - prev_pos.y) / dt
                velocity.z = (position.z - prev_pos.z) / dt
        
        # Store obstacle information
        self.dynamic_obstacles[obstacle_id] = {
            'position': position,
            'velocity': velocity,
            'time': time.time()
        }
    
    def calculate_position(self):
        """Calculate position based on leader's state and RVO collision avoidance."""
        # Get current position and velocity
        current_pos = self.position  # Using property from Drone class
        current_vel = Vector3(x=0.0, y=0.0, z=0.0)
        if hasattr(self, 'velocity'):
            current_vel = self.velocity
        
        # Check if we have leader's state
        if not self.leader_state:
            self.get_logger().warn("No leader state available")
            return current_pos
        
        # Calculate desired position in world frame (leader + formation offset)
        leader_pos = Point(
            x=self.leader_state['x'],
            y=self.leader_state['y'],
            z=self.leader_state['z']
        )
        
        # Calculate formation position in world frame
        formation_pos = Point(
            x=leader_pos.x + self.formation_spacing * self.leader_state['vx'],
            y=leader_pos.y + self.formation_spacing * self.leader_state['vy'],
            z=leader_pos.z + self.formation_spacing * self.leader_state['vz']
        )
        
        # Vector from current position to formation position
        desired_vel = Vector3(
            x=formation_pos.x - current_pos.x,
            y=formation_pos.y - current_pos.y,
            z=formation_pos.z - current_pos.z
        )
        
        # Limit desired velocity magnitude
        desired_speed = math.sqrt(desired_vel.x**2 + desired_vel.y**2)
        if desired_speed > self.collision_radius:
            scaling = self.collision_radius / desired_speed
            desired_vel.x *= scaling
            desired_vel.y *= scaling
        
        # Get neighbor drones (other followers)
        neighbors = []
        for follower in self.leader_state['followers']:
            if follower != self:
                follower_pos = follower['position']
                follower_vel = Vector3(x=0.0, y=0.0, z=0.0)
                if 'velocity' in follower:
                    follower_vel = follower['velocity']
                neighbors.append((follower_pos, follower_vel))
        
        # Add leader as a neighbor too
        neighbors.append((leader_pos, Vector3(
            x=self.leader_state['vx'],
            y=self.leader_state['vy'],
            z=self.leader_state['vz']
        )))
        
        # Add dynamic obstacles as neighbors
        current_time = time.time()
        for obstacle_id, obstacle_data in self.dynamic_obstacles.items():
            # Only consider recent obstacle data (within 1 second)
            if current_time - obstacle_data['time'] < 1.0:
                neighbors.append((obstacle_data['position'], obstacle_data['velocity']))
        
        # Use RVO to compute collision-free velocity
        new_vel = self.orca.compute_velocity(current_pos, current_vel, desired_vel, neighbors, formation_pos)
        
        # Calculate new position based on the computed velocity
        dt = 0.1  # Time step for position update
        new_pos = Point(
            x=current_pos.x + new_vel.x * dt,
            y=current_pos.y + new_vel.y * dt,
            z=current_pos.z + new_vel.z * dt
        )
        
        # Store computed velocity for next iteration
        self.velocity = new_vel
        
        # Debug print occasionally
        if random.random() < 0.01:
            self.get_logger().info(f"Desired vel: ({desired_vel.x:.2f}, {desired_vel.y:.2f}), " +
                                  f"RVO vel: ({new_vel.x:.2f}, {new_vel.y:.2f}), " +
                                  f"Neighbors: {len(neighbors)}, Obstacles: {len(self.dynamic_obstacles)}")
        
        return new_pos

def run_drone(args):
    """Run drones for the stage 4 mission."""
    # Initialize list of drone objects
    drones = []
    
    # Get all namespaces for formation calculation
    all_namespaces = args.namespaces
    
    # Create and initialize drones
    for namespace in all_namespaces:
        drone = Drone(
            namespace,
            args.config,
            verbose=args.verbose, 
            use_sim_time=args.use_sim_time
        )
        # Set all drone namespaces for formation calculation
        drone.all_drone_namespaces = all_namespaces.copy()
        drones.append(drone)
    
    # Set up neighbor tracking between drones
    for i, drone1 in enumerate(drones):
        for j, drone2 in enumerate(drones):
            if i != j:  # Skip self
                # Initialize neighbor state
                # Will be updated with actual positions in the main loop
                drone1.neighbor_states[drone2.namespace] = {
                    'position': Point(x=0.0, y=0.0, z=0.0),
                    'velocity': Vector3(x=0.0, y=0.0, z=0.0)
                }
    
    # Print mission information
    print("\n=== Mission Configuration ===")
    print(f"Number of drones: {len(drones)}")
    leader = drones[0]
    start_x = leader.stage_center[0] + leader.start_point[0]
    start_y = leader.stage_center[1] + leader.start_point[1]
    end_x = leader.stage_center[0] + leader.end_point[0]
    end_y = leader.stage_center[1] + leader.end_point[1]
    print(f"Stage center: ({leader.stage_center[0]:.2f}, {leader.stage_center[1]:.2f})")
    print(f"Start point: ({start_x:.2f}, {start_y:.2f})")
    print(f"End point: ({end_x:.2f}, {end_y:.2f})")
    print(f"Formation spacing: {leader.formation_spacing} meters")
    print("===========================\n")
    
    try:
        # Arm and takeoff all drones
        for drone in drones:
            # Arm and set offboard mode
            if not drone.arm() or not drone.offboard():
                print(f"Failed to arm/offboard drone {drone.namespace}!")
                for d in drones:
                    d.shutdown()
                return
            
            # Takeoff
            print(f"[{drone.namespace}] Taking off to position ({drone.takeoff_x:.2f}, {drone.takeoff_y:.2f}, {drone.altitude:.2f})...")
            drone.do_behavior("takeoff", drone.altitude, 0.7, False)
            drone.change_led_colour((0, 255, 0))
            
        # Wait for all drones to complete takeoff
        for drone in drones:
            while not drone.goal_reached():
                time.sleep(0.1)
        
        # Remove the step where drones move to assigned positions and directly start the mission
        # Provide specific instructions for mission
        print("\n=== MISSION START ===")
        print(f"Drones will start from ({start_x:.2f}, {start_y:.2f}) and move to ({end_x:.2f}, {end_y:.2f})")
        print("Drones will maintain a V-formation while avoiding obstacles")
        print("Collision avoidance has priority over formation maintenance")
        print("===================\n")
        
        # Ensure all drones know their start positions
        for drone in drones:
            x, y, z = drone.position()
            print(f"[{drone.namespace}] Starting from position: ({x:.2f}, {y:.2f}, {z:.2f})")
            print(f"[{drone.namespace}] Moving toward goal: ({end_x:.2f}, {end_y:.2f})")
            
            # Calculate and print distance to goal
            dx = end_x - x
            dy = end_y - y
            dist = math.sqrt(dx*dx + dy*dy)
            print(f"[{drone.namespace}] Distance to goal: {dist:.2f}m")
        
        # Start position updates for all drones
        for drone in drones:
            drone.start()
        
        # Keep updating neighbor states while mission is running
        all_complete = False
        update_count = 0
        mission_start_time = time.time()
        
        while not all_complete and rclpy.ok():
            # Update neighbor states
            for i, drone1 in enumerate(drones):
                for j, drone2 in enumerate(drones):
                    if i != j:  # Skip self
                        try:
                            # Get current position
                            pos_x, pos_y, pos_z = drone2.position()
                            
                            # Update neighbor state with current position
                            position = Point(x=pos_x, y=pos_y, z=pos_z)
                            velocity = Vector3(x=0.0, y=0.0, z=0.0)  # Default velocity
                            
                            drone1.neighbor_states[drone2.namespace] = {
                                'position': position,
                                'velocity': velocity
                            }
                        except Exception as e:
                            print(f"Error updating neighbor state: {e}")
                            # Keep existing state if update fails
            
            # Print status every ~5 seconds (25 * 0.2s sleep)
            update_count += 1
            if update_count % 25 == 0:
                mission_time = time.time() - mission_start_time
                print(f"\n=== Formation Status (Time: {mission_time:.1f}s) ===")
                
                # Print leader's distance to goal
                leader = drones[0]
                leader_x, leader_y, _ = leader.position()
                dx = end_x - leader_x
                dy = end_y - leader_y
                dist_to_goal = math.sqrt(dx*dx + dy*dy)
                print(f"Leader distance to goal: {dist_to_goal:.2f}m")
                
                # Print each drone's formation stats
                for drone in drones:
                    x, y, z = drone.position()
                    form_pos = drone.get_formation_position()
                    dx = form_pos.x - x
                    dy = form_pos.y - y
                    dist = math.sqrt(dx*dx + dy*dy)
                    print(f"{drone.namespace}: Pos=({x:.2f},{y:.2f}) FormPos=({form_pos.x:.2f},{form_pos.y:.2f}) Dist={dist:.2f}m")
                
                # Count obstacles detected
                all_obstacles = set()
                for drone in drones:
                    all_obstacles.update(drone.dynamic_obstacles.keys())
                print(f"Detected obstacles: {len(all_obstacles)}")
                print("=======================\n")
            
            # Check if all drones have completed their mission
            all_complete = True
            for drone in drones:
                if not drone.mission_complete:
                    all_complete = False
                    break
            
            time.sleep(0.2)
        
        # Mission complete - report success
        mission_time = time.time() - mission_start_time
        print(f"\n*** MISSION COMPLETE! Time: {mission_time:.1f} seconds ***")
        
        # Land all drones
        for drone in drones:
            print(f"[{drone.namespace}] Mission complete, landing...")
            drone.stop()
            drone.do_behavior("land", 0.4, False)
            drone.change_led_colour((255, 0, 0))
        
        # Wait for all drones to complete landing
        for drone in drones:
            while not drone.goal_reached():
                time.sleep(0.1)
        
        print("\n*** ALL DRONES HAVE COMPLETED THE MISSION ***\n")
    
    except KeyboardInterrupt:
        print("Operation interrupted by user")
    finally:
        for drone in drones:
            drone.stop()
            drone.shutdown()

def main():
    parser = argparse.ArgumentParser(description="Stage 4: Dynamic Obstacle Avoidance")
    parser.add_argument('-n', '--namespaces', nargs='+',
                        default=['drone0'],
                        help='List of drone namespaces')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Enable verbose output')
    parser.add_argument('-s', '--use_sim_time', action='store_true', default=True,
                        help='Use simulation time')
    parser.add_argument('-c', '--config', type=str, 
                        default='/home/ziyar/project_gazebo_ws/src/challenge_multi_drone/scenarios/scenario1_stage4.yaml',
                        help='Path to scenario configuration file')
    
    args = parser.parse_args()
    
    # Initialize ROS
    rclpy.init()
    
    # Run the drone
    run_drone(args)
    
    # Shutdown ROS
    rclpy.shutdown()

if __name__ == "__main__":
    main() 