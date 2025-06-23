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
from geometry_msgs.msg import PoseStamped, Point, Vector3
from typing import List, Tuple, Optional

# --- RVO Implementation with Formation Bias ---
class RVO:
    """Reciprocal Velocity Obstacles with formation bias"""
    def __init__(self, radius=0.3, time_horizon=1.0, max_speed=2.0, 
                 formation_weight=0.5, neighbor_weight=2.0, obstacle_weight=2.0):
        self.radius = radius  # Collision radius
        self.time_horizon = time_horizon  # Time horizon for collision detection
        self.max_speed = max_speed  # Maximum speed
        
        # Weights for different constraints
        self.formation_weight = formation_weight  # Weight for formation bias - reduced from 1.0 to 0.5
        self.neighbor_weight = neighbor_weight  # Weight for neighbor avoidance - increased from 1.0 to 2.0
        self.obstacle_weight = obstacle_weight  # Weight for obstacle avoidance - increased from 1.5 to 2.0
        
        # Number of samples for RVO
        self.num_samples = 150  # Increased from 100 to 150 for better sampling
        
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
        
        # RVO collision avoidance - updated parameters as requested
        self.rvo = RVO(radius=0.6, time_horizon=1.5, max_speed=2.0, 
                       formation_weight=0.5, neighbor_weight=2.0, obstacle_weight=2.0)
        
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
        
        # Position update thread
        self.update_thread = None

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
            
            # Calculate desired position (end point relative to stage center)
            end_x = self.stage_center[0] + self.end_point[0]
            end_y = self.stage_center[1] + self.end_point[1]
            
            # Calculate direction and distance to target
            dx = end_x - current_pos.x
            dy = end_y - current_pos.y
            distance_to_goal = math.sqrt(dx*dx + dy*dy)
            
            # Print debugging info occasionally
            if random.random() < 0.01:  # 1% chance to print
                print(f"[{self.namespace}] Current position: ({current_pos.x:.2f}, {current_pos.y:.2f})")
                print(f"[{self.namespace}] Goal position: ({end_x:.2f}, {end_y:.2f})")
                print(f"[{self.namespace}] Distance to goal: {distance_to_goal:.2f}m")
            
            # Calculate desired velocity (towards end point)
            # Stronger direct-to-goal factor to ensure forward progress
            direct_to_goal_factor = 0.95  # Increased from 0.8 to 0.95 to prioritize goal movement
            
            # Scale speed based on distance, but ensure significant movement
            # Note: no longer scaling speed down for short distances
            speed_factor = 1.0  # Always use full speed
            
            # Direct velocity toward goal with minimum speed to ensure movement
            min_speed = 0.8  # Increased from 0.3 to 0.8 for more aggressive movement
            if distance_to_goal > 0.5:  # Only apply minimum if not close to goal
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
                    
                    # Apply full speed - always move aggressively toward goal
                    goal_speed = max(self._speed * 1.5, min_speed)  # Increased speed multiplier
                    goal_vel = Vector3(
                        x=norm_dx * goal_speed,
                        y=norm_dy * goal_speed,
                        z=0.0
                    )
            else:
                goal_vel = Vector3(
                    x=dx * self._speed * speed_factor,
                    y=dy * self._speed * speed_factor,
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
            formation_factor = min(0.5, form_dist / 5.0) * (1.0 - direct_to_goal_factor)
            
            # Combined desired velocity - heavily prioritizing goal
            desired_vel = Vector3(
                x=goal_vel.x + form_dx * self._speed * formation_factor * 0.5,  # Reduced formation influence
                y=goal_vel.y + form_dy * self._speed * formation_factor * 0.5,  # Reduced formation influence
                z=0.0
            )
            
            # Check if we're close enough to the goal
            if distance_to_goal < 0.5:  # Within 0.5m of end point
                self.mission_complete = True
                print(f"[{self.namespace}] Reached end point! Distance: {distance_to_goal:.2f}m")
                return current_pos  # Just stay in place
                        
            # Get neighbor information for RVO
            neighbors = []
            for ns, state in self.neighbor_states.items():
                if state is not None and 'position' in state:
                    neighbors.append((state['position'], state['velocity']))
            
            # Add dynamic obstacles as neighbors with higher priority
            for obs_id, obs_pos in self.dynamic_obstacles.items():
                if obs_pos is not None:
                    # Add the obstacle - but only once to reduce constraint priority
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
                # If velocity is too low, inject momentum in goal direction
                new_vel.x += dx / distance_to_goal * 0.2 if distance_to_goal > 0.001 else 0.0
                new_vel.y += dy / distance_to_goal * 0.2 if distance_to_goal > 0.001 else 0.0
            
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
        min_update_interval = 0.05  # Reduced from 0.1 to 0.05 for more frequent updates
        position_threshold = 0.01  # Reduced threshold to 1cm for more responsive updates
        
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
                dx = dx / dist * 2.0  # Move 2 meters in goal direction
                dy = dy / dist * 2.0
                
                # Send initial direct movement command
                next_pos = Point(x=x + dx, y=y + dy, z=z)
                print(f"[{self.namespace}] Initial move directly toward goal: ({next_pos.x:.2f}, {next_pos.y:.2f})")
                self.go_to_position(next_pos.x, next_pos.y, next_pos.z, speed=3.0)  # Higher speed for initial movement
                last_position = next_pos
                last_update_time = time.time()
                
                # Short delay to start movement
                time.sleep(0.05)
            
        except Exception as e:
            print(f"[{self.namespace}] Error in initial position update: {e}")
        
        # Counter for forcing movement
        updates_without_movement = 0
        
        while self.running and not self.mission_complete:
            try:
                # Calculate next position using RVO
                position = self.calculate_position()
                current_time = time.time()
                
                if position:
                    # Check if we need to update position based on time and distance
                    update_position = False
                    
                    # Update if it's the first position, if minimum time has passed, or periodically force updates
                    if (last_position is None or 
                        (current_time - last_update_time >= min_update_interval) or
                        updates_without_movement >= 10):  # Force update after 10 cycles without movement
                        
                        # Calculate distance to previous position
                        if last_position:
                            dx = position.x - last_position.x
                            dy = position.y - last_position.y
                            distance = math.sqrt(dx*dx + dy*dy)
                            
                            # Update if moved more than threshold, it's been a while, or force update
                            if (distance > position_threshold or 
                                (current_time - last_update_time >= 0.2) or
                                updates_without_movement >= 10):
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
                        x, y, z = self.position()
                        dx_goal = end_x - x
                        dy_goal = end_y - y
                        dist_to_goal = math.sqrt(dx_goal*dx_goal + dy_goal*dy_goal)
                        
                        # Always use high speed
                        speed = 2.0
                        
                        # Move to the new position
                        print(f"[{self.namespace}] Moving to ({position.x:.2f}, {position.y:.2f}), distance to goal: {dist_to_goal:.2f}m")
                        self.go_to_position(position.x, position.y, position.z, speed=speed)
                        last_position = position
                        last_update_time = current_time
                    
                    # Very short wait time for position updates
                    time.sleep(0.01)
                
            except Exception as e:
                print(f"[{self.namespace}] Error in update_position: {e}")
                time.sleep(0.1)  # Even shorter wait time before retrying
            
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

class FollowerDrone(Drone):
    """Drone that follows a leader in formation."""
    def __init__(self, name, formation_offset, leader, color=ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)):
        super().__init__(name, color)
        self.formation_offset = formation_offset
        self.leader = leader
        
        # Initialize RVO
        self.rvo = RVO(radius=0.35, time_horizon=1.5, max_speed=2.0)
        
        # For dynamic obstacles tracking
        self.dynamic_obstacles = {}  # {id: {'position': Point, 'velocity': Vector3, 'time': float}}
        
        # Subscribe to dynamic obstacles
        self.obstacles_subscription = self.create_subscription(
            PoseStamped,
            '/dynamic_obstacles/locations',
            self.obstacle_callback,
            10
        )
    
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
        if not self.leader.state:
            self.get_logger().warn("No leader state available")
            return current_pos
        
        # Calculate desired position in world frame (leader + formation offset)
        leader_pos = Point(
            x=self.leader.state['x'],
            y=self.leader.state['y'],
            z=self.leader.state['z']
        )
        
        # Calculate formation position in world frame
        formation_pos = Point(
            x=leader_pos.x + self.formation_offset[0],
            y=leader_pos.y + self.formation_offset[1],
            z=leader_pos.z + self.formation_offset[2]
        )
        
        # Vector from current position to formation position
        desired_vel = Vector3(
            x=formation_pos.x - current_pos.x,
            y=formation_pos.y - current_pos.y,
            z=formation_pos.z - current_pos.z
        )
        
        # Limit desired velocity magnitude
        desired_speed = math.sqrt(desired_vel.x**2 + desired_vel.y**2)
        if desired_speed > self.rvo.max_speed:
            scaling = self.rvo.max_speed / desired_speed
            desired_vel.x *= scaling
            desired_vel.y *= scaling
        
        # Get neighbor drones (other followers)
        neighbors = []
        for follower in self.leader.followers:
            if follower != self:
                follower_pos = follower.position
                follower_vel = Vector3(x=0.0, y=0.0, z=0.0)
                if hasattr(follower, 'velocity'):
                    follower_vel = follower.velocity
                neighbors.append((follower_pos, follower_vel))
        
        # Add leader as a neighbor too
        neighbors.append((leader_pos, Vector3(
            x=self.leader.state.get('vx', 0.0),
            y=self.leader.state.get('vy', 0.0),
            z=self.leader.state.get('vz', 0.0)
        )))
        
        # Add dynamic obstacles as neighbors
        current_time = time.time()
        for obstacle_id, obstacle_data in self.dynamic_obstacles.items():
            # Only consider recent obstacle data (within 1 second)
            if current_time - obstacle_data['time'] < 1.0:
                neighbors.append((obstacle_data['position'], obstacle_data['velocity']))
        
        # Use RVO to compute collision-free velocity
        new_vel = self.rvo.compute_velocity(current_pos, current_vel, desired_vel, neighbors, formation_pos)
        
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