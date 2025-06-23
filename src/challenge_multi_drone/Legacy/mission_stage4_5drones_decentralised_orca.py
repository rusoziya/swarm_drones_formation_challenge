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

# --- ORCA Implementation with Formation Constraints ---
class ORCA:
    def __init__(self, radius=0.5, time_horizon=2.0, max_speed=2.5):
        self.radius = radius  # Collision radius
        self.time_horizon = time_horizon  # Time horizon for velocity obstacles
        self.max_speed = max_speed  # Maximum speed of drones
        self.formation_weight = 0.4  # Reduced weight for formation constraints (0-1)
        self.avoidance_weight = 1.0  # Maximum weight for collision avoidance
        self.obstacle_safe_distance = 1.2  # Increase distance for earlier avoidance
        
    def compute_velocity(self, current_pos, current_vel, desired_vel, neighbors, formation_pos=None):
        """
        Compute new velocity using ORCA for collision avoidance with formation constraints.
        
        Args:
            current_pos: Current position (Point)
            current_vel: Current velocity (Vector3)
            desired_vel: Desired velocity (Vector3) - typically toward goal
            neighbors: List of (position, velocity) tuples for other drones/obstacles
            formation_pos: Desired formation position (Point) or None
            
        Returns:
            New velocity (Vector3)
        """
        try:
            # Convert to numpy arrays for easier math
            pos = np.array([current_pos.x, current_pos.y])
            vel = np.array([current_vel.x, current_vel.y])
            desired = np.array([desired_vel.x, desired_vel.y])
            
            # Initialize with desired velocity (goal-directed)
            new_vel = desired.copy()
            
            # Track if we need to take avoidance action
            avoidance_needed = False
            strongest_avoid_vector = None
            strongest_avoid_magnitude = 0
            
            # Process all neighbors for collision avoidance
            for neighbor_pos, neighbor_vel in neighbors:
                try:
                    # Skip invalid neighbors
                    if neighbor_pos is None:
                        continue
                    
                    # Get neighbor position as numpy array
                    n_pos = np.array([neighbor_pos.x, neighbor_pos.y])
                    
                    # Calculate relative position vector
                    rel_pos = pos - n_pos
                    
                    # Skip if same position
                    if np.linalg.norm(rel_pos) < 0.01:
                        continue
                    
                    # Calculate distance to neighbor
                    dist = np.linalg.norm(rel_pos)
                    
                    # Increased collision radius to make drones more responsive
                    collision_radius = 2 * self.radius + self.obstacle_safe_distance  # More conservative
                    
                    # If we're too close to a neighbor
                    if dist < collision_radius:
                        avoidance_needed = True
                        
                        # Calculate avoidance vector (away from neighbor)
                        avoid_dir = rel_pos / dist  # Normalized direction
                        
                        # Scale based on how close we are (closer = stronger)
                        # Exponential scaling for more aggressive avoidance when very close
                        closeness_factor = 1.0 - (dist / collision_radius)
                        avoid_scale = min(self.max_speed * 2.0, 
                                          self.max_speed * (closeness_factor ** 2) * 5.0)
                        
                        # Keep track of strongest avoidance action
                        if avoid_scale > strongest_avoid_magnitude:
                            strongest_avoid_magnitude = avoid_scale
                            strongest_avoid_vector = avoid_dir * avoid_scale
                        
                        # Print debug info for close obstacles
                        if closeness_factor > 0.7 and random.random() < 0.1:  # 10% chance when very close
                            print(f"CRITICAL AVOIDANCE: dist={dist:.2f}m, radius={collision_radius:.2f}m")
                        
                except Exception as e:
                    # Skip this neighbor if there's a problem
                    continue
            
            # Apply formation constraints using potential wells if formation position is provided
            formation_vector = np.zeros(2)
            if formation_pos is not None and not avoidance_needed:
                # Only apply formation forces if not actively avoiding obstacles
                # Convert formation position to numpy array
                form_pos = np.array([formation_pos.x, formation_pos.y])
                
                # Calculate vector to formation position (formation well)
                form_vec = form_pos - pos
                form_dist = np.linalg.norm(form_vec)
                
                # Only apply formation force if we're not at the formation position
                if form_dist > 0.01:
                    # Normalize and scale by distance (quadratic potential well)
                    # Further from formation = stronger pull
                    form_dir = form_vec / form_dist
                    form_scale = min(self.max_speed, form_dist * 0.8)
                    formation_vector = form_dir * form_scale
                    
                    # Periodically print formation info
                    if random.random() < 0.01:  # 1% chance
                        print(f"Formation pull: distance={form_dist:.2f}m, magnitude={form_scale:.2f}")
            
            # Combine avoidance, formation, and goal-directed velocities
            if avoidance_needed and strongest_avoid_vector is not None:
                # If collision avoidance is needed, heavily prioritize avoidance
                # Minor pull toward goal direction
                new_vel = (strongest_avoid_vector * self.avoidance_weight + 
                          desired * 0.1)  # Very little weight to goal when avoiding
                
                # Periodically print debug info
                if random.random() < 0.05:  # 5% chance
                    print(f"Collision avoidance active! Magnitude: {strongest_avoid_magnitude:.2f}")
            else:
                # No immediate collision danger, blend formation constraints with goal
                # Make formation constraints soft
                new_vel = desired * (1 - self.formation_weight) + formation_vector * self.formation_weight
            
            # Limit speed
            speed = np.linalg.norm(new_vel)
            if speed > self.max_speed:
                new_vel = new_vel / speed * self.max_speed
                
            # Convert back to Vector3
            return Vector3(x=new_vel[0], y=new_vel[1], z=current_vel.z)
        
        except Exception as e:
            # If anything goes wrong, just return the desired velocity
            print(f"Error in compute_velocity: {e}")
            return desired_vel

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
        self._speed = 1.0  # Increased from 0.5 for better maneuverability
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
        self.formation_spacing = 1.5  # Spacing between drones in formation
        self.all_drone_namespaces = []  # Will be set in run_drone function
        
        # ORCA collision avoidance - increase radius and time horizon for more responsive avoidance
        self.orca = ORCA(radius=0.5, time_horizon=2.0, max_speed=2.5)
        
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
        
        # Track neighbor states for ORCA
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
        """Calculate next position using ORCA for collision avoidance and formation constraints."""
        try:
            # Get current position and velocity
            x, y, z = self.position()
            current_pos = Point(x=x, y=y, z=z)
            
            # Get current velocity (default to zero if not available)
            current_vel = Vector3(x=0.0, y=0.0, z=0.0)
            
            # Calculate desired position (end point)
            end_x = self.stage_center[0] + self.end_point[0]
            end_y = self.stage_center[1] + self.end_point[1]
            
            # Calculate direction and distance to target
            dx = end_x - current_pos.x
            dy = end_y - current_pos.y
            distance_to_goal = math.sqrt(dx*dx + dy*dy)
            
            # Calculate desired velocity (towards end point)
            # Always head directly toward the goal 
            direct_to_goal_factor = 0.6  # Strong factor toward goal
            speed_factor = min(1.0, distance_to_goal / 2.0) * direct_to_goal_factor
            
            # Direct velocity toward goal
            goal_vel = Vector3(
                x=dx * self._speed * speed_factor,
                y=dy * self._speed * speed_factor,
                z=0.0  # Maintain current altitude
            )
            
            # Get this drone's desired formation position
            formation_pos = self.get_formation_position()
            
            # Calculate velocity component toward formation position
            form_dx = formation_pos.x - current_pos.x
            form_dy = formation_pos.y - current_pos.y
            form_dist = math.sqrt(form_dx*form_dx + form_dy*form_dy)
            
            # Combine direct-to-goal with formation velocity
            # The closer we are to formation position, the more we prioritize going directly to goal
            formation_factor = min(1.0, form_dist / 3.0) * (1.0 - direct_to_goal_factor)
            
            # Combined desired velocity
            desired_vel = Vector3(
                x=goal_vel.x + form_dx * self._speed * formation_factor,
                y=goal_vel.y + form_dy * self._speed * formation_factor,
                z=0.0
            )
            
            # Periodically print formation information
            if random.random() < 0.01:  # 1% chance to print
                print(f"[{self.namespace}] Formation position: ({formation_pos.x:.2f}, {formation_pos.y:.2f})")
                print(f"[{self.namespace}] Current position: ({current_pos.x:.2f}, {current_pos.y:.2f})")
                print(f"[{self.namespace}] Distance to formation position: {form_dist:.2f}m")
                print(f"[{self.namespace}] Distance to goal: {distance_to_goal:.2f}m")
                print(f"[{self.namespace}] Goal velocity: ({goal_vel.x:.2f}, {goal_vel.y:.2f})")
                print(f"[{self.namespace}] Combined velocity: ({desired_vel.x:.2f}, {desired_vel.y:.2f})")
            
            # Check if we're close enough to the goal
            if distance_to_goal < 0.5:  # Within 0.5m of end point
                self.mission_complete = True
                print(f"[{self.namespace}] Reached end point! Distance: {distance_to_goal:.2f}m")
                return current_pos  # Just stay in place
                        
            # Get neighbor information for ORCA
            neighbors = []
            for ns, state in self.neighbor_states.items():
                if state is not None and 'position' in state:
                    neighbors.append((state['position'], state['velocity']))
            
            # Add dynamic obstacles as neighbors with higher priority
            for obs_id, obs_pos in self.dynamic_obstacles.items():
                if obs_pos is not None:
                    # Add the obstacle multiple times to increase its priority in collision avoidance
                    neighbors.append((obs_pos, Vector3(x=0.0, y=0.0, z=0.0)))
                    neighbors.append((obs_pos, Vector3(x=0.0, y=0.0, z=0.0)))  # Add twice for higher weight
            
            # Compute new velocity using ORCA with formation constraints
            new_vel = self.orca.compute_velocity(
                current_pos,
                current_vel,
                desired_vel,
                neighbors,
                formation_pos  # Include formation position for potential wells
            )
            
            # Calculate new position based on new velocity
            dt = 0.1  # Time step
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
            # Try to use native DroneInterface methods to get position
            # This will vary based on the specific implementation
            x, y, z = self.takeoff_x, self.takeoff_y, self.altitude
            
            # If we have a current behavior, try to use it to get position
            if self.current_behavior and hasattr(self.current_behavior, 'status'):
                # If behavior is still active, return the target position
                # as our current position approximation
                if hasattr(self.current_behavior, '_goal_pose'):
                    pose = self.current_behavior._goal_pose
                    if pose and hasattr(pose, 'position'):
                        x, y, z = pose.position.x, pose.position.y, pose.position.z
            
            return x, y, z
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
        min_update_interval = 0.2  # Minimum time between position updates (seconds)
        position_threshold = 0.05  # 5cm threshold for position updates
        
        while self.running and not self.mission_complete:
            try:
                # Calculate next position using ORCA
                position = self.calculate_position()
                current_time = time.time()
                
                if position:
                    # Check if we need to update position based on time and distance
                    update_position = False
                    
                    # Update if it's the first position or if minimum time has passed
                    if last_position is None or (current_time - last_update_time >= min_update_interval):
                        # Calculate distance to previous position
                        if last_position:
                            dx = position.x - last_position.x
                            dy = position.y - last_position.y
                            distance = math.sqrt(dx*dx + dy*dy)
                            
                            # Update if moved more than threshold or it's been a while
                            if distance > position_threshold or (current_time - last_update_time >= 0.5):
                                update_position = True
                        else:
                            # First position
                            update_position = True
                    
                    if update_position:
                        # Adjust speed based on distance to move
                        if last_position:
                            dx = position.x - last_position.x
                            dy = position.y - last_position.y
                            distance = math.sqrt(dx*dx + dy*dy)
                            
                            # Scale speed with distance, but keep it between 0.8 and 2.0
                            speed = min(max(distance * 2.0, 0.8), 2.0)
                        else:
                            speed = 1.0
                        
                        # Move to the new position
                        self.go_to_position(position.x, position.y, position.z, speed=speed)
                        last_position = position
                        last_update_time = current_time
                    
                    # Quick check if we've reached position, with a short timeout
                    if current_time - last_update_time < 0.5:  # Only wait if we recently sent a command
                        wait_start = time.time()
                        while not self.goal_reached() and self.running and time.time() - wait_start < 0.25:
                            time.sleep(0.02)
            except Exception as e:
                print(f"[{self.namespace}] Error in update_position: {e}")
                time.sleep(0.5)  # Wait a bit before trying again
            
            time.sleep(0.02)
    
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
        
        # Start the mission immediately after takeoff
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