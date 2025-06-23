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
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from as2_msgs.msg import YawMode, BehaviorStatus
from as2_python_api.drone_interface import DroneInterface
from as2_python_api.behavior_actions.behavior_handler import BehaviorHandler
from std_msgs.msg import ColorRGBA, Float32MultiArray, String
from geometry_msgs.msg import PoseStamped, Pose, Vector3

# --- Formation functions for 5 drones ---
def formation_line_5():
    """
    Return offsets for a line formation for 5 drones.
    Evenly spaced along the x-axis with drone2 (index 2) as the leader in the center.
    """
    d = 0.6  # increased distance between drones for more stability
    return [(-2*d, 0.0), (-d, 0.0), (0.0, 0.0), (d, 0.0), (2*d, 0.0)]

def formation_v_5():
    """
    Return offsets for a V formation for 5 drones.
    Middle drone (drone2) at vertex, others form the V shape.
    """
    d = 0.4  # lateral offset
    return [(-2*d, 2*d), (-d, d), (0.0, 0.0), (d, d), (2*d, 2*d)]

def formation_diamond_5():
    """
    Return offsets for a diamond formation for 5 drones.
    Leader (drone2) in the center of the diamond, with followers at the diamond points.
    """
    d = 0.5  # Spacing between drones
    
    # Diamond layout with leader in the center
    return [
        (0.0, d),       # drone0 at front point
        (-d, 0.0),      # drone1 at left point
        (0.0, 0.0),     # drone2 (leader) at center
        (d, 0.0),       # drone3 at right point
        (0.0, -d)       # drone4 at back point
    ]

def formation_circular_orbit_5():
    """
    Return offsets for a circular orbit formation for 5 drones.
    Creates a perfect circle with followers orbiting around the leader in 
    an anticlockwise direction, maintaining a clear circular shape at all times.
    """
    # Circle radius - increased for better visibility
    circle_radius = 1.0  # Increased further for more distinct circular pattern
    
    # Number of followers (4) arranged in a perfect circle
    num_followers = 4
    
    # Calculate positions around the circle in an anticlockwise arrangement
    # These are just the base positions - actual rotation is applied in calculate_position
    offsets = []
    
    # First two followers (drone0, drone1) - add sequentially around the circle
    for i in range(2):
        # Angles in anticlockwise direction (note: using base positions here)
        angle = i * (2.0 * math.pi / num_followers)
        x = circle_radius * math.cos(angle)
        y = circle_radius * math.sin(angle)
        offsets.append((x, y))
    
    # Leader (drone2) at center
    offsets.append((0.0, 0.0))
    
    # Last two followers (drone3, drone4) - continue around the circle
    for i in range(2, 4):
        angle = i * (2.0 * math.pi / num_followers)
        x = circle_radius * math.cos(angle)
        y = circle_radius * math.sin(angle)
        offsets.append((x, y))
    
    return offsets

def formation_grid_5():
    """
    Return offsets for a grid formation for 5 drones.
    Leader (drone2) at front, with followers arranged in a 2x2 grid behind:
        leader
        f1 f2
        f3 f4
    """
    d = 0.6  # Spacing between drones
    
    return [
        (-d, -d),    # drone0 (f1) middle-left
        (d, -d),     # drone1 (f2) middle-right
        (0.0, 0.0),  # drone2 (leader) at front
        (-d, -2*d),  # drone3 (f3) back-left
        (d, -2*d)    # drone4 (f4) back-right
    ]

def formation_staggered_5():
    """
    Return offsets for a staggered formation for 5 drones.
    Creates a pattern with leader (drone2) in the middle.
    """
    d = 0.4
    return [(-2*d, 0.0), (-d, d/2), (0.0, 0.0), (d, d/2), (2*d, 0.0)]

def rotate_offset(offset, angle):
    """Rotate a 2D offset vector by a given angle (in radians)."""
    x, y = offset
    x_new = x * math.cos(angle) - y * math.sin(angle)
    y_new = x * math.sin(angle) + y * math.cos(angle)
    return (x_new, y_new)

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

def load_drone_home_positions():
    """Load drone home positions from the world configuration file."""
    # Try multiple possible paths to find the world_swarm.yaml file
    possible_paths = [
        "config_sim/world/world_swarm.yaml",  # Relative to working directory
        "/home/ziyar/project_gazebo_ws/src/challenge_multi_drone/config_sim/world/world_swarm.yaml",  # Absolute path
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_sim/world/world_swarm.yaml"),  # Relative to script
        os.path.expanduser("~/project_gazebo_ws/src/challenge_multi_drone/config_sim/world/world_swarm.yaml"),  # Expanded user path
    ]
    
    # Try each path
    for file_path in possible_paths:
        if os.path.exists(file_path):
            print(f"Found world config file at: {file_path}")
            try:
                with open(file_path, 'r') as file:
                    config = yaml.safe_load(file)
                    
                    # Extract drone positions
                    drone_positions = {}
                    if 'drones' in config:
                        for drone in config['drones']:
                            name = drone['model_name']
                            xyz = drone['xyz']
                            drone_positions[name] = (xyz[0], xyz[1], xyz[2])
                    
                    # Add hardcoded values for all drones to ensure they're all present
                    # These will only be used if they're not already in the loaded config
                    default_positions = {
                        "drone0": (-1.0, 0.0, 0.2),
                        "drone1": (1.0, 0.0, 0.2),
                        "drone2": (0.0, 0.0, 0.2),
                        "drone3": (0.0, 1.0, 0.2),
                        "drone4": (0.0, -1.0, 0.2)
                    }
                    
                    # Add any missing drones from defaults
                    for drone_name, pos in default_positions.items():
                        if drone_name not in drone_positions:
                            print(f"Adding missing drone {drone_name} with default position {pos}")
                            drone_positions[drone_name] = pos
                    
                    return drone_positions
            except yaml.YAMLError as e:
                print(f"Error parsing YAML file {file_path}: {e}")
                continue
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                continue
    
    # If no file was successfully loaded, use hardcoded defaults
    print("WARNING: Could not load world config file from any path, using hardcoded defaults")
    return {
        "drone0": (-1.0, 0.0, 0.2),
        "drone1": (1.0, 0.0, 0.2),
        "drone2": (0.0, 0.0, 0.2),
        "drone3": (0.0, 1.0, 0.2),
        "drone4": (0.0, -1.0, 0.2)
    }

def compute_euclidean_distance(p1, p2):
    """Compute Euclidean distance between two 3D points."""
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

# --- Vector2 class for position and velocity representations ---
class Vector2:
    """Simple 2D vector class for position and velocity representations"""
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector2(self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar):
        return Vector2(self.x / scalar, self.y / scalar)
    
    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def length_squared(self):
        return self.x * self.x + self.y * self.y
    
    def normalize(self):
        length = self.length()
        if length > 0.0001:
            return Vector2(self.x / length, self.y / length)
        return Vector2(0, 0)
    
    def to_tuple(self):
        return (self.x, self.y)

# --- SwarmDroneNode class for decentralized drone control ---
class SwarmDroneNode(Node):
    """Decentralized drone node that communicates with other drones via ROS topics"""
    
    def __init__(self, drone_id, config_path, altitude=2.0, use_sim_time=True, verbose=False):
        super().__init__(f'{drone_id}_swarm_node')
        
        # Basic properties
        self.drone_id = drone_id
        self.verbose = verbose
        self.altitude = altitude
        self.drone_index = int(drone_id.replace('drone', ''))  # Extract drone index (0-4)
        
        # Current target position for position comparison - explicitly initialize to None
        self.current_target_position = None
        
        # Track exact commanded target position
        self.commanded_target = None
        
        # Track position update sources and timing
        self.last_position_source = "none"
        self.last_position_update_time = time.time()
        self.position_source_counts = {}
        
        # Set logger level
        if verbose:
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
        else:
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)
        
        # Debug counters for communication
        self.position_updates_received = 0
        self.state_messages_received = {d_id: 0 for d_id in [f'drone{i}' for i in range(5)]}
        self.state_messages_sent = 0
        self.last_debug_time = time.time()
        
        # Load configuration
        self.config = load_scenario_config(config_path)
        if not self.config:
            self.get_logger().error(f"Failed to load configuration from {config_path}")
            raise ValueError(f"Failed to load configuration from {config_path}")
            
        # Load home positions
        self.drone_home_positions = load_drone_home_positions()
        self.home_position = self.drone_home_positions.get(drone_id, (0.0, 0.0, 0.2))
        
        # Stage parameters from config
        self.stage_center = self.config['stage1']['stage_center']
        self.diameter = self.config['stage1']['trajectory']['diameter']
        self.radius = self.diameter / 2.0
        
        # Formation parameters
        self.total_formations = 6  # Number of different formations to cycle through
        self.formation_repeats = 1  # How many times to repeat each formation
        self.degrees_per_formation = 180  # How many degrees to travel per formation
        self.angle_deg = 30  # Angle step for trajectory points (in degrees)
        self.angle_step = math.radians(self.angle_deg)
        self.steps_per_formation = int(self.degrees_per_formation / self.angle_deg)
        
        # Special steps for circular orbit
        self.circular_orbit_steps = 18
        self.circular_orbit_angle_deg = self.degrees_per_formation / self.circular_orbit_steps
        self.circular_orbit_angle_step = math.radians(self.circular_orbit_angle_deg)
        
        # Mission state
        self.current_formation = 0
        self.current_angle = 0.0
        self.current_step = 0
        self.total_steps = self.steps_per_formation * self.total_formations
        self.mission_complete = False
        self.all_drones_ready = False
        self.running = False
        
        # Position tracking
        self.last_position_check_time = time.time()
        self.at_target_position = False
        self.position_reach_timeout = 10.0  # seconds before forcing advancement if position not reached
        self.last_advancement_time = time.time()
        self.advancement_timeout = 20.0  # seconds before forcing advancement if stuck at same position
        
        # Formation functions
        self.formation_functions = [
            formation_line_5,
            formation_v_5,
            formation_diamond_5,
            formation_circular_orbit_5,
            formation_grid_5,
            formation_staggered_5
        ]
        self.formation_names = [
            "Line",
            "V-Shape",
            "Diamond",
            "Circular Orbit",
            "Grid",
            "Staggered"
        ]
        
        # Current positions and states of all drones
        self.drone_positions = {}
        self.drone_states = {}  # Will store formation and step for each drone
        
        # Initialize drone interface
        self.drone = DroneInterface(drone_id=drone_id, use_sim_time=use_sim_time, verbose=verbose)
        
        # LED control
        self.led_pub = self.drone.create_publisher(ColorRGBA, f"/{drone_id}/leds/control", 10)
        
        # Create simple QoS profile for all topics
        position_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # ------------------------------------------------------------------------------
        # Subscribe to pose topics for position updates
        # ------------------------------------------------------------------------------
        
        # Main localization source
        topic = f'/{drone_id}/self_localization/pose'
        self.create_subscription(
            PoseStamped,
            topic,
            lambda msg: self.position_callback(msg, "self_localization"),
            position_qos
        )
        self.get_logger().info(f"Subscribed to position topic: {topic} with RELIABLE reliability")
        
        # Backup position source
        backup_topic = f'/{drone_id}/ground_truth/pose'
        self.create_subscription(
            PoseStamped,
            backup_topic,
            lambda msg: self.position_callback(msg, "ground_truth"),
            position_qos
        )
        self.get_logger().info(f"Subscribed to backup position topic: {backup_topic} with RELIABLE reliability")
        
        # Add timer to check if position updates are stalled
        self.position_check_timer = self.create_timer(1.0, self.check_position_updates)
        
        # Create publisher for drone state
        topic_name = f'/swarm/state/{drone_id}'
        self.get_logger().info(f"Creating state publisher on topic: {topic_name}")
        self.state_pub = self.create_publisher(
            String,
            topic_name,
            position_qos
        )
        
        # Create subscription for other drones' states
        self.drone_ids = [f'drone{i}' for i in range(5)]  # All drones in the swarm
        self.state_subscriptions = []  # Keep track of subscriptions
        
        for other_id in self.drone_ids:
            if other_id != drone_id:
                topic_name = f'/swarm/state/{other_id}'  # Changed topic pattern
                self.get_logger().info(f"Subscribing to {topic_name}")
                
                # Fix the callback creation - don't use a factory function that might capture variables incorrectly
                sub = self.create_subscription(
                    String,
                    topic_name,
                    lambda msg, d_id=other_id: self.drone_state_callback(msg, d_id),
                    position_qos
                )
                self.state_subscriptions.append(sub)  # Save subscription to prevent garbage collection
        
        # Consensus state publisher
        topic_name = f'/swarm/consensus/{drone_id}'  # Changed topic pattern
        self.get_logger().info(f"Creating consensus publisher on topic: {topic_name}")
        self.consensus_pub = self.create_publisher(
            String,
            topic_name,
            position_qos
        )
        
        # Consensus state subscribers for other drones
        self.consensus_subscriptions = []  # Keep track of subscriptions
        for other_id in self.drone_ids:
            if other_id != drone_id:
                topic_name = f'/swarm/consensus/{other_id}'  # Changed topic pattern
                self.get_logger().info(f"Subscribing to {topic_name}")
                
                # Fix callback creation
                sub = self.create_subscription(
                    String,
                    topic_name,
                    lambda msg, d_id=other_id: self.consensus_callback(msg, d_id),
                    position_qos
                )
                self.consensus_subscriptions.append(sub)  # Save subscription
        
        # Timer for state publishing (5 Hz)
        self.state_timer = self.create_timer(0.2, self.publish_state)
        
        # Timer for mission logic (10 Hz)
        self.mission_timer = self.create_timer(0.1, self.mission_step)
        
        # Consensus timeouts and votes
        self.formation_votes = {d_id: -1 for d_id in self.drone_ids}
        self.last_consensus_time = time.time()
        self.consensus_timeout = 2.0  # seconds
        
        # Add a recovery timer that triggers less frequently but takes more drastic action
        self.recovery_timer = self.create_timer(15.0, self.perform_recovery_actions)
        self.last_recovery_time = time.time()
        self.recovery_attempts = 0
        
        self.get_logger().info(f"Swarm Drone {drone_id} initialized")
    
    def has_reached_target(self,
                       current: np.ndarray,
                       target: np.ndarray,
                       xy_tol: float = 0.25,
                       z_tol: float = 0.40) -> bool:
        """True if |xy| < xy_tol AND |z| < z_tol (all values in metres)."""
        xy_err = np.linalg.norm(current[:2] - target[:2])
        z_err = abs(current[2] - target[2])
        return xy_err < xy_tol and z_err < z_tol
    
    def check_position_updates(self):
        """Check if position updates have stalled and try to diagnose the issue"""
        current_time = time.time()
        time_since_last_update = current_time - self.last_position_update_time
        
        if time_since_last_update > 1.0:
            self.get_logger().warning(f"No pose update for {time_since_last_update:.2f}s – check QoS or topic names!")
            
            # After 5 seconds without updates, provide more detailed diagnostic info
            if time_since_last_update > 5.0:
                self.get_logger().warning(f"Last update source: {self.last_position_source}")
                self.get_logger().warning(f"Position source counts: {self.position_source_counts}")
                
                # Log all known position topics to help debug localization issues
                try:
                    topics = sorted([t[0] for t in self.get_topic_names_and_types()])
                    position_topics = [t for t in topics if '/pose' in t or '/position' in t or 
                                      '/localization' in t or '/odom' in t or '/platypus' in t]
                    self.get_logger().warning(f"Available position topics: {position_topics}")
                    
                    # For critical diagnostics, provide info for manual checks
                    if time_since_last_update > 15.0 and self.recovery_attempts == 0:
                        self.get_logger().error("Critical position update stall - suggested diagnostics:")
                        try:
                            # Suggest diagnostic commands but don't attempt resubscription
                            self.get_logger().error(f"Check topic info: 'ros2 topic info /{self.drone_id}/self_localization/pose'")
                            self.get_logger().error(f"Check data flow: 'ros2 topic echo /{self.drone_id}/self_localization/pose'")
                            self.get_logger().error(f"Check topic rate: 'ros2 topic hz /{self.drone_id}/self_localization/pose'")
                            
                            # List known publishers for key topics
                            for topic in position_topics:
                                if self.drone_id in topic:
                                    self.get_logger().error(f"Check {topic} info: 'ros2 topic info {topic}'")
                        except:
                            pass
                except Exception as e:
                    self.get_logger().error(f"Error checking position topics: {e}")
        
        # If position updates are stalled for a long time, try recovery actions
        # but NO resubscription since it can cause QoS mismatches
        if time_since_last_update > 20.0:
            self.get_logger().error(f"Position updates stalled for {time_since_last_update:.1f}s - attempting recovery")
            
            # Try reinitializing drone interface
            try:
                # Save current state
                running_state = self.running
                
                # Temporarily set running to False to prevent mission operations during reinitialization
                self.running = False
                
                # Reinitialize drone interface
                self.drone.shutdown()
                time.sleep(1.0)  # Short delay before reinitializing
                
                # Recreate the drone interface
                self.drone = DroneInterface(
                    drone_id=self.drone_id, 
                    use_sim_time=self.drone.use_sim_time, 
                    verbose=True  # Use verbose during recovery
                )
                
                # Restore running state
                self.running = running_state
                
                self.get_logger().info("Drone interface reinitialized")
            except Exception as e:
                self.get_logger().error(f"Error reinitializing drone interface: {e}")
                import traceback
                self.get_logger().error(traceback.format_exc())
    
    def position_callback(self, msg, source="unknown"):
        """Callback for own position updates"""
        # Get the topic from the message frame_id if available, otherwise use provided source
        topic_source = source
        if hasattr(msg.header, 'frame_id') and msg.header.frame_id:
            topic_source = f"{source}:{msg.header.frame_id}"
        
        # Get the topic name from the subscription
        topic_name = "unknown"
        if hasattr(msg, "_topic") and msg._topic:
            topic_name = msg._topic
        
        # Track which sources are providing updates
        if topic_source not in self.position_source_counts:
            self.position_source_counts[topic_source] = 0
        self.position_source_counts[topic_source] += 1
        
        # Update last position source and time
        self.last_position_source = topic_source
        self.last_position_update_time = time.time()
        
        position = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ]
        
        # Important debug - log actual position updates
        prev_position = self.drone_positions.get(self.drone_id, [0.0, 0.0, 0.0])
        
        # Check if position has actually changed
        epsilon = 0.001  # Small threshold to detect real changes (1mm)
        xy_changed = abs(prev_position[0] - position[0]) > epsilon or abs(prev_position[1] - position[1]) > epsilon
        z_changed = abs(prev_position[2] - position[2]) > epsilon
        position_changed = xy_changed or z_changed
        
        # Enhanced logging for position changes
        if position_changed:
            # Log all position changes
            change_type = "XY+Z" if xy_changed and z_changed else "XY only" if xy_changed else "Z only"
            
            if xy_changed:  # Specifically log XY changes as they're what we're troubleshooting
                xy_delta = np.linalg.norm(np.array(position[:2]) - np.array(prev_position[:2]))
                self.get_logger().info(
                    f"XY POSITION CHANGED from {source}: Δ={xy_delta:.3f}m, "
                    f"[{prev_position[0]:.2f}, {prev_position[1]:.2f}] → [{position[0]:.2f}, {position[1]:.2f}]"
                )
            
            # Log all changes periodically
            if self.position_updates_received % 10 == 0 or xy_changed:
                self.get_logger().info(
                    f"Position update ({change_type}) from {topic_source} via {topic_name}: "
                    f"[{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]"
                )
        
        # Track altitude changes specifically
        altitude_changed = abs(prev_position[2] - position[2]) > 0.05  # 5cm threshold for altitude change
        if altitude_changed:
            self.get_logger().info(f"Altitude changed: {prev_position[2]:.2f}m -> {position[2]:.2f}m (delta: {position[2] - prev_position[2]:.2f}m)")
            self.get_logger().info(f"Altitude update from topic: {topic_name}")
            
            # Check if altitude is significantly different from the target
            if hasattr(self, 'altitude') and abs(position[2] - self.altitude) > 0.3:
                self.get_logger().warning(f"Altitude {position[2]:.2f}m differs from target {self.altitude:.2f}m by {position[2] - self.altitude:.2f}m")
            
            # If this is the first real position update after takeoff, log it prominently
            if position_changed and position[2] > 0.5 and prev_position[2] < 0.5:
                self.get_logger().info(f"TAKEOFF DETECTED - First real position: {position} from {topic_name}")
        
        # Update the drone position in our tracker
        self.drone_positions[self.drone_id] = position
        self.position_updates_received += 1
        
        # Debug position updates every 5 seconds
        current_time = time.time()
        if current_time - self.last_debug_time > 5.0:
            self.get_logger().info(f"Position updates received: {self.position_updates_received}")
            self.get_logger().info(f"Current position: {position}")
            self.get_logger().info(f"Current altitude: {position[2]:.2f}m (target: {self.altitude:.2f}m)")
            self.get_logger().info(f"Known drone positions: {list(self.drone_positions.keys())}")
            self.get_logger().info(f"State messages received: {self.state_messages_received}")
            self.get_logger().info(f"State messages sent: {self.state_messages_sent}")
            self.last_debug_time = current_time
        
        # If we have an active target position and we're running, check if we're there
        if self.running and self.commanded_target is not None:
            try:
                pos_array = np.array(position)
                
                # If close enough to target, mark as reached
                if self.has_reached_target(pos_array, self.commanded_target):
                    if not self.at_target_position:
                        self.at_target_position = True
                        xy_err = np.linalg.norm(pos_array[:2] - self.commanded_target[:2])
                        z_err = abs(position[2] - self.commanded_target[2])
                        
                        # Enhanced debug output with detailed position and calculation information
                        dx = position[0] - self.commanded_target[0]
                        dy = position[1] - self.commanded_target[1]
                        dz = position[2] - self.commanded_target[2]
                        
                        self.get_logger().info("TARGET REACHED - Position details:")
                        self.get_logger().info(
                            f"Current: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}], "
                            f"Target: [{self.commanded_target[0]:.2f}, {self.commanded_target[1]:.2f}, {self.commanded_target[2]:.2f}]"
                        )
                        self.get_logger().info(
                            f"Deltas: Δx={dx:.2f}m, Δy={dy:.2f}m, Δz={dz:.2f}m"
                        )
                        self.get_logger().info(
                            f"xy-error: √(Δx² + Δy²) = √({dx:.2f}² + {dy:.2f}²) = {xy_err:.2f}m, z-error: |Δz| = {z_err:.2f}m"
                        )
                        self.get_logger().info(f"Tolerances: xy={0.25}m, z={0.40}m")
            except Exception as e:
                self.get_logger().error(f"Error calculating distance to target: {e}")
                self.get_logger().error(f"Position: {position}, Target: {self.commanded_target}")
    
    def drone_state_callback(self, msg, drone_id):
        """Callback for other drones' state updates"""
        try:
            # More robust string parsing that handles various formats
            data = msg.data.strip()
            
            # Log raw message for debugging
            self.get_logger().debug(f"Raw message from {drone_id}: {data}")
            
            # Handle different string formats that might come from the evaluation
            if data.startswith('{') and data.endswith('}'):
                # Use literal_eval for safer parsing than eval
                import ast
                state_data = ast.literal_eval(data)
                
                # Double check we have required fields
                if 'drone_id' not in state_data:
                    state_data['drone_id'] = drone_id
                
                self.drone_states[drone_id] = state_data
                self.state_messages_received[drone_id] += 1
                
                # Log the first few messages and every 50th message thereafter
                if self.state_messages_received[drone_id] <= 5 or self.state_messages_received[drone_id] % 50 == 0:
                    self.get_logger().info(f"Received state from {drone_id}: #{self.state_messages_received[drone_id]} - {state_data}")
                
                # Update position if available in state
                if 'position' in state_data:
                    self.drone_positions[drone_id] = state_data['position']
                    self.get_logger().debug(f"Updated position for {drone_id}: {state_data['position']}")
            else:
                self.get_logger().warning(f"Received malformed message from {drone_id}: {data}")
                
        except Exception as e:
            self.get_logger().error(f"Error parsing state from {drone_id}: {e} - message was: {msg.data}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def consensus_callback(self, msg, drone_id):
        """Callback for consensus messages from other drones"""
        try:
            # More robust string parsing
            data = msg.data.strip()
            
            # Handle proper parsing
            if data.startswith('{') and data.endswith('}'):
                # Use literal_eval for safer parsing
                import ast
                consensus_data = ast.literal_eval(data)
                
                # Double check fields
                if 'drone_id' not in consensus_data:
                    consensus_data['drone_id'] = drone_id
                
                # Update this drone's vote in our tracking
                if 'formation' in consensus_data:
                    self.formation_votes[drone_id] = consensus_data['formation']
                    self.get_logger().debug(f"Received formation vote from {drone_id}: {consensus_data['formation']}")
                
                # Reset timeout timer since we got a fresh vote
                self.last_consensus_time = time.time()
            else:
                self.get_logger().warning(f"Received malformed consensus message from {drone_id}: {data}")
                
        except Exception as e:
            self.get_logger().error(f"Error parsing consensus from {drone_id}: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def publish_state(self):
        """Publish this drone's state to the swarm"""
        # Get current position (always try to get position, even if not running yet)
        position = self.drone_positions.get(self.drone_id, [0.0, 0.0, 0.0])
        
        # Build state message
        state = {
            'drone_id': self.drone_id,
            'formation': self.current_formation,
            'step': self.current_step,
            'position': position,
            'ready': self.running,  # Use running as ready indicator for early messages
            'timestamp': time.time(),
            'at_target': self.at_target_position,
            'startup': not self.running  # Flag for startup messages
        }
        
        # Debug publish count
        should_log = (self.state_messages_sent < 10) or (self.state_messages_sent % 50 == 0)
        
        # Publish as string (simple serialization)
        state_msg = String()
        state_msg.data = str(state)
        self.state_pub.publish(state_msg)
        self.state_messages_sent += 1
        
        # Log topic name for first few messages
        if should_log:
            topic_name = f'/swarm/state/{self.drone_id}'  # Changed topic pattern
            message_type = "startup" if not self.running else "normal"
            self.get_logger().info(f"Published {message_type} state #{self.state_messages_sent} to {topic_name}")
            
    def publish_consensus(self):
        """Publish consensus vote for next formation"""
        consensus_msg = String()
        consensus_data = {
            'drone_id': self.drone_id,
            'formation': self.current_formation,
            'timestamp': time.time()
        }
        consensus_msg.data = str(consensus_data)
        self.consensus_pub.publish(consensus_msg)
        
        topic_name = f'/swarm/consensus/{self.drone_id}'  # Changed topic pattern
        self.get_logger().debug(f"Published consensus to {topic_name}: {consensus_data['formation']}")
    
    def run_consensus(self):
        """Run consensus to see if all drones agree on formation"""
        # Update our own vote
        self.formation_votes[self.drone_id] = self.current_formation
        
        # Publish our vote
        self.publish_consensus()
        
        # Check if all votes are equal and valid
        values = list(self.formation_votes.values())
        
        # Log vote status occasionally
        if time.time() - self.last_consensus_time > 1.0:
            self.get_logger().info(f"Running consensus. Current votes: {self.formation_votes}")
            self.last_consensus_time = time.time()
        
        # We need at least our own vote and 1 other (allowing for network delays)
        min_votes = 2  
        valid_votes = sum(1 for v in values if v >= 0)
        
        if valid_votes < min_votes:
            # Check for timeout - if we haven't had enough votes for a long time, proceed anyway
            if time.time() - self.last_consensus_time > 10.0:  # Reduced from original timeout
                self.get_logger().warning(f"Consensus timeout - proceeding with only {valid_votes} votes")
                return True
            return False
            
        # Check for agreement (only among drones that have voted)
        votes_cast = [v for v in values if v >= 0]
        if not votes_cast:
            return False
            
        # If most votes agree with our formation, proceed
        our_formation = self.formation_votes[self.drone_id]
        matching_votes = sum(1 for v in votes_cast if v == our_formation)
        
        # If majority of votes match our formation
        consensus_reached = matching_votes >= (len(votes_cast) / 2)
        
        # If consensus timeout has passed, proceed anyway (avoid deadlock)
        consensus_time_passed = time.time() - self.last_consensus_time > self.consensus_timeout
        
        if consensus_reached:
            self.get_logger().info(f"Consensus reached: {matching_votes}/{len(votes_cast)} votes agree")
        elif consensus_time_passed:
            self.get_logger().info("Consensus timeout reached, proceeding anyway")
            
        return consensus_reached or consensus_time_passed
    
    def check_formation_readiness(self):
        """Check if all drones have reached current formation step"""
        # For initial readiness check only, not during formation flight
        if len(self.drone_states) < len(self.drone_ids) - 1:
            # Wait for at least N-1 drones to report (allowing for some lag)
            self.get_logger().info(f"Waiting for other drones: {len(self.drone_states)}/{len(self.drone_ids)-1} reporting")
            self.get_logger().info(f"Known drones states: {list(self.drone_states.keys())}")
            
            # Add check for timeout
            current_time = time.time()
            if not hasattr(self, 'readiness_start_time'):
                self.readiness_start_time = current_time
                self.readiness_check_count = 0
            else:
                self.readiness_check_count += 1
            
            # Check if we've waited too long or have at least 2 drones connected and enough checks
            time_waiting = current_time - self.readiness_start_time
            min_connected = 1  # At least one other drone must be connected
            
            if len(self.drone_states) >= min_connected and self.readiness_check_count > 50:
                self.get_logger().warning(f"Sufficient drone count ({len(self.drone_states)}) and checks ({self.readiness_check_count}). Proceeding.")
                return True
            elif time_waiting > 15.0:  # 15 seconds timeout
                self.get_logger().warning(f"Waited {time_waiting:.1f}s for other drones. Forcing readiness with {len(self.drone_states)} drones.")
                return True
            
            # Debug topic information
            if time.time() - self.last_debug_time > 5.0:
                self.last_debug_time = time.time()
                try:
                    topics = sorted(self.get_topic_names_and_types())
                    state_topics = [t for t, _ in topics if '/state' in t]
                    self.get_logger().info(f"Available state topics: {state_topics}")
                    self.get_logger().info(f"Position received for drones: {list(self.drone_positions.keys())}")
                    self.get_logger().info(f"State messages sent: {self.state_messages_sent}")
                    self.get_logger().info(f"State messages received: {self.state_messages_received}")
                    
                    for sub in self.state_subscriptions:
                        # Try to get subscription info
                        try:
                            topic_name = sub.topic_name
                            self.get_logger().info(f"Subscribed to: {topic_name}")
                        except:
                            self.get_logger().info(f"Subscription info not available")
                except Exception as e:
                    self.get_logger().error(f"Error getting topic info: {e}")
            
            return False
              
        self.get_logger().info(f"All drones reporting. Ready to start formations.")
        return True
    
    def calculate_target_position(self):
        """Calculate target position based on current formation and angle"""
        try:
            # Calculate the center position on the circle
            current_angle_rad = math.radians(self.current_angle)
            center_x = self.stage_center[0] + self.radius * math.cos(current_angle_rad)
            center_y = self.stage_center[1] + self.radius * math.sin(current_angle_rad)
            
            self.get_logger().debug(f"Circle center at ({center_x:.2f}, {center_y:.2f}) for angle {self.current_angle:.1f}°")
            
            # Get formation offsets
            formation_function = self.formation_functions[self.current_formation]
            formation_offsets = formation_function()
            
            # Get offset for this drone
            if self.drone_index < len(formation_offsets):
                offset = formation_offsets[self.drone_index]
                self.get_logger().debug(f"Using offset {offset} for drone {self.drone_index} in formation {self.current_formation}")
            else:
                self.get_logger().warning(f"No offset defined for drone {self.drone_id} in formation {self.current_formation}")
                offset = (0.0, 0.0)  # Default offset
            
            # For circular orbit, add special rotation based on step
            if self.current_formation == 3:  # Circular orbit formation
                # Calculate rotation based on position around the circle
                rotation_angle = current_angle_rad + math.pi  # Rotate drones to face inward
                self.get_logger().debug(f"Circular orbit formation - rotation angle: {math.degrees(rotation_angle):.1f}°")
            else:
                # For other formations, adjust orientation to face forward along the circle
                rotation_angle = current_angle_rad + math.pi/2  # Tangent to the circle
                self.get_logger().debug(f"Standard formation - rotation angle: {math.degrees(rotation_angle):.1f}°")
            
            # Apply rotation to the offset
            rotated_offset = rotate_offset(offset, rotation_angle)
            self.get_logger().debug(f"Rotated offset: {rotated_offset}")
            
            # Apply offset to center position
            drone_x = center_x + rotated_offset[0]
            drone_y = center_y + rotated_offset[1]
            
            # Store the target position for direct comparison in position callback
            target_position = [drone_x, drone_y, self.altitude]
            self.current_target_position = target_position
            
            # Periodically log detailed position calculations for tracking
            # Only log every 5th calculation to avoid excessive output
            if hasattr(self, 'calc_count'):
                self.calc_count += 1
            else:
                self.calc_count = 0
                
            if self.calc_count % 5 == 0:
                # Get current position if available
                current_pos = self.drone_positions.get(self.drone_id, [0.0, 0.0, 0.0])
                
                self.get_logger().info(
                    f"TARGET CALCULATION - Formation: {self.formation_names[self.current_formation]}, Step: {self.current_step}"
                )
                self.get_logger().info(
                    f"Circle center: [{center_x:.2f}, {center_y:.2f}], Angle: {self.current_angle:.1f}°, Rotation: {math.degrees(rotation_angle):.1f}°"
                )
                self.get_logger().info(
                    f"Offset: {offset}, Rotated offset: {rotated_offset}"
                )
                self.get_logger().info(
                    f"Current position: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}], "
                    f"Target position: [{drone_x:.2f}, {drone_y:.2f}, {self.altitude:.2f}]"
                )
                
                # If we have a commanded target already, show error compared to new calculation
                if hasattr(self, 'commanded_target') and self.commanded_target is not None:
                    calc_error = np.linalg.norm(np.array(target_position) - self.commanded_target)
                    if calc_error > 0.01:  # Only show if significant difference
                        self.get_logger().info(
                            f"Previous target: [{self.commanded_target[0]:.2f}, {self.commanded_target[1]:.2f}, {self.commanded_target[2]:.2f}], "
                            f"Calculation drift: {calc_error:.3f}m"
                        )
            
            # Log target position
            self.get_logger().debug(f"Target position: ({drone_x:.2f}, {drone_y:.2f}, {self.altitude:.2f})")
            
            return target_position, rotation_angle
        except Exception as e:
            self.get_logger().error(f"Error calculating target position: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            # Return a safe default position
            safe_position = [0.0, 0.0, self.altitude]
            self.current_target_position = safe_position
            return safe_position, 0.0
    
    def get_formation_color(self):
        """Get LED color for current formation"""
        if self.current_formation == 0:  # Line
            return (1.0, 0.0, 0.0)  # Red
        elif self.current_formation == 1:  # V
            return (0.0, 1.0, 0.0)  # Green
        elif self.current_formation == 2:  # Diamond
            return (0.0, 0.0, 1.0)  # Blue
        elif self.current_formation == 3:  # Circular Orbit
            return (1.0, 1.0, 0.0)  # Yellow
        elif self.current_formation == 4:  # Grid
            return (1.0, 0.0, 1.0)  # Magenta
        elif self.current_formation == 5:  # Staggered
            return (0.0, 1.0, 1.0)  # Cyan
        else:
            return (1.0, 1.0, 1.0)  # White
    
    def change_led_color(self, color):
        """Change LED color"""
        color_msg = ColorRGBA()
        color_msg.r = float(color[0])
        color_msg.g = float(color[1])
        color_msg.b = float(color[2])
        color_msg.a = 1.0
        self.led_pub.publish(color_msg)
    
    def advance_formation(self):
        """Advance to next step or formation"""
        # Calculate angle step based on formation
        if self.current_formation == 3:  # Circular orbit
            angle_step = self.circular_orbit_angle_deg
            max_steps = self.circular_orbit_steps
        else:
            angle_step = self.angle_deg
            max_steps = self.steps_per_formation
        
        # Increment step
        self.current_step += 1
        
        # Update current angle
        self.current_angle += angle_step
        
        # Check if we completed all steps for current formation
        if self.current_step >= max_steps:
            # Move to next formation
            self.current_formation = (self.current_formation + 1) % self.total_formations
            self.current_step = 0
            
            # Reset formation votes
            self.formation_votes = {d_id: -1 for d_id in self.drone_ids}
            
            self.get_logger().info(f"Advancing to formation: {self.formation_names[self.current_formation]}")
        
        # Check if we completed all formations
        if self.current_formation >= self.total_formations:
            self.mission_complete = True
            self.get_logger().info("Mission complete")
    
    def get_target_yaw(self, rotation_angle):
        """Convert rotation angle to yaw angle for drone"""
        # Normalize to 0-2π range
        yaw = rotation_angle % (2 * math.pi)
        return yaw
    
    def mission_step(self):
        """Execute one step of the mission logic"""
        try:
            if not self.running or self.mission_complete:
                return
            
            # Wait for all drones to become ready
            if not self.all_drones_ready:
                self.all_drones_ready = self.check_formation_readiness()
                
                if self.all_drones_ready:
                    self.get_logger().info("All drones ready. Starting formation flight.")
                    # Reset the vote tracking
                    self.formation_votes = {d_id: -1 for d_id in self.drone_ids}
                    
                    # Initialize with first formation position
                    self.current_formation = 0
                    self.current_step = 0
                    self.current_angle = 0.0
                    self.last_advancement_time = time.time()
                    
                    # Start with first formation
                    target_position, rotation_angle = self.calculate_target_position()
                    self.get_logger().info(f"Starting first formation: {self.formation_names[self.current_formation]}")
                    self.get_logger().info(f"Moving to initial position: {target_position}")
                    
                    # Set LED color for first formation
                    color = self.get_formation_color()
                    self.change_led_color(color)
                    
                    # Move to initial position
                    yaw_angle = self.get_target_yaw(rotation_angle)
                    self.get_logger().info(f"Moving to initial position with yaw: {math.degrees(yaw_angle):.1f}°")
                    self.drone.go_to(
                        target_position[0],
                        target_position[1],
                        target_position[2],
                        speed=0.5,
                        yaw_mode=YawMode.PATH_FACING,
                        yaw_angle=yaw_angle,
                        wait=False
                    )
                    # save *exact* waypoint that was commanded
                    self.commanded_target = np.array(target_position, dtype=float)
                return
            
            # Calculate target position for this formation step
            target_position, rotation_angle = self.calculate_target_position()
            
            # Set LED color based on formation
            color = self.get_formation_color()
            self.change_led_color(color)
            
            # Move to target position
            yaw_angle = self.get_target_yaw(rotation_angle)
            
            # Use go_to to move to the position (non-blocking)
            self.drone.go_to(
                target_position[0],
                target_position[1],
                target_position[2],
                speed=0.5,
                yaw_mode=YawMode.PATH_FACING,
                yaw_angle=yaw_angle,
                wait=False  # Non-blocking to allow continued ROS execution
            )
            # save *exact* waypoint that was commanded
            self.commanded_target = np.array(target_position, dtype=float)
            
            # Get the current drone position from the cached value
            current_pos = self.drone_positions.get(self.drone_id, [0.0, 0.0, 0.0])
            current_pos_array = np.array(current_pos)
            
            current_time = time.time()
            
            # Check for position reach timeout (only log once per second)
            if current_time - self.last_position_check_time > 1.0:
                self.last_position_check_time = current_time
                # Only show distance if we haven't reached the target yet
                if not self.at_target_position and self.commanded_target is not None:
                    xy_err = np.linalg.norm(current_pos_array[:2] - self.commanded_target[:2])
                    z_err = abs(current_pos[2] - self.commanded_target[2])
                    xy_tol, z_tol = 0.25, 0.40  # Same as in has_reached_target
                    
                    # Enhanced debug output showing position details and calculation
                    dx = current_pos[0] - self.commanded_target[0]
                    dy = current_pos[1] - self.commanded_target[1]
                    dz = current_pos[2] - self.commanded_target[2]
                    
                    self.get_logger().info(
                        f"Position: current=[{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}], "
                        f"target=[{self.commanded_target[0]:.2f}, {self.commanded_target[1]:.2f}, {self.commanded_target[2]:.2f}]"
                    )
                    self.get_logger().info(
                        f"Deltas: Δx={dx:.2f}m, Δy={dy:.2f}m, Δz={dz:.2f}m"
                    )
                    self.get_logger().info(
                        f"xy-error: √(Δx² + Δy²) = √({dx:.2f}² + {dy:.2f}²) = {xy_err:.2f}m, z-error: |Δz| = {z_err:.2f}m (tolerances: {xy_tol}/{z_tol}m)"
                    )
                    
                    # Log all known position topics to help debug localization issues
                    if self.position_updates_received % 10 == 0:  # Every 10 seconds
                        try:
                            topics = sorted([t[0] for t in self.get_topic_names_and_types()])
                            position_topics = [t for t in topics if '/pose' in t or '/position' in t or '/localization' in t]
                            self.get_logger().info(f"Available position topics: {position_topics}")
                        except Exception as e:
                            self.get_logger().error(f"Error getting topics: {e}")
                
                # Check for timeout for being stuck at same position/formation
                time_since_advancement = current_time - self.last_advancement_time
                if time_since_advancement > self.advancement_timeout:
                    self.get_logger().warning(f"Stuck for {time_since_advancement:.1f}s at step {self.current_step}. Forcing advancement.")
                    if self.commanded_target is not None:
                        xy_err = np.linalg.norm(current_pos_array[:2] - self.commanded_target[:2])
                        z_err = abs(current_pos[2] - self.commanded_target[2])
                        
                        # Enhanced debug output for timeout scenario
                        dx = current_pos[0] - self.commanded_target[0]
                        dy = current_pos[1] - self.commanded_target[1]
                        dz = current_pos[2] - self.commanded_target[2]
                        
                        self.get_logger().warning(
                            f"Position: current=[{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}], "
                            f"target=[{self.commanded_target[0]:.2f}, {self.commanded_target[1]:.2f}, {self.commanded_target[2]:.2f}]"
                        )
                        self.get_logger().warning(
                            f"xy-error: √(Δx² + Δy²) = √({dx:.2f}² + {dy:.2f}²) = {xy_err:.2f}m, z-error: |Δz| = {z_err:.2f}m"
                        )
                    
                    self.advance_formation()
                    self.at_target_position = False
                    self.last_advancement_time = current_time
                    return
            
            # If we've already detected reaching the target position in the position callback,
            # or if we're very close to the target now, run consensus
            if self.at_target_position or (self.commanded_target is not None and 
                                         self.has_reached_target(current_pos_array, self.commanded_target)):
                # If not already marked as at target, do it now
                if not self.at_target_position and self.commanded_target is not None:
                    self.at_target_position = True
                    xy_err = np.linalg.norm(current_pos_array[:2] - self.commanded_target[:2])
                    z_err = abs(current_pos[2] - self.commanded_target[2])
                    
                    # Enhanced debug with detailed position information
                    dx = current_pos[0] - self.commanded_target[0]
                    dy = current_pos[1] - self.commanded_target[1]
                    dz = current_pos[2] - self.commanded_target[2]
                    
                    self.get_logger().info("TARGET REACHED IN MISSION STEP - Position details:")
                    self.get_logger().info(
                        f"Current: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}], "
                        f"Target: [{self.commanded_target[0]:.2f}, {self.commanded_target[1]:.2f}, {self.commanded_target[2]:.2f}]"
                    )
                    self.get_logger().info(
                        f"Deltas: Δx={dx:.2f}m, Δy={dy:.2f}m, Δz={dz:.2f}m"
                    )
                    self.get_logger().info(
                        f"xy-error: √(Δx² + Δy²) = √({dx:.2f}² + {dy:.2f}²) = {xy_err:.2f}m, z-error: |Δz| = {z_err:.2f}m"
                    )
                    self.get_logger().info(f"Current formation: {self.formation_names[self.current_formation]}, step {self.current_step}")
                
                # Run consensus to see if all drones agree to move to next formation
                if self.run_consensus():
                    prev_formation = self.current_formation
                    prev_step = self.current_step
                    self.advance_formation()
                    self.at_target_position = False
                    self.last_advancement_time = current_time
                    
                    # Log advancement
                    if prev_formation != self.current_formation:
                        self.get_logger().info(f"Consensus reached! Advancing to new formation: {self.formation_names[self.current_formation]}")
                    elif prev_step != self.current_step:
                        self.get_logger().info(f"Consensus reached! Advancing to step {self.current_step} in formation {self.formation_names[self.current_formation]}")
            else:
                # If no longer at target position
                if self.at_target_position:
                    self.at_target_position = False
                
                # Check for position reach timeout (only if not at target)
                if not self.at_target_position and current_time - self.last_advancement_time > self.position_reach_timeout:
                    # Get visual confirmation of positions for debugging
                    self.get_logger().warning(f"Failed to reach position within {self.position_reach_timeout}s.")
                    
                    if self.commanded_target is not None:
                        xy_err = np.linalg.norm(current_pos_array[:2] - self.commanded_target[:2])
                        z_err = abs(current_pos[2] - self.commanded_target[2])
                        self.get_logger().warning(f"Current position: {current_pos}, Target: {self.commanded_target}")
                        self.get_logger().warning(f"xy-error: {xy_err:.2f}m, z-error: {z_err:.2f}m")
                    
                    # Check if drone is actually close enough despite measurement issues
                        xy_tol, z_tol = 0.25, 0.40  # Same as in has_reached_target
                        if xy_err < xy_tol and z_err < z_tol:
                            self.get_logger().info("Drone appears to be close enough horizontally and altitude is acceptable. Proceeding.")
                            self.at_target_position = True  # Force recognition of position reached
                        else:
                            self.get_logger().warning("Forcing advancement due to timeout.")
                            self.advance_formation()
                            self.at_target_position = False
                    else:
                        self.get_logger().warning("No commanded target available. Forcing advancement.")
                        self.advance_formation()
                        self.at_target_position = False
                    
                    self.last_advancement_time = current_time
        except Exception as e:
            self.get_logger().error(f"Error in mission_step: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def takeoff_and_start(self):
        """Take off and start the mission"""
        self.get_logger().info(f"Taking off to {self.altitude}m...")
        
        # First, publish initial state to announce presence
        self.publish_state()
        
        # Record initial position before takeoff for comparison
        initial_position = self.drone_positions.get(self.drone_id, [0.0, 0.0, 0.0])
        self.get_logger().info(f"Initial position before takeoff: {initial_position}")
        
        # Offboard mode and arm
        self.drone.offboard()
        self.drone.arm()
        
        # Set LED to blue for takeoff
        self.change_led_color((0.0, 0.0, 1.0))
        
        # Take off - note the altitude target here
        self.get_logger().info(f"Initiating takeoff command to {self.altitude}m with speed 0.7")
        self.drone.takeoff(self.altitude, 0.7)
        
        # Monitor the takeoff progress
        takeoff_start_time = time.time()
        takeoff_timeout = 10.0  # seconds
        position_before_wait = self.drone_positions.get(self.drone_id, initial_position)
        
        self.get_logger().info(f"Position right after takeoff command: {position_before_wait}")
        self.get_logger().info(f"Waiting for stability (3 seconds)...")
        
        # Wait briefly for stability
        time.sleep(3.0)
        
        # Check position after wait
        position_after_wait = self.drone_positions.get(self.drone_id, initial_position)
        altitude_change = position_after_wait[2] - initial_position[2]
        
        self.get_logger().info(f"Position after stability wait: {position_after_wait}")
        self.get_logger().info(f"Altitude change during takeoff: {altitude_change:.2f}m")
        
        # Check if altitude is close to target
        if abs(position_after_wait[2] - self.altitude) > 0.5:  # More than 0.5m difference
            self.get_logger().warning(f"Drone altitude {position_after_wait[2]:.2f}m differs from target {self.altitude:.2f}m by {position_after_wait[2] - self.altitude:.2f}m")
            self.get_logger().warning("This may be due to ground effect, obstacles, or controller issues")
        
        # Start the mission
        self.running = True
        
        # Log subscription status after takeoff
        topics = sorted([t[0] for t in self.get_topic_names_and_types()])
        position_topics = [t for t in topics if 'pose' in t or 'position' in t or 'localization' in t]
        self.get_logger().info(f"After takeoff - Available position topics: {position_topics}")
        self.get_logger().info(f"Position received: {self.position_updates_received} updates")
        self.get_logger().info(f"Current position: {self.drone_positions.get(self.drone_id, 'Unknown')}")
        self.get_logger().info(f"Subscribed drone states: {list(self.drone_states.keys())}")
        topic_name = f'/swarm/state/{self.drone_id}'
        self.get_logger().info(f"Mission started, publishing to topic: {topic_name}")
    
    def land_and_end(self):
        """Land and end the mission"""
        self.running = False
        self.get_logger().info("Landing...")
        
        # Set LED to orange for landing
        self.change_led_color((1.0, 0.5, 0.0))
        
        # Land the drone
        self.drone.land(0.5)
        
        # Disarm
        self.drone.disarm()
        
        self.get_logger().info("Mission ended")
    
    def return_to_home(self):
        """Return to home position"""
        self.get_logger().info("Returning to home position...")
        
        # Set LED to white for return
        self.change_led_color((1.0, 1.0, 1.0))
        
        # Go to position above home
        self.drone.go_to(
            self.home_position[0],
            self.home_position[1],
            self.altitude,
            speed=0.5,
            wait=True
        )
    
    def shutdown(self):
        """Clean shutdown"""
        self.running = False
        self.drone.shutdown()
        self.get_logger().info("Drone shut down")

    def perform_recovery_actions(self):
        """Perform more drastic recovery actions if position updates are stalled"""
        current_time = time.time()
        time_since_last_update = current_time - self.last_position_update_time
        time_since_last_recovery = current_time - self.last_recovery_time
        
        # Only attempt recovery if:
        # 1. Position updates have been stalled for at least 10 seconds
        # 2. We haven't tried recovery in the last 30 seconds
        # 3. We've already attempted position topic resubscription
        if time_since_last_update > 10.0 and time_since_last_recovery > 30.0:
            self.get_logger().warning(f"Position updates stalled for {time_since_last_update:.1f}s - attempting recovery")
            self.recovery_attempts += 1
            self.last_recovery_time = current_time
            
            try:
                # Try force-updating position from config
                if self.home_position:
                    # Force an update to the current position with home position + altitude
                    x, y, _ = self.home_position
                    # Use current altitude if available, otherwise use target altitude
                    z = self.drone_positions.get(self.drone_id, [0, 0, self.altitude])[2]
                    
                    updated_pos = [x, y, z]
                    self.get_logger().warning(f"RECOVERY: Force-updating position to {updated_pos}")
                    
                    # Create a fake message for position update
                    fake_msg = PoseStamped()
                    fake_msg.header.frame_id = "recovery"
                    fake_msg.pose.position.x = x
                    fake_msg.pose.position.y = y
                    fake_msg.pose.position.z = z
                    
                    # Process position update
                    self.position_callback(fake_msg, "recovery")
                
                # Depending on severity, try more drastic measures
                if self.recovery_attempts > 1 and self.recovery_attempts <= 3:
                    # Try querying drone position directly
                    self.get_logger().warning("RECOVERY: Requesting direct position update from drone interface")
                    try:
                        # This is a hypothetical method - adjust based on actual API
                        if hasattr(self.drone, 'get_current_pose'):
                            pose = self.drone.get_current_pose()
                            self.get_logger().info(f"Got direct position from drone interface: {pose}")
                    except Exception as e:
                        self.get_logger().error(f"Failed to get direct position: {e}")
                
                # Last resort - if we're still not getting updates and drone is running
                if self.recovery_attempts >= 4 and self.running:
                    self.get_logger().error("CRITICAL: Multiple recovery attempts failed - forcing mission reset")
                    
                    # Create a more robust reset procedure
                    try:
                        # Save state
                        current_formation = self.current_formation
                        current_step = self.current_step
                        
                        # Reset position tracking
                        current_pos = self.drone_positions.get(self.drone_id, [0, 0, 0])
                        # Keep altitude, reset XY to home position
                        home_x, home_y, _ = self.home_position
                        self.drone_positions[self.drone_id] = [home_x, home_y, current_pos[2]]
                        
                        # Force new target calculation
                        self.current_target_position = None
                        self.commanded_target = None
                        
                        # Reset progress flags
                        self.at_target_position = False
                        
                        # Calculate new targets
                        target_position, rotation_angle = self.calculate_target_position()
                        
                        # If the drone is far from where it should be, try to move it to the right position
                        self.get_logger().warning(f"RECOVERY: Commanding drone to move to target: {target_position}")
                        yaw_angle = self.get_target_yaw(rotation_angle)
                        self.drone.go_to(
                            target_position[0],
                            target_position[1],
                            target_position[2],
                            speed=0.5,
                            yaw_mode=YawMode.PATH_FACING,
                            yaw_angle=yaw_angle,
                            wait=False
                        )
                        self.commanded_target = np.array(target_position, dtype=float)
                        
                    except Exception as e:
                        self.get_logger().error(f"Error during critical recovery: {e}")
                        import traceback
                        self.get_logger().error(traceback.format_exc())
            
            except Exception as e:
                self.get_logger().error(f"Error in recovery action: {e}")
                import traceback
                self.get_logger().error(traceback.format_exc())

# --- Main multi-drone launcher ---
def launch_multi_drone_mission(drone_ids, config_path, use_sim_time=False, verbose=False, wait_time=60.0):
    """Launch multiple drones in a single process with a multi-threaded executor"""
    print(f"Launching decentralized swarm mission with {len(drone_ids)} drones")
    
    # Initialize nodes for each drone
    drone_nodes = {}
    executor = MultiThreadedExecutor(num_threads=len(drone_ids)*2)  # More threads for better handling
    
    # Create and add nodes to executor
    for drone_id in drone_ids:
        print(f"Initializing drone: {drone_id}")
        drone_node = SwarmDroneNode(
            drone_id=drone_id,
            config_path=config_path,
            use_sim_time=use_sim_time,
            verbose=verbose
            )
        
        drone_nodes[drone_id] = drone_node
        executor.add_node(drone_node)
    
    # Allow time for topics to register by spinning the executor
    print("Spinning executor to allow topic registration...")
    for _ in range(20):  # Spin for about 2 seconds total
        executor.spin_once(timeout_sec=0.1)
    
    # Check topics after registration
    print("Available topics after registration:")
    for drone_id, node in drone_nodes.items():
        topics = node.get_topic_names_and_types()
        state_topics = [t for t, _ in topics if '/state' in t]
        if state_topics:
            print(f"{drone_id} state topics: {state_topics}")
    
    # Start takeoff sequence for all drones
    print("Starting takeoff sequence for all drones...")
    takeoff_threads = []
    for drone_id, node in drone_nodes.items():
        t = threading.Thread(target=node.takeoff_and_start, daemon=True)
        t.start()
        takeoff_threads.append(t)
    
    # While waiting for takeoff, keep spinning the executor to process messages
    print("Processing messages during takeoff...")
    while any(t.is_alive() for t in takeoff_threads):
        executor.spin_once(timeout_sec=0.1)
    
    print("All drones are airborne and in formation mode")
    
    # Allow initial message exchange by spinning the executor
    print("Allowing initial state message exchange...")
    for _ in range(50):  # Spin for about 5 seconds total
        executor.spin_once(timeout_sec=0.1)
    
    # Ensure all nodes have received each other's messages
    print("Checking communication between drones...")
    for drone_id, node in drone_nodes.items():
        connected_drones = list(node.drone_states.keys())
        print(f"{drone_id} has received messages from: {connected_drones}")
    
    # Set a timeout for the mission
    start_time = time.time()
    mission_timeout = start_time + wait_time
    
    # Run the executor to process callbacks
    try:
        # Spin with timeout to prevent hanging indefinitely
        print(f"Starting mission with {wait_time} second timeout...")
        while time.time() < mission_timeout:
            executor.spin_once(timeout_sec=0.1)
            
            # Check if any drone has completed its mission
            all_completed = True
            any_completed = False
            for drone_id, node in drone_nodes.items():
                if node.mission_complete:
                    any_completed = True
                else:
                    all_completed = False
            
            # If all drones completed or at least one completed and we've been running a while
            if all_completed or (any_completed and time.time() - start_time > wait_time/2):
                print("Mission objectives achieved, ending mission")
                break
                
            # Debug output every 10 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0 and int(elapsed) > 0 and int(elapsed) % 10 < 0.2:  # Only print once per 10s
                print(f"Mission running for {elapsed:.1f} seconds...")
                # Show current state counts
                for drone_id, node in drone_nodes.items():
                    print(f"{drone_id} has received {sum(node.state_messages_received.values())} state messages")
                    print(f"{drone_id} ready: {node.all_drones_ready}, states: {list(node.drone_states.keys())}")
        
    except KeyboardInterrupt:
        print("Keyboard interrupt received, landing all drones...")
    finally:
        # Land all drones
        print("Mission ending, returning all drones home...")
        for drone_id, node in drone_nodes.items():
            if node.running:
                print(f"Landing {drone_id}...")
                node.return_to_home()
                node.land_and_end()
            node.shutdown()
            node.destroy_node()

def main():
    parser = argparse.ArgumentParser(description='Decentralized multi-drone formation flight')
    parser.add_argument('-n', '--namespace', type=str, nargs='+',
                        help='Drone ID(s) (e.g., drone0 or multiple: drone0 drone1 drone2...)')
    parser.add_argument('--config', type=str, 
                        default='/home/ziyar/project_gazebo_ws/src/challenge_multi_drone/scenarios/scenario1_stage1.yaml',
                        help='Path to scenario configuration YAML file')
    parser.add_argument('--sim-time', action='store_true', help='Use simulation time')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--multi-process', action='store_true', help='Launch multiple processes for multiple drones')
    parser.add_argument('--wait-time', type=float, default=60.0, 
                        help='How long to wait for communication before timing out (seconds)')
    args = parser.parse_args()
    
    # Check if we have drone IDs
    if not args.namespace:
        parser.error("At least one drone ID is required via -n/--namespace")
        return 1
    
    # Check if multi-process mode with multiple drones
    if len(args.namespace) > 1 and args.multi_process:
        print(f"Launching {len(args.namespace)} separate processes for drones: {args.namespace}")
        
        # Launch a separate process for each drone
        import subprocess
        processes = []
        
        script_path = os.path.abspath(__file__)
        for drone_id in args.namespace:
            cmd = [
                'python3', script_path, 
                '-n', drone_id,
                '--config', args.config
            ]
            
            if args.sim_time:
                cmd.append('--sim-time')
            if args.verbose:
                cmd.append('-v')
                
            print(f"Launching: {' '.join(cmd)}")
            proc = subprocess.Popen(cmd)
            processes.append(proc)
        
        # Wait for all processes to complete
        try:
            for proc in processes:
                proc.wait()
        except KeyboardInterrupt:
            print("Keyboard interrupt received, terminating all drone processes...")
            for proc in processes:
                proc.terminate()
        
        return 0

    # Check if multiple drones provided
    if len(args.namespace) > 1:
        # Initialize ROS
        rclpy.init()
        
        try:
            # Launch multi-drone mission in a single process
            launch_multi_drone_mission(
                drone_ids=args.namespace,
                config_path=args.config,
                use_sim_time=args.sim_time,
                    verbose=args.verbose,
                wait_time=args.wait_time
            )
        finally:
            rclpy.shutdown()
        
        return 0
    else:
        # Single drone mode - run just one drone
        drone_id = args.namespace[0]
        
        # Initialize ROS
        rclpy.init()
        
        drone_node = None
        
        try:
            # Create the drone node
            drone_node = SwarmDroneNode(
                drone_id=drone_id,
                config_path=args.config,
                use_sim_time=args.sim_time,
                verbose=args.verbose
            )
            
            # Take off and start mission
            drone_node.takeoff_and_start()
            
            # Spin to process callbacks
            executor = MultiThreadedExecutor()
            executor.add_node(drone_node)
            
            try:
                executor.spin()
            finally:
                executor.shutdown()
            
        except KeyboardInterrupt:
            print("Keyboard interrupt received, shutting down...")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean shutdown
            if drone_node:
                if drone_node.running:
                    drone_node.return_to_home()
                    drone_node.land_and_end()
                drone_node.shutdown()
                drone_node.destroy_node()
        
    rclpy.shutdown()

    return 0

if __name__ == '__main__':
    sys.exit(main())