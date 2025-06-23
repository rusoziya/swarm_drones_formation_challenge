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
import json

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

def compute_euclidean_distance(pos1, pos2):
    """Compute Euclidean distance between two 3D points"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))

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
        
        # Initialize position tracking
        self.position = [0.0, 0.0, 0.0]
        self.position_history = []
        self.position_timestamp = time.time()
        self.position_source = None
        self.has_initial_position = False
        
        # State variables
        self.state = 'INIT'
        self.state_changed_time = time.time()
        self.takeoff_altitude = altitude
        self.mission_speed = 1.0  # m/s
        self.takeoff_start_time = 0.0
        self.landing_start_time = 0.0
        self.target_position = None
        self.leader_position = None
        self.has_reached_target = False
        self.current_formation_index = 0
        self.offset_angle = 0.0
        self.step = 0
        
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

    # ------------------------------------------------------------------------------
    # Callback functions for position updates and communication
    # ------------------------------------------------------------------------------
    
    def position_callback(self, msg, source):
        """Update drone position from pose message"""
        now = time.time()
        
        # Extract position and save with timestamp
        position = msg.pose.position
        self.position = [position.x, position.y, position.z]
        self.position_timestamp = now
        self.position_source = source
        
        # Update position history
        self.position_history.append((self.position, now))
        # Keep only the last 100 positions
        if len(self.position_history) > 100:
            self.position_history.pop(0)
        
        # Handle initial position setting
        if not self.has_initial_position:
            self.has_initial_position = True
            self.update_drone_state('INIT')
            self.get_logger().info(f"Initial position received: {self.position}")
            
            # Set home position if not already set
            if self.home_position is None:
                self.home_position = self.position.copy()
                self.get_logger().info(f"Home position set to: {self.home_position}")
    
    def check_position_updates(self):
        """Check if position updates have stalled"""
        if not self.has_initial_position:
            return
            
        now = time.time()
        # If no position update for 2 seconds, warn
        if now - self.position_timestamp > 2.0:
            source_info = f"last source: {self.position_source}" if self.position_source else "no source"
            self.get_logger().warning(
                f"No position update for {now - self.position_timestamp:.1f} seconds ({source_info})")
    
    def publish_state(self):
        """Publish drone state information for other drones"""
        if not self.has_initial_position:
            return
            
        state_msg = String()
        
        # Create a simple json string with position and state
        state_data = {
            'id': self.drone_id,
            'state': self.state,
            'position': self.position,
            'timestamp': time.time(),
            'formation_index': self.current_formation_index,
            'offset_angle': self.offset_angle,
            'step': self.step
        }
        
        state_msg.data = json.dumps(state_data)
        self.state_pub.publish(state_msg)
    
    def drone_state_callback(self, msg, drone_id):
        """Process state updates from other drones"""
        try:
            state_data = json.loads(msg.data)
            
            # Update tracked state for this drone
            self.drone_states[drone_id] = {
                'state': state_data['state'],
                'position': state_data['position'],
                'timestamp': state_data['timestamp'],
                'formation_index': state_data['formation_index'],
                'offset_angle': state_data['offset_angle'],
                'step': state_data['step'],
                'received_at': time.time()
            }
            
            # Process necessary formation consensus
            self.check_formation_consensus()
            
        except Exception as e:
            self.get_logger().error(f"Error processing state from {drone_id}: {e}")
    
    def consensus_callback(self, msg, drone_id):
        """Process consensus messages from other drones"""
        try:
            consensus_data = json.loads(msg.data)
            
            # Record the vote from this drone
            self.formation_votes[drone_id] = consensus_data.get('formation_vote', -1)
            
            # Update timestamp
            self.last_consensus_time = time.time()
            
            # Check if we have enough votes to change formation
            self.check_formation_consensus()
            
        except Exception as e:
            self.get_logger().error(f"Error processing consensus from {drone_id}: {e}")
    
    def check_formation_consensus(self):
        """Check if there's consensus on formation change"""
        # Only the primary drone broadcasts formation changes
        if self.drone_id != 'drone0':
            return
            
        # Count votes for each formation
        vote_counts = {}
        for drone_id, vote in self.formation_votes.items():
            if vote >= 0:
                vote_counts[vote] = vote_counts.get(vote, 0) + 1
        
        # Check if any formation has majority vote
        for formation_idx, count in vote_counts.items():
            if count >= len(self.drone_ids) // 2 + 1:  # Simple majority
                if formation_idx != self.current_formation_index:
                    self.get_logger().info(f"Formation consensus reached: switching to {self.formation_names[formation_idx]}")
                    self.current_formation_index = formation_idx
                    
                    # Broadcast the consensus
                    self.broadcast_formation_consensus(formation_idx)
                    
                    # Reset votes after successful consensus
                    self.formation_votes = {d_id: -1 for d_id in self.drone_ids}
                    
                    # Reset target reached state
                    self.has_reached_target = False
                break
    
    def broadcast_formation_consensus(self, formation_idx):
        """Broadcast formation consensus to all drones"""
        consensus_msg = String()
        
        consensus_data = {
            'id': self.drone_id,
            'formation_consensus': formation_idx,
            'timestamp': time.time()
        }
        
        consensus_msg.data = json.dumps(consensus_data)
        self.consensus_pub.publish(consensus_msg)
    
    def update_drone_state(self, new_state):
        """Update this drone's state"""
        if self.state != new_state:
            self.get_logger().info(f"State change: {self.state} -> {new_state}")
            self.state = new_state
            
            # On certain state changes, send consensus information
            if new_state in ['READY', 'FORMATION_COMPLETE']:
                # Vote for next formation if we've just completed one
                if new_state == 'FORMATION_COMPLETE' and self.drone_id == 'drone0':
                    next_formation = (self.current_formation_index + 1) % len(self.formation_functions)
                    self.formation_votes[self.drone_id] = next_formation
                    
                    # Broadcast the vote
                    consensus_msg = String()
                    consensus_data = {
                        'id': self.drone_id,
                        'formation_vote': next_formation,
                        'timestamp': time.time()
                    }
                    consensus_msg.data = json.dumps(consensus_data)
                    self.consensus_pub.publish(consensus_msg)
    
    def perform_recovery_actions(self):
        """Perform recovery actions if needed"""
        # Only check if we're in a potentially stuck state
        if self.state not in ['TAKEOFF', 'FORMATION', 'LANDING']:
            # Reset recovery attempts counter when not in recovery-needing state
            self.recovery_attempts = 0
            return
            
        now = time.time()
        # If we've been in the same state for too long
        if now - self.state_changed_time > 60.0:  # 1 minute in same state
            self.recovery_attempts += 1
            self.get_logger().warning(
                f"Recovery attempt {self.recovery_attempts}: Drone stuck in {self.state} state for {now - self.state_changed_time:.1f} seconds")
            
            # Take appropriate recovery action based on state
            if self.state == 'TAKEOFF' and self.recovery_attempts < 3:
                # Try taking off again
                self.drone.takeoff()
                self.get_logger().info("Recovery: Re-attempting takeoff")
                
            elif self.state == 'FORMATION' and self.recovery_attempts < 3:
                # Check if we're making progress towards the target
                if self.has_initial_position and self.target_position:
                    dist = compute_euclidean_distance(self.position, self.target_position)
                    if dist > 0.5:  # If we're far from target
                        # Move again to target
                        self.drone.move_to_position(
                            self.target_position[0], 
                            self.target_position[1], 
                            self.target_position[2],
                            self.mission_speed)
                        self.get_logger().info(f"Recovery: Re-attempting move to {self.target_position}")
                        
            elif self.state == 'LANDING' and self.recovery_attempts < 3:
                # Try landing again
                self.drone.land()
                self.get_logger().info("Recovery: Re-attempting landing")
                
            # If too many recovery attempts, try to reset to a safer state
            elif self.recovery_attempts >= 3:
                self.get_logger().error(f"Multiple recovery attempts failed. Attempting to reset state.")
                
                # Return home and land as last resort
                if self.has_initial_position and self.home_position:
                    self.drone.move_to_position(
                        self.home_position[0],
                        self.home_position[1],
                        self.home_position[2],
                        self.mission_speed)
                    self.get_logger().warning(f"Emergency return to home position {self.home_position}")
                    
                # After some time, try landing
                time.sleep(5.0)
                self.drone.land()
                self.get_logger().warning("Emergency landing initiated")
                self.update_drone_state('EMERGENCY')

    def has_reached_target(self, current, target, xy_tol=0.25, z_tol=0.40):
        """Check if drone has reached target position within tolerance"""
        if current is None or target is None:
            return False
            
        # Check XY distance first
        xy_dist = math.sqrt((current[0] - target[0])**2 + (current[1] - target[1])**2)
        if xy_dist > xy_tol:
            return False
            
        # Then check Z distance
        z_dist = abs(current[2] - target[2])
        if z_dist > z_tol:
            return False
            
        return True

    # ------------------------------------------------------------------------------
    # Mission logic and formation functions
    # ------------------------------------------------------------------------------
    
    def mission_step(self):
        """Main mission logic, executed at regular intervals"""
        if not self.has_initial_position:
            return
        
        # Check if all drones needed for the mission are present
        if not self.check_swarm_readiness():
            return
            
        # State machine for the mission
        if self.state == 'INIT':
            self.start_mission()
            
        elif self.state == 'READY':
            self.execute_takeoff()
            
        elif self.state == 'TAKEOFF':
            self.check_takeoff_completion()
            
        elif self.state == 'FORMATION':
            self.execute_formation_flight()
            
        elif self.state == 'LANDING':
            self.check_landing_completion()
    
    def check_swarm_readiness(self):
        """Check if all drones in the swarm are ready"""
        # Check if we're tracking all expected drones
        now = time.time()
        active_drones = 0
        
        for drone_id in self.drone_ids:
            # Skip own drone
            if drone_id == self.drone_id:
                active_drones += 1
                continue
                
            # Check if we've received state from this drone recently
            if drone_id in self.drone_states:
                state_info = self.drone_states[drone_id]
                if now - state_info['received_at'] < 5.0:  # Within 5 seconds
                    active_drones += 1
        
        # Need at least 3 drones for meaningful formations
        if active_drones >= 3 and active_drones == len(self.drone_ids):
            if self.state == 'INIT' and self.has_initial_position:
                self.update_drone_state('READY')
            return True
        
        if self.state not in ['INIT', 'READY']:
            self.get_logger().warning(f"Only {active_drones}/{len(self.drone_ids)} drones active. Waiting for full swarm.")
        
        return False
    
    def start_mission(self):
        """Initialize mission parameters and prepare for takeoff"""
        if self.state != 'INIT':
            return
            
        # Initialize formation parameters
        self.formation_functions = [
            self.formation_line,
            self.formation_v,
            self.formation_diamond,
            self.formation_circular_orbit,
            self.formation_grid,
            self.formation_staggered
        ]
        
        self.formation_names = [
            "Line Formation", 
            "V Formation", 
            "Diamond Formation", 
            "Circular Orbit", 
            "Grid Formation", 
            "Staggered Formation"
        ]
        
        self.current_formation_index = 0
        self.has_reached_target = False
        self.offset_angle = 0.0
        self.step = 0
        
        # Mark as ready for takeoff
        self.update_drone_state('READY')
        self.get_logger().info("Mission initialized, ready for takeoff")
    
    def execute_takeoff(self):
        """Begin takeoff sequence"""
        if self.state != 'READY':
            return
            
        # Start the takeoff
        self.drone.takeoff()
        self.update_drone_state('TAKEOFF')
        self.get_logger().info("Takeoff initiated")
        self.takeoff_start_time = time.time()
    
    def check_takeoff_completion(self):
        """Check if the drone has reached takeoff altitude"""
        if self.state != 'TAKEOFF':
            return
            
        # Safety timeout for takeoff
        if time.time() - self.takeoff_start_time > 20.0:
            self.get_logger().info("Takeoff timeout reached, assuming completed")
            self.update_drone_state('FORMATION')
            return
            
        # Check if we're at target altitude
        if self.has_initial_position and self.position[2] >= self.takeoff_altitude - 0.3:
            self.get_logger().info(f"Takeoff complete, reached altitude {self.position[2]:.2f}m")
            self.update_drone_state('FORMATION')
    
    def execute_formation_flight(self):
        """Execute the current formation flight pattern"""
        if self.state != 'FORMATION':
            return
            
        # Get position in the current formation
        target_pos = self.calculate_formation_position()
        if target_pos is None:
            self.get_logger().warning("Could not calculate formation position")
            return
            
        # Store target position for recovery if needed
        self.target_position = target_pos
        
        # If we're already at target, don't send redundant commands
        if self.has_reached_target:
            # Circular orbit is special - keep moving even when target is reached
            if self.current_formation_index == 3:  # Circular orbit
                self.offset_angle += 0.05  # Small increment for smooth orbit
                if self.offset_angle >= 2.0 * math.pi:
                    self.offset_angle -= 2.0 * math.pi
                
                # Recalculate position with new angle
                target_pos = self.calculate_formation_position()
                if target_pos is None:
                    return
                    
                # Move to the new position
                self.drone.move_to_position(
                    target_pos[0], target_pos[1], target_pos[2], self.mission_speed)
            return
            
        # Check if we've reached the target position
        if self.has_initial_position:
            dist = compute_euclidean_distance(self.position, target_pos)
            
            if dist < 0.4:  # Within 40cm is close enough
                self.has_reached_target = True
                self.get_logger().info(f"Reached formation position: {target_pos}")
                
                # Check if all drones in formation have reached their positions
                all_reached = True
                for drone_id in self.drone_states:
                    drone_state = self.drone_states[drone_id]['state']
                    if drone_state != 'FORMATION_COMPLETE':
                        all_reached = False
                        break
                
                # If primary drone and all drones are in position, signal formation complete
                if all_reached and self.drone_id == 'drone0':
                    self.update_drone_state('FORMATION_COMPLETE')
                    
                    # After formation, increment step or angle for next time
                    if self.current_formation_index != 3:  # Not circular orbit
                        self.step += 1
                        if self.step >= 4:  # After 4 steps, change formation
                            self.step = 0
                    
                return
            
            # Move to the target position
            self.drone.move_to_position(
                target_pos[0], target_pos[1], target_pos[2], self.mission_speed)
            self.get_logger().debug(f"Moving to formation position: {target_pos}, distance: {dist:.2f}m")
    
    def initiate_landing(self):
        """Begin landing sequence"""
        if self.state not in ['FORMATION', 'FORMATION_COMPLETE']:
            return
            
        # Move to the home position before landing
        if self.home_position:
            self.drone.move_to_position(
                self.home_position[0], self.home_position[1], self.takeoff_altitude, self.mission_speed)
            self.get_logger().info(f"Returning to home position for landing: {self.home_position}")
            
            # Wait to reach the position
            time.sleep(5.0)
            
        # Land
        self.drone.land()
        self.update_drone_state('LANDING')
        self.get_logger().info("Landing initiated")
        self.landing_start_time = time.time()
    
    def check_landing_completion(self):
        """Check if the drone has completed landing"""
        if self.state != 'LANDING':
            return
            
        # Safety timeout for landing
        if time.time() - self.landing_start_time > 20.0:
            self.get_logger().info("Landing timeout reached, assuming completed")
            self.update_drone_state('LANDED')
            return
            
        # Check altitude to determine if landed
        if self.has_initial_position and self.position[2] <= 0.2:
            self.get_logger().info("Landing complete")
            self.update_drone_state('LANDED')
    
    def calculate_formation_position(self):
        """Calculate drone's position in the current formation"""
        if not self.find_formation_leader():
            self.get_logger().warning("Cannot find formation leader position")
            return None
            
        # Get current formation function
        formation_func = self.formation_functions[self.current_formation_index]
        
        # Calculate position based on drone index in formation
        drone_ids = sorted(list(self.drone_ids))
        drone_index = drone_ids.index(self.drone_id)
        
        # Calculate offset from leader
        offset = formation_func(drone_index, len(drone_ids))
        
        # Apply offset to leader position
        target_pos = [
            self.leader_position[0] + offset[0],
            self.leader_position[1] + offset[1],
            self.takeoff_altitude  # Keep consistent altitude
        ]
        
        return target_pos
    
    def find_formation_leader(self):
        """Find the position of the formation leader"""
        # In decentralized architecture, we use a virtual leader concept
        # Use drone0's position as reference, or centroid if drone0 not available
        
        if 'drone0' in self.drone_states:
            # Use drone0 as the leader
            leader_state = self.drone_states['drone0']
            self.leader_position = leader_state['position']
            return True
        elif self.drone_id == 'drone0':
            # If this is drone0, use own position
            self.leader_position = self.position
            return True
        else:
            # Calculate centroid of all known drone positions
            positions = []
            for drone_id, state in self.drone_states.items():
                positions.append(state['position'])
                
            # Add own position
            if self.has_initial_position:
                positions.append(self.position)
                
            if not positions:
                return False
                
            # Calculate centroid
            centroid = [0.0, 0.0, 0.0]
            for pos in positions:
                centroid[0] += pos[0]
                centroid[1] += pos[1]
                centroid[2] += pos[2]
                
            centroid[0] /= len(positions)
            centroid[1] /= len(positions)
            centroid[2] /= len(positions)
            
            self.leader_position = centroid
            return True
    
    # ------------------------------------------------------------------------------
    # Formation calculation functions
    # ------------------------------------------------------------------------------
    
    def formation_line(self, drone_index, total_drones):
        """Calculate position offset for line formation"""
        # Horizontal line with fixed spacing
        spacing = 2.0  # 2 meters between drones
        
        # Center the formation
        start_x = -(total_drones - 1) * spacing / 2
        
        return [start_x + drone_index * spacing, 0.0, 0.0]
    
    def formation_v(self, drone_index, total_drones):
        """Calculate position offset for V formation"""
        # V formation with fixed spacing
        spacing = 2.0  # 2 meters between drones
        angle = math.pi / 4  # 45 degrees
        
        # Position depends on index being odd or even
        if drone_index == 0:  # Leader at the front
            return [0.0, 0.0, 0.0]
            
        # Determine which leg of the V
        if drone_index % 2 == 1:  # Right leg
            leg_index = (drone_index + 1) // 2
            x = leg_index * spacing * math.cos(angle)
            y = -leg_index * spacing * math.sin(angle)
        else:  # Left leg
            leg_index = drone_index // 2
            x = -leg_index * spacing * math.cos(angle)
            y = -leg_index * spacing * math.sin(angle)
            
        return [x, y, 0.0]
    
    def formation_diamond(self, drone_index, total_drones):
        """Calculate position offset for diamond formation"""
        spacing = 2.0  # meters between drones
        
        # Different positions based on drone index
        if drone_index == 0:  # Front
            return [0.0, spacing, 0.0]
        elif drone_index == 1:  # Right
            return [spacing, 0.0, 0.0]
        elif drone_index == 2:  # Back
            return [0.0, -spacing, 0.0]
        elif drone_index == 3:  # Left
            return [-spacing, 0.0, 0.0]
        else:  # Additional drones form outer diamond
            angle = 2.0 * math.pi * (drone_index - 4) / (total_drones - 4)
            radius = spacing * 1.5
            return [radius * math.cos(angle), radius * math.sin(angle), 0.0]
    
    def formation_circular_orbit(self, drone_index, total_drones):
        """Calculate position offset for circular orbit formation"""
        radius = 3.0  # meters from center
        
        # Distribute drones evenly around the circle
        angle = 2.0 * math.pi * drone_index / total_drones + self.offset_angle
        
        return [radius * math.cos(angle), radius * math.sin(angle), 0.0]
    
    def formation_grid(self, drone_index, total_drones):
        """Calculate position offset for grid formation"""
        spacing = 2.0  # meters between drones
        
        # Determine grid dimensions
        side_length = math.ceil(math.sqrt(total_drones))
        
        # Calculate row and column
        row = drone_index // side_length
        col = drone_index % side_length
        
        # Center the grid
        start_x = -(side_length - 1) * spacing / 2
        start_y = -(side_length - 1) * spacing / 2
        
        return [start_x + col * spacing, start_y + row * spacing, 0.0]
    
    def formation_staggered(self, drone_index, total_drones):
        """Calculate position offset for staggered formation"""
        spacing_x = 2.0  # meters between drones in x
        spacing_y = 1.5  # meters between drones in y
        
        # Alternating rows
        row = drone_index // 3
        col = drone_index % 3
        
        # Offset every other row
        offset_x = spacing_x / 2 if row % 2 == 1 else 0.0
        
        # Center the formation
        start_x = -spacing_x
        start_y = -(spacing_y * (total_drones // 3)) / 2
        
        return [start_x + col * spacing_x + offset_x, start_y + row * spacing_y, 0.0] 

class DecentralizedDroneMission(Node):
    """Manages a decentralized drone swarm mission with multiple drones"""
    
    def __init__(self, config_path="config/scenario1.yaml", altitude=2.0, use_sim_time=True, verbose=False):
        super().__init__('decentralized_drone_mission')
        
        # Basic properties
        self.verbose = verbose
        self.altitude = altitude
        self.config_path = config_path
        
        # Set logger level
        if verbose:
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
        else:
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)
        
        # Identify the drone ID from ROS node name or environment variable
        self.drone_id = os.environ.get('DRONE_ID', 'drone0')
        self.get_logger().info(f"Initializing mission for {self.drone_id}")
        
        # Create the swarm drone node
        self.drone_node = SwarmDroneNode(
            drone_id=self.drone_id,
            config_path=self.config_path,
            altitude=self.altitude,
            use_sim_time=use_sim_time,
            verbose=self.verbose
        )
        
    def mission_step(self):
        """Forward the mission step call to the drone node"""
        if hasattr(self.drone_node, 'mission_step'):
            self.drone_node.mission_step()
        else:
            self.get_logger().error("SwarmDroneNode has no mission_step method")
    
    def perform_recovery_actions(self):
        """Forward recovery actions to the drone node"""
        if hasattr(self.drone_node, 'perform_recovery_actions'):
            self.drone_node.perform_recovery_actions()
        else:
            self.get_logger().error("SwarmDroneNode has no perform_recovery_actions method")
    
    def destroy_node(self):
        """Clean up both nodes on shutdown"""
        self.drone_node.destroy_node()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    # Parse command line arguments for drone configuration
    parser = argparse.ArgumentParser(description='Decentralized drone mission')
    parser.add_argument('-n', nargs='+', help='List of drone IDs to launch', default=[])
    parser.add_argument('--drone_id', type=str, help='Single drone ID (e.g., drone0)', default=os.environ.get('DRONE_ID', 'drone0'))
    parser.add_argument('--config', type=str, help='Path to config file', default='config/scenario1.yaml')
    parser.add_argument('--altitude', type=float, help='Flight altitude in meters', default=2.0)
    parser.add_argument('--use_sim_time', action='store_true', help='Use simulation time', default=True)
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parsed_args = parser.parse_args(args)
    
    # If drones are specified with -n, launch multiple drones
    if parsed_args.n:
        launch_multi_drone_mission(
            drone_ids=parsed_args.n,
            config_path=parsed_args.config,
            use_sim_time=parsed_args.use_sim_time,
            verbose=parsed_args.verbose
        )
        return

    # Otherwise, launch a single drone
    # Create the DecentralizedDroneMission node with parameters
    mission_node = DecentralizedDroneMission(
        config_path=parsed_args.config,
        altitude=parsed_args.altitude,
        use_sim_time=parsed_args.use_sim_time,
        verbose=parsed_args.verbose
    )
    
    # Use a MultiThreadedExecutor for parallel callbacks
    executor = MultiThreadedExecutor()
    executor.add_node(mission_node)
    
    try:
        # Run the executor in a separate thread
        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()
        
        # Main loop at 10Hz
        rate = mission_node.create_rate(10)
        
        while rclpy.ok():
            # Run mission step
            mission_node.mission_step()
            
            # Check if recovery actions needed
            mission_node.perform_recovery_actions()
            
            # Sleep to maintain rate
            rate.sleep()
            
    except KeyboardInterrupt:
        mission_node.get_logger().info('Keyboard interrupt, shutting down')
    finally:
        # Clean shutdown
        mission_node.destroy_node()
        rclpy.shutdown()

def launch_multi_drone_mission(drone_ids, config_path="config/scenario1.yaml", use_sim_time=False, verbose=False, wait_time=60.0):
    """Launch a multi-drone mission by spawning separate processes for each drone"""
    import subprocess
    import time
    import signal
    
    print(f"Launching multi-drone mission with drones: {drone_ids}")
    
    # Store processes
    processes = []
    
    try:
        # Launch a separate process for each drone
        for drone_id in drone_ids:
            cmd = [
                sys.executable,  # Python interpreter
                __file__,        # This script
                f"--drone_id={drone_id}",
                f"--config={config_path}"
            ]
            
            if use_sim_time:
                cmd.append("--use_sim_time")
                
            if verbose:
                cmd.append("--verbose")
                
            print(f"Launching: {' '.join(cmd)}")
            
            # Set DRONE_ID environment variable
            env = os.environ.copy()
            env["DRONE_ID"] = drone_id
            
            # Start process
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1  # Line buffered
            )
            
            processes.append((drone_id, process))
            print(f"Launched {drone_id} with PID {process.pid}")
            
            # Small delay between launches to avoid timing issues
            time.sleep(1.0)
            
        # Set up signal handler for clean shutdown
        def signal_handler(sig, frame):
            print("Shutting down all drone processes...")
            for drone_id, process in processes:
                print(f"Terminating {drone_id} (PID {process.pid})")
                process.terminate()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        
        # Monitor processes and their output
        print(f"All drones launched. Mission will run for {wait_time} seconds")
        start_time = time.time()
        
        while time.time() - start_time < wait_time:
            for drone_id, process in processes:
                # Check if process is still running
                if process.poll() is not None:
                    print(f"WARNING: {drone_id} process exited with code {process.returncode}")
                    
                # Read and print output
                output = process.stdout.readline()
                if output:
                    print(f"[{drone_id}] {output.strip()}")
                    
            time.sleep(0.1)  # Reduce CPU usage
            
        # Wait time elapsed, terminate all processes
        print("Mission time elapsed, shutting down all processes")
        for drone_id, process in processes:
            print(f"Terminating {drone_id} (PID {process.pid})")
            process.terminate()
            
    except Exception as e:
        print(f"Error in multi-drone mission: {e}")
        # Cleanup on error
        for drone_id, process in processes:
            if process.poll() is None:  # If process is still running
                process.terminate()
                
    finally:
        # Make sure all processes are terminated
        for drone_id, process in processes:
            if process.poll() is None:  # If process is still running
                try:
                    process.terminate()
                    process.wait(timeout=2.0)
                except:
                    process.kill()  # Force kill if terminate doesn't work
                    
    print("Multi-drone mission completed")
        
if __name__ == '__main__':
    main() 