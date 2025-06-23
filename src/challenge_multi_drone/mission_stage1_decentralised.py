#!/usr/bin/env python3
import argparse
import math
import sys
import time
import threading
import subprocess
import rclpy
import yaml
import os
import numpy as np
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from as2_msgs.msg import YawMode, BehaviorStatus
from as2_python_api.drone_interface import DroneInterface
from as2_python_api.behavior_actions.behavior_handler import BehaviorHandler
from std_msgs.msg import ColorRGBA, Float32MultiArray, String, Bool
from geometry_msgs.msg import PoseStamped, Pose, Vector3

# Constants
TAKE_OFF_HEIGHT = 2.0  # Height in meters
TAKE_OFF_SPEED = 0.7   # Max speed in m/s
POSITION_TOLERANCE = 0.05 # Position tolerance in meters
MOVE_SPEED = 1.0  # Speed for go_to movement in m/s
POSITION_REPORT_INTERVAL = 10.0  # Print positions every 10 seconds

# Formation constants
TOTAL_FORMATIONS = 6
STAGE_RADIUS = 3.0  # Radius of the circle path
STAGE_CENTER = (0.0, 0.0)  # Center point of the formation circle

# Steps and angles
STEPS_PER_FORMATION = 6  # Regular formations use 6 steps (30° each)
CIRCULAR_ORBIT_STEPS = 18  # Circular orbit uses 18 steps (10° each)
ANGLE_DEG = 30.0  # Regular formations advance 30° per step
CIRCULAR_ORBIT_ANGLE_DEG = 10.0  # Circular orbit advances 10° per step

def formation_line_5():
    """
    Return offsets for a line formation for 5 drones.
    Evenly spaced along the x-axis with drone2 (index 2) as the leader in the center.
    """
    d = 0.6  # distance between drones
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
    a clockwise direction, maintaining a clear circular shape at all times.
    """
    # Circle radius - increased for better visibility
    circle_radius = 1.0
    
    # Calculate positions around the circle
    angles = [0, math.pi/2, math.pi, 3*math.pi/2]  # four points on a circle
    offsets = [(circle_radius*math.cos(a), circle_radius*math.sin(a)) for a in angles]
    
    # Return offsets with the center point for the middle drone
    return [offsets[0], offsets[1], (0.0, 0.0), offsets[2], offsets[3]]

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

class DecentralizedDroneNode(Node):
    """Fully decentralized drone node that discovers other drones and coordinates via ROS messaging."""
    
    def __init__(self, drone_id, config_path, altitude=TAKE_OFF_HEIGHT, use_sim_time=True, verbose=False):
        super().__init__(f'{drone_id}_node')
        
        # Basic properties
        self.drone_id = drone_id
        self.verbose = verbose
        self.altitude = altitude
        self.drone_index = int(drone_id.replace('drone', ''))  # Extract drone index (0-4)
        self.running = False
        self.is_in_formation = False
        self.mission_complete = False
        self.all_drones_ready = False
        self.all_drones_started = False
        self.has_printed_mission_complete = False
        
        # Performance metrics tracking
        self.mission_success = False
        self.mission_start_time = None
        self.mission_end_time = None
        self.cpu_samples = []
        self.path_length = 0.0
        self.last_position = None
        self.max_formation_deviation = 0.0
        self.ideal_positions = {}  # Store ideal positions for deviation calculation
        
        # Path length tracking for all drones
        self.all_drone_path_lengths = {d_id: 0.0 for d_id in [f"drone{i}" for i in range(5)]}
        self.last_positions = {d_id: None for d_id in [f"drone{i}" for i in range(5)]}
        
        # Expected list of all drone IDs (for now, we assume 5 drones always, indexed 0-4)
        self.expected_drone_ids = [f"drone{i}" for i in range(5)]
        self.discovered_drones = {d_id: False for d_id in self.expected_drone_ids}
        self.discovered_drones[self.drone_id] = True  # We know about ourselves
        
        # Formation steps tracking
        self.current_step = 0  # Overall step counter
        self.formation_step = 0  # Step within the current formation
        self.current_formation = 0  # Current formation index (0-5)
        self.global_angle = 0.0  # Track overall angle across formations
        self.completed_formations = 0  # Track number of completed formations
        
        # Continuous orbit tracking
        self.orbit_start_time = None  # When we entered the circular orbit formation
        self.orbit_angular_speed = 0.5  # radians per second for continuous orbit
        self.orbit_radius = 1.0  # Radius of the orbit (same as in formation_circular_orbit_5)
        self.orbit_last_update = time.time()
        self.orbit_angle = 0.0  # Current angle in the orbit (for continuous movement)
        
        self.target_position = None
        self.stage_center = STAGE_CENTER
        self.radius = STAGE_RADIUS
        
        # Formation functions list
        self.formation_functions = [
            formation_line_5,
            formation_v_5,
            formation_diamond_5,
            formation_circular_orbit_5,
            formation_grid_5,
            formation_staggered_5
        ]
        
        self.formation_names = [
            "Line Formation", 
            "V Formation", 
            "Diamond Formation", 
            "Circular Orbit", 
            "Grid Formation", 
            "Staggered Formation"
        ]
        
        # Status tracking
        self.drone_ready_status = {d_id: False for d_id in self.expected_drone_ids}
        self.drone_ready_status[self.drone_id] = False  # We're not ready yet
        
        self.drone_formation_status = {d_id: False for d_id in self.expected_drone_ids}
        self.drone_formation_status[self.drone_id] = False
        
        # Home position tracking
        self.drone_home_status = {d_id: False for d_id in self.expected_drone_ids}
        self.drone_home_status[self.drone_id] = False
        self.returning_home = False
        self.landing_initiated = False
        self.landing_time = None
        
        # Position reached tracking
        self.is_moving = False
        self.position_reached_timeout = 30.0  # Maximum time to reach position before considering it failed
        self.last_move_time = 0.0
        
        # Step synchronization
        self.drone_steps = {d_id: 0 for d_id in self.expected_drone_ids}
        self.drone_steps[self.drone_id] = 0
        
        # Set up QoS profiles
        self.pose_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Initialize DroneInterface
        self.get_logger().info(f"Initializing drone interface for {drone_id}")
        self.drone = DroneInterface(drone_id, verbose=verbose, use_sim_time=use_sim_time)
        
        # Set up callback groups for concurrent callbacks
        self.timer_callback_group = MutuallyExclusiveCallbackGroup()
        self.status_callback_group = MutuallyExclusiveCallbackGroup()
        self.sub_callback_group = MutuallyExclusiveCallbackGroup()
        
        # Set up position tracking dictionaries
        self.drone_positions = {drone_id: [0.0, 0.0, 0.0] for drone_id in self.expected_drone_ids}
        self.position_timestamps = {drone_id: 0.0 for drone_id in self.expected_drone_ids}
        
        # Subscribe to our own position updates
        self.position_sub = self.create_subscription(
            PoseStamped,
            f'/{self.drone_id}/self_localization/pose',
            lambda msg: self.position_callback(msg, self.drone_id),
            self.pose_qos,
            callback_group=self.sub_callback_group
        )
        
        # Subscribe to other drones' position updates
        for d_id in self.expected_drone_ids:
            if d_id != self.drone_id:
                self.create_subscription(
                    PoseStamped,
                    f'/{d_id}/self_localization/pose',
                    lambda msg, src=d_id: self.position_callback(msg, src),
                    self.pose_qos,
                    callback_group=self.sub_callback_group
                )
                
        # Publish our own status
        self.status_pub = self.create_publisher(
            Bool,
            f'/{self.drone_id}/ready_status',
            self.reliable_qos
        )
        
        self.formation_pub = self.create_publisher(
            Bool,
            f'/{self.drone_id}/formation_status',
            self.reliable_qos
        )
        
        # Home position publisher and subscribers
        self.home_status_pub = self.create_publisher(
            Bool,
            f'/{self.drone_id}/home_status',
            self.reliable_qos
        )
        
        # Step publication - for synchronization
        self.step_pub = self.create_publisher(
            Float32MultiArray,  # Will contain [current_step, formation_step, formation_index]
            f'/{self.drone_id}/step_status',
            self.reliable_qos
        )
        
        # Subscribe to other drones' status
        for d_id in self.expected_drone_ids:
            if d_id != self.drone_id:
                self.create_subscription(
                    Bool,
                    f'/{d_id}/ready_status',
                    lambda msg, src=d_id: self.ready_status_callback(msg, src),
                    self.reliable_qos,
                    callback_group=self.status_callback_group
                )
                
                self.create_subscription(
                    Bool,
                    f'/{d_id}/formation_status',
                    lambda msg, src=d_id: self.formation_status_callback(msg, src),
                    self.reliable_qos,
                    callback_group=self.status_callback_group
                )
                
                self.create_subscription(
                    Float32MultiArray,
                    f'/{d_id}/step_status',
                    lambda msg, src=d_id: self.step_status_callback(msg, src),
                    self.reliable_qos,
                    callback_group=self.status_callback_group
                )
                
                # Subscribe to other drones' home status
                self.create_subscription(
                    Bool,
                    f'/{d_id}/home_status',
                    lambda msg, src=d_id: self.home_status_callback(msg, src),
                    self.reliable_qos,
                    callback_group=self.status_callback_group
                )
        
        # LED publisher
        self.led_pub = self.create_publisher(
            ColorRGBA, 
            f'/{self.drone_id}/led/control', 
            10
        )
        
        # Set up timers
        self.mission_timer = self.create_timer(
            0.2,  # 5 Hz
            self.mission_step,
            callback_group=self.timer_callback_group
        )
        
        # For continuous orbit updates
        self.orbit_timer = self.create_timer(
            0.05,  # 20 Hz for smoother orbit
            self.orbit_update,
            callback_group=self.timer_callback_group
        )
        
        self.status_timer = self.create_timer(
            1.0,  # 1 Hz
            self.publish_status,
            callback_group=self.status_callback_group
        )
        
        self.position_report_timer = self.create_timer(
            POSITION_REPORT_INTERVAL,
            self.report_position,
            callback_group=self.timer_callback_group
        )
        
        # Landing command publisher and subscribers
        self.landing_command_pub = self.create_publisher(
            Float32MultiArray,  # Will contain [timestamp for synchronized landing]
            f'/formation/landing_command',
            self.reliable_qos
        )
        
        # Subscribe to landing command
        self.landing_command_sub = self.create_subscription(
            Float32MultiArray,
            f'/formation/landing_command',
            self.landing_command_callback,
            self.reliable_qos,
            callback_group=self.status_callback_group
        )
        
        # Calculate initial target position for first formation
        self.calculate_target_position()
        
        # Load initial home positions
        self.home_positions = load_drone_home_positions()
        self.get_logger().info(f"Home position for {self.drone_id}: {self.home_positions.get(self.drone_id, 'Unknown')}")
        
        # Ready to start
        self.change_led_color((1.0, 1.0, 1.0))  # White - ready
        self.get_logger().info(f"{self.drone_id} initialized")
        
        # Mark ourselves as ready and start publishing status
        self.drone_ready_status[self.drone_id] = True
        
    def orbit_update(self):
        """Update orbit position for continuous circular movement."""
        # Only active during circular orbit formation
        if self.current_formation != 3 or not self.running or self.returning_home:
            return
            
        # Only for outer drones, not the center drone
        if self.drone_index == 2:  # Center drone
            return
            
        # If we just entered the orbit formation, initialize the orbit timer
        if self.orbit_start_time is None:
            self.orbit_start_time = time.time()
            self.orbit_last_update = self.orbit_start_time
            self.orbit_angle = 0.0
            
        # Calculate time-based smooth orbital motion
        current_time = time.time()
        delta_time = current_time - self.orbit_last_update
        
        # Update orbit angle - negative for clockwise rotation
        self.orbit_angle -= self.orbit_angular_speed * delta_time
        
        # Normalize angle to [0, 2π]
        while self.orbit_angle < 0:
            self.orbit_angle += 2 * math.pi
        while self.orbit_angle >= 2 * math.pi:
            self.orbit_angle -= 2 * math.pi
            
        self.orbit_last_update = current_time
        
        # Only update position if we have target position
        if self.target_position and self.is_in_formation:
            # Calculate new position with updated orbit angle
            self.update_orbit_position()
    
    def update_orbit_position(self):
        """Update position in orbit without advancing step."""
        if self.current_formation != 3 or self.drone_index == 2:
            return  # Only relevant for outer drones in circular orbit
            
        try:
            # Get the base position (where the center drone is)
            angle_rad = math.radians(self.global_angle)
            base_x = self.stage_center[0] + self.radius * math.cos(angle_rad)
            base_y = self.stage_center[1] + self.radius * math.sin(angle_rad)
            
            # Get formation offsets
            offsets = formation_circular_orbit_5()
            
            # Basic offset position for this drone
            my_offset = offsets[self.drone_index]
            
            # Apply continuous orbit rotation to the offset
            rotated_offset = rotate_offset(my_offset, self.orbit_angle)
            
            # Apply global rotation
            final_offset = rotate_offset(rotated_offset, angle_rad)
            
            # Calculate final target position
            target_x = base_x + final_offset[0]
            target_y = base_y + final_offset[1]
            
            # Update target position for the continuous orbit
            new_position = [target_x, target_y, self.altitude]
            
            # Check if position has changed significantly
            if self.target_position:
                current_x, current_y, current_z = self.target_position
                dx = target_x - current_x
                dy = target_y - current_y
                distance = math.sqrt(dx*dx + dy*dy)
                
                # Only update if position changed enough
                if distance > 0.01:  # 1cm threshold
                    self.target_position = new_position
                    
                    # Move to new position with higher speed for smooth motion
                    self.drone.go_to(
                        self.target_position[0], 
                        self.target_position[1], 
                        self.target_position[2], 
                        MOVE_SPEED * 1.5  # Higher speed for smooth orbit
                    )
                    
                    if self.verbose:
                        orbit_deg = math.degrees(self.orbit_angle)
                        self.get_logger().debug(
                            f"Orbit update: Angle={orbit_deg:.1f}°, "
                            f"Target: [{target_x:.2f}, {target_y:.2f}, {self.altitude:.2f}]"
                        )
        
        except Exception as e:
            self.get_logger().error(f"Error updating orbit position: {e}")
    
    def calculate_target_position(self):
        """Calculate the target position based on current formation and step."""
        try:
            # Determine current formation index and step within formation
            formation_idx = self.current_formation
            step_in_formation = self.formation_step
            
            # Determine angle step size based on formation
            if formation_idx == 3:  # Circular orbit (index 3) uses smaller steps
                angle_step_deg = CIRCULAR_ORBIT_ANGLE_DEG
                
                # Reset orbit timer when entering or advancing in circular orbit
                if self.orbit_start_time is None:
                    self.orbit_start_time = time.time()
                    self.orbit_last_update = self.orbit_start_time
                    self.orbit_angle = 0.0
            else:
                angle_step_deg = ANGLE_DEG
                # Reset orbit tracking when not in circular orbit
                self.orbit_start_time = None
            
            # Calculate current angle in radians - use global angle instead of just the formation step
            angle_rad = math.radians(self.global_angle)
            
            # Calculate base position on the circle
            base_x = self.stage_center[0] + self.radius * math.cos(angle_rad)
            base_y = self.stage_center[1] + self.radius * math.sin(angle_rad)
            
            # Get formation function and offsets
            formation_func = self.formation_functions[formation_idx]
            offsets = formation_func()
            
            # Get this drone's offset
            my_offset = offsets[self.drone_index]
            
            # For circular orbit, apply additional rotation to create clockwise orbit motion
            # Only for outer drones (not for the center drone which is drone2 with index 2)
            if formation_idx == 3 and self.drone_index != 2:  # Circular orbit but not center drone
                # For regular step transitions, use the current orbit angle for continuity
                if self.orbit_start_time is not None:
                    my_offset = rotate_offset(my_offset, self.orbit_angle)
                else:
                    # Fallback if orbit not initialized (shouldn't happen)
                    orbit_angle = -1.0 * math.radians(step_in_formation * CIRCULAR_ORBIT_ANGLE_DEG * 2)
                    my_offset = rotate_offset(my_offset, orbit_angle)
            
            # Rotate offset by current angle
            rotated_offset = rotate_offset(my_offset, angle_rad)
            
            # Calculate final target position
            target_x = base_x + rotated_offset[0]
            target_y = base_y + rotated_offset[1]
            
            # Update target position
            self.target_position = [target_x, target_y, self.altitude]
            
            if self.verbose:
                formation_name = self.formation_names[formation_idx]
                self.get_logger().debug(
                    f"Formation: {formation_name}, Step: {step_in_formation}, "
                    f"Angle: {self.global_angle:.1f}°, "
                    f"Target: [{target_x:.2f}, {target_y:.2f}, {self.altitude:.2f}]"
                )
                
            return self.target_position
            
        except Exception as e:
            self.get_logger().error(f"Error calculating target position: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return None
    
    def advance_formation_step(self):
        """Advance to the next step in the formation sequence."""
        # Don't advance if not all drones have reached target
        if not self.check_all_drones_reached_targets():
            self.get_logger().warn("Cannot advance: not all drones have reached target positions")
            return False

        # Increment step counter
        self.current_step += 1
        
        # Determine current formation index and step within formation
        if self.current_formation == 3:  # Circular orbit (index 3) uses more steps
            total_steps = CIRCULAR_ORBIT_STEPS
            angle_step_deg = CIRCULAR_ORBIT_ANGLE_DEG
            
            # When in circular orbit and advancing a step, preserve the orbit angle
            # for continuous motion between steps
            if self.orbit_start_time is not None:
                # The orbit update will continue from current angle
                pass
        else:
            total_steps = STEPS_PER_FORMATION
            angle_step_deg = ANGLE_DEG
            
            # Reset orbit data when leaving circular orbit
            self.orbit_start_time = None
        
        # Calculate formation step
        self.formation_step = self.current_step % total_steps
        
        # Increment global angle - continue around the circle
        self.global_angle += angle_step_deg
        
        # Normalize global angle to keep it within 0-360 degree range
        if self.global_angle >= 360.0:
            self.global_angle -= 360.0
        
        # Check if we've completed the current formation
        if self.formation_step == 0:
            # Increment completed formations counter
            self.completed_formations += 1
            
            # Move to next formation
            self.current_formation = (self.current_formation + 1) % TOTAL_FORMATIONS
            
            self.get_logger().info(f"Advancing to formation: {self.formation_names[self.current_formation]} at angle {self.global_angle:.1f}°")
            self.get_logger().info(f"Completed formations: {self.completed_formations}")
            
            # Check if we've completed all 6 formations
            if self.completed_formations >= TOTAL_FORMATIONS:
                self.get_logger().info("Completed all formations! Preparing to return home.")
                self.return_to_home()
                return True
            
            # Change LED color based on formation
            self.change_led_color(self.get_formation_color())
            
        # Calculate new target position
        self.calculate_target_position()
        
        # Reset formation status for all drones since we're starting a new step
        for drone_id in self.drone_formation_status:
            self.drone_formation_status[drone_id] = False
        
        # Mark self as not in formation
        self.is_in_formation = False
        self.is_moving = True
        self.last_move_time = time.time()
        
        # Move to new position
        self.move_to_target()
        return True
    
    def move_to_target(self):
        """Move to the calculated target position."""
        if self.target_position is None:
            self.get_logger().warn("Target position is None, cannot move")
            return
            
        self.get_logger().info(f"Moving to target position: {self.target_position}")
        
        # Reset formation status when starting to move
        self.is_in_formation = False
        self.drone_formation_status[self.drone_id] = False
        self.is_moving = True
        self.last_move_time = time.time()
        
        self.drone.go_to(
            self.target_position[0], 
            self.target_position[1], 
            self.target_position[2], 
            MOVE_SPEED
        )
    
    def get_formation_color(self):
        """Get LED color for current formation."""
        colors = [
            (1.0, 0.0, 0.0),  # Red - Line
            (0.0, 1.0, 0.0),  # Green - V
            (0.0, 0.0, 1.0),  # Blue - Diamond
            (1.0, 1.0, 0.0),  # Yellow - Circular Orbit
            (1.0, 0.0, 1.0),  # Magenta - Grid
            (0.0, 1.0, 1.0)   # Cyan - Staggered
        ]
        return colors[self.current_formation % len(colors)]
    
    def report_position(self):
        """Report this drone's position and status"""
        if not self.running:
            return
            
        pos = self.drone_positions.get(self.drone_id, [0.0, 0.0, 0.0])
        target = self.target_position if self.target_position else [0.0, 0.0, 0.0]
        distance = compute_euclidean_distance(pos, target)
        status = "FORMATION" if self.is_in_formation else ("MOVING" if self.is_moving else "IDLE")
        
        # Count how many drones have reached their targets
        drones_in_formation = sum(1 for status in self.drone_formation_status.values() if status)
        
        formation_name = self.formation_names[self.current_formation]
        step_info = (f"Step {self.current_step} (Formation {self.current_formation}: {formation_name}, "
                    f"Step {self.formation_step}, Angle {self.global_angle:.1f}°)")
        
        # For circular orbit, show additional info about orbit rotation
        if self.current_formation == 3:  # Circular orbit
            # Only show rotation info for outer drones
            if self.drone_index != 2:  # Not center drone
                orbit_rotation = math.degrees(self.orbit_angle) if self.orbit_start_time is not None else 0.0
                
                self.get_logger().info(
                    f"STATUS: {step_info} | Position [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] | "
                    f"Target [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}] | "
                    f"Distance: {distance:.2f}m | Status: {status} | Drones in formation: {drones_in_formation}/5 | "
                    f"Orbit angle: {orbit_rotation:.1f}° | Completed formations: {self.completed_formations}"
                )
            else:  # Center drone
                self.get_logger().info(
                    f"STATUS: {step_info} | Position [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] | "
                    f"Target [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}] | "
                    f"Distance: {distance:.2f}m | Status: {status} | Drones in formation: {drones_in_formation}/5 | "
                    f"Center drone (no rotation) | Completed formations: {self.completed_formations}"
                )
        else:
            self.get_logger().info(
                f"STATUS: {step_info} | Position [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] | "
                f"Target [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}] | "
                f"Distance: {distance:.2f}m | Status: {status} | Drones in formation: {drones_in_formation}/5 | "
                f"Completed formations: {self.completed_formations}"
            )
        
    def publish_status(self):
        """Publish this drone's status to other drones"""
        # Publish ready status
        ready_msg = Bool()
        ready_msg.data = self.drone_ready_status[self.drone_id]
        self.status_pub.publish(ready_msg)
        
        # Publish formation status
        formation_msg = Bool()
        formation_msg.data = self.is_in_formation
        self.formation_pub.publish(formation_msg)
        
        # Publish home status when returning home
        if self.returning_home:
            home_msg = Bool()
            home_msg.data = self.drone_home_status[self.drone_id]
            self.home_status_pub.publish(home_msg)
        
        # Publish step status
        step_msg = Float32MultiArray()
        step_msg.data = [float(self.current_step), float(self.formation_step), float(self.current_formation)]
        self.step_pub.publish(step_msg)
    
    def ready_status_callback(self, msg, source):
        """Callback for receiving ready status from other drones"""
        self.drone_ready_status[source] = msg.data
        self.discovered_drones[source] = True
        
        if self.verbose:
            self.get_logger().debug(f"Drone {source} ready status: {msg.data}")
    
    def formation_status_callback(self, msg, source):
        """Callback for receiving formation status from other drones"""
        self.drone_formation_status[source] = msg.data
        self.discovered_drones[source] = True
        
        # If all drones have reached targets, check if we should advance
        if (msg.data and self.is_in_formation and self.check_all_drones_reached_targets() and
            self.current_step % (STEPS_PER_FORMATION if self.current_formation != 3 else CIRCULAR_ORBIT_STEPS) == self.formation_step):
            # Log that all drones are in position
            self.get_logger().info("All drones in position, ready to advance")
            
            # Only the drone with the lowest ID advances to avoid multiple advances
            drones_in_formation = [d_id for d_id in self.expected_drone_ids 
                                if self.drone_formation_status.get(d_id, False)]
            
            # Only proceed if there are drones in formation
            if drones_in_formation:
                lowest_id_ready = min(drones_in_formation)
                
                if self.drone_id == lowest_id_ready:
                    self.get_logger().info("I am the lowest ID drone in formation - advancing step")
                    self.advance_formation_step()
        
        if self.verbose:
            self.get_logger().debug(f"Drone {source} formation status: {msg.data}")
    
    def step_status_callback(self, msg, source):
        """Callback for receiving step status from other drones"""
        current_step = int(msg.data[0])
        formation_step = int(msg.data[1])
        formation_idx = int(msg.data[2])
        
        # Store the step
        self.drone_steps[source] = current_step
        
        # If other drone is ahead of us, sync up
        if current_step > self.current_step and self.running:
            self.get_logger().info(f"Syncing with {source} at step {current_step}")
            
            # Store old values
            old_step = self.current_step
            old_formation = self.current_formation
            
            # Update step values
            self.current_step = current_step
            self.current_formation = formation_idx
            self.formation_step = formation_step
            
            # Calculate global angle based on current step
            if self.current_formation == 3:  # Circular orbit
                self.global_angle = formation_step * CIRCULAR_ORBIT_ANGLE_DEG
                
                # Reset orbit timer when syncing during circular orbit
                self.orbit_start_time = time.time()
                self.orbit_last_update = self.orbit_start_time
                self.orbit_angle = 0.0  # Start from beginning of orbit
                
            else:
                self.global_angle = formation_step * ANGLE_DEG
                # Reset orbit data when not in circular orbit
                self.orbit_start_time = None
            
            # Add angle for completed formations
            formations_completed = self.current_step // (STEPS_PER_FORMATION if self.current_formation != 3 else CIRCULAR_ORBIT_STEPS)
            
            # Update completed formations counter
            if self.current_formation > old_formation:
                self.completed_formations = formation_idx
            elif self.current_formation < old_formation:
                # We've wrapped around to the beginning of the formations list
                self.completed_formations = TOTAL_FORMATIONS
                
            for i in range(formations_completed):
                form_type = i % TOTAL_FORMATIONS
                if form_type == 3:  # Circular orbit
                    self.global_angle += CIRCULAR_ORBIT_STEPS * CIRCULAR_ORBIT_ANGLE_DEG
                else:
                    self.global_angle += STEPS_PER_FORMATION * ANGLE_DEG
            
            # Normalize global angle
            while self.global_angle >= 360.0:
                self.global_angle -= 360.0
            
            # Reset formation status for all drones when syncing to a new step
            for drone_id in self.drone_formation_status:
                self.drone_formation_status[drone_id] = False
            
            # Mark self as not in formation
            self.is_in_formation = False
            
            # Update target position and move
            self.calculate_target_position()
            
            # Only move if the formation or step has actually changed
            if old_step != self.current_step or old_formation != self.current_formation:
                self.move_to_target()
                
            # Check if we need to land (after completing 6 formations)
            if self.completed_formations >= TOTAL_FORMATIONS and not self.returning_home:
                self.get_logger().info("Detected that all formations are complete, preparing to return home")
                self.return_to_home()
    
    def has_reached_target(self, current, target, tolerance=POSITION_TOLERANCE):
        """Check if drone has reached target position within tolerance."""
        if current is None or target is None:
            return False
        
        distance = compute_euclidean_distance(current, target)
        result = distance < tolerance
        
        if result and self.verbose:
            self.get_logger().info(f"Reached target: {distance:.2f}m < {tolerance:.2f}m")
        
        return result
    
    def position_callback(self, msg, source):
        """Process position updates from any drone."""
        try:
            # Extract position from PoseStamped message
            pos = msg.pose.position
            position = [pos.x, pos.y, pos.z]
            
            # Update drone position
            self.drone_positions[source] = position
            self.position_timestamps[source] = time.time()
            self.discovered_drones[source] = True
            
            # For our own drone, calculate path length and formation deviation
            if source == self.drone_id:
                # Calculate path length increment
                if self.last_positions[source] is not None:
                    distance = compute_euclidean_distance(position, self.last_positions[source])
                    self.all_drone_path_lengths[source] += distance
                self.last_positions[source] = position.copy()
                
                # Calculate formation deviation if we have a target position
                if self.target_position is not None and self.running and not self.returning_home:
                    # Get current formation and angle
                    formation_idx = self.current_formation
                    angle_rad = math.radians(self.global_angle)
                    
                    # Get base position on circle
                    base_x = self.stage_center[0] + self.radius * math.cos(angle_rad)
                    base_y = self.stage_center[1] + self.radius * math.sin(angle_rad)
                    
                    # Get formation function and offsets
                    formation_func = self.formation_functions[formation_idx]
                    offsets = formation_func()
                    
                    # Get this drone's offset
                    my_offset = offsets[self.drone_index]
                    
                    # For circular orbit, apply additional rotation
                    if formation_idx == 3 and self.drone_index != 2 and self.orbit_start_time is not None:
                        my_offset = rotate_offset(my_offset, self.orbit_angle)
                    
                    # Rotate offset by current angle
                    rotated_offset = rotate_offset(my_offset, angle_rad)
                    
                    # Calculate ideal position
                    ideal_x = base_x + rotated_offset[0]
                    ideal_y = base_y + rotated_offset[1]
                    ideal_z = self.altitude
                    ideal_position = [ideal_x, ideal_y, ideal_z]
                    
                    # Store ideal position
                    self.ideal_positions[self.current_step] = ideal_position
                    
                    # Calculate deviation
                    deviation = compute_euclidean_distance(position, ideal_position)
                    
                    # Update max deviation if larger
                    if deviation > self.max_formation_deviation:
                        self.max_formation_deviation = deviation
                        if self.verbose:
                            self.get_logger().debug(
                                f"New max deviation: {self.max_formation_deviation:.4f}m at step {self.current_step}, "
                                f"Formation: {self.formation_names[formation_idx]}, Angle: {self.global_angle:.1f}°"
                            )
            
            # Check if we've reached our target position (only for our own drone)
            if source == self.drone_id and self.target_position is not None and self.running:
                if self.has_reached_target(position, self.target_position):
                    if not self.is_in_formation:
                        self.get_logger().info(f"Reached target position: {self.target_position}")
                        self.is_in_formation = True
                        self.is_moving = False
                        
                        # If we're returning home, update home status
                        if self.returning_home:
                            self.drone_home_status[self.drone_id] = True
                            self.get_logger().info("Reached home position")
                            
                            # Land immediately when we reach home
                            self.land_now()
                        else:
                            self.drone_formation_status[self.drone_id] = True
                            
                        self.change_led_color(self.get_formation_color())
                        
                        # Log status of all drones
                        if self.returning_home:
                            all_at_home = self.check_all_drones_at_home()
                            if all_at_home:
                                self.get_logger().info("All drones have reached their home positions")
                        else:
                            all_reached = self.check_all_drones_reached_targets()
                            if all_reached:
                                self.get_logger().info("All drones have reached their targets - ready for next step")
                            
                    # Check if all drones have reached their targets
                    if not self.returning_home and self.check_all_drones_reached_targets():
                        # Move to next formation step
                        self.advance_formation_step()
                elif self.is_moving and (time.time() - self.last_move_time) > self.position_reached_timeout:
                    # If we've been trying to reach this position for too long, just mark it as reached
                    # This prevents drones from getting stuck when facing control or physics issues
                    self.get_logger().warn(
                        f"Position timeout - marking as reached. Current: {position}, Target: {self.target_position}, "
                        f"Distance: {compute_euclidean_distance(position, self.target_position):.2f}m"
                    )
                    self.is_in_formation = True
                    self.is_moving = False
                    
                    # If we're returning home, update home status
                    if self.returning_home:
                        self.drone_home_status[self.drone_id] = True
                        self.get_logger().info("Reached home position (timeout)")
                        
                        # Land immediately when we reach home
                        self.land_now()
                    else:
                        self.drone_formation_status[self.drone_id] = True
            
            if self.verbose and source != self.drone_id:
                self.get_logger().debug(f"Updated {source} position: {position}")
                
        except Exception as e:
            self.get_logger().error(f"Error in position callback: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def check_all_drones_reached_targets(self):
        """Check if all drones have reached their formation targets."""
        result = all(self.drone_formation_status.values())
        if self.verbose and result:
            self.get_logger().debug("All drones have reached their formation targets")
        return result
    
    def all_drones_discovered(self):
        """Check if all expected drones have been discovered through communications."""
        return all(self.discovered_drones.values())
    
    def check_all_drones_ready(self):
        """Check if all drones are ready to start the mission."""
        if self.all_drones_ready:
            return True
            
        all_ready = all(self.drone_ready_status.values())
        
        if all_ready and not self.all_drones_ready:
            self.all_drones_ready = True
            self.get_logger().info("All drones ready to start mission.")
        
        return all_ready
    
    def check_all_drones_in_formation(self):
        """Check if all drones have reached formation."""
        return all(self.drone_formation_status.values())
    
    def mission_step(self):
        """Execute one step of the mission logic."""
        try:
            # Measure CPU time
            cpu_start = time.process_time()
            
            # State machine for the drone
            if not self.running:
                # Check if all drones are discovered and ready
                if self.check_all_drones_ready() and self.all_drones_discovered():
                    self.get_logger().info("Starting mission!")
                    self.start_mission()
                    
            # Record CPU time for this loop iteration
            self.cpu_samples.append(time.process_time() - cpu_start)
                    
            return
        
        except Exception as e:
            self.get_logger().error(f"Error in mission step: {e}")
            # Record CPU time even if there's an error
            self.cpu_samples.append(time.process_time() - cpu_start)
    
    def start_mission(self):
        """Start the mission by taking off and moving to formation position."""
        self.get_logger().info("Starting mission")
        
        # Start running the mission
        self.running = True
        
        # Take off and move to formation
        self.takeoff_and_move_to_formation()
    
    def takeoff_and_move_to_formation(self):
        """Take off and move to formation position."""
        self.get_logger().info(f"Taking off to {self.altitude}m...")
        
        # Record mission start time
        self.mission_start_time = time.time()
        self.get_logger().info(f"Recording mission start time: {self.mission_start_time}")
        
        # Set LED to yellow for takeoff
        self.change_led_color((1.0, 1.0, 0.0))
        
        # Record initial position
        initial_position = self.drone_positions.get(self.drone_id, [0.0, 0.0, 0.0])
        self.get_logger().info(f"Initial position: {initial_position}")
        self.last_positions[self.drone_id] = initial_position.copy()  # Store for path length calculation
        
        # Offboard mode and arm
        self.drone.offboard()
        self.drone.arm()
        
        # Take off
        self.get_logger().info(f"Initiating takeoff to {self.altitude}m with speed {TAKE_OFF_SPEED}")
        self.drone.takeoff(self.altitude, TAKE_OFF_SPEED)
        
        # Wait briefly for takeoff to start
        time.sleep(3.0)
        
        # Set LED to color for current formation
        self.change_led_color(self.get_formation_color())
        
        # Move to first formation position
        self.move_to_target()
    
    def return_to_home(self):
        """Return to home position before landing."""
        if self.returning_home:
            return  # Already returning home
            
        self.returning_home = True
        self.get_logger().info("Returning to home position before landing")
        
        # Get this drone's home position
        home_position = self.home_positions.get(self.drone_id)
        if not home_position:
            self.get_logger().error(f"No home position found for {self.drone_id}, using current position")
            current_pos = self.drone_positions.get(self.drone_id, [0.0, 0.0, 0.0])
            home_position = (current_pos[0], current_pos[1], 0.2)  # Use current XY but low Z
        
        # Set LED to blue for returning home
        self.change_led_color((0.0, 0.0, 1.0))
        
        # Reset home status for all drones
        for drone_id in self.drone_home_status:
            self.drone_home_status[drone_id] = False
        
        # Move to home position with altitude
        home_x, home_y, home_z = home_position
        target_position = [home_x, home_y, self.altitude]
        
        self.get_logger().info(f"Moving to home position: {target_position}")
        
        # Update target and move
        self.target_position = target_position
        self.is_in_formation = False
        self.is_moving = True
        self.last_move_time = time.time()
        self.move_to_target()
    
    def execute_landing(self):
        """Execute landing sequence after all drones are at home position."""
        if self.landing_initiated:
            return  # Already in landing process
            
        self.get_logger().info("All drones at home position, initiating synchronized landing")
        
        # Mark landing as initiated to prevent multiple landing attempts
        self.landing_initiated = True
        
        # Use direct landing method rather than publishing a command
        self.land_all_drones_simultaneously()

    def land_all_drones_simultaneously(self):
        """
        Land all drones simultaneously using daemon threads.
        This is a more robust approach than using timers.
        """
        self.get_logger().info("Executing synchronized landing for all drones")
        
        # Change LED to red for landing
        self.change_led_color((1.0, 0.0, 0.0))
        
        # Directly land this drone
        self.get_logger().info("Executing landing now!")
        self.drone.land()
        
        # Stop mission
        self.running = False
        self.mission_complete = True
        self.landing_initiated = True
        
        # Record mission end time and mark as successful
        if self.mission_start_time is not None and self.mission_end_time is None:
            self.mission_end_time = time.time()
            self.mission_success = True
            
            # Print metrics
            self.print_performance_metrics()

    def publish_landing_command(self):
        """
        Publish landing command with timestamp to all drones.
        This is now a simpler method that just notifies other drones 
        that they should land immediately.
        """
        # Create and publish message - just use current time as the signal
        msg = Float32MultiArray()
        msg.data = [time.time()]
        
        # Publish several times to ensure delivery
        for _ in range(5):
            self.landing_command_pub.publish(msg)
            time.sleep(0.1)
        
        self.get_logger().info("Published synchronized landing command")
        
        # Also land this drone
        self.land_all_drones_simultaneously()

    def landing_command_callback(self, msg):
        """Callback for receiving landing command."""
        if self.landing_initiated:
            return  # Already handling landing
            
        self.get_logger().info("Received synchronized landing command")
        
        # Mark landing as initiated
        self.landing_initiated = True
        
        # Execute landing immediately
        self.land_all_drones_simultaneously()

    def land_and_end(self):
        """Land and end the mission."""
        if not self.running:
            return
            
        if self.returning_home:
            # If we're in the return to home phase, just land after all drones are home
            self.get_logger().info("Returning to home position before landing")
            self.return_to_home()
        else:
            # Direct landing (emergency or when not in a formation)
            self.get_logger().info("Emergency landing...")
            
            # Set LED to red for landing
            self.change_led_color((1.0, 0.0, 0.0))
            
            # Land
            self.drone.land()
            
            # Stop mission
            self.running = False
            self.mission_complete = True
            self.landing_initiated = True
            
            # Set LED to off
            self.change_led_color((0.0, 0.0, 0.0))
            
            # Record mission end time
            if self.mission_start_time is not None and self.mission_end_time is None:
                self.mission_end_time = time.time()
                self.mission_success = False  # Mark as failure for emergency landing
                
                # Print metrics
                self.print_performance_metrics()
    
    def change_led_color(self, color):
        """Change LED color."""
        color_msg = ColorRGBA()
        color_msg.r = float(color[0])
        color_msg.g = float(color[1])
        color_msg.b = float(color[2])
        color_msg.a = 1.0
        self.led_pub.publish(color_msg)
    
    def home_status_callback(self, msg, source):
        """Callback for receiving home status from other drones"""
        self.drone_home_status[source] = msg.data
        self.discovered_drones[source] = True
        
        if self.verbose:
            self.get_logger().debug(f"Drone {source} home status: {msg.data}")
        
        # If already in landing process, don't take action
        if self.landing_initiated:
            return
        
        # If this is our own status and we've reached home, land immediately
        if source == self.drone_id and msg.data == True and self.returning_home:
            self.get_logger().info("Reached home position, landing now")
            self.land_now()
    
    def check_all_drones_at_home(self):
        """Check if all drones have reached their home positions."""
        result = all(self.drone_home_status.values())
        if result and self.verbose:
            self.get_logger().debug("All drones have reached their home positions")
        return result

    def land_now(self):
        """Land this drone immediately without waiting for others"""
        if self.landing_initiated:
            return
        
        self.landing_initiated = True
        self.get_logger().info("Landing now")
        
        # Change LED to red for landing
        self.change_led_color((1.0, 0.0, 0.0))
        
        # Land
        self.drone.land()
        
        # Stop mission
        self.running = False
        self.mission_complete = True
        
        # Record mission end time and mark as successful
        self.mission_end_time = time.time()
        self.mission_success = True
        
        # Print performance metrics
        self.print_performance_metrics()

    def print_performance_metrics(self):
        """Print all performance metrics at the end of the mission."""
        if self.mission_start_time is None or self.mission_end_time is None:
            self.get_logger().error("Cannot calculate metrics: mission timing not recorded properly")
            return
            
        try:
            # Calculate mission time
            mission_time = self.mission_end_time - self.mission_start_time
            
            # Calculate CPU metrics
            if self.cpu_samples:
                total_cpu_time = sum(self.cpu_samples)
                avg_cpu_time = total_cpu_time / len(self.cpu_samples)
                max_cpu_time = max(self.cpu_samples)
            else:
                total_cpu_time = avg_cpu_time = max_cpu_time = 0
            
            # Print metrics summary
            self.get_logger().info("\n" + "="*50)
            self.get_logger().info("MISSION PERFORMANCE METRICS")
            self.get_logger().info("="*50)
            self.get_logger().info(f"Mission success: {self.mission_success}")
            self.get_logger().info(f"Mission time: {mission_time:.2f} s")
            self.get_logger().info(f"CPU time: total={total_cpu_time:.3f} s, avg_loop={avg_cpu_time:.6f} s, max_loop={max_cpu_time:.6f} s")
            self.get_logger().info(f"Average CPU Load: {avg_cpu_time:.6f}")
            self.get_logger().info(f"Path length ({self.drone_id}): {self.all_drone_path_lengths[self.drone_id]:.2f} m")
            self.get_logger().info(f"Max formation deviation: {self.max_formation_deviation:.4f} m")
            self.get_logger().info("="*50)
            
            print("\n" + "="*50)
            print("MISSION PERFORMANCE METRICS")
            print("="*50)
            print(f"Mission success: {self.mission_success}")
            print(f"Mission time: {mission_time:.2f} s")
            print(f"CPU time: total={total_cpu_time:.3f} s, avg_loop={avg_cpu_time:.6f} s, max_loop={max_cpu_time:.6f} s")
            print(f"Average CPU Load: {avg_cpu_time:.6f}")
            print(f"Path length ({self.drone_id}): {self.all_drone_path_lengths[self.drone_id]:.2f} m")
            print(f"Max formation deviation: {self.max_formation_deviation:.4f} m")
            print("="*50)
            
        except Exception as e:
            self.get_logger().error(f"Error calculating metrics: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

def launch_multi_drone_mission(drone_ids, config_path, use_sim_time=False, verbose=False):
    """
    Launch multiple drone processes from a single command.
    This preserves the decentralized architecture while making it easier to start.
    """
    print(f"Launching multi-drone decentralized mission with drones: {drone_ids}")
    
    # Get script path
    script_path = os.path.abspath(__file__)
    
    # Create a list to store processes
    processes = []
    
    # Launch each drone in a separate process
    for drone_id in drone_ids:
        # Build command
        cmd = [
            sys.executable,  # Current Python interpreter
            script_path,     # This script
            '-n', drone_id,  # Single drone ID
            '--config', config_path
        ]
        
        if use_sim_time:
            cmd.append('--sim-time')
            
        if verbose:
            cmd.append('-v')
        
        # Print the command we're running
        print(f"Starting process for {drone_id}: {' '.join(cmd)}")
        
        # Start the process
        drone_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
        )
        
        processes.append((drone_id, drone_process))
    
    # Create threads to display output from each process
    def output_reader(drone_id, process):
        for line in process.stdout:
            print(f"[{drone_id}] {line.strip()}")
    
    threads = []
    for drone_id, process in processes:
        thread = threading.Thread(
            target=output_reader,
            args=(drone_id, process),
            daemon=True
        )
        thread.start()
        threads.append(thread)
    
    try:
        # Monitor all processes
        print("All drones started. Press Ctrl+C to terminate all drones.")
        while True:
            # Check if any process has exited
            all_running = True
            for drone_id, process in processes:
                if process.poll() is not None:
                    # Process has exited
                    all_running = False
                    print(f"Process for {drone_id} has exited with code {process.returncode}")
            
            if not all_running:
                print("Some drones have exited. Terminating all drones.")
                break
                
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Terminating all drones.")
    finally:
        # Terminate all processes
        for drone_id, process in processes:
            if process.poll() is None:  # If still running
                print(f"Terminating process for {drone_id}")
                process.terminate()
                
        # Wait for all processes to end
        for drone_id, process in processes:
            try:
                process.wait(timeout=5)
                print(f"Process for {drone_id} terminated")
            except subprocess.TimeoutExpired:
                print(f"Process for {drone_id} did not terminate in time. Killing.")
                process.kill()

def main():
    parser = argparse.ArgumentParser(description='Decentralized multi-drone line formation')
    parser.add_argument('-n', '--namespace', type=str, nargs='+', required=True,
                        help='One or more drone IDs (e.g., drone0 or multiple: drone0 drone1 drone2...)')
    parser.add_argument('--config', type=str, 
                        default='/home/ziyar/project_gazebo_ws/src/challenge_multi_drone/scenarios/scenario1_stage1.yaml',
                        help='Path to scenario configuration YAML file')
    parser.add_argument('--sim-time', action='store_true', help='Use simulation time')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    # Check if we have multiple drones
    if len(args.namespace) > 1:
        # Launch multi-drone mission with multiple processes
        launch_multi_drone_mission(
            drone_ids=args.namespace,
            config_path=args.config,
            use_sim_time=args.sim_time,
            verbose=args.verbose
        )
        return 0
    
    # Single drone mode - run just one decentralized drone node
    drone_id = args.namespace[0]
    
    # Initialize ROS
    rclpy.init()
    
    try:
        # Create the drone node
        drone_node = DecentralizedDroneNode(
            drone_id=drone_id,
            config_path=args.config,
            use_sim_time=args.sim_time,
            verbose=args.verbose
        )
        
        # Use MultiThreadedExecutor to allow for concurrent callbacks
        executor = MultiThreadedExecutor()
        executor.add_node(drone_node)
        
        print(f"Drone {drone_id} initialized. Starting execution.")
        
        try:
            # Start the execution
            executor.spin()
        except KeyboardInterrupt:
            print(f"Keyboard interrupt received, landing drone {drone_id}...")
            drone_node.land_and_end()
        finally:
            # Cleanup
            executor.shutdown()
            drone_node.destroy_node()
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Shutdown ROS
        rclpy.shutdown()
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 