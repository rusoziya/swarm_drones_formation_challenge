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
import traceback
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from as2_msgs.msg import YawMode, BehaviorStatus
from as2_python_api.drone_interface import DroneInterface
from as2_python_api.behavior_actions.behavior_handler import BehaviorHandler
from std_msgs.msg import ColorRGBA, Float32MultiArray, String
from geometry_msgs.msg import PoseStamped
from tf2_ros import StaticTransformBroadcaster
from types import SimpleNamespace
import subprocess

# --- Formation functions for 5 drones ---
def formation_line_5():
    """
    Return offsets for a line formation for 5 drones.
    Evenly spaced along the x-axis with drone2 (index 2) as the center.
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
    One drone in the center of the diamond, with others at the diamond points.
    """
    d = 0.5  # Spacing between drones
    
    # Diamond layout with center
    return [
        (0.0, d),       # drone0 at front point
        (-d, 0.0),      # drone1 at left point
        (0.0, 0.0),     # drone2 at center
        (d, 0.0),       # drone3 at right point
        (0.0, -d)       # drone4 at back point
    ]

def formation_circular_orbit_5():
    """
    Return offsets for a circular orbit formation for 5 drones.
    Creates a perfect circle with drones orbiting around the center in 
    an anticlockwise direction, maintaining a clear circular shape at all times.
    """
    # Circle radius - increased for better visibility
    circle_radius = 1.0  # Increased for more distinct circular pattern
    
    # Number of drones arranged in a circle (5 total, with 1 in center)
    num_positions = 4
    
    # Calculate positions around the circle in an anticlockwise arrangement
    offsets = []
    
    # First two drones (drone0, drone1) - add sequentially around the circle
    for i in range(2):
        # Angles in anticlockwise direction (note: using base positions here)
        angle = i * (2.0 * math.pi / num_positions)
        x = circle_radius * math.cos(angle)
        y = circle_radius * math.sin(angle)
        offsets.append((x, y))
    
    # One drone (drone2) at center
    offsets.append((0.0, 0.0))
    
    # Last two drones (drone3, drone4) - continue around the circle
    for i in range(2, 4):
        angle = i * (2.0 * math.pi / num_positions)
        x = circle_radius * math.cos(angle)
        y = circle_radius * math.sin(angle)
        offsets.append((x, y))
    
    return offsets

def formation_grid_5():
    """
    Return offsets for a grid formation for 5 drones.
    One drone at front, with others arranged in a 2x2 grid behind:
        front
        f1 f2
        f3 f4
    """
    d = 0.6  # Spacing between drones
    
    return [
        (-d, -d),    # drone0 (f1) middle-left
        (d, -d),     # drone1 (f2) middle-right
        (0.0, 0.0),  # drone2 at front
        (-d, -2*d),  # drone3 (f3) back-left
        (d, -2*d)    # drone4 (f4) back-right
    ]

def formation_staggered_5():
    """
    Return offsets for a staggered formation for 5 drones.
    Creates a pattern with one drone in the middle.
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

# --- SwarmDrone class (replaces both Leader and Follower) ---
class SwarmDrone(DroneInterface):
    """
    Decentralized swarm drone implementation where each drone participates in peer-to-peer
    communication. All drones run identical code and collectively determine formation movement.
    """
    def __init__(self, namespace: str, swarm_namespaces: list, drone_index: int, config_path: str, 
                 verbose: bool = False, use_sim_time: bool = False):
        super().__init__(namespace, verbose=verbose, use_sim_time=use_sim_time)
        self._speed = 0.5
        self._yaw_mode = YawMode.PATH_FACING
        self._yaw_angle = None
        self._frame_id = "earth"
        self.current_behavior: BehaviorHandler = None
        self.led_pub = self.create_publisher(ColorRGBA, f"/{namespace}/leds/control", 10)
        
        # Store drone identification
        self.namespace = namespace
        self.drone_index = drone_index  # Index in formation (0-4)
        self.swarm_namespaces = swarm_namespaces
        
        # Load configuration
        self.config = load_scenario_config(config_path)
        if not self.config:
            raise ValueError(f"Failed to load configuration from {config_path}")
            
        # Load drone home positions
        self.drone_home_positions = load_drone_home_positions()
        if not self.drone_home_positions:
            print("WARNING: Could not load drone home positions from file, using hardcoded defaults")
            self.drone_home_positions = {
                "drone0": (-1.0, 0.0, 0.2),
                "drone1": (1.0, 0.0, 0.2),
                "drone2": (0.0, 0.0, 0.2),
                "drone3": (0.0, 1.0, 0.2),
                "drone4": (0.0, -1.0, 0.2)
            }
        
        # Set this drone's home position
        if namespace in self.drone_home_positions:
            self.home_position = self.drone_home_positions[namespace]
            print(f"[{namespace}] Using home position: {self.home_position}")
        else:
            print(f"[{namespace}] WARNING: Namespace not found in home positions, using default (0,0,0)")
            self.home_position = (0.0, 0.0, 0.2)
            
        # Stage parameters from config
        self.stage_center = self.config['stage1']['stage_center']
        self.diameter = self.config['stage1']['trajectory']['diameter']
        self.radius = self.diameter / 2.0
        self.altitude = 2.0  # Default flight altitude
        
        # Formation parameters
        self.total_formations = 6
        self.formation_repeats = 1
        self.degrees_per_formation = 180
        self.angle_deg = 30  # Decreased angle step for smoother movement
        self.angle_step = math.radians(self.angle_deg)
        self.steps_per_formation = int(self.degrees_per_formation / self.angle_deg)
        
        # Special steps for circular orbit (18 steps per 180 degrees)
        self.circular_orbit_steps = 18
        self.circular_orbit_angle_deg = self.degrees_per_formation / self.circular_orbit_steps
        self.circular_orbit_angle_step = math.radians(self.circular_orbit_angle_deg)
        
        # Movement speed parameters
        self.movement_speed = 0.7  # Slightly increased for more continuous movement
        self.stabilization_time = 0.5  # Reduced stabilization time significantly
        self.formation_switch_pause = 1.0  # Reduced pause between formations
        
        # Ensure we have exactly one full revolution (360 degrees)
        self.total_steps = self.total_formations * self.formation_repeats * self.steps_per_formation
        
        # State variables - each drone tracks its own state and all peers
        self.current_formation = 0
        self.formation_repeat = 0
        self.step_in_formation = 0
        self.angle = 0.0
        self.current_position = (0.0, 0.0, self.altitude)
        self.running = False
        self.mission_complete = False
        self.returning_home = False
        self.at_landing_position = False
        
        # Formation definitions
        self.formations = [
            formation_line_5,
            formation_v_5,
            formation_diamond_5,
            formation_circular_orbit_5,
            formation_grid_5,
            formation_staggered_5
        ]
        
        # Formation names for debugging
        self.formation_names = [
            "Line",
            "V",
            "Diamond",
            "Circular Orbit",
            "Grid",
            "Staggered"
        ]
        
        # For circular orbit rotation tracking
        self.current_rotation_angle = 0.0
        self.last_rotation_update = time.time()
        self.rotation_speed = 1.5  # radians per second
        
        # Initialize ROS publishers and subscribers
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        
        # State publishing and subscription (peer-to-peer communication)
        self.state_pub = self.create_publisher(
            Float32MultiArray, '/formation/swarm_state', 10)
        self.state_sub = self.create_subscription(
            Float32MultiArray, 
            '/formation/swarm_state', 
            self.state_callback, 10)
        
        # Command publishing and subscription
        self.command_publisher = self.create_publisher(
            String, '/formation/command', 10)
            
        self.command_subscription = self.create_subscription(
            String,
            '/formation/command',
            self.command_callback,
            10
        )
        
        # Landing ready confirmation
        self.landing_ready_pub = self.create_publisher(
            String, '/formation/landing_ready', 10)
            
        self.landing_ready_sub = self.create_subscription(
            String,
            '/formation/landing_ready',
            self.landing_ready_callback,
            10
        )
    
        # Tracking peer states - store latest state from each drone
        self.peer_states = {}
        self.landing_ready_drones = set()
        
        # Main execution threads
        self.position_update_thread = None
        self.trajectory_thread = None
        
        # Initialize additional state variables needed for position updates
        self.should_position_update = False
        self.takeoff_complete = False
        self.mission_started = False
        self.current_target_index = 0
        self.swarm_positions = {}
        
        # Create a simple trajectory (for coordinator drone)
        # This is a circular path around the stage center
        self.trajectory = []
        angle_step = math.radians(10)  # 10 degree steps
        for angle in np.arange(0, 2 * math.pi, angle_step):
            x = self.stage_center[0] + self.radius * math.cos(angle)
            y = self.stage_center[1] + self.radius * math.sin(angle)
            z = self.altitude
            self.trajectory.append((x, y, z))
            
        # Set in_air state
        self.in_air = False
        self.takeoff_altitude = 2.0  # Default takeoff altitude
        
        # Initialize traceback for exception handling
        self.traceback = traceback

    def start(self):
        """Start the drone's operation by initializing position updates."""
        print(f"[{self.namespace}] Starting drone operations")
        
        # Initialize the position update thread if not already running
        if not self.position_update_thread or not self.position_update_thread.is_alive():
            self.should_position_update = True
            self.position_update_thread = threading.Thread(target=self.update_position)
            self.position_update_thread.daemon = True
            self.position_update_thread.start()
            print(f"[{self.namespace}] Position update thread started")
            
        # Initialize trajectory execution if this is the coordinator
        if self.drone_index == 0 and (not self.trajectory_thread or not self.trajectory_thread.is_alive()):
            self.trajectory_thread = threading.Thread(target=self.execute_trajectory)
            self.trajectory_thread.daemon = True
            # Don't start trajectory yet - wait for all drones to be ready
            print(f"[{self.namespace}] Trajectory thread initialized (will start when all drones ready)")
    
    def stop(self):
        """Stop the drone's operations and clean up resources."""
        print(f"[{self.namespace}] Stopping drone operations")
        
        # Signal threads to stop
        self.should_position_update = False
        self.running = False
        
        # Wait for threads to finish
        if self.position_update_thread and self.position_update_thread.is_alive():
            try:
                # Don't wait too long - set a timeout
                self.position_update_thread.join(timeout=2.0)
                print(f"[{self.namespace}] Position update thread stopped")
            except Exception as e:
                print(f"[{self.namespace}] Error stopping position thread: {e}")
                
        if self.trajectory_thread and self.trajectory_thread.is_alive():
            try:
                # Don't wait too long - set a timeout
                self.trajectory_thread.join(timeout=2.0)
                print(f"[{self.namespace}] Trajectory thread stopped")
            except Exception as e:
                print(f"[{self.namespace}] Error stopping trajectory thread: {e}")
        
        # If we're in the air, try to return to home
        try:
            # Check if drone is armed using proper method
            is_armed = False
            try:
                # Access arm.status().is_armed property for correct status check
                is_armed = self.arm.status().is_armed
            except Exception as e:
                print(f"[{self.namespace}] Could not check armed status: {e}")
                
            if is_armed and not self.returning_home:
                print(f"[{self.namespace}] Emergency return to home during shutdown")
                self.returning_home = True
                try:
                    self.return_to_home()
                except Exception as e:
                    print(f"[{self.namespace}] Error during emergency return: {e}")
                    # Try to disarm directly as a last resort
                    try:
                        self.disarm()
                        print(f"[{self.namespace}] Emergency disarm completed")
                    except Exception as disarm_error:
                        print(f"[{self.namespace}] Failed emergency disarm: {disarm_error}")
        except Exception as e:
            print(f"[{self.namespace}] Error during stop sequence: {e}")

    def change_led_colour(self, colour):
        """Change the LED color of the drone."""
        msg = ColorRGBA()
        msg.r = colour[0] / 255.0
        msg.g = colour[1] / 255.0
        msg.b = colour[2] / 255.0
        self.led_pub.publish(msg)

    def change_leds_random_colour(self):
        """Set drone LEDs to a random color."""
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
        # Update current position
        self.current_position = (x, y, z)
        
    def set_position(self, x, y, z, speed=1.0) -> None:
        """Set the drone's target position, a simplified version of go_to_position."""
        # Just redirect to go_to_position for now
        self.go_to_position(x, y, z, speed)

    def goal_reached(self) -> bool:
        """Check if the current behavior has finished (IDLE)."""
        if not self.current_behavior:
            return False
        return self.current_behavior.status == BehaviorStatus.IDLE
    
    def publish_state(self):
        """Publish drone state for other drones in the swarm."""
        try:
            # Create state information 
            state_msg = Float32MultiArray()
            
            # Package position and status information
            state_msg.data = [
                float(self.current_position[0]),  # X position
                float(self.current_position[1]),  # Y position
                float(self.current_position[2]),  # Z position
                float(self.drone_index),          # Drone index
                float(self.returning_home),       # Return home flag
                float(self.mission_started),      # Mission started flag
                float(self.takeoff_complete)      # Takeoff complete flag
            ]
            
            # Add formation info if coordinator
            if self.drone_index == 0:
                state_msg.data.extend([
                    float(self.current_formation),  # Current formation
                    float(self.current_target_index)  # Current target in trajectory
                ])
            
            # Publish the message
            self.state_pub.publish(state_msg)
            
        except Exception as e:
            print(f"[{self.namespace}] Error publishing state: {e}")
    
    def publish_command(self, command):
        """Publish command to all drones."""
        msg = String()
        msg.data = f"{self.namespace}:{command}"  # Include sender namespace
        self.command_publisher.publish(msg)
    
    def state_callback(self, msg):
        """Process state messages from other drones in the swarm."""
        try:
            # Skip processing if message is empty
            if len(msg.data) == 0:
                return
            
            # Extract common data from all drones
            sender_x = float(msg.data[0])
            sender_y = float(msg.data[1])
            sender_z = float(msg.data[2])
            sender_index = int(msg.data[3])
            sender_returning = bool(msg.data[4])
            sender_mission_started = bool(msg.data[5])
            sender_takeoff_complete = bool(msg.data[6])
            
            # Skip processing our own messages
            if sender_index == self.drone_index:
                return
                
            # Store position of this drone in our swarm positions map
            sender_namespace = f"drone{sender_index}"
            self.swarm_positions[sender_namespace] = (sender_x, sender_y, sender_z)
                
            # Special handling for coordinator messages (drone0)
            if sender_index == 0:
                # Update coordinator state (if we're a follower)
                if self.drone_index > 0:
                    # Create or update coordinator state object
                    if not hasattr(self, 'coordinator_state'):
                        self.coordinator_state = SimpleNamespace()
                    
                    # Store coordinator position
                    self.coordinator_state.position = SimpleNamespace(
                        x=sender_x, 
                        y=sender_y, 
                        z=sender_z
                    )
                    
                    # Store coordinator status flags
                    self.coordinator_state.returning_home = sender_returning
                    self.coordinator_state.mission_started = sender_mission_started
                    self.coordinator_state.takeoff_complete = sender_takeoff_complete
                    
                    # Get formation information if included
                    if len(msg.data) >= 9:
                        self.current_formation = int(msg.data[7])
                        self.coordinator_target_index = int(msg.data[8])
                    
                    # Sync mission state with coordinator
                    if sender_mission_started and not self.mission_started:
                        self.mission_started = True
                        print(f"[{self.namespace}] Mission started by coordinator")
                    
                    # Follow coordinator's return home command
                    if sender_returning and not self.returning_home:
                        self.returning_home = True
                        print(f"[{self.namespace}] Coordinator is returning home, following")
            
        except Exception as e:
            print(f"[{self.namespace}] Error in state_callback: {e}")
    
    def command_callback(self, msg):
        """Process commands from any drone."""
        command_text = msg.data
        
        # Parse sender and command
        if ':' in command_text:
            sender, command = command_text.split(':', 1)
            
            # Skip our own commands
            if sender == self.namespace:
                return
        else:
            # Legacy format without sender prefix
            command = command_text
        
        print(f"[{self.namespace}] Received command: {command}")
        
        if command == "takeoff":
            # Don't try to take off if already in the process
            if hasattr(self, 'current_behavior') and self.current_behavior and not self.goal_reached():
                print(f"[{self.namespace}] Already executing a behavior, ignoring takeoff command")
                return
                
            # Takeoff procedure
            print(f"[{self.namespace}] Taking off to {self.altitude}m")
            self.do_behavior("takeoff", self.altitude, 0.7, False)
            self.change_led_colour((0, 255, 0))
            
            # Start position updates after takeoff
            def start_after_takeoff():
                start_time = time.time()
                max_wait = 20.0  # 20 seconds maximum wait for takeoff
                
                while not self.goal_reached() and time.time() - start_time < max_wait:
                    time.sleep(0.1)
                
                if self.goal_reached():
                    print(f"[{self.namespace}] Takeoff complete, starting position updates")
                else:
                    print(f"[{self.namespace}] Takeoff may not have completed, but starting position updates anyway")
                
                # Ensure we start position updates
                self.start()
            
            threading.Thread(target=start_after_takeoff, daemon=True).start()
            
        elif command == "prepare_landing":
            # First landing phase: Move to landing position but don't land yet
            print(f"[{self.namespace}] Preparing for landing - moving to landing position")
            self.stop()  # Stop following formation
            
            # Move to home position
            self.return_to_home()
            
            # Wait to reach landing position and notify
            landing_position_wait = threading.Thread(target=self._wait_for_landing_position, daemon=True)
            landing_position_wait.start()
            
        elif command == "execute_landing":
            # Second landing phase: Execute actual landing
            print(f"[{self.namespace}] Received command to execute landing")
            
            # Check if we've reached our landing position
            if hasattr(self, 'at_landing_position') and self.at_landing_position:
                print(f"[{self.namespace}] Already at landing position, executing landing...")
                self._execute_landing()
            else:
                print(f"[{self.namespace}] WARNING: Landing position may not be reached yet!")
                # Brief wait to ensure we're at the landing position
                wait_time = 2.0  # seconds
                start_time = time.time()
                while time.time() - start_time < wait_time and not self.goal_reached():
                    time.sleep(0.1)
                
                # Execute landing anyway
                self._execute_landing()
            
    def landing_ready_callback(self, msg):
        """Callback for landing position confirmations."""
        drone_name = msg.data
        print(f"[{self.namespace}] Received landing position confirmation from {drone_name}")
        self.landing_ready_drones.add(drone_name)
    
    def _wait_for_landing_position(self):
        """Wait until the drone reaches the landing position."""
        # Maximum time to wait for position to be reached
        max_wait_time = 15.0  # seconds
        start_time = time.time()
        
        print(f"[{self.namespace}] Waiting to reach landing position...")
        
        while time.time() - start_time < max_wait_time and not self.goal_reached():
            time.sleep(0.1)
        
        if self.goal_reached():
            print(f"[{self.namespace}] Successfully reached landing position")
            self.at_landing_position = True
            # Change LED to indicate ready for landing
            self.change_led_colour((0, 0, 255))  # Blue for "ready to land"
            
            # Publish confirmation that we're at landing position
            self.publish_landing_ready()
        else:
            print(f"[{self.namespace}] WARNING: Could not reach landing position within timeout")
            # Mark as at landing position anyway to avoid getting stuck
            self.at_landing_position = True
            
            # Still publish confirmation to avoid blocking the mission
            self.publish_landing_ready()
    
    def publish_landing_ready(self):
        """Publish confirmation that this drone is at its landing position."""
        msg = String()
        msg.data = self.namespace
        
        # Publish multiple times to reduce chance of message loss
        for _ in range(5):
            self.landing_ready_pub.publish(msg)
            time.sleep(0.1)
            
        print(f"[{self.namespace}] Published landing position confirmation")
    
    def _execute_landing(self):
        """Execute the actual landing procedure."""
        # Execute landing
        print(f"[{self.namespace}] Executing landing at home position...")
        self.do_behavior("land", 0.4, False)
        
        # Change LED to red to indicate landing
        self.change_led_colour((255, 0, 0))
        
        # Monitor landing status
        landing_timeout = 30  # seconds
        start_time = time.time()
        while not self.goal_reached() and time.time() - start_time < landing_timeout:
            time.sleep(0.5)
            
        if self.goal_reached():
            print(f"[{self.namespace}] Landing completed successfully at home position")
        else:
            print(f"[{self.namespace}] Landing timed out but assuming complete")
    
    def return_to_home(self):
        """Handle return to home and landing sequence safely."""
        import traceback  # Import traceback to ensure it's available
        
        try:
            print(f"[{self.namespace}] Returning to home position")
            
            # First go to home position at safe altitude (2m)
            safe_home = [self.home_position[0], self.home_position[1], 2.0]
            
            # Set position multiple times to ensure command is received
            for _ in range(5):
                self.set_position(*safe_home)
                time.sleep(0.5)
                
            # Check if we've reached the home position
            if self.distance_to_target(safe_home) < 0.5:
                print(f"[{self.namespace}] At home position, starting landing")
                
                # Gradually descend to ground level
                descent_height = 2.0
                descent_step = 0.2
                
                while descent_height > 0:
                    descent_height -= descent_step
                    landing_pos = [self.home_position[0], self.home_position[1], max(0.1, descent_height)]
                    self.set_position(*landing_pos)
                    time.sleep(1.0)
                
                # Final landing command
                final_land = [self.home_position[0], self.home_position[1], 0.0]
                for _ in range(5):
                    self.set_position(*final_land)
                    time.sleep(0.5)
                
                print(f"[{self.namespace}] Landing complete")
                
                # Disarm after landing
                try:
                    # Check if armed using the proper method
                    is_armed = self.arm.status().is_armed
                    if is_armed:
                        self.disarm()
                except Exception as e:
                    print(f"[{self.namespace}] Error disarming: {e}")
            else:
                # If not at home yet, continue moving toward it
                self.set_position(*safe_home)
                time.sleep(0.5)
                
        except Exception as e:
            print(f"[{self.namespace}] Error in return_to_home: {e}")
            traceback.print_exc()
            
    def initiate_landing(self):
        """Initiate the landing sequence for all drones in the swarm."""
        print(f"[{self.namespace}] Initiating landing sequence for all drones")
        
        # First command all drones to prepare for landing (move to home positions)
        self.publish_command("prepare_landing")
        
        # Wait for all drones to report they're at landing positions
        max_wait_time = 30.0  # seconds
        start_time = time.time()
        all_ready = False
        
        # Create a set of expected drones (all except self)
        expected_drones = set()
        for i in range(len(self.swarm_namespaces)):
            if i != self.drone_index:
                expected_drones.add(f"drone{i}")
        
        print(f"[{self.namespace}] Waiting for drones to reach landing positions: {expected_drones}")
        
        while time.time() - start_time < max_wait_time:
            # Check if all expected drones are ready
            if self.landing_ready_drones.issuperset(expected_drones):
                all_ready = True
                break
                
            # Check if we have at least 80% of drones ready (safety fallback)
            if len(self.landing_ready_drones) >= 0.8 * len(expected_drones) and time.time() - start_time > 15.0:
                print(f"[{self.namespace}] 80% of drones ready, proceeding with landing")
                all_ready = True
                break
                
            # Wait a bit before checking again
            time.sleep(1.0)
            
        if all_ready:
            print(f"[{self.namespace}] All drones at landing positions, executing final landing")
            # Command all drones to execute landing
            self.publish_command("execute_landing")
            
            # Also execute our own landing
            self._execute_landing()
        else:
            print(f"[{self.namespace}] Warning: Not all drones reported ready for landing!")
            print(f"[{self.namespace}] Ready drones: {self.landing_ready_drones}")
            print(f"[{self.namespace}] Missing drones: {expected_drones - self.landing_ready_drones}")
            
            # Execute landing anyway after timeout
            print(f"[{self.namespace}] Executing landing sequence anyway")
            self.publish_command("execute_landing")
            self._execute_landing()
    
    def debug_print_all_home_positions(self):
        """Print all drone home positions for debugging."""
        print(f"\n[{self.namespace}] === DEBUG: All drone home positions ===")
        for drone_name, position in self.drone_home_positions.items():
            print(f"[{self.namespace}]   {drone_name}: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
        print(f"[{self.namespace}] =====================================\n")
    
    def calculate_swarm_center(self):
        """Calculate the center of the swarm based on all drone positions."""
        try:
            # Include our own position
            all_positions = [self.current_position]
            
            # Add peer positions
            for state in self.peer_states.values():
                if 'position' in state:
                    peer_position = state['position']
                    # Ensure position is a valid 3D point
                    if peer_position and len(peer_position) == 3:
                        all_positions.append(peer_position)
            
            # Calculate average position (centroid)
            if not all_positions:
                print(f"[{self.namespace}] Warning: No valid positions found, using current position")
                return self.current_position
            
            # Calculate sums
            try:
                sum_x = sum(pos[0] for pos in all_positions)
                sum_y = sum(pos[1] for pos in all_positions)
                sum_z = sum(pos[2] for pos in all_positions)
                
                center_x = sum_x / len(all_positions)
                center_y = sum_y / len(all_positions)
                center_z = sum_z / len(all_positions)
                
                return (center_x, center_y, center_z)
            except (IndexError, TypeError) as e:
                print(f"[{self.namespace}] Error calculating centroid: {e}, positions: {all_positions}")
                return self.current_position
        except Exception as e:
            print(f"[{self.namespace}] Unexpected error in calculate_swarm_center: {e}")
            return self.current_position
    
    def calculate_trajectory_point(self):
        """Calculate the base point on the trajectory circle."""
        # If no peers or returning home, use fallback logic
        if not self.peer_states or self.returning_home:
            # Return own home position if returning home
            if self.returning_home:
                home_x, home_y, home_z = self.home_position
                safe_altitude = max(home_z + 1.0, self.altitude)
                return (home_x, home_y, safe_altitude)
            
            # Otherwise use current position
            return self.current_position
        
        # Get average trajectory parameters from peers
        angles = [self.angle]
        for state in self.peer_states.values():
            if 'angle' in state:
                angles.append(state.get('angle', 0.0))
        
        # Average angle
        avg_angle = sum(angles) / len(angles)
        
        # Calculate base position on the circle
        base_x = self.stage_center[0] + self.radius * math.cos(avg_angle)
        base_y = self.stage_center[1] + self.radius * math.sin(avg_angle)
        
        # Use average altitude
        altitudes = [self.current_position[2]]
        for state in self.peer_states.values():
            if 'position' in state:
                altitudes.append(state['position'][2])
        
        avg_altitude = sum(altitudes) / len(altitudes)
        
        return (base_x, base_y, avg_altitude)
    
    def calculate_position(self):
        """Calculate this drone's position in the current formation."""
        try:
            # If returning home, go to home position
            if self.returning_home:
                home_x, home_y, home_z = self.home_position
                safe_altitude = max(home_z + 1.0, self.altitude)
                return (home_x, home_y, safe_altitude)
            
            # Non-coordinator drones need to wait for drone0 state
            if self.drone_index != 0 and 0 not in self.peer_states:
                # If we haven't received state from coordinator yet, just hover
                print(f"[{self.namespace}] Waiting for coordinator state...")
                return self.current_position
                
            # Get formation function for current formation
            formation_func = self.formations[self.current_formation]
            
            # Get offsets for all drones in this formation
            formation_offsets = formation_func()
            
            # Verify we have enough positions in the formation
            if len(formation_offsets) <= self.drone_index:
                print(f"[{self.namespace}] ERROR: Not enough positions in formation for drone index {self.drone_index}")
                return self.current_position
            
            # Get this drone's offset (2D point)
            my_offset = formation_offsets[self.drone_index]
            
            # For circular orbit (formation 3), apply rotation
            if self.current_formation == 3:  # Circular orbit
                # Update rotation angle continuously for smooth orbit
                current_time = time.time()
                time_delta = current_time - self.last_rotation_update
                
                # Always decrement for anticlockwise movement
                self.current_rotation_angle -= self.rotation_speed * time_delta
                
                # Keep angle in range [0, 2π]
                if self.current_rotation_angle < 0:
                    self.current_rotation_angle += 2.0 * math.pi
                if self.current_rotation_angle >= 2.0 * math.pi:
                    self.current_rotation_angle -= 2.0 * math.pi
                    
                self.last_rotation_update = current_time
                
                # Apply rotation to the offset
                my_offset = rotate_offset(my_offset, self.current_rotation_angle)
            else:
                # For non-circular formations, apply regular trajectory angle
                my_offset = rotate_offset(my_offset, self.angle)
            
            # Use the current angle to determine position on the trajectory circle
            angle_to_use = self.angle
            base_x = self.stage_center[0] + self.radius * math.cos(angle_to_use)
            base_y = self.stage_center[1] + self.radius * math.sin(angle_to_use)
            base_z = self.altitude
            
            # Apply offset to base trajectory point
            position_x = base_x + my_offset[0]
            position_y = base_y + my_offset[1]
            position_z = base_z
            
            # Debug output for follower drones to verify they're moving
            if self.drone_index != 0:
                print(f"[{self.namespace}] Moving to position: ({position_x:.2f}, {position_y:.2f}, {position_z:.2f})")
            
            return (position_x, position_y, position_z)
        except Exception as e:
            print(f"[{self.namespace}] Error in calculate_position: {e}")
            return self.current_position
    
    def update_position(self):
        """
        Main position update loop for the drone. Handles takeoff, formation flight, and mission execution.
        - Coordinator (drone0) leads the mission and manages formation transitions
        - Follower drones (drone1-4) follow the coordinator based on formation parameters
        """
        try:
            # Wait for drone to be armed before attempting position control
            # We need to wait until main() arms the drone and sets offboard mode
            print(f"[{self.namespace}] Position update thread waiting for arming and offboard mode")
            while rclpy.ok() and self.should_position_update:
                try:
                    # Check if armed and in offboard mode by getting status directly from the parent class
                    armed_ok = self.arm.status().is_armed
                    offb_ok = self.offboard.status().status == BehaviorStatus.IDLE
                    
                    if armed_ok and offb_ok:
                        print(f"[{self.namespace}] Drone armed and in offboard mode, starting position control")
                        break
                except Exception as e:
                    # If we can't access the status yet, wait
                    pass
                    
                # Wait a bit before checking again
                time.sleep(1.0)
                
            # Try to get current position, with retry logic
            retry_count = 0
            max_retries = 5
            while retry_count < max_retries:
                if self.current_position is None:
                    print(f"[{self.namespace}] Waiting for valid position data...")
                    time.sleep(1.0)
                    retry_count += 1
                else:
                    break
                    
            if self.current_position is None:
                print(f"[{self.namespace}] Failed to get valid position after {max_retries} attempts")
                return
                
            # Initial takeoff for all drones to 2 meters
            if not self.takeoff_complete:
                print(f"[{self.namespace}] Taking off to 2 meters...")
                takeoff_target = [self.current_position[0], self.current_position[1], 2.0]
                
                # Command takeoff
                for i in range(10):  # Send multiple commands to ensure execution
                    self.set_position(*takeoff_target)
                    time.sleep(0.5)
                    
                # Mark takeoff as complete    
                self.takeoff_complete = True
                print(f"[{self.namespace}] Takeoff complete")
                
                # Publish updated state to inform other drones
                self.publish_state()
                
                # Wait for stabilization
                time.sleep(2.0)
                
            # --- Main position update loop ---
            while rclpy.ok() and self.should_position_update:
                # Get latest position
                self.publish_state()
                
                # Check if returning home
                if self.returning_home:
                    self.return_to_home()
                    return
                
                # Coordinator behavior (drone0)
                if self.drone_index == 0:
                    # Wait for all drones to complete takeoff before starting mission
                    if not self.mission_started:
                        all_ready = True
                        for idx in range(1, len(self.swarm_namespaces)):
                            drone_name = f"drone{idx}"
                            if drone_name not in self.swarm_positions:
                                all_ready = False
                                break
                        
                        if all_ready:
                            print(f"[{self.namespace}] All drones ready, starting mission")
                            self.mission_started = True
                        else:
                            print(f"[{self.namespace}] Waiting for all drones to be ready...")
                            time.sleep(1.0)
                            continue
                    
                    # Proceed with trajectory following
                    if hasattr(self, 'trajectory') and self.current_target_index < len(self.trajectory):
                        target = self.trajectory[self.current_target_index]
                        print(f"[{self.namespace}] Moving to target {self.current_target_index}: {target}")
                        self.set_position(*target)
                        
                        # Check if we've reached the target (within threshold)
                        if self.distance_to_target(target) < 0.5:
                            print(f"[{self.namespace}] Reached target {self.current_target_index}")
                            self.current_target_index += 1
                            
                            # Change formation at specific points in trajectory
                            if self.current_target_index % 3 == 0:
                                self.current_formation = (self.current_formation + 1) % 5
                                print(f"[{self.namespace}] Changing to formation {self.current_formation}")
                    else:
                        # Trajectory complete, return home
                        print(f"[{self.namespace}] Trajectory complete, returning home")
                        self.returning_home = True
                        
                # Follower behavior (drone1-4)
                else:
                    # Wait for mission to start
                    if not self.mission_started:
                        print(f"[{self.namespace}] Waiting for coordinator to start mission...")
                        time.sleep(1.0)
                        continue
                    
                    # Get coordinator position (if available)
                    if hasattr(self, 'coordinator_state') and hasattr(self.coordinator_state, 'position'):
                        coordinator_pos = [
                            self.coordinator_state.position.x,
                            self.coordinator_state.position.y,
                            self.coordinator_state.position.z
                        ]
                        
                        # Calculate formation offset based on current formation
                        offset = self.calculate_formation_offset(
                            self.current_formation,
                            self.drone_index
                        )
                        
                        # Apply offset to coordinator position
                        target_position = [
                            coordinator_pos[0] + offset[0],
                            coordinator_pos[1] + offset[1],
                            coordinator_pos[2] + offset[2]
                        ]
                        
                        # Move to formation position
                        self.set_position(*target_position)
                        print(f"[{self.namespace}] Following coordinator in formation {self.current_formation}")
                    else:
                        print(f"[{self.namespace}] Waiting for coordinator position...")
                
                # Update at consistent rate
                time.sleep(0.5)
            
            print(f"[{self.namespace}] Position update loop ended")
                
        except Exception as e:
            print(f"[{self.namespace}] Error in update_position: {e}")
            traceback.print_exc()  # Using the imported traceback from the top

    def calculate_formation_offset(self, formation_index, drone_index):
        """Calculate position offset from coordinator based on formation and drone index."""
        # Default spacing between drones
        spacing = 2.0
        
        # Relative position based on drone index (1-indexed for followers)
        relative_index = drone_index

        # Offset calculations for different formations
        if formation_index == 0:
            # Line formation
            return [0, spacing * relative_index, 0]
        elif formation_index == 1:
            # Column formation
            return [spacing * relative_index, 0, 0]
        elif formation_index == 2:
            # Diamond formation
            if relative_index == 1:
                return [spacing, 0, 0]
            elif relative_index == 2:
                return [0, spacing, 0]
            elif relative_index == 3:
                return [-spacing, 0, 0] 
            elif relative_index == 4:
                return [0, -spacing, 0]
        elif formation_index == 3:
            # Square formation
            if relative_index == 1:
                return [spacing, spacing, 0]
            elif relative_index == 2:
                return [spacing, -spacing, 0]
            elif relative_index == 3:
                return [-spacing, -spacing, 0]
            elif relative_index == 4:
                return [-spacing, spacing, 0]
        elif formation_index == 4:
            # V formation
            if relative_index == 1:
                return [spacing, spacing, 0]
            elif relative_index == 2:
                return [spacing * 2, spacing * 2, 0]
            elif relative_index == 3:
                return [-spacing, spacing, 0]
            elif relative_index == 4:
                return [-spacing * 2, spacing * 2, 0]
                
        # Default to line formation if unknown
        return [0, spacing * relative_index, 0]
    
    def distance_to_target(self, target):
        """Calculate Euclidean distance to target position."""
        dx = self.current_position[0] - target[0]
        dy = self.current_position[1] - target[1]
        dz = self.current_position[2] - target[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def execute_trajectory(self):
        """Execute the trajectory calculation and movement."""
        self.running = True
        
        # Calculate total steps
        # The total steps calculation needs to account for the special circular orbit steps
        circular_orbit_index = 3
        total_steps = 0
        
        for formation_idx in range(self.total_formations):
            if formation_idx == circular_orbit_index:
                # Circular orbit has more steps
                total_steps += self.formation_repeats * self.circular_orbit_steps
            else:
                # Other formations have regular steps
                total_steps += self.formation_repeats * self.steps_per_formation
        
        # Display formation plan
        print(f"[{self.namespace}] Flight plan: {self.total_formations} formations, "
              f"each repeated {self.formation_repeats} times")
        print(f"[{self.namespace}] Each formation lasts {self.degrees_per_formation}° (half a revolution)")
        print(f"[{self.namespace}] Circular orbit uses {self.circular_orbit_steps} steps per 180°, "
              f"other formations use {self.steps_per_formation} steps")
        print(f"[{self.namespace}] Total flight: {self.degrees_per_formation * self.total_formations * self.formation_repeats}° "
              f"({self.total_formations * self.formation_repeats * self.degrees_per_formation / 360} revolutions)")
        print(f"[{self.namespace}] Total steps: {total_steps}")
        print(f"[{self.namespace}] Formation sequence: {' -> '.join(self.formation_names)}")
        
        # Start at step 0
        step = 0
        last_publish_time = 0
        publish_interval = 0.05  # Publish state 20 times per second
        
        # If we're drone0, we'll lead the trajectory advancement
        is_trajectory_coordinator = (self.drone_index == 0)
        
        if is_trajectory_coordinator:
            print(f"[{self.namespace}] Acting as trajectory coordinator")
        
        while self.running and step < total_steps:
            # Determine if it's time to switch formation
            # Only the coordinator drone manages formation advancement
            if is_trajectory_coordinator:
                if (self.current_formation == 3 and self.step_in_formation >= self.circular_orbit_steps) or \
                   (self.current_formation != 3 and self.step_in_formation >= self.steps_per_formation):
                    self.step_in_formation = 0
                    self.formation_repeat += 1
                    
                    # Move to next formation after all repeats are done
                    if self.formation_repeat >= self.formation_repeats:
                        self.formation_repeat = 0
                        self.current_formation = (self.current_formation + 1) % self.total_formations
                        current_formation_name = self.formation_names[self.current_formation]
                        print(f"\n[{self.namespace}] *** Switching to formation: "
                              f"{self.current_formation} - {current_formation_name} ***\n")
                        
                        # Brief pause between formations
                        time.sleep(self.formation_switch_pause)
                    else:
                        current_formation_name = self.formation_names[self.current_formation]
                        print(f"\n[{self.namespace}] --- Repeating formation {self.current_formation} "
                              f"- {current_formation_name} (repeat {self.formation_repeat+1}"
                              f"/{self.formation_repeats}) ---\n")
            
            # Compute base (center) position on the circle
            # Use different angle step for circular orbit
            if self.current_formation == 3:  # Circular orbit
                current_angle_step = self.circular_orbit_angle_step
            else:
                current_angle_step = self.angle_step
            
            # Publish state at regular intervals
            current_time = time.time()
            if current_time - last_publish_time >= publish_interval:
                self.publish_state()
                last_publish_time = current_time
            
            # Only the coordinator increments the step and angle
            if is_trajectory_coordinator:
                # For circular orbit, ensure we use the right angle step
                if self.current_formation == 3:  # Circular orbit
                    self.angle += self.circular_orbit_angle_step
                else:
                    self.angle += self.angle_step
                    
                self.step_in_formation += 1
                step += 1
                
                # Publish updated state immediately after changing step
                self.publish_state()
                
                # Display status every step (for coordinator)
                current_formation_name = self.formation_names[self.current_formation]
                max_steps = self.circular_orbit_steps if self.current_formation == 3 else self.steps_per_formation
                print(f"[{self.namespace}] Step {step}/{total_steps}: "
                      f"Formation '{current_formation_name}' "
                      f"(repeat {self.formation_repeat+1}/{self.formation_repeats})")
                print(f"[{self.namespace}] Angle = {math.degrees(self.angle):.1f}°")
                
                # Brief stabilization wait between steps
                # Use different timings for different formations
                if self.current_formation == 0:  # Line formation
                    wait_time = self.stabilization_time * 1.5  # Slightly longer for line formation
                elif self.current_formation == 3:  # Circular orbit
                    wait_time = 0.2  # Very short wait for orbit
                else:
                    wait_time = self.stabilization_time
                
                time.sleep(wait_time)
            else:
                # Non-coordinator drones just follow the consensus state
                time.sleep(0.05)
            
            # Check if we're on the last step
            if is_trajectory_coordinator and step >= total_steps:
                print(f"\n[{self.namespace}] *** Mission complete! Returning to start position. ***\n")
                self.mission_complete = True
                
                # Return to home and broadcast to peers
                self.returning_home = True
                self.publish_state()
                self.return_to_home()
                
                # Publish command to prepare for landing
                self.publish_command("prepare_landing")
                break
        
        # For safety, in case the trajectory is stopped before completion
        if self.running and not self.mission_complete and is_trajectory_coordinator:
            print(f"\n[{self.namespace}] *** Trajectory interrupted before completion! ***\n")
    
    def start_mission(self):
        """Start the mission if this drone is the coordinator.
        This function is called on the coordinator drone to initiate the mission.
        """
        if self.drone_index != 0:
            self.get_logger().info("Only the coordinator (drone0) can start the mission")
            return False
            
        self.get_logger().info("Coordinator starting mission...")
        
        # Publish takeoff command for all drones
        self.publish_command("takeoff")
        self.get_logger().info("Takeoff command published to all drones")
        
        # Wait for drones to reach takeoff altitude
        time.sleep(5.0)
        
        # Start the mission sequence
        self.execute_mission()
        
        return True
        
    def execute_mission(self):
        """Execute the mission sequence.
        This can be overridden in subclasses to implement different mission types.
        """
        self.get_logger().info("Executing default mission sequence...")
        
        # Default to a basic up-and-down mission
        # First formation
        self.formation = 'circle'
        self.publish_command("formation1")
        time.sleep(10.0)
        
        # Second formation
        self.formation = 'line'
        self.publish_command("formation2")
        time.sleep(10.0)
        
        # Return to home and land
        self.publish_command("land")
        self.get_logger().info("Mission complete, landing command sent")

    def set_offboard_mode(self):
        """Set the drone to offboard mode by calling the parent class's offboard() method."""
        try:
            print(f"[{self.namespace}] Setting to offboard mode...")
            # Call the parent class's offboard method
            result = self.offboard()
            print(f"[{self.namespace}] Offboard mode result: {result}")
            return result
        except Exception as e:
            print(f"[{self.namespace}] Error setting offboard mode: {e}")
            return False

def run_drone(namespace, all_namespaces, drone_index, config_file, verbose=False, use_sim_time=True):
    """Run a single drone within the swarm.
    
    Args:
        namespace (str): The namespace of this drone.
        all_namespaces (list): List of all drone namespaces in the swarm.
        drone_index (int): Index of this drone in the swarm (0 for coordinator).
        config_file (str): Path to configuration file.
        verbose (bool): Enable verbose output.
        use_sim_time (bool): Use simulation time.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    # Validate inputs
    if drone_index >= len(all_namespaces):
        print(f"Error: Drone index {drone_index} is out of range for namespace list: {all_namespaces}")
        return False
    
    if namespace != all_namespaces[drone_index]:
        print(f"Error: Namespace mismatch. Expected {all_namespaces[drone_index]} but got {namespace}")
        return False
    
    # Check if config file exists and use default if not
    if not os.path.exists(config_file):
        print(f"Warning: Config file {config_file} not found, using default")
        # Try to find a default config in common locations
        possible_configs = [
            "/home/ziyar/project_gazebo_ws/src/challenge_multi_drone/scenarios/scenario1_stage1.yaml",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios/scenario1_stage1.yaml"),
            os.path.expanduser("~/project_gazebo_ws/src/challenge_multi_drone/scenarios/scenario1_stage1.yaml")
        ]
        
        for possible_config in possible_configs:
            if os.path.exists(possible_config):
                config_file = possible_config
                print(f"Using found config file: {config_file}")
                break
    
    # Initialize and set up the drone
    try:
        print(f"Initializing drone: {namespace} (index: {drone_index})")
        
        # Create the SwarmDrone instance
        drone = SwarmDrone(namespace, all_namespaces, drone_index, config_file, verbose, use_sim_time)
        
        # Attempt to arm and set to offboard mode with retries
        arm_attempts = 0
        MAX_ARM_ATTEMPTS = 3
        
        while arm_attempts < MAX_ARM_ATTEMPTS:
            if drone.arm():
                print(f"[{namespace}] Armed successfully")
                break
            arm_attempts += 1
            print(f"[{namespace}] Arm attempt {arm_attempts}/{MAX_ARM_ATTEMPTS} failed, retrying...")
            time.sleep(1.0)
        
        if arm_attempts == MAX_ARM_ATTEMPTS:
            print(f"[{namespace}] Failed to arm after {MAX_ARM_ATTEMPTS} attempts")
            return False
        
        # Set to offboard mode with retries
        offboard_attempts = 0
        MAX_OFFBOARD_ATTEMPTS = 3
        
        while offboard_attempts < MAX_OFFBOARD_ATTEMPTS:
            if drone.set_offboard_mode():
                print(f"[{namespace}] Set to offboard mode successfully")
                break
            offboard_attempts += 1
            print(f"[{namespace}] Offboard mode attempt {offboard_attempts}/{MAX_OFFBOARD_ATTEMPTS} failed, retrying...")
            time.sleep(1.0)
            
        if offboard_attempts == MAX_OFFBOARD_ATTEMPTS:
            print(f"[{namespace}] Failed to set offboard mode after {MAX_OFFBOARD_ATTEMPTS} attempts")
            return False
        
        # If this is the coordinator drone, it will start the mission
        # Otherwise, it will wait for commands from the coordinator
        if drone_index == 0:
            print(f"[{namespace}] I am the coordinator, will start the mission after a delay")
            # Allow time for other drones to initialize
            time.sleep(5.0)
            drone.start_mission()
        else:
            print(f"[{namespace}] I am a follower, waiting for commands from coordinator")
        
        # Keep the drone running until shutdown
        try:
            rclpy.spin(drone)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"[{namespace}] Exception in drone execution: {str(e)}")
        finally:
            # Clean shutdown
            if hasattr(drone, 'in_air') and drone.in_air:
                print(f"[{namespace}] Emergency landing due to shutdown")
                drone.return_to_home()
            drone.destroy_node()
            return True
            
    except Exception as e:
        print(f"[{namespace}] Exception during initialization: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Entry point for the script."""
    parser = argparse.ArgumentParser(description='Run a swarm drone.')
    parser.add_argument('-n', '--namespaces', nargs='+', required=True,
                       help='List of namespaces for all drones in the swarm')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('-s', '--use-sim-time', action='store_true',
                       help='Use simulation time')
    parser.add_argument('-c', '--config', default='/home/ziyar/project_gazebo_ws/src/challenge_multi_drone/scenarios/scenario1_stage1.yaml',
                       help='Path to scenario configuration file')
    parser.add_argument('-r', '--role', choices=['single', 'all'], default='single',
                       help='Role of this script (single=one drone, all=all drones)')
    parser.add_argument('-i', '--index', type=int, default=0,
                       help='Index of the drone to run')
    
    args = parser.parse_args()
    
    # Initialize ROS
    rclpy.init(args=None)
    
    # Print configuration for debugging
    print(f"Configuration:")
    print(f"  Namespaces: {args.namespaces}")
    print(f"  Verbose: {args.verbose}")
    print(f"  Use Sim Time: {args.use_sim_time}")
    print(f"  Config File: {args.config}")
    print(f"  Role: {args.role}")
    print(f"  Index: {args.index}")
    
    # Run all drones if role is "all"
    if args.role == 'all':
        processes = []
        for i, namespace in enumerate(args.namespaces):
            # Build the command correctly - note that we need to include '-n' before each namespace
            cmd = [sys.executable, __file__, '-n']
            # Add each namespace separately to avoid list flattening issues
            for ns in args.namespaces:
                cmd.append(ns) 
            # Add the rest of the arguments
            cmd.extend(['-r', 'single', '-i', str(i), '-c', args.config])
            
            if args.verbose:
                cmd.append('-v')
            if args.use_sim_time:
                cmd.append('-s')
                
            print(f"Starting process for {namespace}: {' '.join(cmd)}")
            p = subprocess.Popen(cmd)
            processes.append(p)
            
        try:
            for p in processes:
                p.wait()
        except KeyboardInterrupt:
            print("Keyboard interrupt received, terminating all drones")
            for p in processes:
                p.terminate()
    else:
        # Run a single drone
        namespace = args.namespaces[args.index]
        success = run_drone(
            namespace=namespace,
            all_namespaces=args.namespaces,
            drone_index=args.index,
            config_file=args.config,
            verbose=args.verbose,
            use_sim_time=args.use_sim_time
        )
        if not success:
            print(f"Failed to run drone {namespace}")
            sys.exit(1)
    
    # Shutdown ROS
    rclpy.shutdown()

if __name__ == '__main__':
    main() 