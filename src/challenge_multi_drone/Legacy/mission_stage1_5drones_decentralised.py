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
from orca_formation import ORCAFormation

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
    Leader (drone2) in the center, others form diamond around it.
    """
    d = 0.4
    return [(0.0, d), (-d, 0.0), (0.0, 0.0), (d, 0.0), (0.0, -d)]

def formation_circular_orbit_5():
    """
    Return offsets for a circular orbit formation for 5 drones.
    Leader (drone2) in center, others positioned evenly along a small circle.
    """
    orbit_radius = 0.4
    offsets = [(0.0, 0.0)]  # Leader at center
    for i in range(4):
        angle = 2 * math.pi * i / 4
        offsets.append((orbit_radius * math.cos(angle), orbit_radius * math.sin(angle)))
    
    # Rearrange to match drone index order (drone2 is leader)
    return [offsets[4], offsets[1], offsets[0], offsets[2], offsets[3]]

def formation_grid_5():
    """
    Return offsets for a grid formation for 5 drones.
    Arranged in a cross pattern with leader (drone2) in center.
    """
    d = 0.4
    return [(0.0, d), (-d, 0.0), (0.0, 0.0), (d, 0.0), (0.0, -d)]

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

# --- Leader Drone class ---
class LeaderDrone(DroneInterface):
    """
    Leader drone that computes trajectory and broadcasts to followers.
    """
    def __init__(self, namespace: str, follower_namespaces: list, config_path: str, 
                 verbose: bool = False, use_sim_time: bool = False):
        super().__init__(namespace, verbose=verbose, use_sim_time=use_sim_time)
        self._speed = 0.5
        self._yaw_mode = YawMode.PATH_FACING
        self._yaw_angle = None
        self._frame_id = "earth"
        self.current_behavior: BehaviorHandler = None
        self.led_pub = self.create_publisher(ColorRGBA, f"/{namespace}/leds/control", 10)
        
        # Load configuration
        self.config = load_scenario_config(config_path)
        if not self.config:
            raise ValueError(f"Failed to load configuration from {config_path}")
            
        # Load drone home positions from world_swarm.yaml
        self.drone_home_positions = load_drone_home_positions()
        if not self.drone_home_positions:
            print("WARNING: Could not load drone home positions from file, using hardcoded defaults")
            # Create default positions based on the provided world_swarm.yaml data
            self.drone_home_positions = {
                "drone0": (-1.0, 0.0, 0.2),
                "drone1": (1.0, 0.0, 0.2),
                "drone2": (0.0, 0.0, 0.2),
                "drone3": (0.0, 1.0, 0.2),
                "drone4": (0.0, -1.0, 0.2)
            }
        
        # Print loaded home positions for debugging
        print(f"Loaded drone home positions: {self.drone_home_positions}")
            
        # Stage parameters
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
        
        # Movement speed parameters
        self.movement_speed = 0.7  # Slightly increased for more continuous movement
        self.stabilization_time = 0.5  # Reduced stabilization time significantly
        self.formation_switch_pause = 1.0  # Reduced pause between formations
        
        # Ensure we have exactly one full revolution (360 degrees)
        self.total_steps = self.total_formations * self.formation_repeats * self.steps_per_formation
        
        # State variables
        self.current_formation = 0
        self.formation_repeat = 0
        self.step_in_formation = 0
        self.angle = 0.0
        self.current_position = (0.0, 0.0, self.altitude)
        self.running = False
        self.mission_complete = False
        
        # Formation definitions
        self.formations = [
            formation_line_5,
            formation_v_5,
            formation_diamond_5,
            formation_circular_orbit_5,
            formation_grid_5,
            formation_staggered_5
        ]
        
        # Communication
        self.state_publisher = self.create_publisher(
            Float32MultiArray, '/formation/leader_state', 10)
        self.command_publisher = self.create_publisher(
            String, '/formation/command', 10)
        
        # Tracking followers
        self.follower_namespaces = follower_namespaces
        
        # Create thread for trajectory calculation and publishing
        self.trajectory_thread = None

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
        # Update current position
        self.current_position = (x, y, z)

    def goal_reached(self) -> bool:
        """Check if the current behavior has finished (IDLE)."""
        if not self.current_behavior:
            return False
        return self.current_behavior.status == BehaviorStatus.IDLE

    def broadcast_state(self, returning_home=False):
        """Broadcast current state to followers."""
        msg = Float32MultiArray()
        # Format: [x, y, z, angle, formation_index, formation_repeat, step_in_formation, returning_home_flag]
        msg.data = [
            float(self.current_position[0]),
            float(self.current_position[1]),
            float(self.current_position[2]),
            float(self.angle),
            float(self.current_formation),
            float(self.formation_repeat),
            float(self.step_in_formation),
            1.0 if returning_home else 0.0  # Add flag to indicate returning home
        ]
        self.state_publisher.publish(msg)
        
        # Add a small delay after publishing to ensure messages are processed
        time.sleep(0.01)
        
    def broadcast_command(self, command):
        """Broadcast command to follower drones."""
        msg = String()
        msg.data = command
        self.command_publisher.publish(msg)
    
    def debug_print_all_home_positions(self):
        """Print all drone home positions for debugging."""
        print("\n=== DEBUG: All drone home positions ===")
        for drone_name, position in self.drone_home_positions.items():
            print(f"  {drone_name}: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
        print("=====================================\n")
        
    def takeoff_all(self, height=2.0, speed=0.7):
        """Takeoff the leader and command followers to takeoff."""
        print(f"[{self.namespace}] Taking off...")
        self.do_behavior("takeoff", height, speed, False)
        self.change_led_colour((0, 255, 0))
        
        # Wait until leader is ready
        while not self.goal_reached():
            time.sleep(0.1)
            
        # Command followers to takeoff
        self.broadcast_command("takeoff")
        
        # Broadcast current state several times to ensure followers get it
        for _ in range(5):
            self.broadcast_state()
            time.sleep(0.1)
            
        # Wait for followers (slightly reduced time)
        time.sleep(3.0)
        print("All drones should now be in the air")
    
    def land_all(self, speed=0.4):
        """Land the leader and command followers to land."""
        # Debug print all home positions
        self.debug_print_all_home_positions()
        
        # Stop trajectory execution
        self.running = False
        if self.trajectory_thread and self.trajectory_thread.is_alive():
            self.trajectory_thread.join()
            
        # Command followers to prepare for landing - move to landing positions
        print("\n*** Commanding followers to prepare for landing ***\n")
        self.broadcast_command("prepare_landing")
        
        # Set up a subscriber to receive landing position confirmations from followers
        self.setup_landing_position_confirmation()
        
        # Wait for followers to confirm they've reached landing positions
        if self.wait_for_followers_ready():
            print("\n*** All followers confirmed at landing positions ***\n")
        else:
            print("\n*** Not all followers confirmed landing positions, proceeding anyway ***\n")
        
        # Now command all followers to execute landing
        print("\n*** Commanding all drones to execute landing ***\n")
        self.broadcast_command("execute_landing")
        time.sleep(1.0)  # Give time for the command to be received
        
        # Make sure drones have time to receive and process the land command
        print("Waiting for followers to start landing...")
        time.sleep(2.0)
        
        # Land the leader
        print(f"[{self.namespace}] Landing...")
        self.do_behavior("land", speed, False)
        
        # Wait until landing is complete
        countdown = 30  # 30 second timeout
        while not self.goal_reached() and countdown > 0:
            time.sleep(0.5)
            countdown -= 0.5
            if countdown % 5 == 0:
                print(f"Waiting for landing to complete... {int(countdown)}s remaining")
        
        if not self.goal_reached():
            print("WARNING: Landing may not have completed properly")
        else:
            print("Leader has landed successfully")
            
        # Wait for followers to complete landing
        print("Waiting for followers to complete landing...")
        time.sleep(5.0)
        
        print("All drones should now be landed")
    
    def setup_landing_position_confirmation(self):
        """Setup subscriber to receive landing position confirmations from followers."""
        self.landing_ready_followers = set()
        self.landing_confirmation_sub = self.create_subscription(
            String,
            '/formation/landing_ready',
            self.landing_ready_callback,
            10
        )
    
    def landing_ready_callback(self, msg):
        """Callback for landing position confirmations."""
        follower_name = msg.data
        print(f"Received landing position confirmation from {follower_name}")
        self.landing_ready_followers.add(follower_name)
    
    def wait_for_followers_ready(self):
        """Wait for all followers to confirm they are at landing positions."""
        # Maximum wait time for all followers to be ready
        max_wait_time = 15.0  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # Check if all followers have confirmed
            if len(self.landing_ready_followers) >= len(self.follower_namespaces):
                return True
                
            # Print progress every 2 seconds
            if int(time.time() - start_time) % 2 == 0:
                ready_count = len(self.landing_ready_followers)
                total_count = len(self.follower_namespaces)
                print(f"Waiting for landing position confirmations: {ready_count}/{total_count} followers ready")
                
            time.sleep(0.1)
            
        # If we've timed out, return False
        return False
    
    def execute_trajectory(self):
        """Execute the trajectory calculation and movement."""
        self.running = True
        
        # Calculate total steps
        total_steps = self.total_steps
        
        # Display formation plan
        print(f"Flight plan: {self.total_formations} formations, each repeated {self.formation_repeats} times")
        print(f"Each formation lasts {self.degrees_per_formation}° (half a revolution)")
        print(f"Total flight: {total_steps * self.angle_deg}° ({total_steps * self.angle_deg / 360} revolutions)")
        print(f"Total steps: {total_steps}")
        
        # Start at step 0
        step = 0
        
        while self.running and step < total_steps:
            # Determine if it's time to switch formation
            if self.step_in_formation >= self.steps_per_formation:
                self.step_in_formation = 0
                self.formation_repeat += 1
                
                # Move to next formation after all repeats are done
                if self.formation_repeat >= self.formation_repeats:
                    self.formation_repeat = 0
                    self.current_formation = (self.current_formation + 1) % self.total_formations
                    print(f"\n*** Switching to formation: {self.current_formation} ***\n")
                    
                    # Brief pause between formations
                    time.sleep(self.formation_switch_pause)
                else:
                    print(f"\n--- Repeating formation {self.current_formation} (repeat {self.formation_repeat+1}/{self.formation_repeats}) ---\n")
            
            # Compute base (center) position on the circle
            base_x = self.stage_center[0] + self.radius * math.cos(self.angle)
            base_y = self.stage_center[1] + self.radius * math.sin(self.angle)
            
            # Calculate leader position (leader is at index 2 for drone2 in formation)
            formation_func = self.formations[self.current_formation]
            offsets = formation_func()
            leader_offset = offsets[2]  # Leader's offset in formation (drone2)
            
            # Rotate leader's offset
            rotated = rotate_offset(leader_offset, self.angle)
            pos_x = base_x + rotated[0]
            pos_y = base_y + rotated[1]
            
            # Display informational message
            print(f"Step {step}/{total_steps}: Formation {self.current_formation} (repeat {self.formation_repeat+1}/{self.formation_repeats})")
            print(f"Base position = ({base_x:.2f}, {base_y:.2f}), Angle = {math.degrees(self.angle):.1f}°")
            print(f"Leader position: ({pos_x:.2f}, {pos_y:.2f}, {self.altitude})")
            
            # Move leader to position
            self.go_to_position(pos_x, pos_y, self.altitude, speed=self.movement_speed)
            
            # Wait until leader is close to target position, with a timeout
            wait_start = time.time()
            max_wait_time = 3.0  # Maximum wait time for position
            goal_check_interval = 0.05  # Check goal more frequently
            broadcast_interval = 0.05  # Broadcast state more frequently
            
            last_broadcast = 0
            while not self.goal_reached() and self.running and (time.time() - wait_start < max_wait_time):
                # Broadcast state periodically to keep followers updated
                current_time = time.time()
                if current_time - last_broadcast >= broadcast_interval:
                    self.broadcast_state()
                    last_broadcast = current_time
                time.sleep(goal_check_interval)
            
            # Brief stabilization wait - different for each formation
            # First determine how much to wait based on formation
            if self.current_formation == 0:  # Line formation
                wait_time = self.stabilization_time * 1.5  # Slightly longer for line formation
            else:
                wait_time = self.stabilization_time
                
            # Broadcast state continuously during the short stabilization wait
            wait_start = time.time()
            while time.time() - wait_start < wait_time and self.running:
                self.broadcast_state()
                time.sleep(0.05)
            
            # Increment counters and angles and continue immediately
            self.angle += self.angle_step
            self.step_in_formation += 1
            step += 1
            
            # Check if we're on the last step
            if step >= total_steps:
                print("\n*** Mission complete! Returning to start position. ***\n")
                self.mission_complete = True
                
                # Return to the start position (center of the stage at our altitude)
                self.return_to_start()
                break
        
        # If we've completed all steps, mark as done
        if step >= total_steps:
            self.mission_complete = True
            
        # For safety, in case the trajectory is stopped before completion
        if self.running and not self.mission_complete:
            print("\n*** Trajectory interrupted before completion! ***\n")
    
    def return_to_start(self):
        """Return to the home position before landing."""
        # Get leader's home position
        home_x, home_y, home_z = self.drone_home_positions.get(self.namespace, (0.0, 0.0, 0.2))
        
        # Verify we're using the correct home position
        print(f"[{self.namespace}] DEBUG: Using home position coordinates: ({home_x:.2f}, {home_y:.2f}, {home_z:.2f})")
        
        # Add some altitude for safety
        safe_altitude = max(home_z + 1.0, self.altitude)
        
        print(f"Returning to home position: ({home_x:.2f}, {home_y:.2f}, {safe_altitude})")
        
        # Move to position above home
        self.go_to_position(home_x, home_y, safe_altitude, speed=0.8)
        
        # Wait for leader to reach position
        wait_start = time.time()
        broadcast_interval = 0.05
        last_broadcast = 0
        
        # Give more time to reach the home position
        max_wait_time = 5.0
        
        while not self.goal_reached() and self.running and (time.time() - wait_start < max_wait_time):
            # Broadcast leader's position and that we're returning home
            # Add a special flag in the broadcast to indicate we're returning home
            self.broadcast_state(returning_home=True)
            time.sleep(0.05)
        
        # Wait at home position briefly
        wait_time = 2.0
        wait_start = time.time()
        
        print("At home position, preparing for landing...")
        
        # Broadcast position continuously during final wait
        while time.time() - wait_start < wait_time and self.running:
            self.broadcast_state(returning_home=True)
            time.sleep(0.05)
    
    def start(self):
        """Start the leader drone's trajectory execution in a separate thread."""
        if self.trajectory_thread is None or not self.trajectory_thread.is_alive():
            self.trajectory_thread = threading.Thread(target=self.execute_trajectory)
            self.trajectory_thread.daemon = True
            self.trajectory_thread.start()
    
    def stop(self):
        """Stop the trajectory execution."""
        self.running = False
        if self.trajectory_thread and self.trajectory_thread.is_alive():
            self.trajectory_thread.join()

# --- Follower Drone class ---
class FollowerDrone(DroneInterface):
    """
    Follower drone that receives state from leader and calculates its own position.
    """
    def __init__(self, namespace: str, follower_index: int, verbose: bool = False, use_sim_time: bool = False):
        super().__init__(namespace, verbose=verbose, use_sim_time=use_sim_time)
        self._speed = 0.5
        self._yaw_mode = YawMode.PATH_FACING
        self._yaw_angle = None
        self._frame_id = "earth"
        self.current_behavior: BehaviorHandler = None
        self.led_pub = self.create_publisher(ColorRGBA, f"/{namespace}/leds/control", 10)
        
        # Publisher for landing position confirmation
        self.landing_ready_pub = self.create_publisher(String, '/formation/landing_ready', 10)
        
        # Follower-specific attributes
        self.follower_index = follower_index
        self.leader_state = None
        self.running = False
        self.altitude = 2.0
        
        # Load drone home positions from world_swarm.yaml
        self.drone_home_positions = load_drone_home_positions()
        
        # Ensure we have correct home positions - hardcode them if needed
        if not self.drone_home_positions or namespace not in self.drone_home_positions:
            print(f"[{namespace}] WARNING: Could not load drone home positions from file, using hardcoded defaults")
            # Create default positions based on the provided world_swarm.yaml data
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
            # This should never happen now due to hardcoded defaults
            print(f"[{namespace}] WARNING: Namespace not found in home positions, using default (0,0,0)")
            self.home_position = (0.0, 0.0, 0.2)
        
        # Formation definitions (same as leader)
        self.formations = [
            formation_line_5,
            formation_v_5,
            formation_diamond_5,
            formation_circular_orbit_5,
            formation_grid_5,
            formation_staggered_5
        ]
        
        # Communication
        callback_group = MutuallyExclusiveCallbackGroup()
        
        # Subscribe to leader state
        self.state_subscription = self.create_subscription(
            Float32MultiArray, 
            '/formation/leader_state',
            self.leader_state_callback,
            10,
            callback_group=callback_group
        )
        
        # Subscribe to commands
        self.command_subscription = self.create_subscription(
            String,
            '/formation/command',
            self.command_callback,
            10,
            callback_group=callback_group
        )
        
        # Position update thread
        self.update_thread = None
        
        # Initialize ORCA
        self.orca = ORCAFormation(
            radius=1.0,  # Collision radius
            time_horizon=2.0,  # Time horizon for collision avoidance
            neighbor_dist=10.0,  # Maximum distance to consider neighbors
            max_neighbors=10  # Maximum number of neighbors to consider
        )
        
        # Add subscribers for neighbor states
        self.neighbor_states = {}
        for ns in self.drone_namespaces:
            if ns != self.namespace:
                self.neighbor_states[ns] = None
                self.create_subscription(
                    Float32MultiArray,
                    f'/{ns}/state',
                    lambda msg, ns=ns: self.neighbor_state_callback(msg, ns),
                    10
                )

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
    
    def leader_state_callback(self, msg):
        """Process incoming leader state messages."""
        # Format: [x, y, z, angle, formation_index, formation_repeat, step_in_formation, returning_home_flag]
        if len(msg.data) >= 8:
            returning_home = (msg.data[7] > 0.5)  # Check if the leader is returning home
            
            self.leader_state = {
                'position': (msg.data[0], msg.data[1], msg.data[2]),
                'angle': msg.data[3],
                'formation': int(msg.data[4]),
                'repeat': int(msg.data[5]),
                'step': int(msg.data[6]),
                'returning_home': returning_home
            }
            
            # If leader is returning home, prepare to return to our own home position
            if returning_home and not hasattr(self, 'returning_home'):
                self.returning_home = True
                print(f"[{self.namespace}] Leader is returning home, preparing to return to own home position")
                self.return_to_home()
    
    def return_to_home(self):
        """Return to this drone's own home position."""
        home_x, home_y, home_z = self.home_position
        
        # Verify we're using the correct home position
        print(f"[{self.namespace}] DEBUG: Using home position coordinates: ({home_x:.2f}, {home_y:.2f}, {home_z:.2f})")
        
        # Add some altitude for safety while moving
        safe_altitude = max(home_z + 1.0, self.altitude)
        
        print(f"[{self.namespace}] Returning to home position: ({home_x:.2f}, {home_y:.2f}, {safe_altitude})")
        
        # Move to position above home
        self.go_to_position(home_x, home_y, safe_altitude, speed=0.8)
    
    def command_callback(self, msg):
        """Process commands from the leader."""
        command = msg.data
        
        if command == "takeoff":
            # Start takeoff procedure
            print(f"[{self.namespace}] Received takeoff command")
            self.do_behavior("takeoff", self.altitude, 0.7, False)
            self.change_led_colour((0, 255, 0))
            
            # Start position updates after takeoff
            def start_after_takeoff():
                while not self.goal_reached():
                    time.sleep(0.1)
                self.start()
            
            threading.Thread(target=start_after_takeoff).start()
            
        elif command == "prepare_landing":
            # First landing phase: Move to landing position but don't land yet
            print(f"[{self.namespace}] Preparing for landing - moving to landing position")
            self.stop()  # Stop following formation
            
            # Double-check home position is correctly loaded
            # This ensures we don't use default positions
            if self.namespace in self.drone_home_positions:
                # Use position from loaded file
                self.home_position = self.drone_home_positions[self.namespace]
                print(f"[{self.namespace}] Confirmed home position: {self.home_position}")
            else:
                # Use hardcoded defaults as a fallback
                default_positions = {
                    "drone0": (-1.0, 0.0, 0.2),
                    "drone1": (1.0, 0.0, 0.2),
                    "drone2": (0.0, 0.0, 0.2),
                    "drone3": (0.0, 1.0, 0.2),
                    "drone4": (0.0, -1.0, 0.2)
                }
                
                if self.namespace in default_positions:
                    self.home_position = default_positions[self.namespace]
                    print(f"[{self.namespace}] Using hardcoded home position: {self.home_position}")
                else:
                    print(f"[{self.namespace}] WARNING: Using center position as fallback")
                    self.home_position = (0.0, 0.0, 0.2)
            
            # Move to home position
            self.return_to_home()
            
            # Wait to reach landing position and notify
            landing_position_wait = threading.Thread(target=self._wait_for_landing_position)
            landing_position_wait.daemon = True
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
            
        elif command == "land":
            # Legacy land command (for backwards compatibility)
            print(f"[{self.namespace}] Received legacy land command")
            self.stop()
            
            # Move to landing position and then land
            self.return_to_home()
            
            # Wait briefly to give time to move toward landing position
            wait_time = 3.0  # seconds
            start_time = time.time()
            while time.time() - start_time < wait_time and not self.goal_reached():
                time.sleep(0.2)
            
            # Execute landing
            self._execute_landing()
    
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
    
    def neighbor_state_callback(self, msg, namespace):
        """Process neighbor state messages."""
        if len(msg.data) >= 8:
            self.neighbor_states[namespace] = {
                'position': Point(x=msg.data[0], y=msg.data[1], z=msg.data[2]),
                'velocity': Vector3(x=msg.data[3], y=msg.data[4], z=msg.data[5])
            }
    
    def update_formation_positions(self):
        """Update formation positions based on leader state."""
        if not self.leader_state:
            return
            
        # Get leader position and angle
        leader_pos = self.leader_state['position']
        leader_angle = self.leader_state['angle']
        
        # Calculate formation positions
        self.formation_positions = []
        for i in range(len(self.drone_namespaces)):
            if i == 0:  # Leader position
                self.formation_positions.append(Point(
                    x=leader_pos[0],
                    y=leader_pos[1],
                    z=leader_pos[2]
                ))
            else:
                # Calculate relative position in formation
                angle = leader_angle + (i - 1) * (2 * math.pi / (len(self.drone_namespaces) - 1))
                distance = 2.0  # Formation radius
                
                self.formation_positions.append(Point(
                    x=leader_pos[0] + distance * math.cos(angle),
                    y=leader_pos[1] + distance * math.sin(angle),
                    z=leader_pos[2]
                ))
                
    def calculate_position(self):
        """Calculate follower's position using ORCA with formation constraints."""
        if not self.leader_state:
            return None
            
        # Update formation positions
        self.update_formation_positions()
        
        # Get current position and velocity
        current_pos = self.get_position()
        current_vel = self.get_velocity()
        
        # Calculate desired velocity (towards formation position)
        formation_index = self.drone_namespaces.index(self.namespace)
        desired_pos = self.formation_positions[formation_index]
        
        # Calculate desired velocity
        desired_vel = Vector3(
            x=(desired_pos.x - current_pos.x) * self.speed,
            y=(desired_pos.y - current_pos.y) * self.speed,
            z=(desired_pos.z - current_pos.z) * self.speed
        )
        
        # Get neighbor information for ORCA
        neighbors = []
        for ns, state in self.neighbor_states.items():
            if state is not None:
                neighbors.append((state['position'], state['velocity']))
        
        # Compute new velocity using ORCA
        new_vel = self.orca.compute_velocity(
            current_pos,
            current_vel,
            desired_vel,
            neighbors,
            self.formation_positions
        )
        
        # Calculate new position based on new velocity
        dt = 0.1  # Time step
        new_pos = Point(
            x=current_pos.x + new_vel.x * dt,
            y=current_pos.y + new_vel.y * dt,
            z=current_pos.z + new_vel.z * dt
        )
        
        return new_pos
    
    def update_position(self):
        """Continuously update follower position based on leader state."""
        self.running = True
        last_position = None
        last_update_time = 0
        min_update_interval = 0.2  # Minimum time between position updates (seconds)
        position_threshold = 0.05  # 5cm threshold for position updates
        
        # Track how many times we've sent identical position commands
        repeat_count = 0
        max_repeats = 3  # Maximum number of repeated commands to same position
        last_command_position = None
        
        while self.running:
            if self.leader_state:
                position = self.calculate_position()
                current_time = time.time()
                
                if position:
                    # Check if we need to update position based on time and distance
                    update_position = False
                    
                    # Update if it's the first position or if minimum time has passed
                    if last_position is None or (current_time - last_update_time >= min_update_interval):
                        # Calculate distance to previous position
                        if last_position:
                            dx = position[0] - last_position[0]
                            dy = position[1] - last_position[1]
                            distance = math.sqrt(dx*dx + dy*dy)
                            
                            # Update if moved more than threshold or it's been a while
                            if distance > position_threshold or (current_time - last_update_time >= 0.5):
                                update_position = True
                        else:
                            # First position
                            update_position = True
                    
                    # Check if this is the same as the last command position
                    if last_command_position and update_position:
                        dx = position.x - last_command_position.x
                        dy = position.y - last_command_position.y
                        cmd_distance = math.sqrt(dx*dx + dy*dy)
                        
                        if cmd_distance < position_threshold:
                            repeat_count += 1
                            # Skip if we've sent too many repeat commands to the same position
                            if repeat_count > max_repeats:
                                update_position = False
                        else:
                            # Reset counter for new position
                            repeat_count = 0
                    
                    if update_position:
                        # Adjust speed based on distance to move
                        if last_position:
                            dx = position.x - last_position.x
                            dy = position.y - last_position.y
                            distance = math.sqrt(dx*dx + dy*dy)
                            
                            # Scale speed with distance, but keep it between 0.8 and 2.0
                            # More aggressive speed scaling for better response
                            speed = min(max(distance * 2.0, 0.8), 2.0)
                        else:
                            speed = 1.0
                            
                        self.go_to_position(position[0], position[1], position[2], speed=speed)
                        last_position = position
                        last_update_time = current_time
                        last_command_position = position
                    
                    # Quick check if we've reached position, with a short timeout
                    # This allows for faster reaction to new leader positions
                    if current_time - last_update_time < 0.5:  # Only wait if we recently sent a command
                        wait_start = time.time()
                        current_state = self.leader_state
                        
                        # Very short timeout - if position not reached quickly, we'll get a new one
                        while (not self.goal_reached() and 
                              self.running and 
                              self.leader_state == current_state and
                              time.time() - wait_start < 0.25):  # Shorter timeout for more responsive updates
                            time.sleep(0.02)  # Check more frequently
            
            # Very short sleep for faster response to new leader states
            time.sleep(0.02)
    
    def start(self):
        """Start the follower's position update thread."""
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

def run_leader(args):
    """Run the leader drone."""
    # Extract follower namespaces (all except the first one, which is the leader)
    follower_namespaces = args.namespaces[1:]
    
    # Create and initialize leader
    leader = LeaderDrone(
        args.namespaces[0], 
        follower_namespaces,
        args.config,
        verbose=args.verbose, 
        use_sim_time=args.use_sim_time
    )
    
    try:
        # Arm and set offboard mode
        if not leader.arm() or not leader.offboard():
            print("Failed to arm/offboard the leader!")
            leader.shutdown()
            return
        
        # Start the flight sequence
        leader.takeoff_all(height=2.0, speed=0.7)
        
        # Start trajectory execution
        leader.start()
        
        # Keep the main thread alive until trajectory execution is complete
        while leader.running:
            time.sleep(1.0)
        
        # Land all drones
        leader.land_all(speed=0.4)
        
    except KeyboardInterrupt:
        print("Leader operation interrupted by user")
    finally:
        leader.shutdown()

def run_follower(args, follower_index):
    """Run a follower drone."""
    namespace = args.namespaces[follower_index + 1]  # +1 because leader is at index 0
    
    # Create and initialize follower
    follower = FollowerDrone(
        namespace,
        follower_index,
        verbose=args.verbose,
        use_sim_time=args.use_sim_time
    )
    
    try:
        # Arm and set offboard mode
        if not follower.arm() or not follower.offboard():
            print(f"Failed to arm/offboard follower {namespace}!")
            follower.shutdown()
            return
        
        # The follower will start following the leader upon receiving the takeoff command
        print(f"Follower {namespace} initialized and waiting for leader commands")
        
        # Keep running until shutdown
        while rclpy.ok():
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        print(f"Follower {namespace} operation interrupted by user")
    finally:
        follower.stop()
        follower.shutdown()

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Decentralized Formation Flight with 5 drones")
    parser.add_argument('-n', '--namespaces', nargs='+',
                        default=['drone0', 'drone1', 'drone2', 'drone3', 'drone4'],
                        help='List of drone namespaces. First one is the leader, rest are followers.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Enable verbose output')
    parser.add_argument('-s', '--use_sim_time', action='store_true', default=True,
                        help='Use simulation time')
    parser.add_argument('-c', '--config', type=str, 
                        default='/home/ziyar/project_gazebo_ws/src/challenge_multi_drone/scenarios/scenario1_stage1.yaml',
                        help='Path to scenario configuration file')
    parser.add_argument('-r', '--role', type=str, choices=['leader', 'follower', 'all'],
                        default='all', help='Role to run (leader, follower, or all)')
    parser.add_argument('-i', '--index', type=int, default=0,
                        help='Index of the follower (0-3) when running a specific follower')
    
    args = parser.parse_args()
    
    # Initialize ROS
    rclpy.init()
    
    # Check if we have enough drones specified
    if len(args.namespaces) < 2:
        print("Error: At least 2 drone namespaces must be provided (1 leader and 1+ followers)")
        return
    
    # Run the specified roles
    if args.role == 'leader':
        run_leader(args)
    elif args.role == 'follower':
        if args.index < 0 or args.index >= len(args.namespaces) - 1:
            print(f"Error: Follower index must be between 0 and {len(args.namespaces) - 2}")
            return
        run_follower(args, args.index)
    else:  # 'all'
        # When running all roles in one process, we need to use MultiThreadedExecutor
        # This is mainly for testing purposes - in a real scenario, each drone would
        # run in its own process or separate machine.
        print("Running leader and all followers in one process (for testing)")
        
        # Create and initialize leader - Use drone2 (index 2) as leader
        leader_namespace = args.namespaces[2]
        follower_namespaces = [ns for i, ns in enumerate(args.namespaces) if i != 2]
        
        leader = LeaderDrone(
            leader_namespace,
            follower_namespaces,
            args.config,
            verbose=args.verbose,
            use_sim_time=args.use_sim_time
        )
        
        # Create followers - all drones except drone2
        followers = []
        
        # Map for drone index to formation index
        # If the physical drones are [drone0, drone1, drone2, drone3, drone4]
        # And the formation indices are [0, 1, 2, 3, 4] 
        # Where drone2 (index 2) is the leader
        # We map:
        # drone0 -> formation index 0
        # drone1 -> formation index 1
        # drone3 -> formation index 3
        # drone4 -> formation index 4
        
        for i, namespace in enumerate(args.namespaces):
            if i != 2:  # Skip leader (drone2)
                # Use actual drone index as the formation index
                follower = FollowerDrone(
                    namespace,
                    i,  # Use the actual drone index as the formation index
                    verbose=args.verbose,
                    use_sim_time=args.use_sim_time
                )
                followers.append(follower)
        
        # Set up executor
        executor = MultiThreadedExecutor()
        
        # Add nodes to executor
        executor.add_node(leader)
        for follower in followers:
            executor.add_node(follower)
            
        # Create a thread to spin the executor
        def spin_executor():
            executor.spin()
            
        executor_thread = threading.Thread(target=spin_executor, daemon=True)
        executor_thread.start()
        
        try:
            # Arm and set offboard for all drones
            print("Arming and setting offboard mode for all drones...")
            if not leader.arm() or not leader.offboard():
                print("Failed to arm/offboard the leader!")
                raise RuntimeError("Leader initialization failed")
            
            for i, follower in enumerate(followers):
                if not follower.arm() or not follower.offboard():
                    print(f"Failed to arm/offboard follower {follower.namespace}!")
                    raise RuntimeError(f"Follower {i} initialization failed")
            
            # Start flight sequence
            leader.takeoff_all(height=2.0, speed=0.7)
            
            # Start trajectory execution
            leader.start()
            
            # Keep the main thread alive until trajectory execution is complete
            while leader.running and rclpy.ok():
                if leader.mission_complete:
                    print("Mission complete detected, preparing to land...")
                    break
                time.sleep(1.0)
            
            # Land all drones
            print("Landing all drones...")
            leader.land_all(speed=0.4)
            
            # Wait for landing to complete
            print("Waiting for all drones to land...")
            time.sleep(10.0)  # Give more time for landing
            
        except KeyboardInterrupt:
            print("Operation interrupted by user")
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            # Clean shutdown
            leader.shutdown()
            for follower in followers:
                follower.stop()
                follower.shutdown()
    
    # Shutdown ROS
    rclpy.shutdown()

if __name__ == "__main__":
    main() 