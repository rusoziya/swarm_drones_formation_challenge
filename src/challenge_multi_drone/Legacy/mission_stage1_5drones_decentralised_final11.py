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
from geometry_msgs.msg import PoseStamped

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
        
        # Formation names for debugging
        self.formation_names = [
            "Line",
            "V",
            "Diamond",
            "Circular Orbit",
            "Grid",
            "Staggered"
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
        print(f"Flight plan: {self.total_formations} formations, each repeated {self.formation_repeats} times")
        print(f"Each formation lasts {self.degrees_per_formation}° (half a revolution)")
        print(f"Circular orbit uses {self.circular_orbit_steps} steps per 180°, other formations use {self.steps_per_formation} steps")
        print(f"Total flight: {self.degrees_per_formation * self.total_formations * self.formation_repeats}° ({self.total_formations * self.formation_repeats * self.degrees_per_formation / 360} revolutions)")
        print(f"Total steps: {total_steps}")
        print(f"Formation sequence: {' -> '.join(self.formation_names)}")
        
        # Start at step 0
        step = 0
        
        while self.running and step < total_steps:
            # Determine if it's time to switch formation
            if (self.current_formation == 3 and self.step_in_formation >= self.circular_orbit_steps) or \
               (self.current_formation != 3 and self.step_in_formation >= self.steps_per_formation):
                self.step_in_formation = 0
                self.formation_repeat += 1
                
                # Move to next formation after all repeats are done
                if self.formation_repeat >= self.formation_repeats:
                    self.formation_repeat = 0
                    self.current_formation = (self.current_formation + 1) % self.total_formations
                    current_formation_name = self.formation_names[self.current_formation]
                    print(f"\n*** Switching to formation: {self.current_formation} - {current_formation_name} ***\n")
                    
                    # Brief pause between formations
                    time.sleep(self.formation_switch_pause)
                else:
                    current_formation_name = self.formation_names[self.current_formation]
                    print(f"\n--- Repeating formation {self.current_formation} - {current_formation_name} (repeat {self.formation_repeat+1}/{self.formation_repeats}) ---\n")
            
            # Compute base (center) position on the circle
            # Use different angle step for circular orbit
            if self.current_formation == 3:  # Circular orbit
                current_angle_step = self.circular_orbit_angle_step
            else:
                current_angle_step = self.angle_step
            
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
            
            # Display informational message with formation name
            current_formation_name = self.formation_names[self.current_formation]
            max_steps = self.circular_orbit_steps if self.current_formation == 3 else self.steps_per_formation
            print(f"Step {step}/{total_steps}: Formation '{current_formation_name}' (repeat {self.formation_repeat+1}/{self.formation_repeats})")
            print(f"Base position = ({base_x:.2f}, {base_y:.2f}), Angle = {math.degrees(self.angle):.1f}°")
            print(f"Leader position: ({pos_x:.2f}, {pos_y:.2f}, {self.altitude})")
            
            # For circular orbit, adjust parameters for better formation
            move_speed = self.movement_speed
            if self.current_formation == 3:  # Circular orbit (index 3)
                # Higher speed for circular orbit for better formation keeping
                move_speed = 1.2  # Faster movement for circular orbit
                
                # Broadcast state immediately with new step information
                self.broadcast_state()
            
            # Move leader to position
            self.go_to_position(pos_x, pos_y, self.altitude, speed=move_speed)
            
            # Wait until leader is close to target position, with a timeout
            wait_start = time.time()
            
            # Set consistent timing for each waypoint in circular orbit
            if self.current_formation == 3:  # Circular orbit (index 3)
                max_wait_time = 0.8  # Shorter wait time for circular orbit (reduced from 1.5)
                goal_check_interval = 0.02  # More frequent checks
                broadcast_interval = 0.02  # More frequent broadcasts
            else:
                max_wait_time = 3.0  # Normal timing for other formations
                goal_check_interval = 0.05
                broadcast_interval = 0.05
            
            last_broadcast = 0
            # More frequent broadcast for circular orbit to ensure followers get updates
            broadcast_count = 0
            while not self.goal_reached() and self.running and (time.time() - wait_start < max_wait_time):
                # Broadcast state at regular intervals
                current_time = time.time()
                if current_time - last_broadcast >= broadcast_interval:
                    self.broadcast_state()
                    last_broadcast = current_time
                    broadcast_count += 1
                time.sleep(goal_check_interval)
            
            # Brief stabilization wait - ensure consistent timing for circular orbit
            if self.current_formation == 0:  # Line formation
                wait_time = self.stabilization_time * 1.5  # Slightly longer for line formation
            elif self.current_formation == 3:  # Circular orbit (index 3)
                # Shorter wait time for circular orbit for smoother movement
                elapsed = time.time() - wait_start
                wait_time = max(0.02, 0.2 - elapsed)  # Very short wait for orbit (reduced from 0.5)
                
                # Ensure sufficient broadcasts have been made
                if broadcast_count < 5 and self.running:
                    # Force at least 5 broadcasts to ensure followers get the updates (reduced from 10)
                    for _ in range(max(0, 5 - broadcast_count)):
                        self.broadcast_state()
                        time.sleep(0.01)
            else:
                wait_time = self.stabilization_time
                
            # Broadcast state during the stabilization wait
            broadcast_interval = 0.01 if self.current_formation == 3 else 0.05  # More frequent for orbit
            wait_start = time.time()
            while time.time() - wait_start < wait_time and self.running:
                self.broadcast_state()
                time.sleep(broadcast_interval)
            
            # Increment counters and angles
            # Use different angle step for circular orbit formation
            if self.current_formation == 3:  # Circular orbit
                self.angle += self.circular_orbit_angle_step
            else:
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
        
        # Rotation tracking for continuous circular orbit
        self.current_rotation_angle = 0.0
        self.last_rotation_update = time.time()
        self.rotation_speed = 1.5  # radians per second - increased for 18 steps per 180 degrees
        
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
        
        # Formation names for debugging
        self.formation_names = [
            "Line",
            "V",
            "Diamond",
            "Circular Orbit",
            "Grid",
            "Staggered"
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
            
            # Check if formation has changed
            new_formation = int(msg.data[4])
            old_formation = -1 if self.leader_state is None else self.leader_state.get('formation', -1)
            formation_changed = new_formation != old_formation
            
            # Debug output for formation changes
            if formation_changed:
                formation_name = self.formation_names[new_formation] if 0 <= new_formation < len(self.formation_names) else "Unknown"
                print(f"[{self.namespace}] Switching to formation: {new_formation} - {formation_name}")
                
                # If we're entering circular orbit, increase rotation speed
                if new_formation == 3:  # Circular orbit
                    self.rotation_speed = 1.5  # Faster rotation for 18 steps
                else:
                    self.rotation_speed = 0.5  # Normal rotation speed for other formations
                
                # If we're entering or leaving circular orbit, reset the rotation tracking
                # to prevent odd jumps, but keep it continuous otherwise
                if new_formation == 3 or old_formation == 3:
                    self.current_rotation_angle = 0.0
                    self.last_rotation_update = time.time()
            
            # Always update last rotation time to prevent sudden jumps if we were paused
            if new_formation == 3:  # If in circular orbit, keep rotation timing current
                current_time = time.time()
                if current_time - self.last_rotation_update > 0.5:  # If too much time passed
                    self.last_rotation_update = current_time  # Reset timer without changing angle
            
            self.leader_state = {
                'position': (msg.data[0], msg.data[1], msg.data[2]),
                'angle': msg.data[3],
                'formation': new_formation,
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
    
    def calculate_position(self):
        """Calculate follower's position based on leader's state."""
        if not self.leader_state:
            return None
        
        # If the leader is returning home and we're also returning home,
        # we should return to our own home position instead of following leader
        if self.leader_state.get('returning_home', False) and hasattr(self, 'returning_home') and self.returning_home:
            # Use our own home position
            home_x, home_y, home_z = self.home_position
            safe_altitude = max(home_z + 1.0, self.altitude)  # Stay at safe altitude until landing
            return (home_x, home_y, safe_altitude)
        
        # Normal formation position calculation
        # Extract leader state
        leader_pos = self.leader_state['position']
        angle = self.leader_state['angle']
        formation_idx = self.leader_state['formation']
        
        # Get formation offsets
        formation_func = self.formations[formation_idx]
        offsets = formation_func()
        
        # For circular orbit formation, apply continuous anticlockwise rotation
        if formation_idx == 3:  # Circular orbit formation (index 3)
            # Update rotation angle continuously
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
            
            # Use current_rotation_angle directly - always ensures anticlockwise movement
            rotation_angle = self.current_rotation_angle
            
            # Create rotated offsets for continuous orbital movement
            rotated_offsets = []
            for offset in offsets:
                # Apply rotation to each position
                x, y = offset
                cos_rot = math.cos(rotation_angle)
                sin_rot = math.sin(rotation_angle)
                new_x = x * cos_rot - y * sin_rot
                new_y = x * sin_rot + y * cos_rot
                rotated_offsets.append((new_x, new_y))
                
            # Use rotated positions
            offsets = rotated_offsets
            
            # Debug rotation (uncomment if needed)
            # print(f"[{self.namespace}] Rotation: {math.degrees(self.current_rotation_angle):.1f}°")
        
        # Leader is at index 2 (drone2), followers are at other indices
        leader_offset = offsets[2]  # drone2's offset in formation
        follower_offset = offsets[self.follower_index]
        
        # Calculate base position (center of formation)
        base_x = leader_pos[0] - leader_offset[0]  # Adjust for leader offset
        base_y = leader_pos[1] - leader_offset[1]
        
        # Apply follower offset with rotation - only for formations other than circular orbit
        # For circular orbit, we've already applied the rotation above
        if formation_idx != 3:
            rotated = rotate_offset(follower_offset, angle)
        else:
            # For circular orbit, don't apply additional rotation based on leader's angle
            # This decouples the orbit rotation from the leader's waypoint movement
            rotated = follower_offset
            
        pos_x = base_x + rotated[0]
        pos_y = base_y + rotated[1]
        
        return (pos_x, pos_y, leader_pos[2])
    
    def update_position(self):
        """Continuously update follower position based on leader state."""
        self.running = True
        last_position = None
        last_update_time = 0
        min_update_interval = 0.2  # Minimum time between position updates (seconds)
        position_threshold = 0.05  # 5cm threshold for position updates
        
        # Shorter update interval specifically for circular orbit formation
        circular_orbit_update_interval = 0.03  # Very frequent updates for circular orbit
        
        # Track how many times we've sent identical position commands
        repeat_count = 0
        max_repeats = 3  # Maximum number of repeated commands to same position
        last_command_position = None
        
        # For tracking waypoint timing
        waypoint_start_time = None
        last_step = -1
        
        while self.running:
            if self.leader_state:
                # Always update rotation angle for circular orbit, even if we don't move
                # This keeps rotation continuous even when not updating position
                current_formation = self.leader_state.get('formation', 0)
                if current_formation == 3:  # Circular orbit
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
                
                position = self.calculate_position()
                current_time = time.time()
                
                # Extract current formation and step
                current_step = self.leader_state.get('step', 0)
                is_circular_orbit = (current_formation == 3)  # Index 3 is circular orbit
                
                # For circular orbit, force frequent position updates for smooth motion
                if is_circular_orbit:
                    # Update position more frequently regardless of other conditions
                    update_position = (current_time - last_update_time >= circular_orbit_update_interval)
                    
                    # Always update at step changes
                    if current_step != last_step:
                        waypoint_start_time = current_time
                        last_step = current_step
                        update_position = True
                else:
                    # Normal update logic for other formations
                    update_position = False
                    
                    # Detect change in waypoint step
                    if current_step != last_step:
                        waypoint_start_time = current_time
                        last_step = current_step
                        update_position = True
                        
                    # Otherwise, determine if update needed based on time and distance
                    elif last_position is None or (current_time - last_update_time >= min_update_interval):
                        # Calculate distance to previous position
                        if last_position:
                            dx = position[0] - last_position[0]
                            dy = position[1] - last_position[1]
                            distance = math.sqrt(dx*dx + dy*dy)
                            
                            # For other formations, use normal distance-based updates
                            if distance > position_threshold or (current_time - last_update_time >= 0.5):
                                update_position = True
                        else:
                            # First position
                            update_position = True
                
                # For non-circular orbit, check for repeated commands
                if last_command_position and update_position and not is_circular_orbit:
                    dx = position[0] - last_command_position[0]
                    dy = position[1] - last_command_position[1]
                    cmd_distance = math.sqrt(dx*dx + dy*dy)
                    
                    if cmd_distance < position_threshold:
                        repeat_count += 1
                        # Skip if we've sent too many repeat commands to the same position
                        if repeat_count > max_repeats:
                            update_position = False
                    else:
                        # Reset counter for new position
                        repeat_count = 0
                
                if position and update_position:
                    # Adjust speed based on distance to move and formation type
                    if last_position:
                        dx = position[0] - last_position[0]
                        dy = position[1] - last_position[1]
                        distance = math.sqrt(dx*dx + dy*dy)
                        
                        # Base speed calculation
                        base_speed = min(max(distance * 2.0, 0.8), 2.0)
                        
                        # For circular orbit, use higher speed and continuous motion
                        if is_circular_orbit:
                            # For continuous motion, use higher minimum speed
                            min_speed = 1.5
                            max_speed = 3.5  # Increased for smoother orbit
                            
                            # Adjust based on distance - further means faster to catch up
                            required_speed = min(max(distance * 3.0, min_speed), max_speed)
                            
                            # Use required speed directly for orbital motion
                            speed = required_speed
                        else:
                            # For waypoint formations, use the base speed
                            speed = base_speed
                    else:
                        # First position - use higher speed for circular orbit
                        speed = 2.0 if is_circular_orbit else 1.0
                        
                    # Execute movement command with adjusted speed
                    self.go_to_position(position[0], position[1], position[2], speed=speed)
                    last_position = position
                    last_update_time = current_time
                    last_command_position = position
                
                # For circular orbit, use very short goal checks for continuous motion
                if update_position and current_time - last_update_time < 0.5:
                    wait_start = time.time()
                    
                    # Minimal wait time for circular orbit
                    wait_timeout = 0.02 if is_circular_orbit else 0.25
                    
                    # Check if goal reached - brief timeout to keep motion continuous
                    while (not self.goal_reached() and 
                          self.running and
                          time.time() - wait_start < wait_timeout):
                        time.sleep(0.01)  # Very fast checks for circular orbit
            
            # Sleep time between updates - very short for circular orbit
            sleep_time = 0.005 if (self.leader_state and self.leader_state.get('formation') == 3) else 0.02
            time.sleep(sleep_time)
    
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
        
        # Set up executor with increased number of threads
        executor = MultiThreadedExecutor(num_threads=8)
        
        # Add a small delay between node initializations to avoid race conditions
        time.sleep(1.0)
        
        # Create leader first
        leader = LeaderDrone(
            leader_namespace,
            follower_namespaces,
            args.config,
            verbose=args.verbose,
            use_sim_time=args.use_sim_time
        )
        
        # Add leader to executor and spin briefly to initialize
        executor.add_node(leader)
        executor.spin_once(timeout_sec=1.0)
        time.sleep(1.0)
        
        # Create followers - all drones except drone2
        followers = []
        
        # Create each follower one at a time with proper delays
        for i, namespace in enumerate(args.namespaces):
            if i != 2:  # Skip leader (drone2)
                # Use actual drone index as the formation index
                print(f"Initializing follower {namespace}...")
                
                # Add a small delay between followers
                time.sleep(0.5)
                
                follower = FollowerDrone(
                    namespace,
                    i,  # Use the actual drone index as the formation index
                    verbose=args.verbose,
                    use_sim_time=args.use_sim_time
                )
                followers.append(follower)
                
                # Add to executor and spin once to initialize
                executor.add_node(follower)
                executor.spin_once(timeout_sec=0.5)
        
        # Create a thread to spin the executor
        def spin_executor():
            try:
                executor.spin()
            except Exception as e:
                print(f"Executor exception: {e}")
            
        executor_thread = threading.Thread(target=spin_executor, daemon=True)
        executor_thread.start()
        
        # Add a small delay to ensure executor is running
        time.sleep(1.0)
        
        try:
            # Arm and set offboard for leader first
            print("Arming and setting offboard mode for leader...")
            if not leader.arm() or not leader.offboard():
                print("Failed to arm/offboard the leader!")
                raise RuntimeError("Leader initialization failed")
            
            # Add delay after leader initialization
            time.sleep(1.0)
            
            # Then arm and set offboard for followers one by one
            print("Arming and setting offboard mode for followers...")
            for i, follower in enumerate(followers):
                print(f"Arming follower {follower.namespace}...")
                if not follower.arm() or not follower.offboard():
                    print(f"Failed to arm/offboard follower {follower.namespace}!")
                    raise RuntimeError(f"Follower {i} initialization failed")
                time.sleep(0.5)  # Add delay between followers
            
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