#!/usr/bin/env python3
import argparse
import math
import sys
import random
import time
import json
import rclpy
import yaml
import os

from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import ColorRGBA, String
from as2_msgs.msg import YawMode, BehaviorStatus
from as2_python_api.drone_interface import DroneInterface
from as2_python_api.behavior_actions.behavior_handler import BehaviorHandler

# --- Formation functions for 5 drones ---
def formation_line_5():
    """
    Return offsets for a line formation for 5 drones.
    Evenly spaced along the x-axis.
    """
    d = 0.4  # distance between drones
    return [(-2 * d, 0.0), (-d, 0.0), (0.0, 0.0), (d, 0.0), (2 * d, 0.0)]

def formation_v_5():
    """
    Return offsets for a V formation for 5 drones.
    Middle drone at vertex, others form the V shape.
    """
    d = 0.4  # lateral offset
    return [(0.0, 0.0), (-d, d), (d, d), (-2 * d, 2 * d), (2 * d, 2 * d)]

def formation_diamond_5():
    """
    Return offsets for a diamond formation for 5 drones.
    One at the top, one in the middle, one at the bottom, and two at the sides.
    """
    d = 0.4
    return [(0.0, d), (-d, 0.0), (d, 0.0), (0.0, -d), (0.0, 0.0)]

def formation_circular_orbit_5():
    """
    Return offsets for a circular orbit formation for 5 drones.
    Drones are positioned evenly along a small circle relative to the center.
    """
    orbit_radius = 0.4
    offsets = []
    for i in range(5):
        angle = 2 * math.pi * i / 5
        offsets.append((orbit_radius * math.cos(angle), orbit_radius * math.sin(angle)))
    return offsets

def formation_grid_5():
    """
    Return offsets for a grid formation for 5 drones.
    Arranged in a cross pattern.
    """
    d = 0.4
    return [(0.0, 0.0), (d, 0.0), (-d, 0.0), (0.0, d), (0.0, -d)]

def formation_staggered_5():
    """
    Return offsets for a staggered formation for 5 drones.
    Creates a zigzag pattern.
    """
    d = 0.4
    return [(0.0, 0.0), (d, d/2), (2 * d, 0.0), (3 * d, d/2), (4 * d, 0.0)]

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

# --- FormationDancer class (unchanged) ---
class FormationDancer(DroneInterface):
    """
    A modified drone interface for formation flight.
    Uses direct go_to commands instead of a predefined path.
    """
    def __init__(self, namespace: str, verbose: bool = False, use_sim_time: bool = False):
        super().__init__(namespace, verbose=verbose, use_sim_time=use_sim_time)
        self._speed = 0.5
        self._yaw_mode = YawMode.PATH_FACING
        self._yaw_angle = None
        self._frame_id = "earth"
        self.current_behavior: BehaviorHandler = None
        self.led_pub = self.create_publisher(ColorRGBA, f"/{namespace}/leds/control", 10)

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

# --- Decentralized Leader Node ---
class LeaderFormationNode(FormationDancer):
    def __init__(self, namespace: str, config, verbose: bool = False, use_sim_time: bool = False):
        super().__init__(namespace, verbose, use_sim_time)
        self.state_pub = self.create_publisher(String, "formation_state", 10)
        self.config = config
        
        # Load stage parameters from config
        self.stage_center = config['stage1']['stage_center']
        diameter = config['stage1']['trajectory']['diameter']
        self.radius = diameter / 2.0
        self.altitude = 2.0  # Default flight altitude
        
        # Formation configuration
        self.total_formations = 6          # number of unique formations
        self.formation_repeats = 1         # each formation repeated (for demo, you can adjust)
        self.degrees_per_formation = 180   # half a revolution per formation
        self.angle_deg = 30                # step size in degrees
        self.angle_step = math.radians(self.angle_deg)
        self.steps_per_formation = int(self.degrees_per_formation / self.angle_deg)
        self.total_steps = self.total_formations * self.formation_repeats * self.steps_per_formation
        
        # Updated formation list
        self.formations = [
            formation_line_5,
            formation_v_5,
            formation_diamond_5,
            formation_circular_orbit_5,
            formation_grid_5,
            formation_staggered_5
        ]
        
    def run(self):
        # Arm, switch to offboard mode, and take off
        print("Leader: Arming and taking off...")
        if not self.arm() or not self.offboard():
            print("Leader: Failed to arm/offboard!")
            sys.exit(1)
        self.do_behavior("takeoff", self.altitude, 0.7, False)
        time.sleep(1)  # brief pause
        
        angle = 0.0
        current_formation = 0
        formation_repeat = 0
        step_in_formation = 0
        
        print("Leader: Starting formation loop.")
        for step in range(self.total_steps):
            # Compute base position on the circle
            base_x = self.stage_center[0] + self.radius * math.cos(angle)
            base_y = self.stage_center[1] + self.radius * math.sin(angle)
            
            # Get the current formation function and offsets
            formation_func = self.formations[current_formation]
            offsets = formation_func()
            # Leader uses its own index; here we assume leader is index 0.
            my_offset = offsets[0]
            rotated = rotate_offset(my_offset, angle)
            pos_x = base_x + rotated[0]
            pos_y = base_y + rotated[1]
            
            # Publish state message (as JSON)
            state = {
                "base_x": base_x,
                "base_y": base_y,
                "angle": angle,
                "current_formation": current_formation,
                "step": step  # can be used by followers to detect update
            }
            msg = String()
            msg.data = json.dumps(state)
            self.state_pub.publish(msg)
            print(f"Leader: Published state {state}")
            
            # Command leader's own movement
            self.go_to_position(pos_x, pos_y, self.altitude, speed=1.0)
            print(f"Leader: Moving to ({pos_x:.2f}, {pos_y:.2f}, {self.altitude})")
            
            # Wait until leader reaches the goal
            while not self.goal_reached():
                rclpy.spin_once(self, timeout_sec=0.05)
            
            # Increment counters and update angle
            angle += self.angle_step
            step_in_formation += 1
            if step_in_formation >= self.steps_per_formation:
                step_in_formation = 0
                formation_repeat += 1
                if formation_repeat >= self.formation_repeats:
                    formation_repeat = 0
                    current_formation = (current_formation + 1) % self.total_formations
                    print(f"\nLeader: Switching to formation {current_formation}\n")
            
        # Land after trajectory
        print("Leader: Landing...")
        self.do_behavior("land", 0.4, False)
        time.sleep(1)
        self.shutdown()

# --- Decentralized Follower Node ---
class FollowerFormationNode(FormationDancer):
    def __init__(self, namespace: str, config, drone_id: int, verbose: bool = False, use_sim_time: bool = False):
        super().__init__(namespace, verbose, use_sim_time)
        self.drone_id = drone_id   # index (0 to 4) in the formation offsets
        self.latest_state = None
        self.state_sub = self.create_subscription(String, "formation_state", self.state_callback, 10)
        self.config = config
        
        # Load stage parameters (should match leader's values)
        self.stage_center = config['stage1']['stage_center']
        diameter = config['stage1']['trajectory']['diameter']
        self.radius = diameter / 2.0
        self.altitude = 2.0
        
        # Formation functions list (must match leader's)
        self.formations = [
            formation_line_5,
            formation_v_5,
            formation_diamond_5,
            formation_circular_orbit_5,
            formation_grid_5,
            formation_staggered_5
        ]
        
        self.last_processed_step = -1  # track the last leader update processed

    def state_callback(self, msg: String):
        try:
            state = json.loads(msg.data)
            self.latest_state = state
        except Exception as e:
            print(f"Follower {self.drone_id}: Error parsing state message: {e}")

    def run(self):
        # Arm, switch to offboard mode, and take off
        print(f"Follower {self.drone_id}: Arming and taking off...")
        if not self.arm() or not self.offboard():
            print(f"Follower {self.drone_id}: Failed to arm/offboard!")
            sys.exit(1)
        self.do_behavior("takeoff", self.altitude, 0.7, False)
        time.sleep(1)
        
        print(f"Follower {self.drone_id}: Waiting for leader state updates...")
        # Main loop: when a new state is received, compute and move.
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_state is None:
                continue
            # Process only new updates based on the leader's step counter
            if self.latest_state["step"] <= self.last_processed_step:
                continue

            # Mark the update as processed
            self.last_processed_step = self.latest_state["step"]
            
            # Read state values
            base_x = self.latest_state["base_x"]
            base_y = self.latest_state["base_y"]
            angle = self.latest_state["angle"]
            formation_index = self.latest_state["current_formation"]
            
            # Retrieve formation function and compute offsets
            formation_func = self.formations[formation_index]
            offsets = formation_func()
            # Check that the drone_id is within range
            if self.drone_id < 0 or self.drone_id >= len(offsets):
                print(f"Follower {self.drone_id}: Invalid drone ID for current formation.")
                continue
            my_offset = offsets[self.drone_id]
            rotated = rotate_offset(my_offset, angle)
            pos_x = base_x + rotated[0]
            pos_y = base_y + rotated[1]
            
            print(f"Follower {self.drone_id}: Received state step {self.latest_state['step']}, "
                  f"moving to ({pos_x:.2f}, {pos_y:.2f}, {self.altitude}).")
                  
            # Command movement for this follower
            self.go_to_position(pos_x, pos_y, self.altitude, speed=1.0)
            # Wait until goal reached before processing the next update
            while not self.goal_reached():
                rclpy.spin_once(self, timeout_sec=0.1)
                
        # (You could add landing logic here if desired)
        self.do_behavior("land", 0.4, False)
        time.sleep(1)
        self.shutdown()

# --- Main function ---
def main():
    parser = argparse.ArgumentParser(description="Decentralized Formation Flight Mission with 5 drones (Leader–Follower)")
    parser.add_argument('--role', choices=['leader', 'follower'], required=True,
                        help='Role of this drone in the formation (leader or follower)')
    parser.add_argument('--id', type=int, default=0,
                        help='Drone ID (index among formation positions). For followers only, typically 0-4.')
    parser.add_argument('-n', '--namespace', type=str, default='drone0',
                        help='Drone namespace')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Enable verbose output')
    parser.add_argument('-s', '--use_sim_time', action='store_true', default=True,
                        help='Use simulation time')
    parser.add_argument('-c', '--config', type=str, 
                        default='/home/ziyar/project_gazebo_ws/src/challenge_multi_drone/scenarios/scenario1_stage1.yaml',
                        help='Path to scenario configuration file')
    args = parser.parse_args()

    config = load_scenario_config(args.config)
    if not config:
        print(f"Failed to load configuration from {args.config}")
        sys.exit(1)
        
    rclpy.init()
    
    # Create and run the node based on role.
    if args.role == "leader":
        node = LeaderFormationNode(args.namespace, config, verbose=args.verbose, use_sim_time=args.use_sim_time)
    else:
        node = FollowerFormationNode(args.namespace, config, drone_id=args.id, verbose=args.verbose, use_sim_time=args.use_sim_time)
    
    try:
        node.run()
    except KeyboardInterrupt:
        print("Shutting down due to keyboard interrupt...")
    finally:
        node.shutdown()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
