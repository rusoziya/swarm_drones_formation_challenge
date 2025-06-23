#!/usr/bin/env python3
import argparse
import math
import sys
import random
import rclpy
import yaml
import os
from rclpy.executors import MultiThreadedExecutor
from as2_msgs.msg import YawMode, BehaviorStatus
from as2_python_api.drone_interface import DroneInterface
from as2_python_api.behavior_actions.behavior_handler import BehaviorHandler
from std_msgs.msg import ColorRGBA

# --- Formation functions for 5 drones ---
def formation_line_5():
    """
    Return offsets for a line formation for 5 drones.
    Evenly spaced along the x-axis.
    """
    d = 0.4  # distance between drones
    return [(-2*d, 0.0), (-d, 0.0), (0.0, 0.0), (d, 0.0), (2*d, 0.0)]

def formation_v_5():
    """
    Return offsets for a V formation for 5 drones.
    Middle drone at vertex, others form the V shape.
    """
    d = 0.4  # lateral offset
    return [(0.0, 0.0), (-d, d), (d, d), (-2*d, 2*d), (2*d, 2*d)]

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
    return [(0.0, 0.0), (d, d/2), (2*d, 0.0), (3*d, d/2), (4*d, 0.0)]

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


# --- FormationDancer class ---
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

# --- FormationSwarmConductor class ---
class FormationSwarmConductor:
    def __init__(self, drones_ns: list, verbose: bool = False, use_sim_time: bool = False):
        self.drones: list[FormationDancer] = []
        for ns in drones_ns:
            self.drones.append(FormationDancer(ns, verbose, use_sim_time))

    def shutdown(self):
        for d in self.drones:
            d.shutdown()

    def wait_all(self):
        # Poll the goal status to check if all drones have finished their behavior.
        all_done = False
        while not all_done:
            import time
            time.sleep(0.01)
            all_done = all(d.goal_reached() for d in self.drones)
        print("All drones reached goal.")
    
    def get_ready(self) -> bool:
        """Arm and set offboard mode for all drones."""
        success = True
        for d in self.drones:
            success_arm = d.arm()
            success_offboard = d.offboard()
            success = success and success_arm and success_offboard
        return success

    def takeoff(self, height=2.0, speed=0.7):
        for d in self.drones:
            d.do_behavior("takeoff", height, speed, False)
            d.change_led_colour((0, 255, 0))
        self.wait_all()

    def land(self, speed=0.4):
        for d in self.drones:
            d.do_behavior("land", speed, False)
        self.wait_all()

    def move_swarm(self, positions, altitude=2.0, speed=1.0):
        """
        positions: list of (x, y) tuples for each drone.
        Commands each drone to move to its corresponding (x, y, altitude) point.
        """
        for d, (px, py) in zip(self.drones, positions):
            print(f"Moving {d.namespace} to ({px:.2f}, {py:.2f}, {altitude})")
            d.go_to_position(px, py, altitude, speed=speed)
        print("Waiting for all drones to reach goal...")
        self.wait_all()

# --- Main orchestration ---
def main():
    parser = argparse.ArgumentParser(description="Stage 1: Formation Flight Mission with 5 drones")
    parser.add_argument('-n', '--namespaces', nargs='+',
                        default=['drone0', 'drone1', 'drone2', 'drone3', 'drone4'],
                        help='List of drone namespaces')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Enable verbose output')
    parser.add_argument('-s', '--use_sim_time', action='store_true', default=True,
                        help='Use simulation time')
    parser.add_argument('-c', '--config', type=str, 
                        default='/home/ziyar/project_gazebo_ws/src/challenge_multi_drone/scenarios/scenario1_stage1.yaml',
                        help='Path to scenario configuration file')
    args = parser.parse_args()

    # Load scenario configuration
    config = load_scenario_config(args.config)
    if not config:
        print(f"Failed to load configuration from {args.config}")
        sys.exit(1)

    # Stage parameters from scenario config
    stage_center = config['stage1']['stage_center']
    diameter = config['stage1']['trajectory']['diameter']
    radius = diameter / 2.0
    altitude = 2.0  # Default flight altitude

    rclpy.init()
    swarm = FormationSwarmConductor(args.namespaces, verbose=args.verbose, use_sim_time=args.use_sim_time)

    # Configuration for formations
    total_formations = 6          # Number of unique formations
    formation_repeats = 1         # Each formation repeats 6 times
    degrees_per_formation = 180   # Each formation lasts half a revolution (180°)
    
    # Calculate required parameters for movement
    angle_deg = 30                 # Base angle step for smooth movement (smaller for more granularity)
    angle_step = math.radians(angle_deg)
    steps_per_formation = int(degrees_per_formation / angle_deg)  # Steps to complete half revolution
    
    # Total steps needed for full flight plan (all formations × repeats × steps per formation)
    total_steps = total_formations * formation_repeats * steps_per_formation

    # Updated formation list with formations for 5 drones
    formations = [
        formation_line_5,
        formation_v_5,
        formation_diamond_5,
        formation_circular_orbit_5,
        formation_grid_5,
        formation_staggered_5
    ]

    # Takeoff procedure
    print("Arming and taking off...")
    if not swarm.get_ready():
        print("Failed to arm/offboard!")
        sys.exit(1)
    swarm.takeoff(height=altitude, speed=0.7)

    # Execute circular trajectory with periodic formation changes
    angle = 0.0
    current_formation = 0     # Index of current formation pattern
    formation_repeat = 0      # Counter for repeating each formation
    step_in_formation = 0     # Steps completed in current formation

    # Display formation plan
    print(f"Flight plan: {total_formations} formations, each repeated {formation_repeats} times")
    print(f"Each formation lasts {degrees_per_formation}° (half a revolution)")
    print(f"Total flight: {total_steps * angle_deg}° ({total_steps * angle_deg / 360} revolutions)")

    for step in range(total_steps):
        # Determine if it's time to switch formation
        if step_in_formation >= steps_per_formation:
            step_in_formation = 0
            formation_repeat += 1
            
            # Move to next formation after all repeats are done
            if formation_repeat >= formation_repeats:
                formation_repeat = 0
                current_formation = (current_formation + 1) % total_formations
                print(f"\n*** Switching to formation: {current_formation} ***\n")
            else:
                print(f"\n--- Repeating formation {current_formation} (repeat {formation_repeat+1}/{formation_repeats}) ---\n")

        # Get the current formation function
        formation_func = formations[current_formation]
        offsets = formation_func()

        # Compute base (center) position on the circle
        base_x = stage_center[0] + radius * math.cos(angle)
        base_y = stage_center[1] + radius * math.sin(angle)

        # Compute final positions for each drone by rotating offsets by the current angle
        positions = []
        for (dx, dy) in offsets:
            rotated = rotate_offset((dx, dy), angle)
            pos_x = base_x + rotated[0]
            pos_y = base_y + rotated[1]
            positions.append((pos_x, pos_y))

        # Display informational message 
        print(f"Step {step}/{total_steps}: Formation {current_formation} (repeat {formation_repeat+1}/{formation_repeats})")
        print(f"Base position = ({base_x:.2f}, {base_y:.2f}), Angle = {math.degrees(angle):.1f}°")
        print(f"Final positions: {[(round(px, 2), round(py, 2)) for px, py in positions]}")

        # Move the swarm
        swarm.move_swarm(positions, altitude=altitude, speed=1.0)
        
        # Increment counters and angles
        angle += angle_step
        step_in_formation += 1

    # Land after trajectory
    print("Landing swarm...")
    swarm.land()
    swarm.shutdown()

if __name__ == "__main__":
    main() 