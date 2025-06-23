#!/usr/bin/env python3
import argparse
import math
import sys
import random
import rclpy
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
    parser = argparse.ArgumentParser(description="Stage 1: Formation Flight Mission with 5 drones (Starting from middle)")
    parser.add_argument('-n', '--namespaces', nargs='+',
                        default=['drone0', 'drone1', 'drone2', 'drone3', 'drone4'],
                        help='List of drone namespaces')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Enable verbose output')
    parser.add_argument('-s', '--use_sim_time', action='store_true', default=True,
                        help='Use simulation time')
    parser.add_argument('-a', '--start-angle', type=float, default=180.0,
                        help='Starting angle in degrees (0-360, default: 180 for middle)')
    parser.add_argument('-f', '--start-formation', type=int, default=3,
                        help='Starting formation index (0-5, default: 3)')
    args = parser.parse_args()

    rclpy.init()
    swarm = FormationSwarmConductor(args.namespaces, verbose=args.verbose, use_sim_time=args.use_sim_time)

    # Stage parameters from scenario
    stage_center = (-6.0, 6.0)       # Stage center for Stage 1
    diameter = 3.0                   # Trajectory diameter => radius = 1.5
    radius = diameter / 2.0
    altitude = 2.0                   # Flight altitude
    angle_deg = 10 
    angle_step = math.radians(angle_deg)    # 10° per iteration

    # Updated formation list with new formations for 5 drones
    formations = [
        formation_line_5,
        formation_v_5,
        formation_diamond_5,
        formation_circular_orbit_5,
        formation_grid_5,
        formation_staggered_5
    ]
    
    # Start from the middle of the scenario (or custom starting point)
    current_formation_idx = args.start_formation
    start_angle_deg = args.start_angle  # Starts from the opposite side (halfway around the circle)
    steps_per_formation = 12         # Switch formation every 12 steps

    # Calculate how many steps to skip
    starting_step = int(start_angle_deg / angle_deg)
    total_steps = int(360 / angle_deg)  # Full circle over n steps
    remaining_steps = total_steps - starting_step

    # Takeoff procedure
    print("Arming and taking off...")
    if not swarm.get_ready():
        print("Failed to arm/offboard!")
        sys.exit(1)
    swarm.takeoff(height=altitude, speed=0.7)

    # Start from the middle of the scenario
    angle = math.radians(start_angle_deg)
    print(f"Starting from angle {start_angle_deg}° (step {starting_step} of {total_steps})")
    print(f"Starting with formation {current_formation_idx}: {formations[current_formation_idx].__name__}")

    # Execute remaining part of the circular trajectory with periodic formation changes
    for step in range(starting_step, total_steps):
        # Switch formation every steps_per_formation iterations, but accounting for our starting point
        adjusted_step = step - starting_step  # Steps since we started
        if adjusted_step > 0 and adjusted_step % steps_per_formation == 0:
            current_formation_idx = (current_formation_idx + 1) % len(formations)
            print(f"Switching formation to index {current_formation_idx}: {formations[current_formation_idx].__name__}")

        # Compute base (center) position on the circle
        base_x = stage_center[0] + radius * math.cos(angle)
        base_y = stage_center[1] + radius * math.sin(angle)

        # Get the current formation offsets (list of (dx, dy) for 5 drones)
        formation_func = formations[current_formation_idx]
        offsets = formation_func()

        # Compute final positions for each drone by rotating offsets by the current angle
        positions = []
        for (dx, dy) in offsets:
            rotated = rotate_offset((dx, dy), angle)
            pos_x = base_x + rotated[0]
            pos_y = base_y + rotated[1]
            positions.append((pos_x, pos_y))

        print(f"Step {step}/{total_steps-1}: Base position = ({base_x:.2f}, {base_y:.2f}), Formation: {formation_func.__name__}")
        print(f"Final positions: {positions}")

        swarm.move_swarm(positions, altitude=altitude, speed=1.0)
        angle += angle_step

    # Land after trajectory
    print("Landing swarm...")
    swarm.land()
    swarm.shutdown()

if __name__ == "__main__":
    main() 