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

# --- Formation functions for 3 drones ---
def formation_line_3():
    """
    Return offsets for a line formation for 3 drones.
    Evenly spaced along the x-axis.
    """
    d = 0.4  # distance between drones
    return [(-d, 0.0), (0.0, 0.0), (d, 0.0)]

def formation_v_3():
    """
    Return offsets for a V (or triangle) formation for 3 drones.
    Middle drone at vertex, the others offset upward.
    """
    d = 0.4  # lateral offset
    return [(0.0, 0.0), (-d, d), (d, d)]

def formation_diamond_3():
    """
    Return offsets for a diamond formation for 3 drones.
    One drone at the top and two drones forming the bottom corners.
    """
    d = 0.4
    return [(0.0, d), (-d, -d), (d, -d)]

def formation_circular_orbit_3():
    """
    Return offsets for a circular orbit formation for 3 drones.
    Drones are positioned evenly along a small circle relative to the center.
    """
    orbit_radius = 0.4
    offsets = []
    for i in range(3):
        angle = 2 * math.pi * i / 3
        offsets.append((orbit_radius * math.cos(angle), orbit_radius * math.sin(angle)))
    return offsets

def formation_grid_3():
    """
    Return offsets for a grid formation for 3 drones.
    Arranged in an L-shaped grid.
    """
    d = 0.4
    return [(0.0, 0.0), (d, 0.0), (0.0, d)]

def formation_staggered_3():
    """
    Return offsets for a staggered formation for 3 drones.
    Creates a zigzag pattern.
    """
    d = 0.4
    return [(0.0, 0.0), (d, d / 2), (2 * d, 0.0)]

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
    def _init_(self, namespace: str, verbose: bool = False, use_sim_time: bool = False):
        super()._init_(namespace, verbose=verbose, use_sim_time=use_sim_time)
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
    def _init_(self, drones_ns: list, verbose: bool = False, use_sim_time: bool = False):
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
    parser = argparse.ArgumentParser(description="Stage 1: Formation Flight Mission")
    parser.add_argument('-n', '--namespaces', nargs='+',
                        default=['drone0', 'drone1', 'drone2'],
                        help='List of drone namespaces')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Enable verbose output')
    parser.add_argument('-s', '--use_sim_time', action='store_true', default=True,
                        help='Use simulation time')
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

    # Updated formation list with new formations in sequential order.
    formations = [
        formation_line_3,
        formation_v_3,
        formation_diamond_3,
        formation_circular_orbit_3,
        formation_grid_3,
        formation_staggered_3
    ]
    current_formation_idx = 0
    steps_per_formation = 12         # Switch formation every 12 steps

    # Takeoff procedure
    print("Arming and taking off...")
    if not swarm.get_ready():
        print("Failed to arm/offboard!")
        sys.exit(1)
    swarm.takeoff(height=altitude, speed=0.7)

    # Execute circular trajectory with periodic formation changes
    angle = 0.0
    total_steps = int(360 / angle_deg)  # Full circle over n steps
    for step in range(total_steps):
        # Switch formation every steps_per_formation iterations
        if step % steps_per_formation == 0 and step != 0:
            current_formation_idx = (current_formation_idx + 1) % len(formations)
            print(f"Switching formation to index {current_formation_idx}")

        # Compute base (center) position on the circle
        base_x = stage_center[0] + radius * math.cos(angle)
        base_y = stage_center[1] + radius * math.sin(angle)

        # Get the current formation offsets (list of (dx, dy) for 3 drones)
        formation_func = formations[current_formation_idx]
        offsets = formation_func()

        # Compute final positions for each drone by rotating offsets by the current angle
        positions = []
        for (dx, dy) in offsets:
            rotated = rotate_offset((dx, dy), angle)
            pos_x = base_x + rotated[0]
            pos_y = base_y + rotated[1]
            positions.append((pos_x, pos_y))

        print(f"Step {step}: Base position = ({base_x:.2f}, {base_y:.2f}), Offsets = {offsets}")
        print(f"Final positions: {positions}")

        swarm.move_swarm(positions, altitude=altitude, speed=1.0)
        angle += angle_step

    # Land after trajectory
    print("Landing swarm...")
    swarm.land()
    swarm.shutdown()
    rclpy.shutdown()
    sys.exit(0)


if __name__ == '__main__':
    main()