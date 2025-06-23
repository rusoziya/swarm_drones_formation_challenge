#!/usr/bin/env python3
import sys
import math
import random
import argparse
import rclpy
import yaml

# Additional imports needed for drone interface functionality.
from rclpy.executors import MultiThreadedExecutor
from as2_msgs.msg import YawMode, BehaviorStatus
from as2_python_api.drone_interface import DroneInterface
from as2_python_api.behavior_actions.behavior_handler import BehaviorHandler
from std_msgs.msg import ColorRGBA

#############################
# Stage 1: Embedded Mission Definitions and Formation Functions
#############################

# --- Formation functions for 3 drones ---
def formation_line_3():
    """
    Return offsets for a line formation for 3 drones.
    """
    d = 0.4  # distance unit
    return [(-d, 0.0), (0.0, 0.0), (d, 0.0)]

def formation_v_3():
    """
    Return offsets for a V formation for 3 drones.
    """
    d = 0.4
    return [
        (0.0, 0.0),    # vertex drone
        (-d, d),
        (d, d)
    ]

# --- Formation functions for 5 drones ---
def formation_line_5():
    """
    Return offsets for a line formation for 5 drones.
    Evenly spaced along the x-axis.
    """
    d = 0.4  # distance unit
    return [(-2*d, 0.0), (-d, 0.0), (0.0, 0.0), (d, 0.0), (2*d, 0.0)]

def formation_v_5():
    """
    Return offsets for a V formation for 5 drones.
    The tip drone is at the vertex and the remaining drones are arranged symmetrically.
    """
    d = 0.4
    return [
        (0.0, 0.0),       # vertex drone
        (-d, d),          # left upper
        (d, d),           # right upper
        (-d, 2*d),        # left further up
        (d, 2*d)          # right further up
    ]

#############################
# Drone Interface and Swarm Conductor Classes
#############################

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

class FormationSwarmConductor:
    def __init__(self, drones_ns: list, verbose: bool = False, use_sim_time: bool = False):
        self.drones = []
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

    def move_swarm_with_altitudes(self, positions, altitudes, speed=1.0):
        """
        positions: list of (x, y) tuples for each drone.
        altitudes: list of altitudes for each drone.
        Commands each drone to move to its corresponding (x, y, altitude) point.
        """
        for d, (px, py), alt in zip(self.drones, positions, altitudes):
            print(f"Moving {d.namespace} to ({px:.2f}, {py:.2f}, {alt})")
            d.go_to_position(px, py, alt, speed=speed)
        print("Waiting for all drones to reach goal...")
        self.wait_all()

########################
# RRT PATH PLANNER CODE
########################

class RRTNode:
    """Node structure for RRT."""
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent

def distance(p1, p2):
    """Euclidean distance."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def nearest_node(nodes, point):
    """Return the node in 'nodes' nearest to 'point'."""
    return min(nodes, key=lambda node: distance((node.x, node.y), point))

def steer(from_node, to_point, step_size=0.5):
    """Generate a new node stepping from 'from_node' toward 'to_point'."""
    theta = math.atan2(to_point[1] - from_node.y, to_point[0] - from_node.x)
    new_x = from_node.x + step_size * math.cos(theta)
    new_y = from_node.y + step_size * math.sin(theta)
    return RRTNode(new_x, new_y, parent=from_node)

def is_collision_free(x, y, obstacles, clearance=0.8):
    """Return True if (x, y) is at least 'clearance' away from all obstacles."""
    for ox, oy in obstacles:
        if distance((x, y), (ox, oy)) < clearance:
            return False
    return True

def generate_rrt_path(start, goal, obstacles, max_iterations=2000, step_size=0.5):
    """
    Generate an RRT path from 'start' to 'goal' avoiding obstacles.
    Returns a list of waypoints [(x, y), ...] if a path is found.
    """
    nodes = [RRTNode(start[0], start[1])]
    goal_reached = False

    for _ in range(max_iterations):
        # Bias 10% of the time towards the goal.
        if random.random() < 0.1:
            random_point = goal
        else:
            # Sample within a bounding box
            min_x = min(start[0], goal[0]) - 2
            max_x = max(start[0], goal[0]) + 2
            min_y = min(start[1], goal[1]) - 2
            max_y = max(start[1], goal[1]) + 2
            random_point = (random.uniform(min_x, max_x), random.uniform(min_y, max_y))

        nearest = nearest_node(nodes, random_point)
        new_node = steer(nearest, random_point, step_size)
        if is_collision_free(new_node.x, new_node.y, obstacles, clearance=0.5):
            nodes.append(new_node)
            if distance((new_node.x, new_node.y), goal) < step_size:
                goal_reached = True
                break

    if not goal_reached:
        return []
    # Reconstruct path
    path = []
    node = nodes[-1]
    while node:
        path.append((node.x, node.y))
        node = node.parent
    return path[::-1]  # From start to goal

########################
# LOCAL RECONFIGURATION
########################

def local_reconfiguration(positions, obstacles, safe_distance=0.3, avoidance_gain=0.9):
    """
    For each formation position, if an obstacle is closer than 'safe_distance',
    compute a repulsion vector and adjust the position.
    """
    new_positions = []
    for pos in positions:
        repulsion = [0.0, 0.0]
        for obs in obstacles:
            d = distance(pos, obs)
            if d < safe_distance:
                dx = pos[0] - obs[0]
                dy = pos[1] - obs[1]
                if d > 0:
                    factor = avoidance_gain * (safe_distance - d) / d
                    repulsion[0] += factor * dx
                    repulsion[1] += factor * dy
        new_positions.append((pos[0] + repulsion[0], pos[1] + repulsion[1]))
    return new_positions

def check_line_collision(start, end, obstacles, clearance=0.8, resolution=0.1):
    # Discretize the line segment between start and end
    dist = math.hypot(end[0]-start[0], end[1]-start[1])
    steps = int(dist / resolution)
    if steps == 0:
        # If start and end are the same, just check the start point for collision.
        for (ox, oy) in obstacles:
            if math.hypot(start[0]-ox, start[1]-oy) < clearance:
                return True
        return False
    for i in range(steps+1):
        t = i / steps
        x = start[0] + t * (end[0]-start[0])
        y = start[1] + t * (end[1]-start[1])
        for (ox, oy) in obstacles:
            if math.hypot(x-ox, y-oy) < clearance:
                return True  # Collision detected
    return False  # No collision along the line

def adjust_waypoint(current, next_wp, obstacles, clearance=0.5, sample_radius=1.8, num_samples=80):
    """
    Randomly sample candidate waypoints around next_wp (within a circle of radius sample_radius)
    and return the candidate that is collision-free along the trajectory from current and has maximum clearance.
    """
    # If the direct path is collision-free, return next_wp.
    if not check_line_collision(current, next_wp, obstacles, clearance):
        return next_wp

    print("Collision detected along trajectory. Sampling candidate waypoints...")
    
    best_candidate = None
    best_candidate_clearance = -1.0

    for _ in range(num_samples):
        # Sample a random angle and radius
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(0, sample_radius)
        candidate_wp = (next_wp[0] + r * math.cos(angle), next_wp[1] + r * math.sin(angle))
        
        # Check if the line from current to candidate is collision-free
        if not check_line_collision(current, candidate_wp, obstacles, clearance):
            # Evaluate candidate clearance: minimal distance to any obstacle
            candidate_clearance = min(math.hypot(candidate_wp[0]-ox, candidate_wp[1]-oy) for (ox, oy) in obstacles)
            if candidate_clearance > best_candidate_clearance:
                best_candidate_clearance = candidate_clearance
                best_candidate = candidate_wp

    if best_candidate is not None:
        print("Robust candidate found:", (round(best_candidate[0],2), round(best_candidate[1],2)),
              "with clearance", round(best_candidate_clearance,2))
        return best_candidate
    else:
        print("No robust candidate found; using original next_wp")
        return next_wp

def computed_next_position(global_path, idx, base_formation):
    next_center = global_path[idx+1] 
    next_positions = [(next_center[0] + dx, next_center[1] + dy) for (dx, dy) in base_formation]
    return next_positions

def check_and_adjust_altitudes(positions, default_altitude=2.5, separation=0.25, collision_threshold=0.4):
    """
    Given formation positions (list of (x,y) tuples), check pairwise distances.
    If any pair is closer than collision_threshold, adjust the altitudes.
    """
    n = len(positions)
    altitudes = [default_altitude for _ in range(n)]
    
    # For each unique pair, check distance
    for i in range(n):
        for j in range(i+1, n):
            d = math.hypot(positions[i][0]-positions[j][0], positions[i][1]-positions[j][1])
            if d < collision_threshold:
                print(f"Collision risk detected between drones {i} and {j} at distance {d:.2f}. Adjusting altitudes.")
                # Simple strategy: one drone gets raised, one lowered.
                altitudes[i] = default_altitude + separation
                altitudes[j] = default_altitude - separation
    return altitudes

def check_drones_trajectory(curr_positions, next_positions, collision_threshold=0.3, num_steps=10):
    """
    Given two lists of positions for the drones (curr_positions and next_positions),
    simulate each drone's straight-line trajectory from current to next (discretized in num_steps)
    and check that at every timestep, the pairwise distance between any two drones
    is at least collision_threshold.
    """
    num = len(curr_positions)
    # Discretize time [0,1] in num_steps+1 points.
    for step in range(num_steps + 1):
        t = step / num_steps
        positions = []
        for pos_curr, pos_next in zip(curr_positions, next_positions):
            # Interpolate: pos = curr + t * (next - curr)
            x = pos_curr[0] + t * (pos_next[0] - pos_curr[0])
            y = pos_curr[1] + t * (pos_next[1] - pos_curr[1])
            positions.append((x, y))
        # Check pairwise distances:
        for i in range(num):
            for j in range(i+1, num):
                d = math.hypot(positions[i][0]-positions[j][0],
                               positions[i][1]-positions[j][1])
                if d < collision_threshold:
                    # A collision risk exists between drones i and j at this timestep.
                    return False
    return True

def load_scenario(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return data

##############################
# STAGE 3 MISSION WITH HIERARCHICAL PLANNING
##############################

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Forest Traversal with Hierarchical (RRT + Local Reconfiguration) Planning")
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

    # ----- Stage 3 Environment Settings -----
    scenario_file = 'scenarios/scenario_flight_arena_stage3.yaml'
    scenario = load_scenario(scenario_file)

    # Parse data
    stage3 = scenario['stage3']
    center = stage3['stage_center']
    start_rel = stage3['start_point']
    goal_rel = stage3['end_point']
    obstacles_rel = stage3['obstacles']

    # Convert relative to absolute positions
    start = (center[0] + start_rel[0], center[1] + start_rel[1])
    goal  = (center[0] + goal_rel[0], center[1] + goal_rel[1])
    obstacles = [(center[0] + ox, center[1] + oy) for (ox, oy) in obstacles_rel]

    # ----- Global Planning with RRT -----
    print("Running global RRT path planning...")
    global_path = generate_rrt_path(start, goal, obstacles, step_size=0.5)
    if not global_path:
        print("No global path found!")
        sys.exit(1)
    print(f"Global RRT path found with {len(global_path)} waypoints.")

    # ----- Takeoff -----
    print("Arming and taking off...")
    if not swarm.get_ready():
        print("Failed to arm/offboard!")
        sys.exit(1)
    swarm.takeoff(height=2.5, speed=0.7)

    # ----- Formation Setup -----
    # By default, we'll use a V formation of 5 drones:
    base_formation = formation_v_5()

    # --------------------------------------------------------------------
    # FIX: Remove or replace the offset that caused a negative X position.
    # If you want the formation to appear exactly at the 'start' coordinate,
    # just move there with no offset:
    # --------------------------------------------------------------------
    swarm.move_swarm(
        [(start[0] + dx, start[1] + dy) for (dx, dy) in base_formation],
        altitude=2.5,
        speed=1.0
    )

    # ----- Global Path Following with Local Reconfiguration -----
    positions_final = []
    flight_speed = 1.0
    safe_distance = 0.9  # If any formation point is closer than this to an obstacle
    for idx, center_point in enumerate(global_path):
        print(f"\n=== Global Waypoint {idx}: {[round(x,2) for x in center_point]} ===")

        # Compute desired formation positions relative to the swarm center.
        desired_positions = [(center_point[0] + dx, center_point[1] + dy) for (dx, dy) in base_formation]
        desired_pos_show = [(round(x, 2), round(y, 2)) for (x, y) in desired_positions]
        print("Desired formation positions:", desired_pos_show)

        # Apply local reconfiguration to avoid obstacles.
        adjusted_positions = local_reconfiguration(
            desired_positions, 
            obstacles, 
            safe_distance=safe_distance, 
            avoidance_gain=0.9
        )
        adjusted_pos_show = [(round(x, 2), round(y, 2)) for (x, y) in adjusted_positions]
        print("Adjusted formation positions:", adjusted_pos_show)

        # Compute safe position for the next waypoint
        if not positions_final:
            # First time, the swarm is already at 'start' in formation.
            safe_next_positions = [(start[0] + dx, start[1] + dy) for (dx, dy) in base_formation]
        else:
            current_positions = positions_final[idx-1]  # the last known safe positions
            next_positions = adjusted_positions
            safe_next_positions = []
            for curr, nxt in zip(current_positions, next_positions):
                safe_wp = adjust_waypoint(curr, nxt, obstacles)
                safe_next_positions.append(safe_wp)

        safe_next_positions_show = [(round(x, 2), round(y, 2)) for (x, y) in safe_next_positions]
        print("Safe next position:", safe_next_positions_show)
        positions_final.append(safe_next_positions)

        # Check if any drones are too close and adjust altitudes accordingly.
        altitudes = check_and_adjust_altitudes(
            safe_next_positions, 
            default_altitude=2.5, 
            separation=0.2, 
            collision_threshold=0.35
        )
        altitudes_show = [round(alt,2) for alt in altitudes]
        print("Assigned altitudes:", altitudes_show)

        # Command the swarm to move to these positions.
        swarm.move_swarm_with_altitudes(safe_next_positions, altitudes, speed=flight_speed)

        # If we are in the last position, check that the landing position corresponds to the goal.
        if idx == len(global_path) - 1:
            print("Final waypoint reached. Checking landing position...")
            if distance(safe_next_positions[0], goal) > 0.1:
                print("Landing position is off. Moving to goal...")
                goal_positions = [(goal[0] + dx, goal[1] + dy) for (dx, dy) in base_formation]
                swarm.move_swarm(goal_positions, altitude=2.5, speed=flight_speed)
            else:
                print("Landing position is correct.")
            
    print("Hierarchical Stage 3 mission complete. Landing swarm.")
    swarm.land()
    swarm.shutdown()
    rclpy.shutdown()
    sys.exit(0)

if __name__ == '__main__':
    main()
