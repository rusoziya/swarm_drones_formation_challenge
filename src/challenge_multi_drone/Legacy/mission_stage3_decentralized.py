#!/usr/bin/env python3
import sys
import math
import random
import argparse
import rclpy
import yaml
import time

# --- Import common formation swarm classes from Stage 1 ---
from mission_formations import FormationSwarmConductor, formation_line_3, formation_v_3, formation_line_5, formation_v_5

with open("scenarios/scenario1_stage3.yaml", "r") as f:
    scenario_data = yaml.safe_load(f)

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

'''
def adjust_waypoint(current, next_wp, obstacles, clearance=0.8):
    # If the straight line between current and next_wp collides,
    # apply a local adjustment (for example, a small step perpendicular to the line)
    if check_line_collision(current, next_wp, obstacles, clearance):
        print("Collision detected along trajectory. Adjusting waypoint...")
        # Compute the unit vector along the line
        dx = next_wp[0] - current[0]
        dy = next_wp[1] - current[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return next_wp  # They are almost the same, nothing to adjust
        ux = dx / dist
        uy = dy / dist
        # Compute a perpendicular unit vector
        perp = (-uy, ux)
        # Nudge the next waypoint away from the obstacle
        adjustment = 0.5  # Tunable parameter: how much to nudge
        adjusted_wp = (next_wp[0] + adjustment * perp[0], next_wp[1] + adjustment * perp[1])
        return adjusted_wp
    return next_wp  # No adjustment needed


def adjust_waypoint(current, next_wp, obstacles, clearance=0.9, initial_adjustment=0.4, max_adjustment=1.8, step=0.2):
    """
    Iteratively adjust the next waypoint until a collision-free candidate is found.
    
    current: tuple (x, y) current safe position.
    next_wp: tuple (x, y) desired next waypoint.
    obstacles: list of obstacle positions [(x,y), ...].
    clearance: effective clearance (including obstacle radius).
    initial_adjustment: starting nudge distance.
    max_adjustment: maximum nudge distance.
    step: incremental increase in adjustment distance.
    
    Returns an adjusted waypoint that yields a collision-free line from current,
    or the best candidate found.
    """
    # If the direct path is already collision-free, return next_wp.
    if not check_line_collision(current, next_wp, obstacles, clearance):
        return next_wp
    
    print("Collision detected along trajectory. Searching for robust adjustment iteratively...")
    best_candidate = None
    best_clearance = -1.0
    best_adjustment = None
    
    adjustment = initial_adjustment
    # Iterate over increasing adjustment distances.
    while adjustment <= max_adjustment:
        candidate_found = False
        # Try a range of candidate angles from -90 to +90 degrees, in 15° increments.
        for angle_deg in range(-90, 91, 15):
            angle_rad = math.radians(angle_deg)
            # Vector from current to next_wp.
            dx = next_wp[0] - current[0]
            dy = next_wp[1] - current[1]
            dist = math.hypot(dx, dy)
            if dist < 1e-3:
                continue  # Too close to adjust.
            # Unit vector along the path.
            ux = dx / dist
            uy = dy / dist
            # Perpendicular vector (rotate by 90°).
            perp = (-uy, ux)
            # Rotate the perpendicular vector by the candidate angle.
            candidate_dir = (
                perp[0] * math.cos(angle_rad) - perp[1] * math.sin(angle_rad),
                perp[0] * math.sin(angle_rad) + perp[1] * math.cos(angle_rad)
            )
            candidate_wp = (
                next_wp[0] + adjustment * candidate_dir[0],
                next_wp[1] + adjustment * candidate_dir[1]
            )
            if not check_line_collision(current, candidate_wp, obstacles, clearance):
                # Evaluate candidate clearance.
                candidate_clearance = min(math.hypot(candidate_wp[0]-ox, candidate_wp[1]-oy) for (ox, oy) in obstacles)
                if candidate_clearance > best_clearance:
                    best_clearance = candidate_clearance
                    best_candidate = candidate_wp
                    best_adjustment = adjustment
                    candidate_found = True
        if candidate_found:
            print(f"Found candidate with adjustment {round(adjustment,2)} and clearance {best_clearance:.2f}")
            return best_candidate
        adjustment += step

    if best_candidate is not None:
        print("No fully collision-free candidate found; using best candidate with adjustment", best_adjustment)
        return best_candidate
    else:
        print("No candidate found; returning original next_wp")
        return next_wp
'''

def adjust_waypoint(current, next_wp, obstacles, clearance=0.5, sample_radius=1.8, num_samples=80):
    """
    Randomly sample candidate waypoints around next_wp (within a circle of radius sample_radius)
    and return the candidate that is collision-free along the trajectory from current and has maximum clearance.
    
    current: tuple (x, y) current safe position.
    next_wp: tuple (x, y) desired next waypoint.
    obstacles: list of obstacle positions [(x, y), ...].
    clearance: effective clearance required (includes obstacle radius).
    sample_radius: radius of the circle around next_wp within which to sample candidates.
    num_samples: number of candidates to sample.
    
    Returns an adjusted waypoint.
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
    
    Parameters:
      positions: list of (x,y) for each drone.
      default_altitude: base altitude for all drones.
      separation: how much to offset altitude when collision risk is detected.
      collision_threshold: horizontal distance under which collision is a risk.
      
    Returns:
      altitudes: list of altitudes (one per drone)
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
    
    Returns True if trajectories are collision-free; False otherwise.
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

    stage3_center =      scenario_data["stage3"]["stage_center"]             # absolute center
    start_point_rel =    scenario_data["stage3"]["start_point"]              # relative to stage center
    end_point_rel =      scenario_data["stage3"]["end_point"]                # relative to stage center

    start =   [stage3_center[0] - start_point_rel[1],     stage3_center[1] - start_point_rel[0]]        # e.g., [6.0, -2.0]
    goal =    [stage3_center[0] - end_point_rel[1],       stage3_center[1] - end_point_rel[0]]          # e.g., [6.0, -10.0]

    obstacles_rel = scenario_data["stage3"]["obstacles"]                    # relative positions from the scenario file
    obstacles = []                                                          # absolute positions        
    for obs in obstacles_rel:
        abs_obs = [stage3_center[0] + obs[0], stage3_center[1] + obs[1]]
        obstacles.append(abs_obs)

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
    time0 = time.time()


    # ----- Formation Setup -----
    # base_formation = formation_line_3()
    base_formation = formation_line_5()
    # base_formation = formation_v_5()


    # ----- Global Path Following with Local Reconfiguration -----
    positions_final = []
    flight_speed = 1.8
    safe_distance = 0.9                        # If any formation point is closer than this to an obstacle
    for idx, center in enumerate(global_path):
        print(f"\n=== Global Waypoint {idx}: {[round(x,2) for x in center]} ===")

        # Compute desired formation positions relative to the swarm center.
        desired_positions = [(center[0] + dx, center[1] + dy) for (dx, dy) in base_formation]
        desired_pos_show = [(round(x, 2), round(y, 2)) for (x, y) in desired_positions]
        print("Desired formation positions:", desired_pos_show)

        # Apply local reconfiguration to avoid obstacles.
        adjusted_positions = local_reconfiguration(desired_positions, obstacles, safe_distance=safe_distance, avoidance_gain=0.9)
        adjusted_pos_show = [(round(x, 2), round(y, 2)) for (x, y) in adjusted_positions]
        print("Adjusted formation positions:", adjusted_pos_show)

        # Compute safe position for the next waypoint
        if positions_final == []:
            safe_next_positions = [(start[0] + dx, start[1] + dy) for (dx, dy) in base_formation]
        else:
            current_positions = positions_final[idx-1]  # the last known safe position
            #next_positions = computed_next_position(global_path, idx, base_formation)
            next_positions = adjusted_positions
            safe_next_positions = []
            for curr, nxt in zip(current_positions, next_positions):
                safe_wp = adjust_waypoint(curr, nxt, obstacles)
                safe_next_positions.append(safe_wp)

        #safe_next_position = adjust_waypoint(current_position, next_position, obstacles, clearance=0.5)
        safe_next_positions_ = [(round(x, 2), round(y, 2)) for (x, y) in safe_next_positions]
        print("Safe next position:", safe_next_positions_)
        positions_final.append(safe_next_positions)

        # Check if any drones are too close and adjust altitudes accordingly.
        altitudes = check_and_adjust_altitudes(safe_next_positions, default_altitude=2.5, separation=0.2, collision_threshold=0.35)
        altitudes_show = [round(alt,2) for alt in altitudes]
        print("Assigned altitudes:", altitudes_show)

        # Command the swarm to move to these positions.
        swarm.move_swarm_with_altitudes(safe_next_positions, altitudes, speed=flight_speed)

        # If we are in the last position, check that the landing position correspond to the goal. Otherwise move it there.
        if idx == len(global_path) - 1:
            print("Final waypoint reached. Checking landing position...")
            if distance(safe_next_positions[0], goal) > 0.1:
                print("Landing position is off. Moving to goal...")
                goal = [(goal[0] + dx, goal[1] + dy) for (dx, dy) in base_formation]
                swarm.move_swarm(goal, altitude=2.5, speed=flight_speed)
            else:
                print("Landing position is correct.")
            

    print("Decentralized Stage 3 mission complete. Landing swarm.")
    time1 = time.time()
    time_taken = time1 - time0
    print(f"Time taken for mission: {time_taken:.2f} seconds.")

    # Landing
    swarm.land()
    swarm.shutdown()
    rclpy.shutdown()
    sys.exit(0)

if __name__ == '__main__':
    main()
