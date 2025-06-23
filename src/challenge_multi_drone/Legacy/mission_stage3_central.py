#!/usr/bin/env python3
import sys
import math
import time
import argparse
import rclpy
import yaml

# Use existing modules from your project (they provide move_swarm, formation_line_3, etc.)
from mission_formations import FormationSwarmConductor, formation_line_3, formation_line_5, formation_vertical_3, formation_vertical_5

with open("scenarios/scenario1_stage3.yaml", "r") as f:
    scenario_data = yaml.safe_load(f)


# ------------------- DYNAMIC PLANNER FORCE -------------------

def compute_dynamic_planner_force(leader_pos, target, obstacles, current_offsets):
    """
    Computes a combined force for the leader:
      - An attractive force toward the target.
      - A repulsive force from each obstacle within influence.
    leader_pos: [x, y]
    target: [x, y]
    obstacles: list of (x, y) positions.
    Returns: (Fx, Fy) force components.
    """
    x, y = leader_pos
    tx, ty = target

    # Attractive force (proportional controller):
    dx = tx - x
    dy = ty - y
    # if abs(dx) < 0.1:
        # dx = 0.5
    dist_target = math.hypot(dx, dy)
    K_att = 0.9
    F_att_x = K_att * dx
    F_att_y = K_att * dy
    print(f"Attractive force: F_att_x = {F_att_x:.2f}, F_att_y = {F_att_y:.2f}, dist_target = {dist_target:.2f}")

    # Repulsive force from obstacles:
    F_rep_x = 0.0
    F_rep_y = 0.0
    K_rep = 2
    influence_radius = 2.7  # Adjust if necessary

    formation_width = get_formation_width(current_offsets)
    buffered_leader_pos_left  = leader_pos[0] - formation_width / 2.0
    buffered_leader_pos_right = leader_pos[0] + formation_width / 2.0

    for idx,obs in enumerate(obstacles):
        ox, oy = obs
        dx_obs = x - ox
        dy_obs = y - oy
        d_obs = math.hypot(dx_obs, dy_obs)


        # Check if the obstacle is within the swarm's width
        within_width = buffered_leader_pos_left - 0.4 <= ox <= buffered_leader_pos_right + 0.4
        within_radius = d_obs < influence_radius and d_obs > 1e-4
        within_lenght = abs(dy_obs) < 3.5

        print(f"{idx}. Obstacle {obs}: dx_obs = {dx_obs:.2f}, dy_obs = {dy_obs:.2f}, d_obs = {d_obs:.2f}, " +
              f"buffered_leader_pos = ({buffered_leader_pos_left:.2f}, {buffered_leader_pos_right:.2f}), " +
              f"ox = {ox}, within_length = {within_lenght}, within_width = {within_width}, ")
        

        if within_width and within_radius:
        # Slight push if perfectly aligned
            if abs(dx_obs) < 0.2:
                dx_obs += 0.3
            # dx_obs += 0.2
            print(f"  → ⚠️ Adjusting dx_obs to avoid perfect X alignment: dx_obs = {dx_obs:.2f}")
            
            repulsion = K_rep * (1.0 / d_obs - 1.0 / influence_radius) / (d_obs**1.5)
            #F_rep_y *= 0.6
            F_rep_x += repulsion * dx_obs * 7
            # F_rep_y += repulsion * dy_obs

            print(f"  → 🛑 Repulsion applied: rep = {repulsion:.3f}, "
                f"F_rep_x = {F_rep_x:.3f}, F_rep_y = {F_rep_y:.3f}")
        

    # Total force:
    Fx = F_att_x + F_rep_x
    Fy = F_att_y + F_rep_y
    return Fx, Fy


def get_formation_width(offsets):
    xs = [x for x, y in offsets]
    return max(xs) - min(xs)

def select_formation(leader_pos, obstacles, current_offsets):
    """
    Determine which formation to use based on how many obstacles are nearby.
    - If two or more obstacles are within the sensing_radius, use a column (squeezed) formation.
    - Otherwise, use the default horizontal line formation.
    """
    # sensing_radius = 1.6
    sensing_radius_front = 1.5
    sensing_radius_back = 1.5
    count = 0
    x, y = leader_pos

    formation_width = get_formation_width(current_offsets)
    buffered_leader_pos_left  = leader_pos[0] - formation_width / 2.0
    buffered_leader_pos_right = leader_pos[0] + formation_width / 2.0


    for obs in obstacles:
        ox, oy = obs
        within_width = buffered_leader_pos_left - 0.2 <= ox <= buffered_leader_pos_right + 0.2
        if y - oy > 0:
            if math.hypot(x - ox, y - oy) < sensing_radius_front and not within_width:
                count += 1
                print(f"  --> ⚠️⚠️⚠️ Obstacle {obs} is in front of the leader: count = {count}, y - oy = {y - oy:.2f}")
        elif y - oy < 0:
            if math.hypot(x - ox, y - oy) < sensing_radius_back and not within_width:
                count += 1
                print(f"  --> ⚠️⚠️⚠️ Obstacle {obs} is behind the leader: count = {count}, y - oy = {y - oy:.2f}")
        

    if count >= 2:
        # return "column", formation_vertical_3()
        return "column", formation_vertical_5()
    else:
        # return "default", formation_line_3()
        return "default", formation_line_5()

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Centralized Leader-Follower Forest Traversal with Dynamic Obstacle Avoidance")
    parser.add_argument('-n', '--namespaces', nargs='+',
                        default=['drone0', 'drone2', 'drone1'], help='Drone namespaces')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('-s', '--use_sim_time', action='store_true', default=True)
    args = parser.parse_args()

    rclpy.init()
    swarm = FormationSwarmConductor(args.namespaces, verbose=args.verbose, use_sim_time=args.use_sim_time)

    # === Scenario parameters (Stage 3: Forest of Trees) ===

    stage_center =      scenario_data["stage3"]["stage_center"]             # absolute center
    start_point_rel =   scenario_data["stage3"]["start_point"]              # relative to stage center
    end_point_rel =     scenario_data["stage3"]["end_point"]                # relative to stage center

    start_wp =  [stage_center[0] - start_point_rel[1],     stage_center[1] - start_point_rel[0]]        # e.g., [6.0, -2.0]
    end_wp =    [stage_center[0] - end_point_rel[1],       stage_center[1] - end_point_rel[0]]          # e.g., [6.0, -10.0]

    obstacles_rel = scenario_data["stage3"]["obstacles"]                    # relative positions from the scenario file
    obstacles_abs = []                                                      # absolute positions        
    for obs in obstacles_rel:
        abs_obs = [stage_center[0] + obs[0], stage_center[1] + obs[1]]
        obstacles_abs.append(abs_obs)

    # Leader starts at the start waypoint:
    leader_pos = list(start_wp)
    altitude = 2.5 

    # Initialize swarm
    print("Arming and taking off...")
    if not swarm.get_ready():
        print("Failed to arm/offboard!")
        sys.exit(1)
    swarm.takeoff(height=altitude, speed=0.7)

    # Set starting formation
    # initial_offsets = formation_line_3()
    initial_offsets = formation_line_5()
    init_positions = [(leader_pos[0] + dx, leader_pos[1] + dy) for (dx, dy) in initial_offsets]
    swarm.move_swarm(init_positions, altitude=altitude, speed=1.0)
    formation_name, formation_offsets = select_formation(leader_pos, obstacles_abs, initial_offsets)


    time0 = time.time()
    dt = 0.15  # time step [seconds]
    step = 0

    # -------- Main Loop --------
    while True:
        print("=" * 60)
        print(f"[Step {step}]")

        dx_end = end_wp[0] - leader_pos[0]
        dy_end = end_wp[1] - leader_pos[1]

        # Check if the leader is near the final waypoint (break the loop)
        dist_to_end = math.hypot(dx_end, dy_end)
        if dist_to_end < 0.2:
            swarm.move_swarm([(leader_pos[0] + dx, leader_pos[1] + dy) for (dx, dy) in initial_offsets], altitude=altitude, speed=1.0)
            print("Leader reached final waypoint.")
            break

        # Dynamic Obstacle-Aware Planning
        Fx, Fy = compute_dynamic_planner_force(leader_pos, end_wp, obstacles_abs, formation_offsets)

        # Limit maximum speed
        max_speed = 1.8
        speed = math.hypot(Fx, Fy)
        if speed > max_speed:
            Fx *= max_speed / speed
            Fy *= max_speed / speed

        # Update leader position
        leader_pos[0] += Fx * dt
        leader_pos[1] += Fy * dt

        # Formation Selection Based on Local Obstacle Density
        formation_name, formation_offsets = select_formation(leader_pos, obstacles_abs, formation_offsets)

        # Compute absolute target positions for each drone (leader + offsets)
        positions = [(leader_pos[0] + off_x, leader_pos[1] + off_y) for (off_x, off_y) in formation_offsets]

        # Debug Logging 
        print(f"Leader: ({leader_pos[0]:.2f}, {leader_pos[1]:.2f}), " +
              f"Dist to goal: {dist_to_end:.2f}, Formation: {formation_name}")
        print(f" → Net Force: Fx = {Fx:.3f}, Fy = {Fy:.3f}, Speed = {speed:.3f}")
        print(" → Drone target positions:")
        for i, p in enumerate(positions):
            print(f"    Drone {i}: ({p[0]:.2f}, {p[1]:.2f})")

        # Obstacle influence debugging
        for i, obs in enumerate(obstacles_abs):
            ox, oy = obs
            dx_obs = leader_pos[0] - ox
            dy_obs = leader_pos[1] - oy
            d_obs = math.hypot(dx_obs, dy_obs)
            if d_obs < 2.0:
                print(f"    ⚠️ Obstacle {i} at ({ox:.2f}, {oy:.2f}) is CLOSE: dist = {d_obs:.2f}")
                if d_obs < 0.5:
                    print("       ‼️  WARNING: Obstacle is VERY CLOSE! Possible collision!")

        # Command the swarm to move:
        swarm.move_swarm(positions, altitude=altitude, speed=1.8)
        time.sleep(0.005)
        step += 1

    print("Mission complete. Landing swarm.")
    swarm.land()
    swarm.shutdown()
    rclpy.shutdown()
    sys.exit(0)

if __name__ == '__main__':
    main()
