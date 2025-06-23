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

def compute_force(leader_pos, target, obstacles):
    """
    Simple force-based approach:
    - Attractive force toward the target
    - Repulsive force from obstacles
    """
    x, y = leader_pos
    tx, ty = target
    
    # Attractive force (proportional controller)
    dx = tx - x
    dy = ty - y
    dist_target = math.hypot(dx, dy)
    K_att = 1.2  # Increased attractive gain for faster movement
    F_att_x = K_att * dx
    F_att_y = K_att * dy
    
    # Repulsive force from obstacles
    F_rep_x = 0.0
    F_rep_y = 0.0
    K_rep = 2.5  # Increased repulsion strength for better obstacle avoidance
    influence_radius = 3.5  # Increased influence radius to detect obstacles from further away
    
    for obs in obstacles:
        ox, oy = obs
        dx_obs = x - ox
        dy_obs = y - oy
        d_obs = math.hypot(dx_obs, dy_obs)
        
        if d_obs < influence_radius and d_obs > 0.1:
            # Calculate repulsion force
            repulsion = K_rep * (1.0 / d_obs - 1.0 / influence_radius) / (d_obs**2)
            F_rep_x += repulsion * dx_obs * 2.5  # Increased horizontal repulsion
            F_rep_y += repulsion * dy_obs * 1.5  # Increased vertical repulsion
    
    # Total force
    Fx = F_att_x + F_rep_x
    Fy = F_att_y + F_rep_y
    
    # Normalize if too large
    force_mag = math.hypot(Fx, Fy)
    if force_mag > 2.0:  # Increased maximum force
        Fx *= 2.0 / force_mag
        Fy *= 2.0 / force_mag
    
    return Fx, Fy

def check_obstacle_proximity(leader_pos, obstacles, formation_offsets):
    """
    Check if any drone in the formation is too close to obstacles
    Returns: (is_near_obstacle, obstacle_count)
    """
    obstacle_count = 0
    is_near_obstacle = False
    
    # Check each drone in the formation
    for dx, dy in formation_offsets:
        drone_x = leader_pos[0] + dx
        drone_y = leader_pos[1] + dy
        
        for obs in obstacles:
            ox, oy = obs
            dist = math.hypot(drone_x - ox, drone_y - oy)
            
            if abs(drone_y - oy) < 0.9:  # Increased vertical detection threshold (was 0.7)
                if dist < 1.5:  # Increased safety distance (was 1.3)
                    is_near_obstacle = True
                    obstacle_count += 1
    return is_near_obstacle, obstacle_count

def find_safe_formation_position(leader_pos, obstacles, formation_offsets, search_range=2.5, step_size=0.2):
    """
    Find a safe x-position for the leader that avoids collisions with obstacles
    Returns: (safe_x, safe_positions)
    """
    original_x = leader_pos[0]
    best_x = original_x
    best_collision_count = float('inf')
    
    # Try different x positions
    for offset in range(-int(search_range/step_size), int(search_range/step_size) + 1):
        test_x = original_x + offset * step_size
        test_pos = [test_x, leader_pos[1]]
        test_positions = [(test_pos[0] + dx, test_pos[1] + dy) for (dx, dy) in formation_offsets]
        
        # Check for collisions
        collision_count = 0
        for pos in test_positions:
            for obs in obstacles:
                if math.hypot(pos[0] - obs[0], pos[1] - obs[1]) < 1.0:  # Increased safety distance (was 0.8)
                    collision_count += 1
                    break
        
        # If this position has fewer collisions, update the best position
        if collision_count < best_collision_count:
            best_collision_count = collision_count
            best_x = test_x
            
            # If we found a position with no collisions, we can stop searching
            if collision_count == 0:
                break
    
    # Create the safe positions with the best x
    safe_positions = [(best_x + dx, leader_pos[1] + dy) for (dx, dy) in formation_offsets]
    
    return best_x, safe_positions

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Centralized Leader-Follower Forest Traversal")
    parser.add_argument('-n', '--namespaces', nargs='+',
                         default=['drone0', 'drone2', 'drone1'], help='Drone namespaces')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('-s', '--use_sim_time', action='store_true', default=True)
    args = parser.parse_args()

    rclpy.init()
    swarm = FormationSwarmConductor(args.namespaces, verbose=args.verbose, use_sim_time=args.use_sim_time)

    # === Scenario parameters (Stage 3: Forest of Trees) ===
    stage_center = scenario_data["stage3"]["stage_center"]
    start_point_rel = scenario_data["stage3"]["start_point"]
    end_point_rel = scenario_data["stage3"]["end_point"]

    start_wp = [stage_center[0] - start_point_rel[1], stage_center[1] - start_point_rel[0]]
    end_wp = [stage_center[0] - end_point_rel[1], stage_center[1] - end_point_rel[0]]

    obstacles_rel = scenario_data["stage3"]["obstacles"]
    obstacles_abs = []
    for obs in obstacles_rel:
        abs_obs = [stage_center[0] + obs[0], stage_center[1] + obs[1]]
        obstacles_abs.append(abs_obs)

    # Leader starts at the start waypoint
    leader_pos = list(start_wp)
    altitude = 2.5

    # Initialize swarm
    print("Arming and taking off...")
    if not swarm.get_ready():
        print("Failed to arm/offboard!")
        sys.exit(1)
    swarm.takeoff(height=altitude, speed=1.5)  # Increased takeoff speed

    # Set starting formation
    current_formation = "line"
    formation_offsets = formation_line_5()
    init_positions = [(leader_pos[0] + dx, leader_pos[1] + dy) for (dx, dy) in formation_offsets]
    swarm.move_swarm(init_positions, altitude=altitude, speed=2.5)  # Increased formation setup speed

    # Formation switching parameters
    formation_switch_time = time.time()
    min_formation_time = 1.5  # Reduced minimum time to stay in a formation
    last_formation = current_formation

    time0 = time.time()
    dt = 0.4  # Increased time step for faster simulation progress
    step = 0

    # -------- Main Loop --------
    while True:
        print("=" * 60)
        print(f"[Step {step}]")

        # Check if we've reached the goal
        dx_end = end_wp[0] - leader_pos[0]
        dy_end = end_wp[1] - leader_pos[1]
        dist_to_end = math.hypot(dx_end, dy_end)
        
        if dist_to_end < 0.4:
            print("Leader reached final waypoint.")
            final_offsets = formation_line_5()
            final_positions = [(leader_pos[0] + dx, leader_pos[1] + dy) for (dx, dy) in final_offsets]
            swarm.move_swarm(final_positions, altitude=altitude, speed=5.0)  # Increased final approach speed
            break

        # Compute force for movement
        Fx, Fy = compute_force(leader_pos, end_wp, obstacles_abs)
        
        # Check if we're near obstacles
        is_near_obstacle, obstacle_count = check_obstacle_proximity(leader_pos, obstacles_abs, formation_offsets)
        
        # Formation switching logic
        time_in_formation = time.time() - formation_switch_time
        
        # Only consider switching if we've been in the current formation for at least min_formation_time
        if time_in_formation >= min_formation_time:
            # Simple rule: switch to vertical if near obstacles, switch back to line if not
            if is_near_obstacle and current_formation == "line":
                print("  --> 🔄 Switching to VERTICAL formation (obstacles detected)")
                
                # Check if the vertical formation would collide with obstacles
                vertical_offsets = formation_vertical_5()
                vertical_positions = [(leader_pos[0] + dx, leader_pos[1] + dy) for (dx, dy) in vertical_offsets]
                
                # Check for collisions
                collision_detected = False
                for pos in vertical_positions:
                    for obs in obstacles_abs:
                        if math.hypot(pos[0] - obs[0], pos[1] - obs[1]) < 0.8:  # Safety distance
                            collision_detected = True
                            print(f"  ⚠️ VERTICAL formation would collide with obstacle at {obs}")
                            break
                    if collision_detected:
                        break
                
                if collision_detected:
                    print("  🔍 Finding safe position for vertical formation...")
                    safe_x, safe_positions = find_safe_formation_position(leader_pos, obstacles_abs, vertical_offsets)
                    
                    # Update leader position to the safe x position
                    leader_pos[0] = safe_x
                    print(f"  ✅ Adjusted leader x-position to {safe_x:.2f} for safe vertical formation")
                    
                    # Use the safe positions for the formation
                    new_positions = safe_positions
                else:
                    # No collision, use the standard vertical formation
                    current_formation = "vertical"
                    formation_offsets = vertical_offsets
                    formation_switch_time = time.time()
                    new_positions = vertical_positions
            elif not is_near_obstacle and current_formation == "vertical":
                print("  --> 🔄 Switching to LINE formation (no obstacles nearby)")
                current_formation = "line"
                formation_offsets = formation_line_5()
                formation_switch_time = time.time()
                new_positions = [(leader_pos[0] + dx, leader_pos[1] + dy) for (dx, dy) in formation_offsets]
            else:
                print(f"  --> Keeping {current_formation.upper()} formation (time in formation: {time_in_formation:.1f}s)")
                new_positions = [(leader_pos[0] + dx, leader_pos[1] + dy) for (dx, dy) in formation_offsets]
        else:
            print(f"  --> Keeping {current_formation.upper()} formation for at least {min_formation_time:.1f} seconds (current: {time_in_formation:.1f}s)")
            new_positions = [(leader_pos[0] + dx, leader_pos[1] + dy) for (dx, dy) in formation_offsets]

        # Calculate new positions based on force
        new_leader_pos = [leader_pos[0] + Fx * dt, leader_pos[1] + Fy * dt]
        
        # Only update positions if we're not already using safe positions from formation switching
        if 'new_positions' not in locals():
            new_positions = [(new_leader_pos[0] + dx, new_leader_pos[1] + dy) for (dx, dy) in formation_offsets]
        
        # Safety check: verify the new position won't cause collisions
        collision_detected = False
        for pos in new_positions:
            for obs in obstacles_abs:
                if math.hypot(pos[0] - obs[0], pos[1] - obs[1]) < 1.0:  # Increased safety distance (was 0.8)
                    collision_detected = True
                    print(f"  ⚠️ COLLISION DETECTED! Drone at {pos} would collide with obstacle at {obs}")
                    break
            if collision_detected:
                break
        
        # If collision detected, reduce movement
        if collision_detected:
            print("  ⚠️ Reducing movement to avoid collision")
            Fx *= 0.2  # More aggressive reduction (was 0.3)
            Fy *= 0.2  # More aggressive reduction (was 0.3)
            # Recalculate new positions with reduced force
            new_leader_pos = [leader_pos[0] + Fx * dt, leader_pos[1] + Fy * dt]
            new_positions = [(new_leader_pos[0] + dx, new_leader_pos[1] + dy) for (dx, dy) in formation_offsets]

        # Update leader position
        leader_pos[0] = new_leader_pos[0]
        leader_pos[1] = new_leader_pos[1]

        # Debug Logging 
        print(f"Leader: ({leader_pos[0]:.2f}, {leader_pos[1]:.2f}), " +
              f"Dist to goal: {dist_to_end:.2f}, Formation: {current_formation}")
        print(f" → Net Force: Fx = {Fx:.3f}, Fy = {Fy:.3f}")
        print(" → Drone target positions:")
        for i, p in enumerate(new_positions):
            print(f"    Drone {i}: ({p[0]:.2f}, {p[1]:.2f})")

        # Command the swarm to move
        swarm.move_swarm(new_positions, altitude=altitude, speed=6.0)  # Significantly increased movement speed
        time.sleep(0.01)
        step += 1

    print("Mission complete. Landing swarm.")
    swarm.land()
    swarm.shutdown()
    rclpy.shutdown()
    sys.exit(0)

if __name__ == '__main__':
    main()



