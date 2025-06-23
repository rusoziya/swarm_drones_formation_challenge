#!/usr/bin/env python3
import sys
import math
import time
import argparse
import rclpy
import yaml

# Using the formation_line_3 and FormationSwarmConductor from your existing code.
from mission_formations import FormationSwarmConductor, formation_line_3, formation_line_5, formation_vertical_3, formation_vertical_5


with open("scenarios/scenario1_stage2.yaml", "r") as f:
    scenario_data = yaml.safe_load(f)


# ------------------- CENTRALIZED LEADER-FOLLOWER CONTROL -------------------
def main():
    parser = argparse.ArgumentParser(description="Stage 2: Centralized Leader-Follower Window Traversal")
    parser.add_argument('-n', '--namespaces', nargs='+',
                        default=['drone0','drone1','drone2'],
                        help='Drone namespaces')
    parser.add_argument('-v','--verbose', action='store_true', default=False)
    parser.add_argument('-s','--use_sim_time', action='store_true', default=True)
    args = parser.parse_args()

    rclpy.init()
    swarm = FormationSwarmConductor(args.namespaces, verbose=args.verbose, use_sim_time=args.use_sim_time)

    stage_center = scenario_data["stage2"]["stage_center"]  # [x0, y0]
    windows_raw = scenario_data["stage2"]["windows"]

    windows = []

    for widx in windows_raw:
        win = windows_raw[widx]
        local_x, local_y = win["center"]  # YAML stores as [y, x]
        
        # Convert to global (x, y)
        global_x = stage_center[0] + local_x
        global_y = stage_center[1] + local_y
        
        window = {
            "center": (global_x, global_y),
            "gap_width": win["gap_width"],
            "gap_height": win["height"],
            "z": win["distance_floor"] + win["height"] / 2.0,  # midpoint height
            "thickness": win["thickness"]
        }
        windows.append(window)
        print(f"Window {widx}: center={window['center']}, gap_width={window['gap_width']}, gap_height={window['gap_height']}, z={window['z']}")


    waypoints = [
        (0.0, 0.0),                                                       # Start waypoint
        (windows[0]["center"][0], windows[0]["center"][1] - 0.4),         # Go through/near window1 center
        (windows[1]["center"][0] + 0.3, windows[1]["center"][1] - 0.4),   # Go through/near window2 center
        (0.0, -10.0)                                                      # End waypoint
    ]

    altitudes = [
        2.5,                # Start: default altitude
        windows[0]["z"],    # At window1: use window-specific altitude
        windows[1]["z"],    # At window2: use window-specific altitude
        2.5                 # End: revert to default altitude
    ]

    # Initialize the leader position to the first waypoint
    leader_pos = list(waypoints[0])
    current_wp_idx = 1  # next waypoint index for the leader

    # Formation selection: use a string variable to switch formations.
    # "default" -> horizontal line formation; "vertical" -> vertical formation for window traversals.
    current_formation_name = "default"  # initially, default formation
    previous_formation_name = current_formation_name
    formation_switch_count = 0

    # Initialize swarm
    print("Arming and taking off...")
    if not swarm.get_ready():
        print("Failed to arm/offboard!")
        sys.exit(1)
    swarm.takeoff(height=2.5, speed=0.7)

    # Move swarm initially to a safe starting formation
    starting_formation = formation_line_3()
    # starting_formation = formation_line_5()
    swarm.move_swarm([(leader_pos[0] + dx, leader_pos[1] + dy) for (dx,dy) in starting_formation], altitude=2.5, speed=1.0)

    dt = 0.2  # time step
    step = 0

    while current_wp_idx < len(waypoints):
        # Compute direction from leader to current target waypoint
        target_wp = waypoints[current_wp_idx]
        dx = target_wp[0] - leader_pos[0]
        dy = target_wp[1] - leader_pos[1]
        distance = math.hypot(dx, dy)

        # Proportional controller for the leader
        gain = 1.0 
        vx = gain * dx
        vy = gain * dy

        # Limit the speed 
        max_speed = 1.8
        speed = math.hypot(vx, vy)
        if speed > max_speed:
            vx *= (max_speed / speed)
            vy *= (max_speed / speed)

        # Update leader position using Euler integration
        leader_pos[0] += vx * dt
        leader_pos[1] += vy * dt

        # Check if leader is near the target waypoint
        if current_wp_idx == 1 or current_wp_idx == 2:  
            dist_y_center = windows[current_wp_idx-1]["center"][1] - leader_pos[1]
            # print(f" Distance from y_center: {dist_y_center:.2f} (y_center: {windows[current_wp_idx]['center'][1]})")
            # If distance is enough, consider the waypoint reached
            if dist_y_center > 0.3:
                print(f"[Waypoint] Leader reached waypoint {current_wp_idx}: {target_wp}")
                current_wp_idx += 1
        else:
            if distance < 0.2:
                current_wp_idx += 1
                if current_wp_idx >= len(waypoints):
                    # Reached final waypoint
                    break

        # Formation switching: if the leader is near a window's center, then use vertical formation
        formation_threshold = 1.3  # distance threshold for switching formations
        window_near = None
        for index, w in enumerate(windows):
            d_window = math.hypot(leader_pos[0] - w["center"][0], leader_pos[1] - w["center"][1])
            if index == 0:
                if d_window < formation_threshold:
                    window_near = w
                    break
            else:
                if d_window < formation_threshold + 0.3:
                    window_near = w
                    break

        if window_near:
            # When near a window, use vertical formation
            current_formation_name = "vertical"
            # formation_offsets = formation_vertical_3()
            formation_offsets = formation_vertical_5()
            current_altitude = window_near["z"]
        else:
            current_formation_name = "default"
            # formation_offsets = formation_line_3()
            formation_offsets = formation_line_5()
            current_altitude = altitudes[current_wp_idx-1] 

        # Count formation switches (only when the formation changes between steps)
        if current_formation_name != previous_formation_name:
            formation_switch_count += 1
            print(f"[Formation Change] Switched from {previous_formation_name} to {current_formation_name} at step {step}. Total switches: {formation_switch_count}")
        previous_formation_name = current_formation_name

        # Compute full positions for all drones relative to leader position
        positions = [(leader_pos[0] + dx_off, leader_pos[1] + dy_off) for (dx_off, dy_off) in formation_offsets]
        
        # Logging the leader and formation info
        print(f"[Step {step}] Leader position: ({leader_pos[0]:.2f}, {leader_pos[1]:.2f}), Formation: {current_formation_name}, Altitude: {current_altitude}")

        # Command the swarm with the new formation positions.
        swarm.move_swarm(positions, altitude=current_altitude, speed=1.0)
        
        time.sleep(0.05)
        step += 1

    print("Final waypoint reached. Landing swarm.")
    swarm.land()
    swarm.shutdown()
    rclpy.shutdown()
    sys.exit(0)

if __name__ == '__main__':
    main()
