#!/usr/bin/env python3
import sys
import math
import time
import argparse
import rclpy
import yaml
import os

# Using the formation_line_3 and FormationSwarmConductor from your existing code.
from mission_formations import FormationSwarmConductor, formation_line_3, formation_line_5, formation_vertical_3, formation_vertical_5

with open("scenarios/scenario1_stage2.yaml", "r") as f:
    scenario_data = yaml.safe_load(f)

def formation_stable(old, new, tol=1e-3):
    """Check if two formations (lists of positions) are effectively the same."""
    return all(math.hypot(x1-x2, y1-y2) < tol for (x1,y1),(x2,y2) in zip(old, new))

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Centralized Leader-Follower Window Traversal")
    parser.add_argument('-n', '--namespaces', nargs='+',
                        default=['drone0','drone1','drone2'],
                        help='Drone namespaces')
    parser.add_argument('-v','--verbose', action='store_true', default=False)
    parser.add_argument('-s','--use_sim_time', action='store_true', default=True)
    parser.add_argument('-d','--debug', action='store_true', default=False, help='Enable debug prints')
    args = parser.parse_args()
    debug = args.debug

    # ── METRIC SETUP ─────────────────────────────────────────────────────
    n_drones = len(args.namespaces)
    success = False
    window_entry = {}      # window_idx -> entry time
    traversal_times = []   # durations per window
    reformation_times = [] # time to re-form after last window
    switch_detect_ts = []  # when switch was detected
    switch_latency = []    # actual latencies
    path_length = [0.0]*n_drones
    formation_switch_count = 0
    prev_positions = None
    switch_start_time = None
    in_nondefault_formation = False
    cpu_samples = []
    # ── END METRIC SETUP ─────────────────────────────────────────────────

    rclpy.init()
    swarm = FormationSwarmConductor(args.namespaces,
                                   verbose=args.verbose,
                                   use_sim_time=args.use_sim_time)

    stage_center = scenario_data["stage2"]["stage_center"]
    windows_raw = scenario_data["stage2"]["windows"]

    # Windows coordinates
    windows = []
    for widx in windows_raw:
        win = windows_raw[widx]
        local_x, local_y = win["center"]
        gx = stage_center[0] + local_x
        gy = stage_center[1] + local_y
        windows.append({
            "center": (gx, gy),
            "gap_width": win["gap_width"],
            "gap_height": win["height"],
            "z": win["distance_floor"] + win["height"]/2.0,
            "thickness": win["thickness"]
        })
    
    print(f"Windows loaded: {len(windows)} windows defined")

    # Waypoints and altitudes
    waypoints = [
        (0.0, 0.0),                                                       # Start waypoint
        (windows[0]["center"][0], windows[0]["center"][1] - 0.4),         # Go through/near window1 center
        (windows[1]["center"][0] + 0.3, windows[1]["center"][1] - 0.4),   # Go through/near window2 center
        (0.0, -10.0)                                                      # End waypoint
    ]
    
    print(f"Path defined: Start → Window1 → Window2 → End")

    altitudes = [
        2.5,                # Start: default altitude
        windows[0]["z"],    # At window1: use window-specific altitude
        windows[1]["z"],    # At window2: use window-specific altitude
        2.5                 # End: revert to default altitude
    ]

    # Leader state
    leader_pos = list(waypoints[0])
    current_wp_idx = 1

    current_formation_name = "default"
    previous_formation_name = current_formation_name

    # Arm & takeoff
    print("Arming and taking off...")
    if not swarm.get_ready():
        print("Failed to arm/offboard!")
        sys.exit(1)
    swarm.takeoff(height=2.5, speed=2.0)

    # Move to start formation
    start_offsets = formation_line_5()
    start_positions = [(leader_pos[0]+dx, leader_pos[1]+dy) for dx,dy in start_offsets]
    swarm.move_swarm(start_positions, altitude=2.5, speed=2.0)
    time0 = time.time()
    prev_positions = start_positions.copy()
    
    print(f"Initial formation: {current_formation_name.upper()}")
    print(f"Starting simulation...")
    print("=" * 40)

    dt = 0.2
    step = 0
    reform_start = None

    while current_wp_idx < len(waypoints):
        cpu_start = time.process_time()
        
        if step % 10 == 0:  # Print every 10 steps to reduce output
            print(f"Step {step} | Leader: ({leader_pos[0]:.2f}, {leader_pos[1]:.2f}) | Target: WP{current_wp_idx}")

        # Leader velocity
        tx, ty = waypoints[current_wp_idx]
        dx = tx - leader_pos[0]
        dy = ty - leader_pos[1]
        dist = math.hypot(dx, dy)

        # simple P-control
        gain = 1.0
        vx, vy = gain*dx, gain*dy
        # limit speed
        max_speed = 3.0
        sp = math.hypot(vx, vy)
        if sp > max_speed:
            vx *= max_speed/sp
            vy *= max_speed/sp

        # integrate
        leader_pos[0] += vx*dt
        leader_pos[1] += vy*dt

        # waypoint checking
        if current_wp_idx in (1,2):
            d_y = windows[current_wp_idx-1]["center"][1] - leader_pos[1]
            if d_y > 0.3:
                print(f"Passed window {current_wp_idx-1}, advancing to waypoint {current_wp_idx+1}")
                current_wp_idx += 1
        else:
            if dist < 0.2:
                print(f"Reached waypoint {current_wp_idx}, advancing to next waypoint")
                current_wp_idx += 1

        # formation switching logic
        formation_threshold = 1.2             # IMPORTANT for parameter tuning
        window_near = None
        for i,w in enumerate(windows):
            d_win = math.hypot(leader_pos[0]-w["center"][0], leader_pos[1]-w["center"][1])
            thresh = formation_threshold + (0.4 if i==1 else 0.0)
            if d_win < thresh:
                window_near = w
                break

        if window_near:
            current_formation_name = "vertical"
            formation_offsets = formation_vertical_5()
            current_alt = window_near["z"]
        else:
            current_formation_name = "default"
            formation_offsets = formation_line_5()
            current_alt = altitudes[current_wp_idx-1]

        # detect switch
        if current_formation_name != previous_formation_name:
            formation_switch_count += 1
            print(f"[Formation Change] Switched from {previous_formation_name.upper()} to {current_formation_name.upper()} at step {step}. Total switches: {formation_switch_count}")
            if previous_formation_name == "default" and current_formation_name != "default":
                switch_start_time = time.time()
                switch_detect_ts.append(switch_start_time)
                in_nondefault_formation = True
                print(f"[Switch Timing] Started tracking switch latency at step {step}")
            elif in_nondefault_formation and previous_formation_name != "default" and current_formation_name == "default":
                reformation_duration = time.time() - switch_start_time
                reformation_times.append(reformation_duration)
                print(f"[Reformation] Time to return to default: {reformation_duration:.2f} s")
                in_nondefault_formation = False

        previous_formation_name = current_formation_name

        # compute positions
        positions = [(leader_pos[0]+dx, leader_pos[1]+dy)
                     for dx,dy in formation_offsets]

        # path length accumulation
        for i, (px, py) in enumerate(positions):
            x0, y0 = prev_positions[i]
            if i < len(path_length):
                path_length[i] += math.hypot(px-x0, py-y0)
        prev_positions = positions.copy()

        # window traversal timing
        for wi,w in enumerate(windows):
            dw = math.hypot(leader_pos[0]-w["center"][0], leader_pos[1]-w["center"][1])
            if wi not in window_entry and dw < formation_threshold:
                window_entry[wi] = time.time()
                print(f"[Window Entry] Entering window {wi} area at step {step}")
            if wi in window_entry and dw > formation_threshold + 0.5:
                traversal_time = time.time() - window_entry.pop(wi)
                traversal_times.append(traversal_time)
                print(f"[Window Exit] Exited window {wi} area, traversal time: {traversal_time:.2f}s")

        # command swarm
        swarm.move_swarm(positions, altitude=current_alt, speed=2.0)

        # capture switch latency once positions settle
        if switch_detect_ts and formation_stable(prev_positions, positions):
            lat = time.time() - switch_detect_ts.pop()
            switch_latency.append(lat)
            print(f"[Switch Latency] Formation switch completed in {lat:.3f} seconds")

        # record CPU usage
        cpu_samples.append(time.process_time() - cpu_start)

        time.sleep(0.005)
        step += 1

    # landing
    print("=" * 40)
    print("Final waypoint reached. Landing swarm.")
    swarm.land()
    swarm.shutdown()
    rclpy.shutdown()
    time1 = time.time()
    total_time = time1 - time0
    success = True


    # Display summary
    print(f"\n[METRICS SUMMARY]")
    print(f"Success: {1 if success else 0}")
    print(f"Total run time: {total_time:.3f}s")
    print(f"Formation switches: {formation_switch_count}")
    if traversal_times:
        print(f"Average Traversal Time: {sum(traversal_times)/len(traversal_times):.3f} sec")
    if reformation_times:
        print(f"Average reformation time: {sum(reformation_times)/len(reformation_times):.3f}s")
    if switch_latency:
        print(f"Average switch latency: {sum(switch_latency)/len(switch_latency):.3f}s")
    print(f"Average path length: {sum(path_length)/n_drones:.3f}m")
    print(f"Average CPU Load: {sum(cpu_samples)/len(cpu_samples):.6f}")


    # ── WRITE METRICS ─────────────────────────────────────────────────────
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    with open("results/stage2_cent.txt","a") as f:
        f.write(f"-------------------------------------------------------------------\n")
        f.write(f"Stage 2: Centralized Leader-Follower Window Traversal\n")
        f.write(f"Success: {1 if success else 0}\n")
        f.write(f"Total time: {total_time:.3f} sec\n")
        if traversal_times:
            f.write(f"Average Traversal Time: {sum(traversal_times)/len(traversal_times):.3f} sec\n")
        f.write(f"Average Reformation Time: {sum(reformation_times)/len(reformation_times):.3f} sec\n")
        if switch_latency:
            f.write(f"Average Switch Latency: {sum(switch_latency)/len(switch_latency):.3f} sec\n")
        avg_path = sum(path_length)/n_drones
        f.write(f"Average Path Length: {avg_path:.3f} m\n")
        f.write(f"Average CPU Load: {sum(cpu_samples)/len(cpu_samples):.6f}\n")
    print("Metrics written to results/stage2_cent.txt")
    

if __name__ == '__main__':
    main()

