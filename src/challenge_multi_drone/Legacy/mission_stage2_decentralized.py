#!/usr/bin/env python3
import sys, math, time, argparse
import rclpy
import random
import yaml
from as2_msgs.msg import YawMode, BehaviorStatus
from as2_python_api.drone_interface import DroneInterface
from as2_python_api.behavior_actions.behavior_handler import BehaviorHandler
from std_msgs.msg import ColorRGBA
from mission_formations import FormationSwarmConductor, formation_line_3, formation_line_5, formation_vertical_3, formation_vertical_5

with open("scenarios/scenario1_stage2.yaml", "r") as f:
    scenario_data = yaml.safe_load(f)


# ------------------- LOCAL APF + SEPARATION SETUP -------------------

def inside_window(x, y, w_center, gap_w, gap_h=1.0):
    half_w = gap_w/2.0
    half_h = gap_h/2.0
    return abs(x - w_center[0]) <= half_w and abs(y - w_center[1]) <= half_h

def compute_local_force(p_i, goal, windows, neighbors, current_phase, desired_sep=0.4, drone_id=None):
    """
    Compute each drone's local APF + neighbor separation force.
    p_i: (x,y) this drone
    goal:   global end waypoint
    walls:  list of (wx_min,wx_max,wy_min,wy_max)
    windows: list of dicts {center, gap_width, gap_height}
    neighbors: list of other drone positions
    """
    x,y = p_i
    Fx = Fy = 0.0

    # 1) Attraction to goal
    dx, dy = goal[0]-x, goal[1]-y
    d_goal = math.hypot(dx,dy)
    if d_goal>1e-3:
        kx, ky = 0.9, 0.6
        Fx_att = kx*dx
        Fy_att = ky*dy
        Fx += Fx_att
        Fy += Fy_att 
        print(f"[Drone {drone_id}] Attraction → dx={dx:.3f}, dy={dy:.3f}, F_att=({Fx_att:.3f},{Fy_att:.3f})")

    # 2) Window‐edge correction only
    #    If the drone is within a vertical band of the window but x is outside the gap,
    #    push it horizontally toward the gap.  Likewise in y.
    kw = 1.3               # gain for window‐edge correction
    infl = 0.9             # how far outside the gap we still apply correction
    if current_phase != 2:
        for widx, w in enumerate(windows):
            cx, cy = w["center"]
            half_w = w["gap_width"]  / 2.0
            half_h = w["gap_height"] / 2.0

            
            # horizontal correction if y is near the window vertically
            # if abs(y - cy) < half_h + infl:
            print(f"[Drone {drone_id}] Window {widx}: cx={cx:.2f}, cy={cy:.2f}, half_w={half_w:.2f}")
            left_edge  = cx - half_w
            right_edge = cx + half_w
            if x < left_edge:
                dxw = left_edge - x
                Fx += kw * dxw * 2
                print(f"[Drone {drone_id}] Window horiz‑corr → x<{left_edge:.2f}, Fx+={kw*dxw:.3f}")
            elif x > right_edge:
                dxw = right_edge - x
                Fx += kw * dxw * 2
                print(f"[Drone {drone_id}] Window horiz‑corr → x>{right_edge:.2f}, Fx+={kw*dxw:.3f}")

    print(f"[Drone {drone_id}] Total local force = ({Fx:.3f}, {Fy:.3f}) at pos ({x:.3f},{y:.3f})\n")
    return Fx, Fy


# ------------------- MAIN -------------------
# --------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stage2: Fully-Decentralized APF + Separation")
    parser.add_argument('-n','--namespaces', nargs='+',
                        default=['drone0','drone1','drone2'],
                        help='Drone namespaces')
    parser.add_argument('-s','--use_sim_time', action='store_true', default=True)
    args = parser.parse_args()

    rclpy.init()
    swarm = FormationSwarmConductor(args.namespaces,
                                   use_sim_time=args.use_sim_time)

    # Stage2 geometry
    stage_center = scenario_data["stage2"]["stage_center"]  # [x0, y0]
    windows_raw = scenario_data["stage2"]["windows"]

    # Windows coordinates
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
            "alt": win["distance_floor"] + win["height"] / 2.0,  # midpoint height
            "thickness": win["thickness"]
        }
        windows.append(window)
        print(f"Window {widx}: center={window['center']}, gap_width={window['gap_width']}, gap_height={window['gap_height']}, alt={window['alt']}")

    # Waypoints (start, end)
    start_wp = (0.0,   0.0)
    end_wp =   (0.0, -10.0)

    # Define some parameters
    dt = 0.2
    max_speed = 1.5
    inter_drone_dx = 0.4
    altitude = 2.5

    # Initialize per-drone positions from a line formation at start
    # offsets = formation_line_3()
    offsets = formation_line_5()

    # Define positions
    positions = [(start_wp[0]+dx, start_wp[1]+dy) for dx,dy in offsets]

    # Arm & takeoff
    if not swarm.get_ready():
        sys.exit("Arming failed")
    swarm.takeoff(height=altitude, speed=0.7)

    # Define and move to starting position
    swarm.move_swarm(positions, altitude=altitude, speed=1.0)

    # Set flags
    passed_first_window = False
    passed_second_window = False

    inter_drone_dx = max(abs(off[0] - offsets[j][0])
                     for i,off in enumerate(offsets)
                     for j in range(len(offsets)))

    # Define three base waypoints
    wp_window1 = windows[0]["center"]
    wp_window2 = windows[1]["center"]
    wp_end     = end_wp

    # track which base waypoint we’re heading toward
    current_phase = 0  # 0→window1, 1→window2, 2→end
    prev_phase = 0
    spread_mode_active = False
    desired_sep = 0.36

    # Main loop: each drone computes its own force
    for step in range(300):
        print("---------------------------")
        print(f"Step {step}:")
        changed_phase = False


        # 0) Decide which window is active
        # avg_y = sum(py for (px,py) in positions) / len(positions)
        y_pos = [p[1] for p in positions]
        if not passed_first_window and all(y < windows[0]["center"][1] - 0.25 for y in y_pos):
            passed_first_window = True
            # swarm.move_swarm(positions, altitude=active_window["alt"], speed=1.0)
            print(f"→ Passed window 0 at step {step}, now targeting window 1")
        if not passed_second_window and all(y < windows[1]["center"][1] - 0.25 for y in y_pos):
            passed_second_window = True
            print(f"→ Passed window 1 at step {step}, now targeting end")

        # Only consider the one active window in our force call
        active_window = windows[1] if passed_first_window else windows[0]
        half_w = active_window["gap_width"]/2.0
        window_to_use = [active_window]

        # in the main loop, after you decide passed_first_window etc:
        if not passed_first_window:
            current_phase = 0
        elif not passed_second_window:  # you’d compute this same as passed_first_window
            current_phase = 1
        else:
            current_phase = 2
        
        print(f"→ Current phase: {current_phase}, passed1={passed_first_window}, passed2={passed_second_window}")

        # Check previous phase
        if current_phase != prev_phase:
            spread_mode_active = False
            prev_phase = current_phase
            changed_phase = True
        
        # pick the “base” for this phase
        base_wp = [wp_window1, wp_window2, wp_end][current_phase]

        new_positions = []
        # window_w = active_window["gap_width"]
        # For each drone i:
        xs = [p[0] for p in positions]
        swarm_dia_x = max(xs) - min(xs)

        # latch spread_mode once it triggers, and never clear until phase change
        if not spread_mode_active and swarm_dia_x > active_window["gap_width"]:
            spread_mode_active = True
            print(f"→ Entering SPREAD mode at step {step}")

        spread_mode = spread_mode_active

        new_positions = []
        for i, pi in enumerate(positions):
            # pick your normal lateral offset
            offx, offy = offsets[i]
            Fx, Fy = 0.0, 0.0

            #drone_name = args.namespaces[i]  # e.g., 'drone0'
            #offx, offy = offset_map[drone_name]
            neighbors = positions[:i] + positions[i+1:]

            # Determine if the drone is close enough to the window center
            cx, cy = active_window["center"]
            dist_to_window_y = pi[1] - cy

            if current_phase != 2 and spread_mode and dist_to_window_y < 3:
                cx, cy = active_window["center"]
                vert_sep = inter_drone_dx + 0.05
                idx = i - (len(offsets)-1)/2
                print(f"[Drone {i}] idx={idx}, offsets={offsets[i]}, vert_sep={vert_sep:.2f}")

                # Spread vertically
                if idx <= 0:
                    target_y = cy - abs(idx*vert_sep) - 0.35
                    Fy = 0.03 * abs(idx)
                if idx > 0:
                    target_y = cy - abs(idx*vert_sep) - 0.10
                    Fy = 0.03 * abs(idx)

                # If drone is already passing the window, start reforming laterally
                if pi[1] < cy - 0.25:
                    target_x = cx + abs(idx)*offx
                    target_y = cy - 0.55
                    print(f"[Drone {i}] SPREAD mode recomposing → target=({target_x:.2f}, {target_y:.2f})")
                else:
                    target_x = cx
                    print(f"[Drone {i}] SPREAD mode → target=({target_x:.2f}, {target_y:.2f})")

                target = (target_x, target_y)
            else:
                # regular per-drone goal at window-center + offset
                target = ( base_wp[0] + offx,
                           base_wp[1] + offy - 0.6)
                print(f"[Drone {i}] Normal target: {target}, dist_to_wind={dist_to_window_y:.2f}")
            
            # print(f"[Drone {i}] Target: {target}, spread_mode={spread_mode}")
            # now compute your local force toward that target:
            Fx_, Fy_ = compute_local_force(
                pi, target, window_to_use, neighbors,
                current_phase, desired_sep, drone_id=i
            )
            Fx += Fx_
            Fy += Fy_
            # … clamp/integrate exactly as before …
            xi, yi = pi[0] + Fx*dt, pi[1] + Fy*dt
            new_positions.append((xi, yi))

        positions = new_positions


        # If changed phase, adjust the formation using the same y offsets as the previous phase but with adjusted x offsets
        if changed_phase:
            print(f"→ Reforming formation at phase {current_phase}")
            
            # Use the average current Y to preserve vertical continuity
            avg_y = sum(p[1] for p in positions) / len(positions)
            reform_center = (base_wp[0], avg_y)

            # Calculate desired positions from formation offsets
            # target_positions_1 = [(reform_center[0] + dx, reform_center[1] + dy - 0.02)
                                # for dx, dy in offsets]
            target_positions = [(reform_center[0] + dx, reform_center[1] + dy - 0.05)
                                for dx, dy in offsets]
            
            N = 3  # number of intermediate steps
            alt_start = altitude  # current altitude
            alt_end = active_window["alt"]  # target altitude

            for i in range(1, N + 1):
                alpha = i / N
                # Interpolate positions
                intermediate_positions = [
                    (p1[0] * (1 - alpha) + p2[0] * alpha,
                    p1[1] * (1 - alpha) + p2[1] * alpha)
                    for p1, p2 in zip(positions, target_positions)
                ]
                # Interpolate altitude
                intermediate_alt = alt_start * (1 - alpha) + alt_end * alpha
                # Move swarm
                swarm.move_swarm(intermediate_positions, altitude=intermediate_alt, speed=1.0)
                time.sleep(0.005)  # Slightly longer delay for smoother transition

            # Move swarm instantly to new positions (adjust altitude too)
            # swarm.move_swarm(target_positions_1, altitude=active_window["alt"], speed=1.0)
            # time.sleep(0.05)  # optional: allow time to stabilize
            positions = target_positions

        # Send each drone its individually computed position
        swarm.move_swarm(positions, altitude=active_window["alt"], speed=1.0)
        time.sleep(0.005)

        # After sending move_swarm and sleeping
        if current_phase == 2:
            # Reuse the same formation offsets to define per-drone goals
            # final_offsets = formation_line_3()  # or whichever final formation you’re using
            final_offsets = formation_line_5()
            all_reached = True
            for (pos, offset) in zip(positions, final_offsets):
                goal_x = end_wp[0] + offset[0]
                goal_y = end_wp[1] + offset[1]
                dist = math.hypot(pos[0] - goal_x, pos[1] - goal_y)
                if dist > 0.3:
                    all_reached = False
                    break
            if all_reached:
                print("→ All drones reached their individual end-goals in Phase 2; exiting loop.")
                break

    swarm.land()
    swarm.shutdown()
    rclpy.shutdown()
    sys.exit(0)

if __name__=='__main__':
    main()

