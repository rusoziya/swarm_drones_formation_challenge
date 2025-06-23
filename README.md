# Swarm Drones Formation Challenge

This repository implements and evaluates centralized and decentralized multi-drone formation strategies across four progressively complex scenarios, using ROS 2, Aerostack2, and Crazyflie quadrotors. Detailed descriptions, hyperparameter analyses, performance metrics, and discussion of trade-offs are drawn from our full coursework report. For folder-specific launch, configuration, and detailed usage, please see the individual README.md files in each source directory (e.g., `src/aerostack2/README.md`, `src/challenge_multi_drone/README.md`).

---

## 📂 Repository Structure

```bash
swarm_drones_formation_challenge/
├── src/
│   ├── aerostack2/                  # Aerostack2 stack customizations and launch files
│   ├── as2_platform_crazyflie/      # Crazyflie bridge for Aerostack2
│   ├── challenge_multi_drone/       # Multi-drone formation mission scripts (focus of this README)
│   │   ├── demo/                       # GIF demonstrations for each scenario
│   │   ├── mission_formations.py        # Defines six formation patterns as relative offsets
│   │   ├── mission_stage1_centralised.py     # Stage 1 centralized pattern switching node
│   │   ├── mission_stage1_decentralised.py   # Stage 1 decentralized consensus node
│   │   ├── mission_stage2_central.py         # Stage 2 centralized narrow-aperture traversal
│   │   ├── mission_stage2_decentralized.py   # Stage 2 decentralized APF + edge-repulsion
│   │   ├── mission_stage3_central.py         # Stage 3 centralized APF forest traversal
│   │   ├── mission_stage3_decentralized.py   # Stage 3 decentralized RRT + local correction
│   │   ├── mission_stage4_centralised.py     # Stage 4 centralized CBS + time-indexed A*
│   │   ├── mission_stage4_decentralised.py   # Stage 4 decentralized ORCA avoidance
│   │   ├── setup.bash                        # Launch all ROS 2 nodes & Gazebo environment
│   │   └── stop.bash                         # Graceful shutdown script
│   ├── build/                        # Colcon build outputs
│   ├── install/                      # Install space for overlay
│   └── log/                          # ROS 2 logs
├── COMP0240_CW2.pdf                  # Full coursework report with methods & quantitative analysis
├── .gitignore
└── README.md (this file)
```

## Detailed Challenge Descriptions & Key Findings

Below we summarize each mission stage, separating the analysis for centralized and decentralized approaches. We include hyperparameters, performance metrics, key observations, and visual demonstrations. For full experiment logs, plots, and parameter sweeps, refer to Sections 3–5 of the report (COMP0240_CW2.pdf).

---

### Stage 1 – Adaptive Formation Patterns

**Scenario:**  
Five drones follow a circular 12 m radius trajectory at 2 m altitude, switching among six predefined patterns (Line, V, Diamond, Circle, Grid, Staggered) every 30°.

#### Centralized Approach
- **Script:** `mission_stage1_centralised.py`
- **Control Strategy:** Leader-follower model where a single node computes trajectories for all drones and broadcasts commands, ensuring tight temporal coordination.
- **Key Hyperparameters:**
  - Waypoint Angular Step (Δθ): 30°
  - Drone Speed: 0.7 m/s
  - Stabilization Pause: 0.5 s
- **Performance Metrics:**
  - Mission Time: 136.3 s
  - Max Formation Error: 2.25 m
  - Total CPU Load: 423.9 s (on the leader node)
- **Observations:**  
  Provides excellent synchronization and precision, completing the mission faster with lower formation error. However, it concentrates the entire computational load on one drone, creating a single point of failure.

#### Decentralized Approach
- **Script:** `mission_stage1_decentralised.py`
- **Control Strategy:** Each drone runs an identical autonomy stack, using a swarm-wide heartbeat (20 Hz) and a leader-election protocol to achieve consensus on formation and trajectory phase.
- **Key Hyperparameters:**
  - Heartbeat Rate: 20 Hz
  - Leader Election Timeout: 0.6 s
  - Drone Speed: 0.7 m/s
- **Performance Metrics:**
  - Mission Time: ~215.0 s
  - Max Formation Error: 4.70 m
  - Total CPU Load: ~55.0 s (across 5 agents)
- **Observations:**  
  Significantly more robust—loss of the leader triggers re-election without halting the mission. Trade-off is longer mission time and greater formation error due to communication latency.

#### Visualization: Stage 1
- [GIF Demo of Centralized Stage 1](src/challenge_multi_drone/docs/centralized_stage1.gif)
- [GIF Demo of Decentralized Stage 1](src/challenge_multi_drone/docs/decentralized_stage1.gif)

*Export to Sheets*

---

### Stage 2 – Passage Through Narrow Apertures

**Scenario:**  
A swarm of five drones traverses two narrow, 1.5 m-wide windows positioned 8 m apart, requiring reconfiguration from a line formation to a vertical stack and back.

#### Centralized Approach
- **Script:** `mission_stage2_central.py`
- **Control Strategy:** Leader drone detects proximity to the window (1.2 m) and triggers a formation switch for the entire swarm using pre-computed offsets.
- **Key Hyperparameters:**
  - Speed: 3.0 m/s
  - Controller Timestep (dt): 0.3 s
  - Formation Switch Threshold: 1.2 m
- **Performance Metrics:**
  - Success Rate: 100%
  - Total Mission Time: 89.5 s
  - Switch Latency: 3.6 s
- **Observations:**  
  Executes formation switches very quickly and cohesively due to a single command source, resulting in low latency.

#### Decentralized Approach
- **Script:** `mission_stage2_decentralized.py`
- **Control Strategy:** Each drone uses an Artificial Potential Field (APF) to navigate. Repulsive forces from window edges (k_rep = 0.8) and attractive forces toward the goal guide each agent independently.
- **Key Hyperparameters:**
  - APF Attractive Gains (k_x, k_y): 1.1, 0.8
  - APF Repulsive Gain (k_rep): 0.8
- **Performance Metrics:**
  - Success Rate: 100%
  - Total Mission Time: 83.6 s
  - Reformation Time: 17.4 s
- **Observations:**  
  Faster overall mission and reformation due to local, parallel adjustments. Introduces higher jitter and slightly increased switch latency.

#### Visualization: Stage 2
- [GIF Demo of Centralized Stage 2](src/challenge_multi_drone/docs/centralized_stage2.gif)
- [GIF Demo of Decentralized Stage 2](src/challenge_multi_drone/docs/decentralized_stage2.gif)

*Export to Sheets*

---

### Stage 3 – Dense Forest Traversal

**Scenario:**  
Navigate a 20×20 m area cluttered with 20 randomly placed cylindrical obstacles (radius 0.5 m). Drones must switch formations in safe zones to pass.

#### Centralized Approach
- **Script:** `mission_stage3_central.py`
- **Control Strategy:** Leader plans a path for the entire swarm using an APF planner (10 Hz), applying repulsive forces from trees and between drones.
- **Key Hyperparameters:**
  - Repulsive Gains: Tree = 30, Neighbor = 15
  - Waypoint Smoothing Look-ahead: 0.8
- **Performance Metrics:**
  - Success Rate: 100%
  - Total Time: 93.7 s
  - Reformation Time: 26.0 s
- **Observations:**  
  Avoids local minima reliably, but reactive path corrections affect the whole swarm, leading to longer mission and reformation times.

#### Decentralized Approach
- **Script:** `mission_stage3_decentralized.py`
- **Control Strategy:** Each agent independently plans its path using RRT and applies local potential field corrections for obstacles within 1.2 m.
- **Key Hyperparameters:**
  - RRT Step Size: 0.5 m
  - RRT Goal Bias: 0.2
  - Local Correction Radius: 1.2 m
- **Performance Metrics:**
  - Success Rate: 100%
  - Total Time: 58.7 s
  - Reformation Time: 5.7 s
- **Observations:**  
  Parallel RRT planning speeds traversal and concurrent local adjustments yield very fast reformation. Higher per-agent CPU load is the trade-off.

#### Visualization: Stage 3
- [GIF Demo of Centralized Stage 3](src/challenge_multi_drone/docs/centralized_stage3.gif)
- [GIF Demo of Decentralized Stage 3](src/challenge_multi_drone/docs/decentralized_stage3.gif)

*Export to Sheets*

---

### Stage 4 – Dynamic Obstacle Avoidance

**Scenario:**  
Five drones maintain a V-formation while avoiding five unpredictably moving balloons in a 15×15 m arena.

#### Centralized Approach
- **Script:** `mission_stage4_centralised.py`
- **Control Strategy:** High-level planner uses Conflict-Based Search (CBS) on a time-indexed A* grid (0.5 m resolution), replanning at 20 Hz with formation compression primitives.
- **Key Hyperparameters:**
  - Grid Resolution: 0.5 m
  - Replanning Frequency: 20 Hz
  - CBS Iteration Limit: 50
- **Performance Metrics:**
  - Success Rate: 80%
  - Mission Time: 53.7 s
  - RMS Formation Error: 1.10 m
- **Observations:**  
  Precise and fast with tight formation, but computationally intensive and sensitive to replanning failures; clearances can be as tight as 1.2 cm.

#### Decentralized Approach
- **Script:** `mission_stage4_decentralised.py`
- **Control Strategy:** Each agent uses Optimal Reciprocal Collision Avoidance (ORCA) to compute collision-free velocities, with a soft formation constraint (bias = 0.1).
- **Key Hyperparameters:**
  - ORCA Time Horizons (Agent/Obstacle): 6 s / 4 s
  - ORCA Agent Radius: 0.625 m
  - Formation Bias: 0.1
- **Performance Metrics:**
  - Success Rate: 60%
  - Mission Time: 93.3 s
  - RMS Formation Error: 7.11 m
- **Observations:**  
  Lightweight and safe with larger clearances (1.9 cm), but local-only reasoning yields lower success rate and significant formation drift.

#### Visualization: Stage 4
- [GIF Demo of Centralized Stage 4](src/challenge_multi_drone/docs/centralized_stage4.gif)
- [GIF Demo of Decentralized Stage 4](src/challenge_multi_drone/docs/decentralized_stage4.gif)

*Export to Sheets*

---

## Further Reading & Report

For full mathematical derivations, hyperparameter sweep plots, and extended discussions, see **COMP0240_CW2.pdf**:

- **Section 3:** Stage designs & algorithms  
- **Section 4:** Experimental setup & parameter tuning  
- **Section 5:** Quantitative results & comparative analysis  

---

## License

MIT © 2025 University College London  
