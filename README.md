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

# 📘 Detailed Challenge Descriptions & Key Findings

Below we summarize each mission stage with hyperparameters, performance metrics, observations, and visual GIF demonstrations located in `src/challenge_multi_drone/demo/`. For full experiment logs, plots, and parameter sweeps, refer to Section 3–5 of the report (`COMP0240_CW2.pdf`).

---

### Stage 1 – Adaptive Formation Patterns

**Scenario:** Five drones follow a circular 12 m radius trajectory at 2 m altitude, switching among six predefined patterns (Line, V, Diamond, Circle, Grid, Staggered) every 30°.

| Aspect | Centralized | Decentralized |
| :--- | :--- | :--- |
| **Script** | `mission_stage1_centralised.py` | `mission_stage1_decentralised.py` |
| **Control** | Leader–follower; central node computes offsets and broadcasts | Consensus Swarm-wide heartbeat (20 Hz) + election |
| **Waypoint dt** | 0.5 s | 0.5 s |
| **Drone speed** | 0.7 m/s | 0.7 m/s |

**Metrics & Tuning**

| Metric | Centralized | Decentralized |
| :--- | :--- | :--- |
| **Mission time** | 136.3 s | 215.0 s |
| **Max formation error** | 2.25 m | 4.70 m |
| **CPU load (total loop time)**| 423.9 s | 55.0 s (shared across 5 agents) |

* **Key hyperparameters:**
    * Pause duration per pattern: 0.5 s (stability vs. speed)
    * Heartbeat rate: 20 Hz; election timeout: 0.6 s
* **Observation:** Centralized offers tighter synchronization but risks full mission failure on leader loss. Decentralized gracefully continues with increased scatter.

**Visualization**

| Centralized | Decentralized |
| :--- | :--- |
| *[GIF Demo]* | *[GIF Demo]* |

---

### Stage 2 – Passage Through Narrow Apertures

**Scenario:** Traverse two 1.5 m wide windows positioned 8 m apart. Drones reconfigure between line (enter) and vertical stack (exit).

| Metric | Centralized | Decentralized |
| :--- | :--- | :--- |
| **Success rate** | 100% | 100% |
| **Total mission time** | 89.5 s | 83.6 s |
| **Reformation time** | 23.6 s | 17.4 s |
| **Switch latency** | 3.6 s | 7.5 s |
| **Avg path length** | 12.6 m | 12.9 m |
| **CPU per step** | 0.289 s | 0.231 s |

**Algorithmic details**

* **Centralized:** Proximity trigger at 1.2 m switches formation via precomputed offsets. Speed tuned at 3.0 m/s, dt=0.3 s.
* **Decentralized:** APF with repulsive gain $k_{ᵧ}=0.8$ at window edges; APF parameters ($k_{x}=1.1, k_{ᵧ}=0.8$) chosen via grid search.
* **Observation:** Decentralized yields faster reformation due to local adjustments, at the expense of higher jitter.

**Visualization**

| Centralized | Decentralized |
| :--- | :--- |
| *[GIF Demo]* | *[GIF Demo]* |

---

### Stage 3 – Dense Forest Traversal

**Scenario:** Navigate through 20 randomly placed cylindrical obstacles (r=0.5 m) in a 20×20 m area. Pattern switches at safe zones.

| Metric | Centralized | Decentralized |
| :--- | :--- | :--- |
| **Success rate** | 100% | 100% |
| **Total time** | 93.7 s | 58.7 s |
| **Reformation ops** | 4 | 4 |
| **Reform. time** | 26.0 s | 5.7 s |
| **Path length** | 10.6 m | 11.1 m |
| **CPU per step** | 0.213 s | 0.351 s |

**Algorithmic details**

* **Centralized:** APF planner at 10 Hz with repulsive gains {tree:30, neighbor:15}, waypoint smoothing look-ahead=0.8.
* **Decentralized:** RRT per agent (step=0.5 m, bias=0.2), local PF correction if obstacles within 1.2 m.
* **Observation:** Decentralized reduces reformation latency via parallel planning; centralized avoids local minima but has longer corrections.

**Visualization**

| Centralized | Decentralized |
| :--- | :--- |
| *[GIF Demo]* | *[GIF Demo]* |

---

### Stage 4 – Dynamic Obstacle Avoidance

**Scenario:** Five balloons move unpredictably in a 15×15 m arena. Drones maintain V formation while avoiding collisions.

| Metric | Centralized | Decentralized |
| :--- | :--- | :--- |
| **Success rate** | 80% | 60% |
| **Mission time** | 53.7 s | 93.3 s |
| **CPU per step** | 0.120 s | 0.361 s |
| **RMS formation error** | 1.10 m | 7.11 m |
| **Min clearance (m)** | 0.012 | 0.019 |

**Algorithmic details**

* **Centralized:** CBS on time-indexed A* grid (0.5 m), replanning 20 Hz, CBS limit=50. Formation compression primitives.
* **Decentralized:** ORCA with formation bias=0.1, radius=0.625 m, horizons (Th=6 s, To=4 s), updates at 10 Hz.
* **Observation:** Centralized is precise and fast but heavy on replanning; decentralized is safer but drifts under density.

**Visualization**

| Centralized | Decentralized |
| :--- | :--- |
| *[GIF Demo]* | *[GIF Demo]* |

---

### 📄 Further Reading & Report

For full mathematical derivations, hyperparameter sweep plots, and extended discussion of trade-offs, see **COMP0240_CW2.pdf**. Key sections:

* **Section 3:** Stage designs & algorithms
* **Section 4:** Experimental setup & parameter tuning
* **Section 5:** Quantitative results & comparative analysis

---

### License

MIT © 2025 University College London
