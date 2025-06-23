#!/usr/bin/env python3
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from cbs_planner import CBSPlanner, DynamicObstacle

def test_simple_scenario():
    """Test CBS with a simple 2-agent scenario"""
    print("Testing CBS with simple 2-agent scenario...")
    
    # Create planner
    planner = CBSPlanner()
    planner.bounds = [-5.0, 5.0, -5.0, 5.0]
    
    # Define start and goal states
    start_states = {
        0: (0.0, -2.0),  # Agent 0 starts at (0, -2)
        1: (0.0, 2.0)    # Agent 1 starts at (0, 2)
    }
    
    goal_states = {
        0: (0.0, 2.0),   # Agent 0 goal at (0, 2)
        1: (0.0, -2.0)   # Agent 1 goal at (0, -2)
    }
    
    # Create a stationary obstacle
    obstacle = DynamicObstacle(
        id="obs1",
        position=(2.0, 0.0, 1.0),
        velocity=(0.0, 0.0, 0.0),
        radius=0.5
    )
    
    # Plan paths
    start_time = time.time()
    next_positions = planner.plan(start_states, goal_states, [obstacle], start_time)
    end_time = time.time()
    
    print(f"Planning time: {(end_time - start_time) * 1000:.2f} ms")
    print(f"Next positions: {next_positions}")
    
    # Visualize the result
    visualize_plan(start_states, goal_states, next_positions, [obstacle])
    
def test_multi_agent_scenario():
    """Test CBS with a 5-agent scenario"""
    print("Testing CBS with 5-agent formation scenario...")
    
    # Create planner
    planner = CBSPlanner()
    planner.bounds = [-10.0, 10.0, -10.0, 10.0]
    
    # Define start states (V formation)
    formation_spacing = 0.5
    start_states = {
        0: (-2*formation_spacing, -2*formation_spacing),
        1: (-formation_spacing, -formation_spacing),
        2: (0.0, 0.0),
        3: (formation_spacing, -formation_spacing),
        4: (2*formation_spacing, -2*formation_spacing)
    }
    
    # Define goal states (5m ahead)
    goal_states = {
        0: (-2*formation_spacing, 5.0-2*formation_spacing),
        1: (-formation_spacing, 5.0-formation_spacing),
        2: (0.0, 5.0),
        3: (formation_spacing, 5.0-formation_spacing),
        4: (2*formation_spacing, 5.0-2*formation_spacing)
    }
    
    # Create dynamic obstacles
    obstacles = [
        DynamicObstacle(
            id="obs1",
            position=(0.0, 2.5, 1.0),
            velocity=(0.0, 0.0, 0.0),
            radius=0.5
        ),
        DynamicObstacle(
            id="obs2",
            position=(-1.0, 3.5, 1.0),
            velocity=(0.2, 0.0, 0.0),
            radius=0.5
        ),
        DynamicObstacle(
            id="obs3",
            position=(1.0, 3.5, 1.0),
            velocity=(-0.2, 0.0, 0.0),
            radius=0.5
        )
    ]
    
    # Plan paths
    start_time = time.time()
    next_positions = planner.plan(start_states, goal_states, obstacles, start_time)
    end_time = time.time()
    
    print(f"Planning time for 5 agents: {(end_time - start_time) * 1000:.2f} ms")
    print(f"Next positions: {next_positions}")
    
    # Visualize the result
    visualize_plan(start_states, goal_states, next_positions, obstacles)

def visualize_plan(start_states, goal_states, next_positions, obstacles):
    """Visualize the planning result"""
    plt.figure(figsize=(10, 8))
    
    # Plot start positions
    for agent_id, pos in start_states.items():
        plt.plot(pos[0], pos[1], 'bo', markersize=10, label=f"Start {agent_id}" if agent_id == 0 else "")
    
    # Plot goal positions
    for agent_id, pos in goal_states.items():
        plt.plot(pos[0], pos[1], 'go', markersize=10, label=f"Goal {agent_id}" if agent_id == 0 else "")
    
    # Plot next positions
    for agent_id, pos in next_positions.items():
        plt.plot(pos[0], pos[1], 'ro', markersize=6, label=f"Next {agent_id}" if agent_id == 0 else "")
        
        # Draw arrow from start to next
        if agent_id in start_states:
            start = start_states[agent_id]
            plt.arrow(
                start[0], start[1], 
                pos[0] - start[0], pos[1] - start[1],
                head_width=0.1, head_length=0.2, fc='r', ec='r'
            )
    
    # Plot obstacles
    for i, obs in enumerate(obstacles):
        circle = plt.Circle((obs.position[0], obs.position[1]), obs.radius, color='gray', alpha=0.7)
        plt.gca().add_patch(circle)
        
        # Add arrow for velocity
        if np.linalg.norm(obs.velocity[:2]) > 0:
            plt.arrow(
                obs.position[0], obs.position[1],
                obs.velocity[0], obs.velocity[1],
                head_width=0.1, head_length=0.2, fc='gray', ec='gray'
            )
    
    # Add legend and labels
    plt.legend()
    plt.grid(True)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('CBS Planner Test')
    plt.axis('equal')
    
    # Show the plot
    plt.tight_layout()
    plt.savefig('cbs_test_result.png')
    print("Plot saved as 'cbs_test_result.png'")
    plt.show()

if __name__ == "__main__":
    # Run tests
    test_simple_scenario()
    test_multi_agent_scenario() 