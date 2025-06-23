#!/usr/bin/env python3
import math
import time
import heapq
import numpy as np
from typing import List, Dict, Tuple, Set, Optional

# Grid and time discretization constants
GRID_RESOLUTION = 0.2    # Grid cell size (m)
TIME_RESOLUTION = 0.1    # Time step size (s)
TIME_HORIZON = 3.0       # Planning horizon (s)
MAX_CT_EXPANSIONS = 100  # Maximum CT node expansions per planning cycle
SAFETY_RADIUS = 0.3      # Safety distance from obstacles (m)
MAX_LOW_LEVEL_EXPANSIONS = 1000  # Maximum A* iterations for low-level search

class GridNode:
    """Node in the time-expanded grid for low-level search"""
    def __init__(self, x: int, y: int, t: int, parent=None):
        self.x = x          # Grid x coordinate
        self.y = y          # Grid y coordinate
        self.t = t          # Time step
        self.parent = parent
        self.g = 0          # Cost from start
        self.h = 0          # Heuristic to goal
        self.f = 0          # Total cost (g + h)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.t == other.t
    
    def __hash__(self):
        return hash((self.x, self.y, self.t))
    
    def __lt__(self, other):
        # For priority queue
        if self.f == other.f:
            return self.h < other.h
        return self.f < other.f

class Constraint:
    """Constraint forbidding an agent from occupying a cell or edge at a specific time"""
    def __init__(self, agent_id: int, x: int, y: int, t: int, x2: int = None, y2: int = None):
        self.agent_id = agent_id  # Agent ID this constraint applies to
        self.x = x                # Grid x coordinate
        self.y = y                # Grid y coordinate
        self.t = t                # Time step
        self.x2 = x2              # For edge constraint: destination x
        self.y2 = y2              # For edge constraint: destination y
    
    def is_vertex_constraint(self):
        """Check if this is a vertex constraint (vs. edge constraint)"""
        return self.x2 is None or self.y2 is None
    
    def conflicts_with(self, agent_id: int, x: int, y: int, t: int, prev_x: int = None, prev_y: int = None) -> bool:
        """Check if this constraint conflicts with a given agent state or move"""
        if self.agent_id != agent_id:
            return False
            
        if self.is_vertex_constraint():
            # Vertex constraint: forbids (agent_id, x, y, t)
            return self.x == x and self.y == y and self.t == t
        else:
            # Edge constraint: forbids (agent_id, x, y, t) -> (agent_id, x2, y2, t+1)
            return (self.x == prev_x and self.y == prev_y and 
                    self.x2 == x and self.y2 == y and 
                    self.t == t - 1)

class Conflict:
    """Represents a conflict between two agents"""
    def __init__(self, agent1: int, agent2: int, x: int, y: int, t: int, 
                 x1_prev: int = None, y1_prev: int = None,
                 x2_prev: int = None, y2_prev: int = None):
        self.agent1 = agent1  # First agent ID
        self.agent2 = agent2  # Second agent ID
        self.x = x            # Conflict location x
        self.y = y            # Conflict location y
        self.t = t            # Conflict time step
        
        # For edge conflicts (agents swap positions)
        self.x1_prev = x1_prev
        self.y1_prev = y1_prev
        self.x2_prev = x2_prev
        self.y2_prev = y2_prev
    
    def is_vertex_conflict(self):
        """Check if this is a vertex conflict (vs. edge conflict)"""
        return (self.x1_prev is None or self.y1_prev is None or
                self.x2_prev is None or self.y2_prev is None)
                
    def get_constraints(self) -> Tuple[Constraint, Constraint]:
        """Get the two alternative constraints to resolve this conflict"""
        if self.is_vertex_conflict():
            # Vertex conflict: two agents at the same position at the same time
            return (
                Constraint(self.agent1, self.x, self.y, self.t),
                Constraint(self.agent2, self.x, self.y, self.t)
            )
        else:
            # Edge conflict: two agents swap positions
            return (
                Constraint(self.agent1, self.x, self.y, self.t, self.x1_prev, self.y1_prev),
                Constraint(self.agent2, self.x, self.y, self.t, self.x2_prev, self.y2_prev)
            )

class CTNode:
    """Node in the Constraint Tree (CT) for high-level search"""
    def __init__(self, paths: Dict[int, List[GridNode]] = None, constraints: List[Constraint] = None):
        self.paths = paths or {}            # Maps agent_id to its path
        self.constraints = constraints or [] # List of constraints
        self.cost = sum(len(path) for path in self.paths.values()) if self.paths else 0
        
    def __lt__(self, other):
        # For priority queue - expand lowest cost nodes first
        return self.cost < other.cost

class CBSPlanner:
    """
    Conflict-Based Search planner for multi-agent path finding with dynamic obstacles
    """
    def __init__(self, grid_resolution=GRID_RESOLUTION, time_resolution=TIME_RESOLUTION):
        self.grid_resolution = grid_resolution
        self.time_resolution = time_resolution
        
        # Grid bounds
        self.bounds = [-10.0, 10.0, -10.0, 10.0]  # Default bounds
        
        # Directions for low-level search (including "wait in place")
        self.directions = [
            (0, 0),   # Wait
            (1, 0),   # Right
            (0, 1),   # Up
            (-1, 0),  # Left
            (0, -1),  # Down
            (1, 1),   # Up-Right
            (-1, 1),  # Up-Left
            (-1, -1), # Down-Left
            (1, -1)   # Down-Right
        ]
        
        # Current CT root for reuse
        self.current_ct_root = None
        self.last_planning_time = 0
        
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        return (round(x / self.grid_resolution), round(y / self.grid_resolution))
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        return (grid_x * self.grid_resolution, grid_y * self.grid_resolution)
    
    def is_collision_with_obstacles(self, x: int, y: int, t: int, obstacles) -> bool:
        """Check if grid cell (x,y) at time t collides with any dynamic obstacle"""
        world_x, world_y = self.grid_to_world(x, y)
        
        for obs in obstacles:
            if hasattr(obs, 'predict_position'):
                # Time relative to obstacle prediction start
                obs_time = self.last_planning_time + t * self.time_resolution
                
                # Predict obstacle position at this time
                obs_pos = obs.predict_position(obs_time)
                
                # Check for collision (x,y are in grid coordinates)
                distance = math.hypot(world_x - obs_pos[0], world_y - obs_pos[1])
                if distance < (SAFETY_RADIUS + obs.radius):
                    return True
            else:
                # Simple position tuple or other obstacle type
                obs_pos = obs[:2] if len(obs) >= 2 else obs
                distance = math.hypot(world_x - obs_pos[0], world_y - obs_pos[1])
                if distance < SAFETY_RADIUS:
                    return True
                    
        return False
    
    def get_obstacle_constraints(self, obstacles, max_timesteps=30) -> List[Constraint]:
        """Generate constraints for all obstacles at all timesteps in planning horizon"""
        constraints = []
        
        # Dummy agent ID for obstacles (-1)
        obs_agent_id = -1
        
        for t in range(max_timesteps):
            for obs in obstacles:
                if hasattr(obs, 'predict_position'):
                    # Time relative to obstacle prediction start
                    obs_time = self.last_planning_time + t * self.time_resolution
                    
                    # Predict obstacle position at this time
                    obs_pos = obs.predict_position(obs_time)
                    
                    # Convert to grid
                    grid_x, grid_y = self.world_to_grid(obs_pos[0], obs_pos[1])
                    
                    # Add constraint to avoid this cell
                    constraints.append(Constraint(obs_agent_id, grid_x, grid_y, t))
                    
                    # For larger obstacles, also forbid neighboring cells
                    if hasattr(obs, 'radius') and obs.radius > self.grid_resolution:
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                if dx == 0 and dy == 0:
                                    continue
                                constraints.append(Constraint(
                                    obs_agent_id, grid_x + dx, grid_y + dy, t))
        
        return constraints
    
    def heuristic(self, node: GridNode, goal: GridNode) -> float:
        """Diagonal distance heuristic for A*"""
        dx = abs(node.x - goal.x)
        dy = abs(node.y - goal.y)
        # Diagonal shortcut cost
        return max(dx, dy) + 0.41421 * min(dx, dy)
    
    def is_valid_state(self, x: int, y: int, t: int, agent_id: int, constraints: List[Constraint], 
                      prev_x: int = None, prev_y: int = None) -> bool:
        """Check if state (x,y,t) is valid for the given agent"""
        # Check grid bounds
        if not (self.bounds[0] <= x * self.grid_resolution <= self.bounds[1] and 
                self.bounds[2] <= y * self.grid_resolution <= self.bounds[3]):
            return False
            
        # Check constraints
        for constraint in constraints:
            if constraint.conflicts_with(agent_id, x, y, t, prev_x, prev_y):
                return False
                
        return True
    
    def low_level_search(self, agent_id: int, start: Tuple[float, float], goal: Tuple[float, float],
                        constraints: List[Constraint], obstacles, start_time: int = 0) -> List[GridNode]:
        """
        A* search for a single agent with constraints and obstacle avoidance
        Returns a path as a list of GridNodes or an empty list if no path found
        """
        # Convert world coordinates to grid
        start_grid = self.world_to_grid(start[0], start[1])
        goal_grid = self.world_to_grid(goal[0], goal[1])
        
        # Create start and goal nodes
        start_node = GridNode(start_grid[0], start_grid[1], start_time)
        goal_node = GridNode(goal_grid[0], goal_grid[1], 0)  # t doesn't matter for goal heuristic
        
        # Initialize start node
        start_node.g = 0
        start_node.h = self.heuristic(start_node, goal_node)
        start_node.f = start_node.g + start_node.h
        
        # Initialize open and closed sets
        open_list = []
        heapq.heappush(open_list, start_node)
        open_dict = {(start_node.x, start_node.y, start_node.t): start_node}  # For faster lookup
        closed_set = set()
        
        # Extract obstacle constraints
        obstacle_constraints = self.get_obstacle_constraints(obstacles)
        all_constraints = constraints + obstacle_constraints
        
        # A* main loop
        iterations = 0
        max_iterations = MAX_LOW_LEVEL_EXPANSIONS
        max_timestep = int(TIME_HORIZON / self.time_resolution)
        
        while open_list and iterations < max_iterations:
            iterations += 1
            
            # Get node with lowest f score
            current = heapq.heappop(open_list)
            del open_dict[(current.x, current.y, current.t)]
            
            # Check if goal reached
            if current.x == goal_node.x and current.y == goal_node.y:
                # Reconstruct and return path
                return self._reconstruct_path(current)
            
            # Add to closed set if not already there
            current_key = (current.x, current.y, current.t)
            if current_key in closed_set:
                continue
            
            closed_set.add(current_key)
            
            # Skip if exceeds time horizon
            if current.t >= max_timestep:
                continue
            
            # Generate successors
            for dx, dy in self.directions:
                # Calculate successor coordinates
                successor_x = current.x + dx
                successor_y = current.y + dy
                successor_t = current.t + 1
                successor_key = (successor_x, successor_y, successor_t)
                
                # Skip if already evaluated
                if successor_key in closed_set:
                    continue
                
                # Check if valid and collision-free
                if not self.is_valid_state(successor_x, successor_y, successor_t, 
                                          agent_id, all_constraints, current.x, current.y):
                    continue
                
                if self.is_collision_with_obstacles(successor_x, successor_y, successor_t, obstacles):
                    continue
                
                # Calculate tentative g score
                if dx == 0 and dy == 0:  # Waiting
                    tentative_g = current.g + 0.5  # Small penalty for waiting
                elif abs(dx) + abs(dy) == 1:  # Cardinal movement
                    tentative_g = current.g + 1.0
                else:  # Diagonal movement
                    tentative_g = current.g + 1.414
                
                # If this successor is already in open list, check if this path is better
                if successor_key in open_dict:
                    successor_node = open_dict[successor_key]
                    if tentative_g < successor_node.g:
                        # This is a better path - update the node
                        successor_node.g = tentative_g
                        successor_node.f = tentative_g + successor_node.h
                        successor_node.parent = current
                        
                        # Reheapify
                        heapq.heapify(open_list)
                else:
                    # New node, add to open list
                    successor_node = GridNode(successor_x, successor_y, successor_t, current)
                    successor_node.g = tentative_g
                    successor_node.h = self.heuristic(successor_node, goal_node)
                    successor_node.f = successor_node.g + successor_node.h
                    
                    heapq.heappush(open_list, successor_node)
                    open_dict[successor_key] = successor_node
        
        # No path found after max iterations
        if iterations >= max_iterations:
            print(f"A* search reached max iterations ({max_iterations}) without finding a path")
        elif not open_list:
            print("A* search exhausted all possible nodes without finding a path")
        
        # If we can't reach the exact goal, find the closest node to goal
        if open_list:
            # Find the closest node to goal in the open list
            closest_node = min(open_list, key=lambda n: self.heuristic(n, goal_node))
            print(f"Returning best partial path to goal (heuristic distance: {closest_node.h:.2f})")
            return self._reconstruct_path(closest_node)
        
        # No path found and no nodes left to explore
        return []
    
    def _reconstruct_path(self, node: GridNode) -> List[GridNode]:
        """Reconstruct path from A* result"""
        path = []
        current = node
        
        while current:
            path.append(current)
            current = current.parent
            
        # Reverse to get path from start to goal
        return path[::-1]
    
    def find_conflicts(self, paths: Dict[int, List[GridNode]]) -> Optional[Conflict]:
        """Find the first conflict in a set of paths"""
        # Check each pair of agents
        agents = list(paths.keys())
        
        for i in range(len(agents)):
            agent1 = agents[i]
            path1 = paths[agent1]
            
            for j in range(i + 1, len(agents)):
                agent2 = agents[j]
                path2 = paths[agent2]
                
                # Determine the maximum timestep to check
                max_t = min(len(path1), len(path2))
                
                # Check for vertex conflicts
                for t in range(max_t):
                    if path1[t].x == path2[t].x and path1[t].y == path2[t].y:
                        # Vertex conflict: same position at same time
                        return Conflict(agent1, agent2, path1[t].x, path1[t].y, t)
                
                # Check for edge conflicts (agents swapping positions)
                for t in range(1, max_t):
                    if (path1[t].x == path2[t-1].x and path1[t].y == path2[t-1].y and
                        path1[t-1].x == path2[t].x and path1[t-1].y == path2[t].y):
                        # Edge conflict: agents swap positions
                        return Conflict(
                            agent1, agent2, 
                            path1[t].x, path1[t].y, t,
                            path1[t-1].x, path1[t-1].y,
                            path2[t-1].x, path2[t-1].y
                        )
        
        # No conflicts found
        return None
    
    def plan(self, start_states: Dict[int, Tuple[float, float]], 
             goal_states: Dict[int, Tuple[float, float]], 
             obstacles, time_start: float) -> Dict[int, Tuple[float, float]]:
        """
        Plan paths for multiple agents avoiding collisions
        Returns a dictionary mapping agent_id to next position (x,y) in world coords
        """
        start_time = time.time()
        self.last_planning_time = time_start
        
        # Cap maximum iterations based on time constraints
        max_ct_expansions = MAX_CT_EXPANSIONS
        
        # Initialize variables
        high_level_expansions = 0
        best_node = None
        
        # Initialize or reuse CT root
        if self.current_ct_root is None:
            # Create new root node
            root = CTNode({}, [])
            
            # Compute initial paths
            for agent_id, start_pos in start_states.items():
                path = self.low_level_search(
                    agent_id, start_pos, goal_states[agent_id], [], obstacles
                )
                
                if path:
                    root.paths[agent_id] = path
            
            self.current_ct_root = root
        else:
            # Reuse previous CT but check if we need to update paths
            root = self.current_ct_root
            
            # Update paths that may have changed
            for agent_id, start_pos in start_states.items():
                if agent_id in root.paths:
                    # If agent has moved significantly from path start, recompute
                    path_start = root.paths[agent_id][0]
                    grid_start = self.world_to_grid(start_pos[0], start_pos[1])
                    
                    if (abs(path_start.x - grid_start[0]) > 1 or 
                        abs(path_start.y - grid_start[1]) > 1):
                        # Recompute path from current position
                        path = self.low_level_search(
                            agent_id, start_pos, goal_states[agent_id], 
                            root.constraints, obstacles
                        )
                        if path:
                            root.paths[agent_id] = path
                else:
                    # New agent, compute initial path
                    path = self.low_level_search(
                        agent_id, start_pos, goal_states[agent_id], 
                        root.constraints, obstacles
                    )
                    if path:
                        root.paths[agent_id] = path
        
        # Check if we already have a valid solution (no conflicts)
        conflict = self.find_conflicts(root.paths)
        if not conflict:
            print("CBS: Initial solution has no conflicts!")
            self.current_ct_root = root
            return self._extract_next_positions(root.paths)
        
        # High-level search
        open_list = []
        heapq.heappush(open_list, root)
        
        while open_list and high_level_expansions < max_ct_expansions:
            high_level_expansions += 1
            
            # Get lowest cost node
            current = heapq.heappop(open_list)
            best_node = current  # Keep track of best node in case we time out
            
            # Check for conflicts
            conflict = self.find_conflicts(current.paths)
            
            if not conflict:
                # No conflict - we've found a valid solution
                print(f"CBS: Found conflict-free solution after {high_level_expansions} expansions")
                self.current_ct_root = current
                return self._extract_next_positions(current.paths)
            
            # Generate child nodes by adding constraints to resolve the conflict
            constraints = conflict.get_constraints()
            
            for i in range(2):
                # Create new constraint set
                child_constraints = current.constraints + [constraints[i]]
                
                # Create new node
                child = CTNode(
                    {agent_id: path.copy() for agent_id, path in current.paths.items()},
                    child_constraints
                )
                
                # Replan path for the constrained agent
                agent_id = constraints[i].agent_id
                
                if agent_id in start_states:
                    new_path = self.low_level_search(
                        agent_id, 
                        start_states[agent_id],
                        goal_states[agent_id],
                        child_constraints, 
                        obstacles
                    )
                    
                    if new_path:
                        # Update path and cost
                        child.paths[agent_id] = new_path
                        child.cost = sum(len(path) for path in child.paths.values())
                        
                        # Add to open list
                        heapq.heappush(open_list, child)
        
        # If we reached the maximum iterations without finding a conflict-free solution,
        # use the best node we've seen so far
        print(f"CBS: Reached max iterations ({high_level_expansions}), returning best partial solution")
        
        if best_node:
            self.current_ct_root = best_node
            return self._extract_next_positions(best_node.paths)
        else:
            # If no valid node found, fall back to direct paths
            return self._extract_next_positions(root.paths)
    
    def _extract_next_positions(self, paths: Dict[int, List[GridNode]]) -> Dict[int, Tuple[float, float]]:
        """Extract next positions for each agent from CBS solution"""
        next_positions = {}
        
        for agent_id, path in paths.items():
            if path and len(path) > 1:
                # Use the second position in the path (first is current position)
                grid_x, grid_y = path[1].x, path[1].y
                world_x, world_y = self.grid_to_world(grid_x, grid_y)
                next_positions[agent_id] = (world_x, world_y)
            elif path and len(path) == 1:
                # If path has only one node, use it (stay in place)
                grid_x, grid_y = path[0].x, path[0].y
                world_x, world_y = self.grid_to_world(grid_x, grid_y)
                next_positions[agent_id] = (world_x, world_y)
        
        return next_positions 