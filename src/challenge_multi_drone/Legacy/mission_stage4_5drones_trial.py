#!/usr/bin/env python3
import sys
import math
import time
import random
import argparse
import rclpy
import yaml
from typing import List, Tuple, Dict, Optional
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from as2_msgs.msg import YawMode, BehaviorStatus
from as2_python_api.drone_interface import DroneInterface
from as2_python_api.behavior_actions.behavior_handler import BehaviorHandler
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import PoseStamped

class DynamicObstacle:
    """Representation of a dynamic obstacle with position, velocity and predicted trajectory."""
    def __init__(self, pos_x: float, pos_y: float, pos_z: float, vel_x: float, vel_y: float, vel_z: float):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.vel_x = vel_x
        self.vel_y = vel_y
        self.vel_z = vel_z
        self.radius = 0.5  # From scenario file
        self.safety_buffer = 0.1  # Additional safety margin
        self.effective_radius = self.radius + self.safety_buffer
        
    def predict_position(self, time_delta: float) -> Tuple[float, float, float]:
        """Predict position of obstacle after time_delta seconds."""
        future_x = self.pos_x + self.vel_x * time_delta
        future_y = self.pos_y + self.vel_y * time_delta
        future_z = self.pos_z + self.vel_z * time_delta
        return (future_x, future_y, future_z)
    
    def will_collide_with(self, pos: Tuple[float, float, float], time_horizon: float = 2.0, 
                          time_steps: int = 10) -> bool:
        """Check if obstacle will collide with position within time horizon."""
        for i in range(time_steps):
            t = i * (time_horizon / time_steps)
            obstacle_pos = self.predict_position(t)
            distance = math.sqrt((obstacle_pos[0] - pos[0])**2 + 
                                (obstacle_pos[1] - pos[1])**2 + 
                                (obstacle_pos[2] - pos[2])**2)
            if distance < self.effective_radius:
                return True
        return False

class DynamicObstacleTracker(Node):
    """Node to track and predict dynamic obstacles."""
    def __init__(self):
        super().__init__('dynamic_obstacle_tracker')
        
        # Dictionary to store current obstacles
        self.obstacles: Dict[str, DynamicObstacle] = {}
        
        # Subscribe to dynamic obstacle locations
        self.obstacle_sub = self.create_subscription(
            PoseStamped,
            '/dynamic_obstacles/locations',
            self.obstacle_callback,
            10
        )
        
        # Last update timestamp for velocity calculation
        self.last_update_time = None
        self.prev_positions = {}
        
        self.get_logger().info('Dynamic obstacle tracker initialized')
    
    def obstacle_callback(self, msg: PoseStamped):
        """Process incoming obstacle data and update obstacle tracking."""
        obstacle_id = msg.header.frame_id  # Assuming frame_id contains unique obstacle ID
        current_time = self.get_clock().now().seconds_nanoseconds()[0]
        
        pos_x = msg.pose.position.x
        pos_y = msg.pose.position.y
        pos_z = msg.pose.position.z
        
        # Calculate velocity if we have previous position data
        vel_x, vel_y, vel_z = 0.0, 0.0, 0.0
        
        if obstacle_id in self.prev_positions and self.last_update_time is not None:
            prev_pos = self.prev_positions[obstacle_id]
            time_delta = current_time - self.last_update_time
            
            if time_delta > 0:
                vel_x = (pos_x - prev_pos[0]) / time_delta
                vel_y = (pos_y - prev_pos[1]) / time_delta
                vel_z = (pos_z - prev_pos[2]) / time_delta
        
        # Update obstacle data
        self.obstacles[obstacle_id] = DynamicObstacle(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z)
        
        # Store current position for next velocity calculation
        self.prev_positions[obstacle_id] = (pos_x, pos_y, pos_z)
        self.last_update_time = current_time
    
    def get_obstacle_positions(self) -> List[Tuple[float, float, float]]:
        """Return current positions of all tracked obstacles."""
        return [(obs.pos_x, obs.pos_y, obs.pos_z) for obs in self.obstacles.values()]
    
    def get_obstacles(self) -> List[DynamicObstacle]:
        """Return all tracked obstacles."""
        return list(self.obstacles.values())

class FormationDancer(DroneInterface):
    """A modified drone interface for formation flight with dynamic obstacle avoidance."""
    def __init__(self, namespace: str, verbose: bool = False, use_sim_time: bool = False):
        super().__init__(namespace, verbose=verbose, use_sim_time=use_sim_time)
        self._speed = 1.0  # Default speed set to 1.0 m/s
        self._yaw_mode = YawMode.PATH_FACING
        self._yaw_angle = None
        self._frame_id = "earth"
        self.current_behavior: BehaviorHandler = None
        self.led_pub = self.create_publisher(ColorRGBA, f"/{namespace}/leds/control", 10)
        self.current_position = (0.0, 0.0, 0.0)  # Initialize position
        self._target_position = None

    def update_position(self, x: float, y: float, z: float):
        """Update stored position of the drone."""
        self.current_position = (x, y, z)
    
    def get_position(self) -> Tuple[float, float, float]:
        """Return current drone position."""
        return self.current_position
    
    def set_target(self, x: float, y: float, z: float):
        """Set target position for the drone."""
        self._target_position = (x, y, z)
    
    def get_target(self) -> Optional[Tuple[float, float, float]]:
        """Get target position of the drone."""
        return self._target_position

    def change_led_colour(self, colour: Tuple[int, int, int]):
        """Change LED color to indicate drone status."""
        msg = ColorRGBA()
        msg.r = colour[0] / 255.0
        msg.g = colour[1] / 255.0
        msg.b = colour[2] / 255.0
        self.led_pub.publish(msg)

    def do_behavior(self, beh, *args) -> None:
        """Start behavior and store it to later check completion."""
        print(f"[{self.namespace}] do_behavior: {beh} with args {args}")
        self.current_behavior = getattr(self, beh)
        self.current_behavior(*args)

    def go_to_position(self, x, y, z, speed=1.0) -> None:
        """Command the drone to move to a specific position."""
        print(f"[{self.namespace}] go_to_position called with x={x}, y={y}, z={z}, speed={speed}")
        self.set_target(x, y, z)
        self.do_behavior("go_to",
                         x, y, z,
                         speed,
                         self._yaw_mode,
                         self._yaw_angle,
                         self._frame_id,
                         False)
        # Update current position (assuming the command will execute)
        self.update_position(x, y, z)

    def goal_reached(self) -> bool:
        """Check if the current behavior has finished (IDLE)."""
        if not self.current_behavior:
            return False
        return self.current_behavior.status == BehaviorStatus.IDLE

class DynamicPathPlanner:
    """Path planner for dynamic environments using rapidly exploring random trees (RRT*)."""
    
    def __init__(self, 
                 start: Tuple[float, float, float],
                 goal: Tuple[float, float, float],
                 bounds: Tuple[float, float, float, float],  # (x_min, x_max, y_min, y_max)
                 drone_radius: float = 0.3,
                 step_size: float = 0.5,
                 max_iterations: int = 2000,
                 goal_sample_rate: float = 0.1,
                 collision_check_resolution: float = 0.1):
        
        self.start = start
        self.goal = goal
        self.bounds = bounds
        self.drone_radius = drone_radius
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_sample_rate = goal_sample_rate
        self.collision_check_resolution = collision_check_resolution
    
    def plan(self, obstacles: List[DynamicObstacle], time_horizon: float = 3.0) -> List[Tuple[float, float, float]]:
        """Generate a path avoiding dynamic obstacles within time horizon."""
        # Initialize the tree with start node
        nodes = [RRTNode(self.start[0], self.start[1], self.start[2])]
        path_found = False
        
        # Try to find a path within max_iterations
        for _ in range(self.max_iterations):
            # With some probability, sample the goal
            if random.random() < self.goal_sample_rate:
                sample = self.goal
            else:
                # Sample a random point within bounds
                sample = (
                    random.uniform(self.bounds[0], self.bounds[1]),  # x
                    random.uniform(self.bounds[2], self.bounds[3]),  # y
                    self.start[2]  # Keep z constant
                )
            
            # Find nearest node
            nearest = self.nearest_node(nodes, sample)
            
            # Steer toward the sample
            new_node = self.steer(nearest, sample)
            
            # Check if new_node is collision-free considering dynamic obstacles
            if self.is_collision_free(nearest, new_node, obstacles, time_horizon):
                nodes.append(new_node)
                
                # Check if we've reached the goal
                if self.distance((new_node.x, new_node.y, new_node.z), self.goal) < self.step_size:
                    goal_node = RRTNode(self.goal[0], self.goal[1], self.goal[2], parent=new_node)
                    nodes.append(goal_node)
                    path_found = True
                    break
        
        if path_found:
            # Extract path from tree
            path = []
            current = nodes[-1]  # Goal node
            while current:
                path.append((current.x, current.y, current.z))
                current = current.parent
            return path[::-1]  # Reverse to get start-to-goal order
        else:
            print("Warning: Path not found within iteration limit")
            return []
    
    def nearest_node(self, nodes: List['RRTNode'], point: Tuple[float, float, float]) -> 'RRTNode':
        """Find the node in nodes closest to point."""
        return min(nodes, key=lambda n: self.distance((n.x, n.y, n.z), point))
    
    def steer(self, from_node: 'RRTNode', to_point: Tuple[float, float, float]) -> 'RRTNode':
        """Create a new node by stepping from from_node toward to_point."""
        d = self.distance((from_node.x, from_node.y, from_node.z), to_point)
        
        if d < self.step_size:
            return RRTNode(to_point[0], to_point[1], to_point[2], parent=from_node)
        else:
            # Normalize direction vector
            dx = (to_point[0] - from_node.x) / d
            dy = (to_point[1] - from_node.y) / d
            dz = (to_point[2] - from_node.z) / d
            
            # Step in that direction
            new_x = from_node.x + dx * self.step_size
            new_y = from_node.y + dy * self.step_size
            new_z = from_node.z + dz * self.step_size
            
            return RRTNode(new_x, new_y, new_z, parent=from_node)
    
    def distance(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
    
    def is_collision_free(self, from_node: 'RRTNode', to_node: 'RRTNode', obstacles: List[DynamicObstacle], 
                          time_horizon: float) -> bool:
        """Check if path between nodes is collision-free with dynamic obstacles."""
        # Calculate distance and direction
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dz = to_node.z - from_node.z
        dist = self.distance((from_node.x, from_node.y, from_node.z), (to_node.x, to_node.y, to_node.z))
        
        # Check path with resolution steps
        steps = max(2, int(dist / self.collision_check_resolution))
        
        for i in range(steps + 1):
            t = i / steps  # Parameter along the line [0, 1]
            
            # Interpolate point along path
            x = from_node.x + t * dx
            y = from_node.y + t * dy
            z = from_node.z + t * dz
            point = (x, y, z)
            
            # Estimate time to reach this point
            time_to_point = t * (dist / 1.0)  # Assuming 1.0 m/s
            
            # Check collision with each obstacle at this point in time
            for obstacle in obstacles:
                # Predict obstacle position at this time
                obs_pos = obstacle.predict_position(time_to_point)
                
                # Check collision
                obstacle_dist = self.distance(point, obs_pos)
                if obstacle_dist < obstacle.effective_radius + self.drone_radius:
                    return False  # Collision detected
        
        return True  # No collision detected along path

class RRTNode:
    """Node structure for RRT path planning."""
    def __init__(self, x: float, y: float, z: float, parent: Optional['RRTNode'] = None):
        self.x = x
        self.y = y
        self.z = z
        self.parent = parent

class FormationManager:
    """Manages drone formations and reconfigurations."""
    
    def __init__(self, num_drones: int = 5, spacing: float = 0.5):
        self.num_drones = num_drones
        self.spacing = spacing
        self.formation = self.v_formation()
        
    def v_formation(self) -> List[Tuple[float, float, float]]:
        """Create a V formation with specified spacing."""
        offsets = []
        for i in range(self.num_drones):
            if i == 0:  # Leader at center front
                offsets.append((0.0, 0.0, 0.0))
            elif i % 2 == 1:  # Odd indices on left wing
                wing_pos = (i + 1) // 2
                offsets.append((-self.spacing * wing_pos, self.spacing * wing_pos, 0.0))
            else:  # Even indices on right wing
                wing_pos = i // 2
                offsets.append((self.spacing * wing_pos, self.spacing * wing_pos, 0.0))
        return offsets
    
    def apply_formation(self, center_pos: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """Apply formation offsets to the given center position."""
        positions = []
        for offset in self.formation:
            positions.append((
                center_pos[0] + offset[0],
                center_pos[1] + offset[1],
                center_pos[2] + offset[2]
            ))
        return positions
    
    def deform_for_obstacles(self, positions: List[Tuple[float, float, float]], 
                             obstacles: List[DynamicObstacle], 
                             time_horizon: float = 3.0) -> List[Tuple[float, float, float]]:
        """
        Deform formation to avoid obstacles while keeping as close to original formation as possible.
        """
        new_positions = positions.copy()
        
        # Check each drone position against obstacles
        for i, pos in enumerate(positions):
            avoid_vector = [0.0, 0.0, 0.0]
            position_modified = False
            
            # Check if any obstacle will collide with this position
            for obstacle in obstacles:
                if obstacle.will_collide_with(pos, time_horizon):
                    # Calculate avoidance vector
                    t_collision = 0  # Find approximate time of collision
                    for t in range(int(time_horizon * 10)):
                        t_check = t / 10.0
                        obs_pos = obstacle.predict_position(t_check)
                        distance = math.sqrt((obs_pos[0] - pos[0])**2 + 
                                            (obs_pos[1] - pos[1])**2 + 
                                            (obs_pos[2] - pos[2])**2)
                        if distance < obstacle.effective_radius + 0.3:  # Drone radius + safety
                            t_collision = t_check
                            break
                    
                    # Get obstacle position at collision time
                    obs_pos = obstacle.predict_position(t_collision)
                    
                    # Direction away from obstacle
                    dir_x = pos[0] - obs_pos[0]
                    dir_y = pos[1] - obs_pos[1]
                    dir_z = pos[2] - obs_pos[2]
                    
                    # Normalize
                    mag = math.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
                    if mag > 0:
                        dir_x /= mag
                        dir_y /= mag
                        dir_z /= mag
                    
                    # Add avoidance vector (stronger for closer collisions)
                    avoid_strength = 1.0  # Base strength
                    avoid_vector[0] += avoid_strength * dir_x
                    avoid_vector[1] += avoid_strength * dir_y
                    avoid_vector[2] += avoid_strength * dir_z
                    position_modified = True
            
            # Apply avoidance vector if needed
            if position_modified:
                # Limit maximum deformation
                mag = math.sqrt(avoid_vector[0]**2 + avoid_vector[1]**2 + avoid_vector[2]**2)
                if mag > 1.5:  # Limit max deviation
                    avoid_vector[0] = (avoid_vector[0] / mag) * 1.5
                    avoid_vector[1] = (avoid_vector[1] / mag) * 1.5
                    avoid_vector[2] = (avoid_vector[2] / mag) * 1.5
                
                new_positions[i] = (
                    pos[0] + avoid_vector[0],
                    pos[1] + avoid_vector[1],
                    pos[2] + avoid_vector[2]
                )
        
        return new_positions
    
    def check_formation_collisions(self, positions: List[Tuple[float, float, float]], 
                                   min_separation: float = 0.4) -> bool:
        """Check if any drones in the formation would collide with each other."""
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dist = math.sqrt((positions[i][0] - positions[j][0])**2 + 
                                 (positions[i][1] - positions[j][1])**2 + 
                                 (positions[i][2] - positions[j][2])**2)
                if dist < min_separation:
                    return True  # Collision detected
        return False  # No collisions

    def adjust_formation_altitudes(self, positions: List[Tuple[float, float, float]], 
                                  base_altitude: float = 2.5,
                                  min_separation: float = 0.4) -> List[Tuple[float, float, float]]:
        """
        Adjust drone altitudes when lateral separation is insufficient.
        """
        new_positions = positions.copy()
        
        # First pass: identify conflicting pairs
        conflicts = []
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                # Calculate 2D distance (ignore altitude)
                dist_2d = math.sqrt(
                    (positions[i][0] - positions[j][0])**2 + 
                    (positions[i][1] - positions[j][1])**2
                )
                if dist_2d < min_separation:
                    conflicts.append((i, j, dist_2d))
        
        # Sort conflicts by severity (ascending distance)
        conflicts.sort(key=lambda x: x[2])
        
        # Resolve conflicts by separating in altitude
        altitude_adjustments = [0.0] * len(positions)
        for i, j, _ in conflicts:
            # Check if either drone already has an altitude adjustment
            if altitude_adjustments[i] == 0 and altitude_adjustments[j] == 0:
                # Neither has adjustment yet, separate them
                altitude_adjustments[i] = 0.5  # Move up
                altitude_adjustments[j] = -0.5  # Move down
            elif altitude_adjustments[i] == 0:
                # j already adjusted, adjust i in opposite direction
                altitude_adjustments[i] = -altitude_adjustments[j]
            elif altitude_adjustments[j] == 0:
                # i already adjusted, adjust j in opposite direction
                altitude_adjustments[j] = -altitude_adjustments[i]
            else:
                # Both already adjusted, make sure they're opposite
                if altitude_adjustments[i] * altitude_adjustments[j] > 0:
                    # Same direction, flip one
                    altitude_adjustments[j] = -altitude_adjustments[j]
        
        # Apply altitude adjustments
        for i, adj in enumerate(altitude_adjustments):
            if adj != 0:
                new_positions[i] = (
                    positions[i][0],
                    positions[i][1],
                    base_altitude + adj
                )
            else:
                # Keep base altitude for drones without conflicts
                new_positions[i] = (
                    positions[i][0],
                    positions[i][1],
                    base_altitude
                )
        
        return new_positions

class FormationSwarmConductor:
    """Manages multiple drones in formation with dynamic obstacle avoidance."""
    
    def __init__(self, drones_ns: List[str], verbose: bool = False, use_sim_time: bool = False):
        self.drones = []
        for ns in drones_ns:
            self.drones.append(FormationDancer(ns, verbose, use_sim_time))
        
        # Initialize formation manager
        self.formation_manager = FormationManager(num_drones=len(drones_ns), spacing=0.5)
        
        # Initialize obstacle tracker
        self.obstacle_tracker = DynamicObstacleTracker()
        
        # Current drone positions
        self.current_positions = [(0.0, 0.0, 0.0)] * len(drones_ns)
        
        # Base flight altitude
        self.default_altitude = 2.5
    
    def shutdown(self):
        """Shutdown all drone interfaces."""
        for d in self.drones:
            d.shutdown()
    
    def wait_all(self):
        """Wait for all drones to reach their current goals."""
        all_done = False
        while not all_done:
            time.sleep(0.01)
            all_done = all(d.goal_reached() for d in self.drones)
        print("All drones reached goal.")
    
    def get_ready(self) -> bool:
        """Arm and set offboard mode for all drones."""
        success = True
        for d in self.drones:
            success_arm = d.arm()
            success_offboard = d.offboard()
            success = success and success_arm and success_offboard
        return success
    
    def takeoff(self, height=2.5, speed=0.7):
        """Command all drones to take off."""
        for d in self.drones:
            d.do_behavior("takeoff", height, speed, False)
            d.change_led_colour((0, 255, 0))  # Green during takeoff
        self.wait_all()
        
        # Update current positions after takeoff
        for i, d in enumerate(self.drones):
            self.current_positions[i] = (0.0, 0.0, height)  # Assuming takeoff is at origin
    
    def land(self, speed=0.4):
        """Command all drones to land."""
        for d in self.drones:
            d.do_behavior("land", speed, False)
        self.wait_all()
    
    def move_swarm(self, positions, speed=1.0):
        """Command each drone to move to its corresponding position."""
        for i, (d, pos) in enumerate(zip(self.drones, positions)):
            print(f"Moving {d.namespace} to ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
            d.go_to_position(pos[0], pos[1], pos[2], speed=speed)
            self.current_positions[i] = pos
        print("Waiting for all drones to reach goal...")
        self.wait_all()
    
    def plan_path(self, start, goal, obstacles):
        """Plan a path from start to goal avoiding obstacles."""
        # Define bounds based on stage size (10x10)
        bounds = (-5.0, 5.0, -5.0, 5.0)  # (x_min, x_max, y_min, y_max)
        
        planner = DynamicPathPlanner(
            start=start,
            goal=goal,
            bounds=bounds,
            step_size=0.5,
            max_iterations=2000
        )
        
        return planner.plan(obstacles)
    
    def execute_mission(self, start_point, end_point, replan_freq=1.0):
        """Execute mission from start to end with dynamic replanning."""
        current_center = start_point
        goal_center = end_point
        
        # Initial formation at start point
        start_formation = self.formation_manager.apply_formation(
            (current_center[0], current_center[1], self.default_altitude)
        )
        self.move_swarm(start_formation)
        
        # Main mission loop
        goal_reached = False
        last_replan_time = time.time()
        
        while not goal_reached:
            current_time = time.time()
            
            # Get current obstacles
            obstacles = self.obstacle_tracker.get_obstacles()
            
            # Replan path if needed
            if current_time - last_replan_time > replan_freq:
                print("Replanning path...")
                path = self.plan_path(
                    (current_center[0], current_center[1], self.default_altitude),
                    (goal_center[0], goal_center[1], self.default_altitude),
                    obstacles
                )
                
                if not path:
                    print("No path found, waiting and retrying...")
                    time.sleep(replan_freq)
                    continue
                
                # Take next waypoint from path
                next_waypoint = path[1] if len(path) > 1 else path[0]
                last_replan_time = current_time
                
                # Apply formation to next waypoint
                desired_positions = self.formation_manager.apply_formation(next_waypoint)
                
                # Deform formation if needed to avoid obstacles
                deformed_positions = self.formation_manager.deform_for_obstacles(
                    desired_positions, obstacles
                )
                
                # Adjust altitudes if needed to avoid inter-drone collisions
                final_positions = self.formation_manager.adjust_formation_altitudes(
                    deformed_positions, self.default_altitude
                )
                
                # Move drones to new positions
                self.move_swarm(final_positions)
                
                # Update current center
                current_center = (next_waypoint[0], next_waypoint[1])
                
                # Check if goal reached
                dist_to_goal = math.sqrt(
                    (current_center[0] - goal_center[0])**2 + 
                    (current_center[1] - goal_center[1])**2
                )
                if dist_to_goal < 0.5:  # Within 0.5m of goal
                    # Final approach to goal
                    goal_formation = self.formation_manager.apply_formation(
                        (goal_center[0], goal_center[1], self.default_altitude)
                    )
                    self.move_swarm(goal_formation)
                    goal_reached = True
            
            # Brief sleep to avoid CPU hogging
            time.sleep(0.1)

def load_scenario(filepath):
    """Load scenario data from YAML file."""
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return data

def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: Dynamic Obstacle Avoidance with Formation Flight")
    parser.add_argument('-n', '--namespaces', nargs='+',
                        default=['drone0', 'drone1', 'drone2', 'drone3', 'drone4'],
                        help='List of drone namespaces')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Enable verbose output')
    parser.add_argument('-s', '--use_sim_time', action='store_true', default=True,
                        help='Use simulation time')
    args = parser.parse_args()

    rclpy.init()

    # Create executor for handling multiple nodes
    executor = MultiThreadedExecutor()

    # Load scenario file
    scenario_file = 'scenarios/scenario1_stage4.yaml'
    try:
        scenario = load_scenario(scenario_file)
        print(f"Loaded scenario: {scenario['name']}")
    except Exception as e:
        print(f"Error loading scenario file: {e}")
        rclpy.shutdown()
        sys.exit(1)

    # Create swarm conductor
    conductor = FormationSwarmConductor(
        args.namespaces, verbose=args.verbose, use_sim_time=args.use_sim_time)

    # Add nodes to executor
    executor.add_node(conductor.obstacle_tracker)
    for drone in conductor.drones:
        executor.add_node(drone)

    # Start executor in a separate thread
    import threading
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Arm and set offboard mode
    if not conductor.get_ready():
        print("Failed to arm or set offboard mode for all drones")
        rclpy.shutdown()
        sys.exit(1)

    # Takeoff
    takeoff_height = scenario.get('takeoff_height', conductor.default_altitude)
    takeoff_speed = scenario.get('takeoff_speed', 0.7)
    conductor.takeoff(height=takeoff_height, speed=takeoff_speed)

    # Extract mission start and goal points
    start_point = tuple(scenario.get('start', [0.0, 0.0]))
    end_point = tuple(scenario.get('goal', [0.0, 0.0]))

    # Execute mission
    replan_freq = scenario.get('replan_frequency', 1.0)
    conductor.execute_mission(start_point, end_point, replan_freq=replan_freq)

    # Land
    land_speed = scenario.get('land_speed', 0.4)
    conductor.land(speed=land_speed)

    # Shutdown
    conductor.shutdown()
    executor.shutdown()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
