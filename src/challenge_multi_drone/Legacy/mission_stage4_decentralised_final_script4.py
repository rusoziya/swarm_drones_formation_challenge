#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Multi-Drone Mission Script for Dynamic Obstacle Avoidance (Stage 4)
#
# This script enables 5 drones to navigate from a starting position to a goal
# position while avoiding dynamic obstacles in real-time. The obstacles' positions
# are obtained from the ROS topic /dynamic_obstacles/locations/posestamped.
#
# Each drone uses ORCA (Optimal Reciprocal Collision Avoidance) for decentralized
# control with soft formation constraints.
# ------------------------------------------------------------------------------

# ------------------------
# Configuration (Modifiable Parameters)
# ------------------------

# Drone motion parameters
TAKE_OFF_HEIGHT = 1.0      # Height in meters at takeoff 
TAKE_OFF_SPEED = 1.0       # m/s for takeoff 
SLEEP_TIME = 0.05          # Minimal delay between commands (seconds) 
SPEED = 2.0                # m/s during flight 
LAND_SPEED = 1.0          # m/s for landing 

# Altitude adjustment parameters
MIN_ALTITUDE = 0.5         # Minimum permitted flying altitude (m)
MAX_ALTITUDE = 1.5         # Maximum permitted flying altitude (m)

# ORCA Parameters
ORCA_TIME_HORIZON = 3.0           # Time horizon for reciprocal collision avoidance (seconds)
ORCA_TIME_HORIZON_OBSTACLE = 2.0  # Time horizon for obstacle avoidance (seconds) - increased from 2.0 to 4.0
ORCA_RADIUS = 0.4                 # Radius of agent (meters)
ORCA_MAX_SPEED = 4             # Maximum speed of agent (m/s)
ORCA_NEIGHBOR_DISTANCE = 2.0      # Maximum distance to consider neighbors (meters)
ORCA_OBSTACLE_DISTANCE = 8.0      # Maximum distance to consider obstacles (meters) - increased from 6.0 to 8.0
FORMATION_WEIGHT = 0.1            # Weight of formation constraints (0-1) - reduced from 0.8 to 0.6 to prioritize obstacle avoidance

# System parameters
REPLAN_FREQUENCY = 5.0            # How often to replan (in Hz) - reduced to 5Hz to ensure control loop can finish
SAFETY_MARGIN = 0.25               # Additional margin (in meters) added around each obstacle

# Number of drones
NUM_DRONES = 5

# ------------------------
# Imports and Setup
# ------------------------
import argparse
import time
import math
import yaml
import logging
import numpy as np
import rclpy
import os
import threading
import json
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped
from queue import Queue

from as2_python_api.drone_interface import DroneInterface

# Remove plotting libraries - we don't need visualization
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.patches as patches
# from matplotlib.lines import Line2D

# ------------------------
# Helper Functions
# ------------------------
def load_scenario(scenario_file):
    """Load scenario from a YAML file."""
    with open(scenario_file, 'r') as f:
        scenario = yaml.safe_load(f)
    return scenario

def compute_euclidean_distance(p1, p2):
    """Compute Euclidean distance between two 3D points."""
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def interpolate_angle(a, b, t):
    """
    Interpolate between angles a and b (in radians) by fraction t.
    Handles angle wrap-around.
    """
    diff = (b - a + math.pi) % (2*math.pi) - math.pi
    return a + diff * t

def predict_obstacle_position(obstacle_data, prediction_time):
    """
    Predict the position of an obstacle after a certain time,
    assuming constant velocity.
    """
    pos = obstacle_data["position"].copy()
    vel = obstacle_data.get("velocity", [0, 0, 0])
    
    # Simple linear prediction
    pos[0] += vel[0] * prediction_time
    pos[1] += vel[1] * prediction_time
    
    return pos

def predict_obstacle_trajectory(obstacle_data, lookahead_time, steps=20):
    """
    Generate a trajectory of positions for an obstacle during the entire lookahead time period.
    Returns a list of predicted positions at different time points.
    """
    pos = obstacle_data["position"].copy()
    vel = obstacle_data.get("velocity", [0, 0, 0])
    diameter = obstacle_data.get("diameter", 0.5)
    
    # Generate trajectory points at evenly spaced intervals during the lookahead period
    trajectory = []
    for t in np.linspace(0, lookahead_time, steps):
        pred_pos = [
            pos[0] + vel[0] * t,
            pos[1] + vel[1] * t,
            pos[2] + vel[2] * t
        ]
        trajectory.append(pred_pos)
    
    return trajectory

def is_state_valid(state, dynamic_obstacles, lookahead_time=0.0):
    """
    Checks whether a given (x, y, z) state is free of collisions with dynamic obstacles.
    Each obstacle is treated as a cylinder with the specified diameter and height.
    If lookahead_time > 0, checks against all possible positions during that time period.
    """
    x, y, z = state[0], state[1], state[2]
    
    # Check if within altitude bounds
    if z < MIN_ALTITUDE or z > MAX_ALTITUDE:
        return False
    
    # If no obstacles data yet, assume it's valid
    if not dynamic_obstacles:
        return True
    
    for obs_id, obs_data in dynamic_obstacles.items():
        # Get obstacle parameters
        diameter = obs_data.get("diameter", 0.5)
        effective_radius = (diameter / 2.0) + SAFETY_MARGIN
        height = obs_data.get("height", 5.0)
        
        if lookahead_time > 0:
            # Check against the entire trajectory
            obstacle_trajectory = predict_obstacle_trajectory(obs_data, lookahead_time)
            
            for pred_pos in obstacle_trajectory:
                # Check horizontal distance (cylinder radius)
                dx = x - pred_pos[0]
                dy = y - pred_pos[1]
                horizontal_dist = math.sqrt(dx*dx + dy*dy)
                
                if horizontal_dist <= effective_radius:
                    # Check vertical distance (cylinder height)
                    if 0 <= z <= height:
                        return False
        else:
            # Just check current position (traditional approach)
            pos = obs_data["position"]
            
            # Check horizontal distance (cylinder radius)
            dx = x - pos[0]
            dy = y - pos[1]
            horizontal_dist = math.sqrt(dx*dx + dy*dy)
            
            if horizontal_dist <= effective_radius:
                # Check vertical distance (cylinder height)
                if 0 <= z <= height:
                    return False
    
    return True

# --------------------------
# ORCA Implementation
# --------------------------
class Vector2:
    """Simple 2D vector class for ORCA implementation"""
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector2(self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar):
        return Vector2(self.x / scalar, self.y / scalar)
    
    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def length_squared(self):
        return self.x * self.x + self.y * self.y
    
    def normalize(self):
        length = self.length()
        if length > 0.0001:
            return Vector2(self.x / length, self.y / length)
        return Vector2(0, 0)
    
    def to_tuple(self):
        return (self.x, self.y)

def dot(vector1, vector2):
    """Compute dot product of two Vector2 instances"""
    return vector1.x * vector2.x + vector1.y * vector2.y

def cross(vector1, vector2):
    """Compute 2D cross product of two Vector2 instances"""
    return vector1.x * vector2.y - vector1.y * vector2.x

def normalize(vector):
    """Normalize a Vector2"""
    length = math.sqrt(vector.x * vector.x + vector.y * vector.y)
    if length > 0.0001:
        return Vector2(vector.x / length, vector.y / length)
    return Vector2(0, 0)

def linear_program1(lines, lineNo, radius, optVelocity, directionOpt):
    """
    Solves a one-dimensional linear program on a specified line.
    
    Args:
        lines: Lines constraining the solution.
        lineNo: The specified line constraint.
        radius: Radius of the agent.
        optVelocity: The optimization velocity.
        directionOpt: Direction of optimization.
    
    Returns:
        True if successful, False otherwise.
    """
    dotProduct = dot(lines[lineNo].point, lines[lineNo].direction)
    discriminant = dotProduct * dotProduct + radius * radius - lines[lineNo].point.length_squared()
    
    if discriminant < 0.0:
        # Max speed circle fully invalidates line lineNo.
        return False
    
    sqrtDiscriminant = math.sqrt(discriminant)
    tLeft = -dotProduct - sqrtDiscriminant
    tRight = -dotProduct + sqrtDiscriminant
    
    for i in range(lineNo):
        denominator = cross(lines[lineNo].direction, lines[i].direction)
        
        if abs(denominator) <= 0.0001:
            # Lines lineNo and i are (almost) parallel.
            if dot(lines[lineNo].direction, lines[i].direction) > 0.0:
                # Lines lineNo and i point in the same direction.
                continue
            else:
                # Lines lineNo and i point in opposite direction.
                return False
        
        t = cross(lines[i].direction, lines[lineNo].point - lines[i].point) / denominator
        
        if denominator >= 0.0:
            # Line i bounds line lineNo on the right.
            tRight = min(tRight, t)
        else:
            # Line i bounds line lineNo on the left.
            tLeft = max(tLeft, t)
        
        if tLeft > tRight:
            return False
    
    if directionOpt:
        # Optimize direction.
        if dot(optVelocity, lines[lineNo].direction) > 0.0:
            # Take right extreme.
            directionOpt.x = lines[lineNo].point.x + tRight * lines[lineNo].direction.x
            directionOpt.y = lines[lineNo].point.y + tRight * lines[lineNo].direction.y
        else:
            # Take left extreme.
            directionOpt.x = lines[lineNo].point.x + tLeft * lines[lineNo].direction.x
            directionOpt.y = lines[lineNo].point.y + tLeft * lines[lineNo].direction.y
    
    return True

def linear_program2(lines, radius, optVelocity, directionOpt, result):
    """
    Solves a two-dimensional linear program subject to linear constraints defined by lines and a circular constraint.
    
    Args:
        lines: Lines constraining the solution.
        radius: Radius of the circular constraint.
        optVelocity: The optimization velocity.
        directionOpt: Direction optimization cone.
        result: Optimal velocity.
    
    Returns:
        True if successful, False otherwise.
    """
    if directionOpt:
        # Optimize direction.
        # Note that the optimization velocity is of unit length.
        result.x = optVelocity.x * radius
        result.y = optVelocity.y * radius
        return True
    
    if optVelocity.length_squared() <= radius * radius:
        # Optimization velocity is within radius, use that.
        result.x = optVelocity.x
        result.y = optVelocity.y
        return True
    
    # Project optimization velocity on direction optimization cone.
    for i in range(len(lines)):
        if cross(lines[i].direction, optVelocity - lines[i].point) > 0.0:
            # Optimization velocity is to the right of constraint line.
            tempResult = Vector2()
            if not linear_program1(lines, i, radius, optVelocity, tempResult):
                return False
            
            # From the definition of the linear program, we know that the optimal solution must lie on
            # the constraint line i.
            for j in range(len(lines)):
                if j != i and cross(lines[j].direction, tempResult - lines[j].point) > 0.0:
                    # Temporary result is to the right of constraint line j.
                    return False
            
            result.x = tempResult.x
            result.y = tempResult.y
            return True
    
    # Optimization velocity is within radius, use that.
    result.x = optVelocity.x
    result.y = optVelocity.y
    return True

class Line:
    """Represents a directed line for ORCA constraints"""
    def __init__(self, point=Vector2(), direction=Vector2()):
        self.point = point
        self.direction = direction

class ORCAPlanner:
    """ORCA planner for a single drone"""
    def __init__(self, drone_id, drone_radius=ORCA_RADIUS, max_speed=ORCA_MAX_SPEED, 
                 neighbor_dist=ORCA_NEIGHBOR_DISTANCE, time_horizon=ORCA_TIME_HORIZON,
                 time_horizon_obstacle=ORCA_TIME_HORIZON_OBSTACLE, formation_weight=FORMATION_WEIGHT):
        self.drone_id = drone_id
        self.position = Vector2()
        self.velocity = Vector2()
        self.preferred_velocity = Vector2()
        self.goal_position = Vector2()
        self.formation_position = Vector2()  # Desired position in formation
        self.radius = drone_radius
        self.max_speed = max_speed
        self.neighbor_dist = neighbor_dist
        self.time_horizon = time_horizon
        self.time_horizon_obstacle = time_horizon_obstacle
        self.formation_weight = formation_weight
        self.velocity_history = []
        self.obstacle_distance = ORCA_OBSTACLE_DISTANCE
        self.neighbors = []
        self.obstacles = []
        
    def set_position(self, x, y, z=None):
        """Set the current 2D or 3D position"""
        self.position = Vector2(x, y)
        self.z = z if z is not None else 0.0
    
    def set_velocity(self, vx, vy, vz=None):
        """Set the current 2D or 3D velocity"""
        self.velocity = Vector2(vx, vy)
        self.vz = vz if vz is not None else 0.0
    
    def set_goal(self, x, y, z=None):
        """Set the goal position"""
        self.goal_position = Vector2(x, y)
        self.goal_z = z if z is not None else 0.0
    
    def set_formation_position(self, x, y, z=None):
        """Set the desired formation position"""
        self.formation_position = Vector2(x, y)
        self.formation_z = z if z is not None else 0.0
    
    def update_neighbors(self, drone_positions, drone_velocities=None):
        """
        Update neighbor information from drone_positions and drone_velocities dicts
        
        Args:
            drone_positions: Dictionary mapping drone_id to position [x, y, z]
            drone_velocities: Dictionary mapping drone_id to velocity [vx, vy, vz]
        """
        self.neighbors = []
        
        # Skip this drone itself
        for drone_id, position in drone_positions.items():
            if drone_id == self.drone_id:
                continue
            
            # Calculate distance to neighbor
            dx = position[0] - self.position.x
            dy = position[1] - self.position.y
            dist_squared = dx * dx + dy * dy
            
            # Only consider neighbors within neighbor_dist
            if dist_squared < self.neighbor_dist * self.neighbor_dist:
                # Get velocity if available, otherwise assume zero
                velocity = [0.0, 0.0, 0.0]
                if drone_velocities and drone_id in drone_velocities:
                    velocity = drone_velocities[drone_id]
                
                neighbor = {
                    'id': drone_id,
                    'position': Vector2(position[0], position[1]),
                    'velocity': Vector2(velocity[0], velocity[1]),
                    'z': position[2] if len(position) > 2 else 0.0,
                    'vz': velocity[2] if len(velocity) > 2 else 0.0
                }
                self.neighbors.append(neighbor)
    
    def update_obstacles(self, obstacle_dict):
        """Update obstacle information from obstacle tracker"""
        self.obstacles = []
        
        for obs_id, obs_data in obstacle_dict.items():
            pos = obs_data["position"]
            vel = obs_data.get("velocity", [0, 0, 0])
            diameter = obs_data.get("diameter", 0.5)
            height = obs_data.get("height", 5.0)
            
            # Calculate distance to obstacle
            dx = pos[0] - self.position.x
            dy = pos[1] - self.position.y
            dist_squared = dx * dx + dy * dy
            
            # Only consider obstacles within obstacle_distance
            if dist_squared < self.obstacle_distance * self.obstacle_distance:
                obstacle = {
                    'id': obs_id,
                    'position': Vector2(pos[0], pos[1]),
                    'velocity': Vector2(vel[0], vel[1]),
                    'radius': diameter / 2.0 + SAFETY_MARGIN,
                    'height': height,
                    'z': pos[2] if len(pos) > 2 else 0.0
                }
                self.obstacles.append(obstacle)
    
    def compute_preferred_velocity(self):
        """Compute the preferred velocity based on goal position"""
        goal_dir = Vector2(
            self.goal_position.x - self.position.x,
            self.goal_position.y - self.position.y
        )
        
        dist_to_goal = goal_dir.length()
        
        # If very close to goal, prefer zero velocity
        if dist_to_goal < 0.1:
            self.preferred_velocity = Vector2(0, 0)
            return
        
        # Normalize and scale by max_speed
        if dist_to_goal > 0.0001:
            goal_dir = goal_dir / dist_to_goal
        else:
            goal_dir = Vector2(0, 0)
        
        # Apply formation constraint with a small weight
        formation_dir = Vector2(
            self.formation_position.x - self.position.x,
            self.formation_position.y - self.position.y
        )
        
        if formation_dir.length() > 0.0001:
            formation_dir = formation_dir.normalize()
        else:
            formation_dir = Vector2(0, 0)
        
        # Blend goal direction with formation constraint
        preferred_dir = Vector2(
            (1.0 - self.formation_weight) * goal_dir.x + self.formation_weight * formation_dir.x,
            (1.0 - self.formation_weight) * goal_dir.y + self.formation_weight * formation_dir.y
        )
        
        if preferred_dir.length() > 0.0001:
            preferred_dir = preferred_dir.normalize()
        
        # Scale by max_speed for preferred velocity
        self.preferred_velocity = preferred_dir * self.max_speed
    
    def compute_new_velocity(self):
        """
        Compute the new velocity using ORCA constraints from neighbors and obstacles.
        Implements the full ORCA algorithm with proper velocity reciprocity.
        Returns a tuple (vx, vy, vz) of the new velocity.
        """
        self.compute_preferred_velocity()
        
        lines = []
        
        # Create ORCA lines for neighbors
        for neighbor in self.neighbors:
            relative_position = Vector2(
                self.position.x - neighbor['position'].x,
                self.position.y - neighbor['position'].y
            )
            
            # Use the neighbor's velocity for proper reciprocity
            relative_velocity = Vector2(
                self.velocity.x - neighbor['velocity'].x,
                self.velocity.y - neighbor['velocity'].y
            )
            
            dist_squared = relative_position.length_squared()
            combined_radius = self.radius + self.radius  # Assume all drones have the same radius
            
            # Use ORCA to compute velocity constraints
            if dist_squared > combined_radius * combined_radius:
                # No collision, compute ORCA line
                w = relative_velocity - relative_position * (1.0 / self.time_horizon)
                w_length_squared = w.length_squared()
                
                dot_product = dot(w, relative_position)
                
                if dot_product < 0.0 and dot_product * dot_product > combined_radius * combined_radius * w_length_squared:
                    # Project on cut-off circle
                    w_length = math.sqrt(w_length_squared)
                    unit_w = w / w_length if w_length > 0.0001 else Vector2(0, 0)
                    
                    line = Line()
                    line.direction = Vector2(unit_w.y, -unit_w.x)
                    
                    # Create ORCA line with reciprocal responsibility (half-half)
                    u = unit_w * (combined_radius / self.time_horizon - w_length)
                    line.point = Vector2(u.x * 0.5, u.y * 0.5)  # Half responsibility
                    lines.append(line)
                    continue
            
            # Project on cone boundary (handles both cases: collision and non-collision)
            # For collision: Calculate escape velocity
            # For non-collision: Calculate proper VO cone legs
            
            invTimeHorizonSquared = 1.0 / (self.time_horizon * self.time_horizon)
            
            if dist_squared < combined_radius * combined_radius:
                # Collision case - use escape velocity
                
                # Vector from cutoff center to relative velocity
                u = relative_velocity - relative_position * invTimeHorizonSquared
                
                w = u * 0.5  # Half responsibility
                
                line = Line()
                line.direction = normalize(Vector2(relative_position.y, -relative_position.x))
                line.point = w
                lines.append(line)
                
                # Add second constraint in the opposite direction
                line2 = Line()
                line2.direction = Vector2(-line.direction.x, -line.direction.y)
                line2.point = w
                lines.append(line2)
            else:
                # No collision - compute proper VO cone legs
                leg = math.sqrt(dist_squared - combined_radius * combined_radius)
                
                # Left leg of the VO cone
                left_leg = Line()
                left_leg.direction = Vector2(
                    relative_position.x * leg - relative_position.y * combined_radius,
                    relative_position.x * combined_radius + relative_position.y * leg
                ).normalize()
                
                # Right leg of the VO cone
                right_leg = Line()
                right_leg.direction = Vector2(
                    relative_position.x * leg + relative_position.y * combined_radius,
                    -relative_position.x * combined_radius + relative_position.y * leg
                ).normalize() * -1.0
                
                # Adjust for reciprocal responsibility (half-half)
                left_leg_point = relative_velocity * 0.5
                right_leg_point = relative_velocity * 0.5
                
                left_leg.point = left_leg_point
                right_leg.point = right_leg_point
                
                lines.append(left_leg)
                lines.append(right_leg)
        
        # Create ORCA lines for obstacles with trajectory prediction
        obstacle_constraints_added = 0
        
        if self.obstacles:
            # Use more trajectory prediction points for obstacles for smoother avoidance
            for obstacle in self.obstacles:
                # Use 5 key points for enhanced trajectory prediction
                trajectory_points = [
                    Vector2(obstacle['position'].x, obstacle['position'].y),  # Current
                    Vector2(  # 25% future
                        obstacle['position'].x + obstacle['velocity'].x * (self.time_horizon_obstacle * 0.25),
                        obstacle['position'].y + obstacle['velocity'].y * (self.time_horizon_obstacle * 0.25)
                    ),
                    Vector2(  # 50% future
                        obstacle['position'].x + obstacle['velocity'].x * (self.time_horizon_obstacle * 0.5),
                        obstacle['position'].y + obstacle['velocity'].y * (self.time_horizon_obstacle * 0.5)
                    ),
                    Vector2(  # 75% future
                        obstacle['position'].x + obstacle['velocity'].x * (self.time_horizon_obstacle * 0.75),
                        obstacle['position'].y + obstacle['velocity'].y * (self.time_horizon_obstacle * 0.75)
                    ),
                    Vector2(  # 100% future
                        obstacle['position'].x + obstacle['velocity'].x * self.time_horizon_obstacle,
                        obstacle['position'].y + obstacle['velocity'].y * self.time_horizon_obstacle
                    )
                ]
                
                # Process each trajectory point
                for pred_pos in trajectory_points:
                    relative_position = Vector2(
                        self.position.x - pred_pos.x,
                        self.position.y - pred_pos.y
                    )
                    
                    # Use the obstacle's velocity
                    relative_velocity = Vector2(
                        self.velocity.x - obstacle['velocity'].x,
                        self.velocity.y - obstacle['velocity'].y
                    )
                    
                    dist_squared = relative_position.length_squared()
                    combined_radius = self.radius + obstacle['radius']
                    
                    # Skip if too far away - optimization
                    if dist_squared > (self.obstacle_distance * self.obstacle_distance):
                        continue
                    
                    # Only add constraints if not in collision
                    if dist_squared > combined_radius * combined_radius:
                        # Non-collision case, add VO constraints
                        
                        # For obstacles, we take full responsibility (no reciprocity)
                        w = relative_velocity - relative_position * (1.0 / self.time_horizon_obstacle)
                        w_length_squared = w.length_squared()
                        
                        dot_product = dot(w, relative_position)
                        
                        if dot_product < 0.0 and dot_product * dot_product > combined_radius * combined_radius * w_length_squared:
                            # Project on cut-off circle
                            w_length = math.sqrt(w_length_squared)
                            unit_w = w / w_length if w_length > 0.0001 else Vector2(0, 0)
                            
                            line = Line()
                            line.direction = Vector2(unit_w.y, -unit_w.x)
                            
                            # No reciprocity for obstacles (we take full responsibility)
                            u = unit_w * (combined_radius / self.time_horizon_obstacle - w_length)
                            line.point = u
                            lines.append(line)
                            obstacle_constraints_added += 1
                            continue
                        
                        # Project on cone boundary
                        leg = math.sqrt(dist_squared - combined_radius * combined_radius)
                        
                        if cross(relative_position, relative_velocity) > 0.0:
                            # Velocity on right side of obstacle
                            line = Line()
                            line.direction = Vector2(
                                relative_position.x * leg - relative_position.y * combined_radius,
                                relative_position.x * combined_radius + relative_position.y * leg
                            ).normalize()
                            
                            # No reciprocity for obstacles
                            line.point = relative_velocity
                            lines.append(line)
                            obstacle_constraints_added += 1
                        else:
                            # Velocity on left side of obstacle
                            line = Line()
                            line.direction = Vector2(
                                relative_position.x * leg + relative_position.y * combined_radius,
                                -relative_position.x * combined_radius + relative_position.y * leg
                            ).normalize() * -1.0
                            
                            # No reciprocity for obstacles
                            line.point = relative_velocity
                            lines.append(line)
                            obstacle_constraints_added += 1
        
        # Linear programming to find new velocity
        new_velocity = Vector2()
        
        if lines:
            # Call the improved linear program solver
            success = self.robust_linear_program(lines, self.max_speed, self.preferred_velocity, new_velocity)
            
            if not success:
                # If LP solver fails, fall back to current velocity or zero
                if self.velocity.length() > 0.0001:
                    new_velocity = self.velocity
                else:
                    new_velocity = Vector2(0, 0)
        else:
            # No constraints, use preferred velocity
            if self.preferred_velocity.length() > self.max_speed:
                new_velocity = self.preferred_velocity.normalize() * self.max_speed
            else:
                new_velocity = self.preferred_velocity
        
        # Calculate vertical velocity separately (simple approach)
        vz = 0.0
        if hasattr(self, 'goal_z') and self.goal_z is not None:
            dz = self.goal_z - self.z
            # Simple proportional control for vertical velocity
            vz = max(min(dz, self.max_speed), -self.max_speed)
        
        # Store velocity for future reference
        self.velocity = new_velocity
        self.vz = vz
        
        return (new_velocity.x, new_velocity.y, vz)
        
    def robust_linear_program(self, lines, radius, optVelocity, result):
        """
        Robust implementation of the linear program for ORCA.
        Ensures all constraints are satisfied through iterative calls to linear_program1.
        
        Args:
            lines: Lines constraining the solution
            radius: Maximum speed
            optVelocity: Preferred velocity
            result: Output parameter for the computed velocity
            
        Returns:
            True if successful, False otherwise
        """
        # IMPORTANT FIX #1: Previously, the code had an early return that would skip 
        # all constraint processing when lines existed, causing drones to ignore obstacles.
        # The original flow has been fixed to ensure constraints are actually applied.
        
        # IMPORTANT FIX #2: The half-plane test was reversed - the cross product
        # inequality needed to be flipped to correctly identify forbidden velocities.
        
        # Debug prints have been added to track solver behavior:
        # - Number of constraint lines
        # - Which path the solver takes (direct, clamped, LP2, fallbacks)
        # - Which constraints are being violated
        # - Results from each linear programming step
        
        # Debug print - number of constraint lines
        print(f"[LP-{self.drone_id}] lines={len(lines)}, pref={optVelocity.to_tuple()}")
        
        # 1) If no constraints, just use preferred velocity (if it's valid).
        if not lines:
            if optVelocity.length_squared() <= radius * radius:
                result.x = optVelocity.x
                result.y = optVelocity.y
                print(f"[LP-{self.drone_id}] No constraints, using preferred velocity")
                return True
            # else fall through to clamp below
            print(f"[LP-{self.drone_id}] No constraints, preferred velocity too fast")

        # 2) Clamp the preferred velocity to max speed
        norm = optVelocity.normalize()
        clamped = Vector2(norm.x * radius, norm.y * radius)
        print(f"[LP-{self.drone_id}] Clamped to {clamped.to_tuple()}")

        # 3) If that clamped velocity doesn't violate any line constraints, use it
        # CRITICAL FIX: The half-plane test was reversed. In ORCA, each Line defines a half-plane 
        # of forbidden velocities. The sign of the cross-product test determines which side of the 
        # line is considered "safe". The inequality was flipped from < 0.0 to > 0.0 to correctly 
        # identify the forbidden half-plane.
        violates = False
        for i, line in enumerate(lines):
            cross_product = cross(line.direction, clamped - line.point)
            if cross_product > 0.0:  # FLIPPED INEQUALITY - originally was < 0.0
                violates = True
                print(f"[LP-{self.drone_id}] Line {i} violated: cross={cross_product}")
                break
                
        print(f"[LP-{self.drone_id}] violates={violates}, using_clamped={not violates}")
        
        if not violates:
            result.x, result.y = clamped.x, clamped.y
            return True

        # 4) Otherwise fall back to the LP solvers
        # Previously, this code was never reached due to the early return bug
        print(f"[LP-{self.drone_id}] Falling back to LP2")
        lp2_result = linear_program2(lines, radius, optVelocity, False, result)
        print(f"[LP-{self.drone_id}] LP2 result: {lp2_result}, velocity={result.x:.2f},{result.y:.2f}")
        
        if lp2_result:
            return True

        # If linear_program2 fails, implement fallback strategy
        print(f"[LP-{self.drone_id}] LP2 failed, trying fallback")
        
        # Project onto the closest constraint line
        min_distance = float('inf')
        closest_line_idx = -1
        
        for i in range(len(lines)):
            # Calculate distance from optVelocity to line
            dist = abs(cross(lines[i].direction, optVelocity - lines[i].point))
            if dist < min_distance:
                min_distance = dist
                closest_line_idx = i
        
        if closest_line_idx >= 0:
            # Try linear_program1 with the closest line
            temp_result = Vector2()
            lp1_result = linear_program1(lines, closest_line_idx, radius, optVelocity, temp_result)
            print(f"[LP-{self.drone_id}] LP1 result: {lp1_result}")
            
            if lp1_result:
                result.x = temp_result.x
                result.y = temp_result.y
                return True
        
        # Last resort: use current velocity or zero if all else fails
        print(f"[LP-{self.drone_id}] All LP methods failed, using fallback velocity")
        if self.velocity.length_squared() <= radius * radius:
            result.x = self.velocity.x
            result.y = self.velocity.y
        else:
            result.x = 0.0
            result.y = 0.0
        
        return False

# --------------------------
# Dynamic Obstacle Tracking
# --------------------------
class DynamicObstacleTracker(Node):
    """Track dynamic obstacles from the /dynamic_obstacles/locations topic"""
    
    def __init__(self):
        super().__init__('dynamic_obstacle_tracker')
        
        # Create QoS profile for obstacle subscription
        obstacle_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribe to the posestamped topic
        self.obstacle_subscriber = self.create_subscription(
            PoseStamped,
            '/dynamic_obstacles/locations',
            self.obstacle_callback,
            obstacle_qos
        )
        
        # Dictionary to store obstacle data
        self.obstacles = {}
        self.last_update_time = time.time()
        
        # Obstacle parameters
        self.obstacle_height = 5.0  # Default height in meters
        self.obstacle_diameter = 0.5  # Default diameter in meters
        
        # Lock for thread-safe access to obstacle data
        self.obstacle_lock = threading.Lock()
        
        self.get_logger().info("Dynamic obstacle tracker initialized")
    
    def set_obstacle_parameters(self, height, diameter):
        """Set obstacle parameters"""
        self.obstacle_height = height
        self.obstacle_diameter = diameter
        self.get_logger().info(f"Set obstacle parameters: height={height}m, diameter={diameter}m")
    
    def obstacle_callback(self, msg):
        """Process incoming obstacle pose messages"""
        with self.obstacle_lock:
            try:
                # Extract obstacle ID from the frame_id (it contains the ID as the last part)
                # Format is typically /world/dynamic_obstacles/<id>
                # Example: /world/dynamic_obstacles/sphere0
                frame_parts = msg.header.frame_id.split('/')
                obstacle_id = frame_parts[-1]
                
                # Store current time for velocity calculation
                current_time = time.time()
                
                # Create or update obstacle data with the latest position
                position = [
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z
                ]
                
                if obstacle_id in self.obstacles:
                    # Calculate velocity if we have a previous position
                    prev_position = self.obstacles[obstacle_id]["position"]
                    time_diff = current_time - self.obstacles[obstacle_id]["timestamp"]
                    
                    if time_diff > 0:
                        velocity = [
                            (position[0] - prev_position[0]) / time_diff,
                            (position[1] - prev_position[1]) / time_diff,
                            (position[2] - prev_position[2]) / time_diff
                        ]
                        
                        # Update with new position and computed velocity
                        self.obstacles[obstacle_id] = {
                            "position": position,
                            "velocity": velocity,
                            "timestamp": current_time,
                            "height": self.obstacle_height,
                            "diameter": self.obstacle_diameter
                        }
                else:
                    # First sighting, can't compute velocity yet
                    self.obstacles[obstacle_id] = {
                        "position": position,
                        "velocity": [0.0, 0.0, 0.0],  # Zero initial velocity
                        "timestamp": current_time,
                        "height": self.obstacle_height,
                        "diameter": self.obstacle_diameter
                    }
                
                # Update the last update time
                self.last_update_time = current_time
                
            except Exception as e:
                self.get_logger().error(f"Error processing obstacle message: {e}")
    
    def get_obstacles(self):
        """Get a copy of the current obstacle data"""
        with self.obstacle_lock:
            return self.obstacles.copy()
    
    def get_obstacle_predictions(self, prediction_time=None, horizon=5.0, timestep=0.5):
        """
        Get predicted obstacle positions for a future time period.
        
        Args:
            prediction_time: Single future time to predict for (seconds), or None for trajectory
            horizon: Time horizon for trajectory prediction (seconds), used if prediction_time is None
            timestep: Time step for trajectory prediction (seconds)
            
        Returns:
            Dictionary of obstacle predictions
        """
        with self.obstacle_lock:
            if prediction_time is not None:
                # Single time prediction
                predictions = {}
                for obs_id, obs_data in self.obstacles.items():
                    predictions[obs_id] = {
                        "position": predict_obstacle_position(obs_data, prediction_time),
                        "velocity": obs_data["velocity"],
                        "height": obs_data["height"],
                        "diameter": obs_data["diameter"]
                    }
                return predictions
            else:
                # Trajectory prediction
                predictions = {}
                for obs_id, obs_data in self.obstacles.items():
                    trajectory = predict_obstacle_trajectory(obs_data, horizon, int(horizon/timestep))
                    predictions[obs_id] = {
                        "trajectory": trajectory,
                        "velocity": obs_data["velocity"],
                        "height": obs_data["height"],
                        "diameter": obs_data["diameter"],
                        "timestep": timestep
                    }
                return predictions

# --------------------------
# Multi-Drone Mission
# --------------------------
class MultiDroneMission:
    def __init__(self, drone_ids, scenario_file, use_sim_time=True, verbose=False):
        """Initialize the mission with multiple drones"""
        self.drone_ids = drone_ids
        self.drones = {}
        self.drone_positions = {}  # Store latest positions
        self.drone_velocities = {}  # Store latest velocities (key addition for ORCA)
        self.drone_last_positions = {}  # Store previous positions for velocity calculation
        self.drone_last_update_times = {}  # Store last update times for velocity calculation
        self.orca_planners = {}    # ORCA planners for each drone
        
        # Load scenario
        self.scenario = load_scenario(scenario_file)
        self.setup_from_scenario(self.scenario)
        
        # Initialize ROS
        rclpy.init()
        
        # Create a simple node for position subscriptions
        self.position_node = rclpy.create_node('position_subscriber_node')
        
        # Create QoS profile that matches the publisher
        position_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Initialize dynamic obstacle tracker
        self.obstacle_tracker = DynamicObstacleTracker()
        
        # Get starting positions in a zigzag pattern
        start_positions = self.get_distributed_positions(self.start_position, len(drone_ids))
        print("Created zigzag formation for starting positions")
        
        # Get end positions in a zigzag pattern for formation
        end_positions = self.get_distributed_positions(self.end_position, len(drone_ids))
        print("Created zigzag formation for end positions")
        
        # Initialize drones, position subscribers, and ORCA planners
        for idx, drone_id in enumerate(drone_ids):
            self.drones[drone_id] = DroneInterface(drone_id=drone_id, use_sim_time=use_sim_time, verbose=verbose)
            
            # Initialize position with the calculated start position
            self.drone_positions[drone_id] = start_positions[idx].copy()
            self.drone_velocities[drone_id] = [0.0, 0.0, 0.0]  # Initialize with zero velocity
            self.drone_last_positions[drone_id] = start_positions[idx].copy()
            self.drone_last_update_times[drone_id] = time.time()
            
            # Create ORCA planner for this drone
            self.orca_planners[drone_id] = ORCAPlanner(drone_id)
            
            # Initialize planner with start position and goal
            self.orca_planners[drone_id].set_position(
                start_positions[idx][0], 
                start_positions[idx][1], 
                start_positions[idx][2]
            )
            
            # Set precise zigzag goal position (not just the center end position)
            self.orca_planners[drone_id].set_goal(
                end_positions[idx][0], 
                end_positions[idx][1], 
                end_positions[idx][2]
            )
            
            # Set formation position (same as the goal position)
            self.orca_planners[drone_id].set_formation_position(
                end_positions[idx][0], 
                end_positions[idx][1], 
                end_positions[idx][2]
            )
            
            # Subscribe to position topic with correct QoS
            topic = f'/{drone_id}/self_localization/pose'
            self.position_node.create_subscription(
                PoseStamped,
                topic,
                lambda msg, d_id=drone_id: self.position_callback(msg, d_id),
                position_qos
            )
            print(f"Subscribed to position topic: {topic} with BEST_EFFORT reliability")
        
        # Create separate executors for each node to avoid threading issues
        self.position_executor = rclpy.executors.SingleThreadedExecutor()
        self.position_executor.add_node(self.position_node)
        
        self.obstacle_executor = rclpy.executors.SingleThreadedExecutor()
        self.obstacle_executor.add_node(self.obstacle_tracker)
        
        # Start position node in a separate thread
        self.position_thread = threading.Thread(target=self.spin_position_node, daemon=True)
        self.position_thread.start()
        
        # Start obstacle tracker in a separate thread
        self.obstacle_thread = threading.Thread(target=self.spin_obstacle_node, daemon=True)
        self.obstacle_thread.start()
        
        # Create timer threads for each drone's ORCA control loop
        self.orca_threads = {}
        self.orca_stop_flags = {}
        
        for drone_id in drone_ids:
            self.orca_stop_flags[drone_id] = threading.Event()
            self.orca_threads[drone_id] = threading.Thread(
                target=self.orca_control_loop,
                args=(drone_id, self.orca_stop_flags[drone_id]),
                daemon=True
            )
        
        # Check if we need to start the scenario runner
        self.check_and_start_scenario_runner(scenario_file)
        
        # Allow more time for obstacle data to be received
        print("Waiting for obstacle data to be published (this may take 10 seconds)...")
        obstacle_wait_time = 10.0  # Increase wait time to 10 seconds
        start_wait = time.time()
        while time.time() - start_wait < obstacle_wait_time:
            obstacles = self.obstacle_tracker.get_obstacles()
            if obstacles:
                print(f"Detected {len(obstacles)} obstacles after {time.time() - start_wait:.1f} seconds!")
                break
            time.sleep(0.5)
            print(".", end="", flush=True)
        print()  # Newline after progress dots
        
        # Set obstacle parameters from scenario
        self.obstacle_tracker.set_obstacle_parameters(
            self.obstacle_height,
            self.obstacle_diameter
        )
        
        print(f"Multi-drone mission initialized with {len(drone_ids)} drones in a zigzag formation")
    
    def orca_control_loop(self, drone_id, stop_flag):
        """ORCA control loop for a single drone - optimized for performance"""
        planner = self.orca_planners[drone_id]
        drone = self.drones[drone_id]
        
        # Set control rate to match REPLAN_FREQUENCY
        control_rate = 1.0 / REPLAN_FREQUENCY
        # Create rate limiter
        rate_limiter = time.time()
        
        # Print debug info 
        print(f"Starting ORCA control loop for {drone_id}")
        print(f"{drone_id}: Goal position: ({planner.goal_position.x:.2f}, {planner.goal_position.y:.2f}, {planner.goal_z:.2f})")
        
        # Debug variables
        debug_timer = 0
        debug_interval = 5.0  # only print debug every 5 seconds
        obstacle_debug_interval = 3.0  # Print obstacle info every 3 seconds
        obstacle_debug_timer = 0
        
        # Check if go_to method is available
        if not hasattr(drone, 'go_to'):
            print(f"{drone_id}: go_to method not available, cannot proceed with ORCA")
            return
        
        while not stop_flag.is_set():
            try:
                # Track loop start time
                loop_start = time.time()
                
                # Get current position (cache to minimize duplicate calls)
                current_position = self.get_drone_position(drone)
                
                # Only update all planners at the control rate frequency
                elapsed = time.time() - rate_limiter
                if elapsed >= control_rate:
                    # Reset rate limiter
                    rate_limiter = time.time()
                    
                    # Update position in planner
                    planner.set_position(current_position[0], current_position[1], current_position[2])
                    
                    # Update velocity if available
                    if drone_id in self.drone_velocities:
                        current_velocity = self.drone_velocities[drone_id]
                        planner.set_velocity(current_velocity[0], current_velocity[1], current_velocity[2])
                    
                    # Get obstacles once per cycle for efficiency
                    obstacles = self.obstacle_tracker.get_obstacles()
                    planner.update_obstacles(obstacles)
                    
                    # Periodically print obstacle debug info
                    obstacle_debug_timer += elapsed
                    if obstacle_debug_timer >= obstacle_debug_interval:
                        obstacle_debug_timer = 0
                        
                        if obstacles:
                            num_obstacles = len(obstacles)
                            closest_obstacle_dist = float('inf')
                            closest_id = None
                            
                            for obs_id, obs_data in obstacles.items():
                                dx = obs_data["position"][0] - current_position[0]
                                dy = obs_data["position"][1] - current_position[1]
                                dist = math.sqrt(dx*dx + dy*dy)
                                
                                if dist < closest_obstacle_dist:
                                    closest_obstacle_dist = dist
                                    closest_id = obs_id
                            
                            if closest_id:
                                obs = obstacles[closest_id]
                                print(f"{drone_id}: {num_obstacles} obstacles detected, closest: {closest_id} at {closest_obstacle_dist:.2f}m " +
                                      f"pos=({obs['position'][0]:.2f}, {obs['position'][1]:.2f}), vel=({obs['velocity'][0]:.2f}, {obs['velocity'][1]:.2f})")
                        else:
                            print(f"{drone_id}: No obstacles detected")
                    
                    # Update neighbor information with velocities for proper reciprocity
                    planner.update_neighbors(self.drone_positions, self.drone_velocities)
                    
                    # Compute new velocity using ORCA with full reciprocity
                    vx, vy, vz = planner.compute_new_velocity()
                    speed = math.sqrt(vx*vx + vy*vy + vz*vz)
                    
                    # Print debug info occasionally
                    debug_timer += elapsed
                    if debug_timer >= debug_interval:
                        debug_timer = 0
                        
                        # Calculate distance to goal
                        dx = planner.goal_position.x - current_position[0]
                        dy = planner.goal_position.y - current_position[1]
                        dz = planner.goal_z - current_position[2]
                        distance_to_goal = math.sqrt(dx*dx + dy*dy + dz*dz)
                        
                        print(f"{drone_id}: Pos=({current_position[0]:.2f}, {current_position[1]:.2f}, {current_position[2]:.2f}), " +
                              f"Goal=({planner.goal_position.x:.2f}, {planner.goal_position.y:.2f}, {planner.goal_z:.2f}), " +
                              f"Dist={distance_to_goal:.2f}, " +
                              f"Vel=({vx:.2f}, {vy:.2f}, {vz:.2f}), Speed={speed:.2f}")
                    
                    # Send velocity command using available methods
                    try:
                        # Create a point slightly ahead in the velocity direction
                        # This simulates velocity control using position setpoints
                        next_point = [
                            current_position[0] + vx * control_rate,
                            current_position[1] + vy * control_rate,
                            current_position[2] + vz * control_rate
                        ]
                        
                        # Use non-blocking go_to for velocity-like control
                        drone.go_to(
                            next_point[0], next_point[1], next_point[2],
                            speed=speed if speed > 0.1 else 0.1,  # Ensure minimum speed
                            wait=False  # Non-blocking call
                        )
                    except Exception as e:
                        print(f"{drone_id}: Error sending velocity command: {e}")
                
                # Check if we've reached the goal - do this on every loop iteration
                dx = planner.goal_position.x - current_position[0]
                dy = planner.goal_position.y - current_position[1]
                dz = planner.goal_z - current_position[2]
                distance_to_goal = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                if distance_to_goal < 0.35:  # Reduced from 0.5 to 0.35 for more precise positioning
                    # We've reached the goal, stop moving
                    print(f"{drone_id}: Goal reached! Distance: {distance_to_goal:.2f}m")
                    # Try to send a go_to command to the exact goal position to stop
                    try:
                        drone.go_to(
                            planner.goal_position.x,
                            planner.goal_position.y,
                            planner.goal_z,
                            speed=0.3,  # Reduced from 0.5 to 0.3 for finer control in final approach
                            wait=True   # Wait for this final command
                        )
                    except Exception as e:
                        print(f"{drone_id}: Error sending final position command: {e}")
                    stop_flag.set()
                    break
                
                # Calculate remaining time in this cycle
                loop_duration = time.time() - loop_start
                sleep_time = max(0.01, control_rate - loop_duration)  # At least 10ms sleep
                
                # Sleep for the remainder of the control cycle
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"{drone_id}: Error in ORCA control loop: {e}")
                time.sleep(0.1)  # Brief sleep to avoid tight error loops
    
    def check_and_start_scenario_runner(self, scenario_file):
        """Check if the scenario runner is active and start it if needed"""
        # Create a temporary node to check for active publishers on the obstacle topic
        check_node = rclpy.create_node('topic_check_node')
        
        # Wait a bit for discovery
        time.sleep(2.0)
        
        # Get publishers on the obstacle topic
        topic_name = '/dynamic_obstacles/locations'
        publishers_info = check_node.get_publishers_info_by_topic(topic_name)
        
        if not publishers_info:
            print(f"No publishers found on {topic_name}. Starting scenario runner...")
            
            # Start the scenario runner in a separate process
            try:
                import subprocess
                
                # Make sure the scenario file exists
                if not os.path.exists(scenario_file):
                    print(f"Error: Scenario file {scenario_file} not found!")
                    scenario_file = "scenarios/scenario1_stage4.yaml"
                    print(f"Falling back to default: {scenario_file}")
                
                # Try to create an absolute path to the scenario file
                current_dir = os.getcwd()
                scenario_path = os.path.join(current_dir, scenario_file)
                print(f"Starting scenario runner with file: {scenario_path}")
                
                # Run the command
                cmd = f"ros2 run config_sim scenario_runner --ros-args -p scenario_file:={scenario_path}"
                print(f"Running command: {cmd}")
                process = subprocess.Popen(cmd, shell=True)
                
                # Wait for scenario runner to start
                print("Waiting for scenario runner to start publishing obstacles...")
                wait_time = 0
                max_wait = 15  # seconds
                while wait_time < max_wait:
                    time.sleep(1.0)
                    wait_time += 1
                    print(".", end="", flush=True)
                    # Check if publishers exist now
                    publishers = check_node.get_publishers_info_by_topic(topic_name)
                    if publishers:
                        print(f"\nScenario runner started successfully after {wait_time}s!")
                        break
                
                if wait_time >= max_wait:
                    print("\nWarning: Scenario runner may not have started properly.")
            except Exception as e:
                print(f"Failed to start scenario runner: {e}")
        else:
            print(f"Found {len(publishers_info)} publishers on {topic_name}")
        
        # Clean up the temporary node
        check_node.destroy_node()
    
    def spin_position_node(self):
        """Spin the position subscriber node in a separate thread"""
        try:
            self.position_executor.spin()
        except Exception as e:
            print(f"Error in position node: {e}")
    
    def spin_obstacle_node(self):
        """Spin the obstacle tracker node in a separate thread"""
        try:
            self.obstacle_executor.spin()
        except Exception as e:
            print(f"Error in obstacle tracker node: {e}")
            
    def position_callback(self, msg, drone_id):
        """Callback for position updates"""
        current_time = time.time()
        current_position = [
            msg.pose.position.x,
            msg.pose.position.y, 
            msg.pose.position.z
        ]
        
        # Compute velocity from position changes if we have a previous position
        if drone_id in self.drone_positions:
            prev_position = self.drone_last_positions[drone_id]
            time_diff = current_time - self.drone_last_update_times[drone_id]
            
            if time_diff > 0.001:  # Avoid division by very small numbers
                velocity = [
                    (current_position[0] - prev_position[0]) / time_diff,
                    (current_position[1] - prev_position[1]) / time_diff,
                    (current_position[2] - prev_position[2]) / time_diff
                ]
                
                # Apply some smoothing to avoid noisy velocity estimates (optional)
                alpha = 0.7  # Smoothing factor
                if drone_id in self.drone_velocities:
                    prev_vel = self.drone_velocities[drone_id]
                    velocity = [
                        alpha * velocity[0] + (1-alpha) * prev_vel[0],
                        alpha * velocity[1] + (1-alpha) * prev_vel[1],
                        alpha * velocity[2] + (1-alpha) * prev_vel[2]
                    ]
                
                # Update velocities
                self.drone_velocities[drone_id] = velocity
        
        # Update position and time records
        self.drone_positions[drone_id] = current_position
        self.drone_last_positions[drone_id] = current_position.copy()
        self.drone_last_update_times[drone_id] = current_time
        
        # Update the ORCA planner position and velocity if available
        if drone_id in self.orca_planners:
            self.orca_planners[drone_id].set_position(
                current_position[0], 
                current_position[1], 
                current_position[2]
            )
            
            if drone_id in self.drone_velocities:
                vel = self.drone_velocities[drone_id]
                self.orca_planners[drone_id].set_velocity(vel[0], vel[1], vel[2])
        
        # Uncomment for debugging
        # print(f"Position update for {drone_id}: {current_position}")
    
    def get_drone_position(self, drone):
        """Get the current position of the drone from stored positions"""
        return self.drone_positions[drone.drone_id].copy()

    def setup_from_scenario(self, scenario):
        """Extract important information from the scenario file"""
        # Stage 4 data
        self.stage4 = scenario.get("stage4", {})
        self.stage_center = self.stage4.get("stage_center", [0.0, 6.0])
        self.start_point = self.stage4.get("start_point", [0.0, -6.0])
        self.end_point = self.stage4.get("end_point", [0.0, 6.0])
        
        # Convert relative points to absolute coordinates
        self.start_position = [
            self.stage_center[0] + self.start_point[0],
            self.stage_center[1] + self.start_point[1],
            TAKE_OFF_HEIGHT
        ]
        
        self.end_position = [
            self.stage_center[0] + self.end_point[0],
            self.stage_center[1] + self.end_point[1],
            TAKE_OFF_HEIGHT
        ]
        
        # Stage bounds from scenario
        stage_size = scenario.get("stage_size", [10.0, 10.0])
        
        # Calculate exact bounds according to stage4.yaml with +2.0 safety margin
        stage_min_x = self.stage_center[0] - stage_size[0]/2.0
        stage_max_x = self.stage_center[0] + stage_size[0]/2.0
        stage_min_y = self.stage_center[1] - stage_size[1]/2.0
        stage_max_y = self.stage_center[1] + stage_size[1]/2.0
        
        # Add safety margin to bounds
        bounds_margin = 2.0
        stage_min_x -= bounds_margin
        stage_max_x += bounds_margin
        stage_min_y -= bounds_margin
        stage_max_y += bounds_margin
        
        # Store bounds for planning
        self.bounds = {
            "low": [stage_min_x, stage_min_y, MIN_ALTITUDE],
            "high": [stage_max_x, stage_max_y, MAX_ALTITUDE]
        }
        
        # Get obstacle parameters
        self.obstacle_height = self.stage4.get("obstacle_height", 5.0)
        self.obstacle_diameter = self.stage4.get("obstacle_diameter", 0.5)
    
    def start_mission(self):
        """Start the mission by taking off all drones"""
        print("Starting mission: taking off all drones")
        
        # First, set all drones to OFFBOARD mode and arm them
        print("Setting all drones to OFFBOARD mode and arming...")
        for drone_id, drone in self.drones.items():
            # Set mode to OFFBOARD for flight control
            print(f"Setting {drone_id} to OFFBOARD mode")
            drone.offboard()
            
            # Arm the drone
            print(f"Arming {drone_id}")
            drone.arm()
        
        # Small delay to ensure all drones are armed
        time.sleep(1.0)
            
        # Now send take-off commands to all drones simultaneously
        print("Commanding all drones to takeoff simultaneously...")
        takeoff_threads = []
        for drone_id, drone in self.drones.items():
            print(f"Taking off {drone_id} to {TAKE_OFF_HEIGHT}m at {TAKE_OFF_SPEED}m/s")
            t = threading.Thread(
                target=drone.takeoff,
                args=(TAKE_OFF_HEIGHT, TAKE_OFF_SPEED),
                daemon=True
            )
            t.start()
            takeoff_threads.append(t)
        
        # Wait for all drones to receive takeoff commands
        for t in takeoff_threads:
            t.join()
            
        # Give some time for drones to stabilize at take-off altitude
        print("All take-off commands sent, waiting for drones to reach altitude")
        time.sleep(2.0)
        
        # Check if all drones have reached target altitude
        all_drones_at_altitude = False
        max_wait_time = 30  # maximum wait time in seconds
        start_time = time.time()
        altitude_tolerance = 0.3  # Increased tolerance to 30cm
        
        while not all_drones_at_altitude and (time.time() - start_time) < max_wait_time:
            all_drones_at_altitude = True
            altitude_issues = []
            
            for drone_id, drone in self.drones.items():
                pos = self.get_drone_position(drone)
                altitude = pos[2]
                
                if abs(altitude - TAKE_OFF_HEIGHT) > altitude_tolerance:
                    all_drones_at_altitude = False
                    altitude_issues.append(f"{drone_id}: {altitude:.2f}m")
            
            if not all_drones_at_altitude:
                print(f"Altitude check: {', '.join(altitude_issues)} (target: {TAKE_OFF_HEIGHT}m ±{altitude_tolerance}m)")
                time.sleep(1.0)
        
        if all_drones_at_altitude:
            print("All drones have reached target altitude. Mission ready to proceed.")
        else:
            print("Warning: Not all drones reached target altitude within time limit. Proceeding anyway.")
        
        # ==== EXPLICIT ZIGZAG FORMATION PHASE ====
        print("\n=== FORMING INITIAL ZIGZAG PATTERN ===")
        
        # Get distributed positions in zigzag pattern over the start position
        initial_formation_positions = self.get_distributed_positions(self.start_position, len(self.drones))
        print("Created zigzag formation for initial positions")
        
        # Move all drones to their respective zigzag positions
        formation_threads = []
        for idx, (drone_id, drone) in enumerate(self.drones.items()):
            target_position = initial_formation_positions[idx]
            print(f"Moving {drone_id} to formation position: ({target_position[0]:.2f}, {target_position[1]:.2f}, {target_position[2]:.2f})")
            
            # Create thread to move drone to its zigzag position
            t = threading.Thread(
                target=drone.go_to,
                args=(target_position[0], target_position[1], target_position[2]),
                kwargs={'speed': 1.0, 'wait': True},
                daemon=True
            )
            t.start()
            formation_threads.append(t)
        
        # Wait for all drones to reach their formation positions
        print("Waiting for all drones to reach their formation positions...")
        for t in formation_threads:
            t.join()
        
        # Verify that all drones are in their initial formation positions
        print("Verifying formation positions...")
        all_drones_in_position = False
        max_position_wait = 20  # maximum wait time in seconds
        start_check_time = time.time()
        position_tolerance = 0.4  # 40cm tolerance
        
        while not all_drones_in_position and (time.time() - start_check_time) < max_position_wait:
            all_drones_in_position = True
            position_issues = []
            
            for idx, (drone_id, drone) in enumerate(self.drones.items()):
                current_pos = self.get_drone_position(drone)
                target_pos = initial_formation_positions[idx]
                
                # Calculate distance to target formation position
                distance = compute_euclidean_distance(current_pos, target_pos)
                
                if distance > position_tolerance:
                    all_drones_in_position = False
                    position_issues.append(f"{drone_id}: {distance:.2f}m off")
            
            if not all_drones_in_position:
                print(f"Formation check: {', '.join(position_issues)} (tolerance: {position_tolerance}m)")
                time.sleep(1.0)
        
        if all_drones_in_position:
            print("All drones have successfully formed the initial zigzag pattern!")
        else:
            print("Warning: Not all drones reached their exact formation positions. Proceeding anyway.")
            
        print("=== INITIAL FORMATION COMPLETE ===\n")
        
        # Update ORCA planner positions with current drone positions
        for drone_id, drone in self.drones.items():
            current_position = self.get_drone_position(drone)
            if drone_id in self.orca_planners:
                self.orca_planners[drone_id].set_position(
                    current_position[0], 
                    current_position[1], 
                    current_position[2]
                )
                
                # Set zero initial velocity
                self.orca_planners[drone_id].set_velocity(0, 0, 0)
        
        return True

    def end_mission(self):
        """End the mission by landing all drones"""
        print("Ending mission: stopping ORCA controllers and landing all drones")
        
        # Stop all ORCA control threads first
        for drone_id in self.drones:
            if drone_id in self.orca_stop_flags:
                self.orca_stop_flags[drone_id].set()
                print(f"Stopped ORCA controller for {drone_id}")
        
        # Wait for ORCA threads to terminate
        for drone_id, thread in self.orca_threads.items():
            if thread.is_alive():
                thread.join(timeout=1.0)
                print(f"ORCA thread for {drone_id} terminated")
        
        # Land all drones
        threads = []
        for drone_id, drone in self.drones.items():
            # Land the drone using the land module
            print(f"Landing {drone_id} at {LAND_SPEED}m/s")
            t = threading.Thread(
                target=drone.land,
                args=(LAND_SPEED,),
                daemon=True
            )
            t.start()
            threads.append(t)
        
        # Wait for landing to complete
        for t in threads:
            t.join()
        
        print("All landing commands completed")
        
        # Check that all drones have landed
        all_drones_landed = False
        max_wait_time = 30  # maximum wait time in seconds
        start_time = time.time()
        
        while not all_drones_landed and (time.time() - start_time) < max_wait_time:
            all_drones_landed = True
            
            for drone_id, drone in self.drones.items():
                pos = self.get_drone_position(drone)
                altitude = pos[2]
                
                if altitude > 0.2:  # 20cm threshold for considering landed
                    all_drones_landed = False
                    print(f"{drone_id} altitude: {altitude:.2f}m (waiting for landing)")
                    break
            
            if not all_drones_landed:
                time.sleep(1.0)
        
        if all_drones_landed:
            print("All drones have successfully landed. Mission complete.")
        else:
            print("Warning: Not all drones confirmed landing within time limit.")
        
        # Disarm all drones
        for drone_id, drone in self.drones.items():
            print(f"Disarming {drone_id}")
            drone.disarm()
        
        return True
    
    def get_distributed_positions(self, center_position, num_drones):
        """
        Get distributed positions around a center point.
        For multiple drones: arrange them in a zigzag pattern.
        """
        positions = []
        
        # Spacing between drones - increased y_spacing for better zigzag visibility
        x_spacing = 1.25
        y_spacing = 0.25
        
        # Calculate the leftmost position (to center the formation)
        left_offset = -x_spacing * (num_drones - 1) / 2
        
        for i in range(num_drones):
            # Place drones in a zigzag pattern
            x_offset = left_offset + (i * x_spacing)
            y_offset = y_spacing if i % 2 == 0 else -y_spacing
            
            # Alternate altitude for better separation
            alt_variation = 0.0
            if i > 0:  # First drone at standard height, others with small variations
                alt_variation = 0.1 if i % 2 == 1 else -0.1  # Opposite of Y pattern for better separation
            
            pos = [
                center_position[0] + x_offset,      # X: horizontal spacing
                center_position[1] + y_offset,      # Y: zigzag offset
                center_position[2] + alt_variation  # Z: altitude with small variation
            ]
            
            # Ensure altitude is within the allowed range
            pos[2] = max(MIN_ALTITUDE, min(MAX_ALTITUDE, pos[2]))
            positions.append(pos)
            
            # Add debug message for position
            print(f"DEBUG: Position {i} in zigzag: X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f}")
            
            # Add debug message for altitude variation
            alt_diff = pos[2] - TAKE_OFF_HEIGHT
            if abs(alt_diff) > 0.01:  # If there's a meaningful difference
                print(f"DEBUG: Position {i} has altitude variation: {alt_diff:.2f}m (Z={pos[2]:.2f}m)")
        
        return positions
    
    def run_mission(self):
        """
        Main mission execution with decentralized ORCA control:
        1. Each drone uses its own ORCA planner for collision avoidance
        2. Drones communicate positions implicitly through self.drone_positions
        3. Formation is maintained through soft constraints in ORCA
        """
        print("Starting mission execution with decentralized ORCA control")
        print("Drones will maintain zigzag formation while navigating to the end position")
        
        # Print current obstacle count
        obstacles = self.obstacle_tracker.get_obstacles()
        print(f"Current obstacle count before starting: {len(obstacles)}")
        
        # Get distributed end positions for the drones in a zigzag pattern
        end_positions = self.get_distributed_positions(self.end_position, len(self.drones))
        print("Created zigzag formation for end positions")
        
        # Set both formation positions AND goal positions for each drone to ensure zigzag pattern
        for idx, drone_id in enumerate(self.drones):
            # Set formation position
            self.orca_planners[drone_id].set_formation_position(
                end_positions[idx][0],
                end_positions[idx][1],
                end_positions[idx][2]
            )
            
            # Also set as goal position to ensure pattern is maintained
            self.orca_planners[drone_id].set_goal(
                end_positions[idx][0],
                end_positions[idx][1],
                end_positions[idx][2]
            )
            
            print(f"Set {drone_id} goal & formation position to: {end_positions[idx]}")
        
        # Start the ORCA control threads for each drone
        print("Starting ORCA control threads for all drones - maintaining zigzag formation")
        for drone_id in self.drones:
            self.orca_threads[drone_id].start()
            print(f"Started ORCA control thread for {drone_id}")
        
        # Track completion status
        all_drones_completed = False
        
        # Main monitoring loop
        try:
            while not all_drones_completed:
                # Check if all drones have reached their goals
                completed_drones = [flag.is_set() for flag in self.orca_stop_flags.values()]
                all_drones_completed = all(completed_drones)
                
                # Display progress
                completion_count = sum(completed_drones)
                if completion_count > 0:
                    print(f"Progress: {completion_count}/{len(self.drones)} drones reached their goals")
                
                # Wait before checking again
                time.sleep(1.0)
            
            print("All drones have successfully reached their goals!")
            return True
                
        except KeyboardInterrupt:
            print("Mission interrupted by user")
            # Stop all ORCA threads
            for drone_id in self.drones:
                self.orca_stop_flags[drone_id].set()
            return False
        except Exception as e:
            print(f"Error during mission execution: {e}")
            # Stop all ORCA threads
            for drone_id in self.drones:
                self.orca_stop_flags[drone_id].set()
            return False
        
    def shutdown(self):
        """Clean shutdown of all resources"""
        print("Shutting down all resources...")
        
        # Stop ORCA control threads
        for drone_id in self.drones:
            if drone_id in self.orca_stop_flags:
                self.orca_stop_flags[drone_id].set()
                print(f"Stopped ORCA controller for {drone_id}")
                
        # Wait for threads to terminate (only if they've been started)
        for drone_id, thread in self.orca_threads.items():
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        # Disconnect from drones
        for drone_id, drone in self.drones.items():
            drone.shutdown()
            print(f"Disconnected from {drone_id}")
        
        # Shutdown ROS nodes
        if hasattr(self, 'position_node') and self.position_node:
            self.position_node.destroy_node()
        
        if hasattr(self, 'obstacle_tracker') and self.obstacle_tracker:
            self.obstacle_tracker.destroy_node()
        
        print("All resources shut down")

# --------------------------
# Main
# --------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-drone mission with ORCA collision avoidance')
    parser.add_argument('--drones', type=str, default='drone1,drone2,drone3,drone4,drone5', help='Comma-separated list of drone IDs')
    parser.add_argument('-n', '--namespace', type=str, nargs='+', help='Space-separated list of drone IDs (legacy format)')
    parser.add_argument('--scenario', type=str, default='scenarios/scenario1_stage4.yaml', help='Path to scenario YAML file')
    parser.add_argument('--sim-time', action='store_true', help='Use simulation time')
    args = parser.parse_args()
    
    # Support both the new (--drones) and legacy (-n/--namespace) formats
    if args.namespace:
        drone_ids = args.namespace
    else:
        drone_ids = args.drones.split(',')
    
    # Print configuration for debugging
    print("\n==== ORCA Configuration ====")
    print(f"ORCA TIME HORIZON: {ORCA_TIME_HORIZON}s")
    print(f"ORCA OBSTACLE TIME HORIZON: {ORCA_TIME_HORIZON_OBSTACLE}s")
    print(f"ORCA RADIUS: {ORCA_RADIUS}m")
    print(f"ORCA MAX SPEED: {ORCA_MAX_SPEED}m/s")
    print(f"REPLAN FREQUENCY: {REPLAN_FREQUENCY}Hz")
    print(f"SAFETY MARGIN: {SAFETY_MARGIN}m")
    print(f"Number of drones: {len(drone_ids)}")
    print("============================\n")
    
    mission = None
    success = False
    
    try:
        # Initialize mission
        mission = MultiDroneMission(
            drone_ids=drone_ids,
            scenario_file=args.scenario,
            use_sim_time=args.sim_time,
            verbose=False
        )
        
        # Start all drones
        success = mission.start_mission()
        
        if success:
            # Run the main mission
            success = mission.run_mission()
        
        if success:
            # End mission and land drones
            mission.end_mission()
        
    except Exception as e:
        print(f"Mission failed with error: {e}")
        success = False
    finally:
        # Ensure clean shutdown
        if mission:
            mission.shutdown()
    
    print("Mission script completed")
    exit(0 if success else 1) 