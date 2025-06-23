import numpy as np
from geometry_msgs.msg import Point, Vector3
import math

class ORCAFormation:
    def __init__(self, radius=1.0, time_horizon=2.0, neighbor_dist=10.0, max_neighbors=10):
        """
        Initialize ORCA with formation constraints
        
        Args:
            radius: Collision radius of the drone
            time_horizon: Time horizon for collision avoidance
            neighbor_dist: Maximum distance to consider neighbors
            max_neighbors: Maximum number of neighbors to consider
        """
        self.radius = radius
        self.time_horizon = time_horizon
        self.neighbor_dist = neighbor_dist
        self.max_neighbors = max_neighbors
        
        # Formation parameters
        self.formation_strength = 0.5  # Strength of formation attraction
        self.formation_threshold = 2.0  # Distance threshold for formation attraction
        
    def compute_velocity(self, position, velocity, goal_velocity, neighbors, formation_positions):
        """
        Compute new velocity using ORCA with formation constraints
        
        Args:
            position: Current position (Point)
            velocity: Current velocity (Vector3)
            goal_velocity: Desired velocity (Vector3)
            neighbors: List of neighbor positions and velocities [(Point, Vector3)]
            formation_positions: List of formation slot positions [Point]
            
        Returns:
            Vector3: New velocity
        """
        # Convert to numpy arrays for easier computation
        pos = np.array([position.x, position.y, position.z])
        vel = np.array([velocity.x, velocity.y, velocity.z])
        goal_vel = np.array([goal_velocity.x, goal_velocity.y, goal_velocity.z])
        
        # Initialize ORCA constraints
        orca_lines = []
        
        # Add constraints from neighbors
        for neighbor_pos, neighbor_vel in neighbors:
            n_pos = np.array([neighbor_pos.x, neighbor_pos.y, neighbor_pos.z])
            n_vel = np.array([neighbor_vel.x, neighbor_vel.y, neighbor_vel.z])
            
            # Compute relative position and velocity
            relative_pos = n_pos - pos
            relative_vel = vel - n_vel
            
            # Compute distance
            dist = np.linalg.norm(relative_pos)
            
            # Skip if too far away
            if dist > self.neighbor_dist:
                continue
                
            # Compute collision time
            collision_time = self._compute_collision_time(relative_pos, relative_vel)
            
            if collision_time < self.time_horizon:
                # Add ORCA constraint
                orca_lines.append(self._compute_orca_line(relative_pos, relative_vel, collision_time))
        
        # Add formation constraints
        formation_force = self._compute_formation_force(pos, formation_positions)
        
        # Find new velocity that satisfies all constraints
        new_vel = self._linear_program2(orca_lines, goal_vel + formation_force)
        
        # Convert back to Vector3
        return Vector3(x=float(new_vel[0]), y=float(new_vel[1]), z=float(new_vel[2]))
    
    def _compute_collision_time(self, relative_pos, relative_vel):
        """Compute time until collision"""
        dist = np.linalg.norm(relative_pos)
        relative_speed = np.linalg.norm(relative_vel)
        
        if relative_speed == 0:
            return float('inf')
            
        # Time until collision
        return (dist - 2 * self.radius) / relative_speed
    
    def _compute_orca_line(self, relative_pos, relative_vel, collision_time):
        """Compute ORCA line for a neighbor"""
        # Normalize relative position
        relative_pos_norm = relative_pos / np.linalg.norm(relative_pos)
        
        # Compute u (change in velocity needed to avoid collision)
        u = (2 * self.radius / collision_time) * relative_pos_norm - relative_vel
        
        # Compute ORCA line
        n = u / np.linalg.norm(u)
        p = 0.5 * (relative_vel + u)
        
        return (n, p)
    
    def _compute_formation_force(self, position, formation_positions):
        """Compute formation attraction force"""
        force = np.zeros(3)
        
        for formation_pos in formation_positions:
            f_pos = np.array([formation_pos.x, formation_pos.y, formation_pos.z])
            relative_pos = f_pos - position
            dist = np.linalg.norm(relative_pos)
            
            if dist > self.formation_threshold:
                # Compute attractive force
                force += self.formation_strength * relative_pos / dist
        
        return force
    
    def _linear_program2(self, orca_lines, goal_velocity):
        """Find new velocity that satisfies ORCA constraints"""
        if not orca_lines:
            return goal_velocity
            
        # Initialize with goal velocity
        result = goal_velocity
        
        # Iteratively project onto ORCA lines
        for i in range(len(orca_lines)):
            n, p = orca_lines[i]
            
            # Project current velocity onto line
            dot_product = np.dot(result - p, n)
            
            if dot_product < 0:
                # Velocity violates constraint, project onto line
                result = result - dot_product * n
        
        return result 