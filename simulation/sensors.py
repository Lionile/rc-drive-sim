"""
Sensor system for the RC car - implements distance sensors.
"""

import numpy as np
import math
import pygame

class DistanceSensor:
    def __init__(self, angle_offset, max_range=180):
        """
        Initialize a distance sensor.
        
        Args:
            angle_offset: Sensor angle relative to car heading (radians)
            max_range: Maximum detection range in pixels (180 pixels = 18cm)
        """
        self.angle_offset = angle_offset
        self.max_range = max_range
        
    def get_distance(self, car_x, car_y, car_heading, track_boundaries):
        """
        Get distance reading from this sensor.
        
        Args:
            car_x, car_y: Car position
            car_heading: Car heading in radians
            track_boundaries: List of boundary polygons
            
        Returns:
            Normalized distance (1.0 = max range, 0.0 = touching obstacle)
        """
        # Calculate sensor direction
        sensor_angle = car_heading + self.angle_offset
        
        # Cast ray and find intersection
        min_distance = self.max_range
        
        # Simple ray casting - sample points along the ray
        num_samples = int(self.max_range)
        for i in range(1, num_samples + 1):
            # Point along the ray
            sample_x = car_x + i * math.cos(sensor_angle)
            sample_y = car_y + i * math.sin(sensor_angle)
            
            # Check if this point hits a boundary (simplified)
            if self._point_hits_boundary(sample_x, sample_y, track_boundaries):
                min_distance = i
                break
        
        # Normalize distance (1.0 = max range, 0.0 = touching)
        return min_distance / self.max_range
    
    def _point_hits_boundary(self, x, y, track_boundaries):
        """
        Simple boundary collision check.
        For now, just check if point is outside map bounds.
        """
        # Map bounds check (assuming 1024x1024 map)
        if x < 0 or x >= 1024 or y < 0 or y >= 1024:
            return True
        
        # TODO: Add proper polygon intersection check when shapely is available
        # For now, use a simple approach based on the map structure
        return False
    
    def get_ray_endpoints(self, car_x, car_y, car_heading, distance_reading):
        """
        Get the endpoints of the sensor ray for visualization.
        
        Args:
            car_x, car_y: Car position
            car_heading: Car heading in radians
            distance_reading: Normalized distance reading (0-1)
            
        Returns:
            Tuple of ((start_x, start_y), (end_x, end_y))
        """
        sensor_angle = car_heading + self.angle_offset
        actual_range = distance_reading * self.max_range
        
        start_pos = (car_x, car_y)
        end_pos = (
            car_x + actual_range * math.cos(sensor_angle),
            car_y + actual_range * math.sin(sensor_angle)
        )
        
        return start_pos, end_pos

class SensorArray:
    def __init__(self):
        """Initialize the sensor array with 3 sensors."""
        self.sensors = [
            DistanceSensor(0.0),                    # Front sensor
            DistanceSensor(math.radians(45)),       # Front-right sensor
            DistanceSensor(math.radians(-45))       # Front-left sensor
        ]
        
    def get_readings(self, car_x, car_y, car_heading, track_boundaries):
        """
        Get readings from all sensors.
        
        Returns:
            List of 3 normalized distance readings
        """
        readings = []
        for sensor in self.sensors:
            reading = sensor.get_distance(car_x, car_y, car_heading, track_boundaries)
            readings.append(reading)
        return readings
    
    def get_sensor_rays(self, car_x, car_y, car_heading, readings):
        """
        Get sensor ray endpoints for visualization.
        
        Returns:
            List of tuples: [((start_x, start_y), (end_x, end_y)), ...]
        """
        rays = []
        for sensor, reading in zip(self.sensors, readings):
            ray = sensor.get_ray_endpoints(car_x, car_y, car_heading, reading)
            rays.append(ray)
        return rays
