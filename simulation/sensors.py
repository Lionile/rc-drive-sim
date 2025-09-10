"""
Sensor system for the RC car - implements distance sensors.
"""

import numpy as np
import math
import pygame
from utils.geometry import segment_intersection, distance

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
        
    def get_distance(self, car_x, car_y, car_heading, boundary_segments):
        """
        Get distance reading from this sensor using segment intersection.
        
        Args:
            car_x, car_y: Car position
            car_heading: Car heading in radians
            boundary_segments: List of ((x1,y1), (x2,y2)) boundary line segments
            
        Returns:
            Normalized distance (1.0 = max range, 0.0 = touching obstacle)
        """
        # Calculate sensor direction
        sensor_angle = car_heading + self.angle_offset
        
        # Calculate ray end point
        end_x = car_x + self.max_range * math.cos(sensor_angle)
        end_y = car_y + self.max_range * math.sin(sensor_angle)
        
        # Find closest intersection with boundary segments
        min_distance = self.max_range
        closest_hit = None
        
        ray_start = (car_x, car_y)
        ray_end = (end_x, end_y)
        
        for segment in boundary_segments:
            intersection = segment_intersection(ray_start, ray_end, segment[0], segment[1])
            if intersection:
                # Calculate distance to intersection
                hit_distance = distance(ray_start, intersection)
                if hit_distance < min_distance:
                    min_distance = hit_distance
                    closest_hit = intersection
        
        # Store hit point for visualization
        self.last_hit = closest_hit
        
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
        
    def get_readings(self, car_x, car_y, car_heading, boundary_segments):
        """
        Get readings from all sensors.
        
        Returns:
            List of 3 normalized distance readings
        """
        readings = []
        for sensor in self.sensors:
            reading = sensor.get_distance(car_x, car_y, car_heading, boundary_segments)
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
