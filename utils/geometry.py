"""
Geometric utility functions for collision detection.
"""

import math

def segment_intersection(p1, p2, q1, q2, epsilon=1e-9):
    """
    Find intersection between two line segments.
    
    Args:
        p1, p2: First segment endpoints as (x, y) tuples
        q1, q2: Second segment endpoints as (x, y) tuples
        epsilon: Tolerance for floating point comparisons
        
    Returns:
        Intersection point as (x, y) tuple if segments intersect, None otherwise
    """
    # Vector from p1 to p2
    r = (p2[0] - p1[0], p2[1] - p1[1])
    # Vector from q1 to q2
    s = (q2[0] - q1[0], q2[1] - q1[1])
    # Vector from p1 to q1
    qp = (q1[0] - p1[0], q1[1] - p1[1])
    
    # Cross product r × s
    cross_r_s = r[0] * s[1] - r[1] * s[0]
    
    # If lines are parallel (cross product is zero)
    if abs(cross_r_s) < epsilon:
        return None  # Parallel or colinear - treat as no intersection
    
    # Calculate parameters t and u
    t = (qp[0] * s[1] - qp[1] * s[0]) / cross_r_s
    u = (qp[0] * r[1] - qp[1] * r[0]) / cross_r_s
    
    # Check if intersection is within both segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        # Calculate intersection point
        intersection_x = p1[0] + t * r[0]
        intersection_y = p1[1] + t * r[1]
        return (intersection_x, intersection_y)
    
    return None

def distance_squared(p1, p2):
    """Calculate squared distance between two points (avoids sqrt for speed)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return dx * dx + dy * dy

def distance(p1, p2):
    """Calculate distance between two points."""
    return math.sqrt(distance_squared(p1, p2))

def contour_to_segments(points):
    """
    Convert a list of contour points to line segments.
    
    Args:
        points: List of (x, y) tuples representing polygon vertices
        
    Returns:
        List of ((x1, y1), (x2, y2)) tuples representing line segments
    """
    if len(points) < 2:
        return []
    
    segments = []
    for i in range(len(points)):
        start = points[i]
        end = points[(i + 1) % len(points)]  # Wrap around to close the polygon
        segments.append((start, end))
    
    return segments
