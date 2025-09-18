import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def downsample_path(path, num_points=30):
    path_length = len(path)
    
    if path_length <= num_points:
        return path
    
    step = path_length / num_points
    downsampled_path = [path[int(i * step)] for i in range(num_points)]
    
    return downsampled_path

def extract_and_scale_contours(image_path, epsilon_factor=0.001, plot=False):
    """
    Extracts the first contour from an image, simplifies it using the RDP algorithm,
    and scales it by the specified factor.

    Args:
        image_path (str): Path to the image file.
        epsilon_factor (float): Factor to control the simplification level (higher means more simplification).

    Returns:
        tuple: A tuple containing two lists of tuples:
            - List of points for the simplified first contour.
            - List of points for the simplified contour.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return [], []

    first_contour = contours[0]
    second_contour = contours[1]

    epsilon = epsilon_factor * cv2.arcLength(first_contour, True)
    simplified_contour_first = cv2.approxPolyDP(first_contour, epsilon, True)
    simplified_contour_second = cv2.approxPolyDP(second_contour, epsilon, True)

    simplified_contour_points_first = [(int(point[0][0]), int(point[0][1])) for point in simplified_contour_first]
    simplified_contour_points_second = [(int(point[0][0]), int(point[0][1])) for point in simplified_contour_second]

    # Ensure contours are closed loops by adding the first point at the end if not already closed
    if simplified_contour_points_first and simplified_contour_points_first[0] != simplified_contour_points_first[-1]:
        simplified_contour_points_first.append(simplified_contour_points_first[0])
    
    if simplified_contour_points_second and simplified_contour_points_second[0] != simplified_contour_points_second[-1]:
        simplified_contour_points_second.append(simplified_contour_points_second[0])

    start_points = find_green_circle_center(image_path)
    headings = [3.14]

    if plot:
        # Visualize the contours
        cv2.drawContours(image, [simplified_contour_first], -1, (0, 255, 0), 2)
        cv2.drawContours(image, [first_contour], -1, (0, 0, 255), 2)
        cv2.drawContours(image, [simplified_contour_second], -1, (255, 0, 0), 2)
        cv2.drawContours(image, [second_contour], -1, (255, 255, 0), 2)
        
        cv2.imshow('Contours', image)
        cv2.waitKey(0)  # Wait for a key press to close
        cv2.destroyAllWindows()

    return simplified_contour_points_first, simplified_contour_points_second, start_points, headings



def are_lines_similar(line1, line2, distance_threshold=10, angle_threshold=5):
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    def distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    distances = [
        distance((x1, y1), (x3, y3)),
        distance((x1, y1), (x4, y4)),
        distance((x2, y2), (x3, y3)),
        distance((x2, y2), (x4, y4)),
    ]

    if min(distances) < distance_threshold:
        def angle(x1, y1, x2, y2):
            return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

        angle1 = angle(x1, y1, x2, y2)
        angle2 = angle(x3, y3, x4, y4)

        if abs(angle1 - angle2) < angle_threshold:
            return True

    return False

def find_green_circle_center(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 40, 40])  # Lower bound of green in HSV
    upper_green = np.array([80, 255, 255])  # Upper bound of green in HSV

    mask = cv2.inRange(hsv, lower_green, upper_green)

    blurred = cv2.GaussianBlur(mask, (9, 9), 2)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=0, maxRadius=0)
    circles_pos = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, radius = circle
            circles_pos.append((x, y))

    return circles_pos


def generate_racing_line(image_path, num_points=100, epsilon_factor=0.001):
    """
    Generates a racing line (the middle of the track) from an image.
    Extracts contours for the track edges, and finds the center line by projecting perpendiculars from one edge to the other.
    Args:
        image_path (str): Path to the image file.
        num_points (int): Number of points in the final racing line.
        scale (float): Scaling factor for the contour.
        epsilon_factor (float): Factor to control the simplification level.
    """
    first_contour, second_contour, start_points, heading = extract_and_scale_contours(image_path, epsilon_factor)

    if not first_contour or not second_contour:
        raise ValueError("Contours could not be extracted from the image.")

    # Downsample both contours to the same number of points for efficiency
    first_contour = downsample_path(first_contour, num_points * 3)
    second_contour = downsample_path(second_contour, num_points * 3)

    # Convert to numpy arrays for vectorized operations
    first_arr = np.array(first_contour)
    second_arr = np.array(second_contour)

    def find_closest_point(point, contour):
        dists = np.linalg.norm(contour - point, axis=1)
        idx = np.argmin(dists)
        return contour[idx]

    # For each point on the first contour, find the closest point on the second contour,
    # then take the midpoint as the racing line point.
    racing_line = []
    for pt in first_arr:
        closest = find_closest_point(pt, second_arr)
        mid = ((pt[0] + closest[0]) // 2, (pt[1] + closest[1]) // 2)
        racing_line.append(mid)

    # Downsample the racing line to the desired number of points
    racing_line = downsample_path(racing_line, num_points)

    def racing_line_generate(racing_line, start_point):
        # Shift the racing line so that the first point is closest to the start point
        if start_point is not None and racing_line:
            dists = [np.linalg.norm(np.array(pt) - np.array(start_point)) for pt in racing_line]
            min_idx = int(np.argmin(dists))
            racing_line = racing_line[min_idx:] + racing_line[:min_idx]

        # append start point
        racing_line.append(start_point)

        # Connect big gaps between consecutive points
        max_gap = 1.5 * np.mean([
            np.linalg.norm(np.array(racing_line[i]) - np.array(racing_line[i - 1]))
            for i in range(1, len(racing_line))
        ])
        connected_line = []
        for i, pt in enumerate(racing_line):
            connected_line.append(pt)
            if i > 0:
                prev_pt = racing_line[i - 1]
                dist = np.linalg.norm(np.array(pt) - np.array(prev_pt))
                if dist > max_gap:
                    # Insert intermediate points to connect the gap
                    num_steps = int(np.ceil(dist / max_gap))
                    for step in range(1, num_steps):
                        interp = (
                            int(prev_pt[0] + (pt[0] - prev_pt[0]) * step / num_steps),
                            int(prev_pt[1] + (pt[1] - prev_pt[1]) * step / num_steps)
                        )
                        connected_line.insert(-1, interp)  # Insert before the current point

        return connected_line[:1:-1], start_point
    
    racing_lines = []
    for start in start_points:
        # Shift the racing line so that the first point is closest to the start point
        shifted_line, start_point = racing_line_generate(racing_line, start)

        # Append the shifted racing line to the list
        racing_lines.append((shifted_line, start_point))
    
    return racing_lines



def load_distance_field(map_path):
    """
    Load distance field image corresponding to the given map path.
    
    Args:
        map_path (str): Path to the map file (e.g., "maps/map_start1.png")
        
    Returns:
        np.ndarray or None: Distance field as float32 array normalized to [0,1], 
                           or None if distance field file not found
    """
    # Extract directory and base filename
    dirname, filename = os.path.split(map_path)
    
    # Simple replacement: map_start1.png -> map_distance1.png
    distance_filename = filename.replace("map_start", "map_distance")
    distance_path = os.path.join(dirname, distance_filename)
    
    # Load distance field if it exists
    if os.path.exists(distance_path):
        try:
            # Load as grayscale
            distance_img = cv2.imread(distance_path, cv2.IMREAD_GRAYSCALE)
            if distance_img is not None:
                # Convert to float32 and normalize to [0,1]
                distance_field = distance_img.astype(np.float32) / 255.0
                return distance_field
        except Exception as e:
            print(f"Warning: Could not load distance field from {distance_path}: {e}")
    
    return None



def visualize_track_lines(image_path):
    # Get track edges (contours)
    first_contour, second_contour, start_points, headings = extract_and_scale_contours(image_path)
    # Get racing lines
    racing_lines = generate_racing_line(image_path, num_points=200, epsilon_factor=0.0001)
    # Load distance field
    distance_field = load_distance_field(image_path)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    # --- Left: Track with edges and racing line ---
    ax = axes[0]
    ax.imshow(cv2.imread(image_path))
    if first_contour:
        x1, y1 = zip(*first_contour)
        ax.plot(x1, y1, color='blue', linewidth=2, label='Inner Edge')
    if second_contour:
        x2, y2 = zip(*second_contour)
        ax.plot(x2, y2, color='red', linewidth=2, label='Outer Edge')
    for i, (racing_line, start_point) in enumerate(racing_lines):
        x, y = zip(*racing_line)
        ax.plot(x, y, marker='o', markersize=3, label=f'Racing Line {i+1}')
        if start_point:
            ax.scatter(*start_point, label=f'Start Point {i+1}')
        for idx, (xi, yi) in enumerate(zip(x, y)):
            ax.text(xi, yi - 5, str(idx), fontsize=8, ha='center')
    ax.legend(loc='upper left')
    ax.axis('off')
    ax.set_title('Track Edges and Racing Lines')

    # --- Right: Distance field ---
    ax2 = axes[1]
    if distance_field is not None:
        ax2.imshow(distance_field, cmap='viridis')
        ax2.set_title('Distance Field')
    else:
        ax2.text(0.5, 0.5, 'Distance field not found', ha='center', va='center', fontsize=16)
        ax2.set_title('Distance Field')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


# test and plot racing line generation
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python map_utils.py <image_path>")
        print("Example: python map_utils.py maps/map_start3.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    visualize_track_lines(image_path)