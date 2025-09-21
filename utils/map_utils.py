import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom
from scipy.signal import savgol_filter
from PIL import Image



def _to_uint8(mask):
    """Accept bool/uint8 and return (uint8_mask, original_dtype)."""
    if mask.dtype == np.bool_:
        m8 = np.where(mask, 255, 0).astype(np.uint8)
        return m8, mask.dtype
    elif mask.dtype == np.uint8:
        return mask.copy(), mask.dtype
    else:
        raise TypeError("Mask must be uint8 (0/255) or bool.")

def _from_uint8(mask_u8, out_dtype):
    if out_dtype == np.bool_:
        return mask_u8 > 127
    return mask_u8

def _disk_kernel(radius_px: int) -> np.ndarray:
    """Elliptical (disk-like) structuring element with given radius in pixels."""
    if radius_px < 1:
        raise ValueError("radius_px must be >= 1")
    k = 2 * radius_px + 1  # odd size
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

def erode_track(mask, radius_px: int = 1, iterations: int = 1):
    """
    Shrinks the black track area by ~radius_px pixels (per iteration),
    keeping background white.
    """
    m8, dtype0 = _to_uint8(mask)
    kernel = _disk_kernel(radius_px)
    # Track is black => invert so track becomes foreground (255)
    inv = 255 - m8
    inv_eroded = cv2.erode(inv, kernel, iterations=iterations)
    out = 255 - inv_eroded
    return _from_uint8(out, dtype0)

def dilate_track(mask, radius_px: int = 1, iterations: int = 1):
    """
    Expands/thickens the black track area by ~radius_px pixels (per iteration),
    keeping background white.
    """
    m8, dtype0 = _to_uint8(mask)
    kernel = _disk_kernel(radius_px)
    inv = 255 - m8
    inv_dilated = cv2.dilate(inv, kernel, iterations=iterations)
    out = 255 - inv_dilated
    return _from_uint8(out, dtype0)



def prep_mask(image_path):
    """
    Loads the image, removes green start circle(s), and returns:
      - track_mask : boolean (False = drivable track, True = background)
      - start_points : list[(x,y)]
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")

    # Detect & remove green circles (paint them as background = white)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    blurred = cv2.GaussianBlur(mask, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,
                               dp=1, minDist=50, param1=50, param2=30,
                               minRadius=0, maxRadius=0)
    start_points = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0, :]:
            start_points.append((int(x), int(y)))
            cv2.circle(image, (int(x), int(y)), int(r) + 5, (0, 0, 0), -1)  # paint white (background)

    # Track = black (0), Background = white (255)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # Boolean mask: True = background, False = track
    track_mask = bw.astype(bool)
    return track_mask, start_points


def edges_from_mask(track_mask):
    """
    Returns (inner_edge, outer_edge) as boolean masks (1-px wide, disjoint).
    track_mask: True = background, False = track
    """
    bg = track_mask.astype(bool)      # background = True
    track = ~bg                       # track = True
    lbl = measure.label(bg, connectivity=1)

    # outer background = any bg touching the border
    border_ids = np.unique(np.r_[lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]])
    outer_id = next(i for i in border_ids if i != 0)

    # inner background (hole) = largest non-outer bg component
    inner_candidates = [i for i in np.unique(lbl) if i not in (0, outer_id)]
    if not inner_candidates:
        raise ValueError("No inner hole found in track mask.")
    inner_id = max(inner_candidates, key=lambda i: (lbl == i).sum())

    se4 = np.array([[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]], np.uint8)
    inner_bg = (lbl == inner_id)
    outer_bg = (lbl == outer_id)

    inner_edge = track & cv2.dilate(inner_bg.astype(np.uint8), se4, 1).astype(bool)
    outer_edge = track & cv2.dilate(outer_bg.astype(np.uint8), se4, 1).astype(bool)

    # make edges disjoint if any pixel got tagged as both
    both = inner_edge & outer_edge
    if np.any(both):
        inner_edge[both] = False
        outer_edge[both] = False

    return inner_edge, outer_edge


def edge_polylines_from_masks(inner_edge, outer_edge, epsilon_px=2.0):
    """
    Converts edge masks to simplified polylines (x,y), closed.
    Returns (outer_poly, inner_poly) as float arrays Nx2.
    """
    def mask_to_poly(mask):
        cnts, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return np.zeros((0, 2), float)
        cnt = max(cnts, key=lambda c: cv2.arcLength(c, True))
        approx = cv2.approxPolyDP(cnt, epsilon_px, True)
        P = approx[:, 0, :].astype(float)
        if np.linalg.norm(P[0] - P[-1]) > 1e-6:
            P = np.vstack([P, P[0]])
        return P

    outer_poly = mask_to_poly(outer_edge)
    inner_poly = mask_to_poly(inner_edge)
    return outer_poly, inner_poly


def distance_fields(inner_edge, outer_edge, track_mask):
    """
    Returns d_in, d_out, d_edge, width as float arrays (NaN on background).
    """
    track = ~track_mask  # track = True
    d_in = distance_transform_edt(~inner_edge).astype(float)
    d_out = distance_transform_edt(~outer_edge).astype(float)
    d_in[~track] = np.nan
    d_out[~track] = np.nan
    d_edge = np.fmin(d_in, d_out)
    width = d_in + d_out
    return d_in, d_out, d_edge, width


def centerline_from_dist(d_in, d_out, track_mask, sigma=1.0):
    """
    Builds phi = d_in - d_out, extracts the zero-level contour,
    and returns the longest closed loop as an (x,y) array (raw, unsmoothed).
    """
    H, W = track_mask.shape
    phi = gaussian_filter(d_in - d_out, sigma)
    contours = measure.find_contours(np.nan_to_num(phi, nan=1e6), 0.0)

    track = ~track_mask  # track = True
    loops = []
    for c in contours:
        rr = np.clip(np.round(c[:, 0]).astype(int), 0, H-1)
        cc = np.clip(np.round(c[:, 1]).astype(int), 0, W-1)
        if np.all(track[rr, cc]):
            loops.append(np.c_[cc, c[:, 0]])  # (x,y)

    if not loops:
        raise RuntimeError("No closed midline found.")
    C = max(loops, key=lambda v: np.linalg.norm(np.diff(v, axis=0), axis=1).sum())
    return C


def resample_closed(C, step=6.0, smooth_win=51, poly=3, min_points=100):
    if np.linalg.norm(C[0] - C[-1]) > 1e-6:
        C = np.vstack([C, C[0]])
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    L = s[-1]
    N = max(min_points, int(round(L / step)))
    S = np.linspace(0, L, N+1)[:-1]
    X = np.interp(S, s, C[:, 0])
    Y = np.interp(S, s, C[:, 1])
    if smooth_win % 2 == 0:
        smooth_win += 1
    X = savgol_filter(X, smooth_win, poly, mode='wrap')
    Y = savgol_filter(Y, smooth_win, poly, mode='wrap')
    return np.column_stack([X, Y])


def compute_centerline(track_mask,
                       sigma=1.0,
                       step=40.0,
                       smooth_win=7,
                       poly=3,
                       return_edge_polys=False,
                       edge_epsilon_px=2.0):
    inner_edge, outer_edge = edges_from_mask(track_mask)
    d_in, d_out, d_edge, width = distance_fields(inner_edge, outer_edge, track_mask)
    raw_center = centerline_from_dist(d_in, d_out, track_mask, sigma=sigma)
    centerline = resample_closed(raw_center, step=step, smooth_win=smooth_win, poly=poly)

    clearance_norm = np.clip(2 * d_edge / np.maximum(width, 1e-6), 0, 1)
    # Handle NaN values before casting to uint8
    clearance_norm = np.nan_to_num(clearance_norm, nan=0.0)
    dist_u8 = (255 * clearance_norm).astype(np.uint8)
    dist_u8[track_mask] = 0  # background = True → 0

    outputs = {
        "centerline": centerline,
        "d_in": d_in,
        "d_out": d_out,
        "d_edge": d_edge,
        "width": width,
        "dist_u8": dist_u8
    }

    if return_edge_polys:
        outer_poly, inner_poly = edge_polylines_from_masks(inner_edge, outer_edge, edge_epsilon_px)
        outputs["outer_poly"] = outer_poly
        outputs["inner_poly"] = inner_poly

    return outputs



def create_stylized_track_image(track_mask, texture_strength=3, seed=42):
    """
    Create a stylized racetrack image from a boolean track mask.
    
    Parameters:
    - track_mask: boolean numpy array (True = track/drivable area, False = background)
    - texture_strength: int, strength of texture noise (default: 3)
    - seed: int, random seed for reproducible textures (default: 42)
    
    Returns:
    - PIL Image in RGB format
    """
    
    # Ensure boolean type
    track_mask = track_mask.astype(bool)
    H, W = track_mask.shape
    S = min(W, H)
    
    # Colors (approx from reference)
    grass = np.array([81, 115, 49], dtype=np.uint8)    # green field
    asphalt = np.array([44, 44, 47], dtype=np.uint8)   # road surface
    shadow = np.array([24, 24, 24], dtype=np.uint8)    # outer shadow ring
    stripe = np.array([235, 234, 230], dtype=np.uint8) # off-white stripe
    
    # Parameters scaled by image size
    outer_shadow_r = max(1, int(0.003 * S))     # slight dark outline outside the road
    inner_white_r = max(2, int(0.004 * S))      # white line inside the track edge
    inner_white_width = max(2, int(0.005 * S))  # thickness of the inner white line
    stripe_offset_r = max(4, int(0.028 * S))    # gap from the road edge to the stripe
    stripe_width_r = max(3, int(0.012 * S))     # stripe thickness
    
    # Build masks for rendering using existing erode/dilate functions
    road = track_mask.copy()
    
    # Outer shadow ring: dilate track and take the difference
    outer_dil = dilate_track(road, outer_shadow_r)
    outer_ring = np.logical_and(outer_dil, ~road)
    
    # Inner white line: erode track to get inner boundary
    road_inner = erode_track(road, inner_white_r)
    road_inner_more = erode_track(road, inner_white_r + inner_white_width)
    inner_white_band = np.logical_and(road_inner, ~road_inner_more)
    
    # Stripe band: erode further inward for the stripe
    road_inward = erode_track(road, stripe_offset_r)
    road_inward_more = erode_track(road, stripe_offset_r + stripe_width_r)
    stripe_band = np.logical_and(road_inward, ~road_inward_more)
    
    # Compose the image
    canvas = np.tile(grass, (H, W, 1))
    
    # Draw shadow ring
    canvas[outer_ring] = shadow
    
    # Draw road fill
    canvas[road] = asphalt
    
    # Draw inner white line
    canvas[inner_white_band] = stripe
    
    # Draw white stripe
    canvas[stripe_band] = stripe
    
    # Add patchy texture for realism
    def generate_patchy_noise(shape, strength, patch_size=8, contrast=2.0, seed=42):
        """Generate patchy, sharp-looking noise with clusters"""
        rng = np.random.default_rng(seed)
        H, W = shape[:2]
        
        # Create base noise at lower resolution for patches
        low_res_h, low_res_w = H // patch_size, W // patch_size
        if low_res_h == 0 or low_res_w == 0:
            low_res_h = max(1, low_res_h)
            low_res_w = max(1, low_res_w)
        base_noise = rng.normal(0, strength, (low_res_h, low_res_w))
        
        # Upsample using nearest neighbor to create sharp patches
        patchy_noise = zoom(base_noise, patch_size, order=0)  # order=0 = nearest neighbor
        
        # Crop or pad to exact size
        if patchy_noise.shape[0] > H:
            patchy_noise = patchy_noise[:H, :]
        if patchy_noise.shape[1] > W:
            patchy_noise = patchy_noise[:, :W]
        if patchy_noise.shape[0] < H or patchy_noise.shape[1] < W:
            padded = np.zeros((H, W))
            padded[:patchy_noise.shape[0], :patchy_noise.shape[1]] = patchy_noise
            patchy_noise = padded
        
        # Add very subtle high-frequency detail noise
        detail_noise = rng.integers(-strength//4, strength//4 + 1, size=(H, W), dtype=np.int16)
        
        # Combine for final patchy texture
        final_noise = (patchy_noise * contrast + detail_noise).astype(np.int16)
        
        # Expand to 3 channels
        return final_noise[..., np.newaxis]
    
    rng = np.random.default_rng(seed)
    grass_mask = ~road
    asphalt_mask = road
    
    canvas = canvas.astype(np.int16)
    
    # Generate textures - more pronounced for grass, keep asphalt subtle
    grass_noise = generate_patchy_noise((H, W), texture_strength * 1.5, patch_size=2, contrast=1.0, seed=seed)
    asphalt_noise = generate_patchy_noise((H, W), texture_strength, patch_size=2, contrast=0.6, seed=seed+1)
    
    canvas[grass_mask] = np.clip(canvas[grass_mask] + grass_noise[grass_mask], 0, 255)
    canvas[asphalt_mask] = np.clip(canvas[asphalt_mask] + asphalt_noise[asphalt_mask], 0, 255)
    canvas = canvas.astype(np.uint8)
    
    # Convert to PIL Image
    styled = Image.fromarray(canvas, mode="RGB")
    
    return styled



def create_stylized_track_image(track_mask, texture_strength=3, seed=42):
    """
    Create a stylized racetrack image from a boolean track mask.

    Parameters:
    - track_mask: boolean numpy array (True = track/drivable area, False = background)
    - texture_strength: int, strength of texture noise (default: 3)
    - seed: int, random seed for reproducible textures (default: 42)

    Returns:
    - PIL Image in RGB format
    """

    # Ensure boolean type
    track_mask = ~track_mask.astype(bool)
    H, W = track_mask.shape
    S = min(W, H)

    # Colors (approx from reference)
    grass = np.array([81, 115, 49], dtype=np.uint8)    # green field
    asphalt = np.array([44, 44, 47], dtype=np.uint8)   # road surface
    shadow = np.array([24, 24, 24], dtype=np.uint8)    # outer shadow ring
    stripe = np.array([235, 234, 230], dtype=np.uint8) # off-white stripe

    # Parameters scaled by image size
    outer_shadow_r = max(1, int(0.003 * S))     # slight dark outline outside the road
    inner_white_r = max(2, int(0.004 * S))      # white line inside the track edge
    inner_white_width = max(2, int(0.005 * S))  # thickness of the inner white line
    stripe_offset_r = max(4, int(0.028 * S))    # gap from the road edge to the stripe
    stripe_width_r = max(3, int(0.012 * S))     # stripe thickness

    # Build masks for rendering using existing erode/dilate functions
    road = track_mask.copy()

    # Outer shadow ring: dilate track and take the difference
    outer_dil = dilate_track(road, outer_shadow_r)
    outer_ring = np.logical_and(outer_dil, ~road)

    # Inner white line: erode track to get inner boundary
    road_inner = erode_track(road, inner_white_r)
    road_inner_more = erode_track(road, inner_white_r + inner_white_width)
    inner_white_band = np.logical_and(road_inner, ~road_inner_more)

    # Stripe band: erode further inward for the stripe
    road_inward = erode_track(road, stripe_offset_r)
    road_inward_more = erode_track(road, stripe_offset_r + stripe_width_r)
    stripe_band = np.logical_and(road_inward, ~road_inward_more)

    # Compose the image
    canvas = np.tile(grass, (H, W, 1))

    # Draw shadow ring
    canvas[outer_ring] = shadow

    # Draw road fill
    canvas[road] = asphalt

    # Draw inner white line
    canvas[inner_white_band] = stripe

    # Draw white stripe
    canvas[stripe_band] = stripe

    # Add patchy texture for realism
    def generate_patchy_noise(shape, strength, patch_size=8, contrast=2.0, seed=42):
        """Generate patchy, sharp-looking noise with clusters"""
        rng = np.random.default_rng(seed)
        H, W = shape[:2]

        # Create base noise at lower resolution for patches
        low_res_h, low_res_w = H // patch_size, W // patch_size
        if low_res_h == 0 or low_res_w == 0:
            low_res_h = max(1, low_res_h)
            low_res_w = max(1, low_res_w)
        base_noise = rng.normal(0, strength, (low_res_h, low_res_w))

        # Upsample using nearest neighbor to create sharp patches
        patchy_noise = zoom(base_noise, patch_size, order=0)  # order=0 = nearest neighbor

        # Crop or pad to exact size
        if patchy_noise.shape[0] > H:
            patchy_noise = patchy_noise[:H, :]
        if patchy_noise.shape[1] > W:
            patchy_noise = patchy_noise[:, :W]
        if patchy_noise.shape[0] < H or patchy_noise.shape[1] < W:
            padded = np.zeros((H, W))
            padded[:patchy_noise.shape[0], :patchy_noise.shape[1]] = patchy_noise
            patchy_noise = padded

        # Add very subtle high-frequency detail noise
        detail_noise = rng.integers(-strength//4, strength//4 + 1, size=(H, W), dtype=np.int16)

        # Combine for final patchy texture
        final_noise = (patchy_noise * contrast + detail_noise).astype(np.int16)

        # Expand to 3 channels
        return final_noise[..., np.newaxis]

    rng = np.random.default_rng(seed)
    grass_mask = ~road
    asphalt_mask = road

    canvas = canvas.astype(np.int16)

    # Generate textures - more pronounced for grass, keep asphalt subtle
    grass_noise = generate_patchy_noise((H, W), texture_strength * 1.5, patch_size=2, contrast=1.0, seed=seed)
    asphalt_noise = generate_patchy_noise((H, W), texture_strength, patch_size=2, contrast=0.6, seed=seed+1)

    canvas[grass_mask] = np.clip(canvas[grass_mask] + grass_noise[grass_mask], 0, 255)
    canvas[asphalt_mask] = np.clip(canvas[asphalt_mask] + asphalt_noise[asphalt_mask], 0, 255)
    canvas = canvas.astype(np.uint8)

    # Convert to PIL Image
    styled = Image.fromarray(canvas, mode="RGB")

    return styled



def project_and_reorder_centerline(centerline: np.ndarray, start_xy: tuple | np.ndarray):
    """
    centerline: Nx2 float array, CLOSED curve *without* a duplicate last point.
    start_xy : (x, y) start marker in image coords.

    Returns:
      new_centerline: Nx2 float array starting at the projected point, then forward
      proj_pt: (x,y) projection of start_xy onto the centerline
      seg_idx: index i of segment [i -> i+1] (wrapped) containing the projection
      t: param in [0,1] along that segment
      dist2: squared distance from start to proj
    """
    C = np.asarray(centerline, dtype=float)
    # If user accidentally passed a closed curve with duplicate final vertex, drop it
    if np.linalg.norm(C[0] - C[-1]) < 1e-9:
        C = C[:-1]

    P = C
    Q = np.roll(C, -1, axis=0)             # segment ends (wrap)
    v = Q - P                               # segment vectors
    s = np.asarray(start_xy, float)

    # projection t for each segment (clamped to [0,1])
    vv = (v[:, 0]**2 + v[:, 1]**2)
    vv = np.where(vv < 1e-12, 1e-12, vv)    # avoid div by zero for degenerate segments
    t = np.clip(((s[0]-P[:,0])*v[:,0] + (s[1]-P[:,1])*v[:,1]) / vv, 0.0, 1.0)

    proj = P + t[:, None] * v               # projected points on each segment
    dist2 = (proj[:, 0] - s[0])**2 + (proj[:, 1] - s[1])**2
    i = int(np.argmin(dist2))               # best segment
    proj_pt = proj[i]

    # Reorder so we start at proj_pt and then follow vertices forward once
    new_centerline = np.vstack([proj_pt, C[i+1:], C[:i+1]])
    return new_centerline, proj_pt, i, float(t[i]), float(dist2[i])



def visualize_track_lines(image_path):
    track_mask, start_points = prep_mask(image_path)

    try:
        out = compute_centerline(track_mask, return_edge_polys=True)
        centerline = out["centerline"]
        outer_poly = out.get("outer_poly", None)
        inner_poly = out.get("inner_poly", None)
        # project the start point onto the centerline, and reorder to start there
        centerline, proj_pt, seg_idx, t, d2 = project_and_reorder_centerline(
            centerline,
            np.array(start_points[0], dtype=float)  # use first start dot
        )
        start_points = [proj_pt]  # replace with projected point
    except Exception as e:
        print(f"Warning: Could not compute centerline and edges: {e}")
        centerline, outer_poly, inner_poly = None, None, None

    # Create subplot layout: 1 row, 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    # Left subplot: Original track analysis
    ax1.imshow(track_mask, cmap="gray", interpolation="nearest")
    ax1.set_title('Track Analysis (Black=Track)')

    def plot_closed(poly, *args, **kwargs):
        if poly is None or len(poly) == 0:
            return
        x = np.r_[poly[:, 0], poly[0, 0]]
        y = np.r_[poly[:, 1], poly[0, 1]]
        ax1.plot(x, y, *args, **kwargs)

    plot_closed(outer_poly, color='cyan', linewidth=2.0, alpha=0.9, label='Outer edge')
    plot_closed(inner_poly, color='magenta', linewidth=2.0, alpha=0.9, label='Inner edge')

    if centerline is not None:
        ax1.plot(np.r_[centerline[:,0], centerline[0,0]],
                np.r_[centerline[:,1], centerline[0,1]],
                '-', linewidth=1.5, color='red', alpha=0.9,
                solid_capstyle='butt', solid_joinstyle='miter', zorder=3, label='Centerline')

    if start_points:
        sx = [p[0] for p in start_points]
        sy = [p[1] for p in start_points]
        ax1.scatter(sx, sy, s=60, c='lime', edgecolors='k',
                   linewidths=1.0, label='Start')

    ax1.legend(loc='upper right')
    ax1.axis('off')

    # Right subplot: Stylized track image
    try:
        stylized_img = create_stylized_track_image(track_mask)
        stylized_array = np.array(stylized_img)
        ax2.imshow(stylized_array)
        ax2.set_title('Stylized Racetrack')
    except Exception as e:
        print(f"Warning: Could not create stylized image: {e}")
        ax2.imshow(track_mask, cmap="gray", interpolation="nearest")
        ax2.set_title('Stylized Image Failed - Showing Track Mask')

    ax2.axis('off')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python map_utils.py <image_path>")
        print("Example: python map_utils.py maps/map_start3.png")
        sys.exit(1)

    image_path = sys.argv[1]
    visualize_track_lines(image_path)
