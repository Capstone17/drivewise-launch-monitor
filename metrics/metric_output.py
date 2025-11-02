# NOTES ------------------------
# - Error checks should be added for extreme angles
# - For first frame of movement, more reliable methods could be found
# ------------------------------


from metric_calculation import *

from pathlib import Path
import json
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# TODO: Remove this function completely
def load_ball_movement_window(json_path):
    """
    Finds the first instance of major movement and returns all frames from impact onward.

    Returns:
    tuple: (window, impact_idx_in_window, pre_frame, post_frame)
    - pre_frame is the impact frame
    - post_frame is the frame directly after impact
    """


    # x and y are much more accurate than z, hence the smaller movement thresholds
    x_threshold = 0.1
    y_threshold = 0.1
    z_threshold = 8.0


    with open(json_path, 'r') as f:
        data = json.load(f)


    if not data:
        return [], None, None, None, None # No data at all


    for i in range(len(data) - 1):
        frame1 = data[i]
        frame2 = data[i + 1]


        dx = abs(frame2['x'] - frame1['x'])
        dy = abs(frame2['y'] - frame1['y'])
        dz = abs(frame2['z'] - frame1['z'])


        if dx > x_threshold or dy > y_threshold or dz > z_threshold:
            # Window is everything from impact onward
            window = data[i:]
            impact_idx_in_window = 0 # Impact is first frame in the window
            pre_frame = window[0]
            post_frame = window[1] if len(window) > 1 else None
            last_frame = window[-1]
            return window, impact_idx_in_window, pre_frame, post_frame, last_frame


    return [], None, None, None, None # No movement found


def load_impact_time(json_path):
    """
    Finds the last sticker frame and returns it as the moment of impact.
    This is because the computer vision algorithm stops tracking the sticker once the ball begins to move.
    
    Returns:
        impact_time: Time of impact
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not data:
        return None

    # Use last frame as the moment of impact
    impact_idx = len(data) - 1
    impact_time = data[impact_idx]['time']

    return impact_time


# Load the marker pose immediately before and immediately after impact
def load_marker_poses_with_impact_time(json_path, t_target, time_window=0.3):
    """
    Load marker poses from JSON and:
      - return poses within ±time_window around t_target
      - return the closest pose before impact
      - return the closest pose after impact

    Args:
        json_path (str): Path to JSON file containing pose data.
        t_target (float): Target time in seconds.
        time_window (float): Time window around t_target for filtering.

    Returns:
        tuple: (filtered_poses, pose_before, pose_after)
            filtered_poses (list): Poses within ±time_window of t_target.
            pose_before (dict or None): Pose closest before t_target in full dataset.
            pose_after (dict or None): Pose closest after t_target in full dataset.
    """
    with open(json_path, 'r') as f:
        poses = json.load(f)

    if not poses:
        return [], None, None

    # Sort by time
    poses.sort(key=lambda p: p['time'])

    # Filtered poses: same as original function
    filtered_poses = [p for p in poses if abs(p['time'] - t_target) <= time_window]

    # Find closest before and after in full dataset
    pose_before = None
    pose_after = None

    for pose in poses:
        if pose['time'] < t_target:
            pose_before = pose
        elif pose['time'] > t_target:
            pose_after = pose
            break
        # If exactly at t_target, skip for before/after calculation

    return filtered_poses, pose_before, pose_after

# Load the marker poses without knowing the moment of impact
def load_marker_poses_without_impact_time(json_path, time_jump_threshold=0.1):
    """
    Returns:
        window (list): The last window of poses after the final time jump,
                       or the entire sequence if there are no jumps.
        middle_pose (dict): Pose closest to the middle of the window.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not data or len(data) < 2:
        return [], None  # Not enough poses to analyze

    last_jump_index = -1

    # Track the last jump index
    for i in range(len(data) - 1):
        dt = data[i + 1]["time"] - data[i]["time"]
        if dt > time_jump_threshold:
            last_jump_index = i + 1

    # Slice from last jump onward (or full sequence if no jump)
    if last_jump_index != -1:
        window = data[last_jump_index:]
    else:
        window = data

    # Find middle pose
    if window:
        middle_index = len(window) // 2
        middle_pose = window[middle_index]
    else:
        middle_pose = None

    return window, middle_pose


# -------------------------
# Ball Velocity Helpers
# -------------------------

def fit_line(t_values, y_values):
    """
    Fit a straight line to (t, y) data using least squares.
    Returns slope (m) and intercept (b).
    """
    t = np.array(t_values, dtype=float)
    y = np.array(y_values, dtype=float)
    A = np.vstack([t, np.ones(len(t))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, b


def calculate_r_squared(t_values, y_values, slope, intercept):
    """
    Calculate R-squared (coefficient of determination) for the fit.
    R² = 1 means perfect fit, R² = 0 means the fit is no better than the mean.
    """
    t = np.array(t_values, dtype=float)
    y = np.array(y_values, dtype=float)
    
    y_pred = slope * t + intercept
    ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
    
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    return r2


def ball_velocity_components(json_path, time_threshold, apply_filter=True, 
                            window_length=11, poly_order=2, 
                            warn_threshold=0.8, verbose=True):
    """
    Compute velocity components from position data after a specified time,
    with optional Savitzky-Golay filtering and fit quality diagnostics.
    
    Args:
        json_path: Path to JSON file with position data (time, x, y, z)
        time_threshold: Only consider frames with time > this value
        apply_filter: If True, apply Savitzky-Golay filter to smooth data
        window_length: Window length for Savitzky-Golay filter (must be odd)
        poly_order: Polynomial order for Savitzky-Golay filter (typically 2-3)
        warn_threshold: R² threshold below which to warn about poor fit quality
        verbose: If True, print fit quality diagnostics
        
    Returns:
        tuple: (x_rate, y_rate, z_rate, diagnostics)
            - x_rate, y_rate, z_rate: velocity components in units/second
            - diagnostics: dict with R² values and other fit quality metrics
        
    Raises:
        ValueError: If insufficient frames available after filtering
    """

    # Load data from JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Filter
    def after_threshold(d, threshold):
        return round(d['time'], 6) > round(threshold, 6)

    frames = [d for d in data if after_threshold(d, time_threshold)]
    print('Filtered time values:', [f['time'] for f in frames])
    print('Frames found:', len(frames))

    # Always use as many frames as possible
    available_frames = len(frames)
    if available_frames < 3:
        raise ValueError("Need at least 3 frames after filtering for fit.")

    t_vals = np.array([f['time'] for f in frames])
    x_vals = np.array([f['x'] for f in frames])
    y_vals = np.array([f['y'] for f in frames])
    z_vals = np.array([f['z'] for f in frames])

    # Savitzky-Golay: Use largest possible odd window length <= available_frames
    if apply_filter and available_frames >= 3:
        max_window = available_frames if available_frames % 2 == 1 else available_frames - 1
        sg_win = min(window_length, max_window)
        if sg_win < 3: sg_win = 3
        sg_poly = min(poly_order, sg_win - 1)
        # Only filter if window makes sense (>= 3 and poly < win)
        if sg_win > sg_poly:
            x_vals = savgol_filter(x_vals, sg_win, sg_poly)
            y_vals = savgol_filter(y_vals, sg_win, sg_poly)
            z_vals = savgol_filter(z_vals, sg_win, sg_poly)
    else:
        sg_win = None
        sg_poly = None

    # Fit lines for x, y, and z
    x_rate, x_intercept = fit_line(t_vals, x_vals)
    y_rate, y_intercept = fit_line(t_vals, y_vals)
    z_rate, z_intercept = fit_line(t_vals, z_vals)

    # R-squared diagnostics
    r2_x = calculate_r_squared(t_vals, x_vals, x_rate, x_intercept)
    r2_y = calculate_r_squared(t_vals, y_vals, y_rate, y_intercept)
    r2_z = calculate_r_squared(t_vals, z_vals, z_rate, z_intercept)

    # RMS error diagnostics
    x_pred = x_rate * t_vals + x_intercept
    y_pred = y_rate * t_vals + y_intercept
    z_pred = z_rate * t_vals + z_intercept
    rmse_x = np.sqrt(np.mean((x_vals - x_pred) ** 2))
    rmse_y = np.sqrt(np.mean((y_vals - y_pred) ** 2))
    rmse_z = np.sqrt(np.mean((z_vals - z_pred) ** 2))

    diagnostics = {
        'r2_x': r2_x,
        'r2_y': r2_y,
        'r2_z': r2_z,
        'r2_min': min(r2_x, r2_y, r2_z),
        'rmse_x': rmse_x,
        'rmse_y': rmse_y,
        'rmse_z': rmse_z,
        'num_frames': available_frames,
        'filtered': apply_filter and available_frames >= 3,
        'window_length': sg_win,
        'poly_order': sg_poly
    }

    # Print diagnostics
    if verbose:
        print(f"\n=== Velocity Fit Diagnostics ===")
        print(f"Frames used: {available_frames}")
        print(f"Time range: {t_vals[0]:.3f} to {t_vals[-1]:.3f} seconds")
        if diagnostics['filtered']:
            print(f"Filtering applied: Yes (Window length: {sg_win}, Polynomial order: {sg_poly})")
        else:
            print(f"Filtering applied: No (insufficient frames for filter or filter disabled)")
        print(f"\nVelocity components:")
        print(f"  x_rate: {x_rate:+.3f} units/sec")
        print(f"  y_rate: {y_rate:+.3f} units/sec")
        print(f"  z_rate: {z_rate:+.3f} units/sec")
        print(f"\nFit quality (R²):")
        print(f"  x: {r2_x:.4f}")
        print(f"  y: {r2_y:.4f}")
        print(f"  z: {r2_z:.4f}")
        print(f"\nRMS errors:")
        print(f"  x: {rmse_x:.4f} units")
        print(f"  y: {rmse_y:.4f} units")
        print(f"  z: {rmse_z:.4f} units")
        if diagnostics['r2_min'] < warn_threshold:
            print(f"\nWARNING: Poor fit quality detected (min R² = {diagnostics['r2_min']:.3f} < {warn_threshold})")
            print("    This suggests either:")
            print("    - Velocity is not constant after the threshold")
            print("    - Data contains significant noise")
            print("    - Time threshold may be incorrectly chosen")
        else:
            print(f"\nGood fit quality (all R² > {warn_threshold})")

    return x_rate, y_rate, z_rate, diagnostics


def finite_diff_velocity(frames, t_target=None):
    """
    Calculate velocity components (x, y, z) at t_target using finite differences.
    Uses the frame at or just before t_target and the next frame.

    Args:
        frames (list of dict): Each dict with keys 'time', 'x', 'y', 'z'.
        t_target (float, optional): Time to compute velocity at. Defaults to middle frame.

    Returns:
        tuple: (x_vel, y_vel, z_vel) in units per second (cm/s if positions are cm).
    """
    if len(frames) < 2:
        raise ValueError("Need at least 2 frames for finite difference velocity.")

    t_vals = [f['time'] for f in frames]
    if t_target is None:
        t_target = t_vals[len(t_vals) // 2]

    # Find index of closest frame with time <= t_target
    idx = 0
    for i, t in enumerate(t_vals):
        if t <= t_target:
            idx = i
        else:
            break

    if idx >= len(frames) - 1:
        idx = len(frames) - 2  # Ensure next frame exists

    dt = t_vals[idx + 1] - t_vals[idx]
    if dt == 0:
        raise ValueError("Two frames have identical timestamps.")

    x_vel = (frames[idx + 1]['x'] - frames[idx]['x']) / dt
    y_vel = (frames[idx + 1]['y'] - frames[idx]['y']) / dt
    z_vel = (frames[idx + 1]['z'] - frames[idx]['z']) / dt

    return x_vel, y_vel, z_vel


# Find the metrics using ball data
def metrics_with_ball(ball_dx, ball_dy, ball_dz, marker_dx, marker_dy, marker_dz) -> dict:

    swing_path = horizontal_movement_angle_from_rates(marker_dx, marker_dz)
    side_angle = horizontal_movement_angle_from_rates(ball_dx, ball_dy)
    face_angle = face_angle_calc(swing_path, side_angle)
    attack_angle = vertical_movement_angle_from_rates(marker_dy, marker_dz)
    face_to_path = face_angle - swing_path

    # Speeds
    club_speed = cmps_to_speed_kmh(marker_dx, marker_dy, marker_dz)
    print(f'Club speed: {club_speed:.2f}\n')

    ball_speed = cmps_to_speed_kmh(ball_dx, ball_dy, ball_dz)
    print(f'Ball speed: {ball_speed:.2f}\n')

    # ---------------------------------
    # Error checking
    # - If any angles are very extreme, set them to zero
    # - This is worst-case scenario, and we don't want it to happen often!
    # ---------------------------------
    
    if (abs(swing_path) > 25):
        print(f"Extreme swing path {swing_path}; using default of 0.00")
        swing_path = 0.00
    if (abs(side_angle) > 25):
        print(f"Extreme side angle {side_angle}; using default of 0.00")
        side_angle = 0.00
    if (abs(face_angle) > 25):
        print(f"Extreme face angle {face_angle}; using default of 0.00")
        face_angle = 0.00
    if (abs(attack_angle) > 25):
        print(f"Extreme attack angle {attack_angle}; using default of 0.00")
        attack_angle = 0.00
    if (abs(face_to_path) > 25):
        print(f"Extreme face-to-path {face_to_path}; using default of 0.00")
        face_to_path = 0.00

    return {
        "face_angle": face_angle,
        "swing_path": swing_path,
        "attack_angle": attack_angle,
        "side_angle": side_angle,
        "face_to_path": face_to_path
    }


# Find the metrics without ball data
def metrics_without_ball(marker_dx, marker_dy, marker_dz, marker_yaw_at_impact) -> dict:

    swing_path = horizontal_movement_angle_from_rates(marker_dx, marker_dz)
    face_angle = marker_yaw_at_impact
    attack_angle = vertical_movement_angle_from_rates(marker_dy, marker_dz)
    side_angle = side_angle_without_ball(swing_path, face_angle)
    face_to_path = face_angle - swing_path

    # Speeds
    club_speed = cmps_to_speed_kmh(marker_dx, marker_dy, marker_dz)
    print(f'Club speed: {club_speed:.2f}\n')

    return {
        "face_angle": face_angle,
        "swing_path": swing_path,
        "attack_angle": attack_angle,
        "side_angle": side_angle,
        "face_to_path": face_to_path
    }


# For testing
def plot_positions(frames):
    """
    Plot x, y, and z positions over time.

    Args:
        frames (list of dict): Each dict with keys 'time', 'x', 'y', 'z'.
    """
    times = [f['time'] for f in frames]
    xs = [f['x'] for f in frames]
    ys = [f['y'] for f in frames]
    zs = [f['z'] for f in frames]

    plt.figure(figsize=(10, 6))
    plt.plot(times, xs, label='x position', marker='o')
    plt.plot(times, ys, label='y position', marker='o')
    plt.plot(times, zs, label='z position', marker='o')

    plt.xlabel('Time (s)')
    plt.ylabel('Position (cm)')
    plt.title('Marker Position Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def return_metrics() -> dict:

    # ---------------------------------
    # Filepaths
    # ---------------------------------
    # Coordinate source paths
    src_coords_path = Path("~/Documents/webcamGolf").expanduser()
    src_coords = str(src_coords_path) + "/"
    # CHANGE: ball_coords_path = os.path.join(src_coords, 'ball_coords.json')
    # CHANGE: sticker_coords_path = os.path.join(src_coords, 'sticker_coords.json')
    ball_coords_path = "../ball_coords.json"
    sticker_coords_path = "../sticker_coords.json"

    # ---------------------------------
    # Find impact time
    # --------------------------------- 
    ball_window, ball_impact_idx, ball_pre_frame, ball_post_frame, ball_last_frame = load_ball_movement_window(ball_coords_path)  # Find moment of impact and its surrounding frames
    impact_time = load_impact_time(sticker_coords_path)
    print(f"Impact time: {impact_time}")

    # TODO: Remove section completely
    # ---------------------------------
    # Load Window & Print
    # ---------------------------------
    marker_window, marker_frame_before_impact, marker_frame_after_impact = load_marker_poses_with_impact_time(sticker_coords_path, ball_pre_frame['time'])

    # Compare the absolute time differences to find the frame closest to impact
    if marker_frame_before_impact and marker_frame_after_impact:

        # Compare absolute time differences
        if abs(marker_frame_before_impact["time"] - ball_pre_frame["time"]) <= abs(marker_frame_after_impact["time"] - ball_pre_frame["time"]):
            marker_target_time = marker_frame_before_impact["time"]
        else:
            marker_target_time = marker_frame_after_impact["time"]
    elif marker_frame_before_impact:
        marker_target_time = marker_frame_before_impact["time"]
    elif marker_frame_after_impact:
        marker_target_time = marker_frame_after_impact["time"]

    for idx, frame in enumerate(ball_window):
        print(f"Frame {idx}: time={frame['time']}, x={frame['x']:.3f}, y={frame['y']:.3f}, z={frame['z']:.3f}")

    # Print marker data
    print("\nMarker Frames:")
    for idx, frame in enumerate(marker_window):
        print(f"Frame {idx}: time={frame['time']:.3f}, x={frame['x']:.3f}, y={frame['y']:.3f}, z={frame['z']:.3f}")

    # ---------------------------------
    # Velocity Approximation
    # ---------------------------------

    # With filtering (recommended)
    ball_dx, ball_dy, ball_dz, diag = ball_velocity_components(
        ball_coords_path, 
        time_threshold=impact_time,
        apply_filter=True,
        window_length=11,
        poly_order=2,
        verbose=True
    )
    
    # Without filtering (for comparison)
    # x_vel, y_vel, z_vel, diag = ball_velocity_components(
    #     'ball_data.json', 
    #     time_threshold=1.462,
    #     apply_filter=False,
    #     verbose=True
    # )

    # Access diagnostics
    print(f"\nMinimum R²: {diag['r2_min']:.4f}")

    print(f"Ball dx: {ball_dx}, Ball dy: {ball_dy}, Ball dz: {ball_dz}")
    marker_dx, marker_dy, marker_dz = finite_diff_velocity(marker_window, t_target=marker_target_time)
    print(f"At time {marker_target_time}, Marker dx: {marker_dx}, Marker dy: {marker_dy}, Marker dz: {marker_dz}")

    # Calculate the metrics
    metrics = metrics_with_ball(ball_dx, ball_dy, ball_dz, marker_dx, marker_dy, marker_dz)
   

    # ---------------------------------
    # Print & Return Metrics
    # ---------------------------------
    print(f'Swing path: {metrics["swing_path"]:.2f}\n')
    print(f'Face angle: {metrics["face_angle"]:.2f}\n')
    print(f'Side angle: {metrics["side_angle"]:.2f}\n')
    print(f'Attack angle: {metrics["attack_angle"]:.2f}\n')
    print(f'Face-to-path: {metrics["face_to_path"]:.2f}\n')

    return metrics

