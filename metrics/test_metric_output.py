from metric_calculation import *

from pathlib import Path
import json
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# -------------------------
# Find Impact
# -------------------------

def load_impact_time(sticker_json_path, ball_json_path):
    """
    Finds the last sticker frame and returns it as the moment of impact.
    This is because the computer vision algorithm stops tracking the sticker once the ball begins to move.
    
    Returns:
        impact_time: Time of impact
    """
    with open(sticker_json_path, 'r') as f:
        data = json.load(f)

    if not data:
        return None

    # Use last frame minus 3 as the moment of impact
    # This has to do with how many frames we check after impact--should be one more than that
    # If there is only one coordinate, our sticker detection has failed, and we should check our ball
    if len(data) <= 1:
        print("Warning: Sticker detection failed.")
        impact_time = detect_ball_movement(ball_json_path)
        
        if (impact_time is None):
            print("Warning: Ball detection failed.")
            return None
    else:
        impact_idx = len(data) - 1 - 2
        impact_time = data[impact_idx]['time']

    return impact_time

def detect_ball_movement(ball_json_path, threshold=5.0):
    """
    Detects the moment when the ball starts moving by finding the first significant 
    position change in the ball tracking data.
    
    Args:
        ball_json_path: Path to the JSON file containing ball position data
        threshold: Minimum distance (in units) to consider as movement (default 5.0)
    
    Returns:
        float: Time when ball movement is detected, or None if no valid movement found
    """
    with open(ball_json_path, 'r') as f:
        data = json.load(f)
    
    # Check if data is invalid (empty, single entry, or all zeros)
    if not data or len(data) <= 1:
        return None
    
    # Check if this is the invalid case (all zeros)
    if all(d['x'] == 0.0 and d['y'] == 0.0 and d['z'] == 0.0 for d in data):
        return None
    
    # Need at least 2 frames to detect movement
    if len(data) < 2:
        return None
    
    # Look for the first significant movement between consecutive frames
    print("Looking for significant displacement")
    for i in range(len(data) - 1):
        x1, y1, z1 = data[i]['x'], data[i]['y'], data[i]['z']
        x2, y2, z2 = data[i+1]['x'], data[i+1]['y'], data[i+1]['z']
        
        # Calculate 3D distance between consecutive positions
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        
        # If movement exceeds threshold, return the time of the frame (when movement occurred)
        if distance > threshold:
            return data[i]['time']
    
    # No significant movement detected
    return None


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


# -------------------------
# Ball Velocity
# -------------------------
def ball_velocity_components(json_path, time_threshold, apply_filter=True, 
                            window_length=11, poly_order=2, 
                            warn_threshold=0.8, verbose=True,
                            detect_z_anomaly=True):
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
        detect_z_anomaly: If True, remove frames where z increases for z-fitting only
        
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

    # Filter by time threshold
    def after_threshold(d, threshold):
        return round(d['time'], 6) >= round(threshold, 6)

    frames = [d for d in data if after_threshold(d, time_threshold)]
    
    # Detect anomalous z-increasing frames for z-only filtering
    # Even if there are only 3 frames, we should be removing these for z calculation.
    #   If this is the case we can rely on our finite difference fallback later.
    frames_for_z = frames
    z_frames_removed = 0
    if detect_z_anomaly and len(frames) >= 3:
        frames_for_z = remove_z_increasing_tail(frames, verbose)
        z_frames_removed = len(frames) - len(frames_for_z)
    
    print(f'Frames found: {len(frames)} (x/y), {len(frames_for_z)} (z)')

    # Always use as many frames as possible
    available_frames = len(frames)
    available_frames_z = len(frames_for_z)
    
    if available_frames < 3:
        return finite_difference_fallback(frames, verbose)

    # Extract arrays for x and y (using all frames)
    t_vals = np.array([f['time'] for f in frames])
    x_vals = np.array([f['x'] for f in frames])
    y_vals = np.array([f['y'] for f in frames])
    
    # Extract arrays for z (using filtered frames only)
    t_vals_z = np.array([f['time'] for f in frames_for_z])
    z_vals = np.array([f['z'] for f in frames_for_z])

    # Savitzky-Golay filtering for x and y
    if apply_filter and available_frames >= 3:
        max_window = available_frames if available_frames % 2 == 1 else available_frames - 1
        sg_win = min(window_length, max_window)
        if sg_win < 3: sg_win = 3
        sg_poly = min(poly_order, sg_win - 1)
        
        if sg_win > sg_poly:
            x_vals = savgol_filter(x_vals, sg_win, sg_poly)
            y_vals = savgol_filter(y_vals, sg_win, sg_poly)
    else:
        sg_win = None
        sg_poly = None
    
    # Savitzky-Golay filtering for z (separate window based on available z frames)
    sg_win_z = None
    sg_poly_z = None
    if apply_filter and available_frames_z >= 3:
        max_window_z = available_frames_z if available_frames_z % 2 == 1 else available_frames_z - 1
        sg_win_z = min(window_length, max_window_z)
        if sg_win_z < 3: sg_win_z = 3
        sg_poly_z = min(poly_order, sg_win_z - 1)
        
        if sg_win_z > sg_poly_z:
            z_vals = savgol_filter(z_vals, sg_win_z, sg_poly_z)

    # Fit lines for x, y, and z (using appropriate datasets)
    x_rate, x_intercept = fit_line(t_vals, x_vals)
    y_rate, y_intercept = fit_line(t_vals, y_vals)
    
    # Handle z separately - may need fallback if too few z frames
    if available_frames_z >= 3:
        z_rate, z_intercept = fit_line(t_vals_z, z_vals)
        r2_z = calculate_r_squared(t_vals_z, z_vals, z_rate, z_intercept)
        z_pred = z_rate * t_vals_z + z_intercept
        rmse_z = np.sqrt(np.mean((z_vals - z_pred) ** 2))
    elif available_frames_z == 2:
        # Use finite difference for z
        dt = t_vals_z[1] - t_vals_z[0]
        z_rate = (z_vals[1] - z_vals[0]) / dt
        z_intercept = None
        r2_z = None
        rmse_z = None
        if verbose:
            print("Warning: Using finite difference for z (only 2 valid z frames)")
    else:
        z_rate = None
        z_intercept = None
        r2_z = None
        rmse_z = None
        if verbose:
            print("Warning: Insufficient z frames for velocity estimation")

    # R-squared diagnostics for x and y
    r2_x = calculate_r_squared(t_vals, x_vals, x_rate, x_intercept)
    r2_y = calculate_r_squared(t_vals, y_vals, y_rate, y_intercept)

    # RMS error diagnostics for x and y
    x_pred = x_rate * t_vals + x_intercept
    y_pred = y_rate * t_vals + y_intercept
    rmse_x = np.sqrt(np.mean((x_vals - x_pred) ** 2))
    rmse_y = np.sqrt(np.mean((y_vals - y_pred) ** 2))

    # Calculate r2_min only from available metrics
    r2_values = [r2_x, r2_y]
    if r2_z is not None:
        r2_values.append(r2_z)
    
    diagnostics = {
        'r2_x': r2_x,
        'r2_y': r2_y,
        'r2_z': r2_z,
        'r2_min': min(r2_values) if r2_values else None,
        'rmse_x': rmse_x,
        'rmse_y': rmse_y,
        'rmse_z': rmse_z,
        'num_frames': available_frames,
        'num_frames_z': available_frames_z,
        'z_frames_removed': z_frames_removed,
        'filtered': apply_filter and available_frames >= 3,
        'window_length': sg_win,
        'poly_order': sg_poly,
        'window_length_z': sg_win_z,
        'poly_order_z': sg_poly_z
    }

    # Print diagnostics
    if verbose:
        print(f"\n=== Velocity Fit Diagnostics ===")
        print(f"Frames used: {available_frames} (x/y), {available_frames_z} (z)")
        if z_frames_removed > 0:
            print(f"  - Removed {z_frames_removed} z-anomaly frames (z increasing)")
        print(f"Time range: {t_vals[0]:.3f} to {t_vals[-1]:.3f} seconds")
        if diagnostics['filtered']:
            print(f"Filtering applied: Yes")
            print(f"  - x/y: Window={sg_win}, Poly={sg_poly}")
            if sg_win_z:
                print(f"  - z: Window={sg_win_z}, Poly={sg_poly_z}")
        else:
            print(f"Filtering applied: No")
        print(f"\nVelocity components:")
        print(f"  x_rate: {x_rate:+.3f} units/sec")
        print(f"  y_rate: {y_rate:+.3f} units/sec")
        if z_rate is not None:
            print(f"  z_rate: {z_rate:+.3f} units/sec")
        else:
            print(f"  z_rate: N/A (insufficient data)")
        print(f"\nFit quality (R²):")
        print(f"  x: {r2_x:.4f}")
        print(f"  y: {r2_y:.4f}")
        if r2_z is not None:
            print(f"  z: {r2_z:.4f}")
        else:
            print(f"  z: N/A")
        print(f"\nRMS errors:")
        print(f"  x: {rmse_x:.4f} units")
        print(f"  y: {rmse_y:.4f} units")
        if rmse_z is not None:
            print(f"  z: {rmse_z:.4f} units")
        else:
            print(f"  z: N/A")
        
        if diagnostics['r2_min'] is not None and diagnostics['r2_min'] < warn_threshold:
            print(f"\nWARNING: Poor fit quality detected (min R² = {diagnostics['r2_min']:.3f} < {warn_threshold})")
            print("    This suggests either:")
            print("    - Velocity is not constant after the threshold")
            print("    - Data contains significant noise")
            print("    - Time threshold may be incorrectly chosen")
        elif diagnostics['r2_min'] is not None:
            print(f"\nGood fit quality (all R² > {warn_threshold})")

    return x_rate, y_rate, z_rate, diagnostics

# Since we usually get some frames at the end where the ball is partially detected, this can mess up the dz calculation.
# To avoid this we can remove these frames when we calcuate for dz.
def remove_z_increasing_tail(frames, verbose=True, increase_threshold=5.0, min_consecutive=2):
    """
    Remove frames at the end where z starts consistently increasing (object appears to move away).
    Since the object is always moving toward camera, z should decrease overall.
    
    Args:
        frames: List of frames with 'z' values
        verbose: If True, print information about removed frames
        increase_threshold: Minimum z increase (in units) to consider anomalous
        min_consecutive: Number of consecutive increases needed to trigger removal
        
    Returns:
        Filtered list of frames with anomalous tail removed
    """
    if len(frames) < 3:
        return frames
    
    z_vals = np.array([f['z'] for f in frames])
    
    # Calculate the overall trend (should be negative for decreasing z)
    overall_slope = (z_vals[-1] - z_vals[0]) / (len(z_vals) - 1)
    
    # Look for where z starts increasing significantly and consistently
    cutoff_idx = len(frames)
    consecutive_increases = 0
    
    for i in range(len(frames) - 1):
        z_change = z_vals[i + 1] - z_vals[i]
        
        # Check if this is a significant increase (not just noise)
        if z_change > increase_threshold:
            consecutive_increases += 1
            
            # If we see sustained increases, mark this as the cutoff
            if consecutive_increases >= min_consecutive:
                # Backtrack to where the increases started
                cutoff_idx = i - consecutive_increases + 1
                break
        else:
            # Reset counter if we don't see an increase
            consecutive_increases = 0
    
    # Additional check: if the LAST FRAME shows a single large anomalous jump
    if cutoff_idx == len(frames) and len(frames) >= 2:
        last_z_change = z_vals[-1] - z_vals[-2]
        if last_z_change > increase_threshold:
            cutoff_idx = len(frames) - 1
            if verbose:
                print(f"Z-anomaly detected: Last frame shows large z-increase of {last_z_change:.2f} units")
    
    # Additional check: if the last few frames show a strong upward trend
    # compared to the overall downward trend, remove them
    if cutoff_idx == len(frames) and len(frames) >= 5:
        # Check last 3-4 frames for anomalous behavior
        tail_length = min(4, len(frames) // 3)
        tail_slope = (z_vals[-1] - z_vals[-tail_length]) / tail_length
        
        # If tail is going up while overall trend is down, it's anomalous
        if tail_slope > 0 and overall_slope < 0 and abs(tail_slope) > abs(overall_slope):
            cutoff_idx = len(frames) - tail_length
    
    if cutoff_idx < len(frames):
        removed_count = len(frames) - cutoff_idx
        if verbose:
            print(f"Z-anomaly detected at frame {cutoff_idx}: Will use only first {cutoff_idx} frames for z-fitting")
            print(f"  Removed {removed_count} frame(s) where z increased anomalously")
        return frames[:cutoff_idx]
    
    return frames


# Worst-case scenario: If we only have 2 frames, we must use finite difference
def finite_difference_fallback(frames, verbose=True):
    """
    Calculate velocity using finite differences when fewer than 3 frames available.
    
    Args:
        frames: List of 1-2 frames with 'time', 'x', 'y', 'z' keys
        verbose: If True, print diagnostic information
        
    Returns:
        tuple: (x_rate, y_rate, z_rate, diagnostics)
    """
    if len(frames) < 2:
        raise ValueError("Need at least 2 frames for finite difference velocity estimation.")
    
    # Use last two frames for velocity estimate
    dt = frames[-1]['time'] - frames[-2]['time']
    if dt == 0:
        raise ValueError("Last two frames have identical timestamps.")
    
    x_rate = (frames[-1]['x'] - frames[-2]['x']) / dt
    y_rate = (frames[-1]['y'] - frames[-2]['y']) / dt
    z_rate = (frames[-1]['z'] - frames[-2]['z']) / dt
    
    diagnostics = {
        'r2_x': None,
        'r2_y': None,
        'r2_z': None,
        'r2_min': None,
        'rmse_x': None,
        'rmse_y': None,
        'rmse_z': None,
        'num_frames': len(frames),
        'filtered': False,
        'window_length': None,
        'poly_order': None,
        'method': 'finite_difference'
    }
    
    if verbose:
        print(f"\n=== Velocity Finite Difference Estimation ===")
        print(f"Frames used: {len(frames)} (insufficient for fitting)")
        print(f"Method: Simple finite difference between last two frames")
        print(f"\nVelocity components:")
        print(f"  x_rate: {x_rate:+.3f} units/sec")
        print(f"  y_rate: {y_rate:+.3f} units/sec")
        print(f"  z_rate: {z_rate:+.3f} units/sec")
        print(f"\nNote: No fit quality metrics available for finite difference method.")
    
    return x_rate, y_rate, z_rate, diagnostics


# -------------------------
# Club Velocity Calculation
# -------------------------
# Polyorder=2 gives a good balance of smoothness, adaptability to acceleration/deceleration, 
#   and resistance to overfitting random noise for real-world motion estimation from short, 
#   noisy time series.
# Note that Savitzky-Golay is meant for local smoothing, not global trend fitting.
#   This is why we have a maximum window size of 7-13
def savgol_velocity(json_path, polyorder=2, max_window=13):
    """
    Estimate velocity components (x, y, z) at the last time point in a JSON file
    using Savitzky-Golay smoothing/derivative, or finite difference if only two frames.
    Ensures dz is always negative (object moving toward camera).

    Args:
        json_path (str): Path to JSON file with position data (time, x, y, z).
        polyorder (int): Polynomial order for S-G filter (default 2).
        max_window (int): Maximum window length (default 13). Should be odd.

    Returns:
        tuple: (x_vel, y_vel, z_vel) at the last time point in units/sec.

    Raises:
        ValueError: If fewer than 2 frames in the file.
    """
    # Load data from JSON file
    with open(json_path, 'r') as f:
        frames = json.load(f)
    
    N = len(frames)
    
    # Finite difference fallback for exactly 2 frames
    if N == 2:
        dt = frames[1]['time'] - frames[0]['time']
        if dt == 0:
            raise ValueError("Timestamps of the two frames are identical.")
        x_vel = (frames[1]['x'] - frames[0]['x']) / dt
        y_vel = (frames[1]['y'] - frames[0]['y']) / dt
        z_vel = (frames[1]['z'] - frames[0]['z']) / dt
        return x_vel, y_vel, z_vel
    
    if N < 3:
        print(f"Need at least 3 frames for Savitzky-Golay velocity. Got {N}.")
        return None, None, None

    # Extract arrays and convert time to float (in case stored as string)
    t_vals = np.array([f['time'] for f in frames])
    x_vals = np.array([f['x'] for f in frames])
    y_vals = np.array([f['y'] for f in frames])
    z_vals = np.array([f['z'] for f in frames])

    # Sort by time (in case not already sorted)
    idx = np.argsort(t_vals)
    t_vals = t_vals[idx]
    x_vals = x_vals[idx]
    y_vals = y_vals[idx]
    z_vals = z_vals[idx]

    # Iteratively remove tail frames until dz becomes negative
    max_removal = min(3, N - 3)  # Remove at most 3 frames, keep at least 3
    frames_removed = 0
    
    for attempt in range(max_removal + 1):
        current_N = N - frames_removed
        
        # Choose window length
        window_length = min(max_window, current_N)
        if window_length % 2 == 0:
            window_length -= 1
        if window_length < 3:
            window_length = 3
        poly = min(polyorder, window_length - 1)

        # Median time step for derivative scaling
        dt = np.median(np.diff(t_vals[:current_N]))
        if dt == 0:
            raise ValueError("Time values must be distinct for velocity estimation.")

        # Compute Savitzky-Golay derivatives
        x_deriv = savgol_filter(x_vals[:current_N], window_length, poly, deriv=1, delta=dt)
        y_deriv = savgol_filter(y_vals[:current_N], window_length, poly, deriv=1, delta=dt)
        z_deriv = savgol_filter(z_vals[:current_N], window_length, poly, deriv=1, delta=dt)

        # Check if dz is negative (as expected)
        if z_deriv[-1] < 0:
            # Good! Return velocities
            return x_deriv[-1], y_deriv[-1], z_deriv[-1]
        else:
            # dz is positive - remove one more frame from tail and retry
            frames_removed += 1
            if frames_removed <= max_removal:
                print(f"Warning: dz positive ({z_deriv[-1]:.2f}), removing 1 tail frame (attempt {attempt+1})")

    # If still positive after max removals, force it negative
    print(f"Warning: Could not achieve negative dz after removing {frames_removed} frames. Forcing sign.")
    return x_deriv[-1], y_deriv[-1], -abs(z_deriv[-1])


# -------------------------
# Find Metrics
# -------------------------

# Find the metrics using ball data
def metrics_with_ball(ball_dx, ball_dy, ball_dz, marker_dx, marker_dy, marker_dz) -> dict:
    # ---------------------------------
    # Error checking
    # - If club or ball was not detected, return None respectively
    # - If any angles are very extreme, return None
    # - This is worst-case scenario, and we don't want it to happen often!
    # ---------------------------------
    # If club is undetected
    if (marker_dx is None) or (marker_dy is None) or (marker_dz is None):
        print("Warning: Club not detected, using ball only")
        swing_path = None
        attack_angle = None
    else:
        swing_path = horizontal_movement_angle_from_rates(marker_dx, marker_dz)
        attack_angle = vertical_movement_angle_from_rates(marker_dy, marker_dz)
        
        # Check for extreme return values
        # If one angle is extreme, both cannot be trusted
        if (abs(swing_path) > 40):
            print(f"Extreme swing path {swing_path}; using default of 0.00")
            swing_path = None
            attack_angle = None
        elif (abs(attack_angle) > 50):
            print(f"Extreme attack angle {attack_angle}; using default of 0.00")
            attack_angle = None
            swing_path = None
        
        club_speed = cmps_to_speed_kmh(marker_dx, marker_dy, marker_dz)
        print(f'Club speed: {club_speed:.2f}\n')
        
    # If ball is undetected
    if (ball_dx is None) or (ball_dy is None) or (ball_dz is None):
        print("Warning: Ball not detected, using club only")
        side_angle = None
    else:
        side_angle = horizontal_movement_angle_from_rates(ball_dx, ball_dz)
        
        # Check for extreme return values
        if (abs(side_angle) > 50):
            print(f"Extreme side angle {side_angle}; using default of 0.00")
            side_angle = None
        
        ball_speed = cmps_to_speed_kmh(ball_dx, ball_dy, ball_dz)
        print(f'Ball speed: {ball_speed:.2f}\n')

    # If ball or club was not detected
    if (side_angle is None) or (swing_path is None):
        face_angle = None
        face_to_path = None
    else:
        face_angle = face_angle_calc(swing_path, side_angle)
        face_to_path = face_angle - swing_path

    return {
        "face_angle": face_angle,
        "swing_path": swing_path,
        "attack_angle": attack_angle,
        "side_angle": side_angle,
        "face_to_path": face_to_path
    }


def return_metrics() -> dict:

    # ---------------------------------
    # Filepaths
    # ---------------------------------
    # Coordinate source paths
    src_coords_path = Path("~/Documents/webcamGolf").expanduser()
    src_coords = str(src_coords_path) + "/"
    # ball_coords_path = os.path.join(src_coords, 'ball_coords.json')  # PIPELINE
    # sticker_coords_path = os.path.join(src_coords, 'sticker_coords.json')  # PIPELINE
    ball_coords_path = "../ball_coords.json"  # STANDALONE
    sticker_coords_path = "../sticker_coords.json"  # STANDALONE

    # ---------------------------------
    # Find impact time
    # - If impact time is Nonetype, our detection has failed because we do not have a ball or sticker
    # --------------------------------- 
    impact_time = load_impact_time(sticker_coords_path, ball_coords_path)
    print(f"Impact time: {impact_time}")

    if (impact_time is None):
        return {
            "face_angle": None,
            "swing_path": None,
            "attack_angle": None,
            "side_angle": None,
            "face_to_path": None
    }

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
    if diag['r2_min'] is not None:
        print(f"\nMinimum R²: {diag['r2_min']:.4f}")
    else:
        print("\nMinimum R²: Not available (finite difference method used)")

    print(f"Ball dx: {ball_dx}, Ball dy: {ball_dy}, Ball dz: {ball_dz}")
    marker_dx, marker_dy, marker_dz = savgol_velocity(sticker_coords_path)
    print(f"At time {impact_time}, Marker dx: {marker_dx}, Marker dy: {marker_dy}, Marker dz: {marker_dz}")

    # ---------------------------------
    # Calculate Metrics
    # ---------------------------------
    metrics = metrics_with_ball(ball_dx, ball_dy, ball_dz, marker_dx, marker_dy, marker_dz)


    # ---------------------------------
    # Print & Return Metrics
    # ---------------------------------
    print(f'\nSwing path: {metrics["swing_path"] if metrics["swing_path"] is None else "%.2f" % metrics["swing_path"]}')
    print(f'Face angle: {metrics["face_angle"] if metrics["face_angle"] is None else "%.2f" % metrics["face_angle"]}')
    print(f'Side angle: {metrics["side_angle"] if metrics["side_angle"] is None else "%.2f" % metrics["side_angle"]}')
    print(f'Attack angle: {metrics["attack_angle"] if metrics["attack_angle"] is None else "%.2f" % metrics["attack_angle"]}')
    print(f'Face-to-path: {metrics["face_to_path"] if metrics["face_to_path"] is None else "%.2f" % metrics["face_to_path"]}')
    print('\n')

    return metrics

