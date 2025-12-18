from metric_calculation import *

from pathlib import Path
import json
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


FACTOR_BALL_X_Y_DELTA = 0.4  # Hardcoded factor to multiply dx and dy, since there is not enough time to fix it in the ball detector code
FACTOR_CLUB_X_Y_DELTA = 0.2  # Hardcoded factor to multiply dx and dy, since there is not enough time to fix it in the ball detector code


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
    """

    # Load data from JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Filter by time threshold
    def after_threshold(d, threshold):
        return round(d['time'], 6) >= round(threshold, 6)

    frames = [d for d in data if after_threshold(d, time_threshold)]

    # Detect anomalous z-increasing frames; this now drives x,y,t as well
    frames_for_fit = frames
    z_frames_removed = 0
    if detect_z_anomaly and len(frames) >= 3:
        result = remove_z_increasing_tail(frames, verbose)
        if result is None:
            # Too many frames removed, use finite difference fallback on full frames
            return finite_difference_fallback(frames, verbose)
        frames_for_fit = result
        z_frames_removed = len(frames) - len(frames_for_fit)

    print(f'Frames found: {len(frames)} (raw), {len(frames_for_fit)} (used for x/y/z)')

    available_frames = len(frames_for_fit)
    if available_frames < 3:
        # Not enough clean frames; fall back to simple finite difference on full frames
        return finite_difference_fallback(frames_for_fit, verbose)

    # Extract arrays for t, x, y, z using the same filtered frames
    t_vals = np.array([f['time'] for f in frames_for_fit])
    x_vals = np.array([f['x'] for f in frames_for_fit])
    y_vals = np.array([f['y'] for f in frames_for_fit])
    z_vals = np.array([f['z'] for f in frames_for_fit])

    # Savitzky-Golay filtering for x, y, z on the same frame set
    sg_win = sg_poly = None
    if apply_filter and available_frames >= 3:
        max_window = available_frames if available_frames % 2 == 1 else available_frames - 1
        sg_win = min(window_length, max_window)
        if sg_win < 3:
            sg_win = 3
        sg_poly = min(poly_order, sg_win - 1)

        if sg_win > sg_poly:
            x_vals = savgol_filter(x_vals, sg_win, sg_poly)
            y_vals = savgol_filter(y_vals, sg_win, sg_poly)
            z_vals = savgol_filter(z_vals, sg_win, sg_poly)

    # Fit lines for x, y
    x_rate, x_intercept = fit_line(t_vals, x_vals)
    y_rate, y_intercept = fit_line(t_vals, y_vals)

    # Handle z separately - may need finite-difference fallback
    if available_frames >= 3:
        z_rate, z_intercept = fit_line(t_vals, z_vals)
        r2_z = calculate_r_squared(t_vals, z_vals, z_rate, z_intercept)
        z_pred = z_rate * t_vals + z_intercept
        rmse_z = np.sqrt(np.mean((z_vals - z_pred) ** 2))
    elif available_frames == 2:
        # Use finite difference for z over last two decreasing frames in frames_for_fit
        z_rate, z_intercept, r2_z, rmse_z = finite_difference_z(frames_for_fit, verbose)
    else:
        z_rate = z_intercept = r2_z = rmse_z = None
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

    # r2_min over available components
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
        'num_frames': len(frames),
        'num_frames_used': available_frames,
        'z_frames_removed': z_frames_removed,
        'filtered': apply_filter and available_frames >= 3,
        'window_length': sg_win,
        'poly_order': sg_poly,
    }

    if verbose:
        print(f"\n=== Velocity Fit Diagnostics ===")
        print(f"Frames found: {len(frames)}, used: {available_frames}")
        if z_frames_removed > 0:
            print(f"  - Removed {z_frames_removed} anomalous frame(s) (x,y,z)")
        print(f"Time range (used): {t_vals[0]:.3f} to {t_vals[-1]:.3f} seconds")
        if diagnostics['filtered']:
            print(f"Filtering applied: Yes (Window={sg_win}, Poly={sg_poly})")
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
            print("  z: N/A")
        print(f"\nRMS errors:")
        print(f"  x: {rmse_x:.4f} units")
        print(f"  y: {rmse_y:.4f} units")
        if rmse_z is not None:
            print(f"  z: {rmse_z:.4f} units")
        else:
            print("  z: N/A")

        if diagnostics['r2_min'] is not None and diagnostics['r2_min'] < warn_threshold:
            print(f"\nWARNING: Poor fit quality detected (min R² = {diagnostics['r2_min']:.3f} < {warn_threshold})")
        elif diagnostics['r2_min'] is not None:
            print(f"\nGood fit quality (all R² > {warn_threshold})")


    # Lessen the extremity of the x and y rate of change
    x_rate = x_rate * FACTOR_BALL_X_Y_DELTA
    y_rate = y_rate * FACTOR_BALL_X_Y_DELTA

    return x_rate, y_rate, z_rate, diagnostics


def finite_difference_z(frames, verbose=True):
    """
    Calculate z velocity using finite differences between the last two frames
    that show decreasing z (normal behavior for object moving toward camera).
    
    Args:
        frames: List of frames with 'time' and 'z' keys
        verbose: If True, print diagnostic information
        
    Returns:
        tuple: (z_rate, z_intercept, r2_z, rmse_z)
               z_intercept, r2_z, and rmse_z are None for finite difference
    """
    if len(frames) < 2:
        if verbose:
            print("Warning: Need at least 2 frames for z finite difference")
        return None, None, None, None
    
    # Find last two frames with decreasing z
    last_decreasing_idx = None
    second_last_decreasing_idx = None
    
    for i in range(len(frames) - 1, 0, -1):
        z_change = frames[i]['z'] - frames[i-1]['z']
        
        # Found a decreasing pair
        if z_change < 0:
            if last_decreasing_idx is None:
                last_decreasing_idx = i
                second_last_decreasing_idx = i - 1
                break
    
    # If no decreasing pair found, just use last two frames
    if last_decreasing_idx is None:
        if verbose:
            print("Warning: No decreasing z frames found, using last 2 frames for z finite difference")
        last_decreasing_idx = len(frames) - 1
        second_last_decreasing_idx = len(frames) - 2
    
    # Calculate finite difference
    dt = frames[last_decreasing_idx]['time'] - frames[second_last_decreasing_idx]['time']
    if dt == 0:
        if verbose:
            print("Warning: Selected frames have identical timestamps")
        return None, None, None, None
    
    z_rate = (frames[last_decreasing_idx]['z'] - frames[second_last_decreasing_idx]['z']) / dt
    
    if verbose:
        print(f"Using finite difference for z between frames {second_last_decreasing_idx} and {last_decreasing_idx}")
        print(f"  Frame {second_last_decreasing_idx}: t={frames[second_last_decreasing_idx]['time']:.3f}, z={frames[second_last_decreasing_idx]['z']:.2f}")
        print(f"  Frame {last_decreasing_idx}: t={frames[last_decreasing_idx]['time']:.3f}, z={frames[last_decreasing_idx]['z']:.2f}")
        print(f"  z_rate: {z_rate:+.3f} units/sec")
    
    return z_rate, None, None, None


def remove_z_increasing_tail(frames, verbose=True, min_frames_required=3):
    """
    Remove frames at the end where z shows large anomalous increases.
    Since the object is always moving toward camera, large z increases indicate detection errors.
    Scans backwards to find the EARLIEST anomalous frame.
    
    Args:
        frames: List of frames with 'z' values
        verbose: If True, print information about removed frames
        min_frames_required: Minimum frames to keep (to allow fitting). If we would
                           remove too many, return None to trigger finite difference fallback.
        
    Returns:
        Filtered list of frames with anomalous tail removed, or None if too many removed
    """
    if len(frames) < 3:
        return frames
    
    z_vals = np.array([f['z'] for f in frames])
    
    # Find the EARLIEST anomalous transition (large z increase) by scanning backwards
    # We keep updating to earlier indices as we find them
    cutoff_idx = None
    
    for i in range(len(frames) - 1, 0, -1):
        z_change = z_vals[i] - z_vals[i-1]
        
        # If we see a large POSITIVE change (z increasing = moving away)
        if z_change > 5.0:
            # Keep this as the cutoff - scanning backwards means this is earlier
            cutoff_idx = i
            # DON'T break - keep looking for even earlier anomalies
    
    # If no anomaly found, return all frames
    if cutoff_idx is None:
        return frames
    
    # Safety check: Don't remove so many frames that we can't fit
    if cutoff_idx < min_frames_required:
        if verbose:
            print(f"Z-anomaly detection would remove too many frames ({len(frames) - cutoff_idx} removed, {cutoff_idx} remaining)")
            print(f"  Falling back to finite difference method")
        return None
    
    # Report what we're removing
    removed_count = len(frames) - cutoff_idx
    if verbose:
        print(f"Z-anomaly detected: Removed last {removed_count} frame(s)")
        print(f"  Kept frames 0-{cutoff_idx-1} with z values: {[round(f['z'], 2) for f in frames[:min(5, cutoff_idx)]]}{' ...' if cutoff_idx > 5 else ''}")
        print(f"  Removed frames {cutoff_idx}-{len(frames)-1} with z values: {[round(f['z'], 2) for f in frames[cutoff_idx:]]}")
    return frames[:cutoff_idx]


# Worst case!
def finite_difference_fallback(frames, verbose=True):
    """
    Calculate velocity using finite differences when fewer than 3 frames available.
    Uses the last two frames where z is decreasing (object moving toward camera).
    
    Args:
        frames: List of 1-2 frames with 'time', 'x', 'y', 'z' keys
        verbose: If True, print diagnostic information
        
    Returns:
        tuple: (x_rate, y_rate, z_rate, diagnostics)
    """
    if len(frames) < 2:
        raise ValueError("Need at least 2 frames for finite difference velocity estimation.")
    
    # Find the last two frames where z is decreasing
    last_decreasing_idx = None
    second_last_decreasing_idx = None
    
    for i in range(len(frames) - 1, 0, -1):
        z_change = frames[i]['z'] - frames[i-1]['z']
        
        # Found a decreasing pair
        if z_change < 0:
            last_decreasing_idx = i
            second_last_decreasing_idx = i - 1
            break
    
    # If no decreasing pair found, use last two frames
    if last_decreasing_idx is None:
        if verbose:
            print("Warning: No decreasing z frames found, using last 2 frames for finite difference")
        last_decreasing_idx = len(frames) - 1
        second_last_decreasing_idx = len(frames) - 2
    
    # Calculate finite difference using the selected frames
    dt = frames[last_decreasing_idx]['time'] - frames[second_last_decreasing_idx]['time']
    if dt == 0:
        raise ValueError("Selected frames have identical timestamps.")
    
    x_rate = (frames[last_decreasing_idx]['x'] - frames[second_last_decreasing_idx]['x']) / dt
    y_rate = (frames[last_decreasing_idx]['y'] - frames[second_last_decreasing_idx]['y']) / dt
    z_rate = (frames[last_decreasing_idx]['z'] - frames[second_last_decreasing_idx]['z']) / dt
    
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
        print(f"Method: Finite difference between frames {second_last_decreasing_idx} and {last_decreasing_idx} (z decreasing)")
        print(f"  Frame {second_last_decreasing_idx}: t={frames[second_last_decreasing_idx]['time']:.3f}, z={frames[second_last_decreasing_idx]['z']:.2f}")
        print(f"  Frame {last_decreasing_idx}: t={frames[last_decreasing_idx]['time']:.3f}, z={frames[last_decreasing_idx]['z']:.2f}")
        print(f"\nVelocity components:")
        print(f"  x_rate: {x_rate:+.3f} units/sec")
        print(f"  y_rate: {y_rate:+.3f} units/sec")
        print(f"  z_rate: {z_rate:+.3f} units/sec")
        print(f"\nNote: No fit quality metrics available for finite difference method.")

    # Apply scaling factor to x and y
    x_rate = x_rate * FACTOR_BALL_X_Y_DELTA
    y_rate = y_rate * FACTOR_BALL_X_Y_DELTA
    
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
    Applies unit correction to x and y components.

    Args:
        json_path (str): Path to JSON file with position data (time, x, y, z).
        polyorder (int): Polynomial order for S-G filter (default 2).
        max_window (int): Maximum window length (default 13). Should be odd.

    Returns:
        tuple: (x_vel, y_vel, z_vel) at the last time point in units/sec.
               x_vel and y_vel are scaled by FACTOR_CLUB_X_Y_DELTA.

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
        x_vel = (frames[1]['x'] - frames[0]['x']) / dt * FACTOR_CLUB_X_Y_DELTA
        y_vel = (frames[1]['y'] - frames[0]['y']) / dt * FACTOR_CLUB_X_Y_DELTA
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
            # Good! Return velocities with scaling applied to x and y
            return x_deriv[-1] * FACTOR_CLUB_X_Y_DELTA, y_deriv[-1] * FACTOR_CLUB_X_Y_DELTA, z_deriv[-1]
        else:
            # dz is positive - remove one more frame from tail and retry
            frames_removed += 1
            if frames_removed <= max_removal:
                print(f"Warning: dz positive ({z_deriv[-1]:.2f}), removing 1 tail frame (attempt {attempt+1})")

    # If still positive after max removals, force it negative
    print(f"Warning: Could not achieve negative dz after removing {frames_removed} frames. Forcing sign.")
    return x_deriv[-1] * FACTOR_CLUB_X_Y_DELTA, y_deriv[-1] * FACTOR_CLUB_X_Y_DELTA, -abs(z_deriv[-1])



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
        if (abs(side_angle) > 56):
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

