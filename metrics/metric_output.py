# NOTES ------------------------
# - Error checks should be added for extreme angles
# - For first frame of movement, more reliable methods could be found
# ------------------------------


from .metric_calculation import *

from pathlib import Path
import json
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# -------------------------
# Find Impact
# -------------------------

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


# -------------------------
# Club Velocity Calculation
# -------------------------
# Polyorder=2 gives a good balance of smoothness, adaptability to acceleration/deceleration, 
#   and resistance to overfitting random noise for real-world motion estimation from short, 
#   noisy time series.
def savgol_velocity(json_path, polyorder=2):
    """
    Estimate velocity components (x, y, z) at the last time point in a JSON file
    using Savitzky-Golay smoothing/derivative.

    Args:
        json_path (str): Path to JSON file with position data (time, x, y, z).
        polyorder (int): Polynomial order for S-G filter (default 2).

    Returns:
        tuple: (x_vel, y_vel, z_vel) at the last time point in units/sec.

    Raises:
        ValueError: If fewer than 3 frames in the file.
    """
    # Load data from JSON file
    
    with open(json_path, 'r') as f:
        frames = json.load(f)
    
    N = len(frames)
    if N < 3:
        raise ValueError(f"Need at least 3 frames for Savitzky-Golay velocity. Got {N}.")

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

    # Choose largest odd window length ≤ N
    window_length = N if N % 2 == 1 else N - 1
    if window_length < 3:
        window_length = 3
    polyorder = min(polyorder, window_length - 1)

    # Median time step for derivative scaling
    dt = np.median(np.diff(t_vals))
    if dt == 0:
        raise ValueError("Time values must be distinct for velocity estimation.")

    # Compute Savitzky-Golay derivatives (velocity estimates) at each time point
    x_deriv = savgol_filter(x_vals, window_length, polyorder, deriv=1, delta=dt)
    y_deriv = savgol_filter(y_vals, window_length, polyorder, deriv=1, delta=dt)
    z_deriv = savgol_filter(z_vals, window_length, polyorder, deriv=1, delta=dt)

    # Return velocities at the last frame
    return x_deriv[-1], y_deriv[-1], z_deriv[-1]


# -------------------------
# Find Metrics
# -------------------------

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


def return_metrics() -> dict:

    # ---------------------------------
    # Filepaths
    # ---------------------------------
    # Coordinate source paths
    src_coords_path = Path("~/Documents/webcamGolf").expanduser()
    src_coords = str(src_coords_path) + "/"
    ball_coords_path = os.path.join(src_coords, 'ball_coords.json')  # PIPELINE
    sticker_coords_path = os.path.join(src_coords, 'sticker_coords.json')  # PIPELINE
    # ball_coords_path = "../ball_coords.json"  # STANDALONE
    # sticker_coords_path = "../sticker_coords.json"  # STANDALONE

    # ---------------------------------
    # Find impact time
    # --------------------------------- 
    impact_time = load_impact_time(sticker_coords_path)
    print(f"Impact time: {impact_time}")

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
    marker_dx, marker_dy, marker_dz = savgol_velocity(sticker_coords_path)
    print(f"At time {impact_time}, Marker dx: {marker_dx}, Marker dy: {marker_dy}, Marker dz: {marker_dz}")

    # ---------------------------------
    # Calculate Metrics
    # ---------------------------------
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

