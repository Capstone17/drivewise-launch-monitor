from .metric_calculation_3_metrics import *
import json
import os
import matplotlib.pyplot as plt



def load_ball_movement_window(json_path, threshold=5.0):
    """
    Finds the first instance of major movement and returns all frames from impact onward.
    
    Returns:
        tuple: (window, impact_idx_in_window, pre_frame, post_frame)
        - pre_frame is the impact frame
        - post_frame is the frame directly after impact
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not data:
        return [], None, None, None, None  # No data at all

    for i in range(len(data) - 1):
        frame1 = data[i]
        frame2 = data[i + 1]

        dx = abs(frame2['x'] - frame1['x'])
        dy = abs(frame2['y'] - frame1['y'])
        dz = abs(frame2['z'] - frame1['z'])

        if dx > threshold or dy > threshold or dz > threshold:
            # Window is everything from impact onward
            window = data[i:]
            impact_idx_in_window = 0  # Impact is first frame in the window
            pre_frame = window[0]
            post_frame = window[1] if len(window) > 1 else None
            last_frame = window[-1]
            return window, impact_idx_in_window, pre_frame, post_frame, last_frame

    return [], None, None, None, None  # No movement found


def load_marker_poses_using_impact_time(json_path, t_target, time_window=0.3):
    """
    Load marker poses from JSON and return poses within ±time_window of t_target,
    along with the impact pose (exact or closest to t_target).

    Args:
        json_path (str): Path to JSON file containing pose data.
        t_target (float): Target time in seconds.
        time_window (float): Time window around t_target to filter poses.

    Returns:
        tuple: (filtered_poses, impact_pose)
            filtered_poses (list): Poses within ±time_window of t_target.
            impact_pose (dict or None): Pose matching t_target or closest if no exact match.
    """
    with open(json_path, 'r') as f:
        poses = json.load(f)

    if not poses:
        return [], None

    # Filter poses within the time window
    filtered_poses = [p for p in poses if abs(p['time'] - t_target) <= time_window]

    if not filtered_poses:
        return [], None

    # Try exact match first
    for pose in filtered_poses:
        if pose['time'] == t_target:
            return filtered_poses, pose

    # No exact match → find closest time within filtered poses
    impact_pose = min(filtered_poses, key=lambda p: abs(p['time'] - t_target))

    return filtered_poses, impact_pose


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


# Ball velocity components
def velocity_components(frames):
    """
    Given a window of ≥ 3 frames, fit best-fit lines for x and z vs. time.
    Returns velocity components (x_rate, z_rate) in units per second.
    """
    if len(frames) < 3:
        raise ValueError("Need at least 3 frames for a fit.")

    t_vals = [f['time'] for f in frames]

    # Fit lines for x and z; slopes = velocity components
    x_rate, _ = fit_line(t_vals, [f['x'] for f in frames])
    y_rate, _ = fit_line(t_vals, [f['z'] for f in frames])
    z_rate, _ = fit_line(t_vals, [f['z'] for f in frames])

    return x_rate, y_rate, z_rate


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
    src_coords = './'
    ball_coords_path = os.path.join(src_coords, 'ball_coords.json')
    sticker_coords_path = os.path.join(src_coords, 'sticker_coords.json')


    # ---------------------------------
    # Load movement windows
    # ---------------------------------
    # Find ball data
    ball_window, ball_impact_idx, ball_pre_frame, ball_post_frame, ball_last_frame = load_ball_movement_window(ball_coords_path, threshold=5.0)  # Find moment of impact and its surrounding frames
    
    # Check if no movement was found
    if not ball_window or ball_impact_idx is None:

        # ---------------------------------
        # Load Window & Print
        # ---------------------------------
        print("No movement detected, using sticker only")
        marker_window, marker_frame_closest_impact = load_marker_poses_without_impact_time(sticker_coords_path)
        marker_target_time = marker_frame_closest_impact["time"] 

        # Print marker data
        print("\nMarker Frames:")  
        print("Marker impact frame:", marker_target_time)
        for idx, frame in enumerate(marker_window):
            print(f"Frame {idx}: time={frame['time']:.3f}, x={frame['x']:.3f}, y={frame['y']:.3f}, z={frame['z']:.3f}, yaw={frame['yaw']:.2f}")
       
        # ---------------------------------
        # Marker Velocity Approximation
        # ---------------------------------
        marker_dx, marker_dy, marker_dz = finite_diff_velocity(marker_window, t_target=marker_target_time)
        print(f"At time {marker_target_time}, Marker dx: {marker_dx}, Marker dy: {marker_dy}, Marker dz: {marker_dz}")
        # plot_positions(marker_window)  # For testing
        
        marker_yaw_at_impact = marker_frame_closest_impact["yaw"] 
        metrics = metrics_without_ball(marker_dx, marker_dy, marker_dz, marker_yaw_at_impact)
        
    else:

        # ---------------------------------
        # Load Window & Print
        # ---------------------------------
        print("Movement detected, using ball and sticker")
        marker_window, marker_frame_closest_impact = load_marker_poses_using_impact_time(sticker_coords_path, ball_pre_frame['time'])
        marker_target_time = marker_frame_closest_impact["time"]

        # Print ball data
        print("Impact frame index in window:", ball_impact_idx)
        print("Pre-impact frame:", ball_pre_frame)
        print("Post-impact frame:", ball_post_frame)
        print("Ball last frame:", ball_last_frame)
        for idx, frame in enumerate(ball_window):
            print(f"Frame {idx}: time={frame['time']}, x={frame['x']:.3f}, y={frame['y']:.3f}, z={frame['z']:.3f}")

        # Print marker data
        print("\nMarker Frames:")  
        print("Marker impact frame:", marker_target_time)
        for idx, frame in enumerate(marker_window):
            print(f"Frame {idx}: time={frame['time']:.3f}, x={frame['x']:.3f}, y={frame['y']:.3f}, z={frame['z']:.3f}")

        # ---------------------------------
        # Velocity Approximation
        # ---------------------------------
        ball_dx, ball_dy, ball_dz = velocity_components(ball_window)
        print(f"Ball dx: {ball_dx}, Ball dy: {ball_dy}, Ball dz: {ball_dz}")
        marker_dx, marker_dy, marker_dz = finite_diff_velocity(marker_window, t_target=marker_target_time)
        print(f"At time {marker_target_time}, Marker dx: {marker_dx}, Marker dy: {marker_dy}, Marker dz: {marker_dz}")

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

