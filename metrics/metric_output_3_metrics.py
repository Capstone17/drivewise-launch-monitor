from metric_calculation_3_metrics import * # face_angle_calc, swing_path_calc, attack_angle_calc, side_angle_calc 
import json
import os



def load_movement_window(json_path, threshold=5.0, pre_frames=4, post_frames=4):
    """
    Finds the first instance of major movement and returns a window of frames.
    
    Returns:
        tuple: (window, impact_idx_in_window, pre_frame, post_frame)

    Printing:
        Entire window:
            for idx, frame in enumerate(window):
                print(f"Frame {idx}: x={frame['x']:.3f}, y={frame['y']:.3f}, z={frame['z']:.3f}")
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    if len(data) < (pre_frames + post_frames + 1):
        raise ValueError("Not enough data points to extract full window.")

    for i in range(len(data) - 1):
        frame1 = data[i]
        frame2 = data[i + 1]

        dx = abs(frame2['x'] - frame1['x'])
        dy = abs(frame2['y'] - frame1['y'])
        dz = abs(frame2['z'] - frame1['z'])

        if dx > threshold or dy > threshold or dz > threshold:
            start_idx = max(0, i - pre_frames)
            end_idx = min(len(data), i + 1 + post_frames)

            window = data[start_idx:end_idx]

            # Position of the impact frame inside the window
            impact_idx_in_window = i - start_idx  

            # Exact frames
            pre_frame = window[impact_idx_in_window] if impact_idx_in_window >= 0 else None
            post_frame = window[impact_idx_in_window + 1] if impact_idx_in_window + 1 < len(window) else None

            return window, impact_idx_in_window, pre_frame, post_frame

    return [], None, None, None  # No movement found


def load_pose_at_impact(json_path, frame_before_impact, threshold=2.0):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if len(data) < 2:
        raise ValueError("Not enough data points to compare.")

    # Search for the first two frames with a coordinate change larger than the threshold
    # Note: for enhanced accuracy and error safety, could return more points surrounding the moment of impact
    # Note: consider error in x y and z measurement
    for i in range(len(data) - 1):
        pose1 = data[i]
        pose2 = data[i + 1]

        if abs(pose1['time'] - frame_before_impact['time']) < threshold:
            return pose1, pose2

    return None, None  # No significant movement found


# Find the pose of a sticker at a given time
def load_pose_at_time(json_path, target_time, tolerance=0.02):
    """
    Finds the pose dictionary in the JSON that matches the given time (within a small tolerance).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    for entry in data:
        if abs(entry["time"] - target_time) < tolerance:
            return entry  # Return full pose dict
    
    raise ValueError(f"No pose found for time ≈ {target_time}")


# Find the reference yaw
def load_reference_yaw(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if not data or 'yaw' not in data[0]:
        raise ValueError("Invalid JSON format or missing 'yaw' field.")
    
    return data[0]['yaw']


def return_metrics():
    # Coordinate source paths
    src_coords = '../'
    ball_coords_path = os.path.join(src_coords, 'ball_coords.json')
    sticker_coords_path = os.path.join(src_coords, 'sticker_coords.json')
    stationary_sticker_path = os.path.join(src_coords, 'stationary_sticker.json')


    # Find ball data
    window, impact_idx, pre_frame, post_frame = load_movement_window(ball_coords_path, threshold=5.0)  # Find moment of impact and its surrounding frames
    print("Impact frame index in window:", impact_idx)
    print("Pre-impact frame:", pre_frame)
    print("Post-impact frame:", post_frame)

    # # Find club data
    # pose_before_impact1, pose_after_impact1 = load_pose_at_impact(sticker_coords_path, frame_before_impact1)
    # print(f'Pose before impact: {pose_before_impact1}')
    # print(f'Pose after impact: {pose_after_impact1}\n')

    # face_angle = face_angle_calc(pose_before_impact1, yaw_ideal)
    # print(f'Face angle: {face_angle}\n')

    # swing_path = swing_path_calc(pose_before_impact1, pose_after_impact1, reference_vector)
    # print(f'Swing path: {swing_path}\n')

    # attack_angle = attack_angle_calc(pose_before_impact1, pose_after_impact1)
    # print(f'Attack angle: {attack_angle}\n')

    side_angle = side_angle_calc(pre_frame, post_frame)
    print(f'Side angle: {side_angle}\n')

    # face_to_path = face_angle - swing_path
    # print(f'Face-to-path: {face_to_path}\n')

    # return {
    #     'face_angle': face_angle,
    #     'swing_path': swing_path,
    #     'attack_angle': attack_angle,
    #     'side_angle': side_angle,
    #     'face_to_path': face_to_path
    # }

