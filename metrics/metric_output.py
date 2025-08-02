from metric_calculation import * # reference_vector_calc, face_angle_calc, swing_path_calc, attack_angle_calc, side_angle_calc 
import json
import os

def load_first_movement_pair(json_path, threshold=2.0):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if len(data) < 3:
        raise ValueError("Not enough data points to compare.")

    # Search for the first two frames with a coordinate change larger than the threshold
    # Note: for enhanced accuracy and error safety, could return more points surrounding the moment of impact
    # Note: consider error in x y and z measurement
    for i in range(len(data) - 2):
        frame1 = data[i]
        frame2 = data[i + 1]
        frame3 = data[i + 2]

        dx = abs(frame2['x'] - frame1['x'])
        dy = abs(frame2['y'] - frame1['y'])
        dz = abs(frame2['z'] - frame1['z'])

        if dx > threshold or dy > threshold or dz > threshold:
            return frame1, frame2, frame3  # Return the first pair with significant movement

    return None, None, None  # No significant movement found


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
    frame_before_impact1, frame_after_impact1, frame_after_impact2 = load_first_movement_pair(ball_coords_path)  # Find the moment of impact and its surrounding frames
    print(f'Frame before impact: {frame_before_impact1}')
    print(f'Frame after impact: {frame_after_impact1}')
    print(f'Frame 2 after impact: {frame_after_impact2}\n')

    # Find club data
    pose_before_impact1, pose_after_impact1 = load_pose_at_impact(sticker_coords_path, frame_before_impact1)
    print(f'Pose before impact: {pose_before_impact1}')
    print(f'Pose after impact: {pose_after_impact1}\n')

    # Find reference data
    yaw_ideal = load_reference_yaw(stationary_sticker_path)  # Load the reference yaw
    print(f'Ideal yaw: {yaw_ideal}\n')

    reference_vector = reference_vector_calc(yaw_ideal)
    print(f'Reference vector: {reference_vector}\n')

    face_angle = face_angle_calc(pose_before_impact1, yaw_ideal)
    print(f'Face angle: {face_angle}\n')

    swing_path = swing_path_calc(pose_before_impact1, pose_after_impact1, reference_vector)
    print(f'Swing path: {swing_path}\n')

    attack_angle = attack_angle_calc(pose_before_impact1, pose_after_impact1)
    print(f'Attack angle: {attack_angle}\n')

    side_angle = side_angle_calc(frame_before_impact1, frame_after_impact1, reference_vector)
    print(f'Side angle: {side_angle}\n')

    face_to_path = face_angle - swing_path
    print(f'Face-to-path: {face_to_path}\n')

    return {
        'face_angle': face_angle,
        'swing_path': swing_path,
        'attack_angle': attack_angle,
        'side_angle': side_angle,
        'face_to_path': face_to_path
    }

