import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


# REFERENCE VECTOR
def reference_vector_calc(yaw_deg):
    """
    Converts yaw angle to a unit direction vector in the XZ plane.
    yaw_deg: yaw in degrees (rotation around Y-axis)
    Returns: [x, z] unit vector
    """
    yaw_rad = np.radians(yaw_deg)
    x = np.sin(yaw_rad)
    z = np.cos(yaw_rad)
    return np.array([x, z])


# FACE ANGLE
def face_angle_calc(pose1: dict, yaw_ideal: float) -> float:
    """
    Compute the yaw difference (in degrees) between two Euler yaw angles.

    Args:
        yaw_current (float): The current yaw angle in degrees.
        yaw_ideal (float): The ideal/reference yaw angle in degrees.

    Returns:
        float: The yaw angle difference in degrees, wrapped to [-180, 180).
    """
    yaw_current = pose1['yaw']
    yaw_error = (yaw_current - yaw_ideal + 180) % 360 - 180
    return yaw_error


# SWING PATH
def swing_path_calc(pose1: dict, pose2: dict, reference_vector):
    """
    Analyze horizontal motion and orientation change between two poses.

    Args:
        pose1 (dict): First marker pose with x, y, z, roll, pitch, yaw.
        pose2 (dict): Second marker pose with same format.
        reference_vector (list or np.ndarray): Direction of reference (default [0, 1] = +Z forward).

    Returns:
        dict: {
            "horizontal_displacement": np.ndarray [dx, dz],
            "distance": float,
            "motion_angle_deg": float
        }
    """
    # Extract horizontal positions (X, Z)
    p1 = np.array([pose1["x"], pose1["z"]], dtype=float)
    p2 = np.array([pose2["x"], pose2["z"]], dtype=float)

    # Compute displacement vector and distance
    delta_xz = p2 - p1

    # Check for the [0, 0] case
    if np.linalg.norm(delta_xz) == 0:
        return {
            "horizontal_displacement": delta_xz,
            "distance": 0.0,
            "motion_angle_deg": 0.0
        }
    distance = np.linalg.norm(delta_xz)

    # Compute motion direction angle relative to reference vector
    v_move = delta_xz / np.linalg.norm(delta_xz)
    v_ref = np.array(reference_vector, dtype=float)
    v_ref /= np.linalg.norm(v_ref)

    cross = v_ref[0]*v_move[1] - v_ref[1]*v_move[0]
    dot = np.dot(v_ref, v_move)
    motion_angle_deg = np.degrees(np.arctan2(cross, dot))  # signed angle

    return {
        "horizontal_displacement": delta_xz,
        "distance": distance,
        "motion_angle_deg": motion_angle_deg
    }


# ATTACK ANGLE
# This function assumes that the camera is perfectly level (flat)
# This means that movement in the "y" direction represents true vertical displacement 
def attack_angle_calc(pose1: dict, pose2: dict) -> float:
    """
    Calculate the vertical (Y-axis) displacement of the club.

    Args:
        pose1 (dict): First pose with key "y".
        pose2 (dict): Second pose with key "y".

    Returns:
        float: Vertical displacement (positive = upward, negative = downward).
    """
    return pose2["y"] - pose1["y"]



# SIDE ANGLE
def side_angle_calc(pos1_dict, pos2_dict, reference_vector):
    """
    Computes horizontal angle of movement (in XZ plane) relative to a reference direction.

    Parameters:
    - pos1_dict: Dictionary with keys 'x', 'y', 'z' (e.g., first ball position)
    - pos2_dict: Dictionary with keys 'x', 'y', 'z' (e.g., second ball position)
    - reference_vector: 2D reference direction in XZ plane (e.g., [0, 1] for forward)

    Returns:
    Dictionary with:
      - delta_xz: XZ movement vector
      - distance_xz: distance in XZ plane
      - angle_deg: signed angle to reference vector
    """
    p1_xz = np.array([pos1_dict["x"], pos1_dict["z"]], dtype=float)
    p2_xz = np.array([pos2_dict["x"], pos2_dict["z"]], dtype=float)

    delta_xz = p2_xz - p1_xz
    distance_xz = np.linalg.norm(delta_xz)

    move_vec = delta_xz / distance_xz if distance_xz != 0 else np.zeros(2)
    ref_vec = np.array(reference_vector, dtype=float)
    ref_vec /= np.linalg.norm(ref_vec)

    cross = ref_vec[0] * move_vec[1] - ref_vec[1] * move_vec[0]
    dot = np.dot(ref_vec, move_vec)
    angle_rad = np.arctan2(cross, dot)
    angle_deg = np.degrees(angle_rad)

    return {
        "delta_xz": delta_xz,
        "distance_xz": distance_xz,
        "angle_deg": angle_deg
    }


