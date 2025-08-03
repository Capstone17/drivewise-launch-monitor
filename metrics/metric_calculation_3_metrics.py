# Notes:
# - Improvements to be made: Consider more frames than just two, in case of error.
# - Could combine swing path and attack angle since calculation is very similar
# - Side angle and launch angle are essentially the same calculations as swing path and attack angle, could be modularized more

import numpy as np
from scipy.spatial.transform import Rotation as R


# FACE ANGLE
def face_angle_calc(swing_path, side_angle) -> float:
    """
    Estimate the face angle using the swing path and the side angle.
    The formula used is from https://www.researchgate.net/publication/323373897_The_Influence_of_Face_Angle_and_Club_Path_on_the_Resultant_Launch_Angle_of_a_Golf_Ball.
    For 7-iron:
        Horizontal launch = Side angle = H, Face angle = F, Swing path = P
        H =~ 0.7F + 0.3P - 0.013
        F = (H - 0.3P) / 0.7 + 0.019
    I have done some testing on this based on videos such as this one: https://youtu.be/sOlladSBOak.
        We should be within a degree of accuracy for face angle.
    Currently I only use path and face angle, not considering gear effect or any other factor.

    Args:
        swing_path (float): The angular path of the club in the XZ plane.
        side_angle (float): The angular path of the ball in the XZ plane.

    Returns:
        float: The angle of the club face in degrees.
    """
    
    face_angle = (side_angle - 0.3 * swing_path) / 0.7 + 0.019
    return face_angle


# SWING PATH
def swing_path_calc(pose1: dict, pose2: dict, reference_vector=[0, 1]):
    """
    Analyze horizontal motion and orientation change between two poses.

    Args:
        pose1 (dict): First marker pose with x, y, z, roll, pitch, yaw.
        pose2 (dict): Second marker pose with same format.
        reference_vector (list or np.ndarray): Direction of reference (default [0, 1] = +Z forward).

    Returns:
        swing_path (float): The angular path of the club in the XZ plane.
            This function should take into account sign of the direction
    """

    movement_vector = [pose2["x"] - pose1["x"], pose2["z"] - pose1["z"]]

    v = np.array(movement_vector, dtype=float)
    ref = np.array(reference_vector, dtype=float)
    
    # Guard against zero-length vectors
    if np.linalg.norm(v) == 0 or np.linalg.norm(ref) == 0:
        return 0.0
    
    # Normalize vectors
    v /= np.linalg.norm(v)
    ref /= np.linalg.norm(ref)
    
    # Signed angle using atan2
    swing_path = np.degrees(np.arctan2(v[0]*ref[1] - v[1]*ref[0], np.dot(v, ref)))    

    return swing_path

# ATTACK ANGLE
# This function assumes that the camera is perfectly level (flat)
# This means that movement in the "y" direction represents true vertical displacement 
def attack_angle_calc(pose1: dict, pose2: dict, reference_vector=[0, 1]) -> float:
    """
    Calculate the vertical (Y-axis) displacement of the club.

    Args:
        pose1 (dict): First marker pose with x, y, z, roll, pitch, yaw.
        pose2 (dict): Second marker pose with same format.
        reference_vector (list or np.ndarray): Direction of reference (default [0, 1] = +Z forward).

    Returns:
        attack_angle (float): The angular path of the club in the YZ plane.
    """
    
    movement_vector = [pose2["y"] - pose1["y"], pose2["z"] - pose1["z"]]

    v = np.array(movement_vector, dtype=float)
    ref = np.array(reference_vector, dtype=float)
    
    # Guard against zero-length vectors
    if np.linalg.norm(v) == 0 or np.linalg.norm(ref) == 0:
        return 0.0
    
    # Normalize vectors
    v /= np.linalg.norm(v)
    ref /= np.linalg.norm(ref)
    
    # Signed angle using atan2
    attack_angle = np.degrees(np.arctan2(v[0]*ref[1] - v[1]*ref[0], np.dot(v, ref)))    

    return attack_angle



# SIDE ANGLE
# - "Toward the camera" is negative
def side_angle_calc(pose1: dict, pose2: dict, reference_vector=[0, -1]):
    """
    Analyze horizontal motion and orientation change between two poses.

    Args:
        pose1 (dict): First ball pose with x, y, z.
        pose2 (dict): Second ball pose with same format.
        reference_vector (list or np.ndarray): Direction of reference (default [0, 1] = +Z forward).

    Returns:
        side_angle (float): The angular path of the ball in the XZ plane
    """

    movement_vector = [pose2["x"] - pose1["x"], pose2["z"] - pose1["z"]]

    v = np.array(movement_vector, dtype=float)
    ref = np.array(reference_vector, dtype=float)
    
    # Guard against zero-length vectors
    if np.linalg.norm(v) == 0 or np.linalg.norm(ref) == 0:
        return 0.0
    
    # Normalize vectors
    v /= np.linalg.norm(v)
    ref /= np.linalg.norm(ref)
    
    # Signed angle using atan2
    side_angle = np.degrees(np.arctan2(v[0]*ref[1] - v[1]*ref[0], np.dot(v, ref)))    

    return side_angle


