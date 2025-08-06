# Notes:
# - Improvements to be made: Consider more frames than just two, in case of error.
# - Could combine swing path and attack angle since calculation is very similar
# - Side angle and launch angle are essentially the same calculations as swing path and attack angle, could be modularized more

import numpy as np
import math
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



# SWING PATH & SIDE ANGLE
# - Uses velocity components to calculate side angle
def horizontal_movement_angle_from_rates(dx: float, dz: float, reference_vector=[0, -1]) -> float:
    """
    Calculate the horizontal angular path in the XZ plane using velocity components.

    Args:
        dx (float): Rate of change in X (units/sec).
        dz (float): Rate of change in Z (units/sec).
        reference_vector (list or np.ndarray): Reference direction (default [0, -1] = -Z forward).

    Returns:
        side_angle (float): Signed angle in degrees between movement vector and reference.
    """
    movement_vector = np.array([dx, dz], dtype=float)
    ref = np.array(reference_vector, dtype=float)

    # Guard against zero-length vectors
    if np.linalg.norm(movement_vector) == 0 or np.linalg.norm(ref) == 0:
        return 0.0

    # Normalize vectors
    movement_vector /= np.linalg.norm(movement_vector)
    ref /= np.linalg.norm(ref)

    # Signed angle using atan2
    horizontal_movement_angle = np.degrees(
        np.arctan2(
            movement_vector[0] * ref[1] - movement_vector[1] * ref[0],
            np.dot(movement_vector, ref)
        )
    )

    return horizontal_movement_angle  # Negate the angle due to mirroring


# ATTACK ANGLE
def vertical_movement_angle_from_rates(dy: float, dz: float, reference_vector=[0, -1]) -> float:
    """
    Calculate the vertical angular path in the YZ plane using velocity components.

    Args:
        dy (float): Rate of change in Y (vertical, units/sec).
        dz (float): Rate of change in Z (units/sec).
        reference_vector (list or np.ndarray): Reference direction (default [0, -1] = -Z forward).

    Returns:
        vertical_angle (float): Signed angle in degrees between movement vector and reference.
    """
    movement_vector = np.array([dy, dz], dtype=float)
    ref = np.array(reference_vector, dtype=float)

    # Guard against zero-length vectors
    if np.linalg.norm(movement_vector) == 0 or np.linalg.norm(ref) == 0:
        return 0.0

    # Normalize vectors
    movement_vector /= np.linalg.norm(movement_vector)
    ref /= np.linalg.norm(ref)

    # Signed angle using atan2
    vertical_angle = np.degrees(
        np.arctan2(
            movement_vector[0] * ref[1] - movement_vector[1] * ref[0],
            np.dot(movement_vector, ref)
        )
    )

    return vertical_angle


# SPEED
def cmps_to_speed_kmh(x_rate_cmps, y_rate_cmps, z_rate_cmps):
    """
    Convert velocity components from cm/s to total speed in km/h.
    """
    factor = 0.036  # 1 cm/s = 0.036 km/h
    speed_cmps = math.sqrt(x_rate_cmps**2 + y_rate_cmps**2 + z_rate_cmps**2)
    return speed_cmps * factor
