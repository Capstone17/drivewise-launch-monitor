import numpy as np

def adjust_camera_matrix(K, crop_offset, new_size=None, original_size=None):
    """
    Adjusts a camera intrinsic matrix after cropping and optional resizing.
    
    Parameters:
        K             : np.ndarray, shape (3,3)
                        Original camera matrix from calibration
        crop_offset   : (int, int)
                        (x_offset, y_offset) from top-left in pixels
        new_size      : (int, int) or None
                        (new_width, new_height) after resize. None = no resize
        original_size : (int, int) or None
                        (orig_width, orig_height) used during calibration.
                        Required if resizing.
    
    Returns:
        np.ndarray, shape (3,3): Updated camera matrix
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_off, y_off = crop_offset

    # Adjust for cropping
    cx -= x_off
    cy -= y_off

    # Adjust for resizing if needed
    if new_size and original_size:
        cropped_width = original_size[0] - 2 * x_off
        cropped_height = original_size[1] - 2 * y_off
        scale_x = new_size[0] / cropped_width
        scale_y = new_size[1] / cropped_height

        fx *= scale_x
        fy *= scale_y
        cx *= scale_x
        cy *= scale_y

    return np.array([[fx, 0,   cx],
                     [0,  fy,  cy],
                     [0,  0,   1]], dtype=float)
