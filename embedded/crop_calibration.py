import logging

from embedded.image_ball_locator import find_ball_y_in_image


def calibrate_crop(exposure):
    
    # Find the bottom of the ball in the image with the best exposure
    filename = f"{exposure}_exposure.jpg"
    y_pos = find_ball_y_in_image(filename)
    
    if y_pos is None:
        logging.error("[calibrate_crop] Ball not detected in %s", filename)
        # Additional error handling or early return, if needed
        return
    else:
        print(f"Ball position is currently at {y_pos}, recropping")

    # Configure the camera using this crop value
    
    


