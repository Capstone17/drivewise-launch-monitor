import logging

from embedded.image_ball_locator import find_ball_y_in_image


def calibrate_crop(exposure):
    
    # Find the bottom of the ball in the image with the best exposure
    filename = f"{exposure}_exposure.jpg"
    result = find_ball_y_in_image(filename)
    
    if result is None:
        print(f"[calibrate_crop] Warning: Ball not detected in {filename}")
        return  # Or handle as needed

    px_top, px_bottom = result
    logging.info(f"[calibrate_crop] Detected ball bottom at {px_top:.1f} px from top, {px_bottom:.1f} px from bottom")

    # Configure the camera using this crop value
    
    


