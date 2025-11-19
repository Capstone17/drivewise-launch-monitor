import logging
import subprocess
from pathlib import Path

from embedded.image_ball_locator import find_ball_y_in_image


def calculate_crop_offset(pixels_bottom, threshold=5):
    # If the bottom of the ball is too close to the bottom
    if pixels_bottom < threshold:
        return threshold
    # If the ball is too far up, crop such that the ball is 5 pixels away from the bottom
    elif (pixels_bottom >= 2*threshold):
        return -round(pixels_bottom - threshold)
    # Else our crop is already working
    else:
        return 0
    
    
def configure_new_crop(new_crop_offset, exposure, path):
    # Define the exposure sample photo path

    # Define the commands as a list of lists
    commands = [
        ['echo', 'Starting crop calibration command series...'],
        ['echo', 'Recalibrating camera...'],
        ['echo', './GS_config', '224', '128', new_crop_offset],
        ['echo', 'Capturing exposures...'],
        ['rpicam-vid', '-o', path + exposure + '_exposure.mp4', '--level', '4.2', '--camera', '0', '--width', '224', '--height', '128', '--hflip', '--vflip', '--no-raw', '-n', '--shutter', '50', '--frames', '1'],
        ['echo', 'Extracting frames...'],
        ['ffmpeg', '-y', '-loglevel', 'error', '-i', path + exposure + '_exposure.mp4', '-frames:v', '1', '-update', '1', path + exposure + '_exposure.jpg', '-y']
    ]

    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            print(f"Command: {' '.join(cmd)}")
            print(f"Stdout:\n{result.stdout}")
            if result.stderr:
                print(f"Stderr:\n{result.stderr}")

        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {' '.join(cmd)}")
            print(f"Exit Code: {e.returncode}")
            print(f"Stderr:\n{e.stderr}")
            break
        except FileNotFoundError:
            print(f"Error: Command not found - {' '.join(cmd)}")
            break

    print("Command series finished.")


def calibrate_crop(exposure):
    
    # Find the bottom of the ball in the image with the best exposure
    exposure_samples_path = Path("~/Documents/webcamGolf/embedded/exposure_samples/").expanduser()
    exposure_samples_path_as_str = str(exposure_samples_path) + "/"
    filename = f"{exposure}_exposure.jpg"
    result = find_ball_y_in_image(exposure_samples_path_as_str + filename)
    
    if result is None:
        print(f"[calibrate_crop] Warning: Ball not detected in {filename}")
        return  # Or handle as needed

    px_top, px_bottom = result
    logging.info(f"[calibrate_crop] Detected ball bottom at {px_top:.1f} px from top, {px_bottom:.1f} px from bottom")


    # --------------------
    # Calculate New Crop
    # - A positive crop value will move the crop downwards, since the camera is flipped
    # --------------------
    calibrated = False
    while (calibrated == False):
        crop_offset = calculate_crop_offset(px_bottom, 5)
        logging.info(f"[calibrate_crop] New crop value is {crop_offset}")

        # If the cropping is already within the threshold, quit
        if (crop_offset == 0):
            calibrated = True
        else:
            configure_new_crop(crop_offset, exposure, exposure_samples_path_as_str)
    
   
    
    


