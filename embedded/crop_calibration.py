import logging
import subprocess
from pathlib import Path
import time

from embedded.image_ball_locator import find_ball_y_in_image


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


DEFAULT_CROP_OFFSET = 50
MAX_CALIBRATION_ATTEMPTS = 10
TARGET_PIXELS_FROM_BOTTOM = 2
TOLERANCE = 1  # pixels - acceptable deviation from target


def calculate_crop_adjustment(pixels_bottom, target=TARGET_PIXELS_FROM_BOTTOM, tolerance=TOLERANCE):
    """
    Calculate how much to adjust the crop offset.
    
    Returns:
        int: Adjustment needed (positive = move crop down, negative = move up)
             0 if already within tolerance
    """
    deviation = pixels_bottom - target
    
    # Already within tolerance
    if abs(deviation) <= tolerance:
        return 0
    
    # Need to adjust crop
    # Since camera is flipped, positive offset moves crop down (ball appears higher)
    # If ball is too high (pixels_bottom > target), we need positive adjustment
    # If ball is too low (pixels_bottom < target), we need negative adjustment
    return -round(deviation)


def run_command(cmd, description=""):
    """Run a subprocess command with error handling."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stdout:
            logger.debug(f"{description} stdout: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing {description}: {' '.join(cmd)}")
        logger.error(f"Exit Code: {e.returncode}")
        logger.error(f"Stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error(f"Command not found: {' '.join(cmd)}")
        return False


def configure_new_crop(crop_offset, exposure, exposure_samples_path, config_path):
    """
    Configure camera crop and capture a new image.
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Configuring crop with offset: {crop_offset}")
    
    output_video = exposure_samples_path + f"{exposure}_exposure.mp4"
    output_image = exposure_samples_path + f"{exposure}_exposure.jpg"
    
    # Define the commands
    commands = [
        {
            'cmd': [config_path + 'GS_config.sh', '224', '128', str(crop_offset)],
            'desc': 'Configure crop'
        },
        {
            'cmd': ['rpicam-vid', '-o', output_video, '--level', '4.2', '--camera', '0', 
                    '--width', '224', '--height', '128', '--hflip', '--vflip', 
                    '--no-raw', '-n', '--shutter', str(exposure), '--frames', '1'],
            'desc': 'Capture video frame'
        },
        {
            'cmd': ['ffmpeg', '-y', '-loglevel', 'error', '-i', output_video, 
                    '-frames:v', '1', '-update', '1', output_image],
            'desc': 'Extract JPEG frame'
        }
    ]
    
    # Execute commands in sequence
    for command_info in commands:
        success = run_command(command_info['cmd'], command_info['desc'])
        if not success:
            logger.error(f"Failed at step: {command_info['desc']}")
            return False
    
    # Wait a moment to ensure file is written
    time.sleep(0.1)
    
    logger.info("Crop configuration and capture completed successfully")
    return True


def calibrate_crop(exposure):
    """
    Calibrate camera crop so the ball appears ~2 pixels from the bottom of the frame.
    
    Args:
        exposure: Shutter speed/exposure value for the camera
        
    Returns:
        tuple: (success: bool, final_crop_offset: int, px_bottom: float)
    """
    # Setup paths
    exposure_samples_path = Path("~/Documents/webcamGolf/embedded/exposure_samples/").expanduser()
    exposure_samples_path_as_str = str(exposure_samples_path) + "/"
    config_path = Path("~/Documents/webcamGolf/embedded/").expanduser()
    config_path_as_str = str(config_path) + "/"
    filename = f"{exposure}_exposure.jpg"
    full_image_path = exposure_samples_path_as_str + filename
    
    logger.info(f"Starting crop calibration for exposure={exposure}")
    
    # Start with default crop offset
    current_crop_offset = DEFAULT_CROP_OFFSET
    
    # Capture initial image
    logger.info("Capturing initial image with default crop")
    if not configure_new_crop(current_crop_offset, exposure, exposure_samples_path_as_str, config_path_as_str):
        logger.error("Failed to capture initial image")
        return False, current_crop_offset, None
    
    # Iterative calibration loop
    for iteration in range(MAX_CALIBRATION_ATTEMPTS):
        logger.info(f"Calibration iteration {iteration + 1}/{MAX_CALIBRATION_ATTEMPTS}")
        
        # Detect ball in current image
        result = find_ball_y_in_image(full_image_path)
        
        if result is None:
            logger.warning(f"Ball not detected in {filename}")
            # If we can't find the ball, try moving the crop down significantly
            if iteration == 0:
                logger.info("First attempt failed - trying larger crop offset")
                current_crop_offset += 30
                if not configure_new_crop(current_crop_offset, exposure, exposure_samples_path_as_str, config_path_as_str):
                    return False, current_crop_offset, None
                continue
            else:
                logger.error("Ball detection failed after adjustment")
                return False, current_crop_offset, None
        
        px_top, px_bottom = result
        logger.info(f"Ball detected: {px_top:.1f}px from top, {px_bottom:.1f}px from bottom")
        
        # Calculate needed adjustment
        adjustment = calculate_crop_adjustment(px_bottom, TARGET_PIXELS_FROM_BOTTOM, TOLERANCE)
        
        if adjustment == 0:
            logger.info(f"Calibration successful! Ball is {px_bottom:.1f}px from bottom (target: {TARGET_PIXELS_FROM_BOTTOM}px)")
            return True, current_crop_offset, px_bottom
        
        # Apply adjustment
        logger.info(f"Ball needs adjustment: {adjustment}px (current: {px_bottom:.1f}px, target: {TARGET_PIXELS_FROM_BOTTOM}px)")
        current_crop_offset += adjustment
        
        # Capture new image with adjusted crop
        if not configure_new_crop(current_crop_offset, exposure, exposure_samples_path_as_str, config_path_as_str):
            logger.error("Failed to recapture with adjusted crop")
            return False, current_crop_offset, px_bottom
    
    # Max iterations reached
    logger.warning(f"Max calibration attempts ({MAX_CALIBRATION_ATTEMPTS}) reached")
    logger.warning(f"Final position: {px_bottom:.1f}px from bottom (target: {TARGET_PIXELS_FROM_BOTTOM}px)")
    return False, current_crop_offset, px_bottom

