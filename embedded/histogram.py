import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
import re


def extract_exposure_from_filename(filename: str) -> Optional[float]:
    """
    Attempt to extract a numeric exposure value from the filename.
    Example filenames:
        - image_100.jpg -> 100
        - exp200.png -> 200
    """
    match = re.search(r"(\d+)", filename)
    return float(match.group(1)) if match else None


def analyze_exposure_in_folder(folder_path: str,
                               visualize: bool = True,
                               target_brightness: float = 128.0) -> Tuple[List[float], List[float], float]:
    """
    Analyze all images in a folder to determine the ideal exposure.

    Args:
        folder_path (str): Path to the folder containing images.
        visualize (bool): Whether to display histograms for each image.
        target_brightness (float): Desired mean brightness (0-255).

    Returns:
        Tuple[List[float], List[float], float]:
            - List of exposure values
            - List of corresponding brightness values
            - Estimated ideal exposure (may be interpolated)
    """
    folder = Path(folder_path).expanduser()
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    # Gather and sort image files
    image_files = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]
    if not image_files:
        raise FileNotFoundError(f"No images found in folder: {folder_path}")

    # Sort files by extracted exposure value if possible
    image_files = sorted(image_files, key=lambda x: extract_exposure_from_filename(x.name) or 0)

    exposures = []
    brightness_values = []

    for img_path in image_files:
        exposure_val = extract_exposure_from_filename(img_path.name)
        if exposure_val is None:
            print(f"Warning: Could not determine exposure for {img_path.name}, skipping.")
            continue

        # Load image as grayscale
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue

        # Calculate mean brightness
        mean_brightness = float(np.mean(img))
        exposures.append(exposure_val)
        brightness_values.append(mean_brightness)

        # Calculate histogram for visualization
        if visualize:
            hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
            plt.figure()
            plt.title(f"Histogram for {img_path.name} (Exposure: {exposure_val})")
            plt.xlabel("Pixel Intensity (0-255)")
            plt.ylabel("Frequency")
            plt.plot(hist)
            plt.xlim([0, 256])
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.show()

    # Check if we have enough data
    if len(exposures) < 2:
        raise ValueError("Need at least two images with valid exposure values to interpolate.")

    # Find ideal exposure using linear interpolation
    exposures = np.array(exposures)
    brightness_values = np.array(brightness_values)

    # Interpolate exposure at target brightness
    estimated_exposure = np.interp(target_brightness, brightness_values, exposures)

    # Plot brightness vs exposure
    if visualize:
        plt.figure(figsize=(8, 5))
        plt.title("Exposure vs. Brightness")
        plt.xlabel("Exposure Value")
        plt.ylabel("Mean Brightness")
        plt.plot(exposures, brightness_values, 'o-', label="Measured Data")
        plt.axhline(y=target_brightness, color='r', linestyle='--', label=f"Target Brightness ({target_brightness})")
        plt.axvline(x=estimated_exposure, color='g', linestyle='--', label=f"Estimated Ideal Exposure ({estimated_exposure:.2f})")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    return exposures.tolist(), brightness_values.tolist(), float(estimated_exposure)


# Example usage
folder_path=Path("~/Documents/webcamGolf/embedded/exposure_samples/").expanduser()
exposures, brightness, best_exposure = analyze_exposure_in_folder(folder_path, visualize=True)

print("Exposures:", exposures)
print("Brightness per image:", brightness)
print(f"Estimated Ideal Exposure: {best_exposure:.2f}")
