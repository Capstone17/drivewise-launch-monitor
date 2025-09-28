import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple


def analyze_exposure_in_folder(folder_path: str, visualize: bool = True) -> Tuple[List[float], str]:
    """
    Analyze all images in a folder to find the ideal exposure.
    
    Args:
        folder_path (str): Path to the folder containing images.
        visualize (bool): Whether to display histograms for each image.
        
    Returns:
        Tuple[List[float], str]: 
            - List of average brightness values for each image.
            - Suggestion for ideal exposure ("Increase", "Decrease", or "Good").
    """
    folder = Path(folder_path)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    
    # Gather all image files
    image_files = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        raise FileNotFoundError(f"No images found in folder: {folder_path}")

    brightness_values = []

    for idx, img_path in enumerate(image_files):
        # Load image as grayscale
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue

        # Calculate histogram
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()

        # Calculate mean brightness
        mean_brightness = np.mean(img)
        brightness_values.append(mean_brightness)

        # Visualize histogram
        if visualize:
            plt.figure()
            plt.title(f"Histogram for {img_path.name}")
            plt.xlabel("Pixel Intensity (0-255)")
            plt.ylabel("Frequency")
            plt.plot(hist)
            plt.xlim([0, 256])
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.show()

    # Compute overall exposure suggestion
    avg_brightness = np.mean(brightness_values)
    
    if avg_brightness < 85:
        suggestion = "Increase Exposure (too dark)"
    elif avg_brightness > 170:
        suggestion = "Decrease Exposure (too bright)"
    else:
        suggestion = "Exposure is Good"

    return brightness_values, suggestion


# Example usage
folder_path = Path("~/Documents/webcamGolf/embedded/exposure_samples/").expanduser()
brightness, recommendation = analyze_exposure_in_folder(str(folder_path), visualize=True)

print("Brightness per image:", brightness)
print("Overall Recommendation:", recommendation)
