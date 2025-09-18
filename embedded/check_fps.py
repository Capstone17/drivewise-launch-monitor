import cv2
import sys
import time

def check_video_fps(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Get FPS from metadata
    metadata_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Metadata FPS: {metadata_fps:.2f}")
    
    # Count frames to verify FPS
    print("Counting frames, please wait...")
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        frame_count += 1
    
    elapsed_time = time.time() - start_time
    measured_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    print(f"Total frames: {frame_count}")
    print(f"Measured FPS (actual read speed): {measured_fps:.2f}")
    
    cap.release()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_fps.py <video.mkv>")
    else:
        check_video_fps(sys.argv[1])
