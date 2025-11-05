#!/bin/bash

# Set duration of each segment in milliseconds (e.g., 300000 ms = 5 minutes)
SEGMENT_DURATION=10000

# Output file naming pattern
OUTPUT_PATTERN="dashcam-%Y%m%d-%H%M%S.h264"

# Start continuous segmented recording
rpicam-vid -t 0 --segment $SEGMENT_DURATION -o $OUTPUT_PATTERN
