#!/bin/bash

# Create executable: 
#   chmod +x record_test_vid.sh
# Usage:
#   ./media_config.sh <width> <height> <framerate> <duration_ms> [shutter_us]
# Example:
#   ./media_config.sh 640 480 500 2000 1000

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <width> <height> <framerate> <duration_ms> [shutter_us]"
    exit 1
fi

WIDTH=$1
HEIGHT=$2
FPS=$3
DURATION=$4
SHUTTER_US=${5:-""}  # Optional

# Ensure width and height are even (required by V4L2)
if (( WIDTH % 2 != 0 )); then echo "Width must be even"; exit 1; fi
if (( HEIGHT % 2 != 0 )); then echo "Height must be even"; exit 1; fi

# Sensor size for IMX296 (full resolution: 1440x1088)
SENSOR_W=1440
SENSOR_H=1088

# Calculate centered crop origin
CROP_X=$(( (SENSOR_W - WIDTH) / 2 ))
CROP_Y=$(( (SENSOR_H - HEIGHT) / 2 ))

# Attempt to configure the media pipeline
echo "Configuring crop to ${WIDTH}x${HEIGHT} at (${CROP_X}, ${CROP_Y})"
for ((m=0; m<=5; ++m)); do
    media-ctl -d /dev/media$m \
      --set-v4l2 "'imx296 10-001a':0 [fmt:SBGGR10_1X10/${WIDTH}x${HEIGHT} crop:(${CROP_X},${CROP_Y})/${WIDTH}x${HEIGHT}]" -v && break
done

# Check if configuration succeeded
if [[ $? -ne 0 ]]; then
    echo "Failed to configure media pipeline"
    exit 1
fi

# Set shutter argument if specified
SHUTTER_ARG=""
if [[ -n "$SHUTTER_US" ]]; then
    SHUTTER_ARG="--shutter $SHUTTER_US"
    echo "Using shutter: $SHUTTER_US us"
fi

# Use rpicam-vid to record video headlessly
echo "Recording video at ${FPS} fps for ${DURATION} ms..."
rpicam-vid \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --framerate "$FPS" \
    $SHUTTER_ARG \
    --denoise cdn_off \
    -t "$DURATION" \
    -o /dev/shm/test.mp4 \
    -n

echo "Video saved to /dev/shm/test.mp4"
