#!/bin/bash

# Create executable: chmod +x media_config.sh
# Usage: ./set_sensor_crop.sh <width> <height>
# Example: ./set_sensor_crop.sh 128 96

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <width> <height>"
    exit 1
fi

WIDTH=$1
HEIGHT=$2

# Ensure width and height are even (required by V4L2)
if (( WIDTH % 2 != 0 )); then echo "Width must be even"; exit 1; fi
if (( HEIGHT % 2 != 0 )); then echo "Height must be even"; exit 1; fi

# Sensor size for IMX296 (full resolution: 1440x1088)
SENSOR_W=1440
SENSOR_H=1088

# Calculate centered crop origin
CROP_X=$(( (SENSOR_W - WIDTH) / 2 ))
CROP_Y=$(( (SENSOR_H - HEIGHT) / 2 ))

# Attempt to find the correct media device
for ((m=0; m<=5; ++m)); do
    media-ctl -d /dev/media$m --set-v4l2 "'imx296 10-001a':0 [fmt:SBGGR10_1X10/${WIDTH}x${HEIGHT} crop:(${CROP_X},${CROP_Y})/${WIDTH}x${HEIGHT}]" -v && exit 0
done

echo "Failed to configure media pipeline"
exit 1
