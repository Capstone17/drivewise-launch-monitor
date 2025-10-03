#!/bin/bash
# shellcheck disable=SC2154
# (silence shellcheck wrt $cam1 environment variable)

# RUN INSTRUCTIONS:
#   Make executable: chmod +x GS_config.sh
#   Usage: ./GS_config.sh <width> <height>
#   Example: ./GS_config.sh 816 144
#   Example: ./GS_config.sh 672 128

# -------------------------
# Default Values
# -------------------------
DEFAULT_WIDTH=196
DEFAULT_HEIGHT=128

# -------------------------
# Handle Arguments
# -------------------------
if [[ $# -eq 0 ]]; then
    echo "No arguments provided. Using defaults: width=$DEFAULT_WIDTH, height=$DEFAULT_HEIGHT"
    width=$DEFAULT_WIDTH
    height=$DEFAULT_HEIGHT
elif [[ $# -eq 2 ]]; then
    width=$1
    height=$2
else
    echo "Usage: $0 [<width> <height>]"
    echo "Example: $0 196 128"
    exit 1
fi

# -------------------------
# Validate Even Dimensions
# -------------------------
if (( width % 2 != 0 )); then
    echo "Error: width must be an even number (got $width)"
    exit 1
fi

if (( height % 2 != 0 )); then
    echo "Error: height must be an even number (got $height)"
    exit 1
fi

# -------------------------
# Compute Center Crop
# -------------------------
crop_x=$(((1456 - width) / 2))
crop_y=$((((1088 - height) / 2) + 30))
max_crop_y=$((1088 - height))

if (( crop_y > max_crop_y )); then
    crop_y=$max_crop_y
fi

echo "Cropping at: ($crop_x, $crop_y)"

# -------------------------
# Detect IMX296 Entity Dynamically
# -------------------------
CAMERA_NAME=""
MEDIA_DEV=""

for dev in /dev/media0 /dev/media1 /dev/media2 /dev/media3; do
    CAMERA_NAME=$(media-ctl -d "$dev" -p 2>/dev/null | grep -oP "imx296 \d+-[0-9a-f]+" | head -n1)
    if [[ -n "$CAMERA_NAME" ]]; then
        MEDIA_DEV="$dev"
        break
    fi
done

if [[ -z "$CAMERA_NAME" ]]; then
    echo "ERROR: IMX296 camera not found on /dev/media0, /dev/media1, /dev/media2, or /dev/media3!"
    exit 1
fi

echo "Found camera $CAMERA_NAME on $MEDIA_DEV"

# -------------------------
# Configure Media Device
# -------------------------
media-ctl -d "$MEDIA_DEV" --set-v4l2 "'${CAMERA_NAME}':0 [fmt:SBGGR10_1X10/${width}x${height} crop:(${crop_x},${crop_y})/${width}x${height}]" -v

# Confirm configuration
media-ctl -d "$MEDIA_DEV" --get-v4l2
