#!/bin/bash
# shellcheck disable=SC2154
# (silence shellcheck wrt $cam1 environment variable)

# RUN INSTRUCTIONS:
#   Make executable: chmod +x GS_config.sh
#   Usage: ./GS_config.sh <width> <height>
#   Example: ./GS_config.sh 816 144 
#   Example: ./GS_config.sh 672 128 

# -------------------------
# Input Validation
# -------------------------

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <width> <height>"
    echo "Example: $0 196 128"
    exit 1
fi

width=$1
height=$2
framerate=$3
duration=$4
shutter=$5

if (( width % 2 != 0 )); then
    echo "Error: width must be an even number (got $width)"
    exit 1
fi
if (( height % 2 != 0 )); then
    echo "Error: height must be an even number (got $height)"
    exit 1
fi


# -------------------------
# media-ctl Setup (center crop)
# -------------------------

crop_x=$(((1456 - width) / 2))
crop_y=$((((1088 - height) / 2) + 5))
max_crop_y=$((1088 - height))

if (( crop_y > max_crop_y )); then
    crop_y=$max_crop_y
fi

echo "Cropping at: ($crop_x, $crop_y)"

# Configure media
media-ctl -d /dev/media0 --set-v4l2 "'imx296 11-001a':0 [fmt:SBGGR10_1X10/${width}x${height} crop:(${crop_x},${crop_y})/${width}x${height}]" -v

# Confirm config
media-ctl -d /dev/media$m --get-v4l2
