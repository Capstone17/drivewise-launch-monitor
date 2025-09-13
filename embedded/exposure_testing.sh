#!/bin/bash
# GScrop_single_photo.sh
# Usage: ./GScrop_single_photo.sh <width> <height> [shutter_us]
# Example: ./GScrop_single_photo.sh 816 144 2300

# -------------------------
# Default Parameters
# -------------------------
DEFAULT_WIDTH=200
DEFAULT_HEIGHT=144
DEFAULT_SHUTTER=700  # optional in microseconds

# -------------------------
# Parse Input
# -------------------------
width=${1:-$DEFAULT_WIDTH}
height=${2:-$DEFAULT_HEIGHT}
shutter=${3:-$DEFAULT_SHUTTER}

# -------------------------
# Validate even dimensions
# -------------------------
if (( width % 2 != 0 )); then
    echo "Error: width must be even (got $width)"
    exit 1
fi
if (( height % 2 != 0 )); then
    echo "Error: height must be even (got $height)"
    exit 1
fi

# -------------------------
# Calculate center crop
# -------------------------
crop_x=$(((1456 - width) / 2))
crop_y=$((((1088 - height) / 2) + 40))
max_crop_y=$((1088 - height))
if (( crop_y > max_crop_y )); then
    crop_y=$max_crop_y
fi
echo "Cropping at: ($crop_x, $crop_y)"

# Apply crop to media devices
for ((m=0; m<=5; ++m)); do
    media-ctl -d /dev/media$m \
        --set-v4l2 "'imx296 10-001a':0 [fmt:SBGGR10_1X10/${width}x${height} crop:(${crop_x},${crop_y})/${width}x${height}]" -v
    if [[ $? -eq 0 ]]; then
        media-ctl -d /dev/media$m --get-v4l2  # Confirm crop
        break
    fi
done

# -------------------------
# Prepare output
# -------------------------
output_dir=~/Documents/webcamGolf
mkdir -p "$output_dir"
output_file="$output_dir/photo_manual.jpg"

# -------------------------
# Capture single image
# -------------------------
# Use libcamera-still for Pi Global Shutter (manual shutter)
libcamera-still --width "$width" --height "$height" \
    --shutter "$shutter" \
    --denoise cdn_off \
    --hflip --vflip \
    -o "$output_file" -n

echo "Saved single manual exposure image to $output_file"
