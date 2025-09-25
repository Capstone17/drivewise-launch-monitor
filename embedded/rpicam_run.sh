#!/bin/bash
set -e  # Exit on error
set -x  # Debug: print each command

# -------------------------
# Directories
# -------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VIDEO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Script directory: $SCRIPT_DIR"
echo "Video directory:  $VIDEO_DIR"

# -------------------------
# Default Values
# -------------------------
DEFAULT_TIME="1s"
DEFAULT_SHUTTER=10

# -------------------------
# Argument Handling
# -------------------------
if [[ $# -eq 0 ]]; then
    echo "No arguments provided. Using defaults: time=$DEFAULT_TIME, shutter=$DEFAULT_SHUTTER"
    capture_time=$DEFAULT_TIME
    shutter_speed=$DEFAULT_SHUTTER
elif [[ $# -eq 2 ]]; then
    capture_time=$1
    shutter_speed=$2
else
    echo "Usage: $0 [<time> <shutter_speed>]"
    echo "Example: $0 2s 15"
    exit 1
fi

# -------------------------
# Find Next Video File Name
# -------------------------
counter=0
while [[ $counter -lt 100 ]]; do
    filename=$(printf "vid%02d.mp4" "$counter")
    filepath="$VIDEO_DIR/$filename"

    if [[ ! -f "$filepath" ]]; then
        break
    fi
    ((counter++))
done

if [[ $counter -ge 100 ]]; then
    echo "Error: Too many videos, cleanup required." >&2
    exit 1
fi

echo "Saving video as: $filepath"

# -------------------------
# Run rpicam-vid
# -------------------------
rpicam-vid --level 4.2 \
    -t "$capture_time" \
    --camera 0 \
    --width 196 \
    --height 128 \
    --no-raw \
    --denoise cdn_off \
    -o "$filepath" \
    -n \
    --framerate 550 \
    --shutter "$shutter_speed"

echo "Video capture complete: $filepath"
