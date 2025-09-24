#!/bin/bash
set -e  # Exit on error
set -x  # Uncomment to print each command before execution

# -------------------------
# Default Values
# -------------------------
DEFAULT_TIME="1s"
DEFAULT_SHUTTER=10
VIDEO_DIR="$(dirname "$0")/.."  # Save one directory up from the script location

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
while true; do
    filename=$(printf "vid%02d.mp4" "$counter")
    filepath="$VIDEO_DIR/$filename"

    if [[ ! -f "$filepath" ]]; then
        break
    fi
    ((counter++))
done

echo "Saving video as: $filepath"

# -------------------------
# Run rpicam-vid (outputs will display in terminal)
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

# -------------------------
# Completion Message
# -------------------------
echo "Video capture complete: $filepath"
