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
# Prepare Output Directory
# -------------------------

output_dir=~/Documents/webcamGolf

# Find next available output file name (tst.mp4, tst1.mp4, tst2.mp4, ...)
find_next_output_file() {
    base="$1"
    ext="$2"
    n=0
    while :; do
        if [[ $n -eq 0 ]]; then
            f="$output_dir/${base}${cam1:+1}.$ext"
        else
            f="$output_dir/${base}${cam1:+1}_$n.$ext"
        fi
        [[ ! -e "$f" ]] && { echo "$f"; return; }
        ((n++))
    done
}

filepath=$(find_next_output_file "vid" "mp4")

echo "Saving video as: $filepath"

# -------------------------
# Run rpicam-vid
# -------------------------
rpicam-vid --level 4.2 \
    -t "$capture_time" \
    --camera 0 \
    --width 224 \
    --height 128 \
    --no-raw \
    --denoise cdn_off \
    --hflip --vflip \
    -o "$filepath" \
    -n \
    --framerate 550 \
    --shutter "$shutter_speed"

echo "Video capture complete: $filepath"
