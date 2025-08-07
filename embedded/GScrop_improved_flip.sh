#!/bin/bash
# shellcheck disable=SC2154
# (silence shellcheck wrt $cam1 environment variable)

# RUN INSTRUCTIONS:
#   Make executable: chmod +x GScrop_improved_flip.sh
#   Usage: ./GScrop_improved_flip.sh <width> <height> <framerate> <duration_ms> [shutter_us]
#   Example: ./GScrop_improved_flip.sh 816 144 387 2000 2300
#   Example: ./GScrop_improved_flip.sh 672 128 425 2000 2100

# -------------------------
# Input Validation
# -------------------------

if [[ $# -lt 4 ]]; then
    echo "Usage: [narrow=1] [cam1=1] $0 <width> <height> <framerate> <duration_ms> [shutter_us]"
    echo "Example: $0 960 240 250 5000 4000"
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

export SHTR=""
if [[ $# -gt 4 ]]; then
    SHTR="--shutter"
fi

export workaround=""
if grep -q '=bookworm' /etc/os-release; then
    workaround="--no-raw"
fi

export d=10
if grep -q "Revision.*: ...17." /proc/cpuinfo; then
    if [[ -n "$cam1" ]]; then
        d=11
    fi
fi

# -------------------------
# media-ctl Setup (center crop)
# -------------------------

crop_x=$(((1456 - width) / 2))
crop_y=$((((1088 - height) / 2) + 15))
max_crop_y=$((1088 - height))

if (( crop_y > max_crop_y )); then
    crop_y=$max_crop_y
fi

echo "Cropping at: ($crop_x, $crop_y)"

for ((m=0; m<=5; ++m)); do
    media-ctl -d /dev/media$m \
        --set-v4l2 "'imx296 $d-001a':0 [fmt:SBGGR10_1X10/${width}x${height} crop:(${crop_x},${crop_y})/${width}x${height}]" -v
    if [[ $? -eq 0 ]]; then
        media-ctl -d /dev/media$m --get-v4l2  # <-- Confirm crop
        break
    fi
done

# for ((m=0; m<=5; ++m)); do
#     media-ctl -d /dev/media$m --set-v4l2 "'imx296 $d-001a':0 [fmt:SBGGR10_1X10/${width}x${height} crop:($(((1456 - width) / 2)),$(((1088 - height) / 2) + 100))/${width}x${height}]" -v
#     if [[ $? -eq 0 ]]; then
#         break
#     fi
# done

# -------------------------
# Prepare Output Directory
# -------------------------

output_dir=~/Documents/webcamGolf
mkdir -p "$output_dir"
rm -f "$output_dir/tst.pts"

# -------------------------
# Run Camera Capture (with live flip)
# -------------------------

libcamera-hello --list-cameras

echo
if grep -q "Revision.*: ...17." /proc/cpuinfo; then
    # Raspberry Pi 5 with rpicam-vid
    output_file="$output_dir/tst${cam1:+1}.mp4"
    rpicam-vid "$workaround" ${cam1:+--camera 1} --width "$width" --height "$height" \
        --denoise cdn_off --framerate "$framerate" -t "$duration" "$SHTR" "$shutter" \
        --hflip --vflip \
        -o "$output_file" -n

    # echo
    # ~/venv/bin/python ~/rpicam-apps/utils/timestamp.py --plot ${narrow:+--narrow} "$output_file"

else
    # Other Pi models using libcamera-vid
    output_file="$output_dir/tst.h264"
    pts_file="$output_dir/tst.pts"
    libcamera-vid "$workaround" --width "$width" --height "$height" \
        --denoise cdn_off --framerate "$framerate" --save-pts "$pts_file" \
        -t "$duration" "$SHTR" "$shutter" \
        --hflip --vflip \
        -o "$output_file" -n

    echo
    rm -f tstamps.csv
    if command -v ptsanalyze >/dev/null 2>&1; then
        echo "Calculating actual FPS from pts file..."
        ptsanalyze "$pts_file" | grep "average framerate"
    else
        echo "Note: ptsanalyze not found, skipping FPS benchmark."
    fi
fi
