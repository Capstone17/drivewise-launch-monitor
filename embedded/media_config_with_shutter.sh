#!/bin/bash
# Configures Raspberry Pi Global Shutter Camera (IMX296) with resolution and exposure
# Usage: ./configure_gs_camera.sh <width> <height> <shutter_us>
# Example: ./configure_gs_camera.sh 960 240 4000


if [[ $# -lt 3 ]]; then
    echo "Usage: $0 width height shutter_us"
    exit 1
fi

if [[ "$(( $1 % 2 ))" -eq 1 ]]; then
    echo "Width must be even"
    exit 1
fi

if [[ "$(( $2 % 2 ))" -eq 1 ]]; then
    echo "Height must be even"
    exit 1
fi

WIDTH=$1
HEIGHT=$2
SHUTTER_US=$3

# Determine camera index
export d=10
if grep -q "Revision.*: ...17." /proc/cpuinfo; then
    if [[ "$cam1" != "" ]]; then
        d=11
    fi
fi

# Set media crop using media-ctl
for ((m=0; m<=5; ++m)); do
    media-ctl -d /dev/media$m \
        --set-v4l2 "'imx296 $d-001a':0 [fmt:SBGGR10_1X10/${WIDTH}x${HEIGHT} crop:($(( (1440 - $WIDTH) / 2 )),$(( (1088 - $HEIGHT) / 2 )))/${WIDTH}x${HEIGHT}]" -v
    if [[ $? -eq 0 ]]; then
        echo "✔ Media crop set on /dev/media$m"
        break
    fi
done

# Set shutter using rpicam-vid (without saving or previewing)
echo "✔ Applying shutter $SHUTTER_US µs using rpicam-vid"
rpicam-vid --width "$WIDTH" --height "$HEIGHT" --shutter "$SHUTTER_US" -t 1 -n --framerate 30 >/dev/null 2>&1

if [[ $? -eq 0 ]]; then
    echo "✔ Shutter configuration applied"
else
    echo "⚠ Failed to apply shutter via rpicam-vid"
fi