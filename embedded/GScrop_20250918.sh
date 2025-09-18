#!/usr/bin/bash

# Exit immediately if any command fails
set -e

# Configure the IMX296 camera pipeline
media-ctl -d /dev/media0 --set-v4l2 "'imx296 11-001a':0 [fmt:SBGGR10_1X10/672x128 crop:(0,0)/224x96]" -v

echo "Camera configuration applied successfully."
