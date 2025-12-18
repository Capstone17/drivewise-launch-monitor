# DriveWise Launch Monitor – Setup Guide

## Purpose

This guide covers installing dependencies and preparing your environment to run DriveWise. For detailed instructions on camera placement, fiducial marker usage, Bluetooth pairing, configuration, and typical shot workflows, see **user_manual.pdf** included in this repository.

## Prerequisites

Before starting, ensure you have:

- **OS**: Ubuntu 22.04 LTS, Raspberry Pi OS, or similar Linux distribution
- **Python**: Version 3.10 or higher
- **Hardware**:
  - Camera (Raspberry Pi Global Shutter Camera recommended)
  - Host with Bluetooth support (or USB Bluetooth adapter)
  - Mobile device with the DriveWise companion app installed
- **Physical space**: A small area for hitting golf balls (indoor or outdoor)

## 1. Clone the repository and create a virtual environment

```bash
# Clone the repository
git clone https://github.com/Capstone17/drivewise-launch-monitor.git
cd webcamGolf

# Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

## 2. Install system dependencies

Depending on your platform, you may need to install additional system-level packages:

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y \
  python3-dev \
  libatlas-base-dev \
  libjasper-dev \
  libtiff5 \
  libjasper1 \
  libharfbuzz0b \
  libwebp6 \
  libc6 \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libopenblas-dev \
  liblapack-dev \
  gfortran \
  bluetooth \
  bluez \
  python3-dbus
```

### Raspberry Pi OS

```bash
sudo apt-get update
sudo apt-get install -y \
  python3-dev \
  libjasper-dev \
  libtiff5 \
  libjasper1 \
  libharfbuzz0b \
  libwebp6 \
  libatlas-base-dev \
  libc6 \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  bluetooth \
  bluez \
  python3-dbus
```

## 3. Install Python dependencies

From the project root with your virtual environment active:

```bash
pip install -e .
```

This will install all dependencies listed in `pyproject.toml`, including OpenCV, TensorFlow/TFLite, NumPy, SciPy, and Bluetooth libraries.

## 4. Hardware setup

For detailed instructions on camera placement, fiducial marker attachment, and hitting area layout, see **user_manual.pdf – Section 2 (Hardware Setup)**.

## 5. Bluetooth and mobile app pairing

For Bluetooth pairing with the mobile app and in-app configuration, see **user_manual.pdf – Operating Instructions**.

To enable the Bluetooth service on your host:

```bash
sudo systemctl start bluetooth
sudo systemctl enable bluetooth
sudo systemctl status bluetooth
```

## 6. Configuration

For example configurations and recommended settings, see **user_manual.pdf – Operating Instructions**.

## 7. Running the system

For the full step-by-step process of starting the system, hitting shots, and interpreting metrics, refer to **user_manual.pdf – Operating Instructions**.

## 8. Troubleshooting

### Installation issues

- **Python version error**: Ensure you're using Python 3.10 or higher:
  ```bash
  python3 --version
  ```
- **Pip install fails**: Try upgrading pip and setuptools:
  ```bash
  pip install --upgrade pip setuptools wheel
  ```

### Runtime issues

For a complete troubleshooting guide covering camera issues, detection failures, Bluetooth problems, performance optimization, and model loading errors, see **user_manual.pdf – Maintenance and Troubleshooting**.

## 9. Next steps

- Open the DriveWise mobile app and verify shot data arrives in real time
- Perform a few test swings and adjust camera placement as needed
- See **user_manual.pdf** for detailed guidance on operation and optimization
- See **architecture.md** for technical details on the system design