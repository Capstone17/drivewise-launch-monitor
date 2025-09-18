/*
SUMMARY:
- Name: GScrop.cpp
- This is the updated GScrop that works for new RPi firmware from September 15th, 2025

COMPILE AND RUN:
- g++ -O2 -o GScrop GScrop.cpp
*/

#include <cstdlib>
#include <iostream>

int main() {
    const char* cmd = 
        "media-ctl -d /dev/media0 "
        "--set-v4l2 \"'imx296 11-001a':0 [fmt:SBGGR10_1X10/672x128 crop:(0,0)/224x128]\"";

    int ret = std::system(cmd);
    if (ret != 0) {
        std::cerr << "ERROR: Failed to configure media pipeline\n";
        return 1;
    }

    return 0;
}
