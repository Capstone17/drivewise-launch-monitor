#!/usr/bin/env python3
"""Simple standalone test that lights the yellow status LED using the shared helper."""

import time

from status_led import cleanup_status_led, set_status_led_color


def main() -> None:
    set_status_led_color("yellow")
    print("Yellow LED should now be on (GPIO22 high). Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup_status_led()


if __name__ == "__main__":
    main()
