#!/usr/bin/env python3

"""Simple standalone test that lights the yellow status LED."""

import time

import RPi.GPIO as GPIO

YELLOW_LED_PIN = 22  # BCM numbering


def main() -> None:
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(YELLOW_LED_PIN, GPIO.OUT, initial=GPIO.LOW)

    GPIO.output(YELLOW_LED_PIN, GPIO.HIGH)
    print("Yellow LED should now be on (GPIO22 high). Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.output(YELLOW_LED_PIN, GPIO.LOW)
        GPIO.cleanup(YELLOW_LED_PIN)


if __name__ == "__main__":
    main()
