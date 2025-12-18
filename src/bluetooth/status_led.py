"""Status LED control utilities for the webcamGolf project."""

import logging
from typing import Iterable, Tuple

try:
    import RPi.GPIO as _LED_GPIO
except Exception:  # pragma: no cover - handled on non-RPi environments
    _LED_GPIO = None

logger = logging.getLogger(__name__)

_STATUS_LED_RED_PIN = 17
_STATUS_LED_YELLOW_PIN = 22
_STATUS_LED_GREEN_PIN = 27
_STATUS_LED_ALL_PINS: Tuple[int, int, int] = (
    _STATUS_LED_RED_PIN,
    _STATUS_LED_YELLOW_PIN,
    _STATUS_LED_GREEN_PIN,
)
_STATUS_LED_COLOR_PINS = {
    "off": (),
    "red": (_STATUS_LED_RED_PIN,),
    "yellow": (_STATUS_LED_YELLOW_PIN,),
    "green": (_STATUS_LED_GREEN_PIN,),
}

_status_led_initialized = False
_status_led_available = True
_status_led_last_color = None
_status_led_disable_logged = False


def _ensure_status_led() -> bool:
    global _status_led_initialized
    global _status_led_available
    global _status_led_disable_logged

    if not _status_led_available:
        return False
    if _LED_GPIO is None:
        if not _status_led_disable_logged:
            logger.debug("RPi.GPIO not available; status LED control disabled.")
            _status_led_disable_logged = True
        _status_led_available = False
        return False
    if not _status_led_initialized:
        try:
            _LED_GPIO.setwarnings(False)
            _LED_GPIO.setmode(_LED_GPIO.BCM)
            for pin in _STATUS_LED_ALL_PINS:
                _LED_GPIO.setup(pin, _LED_GPIO.OUT, initial=_LED_GPIO.LOW)
        except Exception:  # pragma: no cover - hardware specific
            if not _status_led_disable_logged:
                logger.exception(
                    "Failed to initialise status LED; disabling further LED control."
                )
                _status_led_disable_logged = True
            _status_led_available = False
            return False
        _status_led_initialized = True
    return True


def _set_gpio_low(pins: Iterable[int]) -> None:
    for pin in pins:
        _LED_GPIO.output(pin, _LED_GPIO.LOW)


def _set_gpio_high(pins: Iterable[int]) -> None:
    for pin in pins:
        _LED_GPIO.output(pin, _LED_GPIO.HIGH)


def set_status_led_color(color: str) -> None:
    """Update the LED to the desired color if hardware is present."""
    global _status_led_last_color
    global _status_led_available
    global _status_led_disable_logged

    if not _ensure_status_led():
        return

    normalized = color.lower()
    pins = _STATUS_LED_COLOR_PINS.get(normalized)
    if pins is None:
        logger.debug("Unsupported status LED color '%s'; ignoring.", color)
        return
    if normalized == _status_led_last_color:
        return
    try:
        _set_gpio_low(_STATUS_LED_ALL_PINS)
        _set_gpio_high(pins)
    except Exception:  # pragma: no cover - hardware specific
        if not _status_led_disable_logged:
            logger.exception(
                "Failed to update status LED state; disabling LED control."
            )
            _status_led_disable_logged = True
        _status_led_available = False
        return
    _status_led_last_color = normalized


def cleanup_status_led() -> None:
    """Return the LED pins to a safe state and release GPIO resources."""
    global _status_led_initialized
    global _status_led_last_color

    if not _status_led_available or _LED_GPIO is None:
        return
    try:
        _set_gpio_low(_STATUS_LED_ALL_PINS)
        _LED_GPIO.cleanup(_STATUS_LED_ALL_PINS)
    except Exception:  # pragma: no cover - hardware specific
        logger.exception("Failed to clean up status LED GPIO pins.")
    finally:
        _status_led_initialized = False
        _status_led_last_color = None
