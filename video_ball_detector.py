import bisect
import json
import math
import os
import sys
import time
import warnings
from collections import deque, OrderedDict
from dataclasses import dataclass
from typing import Optional, Sequence

# Ensure Ultralytics can write its settings locally and skip auto-installation
os.environ.setdefault(
    "YOLO_CONFIG_DIR", os.path.join(os.path.dirname(__file__), ".yolo")
)
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

import cv2
import numpy as np

try:
    from numpy import RankWarning as _NP_RANK_WARNING
except ImportError:
    class _NP_RANK_WARNING(RuntimeWarning):
        """Fallback warning type when numpy.RankWarning is unavailable."""
        pass

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:  # pragma: no cover - fallback when tflite-runtime is unavailable
    from tensorflow.lite.python.interpreter import Interpreter

# Light Pi-friendly OpenCV tweaks
try:
    cv2.setNumThreads(2)
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

ACTUAL_BALL_RADIUS = 2.38
FOCAL_LENGTH = 1755.0  # pixels

MIN_BALL_RADIUS_PX = 9  # pixels
EDGE_MARGIN_PX = 1
BALL_SCORE_THRESHOLD = 0.25
MOTION_WINDOW_SCORE_THRESHOLD = 0.1
MOTION_WINDOW_MIN_ASPECT_RATIO = 0.65
MOTION_WINDOW_FORWARD_SHIFT = 10
MAX_CENTER_JUMP_PX = 120.0

MOTION_WINDOW_DEBUG = os.environ.get("MOTION_WINDOW_DEBUG", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

try:
    _early_exit_env = int(os.environ.get("MOTION_WINDOW_EARLY_EXIT_MISSES", "0"))
except ValueError:
    _early_exit_env = 0
MOTION_WINDOW_EARLY_EXIT_MISSES = _early_exit_env if _early_exit_env > 0 else None
MOTION_WINDOW_MAX_SCAN_MULTIPLIER = 4
MOTION_WINDOW_MIN_SCAN_BUDGET = 1024
MOTION_WINDOW_DECODE_FAILURE_LIMIT = 64

# Load camera calibration parameters
_calib_path = os.path.join(os.path.dirname(__file__), "calibration", "camera_calib.npz")
_calib_data = np.load(_calib_path)
CAMERA_MATRIX = _calib_data["camera_matrix"]
DIST_COEFFS = _calib_data["dist_coeffs"]

FX = float(CAMERA_MATRIX[0, 0])
FY = float(CAMERA_MATRIX[1, 1])
CX = float(CAMERA_MATRIX[0, 2])
CY = float(CAMERA_MATRIX[1, 2])

_DEFAULT_CAMERA_MATRIX = CAMERA_MATRIX.astype(np.float32, copy=True)
_DEFAULT_DIST_COEFFS = DIST_COEFFS.astype(np.float32, copy=True)
_DEFAULT_FOCAL_LENGTH = float(FOCAL_LENGTH)
_DEFAULT_BALL_RADIUS = float(ACTUAL_BALL_RADIUS)
_CURRENT_CALIBRATION: dict[str, object] | None = None


def apply_calibration(calibration: dict[str, object] | None = None) -> dict[str, object]:
    """Update module-wide calibration parameters from ``calibration`` and return the resolved set."""
    global CAMERA_MATRIX, DIST_COEFFS, FX, FY, CX, CY, FOCAL_LENGTH, ACTUAL_BALL_RADIUS, _CURRENT_CALIBRATION

    resolved_matrix = _DEFAULT_CAMERA_MATRIX
    resolved_dist = _DEFAULT_DIST_COEFFS
    resolved_focal = _DEFAULT_FOCAL_LENGTH
    resolved_radius = _DEFAULT_BALL_RADIUS

    if calibration:
        try:
            if "camera_matrix" in calibration and calibration["camera_matrix"] is not None:
                mat = np.asarray(calibration["camera_matrix"], dtype=np.float32)
                if mat.shape == (3, 3):
                    resolved_matrix = mat
        except Exception:
            pass
        try:
            if "dist_coeffs" in calibration and calibration["dist_coeffs"] is not None:
                dist = np.asarray(calibration["dist_coeffs"], dtype=np.float32)
                if dist.ndim == 1:
                    dist = dist.reshape(1, -1)
                if dist.ndim == 2:
                    resolved_dist = dist
        except Exception:
            pass
        try:
            if "focal_length" in calibration and calibration["focal_length"] is not None:
                resolved_focal = float(calibration["focal_length"])
        except (TypeError, ValueError):
            pass
        try:
            if "ball_radius" in calibration and calibration["ball_radius"] is not None:
                resolved_radius = float(calibration["ball_radius"])
        except (TypeError, ValueError):
            pass

        # Allow direct overrides of fx/fy/cx/cy without supplying a full matrix.
        overrides = {}
        for key in ("fx", "fy", "cx", "cy"):
            if calibration.get(key) is None:
                continue
            try:
                overrides[key] = float(calibration[key])
            except (TypeError, ValueError):
                continue
        if overrides:
            matrix = resolved_matrix.astype(np.float32, copy=True)
            if "fx" in overrides:
                matrix[0, 0] = overrides["fx"]
            if "fy" in overrides:
                matrix[1, 1] = overrides["fy"]
            if "cx" in overrides:
                matrix[0, 2] = overrides["cx"]
            if "cy" in overrides:
                matrix[1, 2] = overrides["cy"]
            resolved_matrix = matrix

    CAMERA_MATRIX = resolved_matrix.astype(np.float32, copy=True)
    DIST_COEFFS = resolved_dist.astype(np.float32, copy=True)
    FOCAL_LENGTH = float(resolved_focal)
    ACTUAL_BALL_RADIUS = float(resolved_radius)

    FX = float(CAMERA_MATRIX[0, 0])
    FY = float(CAMERA_MATRIX[1, 1])
    CX = float(CAMERA_MATRIX[0, 2])
    CY = float(CAMERA_MATRIX[1, 2])

    _CURRENT_CALIBRATION = {
        "camera_matrix": CAMERA_MATRIX.copy(),
        "dist_coeffs": DIST_COEFFS.copy(),
        "focal_length": FOCAL_LENGTH,
        "ball_radius": ACTUAL_BALL_RADIUS,
        "fx": FX,
        "fy": FY,
        "cx": CX,
        "cy": CY,
    }
    return _CURRENT_CALIBRATION


apply_calibration()

MAX_MOTION_FRAMES = 80
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
USE_BLUR = False

CLUBFACE_MAX_JUMP_PX = 80.0
CLUB_PRECONTACT_MIN_VISIBLE_FRAC = 0.012
CLUB_PRECONTACT_MIN_COMPONENT_AREA_PX = 820.0
CLUB_PRECONTACT_MIN_ELLIPSE_MAJOR_PX = 24.0
CLUB_PRECONTACT_MIN_ELLIPSE_MINOR_PX = 8.0
CLUB_BALL_CONTACT_MARGIN_PX = 12.0
CLUB_BALL_CONTACT_HOLD_FRAMES = 5
CLUB_BALL_CONTACT_UNION_MARGIN_PX = 3.0
CLUB_ELLIPSE_MAX_DRIFT_PX = 45.0
CLUB_ELLIPSE_CONTACT_BLEND = 0.55
CLUB_ELLIPSE_HOLD_FRAMES = 4

CLUB_TEMPLATE_MIN_AREA_PX = 1500
CLUB_TEMPLATE_UPDATE_MARGIN = 1.12
CLUB_TEMPLATE_RECOVERY_RATIO = 0.82
CLUB_TEMPLATE_MIN_EXTENSION_PX = 4.0
CLUB_TEMPLATE_MAX_EXTENSION_PX = 42.0
CLUB_RESUME_DELAY_FRAMES = 6
CLUB_STAT_LIMIT_MULTIPLIER = 4.5
CLUB_DEPTH_BASELINE_FRAMES = 20
CLUB_DEPTH_BASELINE_MULTIPLIER = 1.6
CLUB_DEPTH_MIN_CM = 10.0
CLUB_DEPTH_MAX_CM = 1500.0
CLUB_MAX_FRAME_WINDOW = 34
CLUB_ROUND_SELECT_RATIO = 0.6
CLUBFACE_ELLIPSE_CORE_FRACTION = 0.45
CLUBFACE_ELLIPSE_CORE_MIN_PX = 2.5
CLUBFACE_ELLIPSE_REQUIRED_PIXELS = 24
CLUBFACE_ELLIPSE_MAX_ASPECT = 2.05
CLUBFACE_ELLIPSE_ERODE_SIZE = 5
CLUBFACE_ELLIPSE_BORDER_EXPAND_RATIO = 0.35
CLUBFACE_ELLIPSE_BORDER_EXPAND_MAX = 18.0
CLUBFACE_ELLIPSE_CORE_PERCENTILE = 78.0
CLUBFACE_ELLIPSE_MIN_PERCENTILE = 55.0
CLUBFACE_ELLIPSE_BORDER_KERNEL = 3
CLUBFACE_ELLIPSE_LEFT_BAND_FRAC = 0.22
CLUBFACE_ELLIPSE_LEFT_BAND_MAX = 26.0
CLUBFACE_ELLIPSE_TOP_BAND_FRAC = 0.55
CLUBFACE_ELLIPSE_TOPLEFT_CORE_PERCENTILE = 60.0
CLUBFACE_ELLIPSE_BOTTOM_DIST_FRAC = 0.42
CLUBFACE_ELLIPSE_BOTTOM_MIN_PX = 3.0
CLUB_FINAL_TRIM_MARGIN = 2
CLUB_TARGET_ALIGN_MIN_COVERAGE = 0.55
CLUB_TARGET_ALIGN_BONUS = 0.05
CLUB_LEFT_WEIGHT_BIAS = 1.8
CLUB_ANNOTATION_MIN_VISIBLE_FRAC = 0.0015
CLUB_ANNOTATION_MIN_FILE_BYTES = 10_240
CLUB_DEPTH_ANCHOR_TARGET_CM = 130.0
CLUB_DEPTH_ANCHOR_SAMPLES = 3
CLUB_TAIL_CONCAVITY_THRESHOLD = 0.58
CLUB_TAIL_CONCAVITY_MAX_REMOVALS = 4
CLUB_TAIL_CONCAVITY_WINDOW = 6
CLUB_TAIL_CONCAVITY_MIN_SEGMENT = 1.5
CLUB_TAIL_MIN_KEEP_RATIO = 0.6

# Adaptive blackout relaxation for tough lighting (e.g., outdoor clips)
CLUB_ALPHA_RELAX_STREAK = 5
CLUB_ALPHA_RELAX_MAX = 4
CLUB_ALPHA_RELAX_VISIBLE_STEP = 0.055
CLUB_ALPHA_RELAX_VISIBLE_CAP = 0.32
CLUB_ALPHA_RELAX_THRESHOLD_SCALE_MIN = 0.12
CLUB_ALPHA_RELAX_THRESHOLD_SCALE_FACTOR = 0.52
CLUB_ALPHA_RELAX_MIN_COVERAGE = 0.38
CLUB_ALPHA_RELAX_COVERAGE_DROP = 0.14
CLUB_ALPHA_RELAX_GUARD_MARGIN = 2.2
CLUB_ALPHA_RELAX_VISIBLE_TRIGGER = 0.006
CLUB_ALPHA_RELAX_DISABLE_BLACKOUT_LEVEL = 2
CLUB_ALPHA_RELAX_BLEND_LEVELS = (0.22, 0.33, 0.44, 0.55)
CLUB_ALPHA_RELAX_SAMPLE_TARGET = 6

CLUB_DEPTH_ANCHOR_TARGET_CM = 150.0
CLUB_DEPTH_ANCHOR_SAMPLES = 3

CLUB_COMPONENT_MIN_AREA_PX = 420.0
CLUB_COMPONENT_MIN_AREA_FRAC = 5.5e-4
CLUB_COMPONENT_MIN_VISIBLE_FRAC = 0.001
CLUB_COMPONENT_MIN_WIDTH_PX = 18.0
CLUB_COMPONENT_MIN_MOVEMENT_PX = 3.2
CLUB_COMPONENT_STATIONARY_MAX = 7
CLUB_PATH_MIN_SPAN_X_PX = 22.0
CLUB_PATH_MIN_SPAN_Y_PX = 12.0


class TimingCollector:
    """Collect sequential timing measurements and preserve insertion order."""

    def __init__(self) -> None:
        self._sections: OrderedDict[str, float] = OrderedDict()

    def add(self, name: str, duration: float) -> None:
        self._sections[name] = self._sections.get(name, 0.0) + float(duration)

    def items(self):
        return self._sections.items()

    def as_dict(self) -> dict[str, float]:
        return dict(self._sections)

    def total(self) -> float:
        return sum(self._sections.values())


@dataclass
class TailCheckResult:
    ball_present: bool
    hits: int
    frames_checked: int
    scores: list[float]
    frame_indices: list[int]


class MotionWindowError(RuntimeError):
    """Base error for failures while establishing a ball motion window."""

    def __init__(self, message: str, *, stats: Optional[dict[str, object]] = None) -> None:
        super().__init__(message)
        self.stats = stats or {}


class MotionWindowNotFoundError(MotionWindowError):
    """Raised when the search could not find any candidate ball frames."""


class MotionWindowDegenerateError(MotionWindowError):
    """Raised when detections are present but cannot form a valid window."""





def _polyfit_with_mask(
    times: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    degree: int,
) -> np.ndarray | None:
    """Return polynomial coefficients for the selected ``times``/``values`` subset."""
    count = int(np.count_nonzero(mask))
    if count == 0:
        return None
    deg = int(max(0, min(degree, count - 1)))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", _NP_RANK_WARNING)
            coeffs = np.polyfit(times[mask], values[mask], deg)
    except (np.linalg.LinAlgError, ValueError):
        return None
    return coeffs


def _robust_polyfit(
    times: np.ndarray,
    values: np.ndarray,
    *,
    degree: int,
    max_iter: int = 5,
    sigma: float = 2.5,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Robust polynomial fit that iteratively masks outliers using a MAD threshold."""
    finite_mask = np.isfinite(values)
    if int(finite_mask.sum()) == 0:
        return None, np.full_like(values, np.nan, dtype=np.float64), finite_mask

    current_mask = finite_mask.copy()
    coeffs = _polyfit_with_mask(times, values, current_mask, degree)
    if coeffs is None:
        return None, np.full_like(values, np.nan, dtype=np.float64), current_mask

    for _ in range(max_iter):
        fitted = np.polyval(coeffs, times)
        residuals = values - fitted
        inlier_residuals = residuals[current_mask]
        if inlier_residuals.size == 0:
            break
        scale = float(np.median(np.abs(inlier_residuals)))
        if not np.isfinite(scale) or scale < 1e-6:
            break
        threshold = sigma * 1.4826 * scale + 1e-6
        new_mask = current_mask & (np.abs(residuals) <= threshold)
        if int(new_mask.sum()) == int(current_mask.sum()):
            break
        current_mask = new_mask
        coeffs = _polyfit_with_mask(times, values, current_mask, degree)
        if coeffs is None:
            break

    if coeffs is None:
        coeffs = _polyfit_with_mask(times, values, finite_mask, 0)
    fitted = np.polyval(coeffs, times) if coeffs is not None else np.copy(values)
    return coeffs, fitted, current_mask




def _median_mad_limits(
    values: Sequence[float],
    *,
    multiplier: float = 3.5,
) -> tuple[float | None, float | None]:
    """Return robust lower/upper bounds using the median and MAD (or heuristics if degenerate)."""
    try:
        arr = np.asarray(
            [float(v) for v in values if v is not None and np.isfinite(float(v))],
            dtype=np.float64,
        )
    except (TypeError, ValueError):
        return None, None

    if arr.size == 0:
        return None, None
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    lower: float | None
    upper: float | None
    if not np.isfinite(mad) or mad < 1e-6:
        std = float(np.std(arr))
        if not np.isfinite(std) or std < 1e-6:
            max_val = float(np.max(arr))
            if max_val <= 0.0:
                return None, None
            upper = max(median * 1.5, max_val * 1.1)
            lower = median * 0.5 if median > 0.0 else None
        else:
            scale = std
            upper = median + multiplier * scale
            lower = median - multiplier * scale
    else:
        scale = 1.4826 * mad
        upper = median + multiplier * scale
        lower = median - multiplier * scale

    if upper is not None and median > 0.0 and upper < median:
        upper = median * 1.1
    if lower is not None:
        if median > 0.0 and lower > median:
            lower = median * 0.9
        if lower < 0.0:
            lower = 0.0
    return (float(lower) if lower is not None else None, float(upper) if upper is not None else None)


def _compute_depth_baseline_limit(
    points: Sequence[dict[str, float]],
    *,
    max_samples: int = CLUB_DEPTH_BASELINE_FRAMES,
    multiplier: float = CLUB_DEPTH_BASELINE_MULTIPLIER,
) -> float | None:
    """Derive an upper depth bound from the first few frames (club closest to camera)."""
    baseline: list[float] = []
    for entry in points:
        if len(baseline) >= max_samples:
            break
        depth_val = entry.get("_raw_depth", entry.get("z"))
        if depth_val is None:
            continue
        try:
            depth_float = float(depth_val)
        except (TypeError, ValueError):
            continue
        if np.isfinite(depth_float):
            baseline.append(depth_float)
    if len(baseline) < 3:
        return None

    arr = np.asarray(baseline, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    if not np.isfinite(mad) or mad < 1e-6:
        std = float(np.std(arr))
        if not np.isfinite(std) or std < 1e-6:
            return median * multiplier
        scale = std
    else:
        scale = 1.4826 * mad
    limit = median + multiplier * scale
    if limit < median and median > 0.0:
        limit = median * (1.0 + 0.05 * multiplier)
    return float(limit)


def _is_finite_number(value: float | None) -> bool:
    try:
        return value is not None and np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _linear_lookup(frame: int, frames: list[int], values: list[float | None]) -> float | None:
    """Return a linearly interpolated/extrapolated value for ``frame``."""
    if not frames:
        return None
    pos = bisect.bisect_left(frames, frame)

    def left(idx: int) -> tuple[int | None, float | None]:
        while idx >= 0:
            val = values[idx]
            if _is_finite_number(val):
                return frames[idx], float(val)  # type: ignore[arg-type]
            idx -= 1
        return None, None

    def right(idx: int) -> tuple[int | None, float | None]:
        limit = len(frames)
        while idx < limit:
            val = values[idx]
            if _is_finite_number(val):
                return frames[idx], float(val)  # type: ignore[arg-type]
            idx += 1
        return None, None

    if pos == 0:
        if len(frames) >= 2:
            f1, v1 = right(0)
            f2, v2 = right(1)
            if f1 is not None and f2 is not None and v1 is not None and v2 is not None and f2 != f1:
                return v1 + (frame - f1) * (v2 - v1) / (f2 - f1)
        _, v = right(0)
        return v

    if pos == len(frames):
        if len(frames) >= 2:
            f2, v2 = left(len(frames) - 1)
            f1, v1 = left(len(frames) - 2)
            if f1 is not None and f2 is not None and v1 is not None and v2 is not None and f2 != f1:
                return v2 + (frame - f2) * (v2 - v1) / (f2 - f1)
        _, v = left(len(frames) - 1)
        return v

    f1, v1 = left(pos - 1)
    f2, v2 = right(pos)
    if f1 is None or v1 is None:
        return v2
    if f2 is None or v2 is None or f2 == f1:
        return v1
    return v1 + (frame - f1) * (v2 - v1) / (f2 - f1)


def interpolate_club_points(
    points: list[dict[str, object]],
    *,
    target_frame: int | None,
    fps: float,
    start_frame: int | None = None,
) -> tuple[list[dict[str, object]], dict[str, int | float | None]]:
    """Interpolate club samples so every frame up to ``target_frame`` has coverage."""
    summary: dict[str, int | float | None] = {
        "added": 0,
        "start_frame": None,
        "target_frame": target_frame,
        "end_frame": target_frame,
    }
    if target_frame is None or fps <= 0.0 or not points:
        return points, summary

    frame_map: dict[int, dict[str, object]] = {}
    frames: list[int] = []
    for entry in points:
        raw_frame = entry.get("_frame")
        if raw_frame is None:
            continue
        frame_idx = int(raw_frame)
        if frame_idx in frame_map:
            continue
        frame_map[frame_idx] = entry
        frames.append(frame_idx)

    if len(frames) < 2:
        if frames:
            range_start = frames[0] if start_frame is None else max(0, int(start_frame))
            summary["start_frame"] = range_start
        return points, summary

    frames.sort()
    range_start = frames[0] if start_frame is None else int(start_frame)
    if range_start < 0:
        range_start = 0
    summary["start_frame"] = range_start
    if target_frame < range_start:
        return points, summary

    x_values = [float(frame_map[f]["x"]) for f in frames]
    y_values = [float(frame_map[f]["y"]) for f in frames]
    z_values = [
        float(frame_map[f]["z"]) if _is_finite_number(frame_map[f].get("z")) else None
        for f in frames
    ]

    added = 0
    for frame in range(range_start, target_frame + 1):
        if frame in frame_map:
            continue
        x_new = _linear_lookup(frame, frames, x_values)
        y_new = _linear_lookup(frame, frames, y_values)
        if x_new is None or y_new is None:
            continue
        z_new = _linear_lookup(frame, frames, z_values)
        time_val = frame / fps
        entry = {
            "time": round(float(time_val), 3),
            "x": round(float(x_new), 2),
            "y": round(float(y_new), 2),
        }
        if _is_finite_number(z_new):
            entry["z"] = round(float(z_new), 2)
        entry["_frame"] = frame
        entry["_interpolated"] = True
        pos = bisect.bisect_left(frames, frame)
        frames.insert(pos, frame)
        x_values.insert(pos, float(entry["x"]))
        y_values.insert(pos, float(entry["y"]))
        z_values.insert(pos, float(entry["z"]) if _is_finite_number(entry.get("z")) else None)
        frame_map[frame] = entry
        added += 1

    summary["added"] = added
    summary["end_frame"] = frames[-1] if frames else summary.get("end_frame")
    result = [frame_map[f] for f in sorted(frame_map)]
    result.sort(key=lambda p: (p.get("time", 0.0), p.get("_frame", 0)))
    return result, summary


def enforce_club_depth_range(
    points: list[dict[str, object]],
    *,
    min_depth: float,
    max_depth: float,
) -> tuple[list[dict[str, object]], dict[str, int | float | bool]]:
    """Ensure club depth values stay within ``[min_depth, max_depth]`` using interpolation."""
    summary: dict[str, int | float | bool] = {
        "total": 0,
        "filled": 0,
        "fallback_used": False,
    }
    if not points:
        return points, summary

    frame_entries: dict[int, dict[str, object]] = {}
    frames: list[int] = []
    for entry in points:
        raw_frame = entry.get("_frame")
        if raw_frame is None:
            continue
        frame_idx = int(raw_frame)
        if frame_idx in frame_entries:
            continue
        frame_entries[frame_idx] = entry
        frames.append(frame_idx)

    if not frames:
        return points, summary

    frames.sort()
    summary["total"] = len(frames)

    valid_frames: list[int] = []
    valid_depths: list[float] = []
    for frame in frames:
        entry = frame_entries[frame]
        depth_val = entry.get("z")
        if _is_finite_number(depth_val):
            depth_float = abs(float(depth_val))  # type: ignore[arg-type]
            if min_depth <= depth_float <= max_depth:
                valid_frames.append(frame)
                valid_depths.append(depth_float)
                continue
        entry.pop("z", None)

    fallback_default = (min_depth + max_depth) / 2.0

    for frame in frames:
        entry = frame_entries[frame]
        depth_val = entry.get("z")
        if _is_finite_number(depth_val):
            depth_float = float(depth_val)
            depth_float = float(np.clip(depth_float, min_depth, max_depth))
            entry["z"] = round(depth_float, 2)
            continue

        new_depth: float | None = None
        if valid_frames:
            new_depth = _linear_lookup(frame, valid_frames, valid_depths)
        if new_depth is None:
            new_depth = fallback_default
            summary["fallback_used"] = True
        new_depth = float(np.clip(float(new_depth), min_depth, max_depth))
        entry["z"] = round(new_depth, 2)
        valid_frames.append(frame)
        valid_depths.append(new_depth)
        summary["filled"] += 1

    return points, summary


def trim_club_points_to_range(
    points: list[dict[str, object]],
    *,
    min_frame: int | None,
    max_frame: int | None,
) -> tuple[list[dict[str, object]], dict[str, int | None]]:
    """Drop club samples outside ``[min_frame, max_frame]`` when frame indices are known."""
    if min_frame is None and max_frame is None:
        return points, {"removed": 0, "min_frame": min_frame, "max_frame": max_frame}

    kept: list[dict[str, object]] = []
    removed = 0
    for entry in points:
        frame_val = entry.get("_frame")
        if frame_val is None:
            kept.append(entry)
            continue
        try:
            frame_idx = int(frame_val)
        except (TypeError, ValueError):
            kept.append(entry)
            continue
        if min_frame is not None and frame_idx < min_frame:
            removed += 1
            continue
        if max_frame is not None and frame_idx > max_frame:
            removed += 1
            continue
        kept.append(entry)

    summary = {
        "removed": removed,
        "min_frame": min_frame,
        "max_frame": max_frame,
    }
    return kept, summary


def trim_tail_concavity_outliers(
    points: list[dict[str, object]],
    *,
    threshold: float = CLUB_TAIL_CONCAVITY_THRESHOLD,
    max_removals: int = CLUB_TAIL_CONCAVITY_MAX_REMOVALS,
    window: int = CLUB_TAIL_CONCAVITY_WINDOW,
    min_segment: float = CLUB_TAIL_CONCAVITY_MIN_SEGMENT,
    min_keep_ratio: float = CLUB_TAIL_MIN_KEEP_RATIO,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Remove sharp concave bends near the tail of the club trajectory."""

    summary: dict[str, object] = {
        "removed": 0,
        "threshold": threshold,
        "max_score": 0.0,
        "reverted": False,
    }
    if not points or max_removals <= 0 or threshold <= 0.0:
        return points, summary

    total_points = len(points)
    ordered: list[tuple[int, float, int, float, float]] = []
    for idx, entry in enumerate(points):
        try:
            x_val = float(entry["x"])
            y_val = float(entry["y"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            time_val = float(entry.get("time", 0.0))
        except (TypeError, ValueError):
            time_val = float(idx)
        frame_val_raw = entry.get("_frame")
        try:
            frame_val = int(frame_val_raw) if frame_val_raw is not None else idx
        except (TypeError, ValueError):
            frame_val = idx
        ordered.append((idx, time_val, frame_val, x_val, y_val))

    if len(ordered) < 5:
        return points, summary

    ordered.sort(key=lambda item: (item[1], item[2], item[0]))
    window = max(3, int(window))
    removed_indices: set[int] = set()
    removal_scores: list[float] = []

    while len(ordered) >= 3 and len(removal_scores) < max_removals:
        flag_idx = None
        flag_score = 0.0
        n = len(ordered)
        lower_bound = max(2, n - window)
        for pos in range(n - 1, lower_bound - 1, -1):
            if pos < 2:
                continue
            ax, ay = ordered[pos - 2][3], ordered[pos - 2][4]
            bx, by = ordered[pos - 1][3], ordered[pos - 1][4]
            cx, cy = ordered[pos][3], ordered[pos][4]
            v1x, v1y = bx - ax, by - ay
            v2x, v2y = cx - bx, cy - by
            len1 = math.hypot(v1x, v1y)
            len2 = math.hypot(v2x, v2y)
            if len1 < min_segment or len2 < min_segment:
                continue
            denom = len1 * len2
            if denom <= 1e-5:
                continue
            cross = abs(v1x * v2y - v1y * v2x)
            score = cross / denom
            if score >= threshold and score > flag_score:
                flag_idx = pos
                flag_score = score
        if flag_idx is None:
            break
        removal_scores.append(flag_score)
        summary["max_score"] = max(summary["max_score"], flag_score)
        removed_indices.add(ordered[flag_idx][0])
        ordered.pop(flag_idx)

    if not removed_indices:
        return points, summary

    filtered_points = [
        entry for idx, entry in enumerate(points) if idx not in removed_indices
    ]

    min_keep = max(5, int(math.ceil(total_points * max(0.0, min_keep_ratio))))
    if len(filtered_points) < min_keep:
        summary["reverted"] = True
        return points, summary

    summary["removed"] = len(removed_indices)
    summary["scores"] = [round(score, 3) for score in removal_scores]
    summary["remaining"] = len(filtered_points)
    return filtered_points, summary


def filter_club_point_outliers(
    points: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Trim club trajectory points with implausible sizes or distances."""
    total_points = len(points)
    width_values: list[float] = []
    area_values: list[float] = []
    depth_values: list[float] = []
    visible_values: list[float] = []
    for entry in points:
        width_val = entry.get("_width_px")
        if width_val is not None:
            try:
                width_float = float(width_val)
            except (TypeError, ValueError):
                width_float = None
            if width_float is not None and np.isfinite(width_float):
                width_values.append(width_float)
        area_val = entry.get("_area_px")
        if area_val is not None:
            try:
                area_float = float(area_val)
            except (TypeError, ValueError):
                area_float = None
            if area_float is not None and np.isfinite(area_float):
                area_values.append(area_float)
        depth_val = entry.get("_raw_depth", entry.get("z"))
        if depth_val is not None:
            try:
                depth_float = float(depth_val)
            except (TypeError, ValueError):
                depth_float = None
            if depth_float is not None and np.isfinite(depth_float):
                depth_values.append(depth_float)
        visible_val = entry.get("_visible_frac")
        if visible_val is not None:
            try:
                visible_float = float(visible_val)
            except (TypeError, ValueError):
                visible_float = None
            if visible_float is not None and np.isfinite(visible_float):
                visible_values.append(visible_float)

    width_min, width_max = _median_mad_limits(width_values, multiplier=CLUB_STAT_LIMIT_MULTIPLIER)
    area_min, area_max = _median_mad_limits(area_values, multiplier=CLUB_STAT_LIMIT_MULTIPLIER)
    _, depth_max = _median_mad_limits(depth_values, multiplier=CLUB_STAT_LIMIT_MULTIPLIER)
    visible_min, visible_max = _median_mad_limits(visible_values, multiplier=CLUB_STAT_LIMIT_MULTIPLIER)

    if width_min is not None:
        width_min = max(0.0, width_min * 0.9)
    if width_max is not None:
        width_max *= 1.05
    if area_min is not None:
        area_min = max(0.0, area_min * 0.85)
    if area_max is not None:
        area_max *= 1.08
    if visible_min is not None:
        visible_min = max(CLUB_ANNOTATION_MIN_VISIBLE_FRAC, visible_min * 0.8)
    if visible_max is not None:
        visible_max = min(1.0, visible_max * 1.2)
    baseline_depth_limit = _compute_depth_baseline_limit(points)
    if baseline_depth_limit is not None:
        if depth_max is None or baseline_depth_limit < depth_max:
            depth_max = baseline_depth_limit
    if depth_max is not None:
        depth_max *= 1.05

    filtered: list[dict[str, object]] = []
    removed = 0
    for entry in points:
        remove = False
        width_val = entry.get("_width_px")
        if width_val is not None:
            try:
                width_float = float(width_val)
            except (TypeError, ValueError):
                width_float = None
            if width_float is not None and np.isfinite(width_float):
                if width_min is not None and width_float < width_min:
                    remove = True
                if width_max is not None and width_float > width_max:
                    remove = True
        area_val = entry.get("_area_px")
        if area_val is not None:
            try:
                area_float = float(area_val)
            except (TypeError, ValueError):
                area_float = None
            if area_float is not None and np.isfinite(area_float):
                if area_min is not None and area_float < area_min:
                    remove = True
                if area_max is not None and area_float > area_max:
                    remove = True
        depth_val = entry.get("z")
        if depth_max is not None and depth_val is not None:
            try:
                depth_float = float(depth_val)
            except (TypeError, ValueError):
                depth_float = None
            if depth_float is not None and np.isfinite(depth_float) and depth_float > depth_max:
                remove = True
        visible_val = entry.get("_visible_frac")
        if visible_val is not None:
            try:
                visible_float = float(visible_val)
            except (TypeError, ValueError):
                visible_float = None
            if visible_float is not None and np.isfinite(visible_float):
                if visible_min is not None and visible_float < visible_min:
                    remove = True
                if visible_max is not None and visible_float > visible_max:
                    remove = True
        if remove:
            removed += 1
            continue
        filtered.append(entry)

    if removed and total_points:
        minimum_keep = max(5, int(math.ceil(total_points * 0.6)))
        if len(filtered) < minimum_keep:
            filtered = points
            removed = 0

    def _round_limit(value: float | None) -> float | None:
        return round(float(value), 2) if value is not None else None

    summary = {
        "applied": bool(removed),
        "total": len(points),
        "removed": removed,
        "width_min": _round_limit(width_min),
        "width_max": _round_limit(width_max),
        "area_min": _round_limit(area_min),
        "area_max": _round_limit(area_max),
        "depth_max": _round_limit(depth_max),
        "depth_baseline_limit": _round_limit(baseline_depth_limit),
        "visible_min": _round_limit(visible_min),
        "visible_max": _round_limit(visible_max),
    }
    return filtered, summary


def smooth_sticker_pixels(
    pixels: list[dict[str, float]],
    *,
    degree: int = 2,
) -> tuple[list[dict[str, float]], dict[str, int]]:
    """Fit a quadratic curve to sticker pixel samples and return a smoothed copy."""
    if degree < 1:
        degree = 1
    valid_indices: list[int] = []
    times: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for idx, entry in enumerate(pixels):
        try:
            t = float(entry["time"])
            x_val = float(entry["x"])
            y_val = float(entry["y"])
        except (KeyError, TypeError, ValueError):
            continue
        times.append(t)
        xs.append(x_val)
        ys.append(y_val)
        z_val = entry.get("z")
        if z_val is None:
            zs.append(np.nan)
        else:
            try:
                zs.append(float(z_val))
            except (TypeError, ValueError):
                zs.append(np.nan)
        valid_indices.append(idx)

    total_valid = len(valid_indices)
    if total_valid < 3:
        return pixels, {"applied": False, "total": total_valid, "inliers": total_valid, "outliers": 0}

    times_arr = np.array(times, dtype=np.float64)
    xs_arr = np.array(xs, dtype=np.float64)
    ys_arr = np.array(ys, dtype=np.float64)
    zs_arr = np.array(zs, dtype=np.float64)

    degree_use = int(min(degree, max(1, times_arr.size - 1)))
    coeff_x, _, mask_x = _robust_polyfit(times_arr, xs_arr, degree=degree_use)
    coeff_y, _, mask_y = _robust_polyfit(times_arr, ys_arr, degree=degree_use)
    if coeff_x is None or coeff_y is None:
        return pixels, {"applied": False, "total": total_valid, "inliers": total_valid, "outliers": 0}

    combined_mask = mask_x & mask_y
    if int(combined_mask.sum()) < 3:
        combined_mask = mask_x | mask_y
    inliers = int(combined_mask.sum())
    if inliers < 2:
        return pixels, {"applied": False, "total": total_valid, "inliers": total_valid, "outliers": 0}

    coeff_x_final = _polyfit_with_mask(times_arr, xs_arr, combined_mask, degree_use)
    coeff_y_final = _polyfit_with_mask(times_arr, ys_arr, combined_mask, degree_use)
    if coeff_x_final is None or coeff_y_final is None:
        return pixels, {"applied": False, "total": total_valid, "inliers": total_valid, "outliers": 0}

    fitted_x = np.polyval(coeff_x_final, times_arr)
    fitted_y = np.polyval(coeff_y_final, times_arr)

    coeff_z_final: np.ndarray | None = None
    if np.isfinite(zs_arr).sum() >= 3:
        coeff_z, _, mask_z = _robust_polyfit(times_arr, zs_arr, degree=degree_use)
        if coeff_z is not None:
            combined_z_mask = mask_z & np.isfinite(zs_arr)
            if int(combined_z_mask.sum()) >= 2:
                coeff_z_final = _polyfit_with_mask(times_arr, zs_arr, combined_z_mask, degree_use)
    fitted_z = (
        np.polyval(coeff_z_final, times_arr) if coeff_z_final is not None else zs_arr
    )

    smoothed = [dict(entry) for entry in pixels]
    for idx_local, (x_fit, y_fit, z_fit) in enumerate(zip(fitted_x, fitted_y, fitted_z)):
        target_idx = valid_indices[idx_local]
        entry = smoothed[target_idx]
        if np.isfinite(x_fit):
            entry["x"] = round(float(x_fit), 2)
        if np.isfinite(y_fit):
            entry["y"] = round(float(y_fit), 2)
        if entry.get("z") is not None and np.isfinite(z_fit):
            entry["z"] = round(float(z_fit), 2)

    outliers = total_valid - inliers
    summary = {"applied": True, "total": total_valid, "inliers": inliers, "outliers": outliers}
    return smoothed, summary


def _annotate_sticker_frames(
    frames_dir: str,
    pixels: Sequence[dict[str, float]],
) -> None:
    """Overlay the (potentially smoothed) sticker trajectory onto saved frames."""
    if not frames_dir or not os.path.isdir(frames_dir):
        return
    try:
        entries = [
            (int(name[6:-4]), os.path.join(frames_dir, name))
            for name in os.listdir(frames_dir)
            if name.startswith("frame_") and name.endswith(".png")
        ]
    except Exception:
        return
    if not entries:
        return

    raw_points: list[tuple[float, float]] = []
    for entry in pixels:
        try:
            x_val = float(entry["x"])
            y_val = float(entry["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (np.isfinite(x_val) and np.isfinite(y_val)):
            continue
        raw_points.append((x_val, y_val))
    if not raw_points:
        return

    for _, frame_path in sorted(entries):
        image = cv2.imread(frame_path)
        if image is None:
            continue
        height, width = image.shape[:2]
        path_points = [
            (int(round(x)), int(round(y)))
            for x, y in raw_points
            if 0 <= x < width and 0 <= y < height
        ]
        if not path_points:
            continue
        if len(path_points) >= 2:
            cv2.polylines(
                image,
                [np.array(path_points, dtype=np.int32)],
                isClosed=False,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        else:
            cv2.circle(image, path_points[0], 3, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.imwrite(frame_path, image)



def bbox_to_ball_metrics(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    """Return center, radius and distance estimates for a bounding box."""

    w = float(x2 - x1)
    h = float(y2 - y1)
    raw_radius = (w + h) / 4.0
    radius_for_distance = max(raw_radius, 1e-6)
    cx = float(x1 + x2) / 2.0
    cy = float(y1 + y2) / 2.0
    distance = FOCAL_LENGTH * ACTUAL_BALL_RADIUS / radius_for_distance
    return cx, cy, raw_radius, distance


def bbox_within_image(
    bbox: tuple[float, float, float, float],
    width: float,
    height: float,
    *,
    margin: float = EDGE_MARGIN_PX,
) -> bool:
    """Return ``True`` when ``bbox`` stays inside the image bounds with ``margin`` padding."""

    x1, y1, x2, y2 = bbox
    return (
        x1 > margin
        and y1 > margin
        and x2 < width - margin
        and y2 < height - margin
    )


def preprocess_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Enhance ``frame`` for low light and return both color and gray images."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = CLAHE.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    gray = CLAHE.apply(gray)
    if USE_BLUR:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return enhanced, gray


def blackout_green_pixels(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 40, 40], dtype=np.uint8)
    upper = np.array([90, 255, 255], dtype=np.uint8)
    green_mask = cv2.inRange(hsv, lower, upper)
    if int(np.count_nonzero(green_mask)) == 0:
        return image
    result = image.copy()
    result[green_mask > 0] = 0
    return result


class AdaptiveAlphaMapper:
    """Adaptive, multi-cluster chroma keyer with temporal stabilization.

    Features:
    - Lab a/b modeling with up to K clusters; Mahalanobis thresholds calibrated by
      chi-square coverage for crisp, color-consistent keys.
    - Shadow-aware: uses a/b only; allows large L deviations so shadows/highlights
      don't break the key.
    - Ball midline hard cutoff to avoid club/tee zone during calibration and runtime.
    - Temporal adaptation: softly updates cluster means from background-labeled pixels.
    - Self-correcting tolerance: expands/contracts threshold to maintain background
      coverage and keep foreground visibility.
    - Safety fallbacks: re-center on current frame if key over-suppresses; green key
      fallback if calibration missing.
    """

    def __init__(
        self,
        *,
        target_coverage: float = 0.83,
        k_clusters: int = 3,
        k_max: int = 5,
        silhouette_min: float = 0.10,
        k_penalty: float = 0.08,
        max_samples: int = 250_000,
        min_radius: float = 6.0,
        max_radius: float = 28.0,
        initial_quantile: float = 0.92,
        min_visible_fraction: float = 0.028,
        temporal_alpha: float = 0.02,
        morph_soften: bool = True,
        guard_chroma_thresh: float = 20.0,
        guard_margin: float = 4.0,
    ) -> None:
        self.target_coverage = float(max(0.5, min(0.98, target_coverage)))
        self.k_clusters = int(max(2, k_clusters))
        self.k_max = int(max(self.k_clusters, k_max))
        self.silhouette_min = float(max(0.0, min(0.6, silhouette_min)))
        self.k_penalty = float(max(0.0, min(0.5, k_penalty)))
        self.max_samples = int(max(10_000, max_samples))
        self.min_radius = float(max(1.0, min_radius))
        self.max_radius = float(max(self.min_radius, max_radius))
        self.initial_quantile = float(max(0.8, min(0.995, initial_quantile)))
        self.min_visible_fraction = float(max(0.002, min(0.1, min_visible_fraction)))
        self.temporal_alpha = float(max(0.0, min(0.25, temporal_alpha)))
        self.morph_soften = bool(morph_soften)
        self.guard_chroma_thresh = float(max(0.0, guard_chroma_thresh))
        self.guard_margin = float(max(0.0, guard_margin))

        # Single-cluster fallback (compatibility)
        self.bg_center_ab: np.ndarray | None = None
        self.bg_radius: float | None = None
        # Multi-cluster parameters (preferred)
        self.centers_ab: np.ndarray | None = None      # (k,2)
        self.cov_inv: list[np.ndarray] | None = None   # k x (2,2)
        self.cov_det: np.ndarray | None = None         # (k,)
        self.threshold_mah2: float | None = None       # chi-square threshold (squared)
        self.threshold_scale: float = 1.0
        self.cluster_weights: np.ndarray | None = None # (k,)
        self.guarded_indices: list[int] = []
        self.ball_mid_y: int | None = None
        self.calib_frame_index: int | None = None
        self.meta: dict[str, object] = {}

    @staticmethod
    def _to_lab_ab(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        a = lab[:, :, 1].astype(np.float32)
        b = lab[:, :, 2].astype(np.float32)
        return a, b

    @staticmethod
    def _sample_ab(a: np.ndarray, b: np.ndarray, mask: np.ndarray, max_samples: int) -> np.ndarray:
        ys, xs = np.where(mask)
        if ys.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        if ys.size > max_samples:
            idx = np.random.choice(ys.size, size=max_samples, replace=False)
            ys = ys[idx]
            xs = xs[idx]
        samples = np.column_stack((a[ys, xs], b[ys, xs])).astype(np.float32)
        return samples

    @staticmethod
    def _calc_cov_inv(samples: np.ndarray, center: np.ndarray, eps: float = 1.5) -> tuple[np.ndarray, float]:
        if samples.size == 0:
            return np.eye(2, dtype=np.float32), 1.0
        diffs = samples - center[None, :]
        cov = np.cov(diffs.T)
        if cov.shape != (2, 2):
            cov = np.eye(2, dtype=np.float32)
        cov = cov.astype(np.float32)
        cov += np.eye(2, dtype=np.float32) * float(eps)
        try:
            cov_inv = np.linalg.inv(cov).astype(np.float32)
            det = float(max(1e-12, np.linalg.det(cov)))
        except Exception:
            cov_inv = np.linalg.pinv(cov).astype(np.float32)
            det = 1.0
        return cov_inv, det

    @staticmethod
    def _mah2_grid(ab: np.ndarray, center: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
        dif = ab - center.reshape(1, 1, 2)
        left = np.einsum('ijk,kl->ijl', dif, cov_inv)
        d2 = np.einsum('ijk,ijk->ij', left, dif)
        return d2

    @staticmethod
    def _chi2_quantile_2df(p: float) -> float:
        table = [
            (0.90, 4.605),
            (0.95, 5.991),
            (0.975, 7.378),
            (0.98, 7.824),
            (0.99, 9.210),
            (0.995, 10.597),
            (0.999, 13.816),
        ]
        p = float(np.clip(p, 0.90, 0.999))
        for i in range(len(table) - 1):
            p0, x0 = table[i]
            p1, x1 = table[i + 1]
            if p0 <= p <= p1:
                t = (p - p0) / max(1e-9, (p1 - p0))
                return x0 + t * (x1 - x0)
        return table[-1][1]

    @staticmethod
    def _kmeans_centroid_ab(samples: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if samples.size == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.int32)
        Z = samples.astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
        attempts = 2
        flags = cv2.KMEANS_PP_CENTERS
        compactness, labels, centers = cv2.kmeans(Z, k, None, criteria, attempts, flags)
        labels = labels.flatten().astype(np.int32)
        centers = centers.astype(np.float32)
        return centers, labels

    @staticmethod
    def _kmeans_run(samples: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        if samples.size == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.int32), 0.0, np.zeros(0)
        Z = samples.astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
        attempts = 3
        flags = cv2.KMEANS_PP_CENTERS
        compactness, labels, centers = cv2.kmeans(Z, k, None, criteria, attempts, flags)
        labels = labels.flatten().astype(np.int32)
        centers = centers.astype(np.float32)
        return centers, labels, float(compactness), Z

    def _select_k(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        N = samples.shape[0]
        if N < 400 or self.k_max <= 1:
            centers, labels, _, _ = self._kmeans_run(samples, 1)
            return centers, labels
        best_idx = 1
        best_centers = None
        best_labels = None
        best_j = float('inf')
        sse1 = None
        silhouettes: dict[int, float] = {}
        js: dict[int, float] = {}
        centers_cache: dict[int, np.ndarray] = {}
        labels_cache: dict[int, np.ndarray] = {}
        sse_cache: dict[int, float] = {}
        for k in range(1, self.k_max + 1):
            centers, labels, compactness, Z = self._kmeans_run(samples, k)
            centers_cache[k] = centers
            labels_cache[k] = labels
            # OpenCV returns sum of squared distances (SSE)
            sse_cache[k] = float(compactness)
            if k == 1:
                sse1 = float(compactness) if compactness > 0 else 1.0
                silhouettes[k] = -1.0
            else:
                # Cluster-level silhouette using centroid spacing and intra-cluster mean distance
                # Compute mean intra distance per cluster
                intra = []
                counts = []
                for ci in range(k):
                    sel = labels == ci
                    cnt = int(np.count_nonzero(sel))
                    counts.append(cnt)
                    if cnt == 0:
                        intra.append(0.0)
                        continue
                    pts = Z[sel]
                    d = np.linalg.norm(pts - centers[ci:ci+1, :], axis=1)
                    intra.append(float(np.mean(d)))
                counts = np.array(counts, dtype=np.float32)
                # Distances between centers
                if k > 1:
                    cen_diff = centers[:, None, :] - centers[None, :, :]
                    cen_dist = np.linalg.norm(cen_diff, axis=2) + np.eye(k, dtype=np.float32) * 1e9
                    nearest = np.min(cen_dist, axis=1)
                else:
                    nearest = np.zeros(1, dtype=np.float32)
                sil = []
                for ci in range(k):
                    a = float(intra[ci])
                    b = float(nearest[ci]) if k > 1 else 0.0
                    if (a <= 1e-6 and b <= 1e-6) or max(a, b) <= 1e-6:
                        s = 0.0
                    else:
                        s = (b - a) / max(a, b)
                    sil.append(s)
                if counts.sum() > 0:
                    silhouettes[k] = float(np.sum((counts / counts.sum()) * np.array(sil, dtype=np.float32)))
                else:
                    silhouettes[k] = 0.0

        # Normalize SSE by K=1 and add small penalty per extra cluster
        for k in range(1, self.k_max + 1):
            sse = sse_cache[k]
            sse_norm = (sse / sse1) if sse1 and sse1 > 0 else 1.0
            js[k] = sse_norm + self.k_penalty * (k - 1)

        # Choose smallest K close to the best j and with silhouette >= min
        best_k = min(js, key=lambda kk: js[kk])
        best_j = js[best_k]
        candidate_ks = [k for k in range(1, self.k_max + 1) if js[k] <= best_j * 1.06]
        # Prefer smallest with acceptable silhouette
        chosen = None
        for k in sorted(candidate_ks):
            if k == 1 or silhouettes.get(k, -1.0) >= self.silhouette_min:
                chosen = k
                break
        if chosen is None:
            chosen = best_k
        best_centers = centers_cache[chosen]
        best_labels = labels_cache[chosen]
        return best_centers, best_labels

    @staticmethod
    def _quantile_radius(center: np.ndarray, samples: np.ndarray, q: float) -> float:
        if samples.size == 0:
            return 0.0
        diffs = samples - center[None, :]
        d = np.linalg.norm(diffs, axis=1)
        q = float(np.clip(q, 0.5, 0.999))
        return float(np.quantile(d, q))

    def calibrate(
        self,
        frame: np.ndarray,
        *,
        ball_bbox: tuple[float, float, float, float] | None,
        frame_index: int | None = None,
        save_dir: str | None = None,
    ) -> None:
        if frame is None or frame.size == 0:
            return
        h, w = frame.shape[:2]
        # Determine the ball mid Y for the line to blackout below
        mid_y: int | None = None
        if ball_bbox is not None:
            try:
                _, y1, _, y2 = ball_bbox
                mid_y = int(np.clip(round((float(y1) + float(y2)) / 2.0), 0, h - 1))
            except Exception:
                mid_y = None
        if mid_y is None:
            mid_y = int(h * 0.6)
        self.ball_mid_y = mid_y
        self.calib_frame_index = frame_index

        # Build mask for the region above ball midline
        region_mask = np.zeros((h, w), dtype=np.uint8)
        region_mask[: mid_y + 1, :] = 1
        a, b = self._to_lab_ab(frame)
        samples = self._sample_ab(a, b, region_mask.astype(bool), self.max_samples)
        if samples.size == 0:
            return

        # Cluster in a/b space to find background; prefer multi-cluster w/ Mahalanobis
        # Determine K automatically (prefers fewer clusters)
        centers, labels = self._select_k(samples)
        k = max(1, centers.shape[0])
        if centers.size == 0:
            return

        # Guard clusters near neutral chroma (likely metallic grey club)
        chroma = np.linalg.norm(centers, axis=1)
        guarded = chroma <= (self.guard_chroma_thresh)
        self.guarded_indices = [int(i) for i in np.where(guarded)[0].tolist()]

        cov_inv_list: list[np.ndarray] = []
        cov_det_list: list[float] = []
        weights = []
        for c in range(k):
            samp_c = samples[labels == c]
            cov_inv_c, det_c = self._calc_cov_inv(samp_c, centers[c])
            cov_inv_list.append(cov_inv_c)
            cov_det_list.append(det_c)
            weights.append(float(samp_c.shape[0]))
        weights_arr = np.array(weights, dtype=np.float32)
        if weights_arr.sum() > 0:
            weights_arr /= float(weights_arr.sum())

        mah2_thresh = self._chi2_quantile_2df(self.target_coverage)

        ab_all = np.dstack((a, b)).astype(np.float32)
        y_coords = np.arange(h)[:, None]
        mask_union = np.zeros((h, w), dtype=bool)
        for c in range(k):
            # Skip guarded clusters when forming background mask
            if c in self.guarded_indices:
                continue
            d2 = self._mah2_grid(ab_all, centers[c], cov_inv_list[c])
            mask_union |= (d2 <= mah2_thresh)
        mask_union &= (y_coords <= mid_y)
        coverage = float(np.count_nonzero(mask_union)) / float((mid_y + 1) * max(1, w))

        # Persist multi-cluster model
        self.centers_ab = centers.astype(np.float32)
        self.cov_inv = [c.copy() for c in cov_inv_list]
        self.cov_det = np.array(cov_det_list, dtype=np.float32)
        self.cluster_weights = weights_arr
        self.threshold_mah2 = float(mah2_thresh)
        self.threshold_scale = 1.0

        # Single-cluster fallback parameters
        # Prefer dominant non-guarded cluster
        if weights_arr.size:
            order = np.argsort(weights_arr)[::-1]
            dom_idx = int(next((int(i) for i in order if int(i) not in self.guarded_indices), int(order[0])))
        else:
            dom_idx = 0
        dom_samples = samples[labels == dom_idx]
        dom_center = centers[dom_idx]
        radius_q = self._quantile_radius(dom_center, dom_samples, self.initial_quantile)
        radius = float(np.clip(radius_q, self.min_radius, self.max_radius))
        self.bg_center_ab = dom_center.astype(np.float32)
        self.bg_radius = float(radius)

        self.meta = {
            "centers_ab": self.centers_ab.tolist() if self.centers_ab is not None else None,
            "cov_det": self.cov_det.tolist() if self.cov_det is not None else None,
            "weights": self.cluster_weights.tolist() if self.cluster_weights is not None else None,
            "mah2_threshold": float(self.threshold_mah2) if self.threshold_mah2 is not None else None,
            "fallback_center_ab": [float(self.bg_center_ab[0]), float(self.bg_center_ab[1])] if self.bg_center_ab is not None else None,
            "fallback_radius_ab": float(self.bg_radius) if self.bg_radius is not None else None,
            "coverage": float(coverage),
            "clusters": int(k),
            "guarded_indices": self.guarded_indices,
            "ball_mid_y": int(mid_y),
            "frame_index": (int(frame_index) if frame_index is not None else None),
        }

        # Persist calibration frame and metadata for debugging
        if save_dir:
            try:
                os.makedirs(save_dir, exist_ok=True)
                if frame_index is None:
                    fname = os.path.join(save_dir, f"calibration.png")
                else:
                    fname = os.path.join(save_dir, f"calibration_{int(frame_index):06d}.png")
                cv2.imwrite(fname, frame)
                with open(os.path.join(save_dir, "alpha_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(self.meta, f, indent=2)
            except Exception:
                pass

    def apply(
        self,
        image: np.ndarray,
        *,
        motion_guard: dict[str, object] | None = None,
        min_visible_fraction: float | None = None,
    ) -> np.ndarray:
        """Apply adaptive background black-out to image with optional motion guard.

        Behavior:
        - Black out background pixels close to the calibrated a/b center within radius.
        - Also black out everything below the calibration ball midpoint, if available.
        - If output has too few visible pixels, gradually shrink radius to reveal more.
        - Protects motion-guarded regions/colors from black-out when provided.
        """
        if image is None or image.size == 0:
            return image
        if (self.centers_ab is None or self.cov_inv is None or self.threshold_mah2 is None):
            return blackout_green_pixels(image)

        h, w = image.shape[:2]
        a, b = self._to_lab_ab(image)
        ab_all = np.dstack((a, b)).astype(np.float32)
        y_coords = np.arange(h)[:, None]

        if self.ball_mid_y is not None:
            mid_y = int(self.ball_mid_y)
            below = y_coords > mid_y
            above = y_coords <= mid_y
            area_above = max(1, int(np.count_nonzero(above)))
        else:
            mid_y = None
            below = None
            above = np.ones((h, w), dtype=bool)
            area_above = max(1, h * w)

        guard_total: np.ndarray | None = None
        guard_neighbors: np.ndarray | None = None
        motion_mean_ab: np.ndarray | None = None
        color_tol: float | None = None
        if motion_guard:
            guard_masks: list[np.ndarray] = []
            guard_mask_val = motion_guard.get("mask")
            if guard_mask_val is not None:
                mask_bool = np.asarray(guard_mask_val, dtype=bool)
                if mask_bool.shape == (h, w):
                    guard_masks.append(mask_bool)
                    guard_neighbors = mask_bool
            neighbor_mask_val = motion_guard.get("neighbor_mask")
            if neighbor_mask_val is not None:
                neighbor_bool = np.asarray(neighbor_mask_val, dtype=bool)
                if neighbor_bool.shape == (h, w):
                    guard_neighbors = neighbor_bool
                    guard_masks.append(neighbor_bool)
            fast_mask_val = motion_guard.get("fast_mask")
            if fast_mask_val is not None:
                fast_bool = np.asarray(fast_mask_val, dtype=bool)
                if fast_bool.shape == (h, w):
                    guard_masks.append(fast_bool)
            mean_ab_val = motion_guard.get("mean_ab")
            if mean_ab_val is not None:
                motion_mean_ab = np.array(mean_ab_val, dtype=np.float32).reshape(2)
                color_tol = motion_guard.get("color_radius")
            if guard_neighbors is None and guard_masks:
                guard_neighbors = guard_masks[0]
            if motion_mean_ab is not None:
                try:
                    color_dist = np.sqrt(
                        (ab_all[:, :, 0] - motion_mean_ab[0]) ** 2
                        + (ab_all[:, :, 1] - motion_mean_ab[1]) ** 2
                    )
                    tol = float(color_tol) if color_tol is not None else (self.guard_chroma_thresh + self.guard_margin * 2.0)
                    color_guard = color_dist <= (tol + self.guard_margin)
                    if guard_neighbors is not None:
                        color_guard = np.logical_and(color_guard, guard_neighbors)
                    guard_masks.append(color_guard)
                except Exception:
                    pass
            if guard_masks:
                guard_total = guard_masks[0].astype(bool)
                for extra in guard_masks[1:]:
                    guard_total = np.logical_or(guard_total, np.asarray(extra, dtype=bool))

        global_guard: np.ndarray | None = None
        try:
            chroma = np.sqrt(ab_all[:, :, 0] ** 2 + ab_all[:, :, 1] ** 2)
            global_guard = chroma <= (self.guard_chroma_thresh + self.guard_margin)
            if mid_y is not None:
                global_guard = np.logical_and(global_guard, above)
        except Exception:
            global_guard = None

        def enforce_guards(mask: np.ndarray) -> np.ndarray:
            guarded = mask
            if global_guard is not None:
                guarded = np.logical_and(guarded, np.logical_not(global_guard))
            if guard_total is not None:
                guarded = np.logical_and(guarded, np.logical_not(guard_total))
            return guarded

        def compute_background_mask(scale: float) -> tuple[np.ndarray, np.ndarray]:
            thresh_val = float(self.threshold_mah2) * float(scale)
            mask = np.zeros((h, w), dtype=bool)
            for i, center in enumerate(self.centers_ab):
                if i in self.guarded_indices:
                    continue
                d2 = self._mah2_grid(ab_all, center, self.cov_inv[i])
                mask |= (d2 <= thresh_val)
            base_mask = mask.copy()
            if below is not None:
                mask = np.logical_or(mask, below)
            mask = enforce_guards(mask)
            return mask, base_mask

        bg_mask, base_mask = compute_background_mask(self.threshold_scale)
        result = image.copy()
        result[bg_mask] = 0

        if self.morph_soften:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            _, hard = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            hard = cv2.morphologyEx(
                hard,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            hard = cv2.morphologyEx(
                hard,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
            soft = cv2.GaussianBlur(hard, (0, 0), 1.2)
            _, hard = cv2.threshold(soft, 40, 255, cv2.THRESH_BINARY)
            result = cv2.bitwise_and(image, image, mask=hard)

        try:
            coverage_now = float(np.count_nonzero(base_mask & above)) / float(area_above)
            if coverage_now < self.target_coverage * 0.85:
                self.threshold_scale = min(1.8, self.threshold_scale * 1.06)
                bg_mask, base_mask = compute_background_mask(self.threshold_scale)
                result = image.copy()
                result[bg_mask] = 0
        except Exception:
            pass

        visible_frac_target = float(self.min_visible_fraction if min_visible_fraction is None else min_visible_fraction)
        for _ in range(3):
            non_black = int(np.count_nonzero(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)))
            total = int(h * w)
            if total <= 0:
                break
            visible_frac = non_black / float(total)
            if visible_frac >= visible_frac_target:
                break
            self.threshold_scale = max(0.55, self.threshold_scale * 0.88)
            bg_mask, base_mask = compute_background_mask(self.threshold_scale)
            result = image.copy()
            result[bg_mask] = 0

        try:
            non_black = int(np.count_nonzero(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)))
            total = int(h * w)
            if total > 0 and (non_black / float(total)) < visible_frac_target:
                region_mask = np.ones((h, w), dtype=bool)
                if mid_y is not None:
                    region_mask[y_coords > mid_y] = False
                samples2 = self._sample_ab(a, b, region_mask, min(self.max_samples, 150_000))
                if samples2.size >= 2000:
                    new_center = samples2.mean(axis=0)
                    diffs2 = samples2 - new_center[None, :]
                    d2 = np.linalg.norm(diffs2, axis=1)
                    r2 = float(np.quantile(d2, 0.90))
                    r2 = float(np.clip(r2, self.min_radius, self.max_radius))
                    self.bg_center_ab = new_center.astype(np.float32)
                    self.bg_radius = r2
                    dist2 = np.linalg.norm(ab_all - self.bg_center_ab.reshape(1, 1, 2), axis=2)
                    mask2 = dist2 <= self.bg_radius
                    base_mask = mask2.copy()
                    if below is not None:
                        mask2 = np.logical_or(mask2, below)
                    mask2 = enforce_guards(mask2)
                    bg_mask = mask2
                    result = image.copy()
                    result[bg_mask] = 0
        except Exception:
            pass

        try:
            if self.temporal_alpha > 1e-6 and self.centers_ab is not None and self.cov_inv is not None:
                d2_stack = []
                for i, center in enumerate(self.centers_ab):
                    d2_stack.append(self._mah2_grid(ab_all, center, self.cov_inv[i]))
                d2_stack = np.stack(d2_stack, axis=-1)
                assign = np.argmin(d2_stack, axis=-1)

                yx_bg = np.where(bg_mask)
                if yx_bg[0].size > 0:
                    if yx_bg[0].size > 20000:
                        idx = np.random.choice(yx_bg[0].size, size=20000, replace=False)
                        ys = yx_bg[0][idx]
                        xs = yx_bg[1][idx]
                    else:
                        ys, xs = yx_bg
                    ab_bg = ab_all[ys, xs, :]
                    asg = assign[ys, xs]
                    for i in range(self.centers_ab.shape[0]):
                        sel = (asg == i)
                        if int(np.count_nonzero(sel)) < 50:
                            continue
                        mean_i = ab_bg[sel].mean(axis=0)
                        self.centers_ab[i] = (1.0 - self.temporal_alpha) * self.centers_ab[i] + self.temporal_alpha * mean_i
        except Exception:
            pass

        return result


class MotionGuardTracker:
    """Track motion during the swing to guard club pixels from alpha mapping."""

    def __init__(
        self,
        *,
        history: int = 150,
        var_threshold: float = 18.0,
        min_area: int = 600,
        neighbor_dilate: int = 6,
        color_quantile: float = 0.90,
        fast_percentile: float = 85.0,
        flow_downscale: float = 0.6,
    ) -> None:
        self.history = max(30, int(history))
        self.var_threshold = float(max(4.0, var_threshold))
        self.min_area = max(120, int(min_area))
        self.neighbor_dilate = max(1, int(neighbor_dilate))
        self.color_quantile = float(max(0.5, min(0.99, color_quantile)))
        self.fast_percentile = float(max(60.0, min(99.0, fast_percentile)))
        self.flow_downscale = float(np.clip(flow_downscale, 0.3, 1.0))

        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=False,
        )
        self._prev_gray: np.ndarray | None = None
        self._prev_flow_gray: np.ndarray | None = None
        self._prev_centroid: tuple[float, float] | None = None
        self._prev_velocity: tuple[float, float] | None = None
        self._last_guard: dict[str, object] | None = None

    def reset(self) -> None:
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=False,
        )
        self._prev_gray = None
        self._prev_flow_gray = None
        self._prev_centroid = None
        self._prev_velocity = None
        self._last_guard = None

    def _predict_guard_from_history(self, shape: tuple[int, int]) -> dict[str, object] | None:
        if self._last_guard is None or self._prev_velocity is None:
            return None
        mask_prev = self._last_guard.get("mask")
        if mask_prev is None:
            return None
        dx, dy = self._prev_velocity
        if abs(dx) < 0.3 and abs(dy) < 0.3:
            return None
        h, w = shape
        M = np.float32([[1, 0, dx], [0, 1, dy]])

        def _warp(src: np.ndarray | None) -> np.ndarray | None:
            if src is None:
                return None
            warped = cv2.warpAffine(
                src.astype(np.uint8),
                M,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            return warped.astype(bool)

        guard_mask = _warp(mask_prev)
        if guard_mask is None or not np.any(guard_mask):
            return None
        neighbor_mask = _warp(self._last_guard.get("neighbor_mask"))
        core_mask = _warp(self._last_guard.get("core_mask"))
        fast_mask = _warp(self._last_guard.get("fast_mask"))
        guard = {
            "mask": guard_mask,
            "core_mask": core_mask if core_mask is not None else guard_mask,
            "neighbor_mask": neighbor_mask if neighbor_mask is not None else guard_mask,
            "fast_mask": fast_mask,
            "mean_ab": self._last_guard.get("mean_ab"),
            "color_radius": self._last_guard.get("color_radius"),
            "centroid": self._last_guard.get("centroid"),
            "area": self._last_guard.get("area"),
        }
        self._last_guard = guard
        return guard

    def _compute_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray | None:
        if prev_gray is None or curr_gray is None:
            return None
        h, w = curr_gray.shape
        if self.flow_downscale < 0.99:
            dw = max(8, int(round(w * self.flow_downscale)))
            dh = max(8, int(round(h * self.flow_downscale)))
            prev_small = cv2.resize(prev_gray, (dw, dh), interpolation=cv2.INTER_AREA)
            curr_small = cv2.resize(curr_gray, (dw, dh), interpolation=cv2.INTER_AREA)
            flow = cv2.calcOpticalFlowFarneback(
                prev_small,
                curr_small,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=21,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) / max(1e-3, self.flow_downscale)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                curr_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=21,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mag_masked = mag[mask]
        if mag_masked.size == 0:
            return None
        thresh = float(np.percentile(mag_masked, self.fast_percentile))
        thresh = max(thresh, 0.6)
        fast = np.zeros_like(mag, dtype=bool)
        fast[mask] = mag_masked >= thresh
        return fast

    def update(self, frame: np.ndarray, *, in_window: bool) -> dict[str, object] | None:
        if frame is None or frame.size == 0:
            return None
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(frame_gray, (5, 5), 0)
        fg_mask = self._bg_subtractor.apply(frame)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        if self._prev_gray is not None:
            diff = cv2.absdiff(blurred, self._prev_gray)
            diff = cv2.GaussianBlur(diff, (5, 5), 0)
            _, diff_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
            fg_mask = cv2.bitwise_or(fg_mask, diff_mask)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        self._prev_gray = blurred

        if not in_window:
            self._prev_flow_gray = blurred
            self._last_guard = None
            self._prev_velocity = None
            self._prev_centroid = None
            return None

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask)
        if num_labels <= 1:
            guard = self._predict_guard_from_history(frame.shape[:2])
            self._prev_flow_gray = blurred
            return guard

        areas = stats[1:, cv2.CC_STAT_AREA]
        idx_rel = int(np.argmax(areas))
        area = float(areas[idx_rel])
        if area < self.min_area:
            guard = self._predict_guard_from_history(frame.shape[:2])
            self._prev_flow_gray = blurred
            return guard
        idx = idx_rel + 1

        component_mask = (labels == idx).astype(np.uint8)
        component_mask = cv2.morphologyEx(component_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        guard_mask = cv2.dilate(
            component_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.neighbor_dilate, self.neighbor_dilate)),
            iterations=1,
        )
        neighbor_mask = cv2.dilate(
            guard_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.neighbor_dilate, self.neighbor_dilate)),
            iterations=1,
        )

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        a_chan = lab[:, :, 1].astype(np.float32)
        b_chan = lab[:, :, 2].astype(np.float32)
        guard_bool = guard_mask.astype(bool)
        if not np.any(guard_bool):
            self._prev_flow_gray = blurred
            self._last_guard = None
            return None

        mean_a = float(a_chan[guard_bool].mean())
        mean_b = float(b_chan[guard_bool].mean())
        motion_mean_ab = np.array([mean_a, mean_b], dtype=np.float32)
        diffs = np.sqrt((a_chan[guard_bool] - mean_a) ** 2 + (b_chan[guard_bool] - mean_b) ** 2)
        if diffs.size:
            color_radius = float(np.quantile(diffs, self.color_quantile))
        else:
            color_radius = 15.0
        color_radius = max(6.0, color_radius)

        fast_mask = None
        if self._prev_flow_gray is not None:
            fast_mask = self._compute_flow(self._prev_flow_gray, blurred, guard_bool)

        centroid = centroids[idx]
        centroid_xy = (float(centroid[0]), float(centroid[1]))
        velocity = None
        if self._prev_centroid is not None:
            velocity = (
                centroid_xy[0] - self._prev_centroid[0],
                centroid_xy[1] - self._prev_centroid[1],
            )
        self._prev_centroid = centroid_xy
        if velocity is not None:
            self._prev_velocity = velocity

        self._prev_flow_gray = blurred

        guard_info: dict[str, object] = {
            "mask": guard_bool,
            "core_mask": component_mask.astype(bool),
            "neighbor_mask": neighbor_mask.astype(bool),
            "fast_mask": fast_mask.astype(bool) if isinstance(fast_mask, np.ndarray) else None,
            "mean_ab": motion_mean_ab.tolist(),
            "color_radius": float(color_radius),
            "centroid": (centroid_xy[0], centroid_xy[1]),
            "area": area,
        }

        self._last_guard = guard_info
        return guard_info




class MotionForegroundExtractor:
    """Build a background model and extract foreground masks around the club and ball."""

    def __init__(
        self,
        *,
        min_background_frames: int = 15,
        learning_rate: float = 0.02,
        min_component_area: int = 60,
        max_component_area: int = 150_000,
        history_length: int = 3,
        persistence_threshold: int = 2,
        median_kernel_size: int = 5,
    ) -> None:
        self.min_background_frames = max(1, int(min_background_frames))
        self.learning_rate = float(learning_rate)
        self.min_component_area = max(1, int(min_component_area))
        self.max_component_area = max(self.min_component_area, int(max_component_area))
        self.history_length = max(1, int(history_length))
        self.persistence_threshold = max(1, min(self.history_length, int(persistence_threshold)))
        kernel = max(1, int(median_kernel_size))
        if kernel % 2 == 0:
            kernel += 1
        self.median_kernel_size = kernel
        self._accumulator: np.ndarray | None = None
        self._frames_seen = 0
        self._history: deque[np.ndarray] = deque(maxlen=self.history_length)
        self._prev_mask: np.ndarray | None = None
        self._prev_centroid: tuple[float, float] | None = None
        self._prev_motion_vec: tuple[float, float] | None = None
        self._template_contour: np.ndarray | None = None
        self._template_area: float = 0.0
        self._template_size: tuple[int, int] = (0, 0)

    def reset(self) -> None:
        self._accumulator = None
        self._frames_seen = 0
        self._history.clear()
        self._prev_mask = None
        self._prev_centroid = None
        self._prev_motion_vec = None
        self._template_contour = None
        self._template_area = 0.0
        self._template_size = (0, 0)

    @staticmethod
    def _compute_centroid(mask: np.ndarray | None) -> tuple[float, float] | None:
        if mask is None or mask.size == 0:
            return None
        moments = cv2.moments(mask, binaryImage=True)
        area = moments["m00"]
        if area <= 1e-6:
            return None
        cx = float(moments["m10"] / area)
        cy = float(moments["m01"] / area)
        return cx, cy

    @staticmethod
    def _extract_primary_contour(mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None or mask.size == 0:
            return None
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def _maybe_update_template(
        self,
        mask: np.ndarray,
        centroid: tuple[float, float],
        area: float,
    ) -> None:
        if area < CLUB_TEMPLATE_MIN_AREA_PX:
            return
        contour = self._extract_primary_contour(mask)
        if contour is None:
            return
        update_required = self._template_contour is None or area > self._template_area * CLUB_TEMPLATE_UPDATE_MARGIN
        if not update_required:
            return
        contour_f = contour.reshape(-1, 2).astype(np.float32)
        centered = contour_f - np.array([[centroid[0], centroid[1]]], dtype=np.float32)
        self._template_contour = centered
        self._template_area = float(area)
        x, y, w, h = cv2.boundingRect(contour)
        self._template_size = (int(w), int(h))

    def _render_template(
        self,
        frame_shape: tuple[int, int],
        centroid: tuple[float, float] | None,
    ) -> np.ndarray | None:
        if self._template_contour is None or centroid is None:
            return None
        h, w = frame_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        contour_shifted = np.round(
            self._template_contour + np.array([[centroid[0], centroid[1]]], dtype=np.float32)
        ).astype(np.int32)
        cv2.fillPoly(mask, [contour_shifted.reshape(-1, 1, 2)], 255)
        return mask

    def _augment_with_motion_template(
        self,
        base_mask: np.ndarray,
        centroid: tuple[float, float] | None,
        movement_vec: tuple[float, float] | None,
    ) -> np.ndarray:
        if self._template_contour is None or self._template_area <= 0.0:
            return base_mask

        frame_shape = base_mask.shape
        anchor = centroid or self._prev_centroid
        if anchor is None:
            return base_mask

        template_masks: list[np.ndarray] = []
        anchor_template = self._render_template(frame_shape, anchor)
        if anchor_template is not None:
            template_masks.append(anchor_template)

        if (
            movement_vec is not None
            and self._template_size != (0, 0)
            and self._template_area > 0.0
        ):
            current_area = float(np.count_nonzero(base_mask))
            ratio = current_area / float(self._template_area)
            if ratio < CLUB_TEMPLATE_RECOVERY_RATIO:
                dx, dy = movement_vec
                magnitude = math.hypot(dx, dy)
                if magnitude > 1e-3:
                    deficit = max(0.0, CLUB_TEMPLATE_RECOVERY_RATIO - ratio)
                    scale = deficit / CLUB_TEMPLATE_RECOVERY_RATIO
                    max_dim = float(max(self._template_size))
                    extent = scale * max_dim
                    extent = max(extent, CLUB_TEMPLATE_MIN_EXTENSION_PX)
                    extent = min(extent, CLUB_TEMPLATE_MAX_EXTENSION_PX)
                    if extent > 0.5:
                        ux = dx / magnitude
                        uy = dy / magnitude
                        lead_centroid = (
                            anchor[0] + ux * extent,
                            anchor[1] + uy * extent,
                        )
                        lead_template = self._render_template(frame_shape, lead_centroid)
                        if lead_template is not None:
                            template_masks.append(lead_template)

        if not template_masks:
            return base_mask

        template_union = template_masks[0].copy()
        for extra in template_masks[1:]:
            template_union = cv2.bitwise_or(template_union, extra)

        combined = cv2.bitwise_or(base_mask, template_union)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        return combined

    def _predict_from_template(self, frame_shape: tuple[int, int]) -> np.ndarray | None:
        if self._template_contour is None or self._prev_centroid is None:
            return None
        target = self._prev_centroid
        if self._prev_motion_vec is not None:
            target = (
                target[0] + self._prev_motion_vec[0],
                target[1] + self._prev_motion_vec[1],
            )
        predicted = self._render_template(frame_shape, target)
        if predicted is None:
            return None
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        predicted = cv2.morphologyEx(predicted, cv2.MORPH_CLOSE, kernel)
        return predicted

    def update_background(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            return
        frame_f32 = frame.astype(np.float32)
        if self._accumulator is None:
            self._accumulator = frame_f32.copy()
        else:
            cv2.accumulateWeighted(frame_f32, self._accumulator, self.learning_rate)
        self._frames_seen += 1

    @property
    def ready(self) -> bool:
        return self._accumulator is not None and self._frames_seen >= self.min_background_frames

    def _build_support_mask(
        self,
        shape: tuple[int, int],
        ball_bbox: tuple[float, float, float, float] | None,
        dot_points: Sequence[tuple[float, float]] | None,
        club_center: tuple[float, float] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        support = np.zeros(shape, dtype=np.uint8)
        ball_mask = np.zeros(shape, dtype=np.uint8)
        if dot_points:
            for point in dot_points:
                try:
                    u = int(round(float(point[0])))
                    v = int(round(float(point[1])))
                except (TypeError, ValueError):
                    continue
                cv2.circle(support, (u, v), 12, 255, -1, lineType=cv2.LINE_AA)
        if club_center is not None:
            try:
                u = int(round(float(club_center[0])))
                v = int(round(float(club_center[1])))
            except (TypeError, ValueError):
                pass
            else:
                cv2.circle(support, (u, v), 18, 255, -1, lineType=cv2.LINE_AA)
        if ball_bbox is not None:
            x1, y1, x2, y2 = ball_bbox
            cx = int(round((x1 + x2) / 2.0))
            cy = int(round((y1 + y2) / 2.0))
            radius = int(round(max(x2 - x1, y2 - y1) * 0.65))
            radius = max(radius, 6)
            cv2.circle(ball_mask, (cx, cy), radius, 255, -1, lineType=cv2.LINE_AA)
        if np.count_nonzero(support):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            support = cv2.dilate(support, kernel, iterations=1)
        if np.count_nonzero(ball_mask):
            kernel_ball = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            ball_mask = cv2.dilate(ball_mask, kernel_ball, iterations=1)
        return support, ball_mask

    def _smooth_primary_region(
        self,
        mask: np.ndarray,
        brightness: np.ndarray | None,
    ) -> np.ndarray | None:
        if mask is None or mask.size == 0:
            return None
        binary = (mask > 0).astype(np.uint8)
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=2,
        )
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )

        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        core = None
        for radius in (10.0, 8.0, 6.0, 4.0):
            candidate = (dist >= radius).astype(np.uint8)
            if int(candidate.sum()) > 0:
                core = candidate
                break
        if core is None:
            core = binary.copy()

        core = cv2.morphologyEx(
            core,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        core = cv2.dilate(
            core,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
            iterations=1,
        )

        soft = cv2.GaussianBlur(core.astype(np.float32), (0, 0), sigmaX=8.0, sigmaY=8.0)
        if soft.max() > 0:
            soft = cv2.normalize(soft, None, 0.0, 255.0, cv2.NORM_MINMAX)
        soft = soft.astype(np.uint8)
        _, refined = cv2.threshold(soft, 120, 255, cv2.THRESH_BINARY)
        refined = cv2.morphologyEx(
            refined,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )
        refined = cv2.GaussianBlur(refined, (9, 9), 0)
        _, refined = cv2.threshold(refined, 1, 255, cv2.THRESH_BINARY)

        dilated_original = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)),
            iterations=1,
        )
        refined = cv2.bitwise_and(refined, dilated_original)

        if brightness is not None and brightness.size:
            if not np.any((refined > 0) & brightness):
                return None
        return refined if int(np.count_nonzero(refined)) else None

    def make_mask(
        self,
        frame: np.ndarray,
        *,
        ball_bbox: tuple[float, float, float, float] | None = None,
        dot_points: Sequence[tuple[float, float]] | None = None,
        club_center: tuple[float, float] | None = None,
    ) -> np.ndarray | None:
        height, width = frame.shape[:2]
        support, ball_support = self._build_support_mask((height, width), ball_bbox, dot_points, club_center)
        support_pixels = int(np.count_nonzero(support))
        ball_pixels = int(np.count_nonzero(ball_support))
        if not self.ready:
            if support_pixels or ball_pixels:
                preliminary = cv2.bitwise_or(support, ball_support)
                if ball_pixels:
                    preliminary = cv2.bitwise_and(preliminary, cv2.bitwise_not(ball_support))
                return preliminary if int(np.count_nonzero(preliminary)) else None
            return None

        if self._accumulator is None or self._accumulator.shape != frame.shape:
            self.reset()
            if support_pixels or ball_pixels:
                preliminary = cv2.bitwise_or(support, ball_support)
                if ball_pixels:
                    preliminary = cv2.bitwise_and(preliminary, cv2.bitwise_not(ball_support))
                return preliminary if int(np.count_nonzero(preliminary)) else None
            return None

        background = cv2.convertScaleAbs(self._accumulator)
        diff = cv2.absdiff(frame, background)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_gray = cv2.GaussianBlur(diff_gray, (5, 5), 0)
        _, motion_mask = cv2.threshold(diff_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if int(np.count_nonzero(motion_mask)) == 0:
            _, motion_mask = cv2.threshold(diff_gray, 18, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.morphologyEx(
            motion_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        motion_mask = cv2.morphologyEx(
            motion_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=2,
        )
        if self.median_kernel_size > 1:
            motion_mask = cv2.medianBlur(motion_mask, self.median_kernel_size)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        mean_val, std_val = cv2.meanStdDev(gray_blur)
        mean_scalar = float(mean_val[0, 0]) if mean_val.size else 0.0
        std_scalar = float(std_val[0, 0]) if std_val.size else 0.0
        bright_thresh = mean_scalar + 1.5 * std_scalar
        bright_thresh = max(0.0, min(255.0, bright_thresh))
        if bright_thresh < 180.0:
            bright_thresh = 180.0
        _, bright_mask = cv2.threshold(gray_blur, bright_thresh, 255, cv2.THRESH_BINARY)
        bright_mask = cv2.dilate(
            bright_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )

        combined = cv2.bitwise_or(motion_mask, bright_mask)
        if support_pixels:
            combined = cv2.bitwise_or(combined, support)
        if ball_pixels:
            combined = cv2.bitwise_or(combined, ball_support)
        if int(np.count_nonzero(combined)) == 0:
            return None

        if self.history_length > 1:
            current_binary = (combined > 0).astype(np.uint8)
            if self._history and self._history[0].shape != current_binary.shape:
                self._history.clear()
            self._history.append(current_binary)
            accum = np.zeros_like(current_binary, dtype=np.uint16)
            for past in self._history:
                accum += past
            stable = np.where(accum >= self.persistence_threshold, 255, 0).astype(np.uint8)
        else:
            stable = combined

        bright_bool = bright_mask.astype(bool)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(stable)
        filtered = np.zeros_like(stable)
        support_bool = support.astype(bool)
        support_has_pixels = bool(support_pixels)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_component_area or area > self.max_component_area:
                continue
            component_mask = labels == label
            if support_has_pixels and not np.any(component_mask & support_bool):
                continue
            if not np.any(component_mask & bright_bool):
                continue
            filtered[component_mask] = 255

        if int(np.count_nonzero(filtered)) == 0 and support_has_pixels:
            filtered = support.copy()
        else:
            filtered = cv2.bitwise_or(filtered, support)

        if ball_pixels:
            filtered = cv2.bitwise_and(filtered, cv2.bitwise_not(ball_support))

        filtered = cv2.morphologyEx(
            filtered,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, closing_kernel, iterations=1)
        filtered = cv2.dilate(
            filtered,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        nonzero = int(np.count_nonzero(filtered))
        if nonzero == 0:
            return None

        num_labels_final, labels_final, stats_final, _ = cv2.connectedComponentsWithStats(filtered)
        if num_labels_final <= 1:
            return filtered
        areas_final = stats_final[1:, cv2.CC_STAT_AREA]
        if areas_final.size == 0:
            return None
        largest_label = 1 + int(np.argmax(areas_final))
        largest_mask = np.where(labels_final == largest_label, 255, 0).astype(np.uint8)
        if not np.any((labels_final == largest_label) & bright_bool):
            return None
        frame_shape = (height, width)
        smoothed = self._smooth_primary_region(largest_mask, bright_bool)
        if smoothed is None:
            fallback = self._predict_from_template(frame_shape)
            if fallback is not None:
                fallback_centroid = self._compute_centroid(fallback)
                prev_centroid = self._prev_centroid
                if fallback_centroid is not None and prev_centroid is not None:
                    self._prev_motion_vec = (
                        fallback_centroid[0] - prev_centroid[0],
                        fallback_centroid[1] - prev_centroid[1],
                    )
                elif fallback_centroid is not None:
                    self._prev_motion_vec = None
                self._prev_mask = fallback.copy()
                self._prev_centroid = fallback_centroid
                return fallback
            self._prev_mask = None
            self._prev_centroid = None
            self._prev_motion_vec = None
            return None

        area_est = float(np.count_nonzero(smoothed))
        centroid = self._compute_centroid(smoothed)
        movement_vec: tuple[float, float] | None = None
        if centroid is not None:
            if self._prev_centroid is not None:
                movement_vec = (
                    centroid[0] - self._prev_centroid[0],
                    centroid[1] - self._prev_centroid[1],
                )
            self._maybe_update_template(smoothed, centroid, area_est)
        elif self._prev_motion_vec is not None:
            movement_vec = self._prev_motion_vec

        enforced = self._augment_with_motion_template(smoothed, centroid, movement_vec)
        final_centroid = self._compute_centroid(enforced)
        if final_centroid is not None and self._prev_centroid is not None:
            self._prev_motion_vec = (
                final_centroid[0] - self._prev_centroid[0],
                final_centroid[1] - self._prev_centroid[1],
            )
        elif final_centroid is not None:
            self._prev_motion_vec = None
        else:
            self._prev_motion_vec = movement_vec

        self._prev_mask = enforced.copy()
        self._prev_centroid = final_centroid
        return enforced


class TFLiteBallDetector:
    """Thin wrapper around the exported YOLOv8 TFLite model with custom post-processing."""

    def __init__(
        self,
        model_path: str,
        *,
        conf_threshold: float = 0.01,
        iou_threshold: float = 0.4,
        max_detections: int = 10,
    ) -> None:
        self.interpreter = Interpreter(model_path=model_path, num_threads=4)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        _, self.input_height, self.input_width, _ = self.input_details["shape"]
        self.input_dtype = self.input_details["dtype"]
        q_scale, q_zero_point = self.input_details.get("quantization", (0.0, 0))
        self.input_scale = float(q_scale) if isinstance(q_scale, (np.ndarray, list)) else float(q_scale)
        self.input_zero_point = (
            int(q_zero_point[0])
            if isinstance(q_zero_point, (np.ndarray, list))
            else int(q_zero_point)
        )
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run inference on ``frame`` and return a list of detections sorted by confidence."""

        letterboxed, ratio, pad = self._letterbox(frame, (self.input_height, self.input_width))
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        input_tensor = rgb.astype(np.float32) / 255.0
        if self.input_dtype != np.float32:
            if self.input_scale == 0.0:
                raise RuntimeError("Quantized model lacks scale factor")
            input_tensor = np.round(input_tensor / self.input_scale + self.input_zero_point).astype(
                self.input_dtype
            )
        else:
            input_tensor = input_tensor.astype(np.float32)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        self.interpreter.set_tensor(self.input_details["index"], input_tensor)
        self.interpreter.invoke()
        raw_output = self.interpreter.get_tensor(self.output_details["index"])
        preds = np.squeeze(raw_output, axis=0).transpose(1, 0)  # (anchors, 5)
        if preds.size == 0:
            return []
        boxes_xywh = preds[:, :4]
        scores = preds[:, 4]
        mask = scores >= self.conf_threshold
        if not np.any(mask):
            return []
        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)
        left, top = pad
        boxes_xyxy[:, [0, 2]] -= left
        boxes_xyxy[:, [1, 3]] -= top
        if ratio == 0:
            raise RuntimeError("Letterbox ratio is zero")
        inv_ratio = 1.0 / ratio
        boxes_xyxy[:, [0, 2]] *= inv_ratio
        boxes_xyxy[:, [1, 3]] *= inv_ratio
        h, w = frame.shape[:2]
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0.0, w - 1)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0.0, h - 1)
        keep = self._nms(boxes_xyxy, scores)
        detections = []
        for idx in keep[: self.max_detections]:
            x1, y1, x2, y2 = boxes_xyxy[idx]
            detections.append(
                {
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "score": float(scores[idx]),
                }
            )
        return detections

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        x, y, w, h = boxes.T
        half_w = w / 2.0
        half_h = h / 2.0
        x1 = x - half_w
        y1 = y - half_h
        x2 = x + half_w
        y2 = y + half_h
        return np.stack((x1, y1, x2, y2), axis=1)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> list[int]:
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            ious = self._iou(boxes[i], boxes[rest])
            order = rest[ious <= self.iou_threshold]
        return keep

    @staticmethod
    def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if boxes.size == 0:
            return np.empty(0)
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter_w = np.maximum(0.0, x2 - x1)
        inter_h = np.maximum(0.0, y2 - y1)
        inter = inter_w * inter_h
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_box + area_boxes - inter + 1e-6
        return inter / union

    @staticmethod
    def _letterbox(
        image: np.ndarray,
        new_shape: tuple[int, int],
        color: tuple[int, int, int] = (0, 0, 0),
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        """Resize ``image`` with unchanged aspect ratio using padding."""

        shape = image.shape[:2]  # h, w
        target_h, target_w = new_shape
        if shape[0] == 0 or shape[1] == 0:
            raise ValueError("Invalid frame shape for letterbox")
        ratio = min(target_h / shape[0], target_w / shape[1])
        new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
        if new_unpad[0] != shape[1] or new_unpad[1] != shape[0]:
            resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            resized = image
        dw = target_w - new_unpad[0]
        dh = target_h - new_unpad[1]
        dw /= 2
        dh /= 2
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        if padded.shape[0] != target_h or padded.shape[1] != target_w:
            padded = cv2.resize(padded, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return padded, ratio, (left, top)


def check_tail_for_ball(
    video_path: str,
    *,
    detector: TFLiteBallDetector | None = None,
    calibration: dict[str, object] | None = None,
    frames_to_check: int = 12,
    stride: int = 1,
    score_threshold: float = 0.25,
    min_hits: int = 2,
) -> TailCheckResult:
    """Inspect the tail end of a clip and report whether the ball remains visible."""

    if calibration is not None:
        apply_calibration(calibration)

    if frames_to_check <= 0:
        return TailCheckResult(False, 0, 0, [], [])
    if stride <= 0:
        raise ValueError("stride must be positive")

    owned_detector = detector is None
    if detector is None:
        detector = TFLiteBallDetector("golf_ball_detector.tflite", conf_threshold=0.01)

    frame_loop_start = time.perf_counter()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for tail check: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    scores: list[float] = []
    frame_indices: list[int] = []
    hits = 0
    processed = 0

    if total_frames > 0:
        last_idx = max(0, total_frames - 1)
        span = frames_to_check * stride
        start_idx = max(0, last_idx - span + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        current_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) or start_idx
        while processed < frames_to_check and current_idx <= last_idx:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            detections = detector.detect(frame)
            best_score = max((det.get("score", 0.0) for det in detections), default=0.0)
            scores.append(float(best_score))
            frame_indices.append(int(current_idx))
            if best_score >= score_threshold:
                hits += 1
            processed += 1
            if processed >= frames_to_check:
                break
            skip = stride - 1
            while skip > 0 and current_idx < last_idx:
                ok_skip = cap.grab()
                current_idx += 1
                if not ok_skip:
                    break
                skip -= 1
            current_idx += 1
    else:
        while processed < frames_to_check:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            detections = detector.detect(frame)
            best_score = max((det.get("score", 0.0) for det in detections), default=0.0)
            scores.append(float(best_score))
            frame_indices.append(processed)
            if best_score >= score_threshold:
                hits += 1
            processed += 1
            for _ in range(stride - 1):
                if not cap.grab():
                    break

    cap.release()
    if owned_detector and hasattr(detector, "interpreter"):
        del detector

    ball_present = hits >= min_hits and hits > 0
    return TailCheckResult(ball_present, hits, processed, scores, frame_indices)


def find_motion_window(
    video_path: str,
    detector: TFLiteBallDetector,
    *,
    pad_frames: int = MAX_MOTION_FRAMES,
    max_frames: int = MAX_MOTION_FRAMES,
    confirm_radius: int = 3,
    score_threshold: float = MOTION_WINDOW_SCORE_THRESHOLD,
    debug: bool = False,
) -> tuple[int, int, bool, dict[str, int | bool | None]]:
    """Return a motion window around the last confident ball sighting.

    The search prioritises late frames by sampling the video with progressively
    finer strides until the first positive detection is found, then refines the
    exact exit frame with a short forward scan. The helper records lightweight
    stats describing how many frames were actually decoded so we can validate
    efficiency claims."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    miss_limit = (
        MOTION_WINDOW_EARLY_EXIT_MISSES
        if MOTION_WINDOW_EARLY_EXIT_MISSES is not None
        else max(confirm_radius * 3, 9)
    )
    max_eval_budget = max(
        total * MOTION_WINDOW_MAX_SCAN_MULTIPLIER,
        total + 512,
        MOTION_WINDOW_MIN_SCAN_BUDGET,
    )
    decode_failure_limit = max(
        MOTION_WINDOW_DECODE_FAILURE_LIMIT,
        confirm_radius * 4,
    )

    stats: dict[str, int | bool | None] = {
        "total_frames": total,
        "frames_evaluated": 0,
        "frames_decoded": 0,
        "frames_with_ball": 0,
        "detector_runs": 0,
        "seeks": 0,
        "coarse_step": None,
        "coarse_frames_scanned": 0,
        "coarse_found_frame": None,
        "coarse_false_frame": None,
        "fallback_full_scan": False,
        "refine_frames": 0,
        "refine_last_frame": None,
        "miss_limit": miss_limit,
        "max_eval_budget": int(max_eval_budget),
        "decode_failure_limit": int(decode_failure_limit),
        "ball_always_visible": False,
        "ball_never_observed": False,
    }

    detection_cache: dict[int, dict | None] = {}
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    current_pos: int | None = None
    decode_failures = 0

    def finalize_stats() -> dict[str, int | bool | None]:
        snapshot = dict(stats)
        snapshot["frames_processed"] = len(detection_cache)
        snapshot["detector_calls"] = stats["detector_runs"]
        snapshot.setdefault("decode_failures", decode_failures)
        return snapshot

    def debug_break(reason: str, frame_idx: int | None = None) -> None:
        if not debug:
            return
        print(f"[motion_debug] {reason} frame={frame_idx}")
        breakpoint()

    def decode_frame(idx: int) -> tuple[bool, np.ndarray | None]:
        nonlocal current_pos, frame_width, frame_height, decode_failures
        if idx < 0 or idx >= total:
            return False, None
        if current_pos is None or idx != current_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            stats["seeks"] += 1
        ret, frame = cap.read()
        if not ret:
            current_pos = None
            decode_failures += 1
            if decode_failures > decode_failure_limit:
                stats["decode_failures"] = decode_failures
                raise MotionWindowError(
                    f"Exceeded decode failure limit ({decode_failures}) while searching for the motion window",
                    stats=finalize_stats(),
                )
            return False, None
        decode_failures = 0
        current_pos = idx + 1
        stats["frames_decoded"] += 1
        if frame_width == 0 or frame_height == 0:
            frame_height, frame_width = frame.shape[:2]
        return True, frame

    def detect_frame(idx: int) -> dict | None:
        cached = detection_cache.get(idx, ...)
        if cached is not ...:
            return cached  # type: ignore[return-value]
        stats["frames_evaluated"] += 1
        if stats["frames_evaluated"] > max_eval_budget:
            raise MotionWindowError(
                "Motion window search exceeded the evaluation budget",
                stats=finalize_stats(),
            )
        ok, frame = decode_frame(idx)
        if not ok or frame is None:
            detection_cache[idx] = None
            return None
        enhanced, _ = preprocess_frame(frame)
        stats["detector_runs"] += 1
        detections = detector.detect(enhanced)
        best: dict | None = None
        best_score = float("-inf")
        for det in detections:
            score = det["score"]
            if score < score_threshold:
                continue
            x1, y1, x2, y2 = det["bbox"]
            width = float(x2 - x1)
            height = float(y2 - y1)
            if width <= 0.0 or height <= 0.0:
                continue
            aspect = min(width, height) / max(width, height)
            if aspect < MOTION_WINDOW_MIN_ASPECT_RATIO:
                continue
            if frame_width and frame_height:
                if not bbox_within_image(det["bbox"], float(frame_width), float(frame_height)):
                    continue
            cx, cy, radius, _ = bbox_to_ball_metrics(x1, y1, x2, y2)
            if radius < MIN_BALL_RADIUS_PX:
                continue
            if score > best_score:
                center = np.array([cx, cy], dtype=float)
                best = {
                    "frame": idx,
                    "center": center,
                    "radius": float(radius),
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "score": float(score),
                }
                best_score = score
        detection_cache[idx] = best
        if best is not None:
            stats["frames_with_ball"] += 1
        return best

    try:
        if total == 0:
            stats["ball_never_observed"] = True
            raise MotionWindowNotFoundError(
                "Video contains no frames to analyse for ball motion",
                stats=finalize_stats(),
            )

        coarse_true: int | None = None
        coarse_false: int | None = None
        coarse_frames_total = 0
        coarse_steps = (64, 32, 16, 8, 4)
        for step in coarse_steps:
            idx = total - 1
            last_false = total
            frames_this_step = 0
            while idx >= 0:
                frames_this_step += 1
                det = detect_frame(idx)
                if det is not None:
                    coarse_true = idx
                    coarse_false = last_false if last_false < total else None
                    stats["coarse_step"] = step
                    stats["coarse_found_frame"] = idx
                    stats["coarse_false_frame"] = coarse_false
                    debug_break("coarse-hit", idx)
                    break
                last_false = idx
                idx -= step
            coarse_frames_total += frames_this_step
            if coarse_true is not None:
                break

        if coarse_true is None:
            stats["fallback_full_scan"] = True
            idx = total - 1
            frames_this_step = 0
            while idx >= 0:
                frames_this_step += 1
                det = detect_frame(idx)
                if det is not None:
                    coarse_true = idx
                    coarse_false = idx + 1 if idx + 1 < total else None
                    stats["coarse_step"] = 1
                    stats["coarse_found_frame"] = idx
                    stats["coarse_false_frame"] = coarse_false
                    debug_break("fallback-hit", idx)
                    break
                idx -= 1
            coarse_frames_total += frames_this_step

        stats["coarse_frames_scanned"] = coarse_frames_total

        if coarse_true is None:
            stats["ball_never_observed"] = True
            raise MotionWindowNotFoundError(
                "Unable to locate any ball detections in the video",
                stats=finalize_stats(),
            )

        initial_det = detect_frame(coarse_true)
        if initial_det is None:
            stats["ball_never_observed"] = True
            raise MotionWindowNotFoundError(
                "Initial coarse detection vanished before refinement",
                stats=finalize_stats(),
            )

        last_detection_idx = coarse_true
        last_center = initial_det["center"]
        refine_idx = coarse_true + 1
        misses = 0
        refine_frames = 0
        center_jump_limit = MAX_CENTER_JUMP_PX * 1.25

        while refine_idx < total and misses < miss_limit:
            det = detect_frame(refine_idx)
            refine_frames += 1
            if det is not None:
                center = det["center"]
                if last_center is None or np.linalg.norm(center - last_center) <= center_jump_limit:
                    last_detection_idx = refine_idx
                    last_center = center
                    misses = 0
                else:
                    misses += 1
            else:
                misses += 1
            refine_idx += 1

        stats["refine_frames"] = refine_frames
        stats["refine_last_frame"] = last_detection_idx
        if coarse_false is None and last_detection_idx >= total - 1:
            stats["ball_always_visible"] = True
            raise MotionWindowDegenerateError(
                "Ball detections persist through the final frame; unable to determine when the ball leaves the shot",
                stats=finalize_stats(),
            )

        start_frame = max(0, last_detection_idx - pad_frames)
        end_frame = min(total, last_detection_idx + confirm_radius + 1)
        if end_frame - start_frame > max_frames:
            start_frame = max(0, end_frame - max_frames)

        original_start = start_frame
        original_end = end_frame

        shift = MOTION_WINDOW_FORWARD_SHIFT
        if shift != 0:
            start_frame = min(total, start_frame + shift)
            end_frame = min(total, end_frame + shift)
            if end_frame - start_frame > max_frames:
                start_frame = max(0, end_frame - max_frames)
            if start_frame >= end_frame:
                start_frame = max(0, end_frame - 1)
        applied_shift = start_frame - original_start

        stats["frames_processed"] = len(detection_cache)
        stats["forward_shift"] = applied_shift
        stats["initial_window"] = (original_start, original_end)
        stats["detector_calls"] = stats["detector_runs"]
        stats["coarse_false_frame"] = coarse_false
        stats["last_detection_frame"] = last_detection_idx

        return start_frame, end_frame, True, stats
    finally:
        cap.release()


def process_video(
    video_path: str,
    ball_path: str,
    sticker_path: str,
    frames_dir: str = "ball_frames",
    *,
    tail_check: TailCheckResult | None = None,
    calibration: dict[str, object] | None = None,
    tail_frames_to_check: int = 12,
    tail_stride: int = 1,
    tail_score_threshold: float = 0.25,
    tail_min_hits: int = 2,
) -> dict[str, object]:
    """Process ``video_path`` and persist ball + club trajectories alongside tail metadata.

    Club detection is simplified: blackout green pixels and use the largest
    remaining non-black connected component as the club "island". Its centroid
    per frame defines a trajectory. Teleporting points (large inter-frame jumps)
    are rejected. The helper reuses the same TFLite detector to quickly
    determine whether the ball remains visible in the clip tail so callers can
    decide whether additional captures are required before running the full
    pipeline.
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    current_calibration = apply_calibration(calibration)

    timings = TimingCollector()
    total_start = time.perf_counter()

    ball_compile_start = time.perf_counter()
    detector = TFLiteBallDetector("golf_ball_detector.tflite", conf_threshold=0.01)
    ball_compile_time = time.perf_counter() - ball_compile_start
    timings.add("detector_initialisation", ball_compile_time)

    tail_time = 0.0
    if tail_check is None:
        tail_start = time.perf_counter()
        tail_check = check_tail_for_ball(
            video_path,
            detector=detector,
            calibration=current_calibration,
            frames_to_check=tail_frames_to_check,
            stride=tail_stride,
            score_threshold=tail_score_threshold,
            min_hits=tail_min_hits,
        )
        tail_time = time.perf_counter() - tail_start
    timings.add("tail_check", tail_time)

    tail_summary = "n/a"
    if tail_check is not None:
        tail_summary = (
            f"{tail_check.hits}/{tail_check.frames_checked} frames >= {tail_score_threshold:.2f}"
        )
        state = "yes" if tail_check.ball_present else "no"
        print(f"Tail ball present: {state} ({tail_summary})")

    motion_start = time.perf_counter()
    try:
        start_frame, end_frame, _, motion_stats = find_motion_window(
            video_path,
            detector,
            debug=MOTION_WINDOW_DEBUG,
        )
    except MotionWindowError as exc:
        timings.add("motion_window", time.perf_counter() - motion_start)
        raise RuntimeError(f"Motion window detection failed: {exc}") from exc
    motion_window_time = time.perf_counter() - motion_start
    timings.add("motion_window", motion_window_time)
    processed = motion_stats.get("frames_processed", 0)
    detector_calls = motion_stats.get("detector_calls", 0)
    coarse_step = motion_stats.get("coarse_step")
    refine_frames = motion_stats.get("refine_frames")
    parts = [
        f"decoded {processed} frames",
        f"detector runs {detector_calls}",
    ]
    if isinstance(coarse_step, int) and coarse_step > 0:
        parts.append(f"coarse step {coarse_step}")
    if isinstance(refine_frames, int) and refine_frames > 0:
        parts.append(f"refine frames {refine_frames}")
    print(
        "Motion window frames: "
        f"{start_frame}-{end_frame} (" + ", ".join(parts) + ")"
    )

    # Calibrate adaptive alpha mapping 20 frames before motion window start
    alpha_mapper = AdaptiveAlphaMapper()
    motion_guard_tracker = MotionGuardTracker()
    calib_dir = "alphamapping"
    alpha_cal_time = 0.0
    alpha_cal_start = time.perf_counter()
    cap_cal: cv2.VideoCapture | None = None
    try:
        cap_cal = cv2.VideoCapture(video_path)
        if not cap_cal.isOpened():
            raise RuntimeError("Unable to open video for alpha calibration")
        calib_idx = max(0, int(start_frame) - 20)
        cap_cal.set(cv2.CAP_PROP_POS_FRAMES, calib_idx)
        ok_cal, calib_frame = cap_cal.read()
        if ok_cal and calib_frame is not None:
            enh_cal, _ = preprocess_frame(calib_frame)
            dets_cal = detector.detect(enh_cal)
            ball_bbox_cal: tuple[float, float, float, float] | None = None
            if dets_cal:
                # choose best detection by score within reasonable size
                best_det_cal = max(dets_cal, key=lambda d: d.get("score", 0.0))
                if best_det_cal.get("score", 0.0) >= 0.01:
                    bb = best_det_cal.get("bbox")
                    if bb and all(np.isfinite(bb)):
                        ball_bbox_cal = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
            alpha_mapper.calibrate(
                enh_cal,
                ball_bbox=ball_bbox_cal,
                frame_index=calib_idx,
                save_dir=calib_dir,
            )
    except Exception:
        # Non-fatal; we will fall back to green-only if calibration didn't initialize
        pass
    finally:
        if cap_cal is not None:
            cap_cal.release()
        alpha_cal_time = time.perf_counter() - alpha_cal_start
        timings.add("alpha_calibration", alpha_cal_time)

    # Simplified club tracking via non-green islands (no dot/column logic)
    MAX_TELEPORT_JUMP_PX = CLUBFACE_MAX_JUMP_PX
    club_pixels: list[dict[str, object]] = []  # {'time': t, 'u': u, 'v': v}
    path_points: list[tuple[int, int]] = []   # for per-frame overlay
    # Shape-consistency state
    prev_club_mask: np.ndarray | None = None
    prev_centroid: tuple[float, float] | None = None
    prev_motion: tuple[float, float] | None = None
    club_small_component_streak = 0
    club_stationary_streak = 0
    club_path_min_u: float | None = None
    club_path_max_u: float | None = None
    club_path_confirmed = False
    club_path_min_v: float | None = None
    club_path_max_v: float | None = None
    # Stationary blackout line (ball midpoint Y), set once when first available
    fixed_blackout_y: int | None = alpha_mapper.ball_mid_y if hasattr(alpha_mapper, "ball_mid_y") else None
    # Stop collecting club points once ball starts moving
    BALL_MOVE_THRESHOLD_PX = 3.0
    club_recording_enabled = True
    club_motion_pause_triggered = False
    club_recording_resume_delay = 0
    prev_club_depth: float | None = None
    club_depth_offset: float | None = None
    club_depth_anchor_samples: list[float] = []
    club_depth_anchor_samples_collected = 0
    impact_frame_idx: int | None = None
    impact_time: float | None = None
    annotated_visible_fractions: dict[int, float] = {}
    annotated_frame_sizes: dict[int, int] = {}
    blackout_outlier_frames: set[int] = set()
    blackout_filter_info: dict[str, object] = {
        "applied": False,
        "frames_removed": 0,
        "club_samples_removed": 0,
        "min_visible_frac": CLUB_ANNOTATION_MIN_VISIBLE_FRAC,
        "min_file_bytes": CLUB_ANNOTATION_MIN_FILE_BYTES,
    }
    club_miss_streak = 0
    alpha_relax_level = 0
    alpha_relax_info: dict[str, object] = {
        "level": 0,
        "events": [],
        "streak_threshold": CLUB_ALPHA_RELAX_STREAK,
        "visible_trigger": CLUB_ALPHA_RELAX_VISIBLE_TRIGGER,
    }
    last_ellipse_center: tuple[float, float] | None = None
    last_ellipse_axes: tuple[float, float] | None = None
    last_ellipse_angle: float | None = None
    last_ellipse_frame_idx = -1
    ball_contact_timer = 0
    ball_contact_occurred = False
    # Tunables for shape consistency
    IOU_SELECT_THRESHOLD = 0.15
    OVERLAP_REQUIRED_FRAC = 0.25
    DILATE_FOR_OVERLAP = 7
    MAX_SPILLOVER_RATIO = 1.6

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    inference_start = max(0, start_frame)
    inference_end = min(total_frames, end_frame)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)
        for name in os.listdir(frames_dir):
            try:
                os.remove(os.path.join(frames_dir, name))
            except OSError:
                pass

    ball_time = 0.0
    clubface_time = 0.0
    ball_coords: list[dict] = []
    last_ball_center: np.ndarray | None = None
    last_ball_radius: float | None = None
    ball_velocity = np.zeros(2, dtype=float)

    frame_loop_start = time.perf_counter()
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        orig = frame
        enhanced, _ = preprocess_frame(frame)
        base_frame = enhanced.copy()
        # No background modeling for club; we work off blackout-only
        if h is None:
            h, w = base_frame.shape[:2]
        frame_pixel_count = float(h * w) if h is not None and w is not None else None
        # No IR toggling; use color frames only for club path
        t = frame_idx / video_fps
        in_window = inference_start <= frame_idx < inference_end

        if not club_recording_enabled and club_motion_pause_triggered:
            if club_recording_resume_delay > 0:
                club_recording_resume_delay -= 1
            if club_recording_resume_delay <= 0 or frame_idx >= inference_end:
                club_recording_enabled = True

        ball_overlay: dict[str, object] | None = None
        ball_prediction_overlay: dict[str, object] | None = None
        text_overlays: list[dict[str, object]] = []
        ball_bbox_for_mask: tuple[float, float, float, float] | None = None

        detections: list[dict] = []
        if in_window:
            start = time.perf_counter()
            detections = detector.detect(enhanced)
            ball_time += time.perf_counter() - start

        detected = False
        if in_window and detections:
            edge_safe = [
                det
                for det in detections
                if bbox_within_image(det["bbox"], w, h)
            ]
            candidates = edge_safe if edge_safe else detections
            best_det = max(candidates, key=lambda d: d["score"])
            if best_det["score"] >= BALL_SCORE_THRESHOLD:
                x1, y1, x2, y2 = best_det["bbox"]
                cx, cy, rad, distance = bbox_to_ball_metrics(x1, y1, x2, y2)
                if rad >= MIN_BALL_RADIUS_PX:
                    center = np.array([cx, cy], dtype=float)
                    if last_ball_center is not None and np.linalg.norm(center - last_ball_center) > MAX_CENTER_JUMP_PX:
                        pass
                    else:
                        bx = (cx - w / 2.0) * distance / FOCAL_LENGTH
                        by = (cy - h / 2.0) * distance / FOCAL_LENGTH
                        bz = distance - 30.0
                        ball_coords.append(
                            {
                                "time": round(t, 3),
                                "x": round(bx, 2),
                                "y": round(by, 2),
                                "z": round(bz, 2),
                            }
                        )
                        ball_overlay = {
                            "center": (int(round(cx)), int(round(cy))),
                            "radius": int(round(rad)),
                        }
                        text_overlays.append(
                            {
                                "value": f"x:{bx:.2f} y:{by:.2f} z:{bz:.2f}",
                                "position": (int(round(cx)) + 10, int(round(cy))),
                                "color": (0, 255, 0),
                                "scale": 0.5,
                                "thickness": 1,
                            }
                        )
                        ball_bbox_for_mask = (x1, y1, x2, y2)
                        if last_ball_center is not None:
                            ball_velocity = center - last_ball_center
                        last_ball_center = center
                        last_ball_radius = rad
                        detected = True
                        # If the ball started moving, stop recording club points
                        move_mag = float(np.linalg.norm(ball_velocity))
                        if (not club_motion_pause_triggered) and move_mag > BALL_MOVE_THRESHOLD_PX:
                            club_recording_enabled = False
                            club_motion_pause_triggered = True
                            club_recording_resume_delay = CLUB_RESUME_DELAY_FRAMES
                            if impact_frame_idx is None:
                                impact_frame_idx = frame_idx
                                impact_time = round(float(t), 3)

        if not detected and last_ball_center is not None and in_window:
            if last_ball_radius is not None:
                motion = ball_velocity
                if np.linalg.norm(motion) <= MAX_CENTER_JUMP_PX:
                    cx, cy = last_ball_center + motion
                    radius = last_ball_radius
                    center_int = (int(round(cx)), int(round(cy)))
                    ball_prediction_overlay = {
                        "center": center_int,
                        "radius": int(round(radius)),
                    }
                    ball_bbox_for_mask = (
                        float(cx - radius),
                        float(cy - radius),
                        float(cx + radius),
                        float(cy + radius),
                    )

        ball_center_px: tuple[float, float] | None = None
        ball_radius_px: float | None = None
        overlay_source = ball_overlay if ball_overlay is not None else ball_prediction_overlay
        if overlay_source is not None:
            cx_src, cy_src = overlay_source["center"]
            ball_center_px = (float(cx_src), float(cy_src))
            ball_radius_px = float(overlay_source["radius"])

        ball_contact_candidate = False
        if ball_center_px is not None:
            contact_threshold = (ball_radius_px if ball_radius_px is not None else 0.0) + CLUB_BALL_CONTACT_MARGIN_PX
            if prev_centroid is not None:
                if math.hypot(ball_center_px[0] - prev_centroid[0], ball_center_px[1] - prev_centroid[1]) <= contact_threshold:
                    ball_contact_candidate = True
            if (not ball_contact_candidate) and last_ellipse_center is not None:
                if math.hypot(ball_center_px[0] - last_ellipse_center[0], ball_center_px[1] - last_ellipse_center[1]) <= contact_threshold:
                    ball_contact_candidate = True

        if ball_contact_candidate:
            ball_contact_timer = int(CLUB_BALL_CONTACT_HOLD_FRAMES)
        elif ball_contact_timer > 0:
            ball_contact_timer -= 1
        ball_contact_active = ball_contact_timer > 0
        if ball_contact_active:
            ball_contact_occurred = True

        guard_info = motion_guard_tracker.update(base_frame, in_window=in_window)
        # Build a simplified club mask: adaptive alpha mapping (dominant background),
        # then take the largest non-black connected component as the "club" island.
        # Exclude the ball bbox.
        output_frame = alpha_mapper.apply(base_frame.copy(), motion_guard=guard_info)
        # Establish a fixed blackout line at ball midpoint the first time we see the ball
        if fixed_blackout_y is None and ball_bbox_for_mask is not None and h is not None:
            try:
                _, _y1_fix, _, _y2_fix = ball_bbox_for_mask
                fixed_blackout_y = int(round((float(_y1_fix) + float(_y2_fix)) / 2.0))
                fixed_blackout_y = int(np.clip(fixed_blackout_y, 0, h - 1))
            except Exception:
                fixed_blackout_y = None

        apply_blackout_line = (
            fixed_blackout_y is not None and alpha_relax_level < CLUB_ALPHA_RELAX_DISABLE_BLACKOUT_LEVEL
        )
        if apply_blackout_line:
            output_frame[fixed_blackout_y + 1 :, :, :] = 0

        if alpha_relax_level > 0:
            blend_idx = min(alpha_relax_level, len(CLUB_ALPHA_RELAX_BLEND_LEVELS)) - 1
            blend_ratio = float(CLUB_ALPHA_RELAX_BLEND_LEVELS[blend_idx])
            blend_ratio = float(np.clip(blend_ratio, 0.0, 0.75))
            output_frame = cv2.addWeighted(
                output_frame,
                float(max(0.0, 1.0 - blend_ratio)),
                base_frame,
                blend_ratio,
                0.0,
            )

        gray_preview = cv2.cvtColor(output_frame, cv2.COLOR_BGR2GRAY)
        total_preview = gray_preview.size
        frame_visible_fraction = 0.0
        if total_preview > 0:
            frame_visible_fraction = float(np.count_nonzero(gray_preview)) / float(total_preview)

        pre_contact_heavy_blackout = False
        if not ball_contact_occurred and frame_visible_fraction < CLUB_PRECONTACT_MIN_VISIBLE_FRAC:
            pre_contact_heavy_blackout = True

        club_sample_created = False

        if in_window:
            club_stage_start = time.perf_counter()
            gray_blk = gray_preview
            threshold_val = max(3, 10 - alpha_relax_level * 2)
            _, nb = cv2.threshold(gray_blk, threshold_val, 255, cv2.THRESH_BINARY)
            # Apply the same stationary cutoff on the binary mask
            if apply_blackout_line:
                nb[fixed_blackout_y + 1 :, :] = 0
            if not ball_contact_active:
                if ball_center_px is not None and ball_radius_px is not None:
                    cx_i = int(round(ball_center_px[0]))
                    cy_i = int(round(ball_center_px[1]))
                    rad_i = max(1, int(round(ball_radius_px))) + 2
                    if 0 <= cx_i < w and 0 <= cy_i < h:
                        cv2.circle(nb, (cx_i, cy_i), rad_i, 0, -1, cv2.LINE_AA)
                elif ball_bbox_for_mask is not None:
                    x1, y1, x2, y2 = map(int, map(round, ball_bbox_for_mask))
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w - 1, x2)
                    y2 = min(h - 1, y2)
                    if x2 > x1 and y2 > y1:
                        cx_mid = int(round((x1 + x2) / 2))
                        cy_mid = int(round((y1 + y2) / 2))
                        rad_mid = int(round(max(x2 - x1, y2 - y1) / 2)) + 2
                        cv2.circle(nb, (cx_mid, cy_mid), max(1, rad_mid), 0, -1, cv2.LINE_AA)
            if alpha_relax_level <= 1:
                nb = cv2.morphologyEx(nb, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
                nb = cv2.morphologyEx(nb, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)))
            else:
                nb = cv2.morphologyEx(
                    nb,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                )
            club_mask = None
            if pre_contact_heavy_blackout:
                blackout_outlier_frames.add(frame_idx)
                clubface_time += time.perf_counter() - club_stage_start
                continue
            if int(np.count_nonzero(nb)):
                num, labels, stats, _ = cv2.connectedComponentsWithStats(nb)
                # Predict where the shape should be based on previous motion
                predicted: np.ndarray | None = None
                if prev_club_mask is not None and prev_centroid is not None:
                    dx, dy = (0.0, 0.0)
                    if prev_motion is not None:
                        dx, dy = prev_motion
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    predicted = cv2.warpAffine(
                        prev_club_mask,
                        M,
                        (int(w), int(h)),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    if DILATE_FOR_OVERLAP > 0:
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_FOR_OVERLAP, DILATE_FOR_OVERLAP))
                        predicted = cv2.dilate(predicted, kernel, iterations=1)

                best_label = None
                best_score = -1.0
                best_mask_local = None
                prev_area = float(np.count_nonzero(prev_club_mask)) if prev_club_mask is not None else 0.0
                for lbl in range(1, num):
                    region = (labels == lbl)
                    reg_mask = np.where(region, 255, 0).astype(np.uint8)
                    score = float(stats[lbl, cv2.CC_STAT_AREA])
                    if predicted is not None:
                        inter = int(np.count_nonzero(cv2.bitwise_and(predicted, reg_mask)))
                        union = int(np.count_nonzero(cv2.bitwise_or(predicted, reg_mask)))
                        iou = (inter / union) if union > 0 else 0.0
                        overlap_frac = (inter / max(1.0, float(np.count_nonzero(reg_mask))))
                        # Prefer overlap with prediction; penalize huge spillover vs last area
                        area = float(stats[lbl, cv2.CC_STAT_AREA])
                        spill_penalty = 0.0
                        if prev_area > 0.0 and area > prev_area * MAX_SPILLOVER_RATIO:
                            spill_penalty = 0.25
                        score = 2.0 * iou + 0.5 * overlap_frac - spill_penalty
                    if score > best_score:
                        best_score = score
                        best_label = lbl
                        best_mask_local = reg_mask

                # If we had a prediction, enforce consistency constraints
                if predicted is not None and best_mask_local is not None:
                    inter = int(np.count_nonzero(cv2.bitwise_and(predicted, best_mask_local)))
                    sel_area = int(np.count_nonzero(best_mask_local))
                    pred_area = int(np.count_nonzero(predicted))
                    iou = inter / float(max(1, sel_area + pred_area - inter))
                    overlap_frac = inter / float(max(1, sel_area))
                    if iou < IOU_SELECT_THRESHOLD or overlap_frac < OVERLAP_REQUIRED_FRAC:
                        # Too different from expected; stick to predicted region
                        best_mask_local = predicted.copy()

                    # Trim spillover to predicted support
                    best_mask_local = cv2.bitwise_and(best_mask_local, predicted)

                club_mask = best_mask_local

                if club_mask is None and num > 1:
                    # Fallback to largest area if no predictor
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    largest = 1 + int(np.argmax(areas))
                    club_mask = np.where(labels == largest, 255, 0).astype(np.uint8)
                elif club_mask is None:
                    club_mask = nb

                centroid_mask = club_mask.copy() if club_mask is not None else None
                if (
                    club_mask is not None
                    and ball_contact_active
                    and ball_center_px is not None
                    and ball_radius_px is not None
                ):
                    ball_patch = np.zeros_like(club_mask)
                    cx_union = int(round(ball_center_px[0]))
                    cy_union = int(round(ball_center_px[1]))
                    contact_radius = max(1, int(round(ball_radius_px)))
                    cv2.circle(ball_patch, (cx_union, cy_union), contact_radius, 255, -1, cv2.LINE_AA)
                    gate_mask: np.ndarray | None = None
                    if last_ellipse_center is not None and last_ellipse_axes is not None:
                        ellipse_gate = np.zeros_like(club_mask)
                        gate_axes = (
                            float(max(1.0, last_ellipse_axes[0] + 2.0 * CLUB_BALL_CONTACT_UNION_MARGIN_PX)),
                            float(max(1.0, last_ellipse_axes[1] + 2.0 * CLUB_BALL_CONTACT_UNION_MARGIN_PX)),
                        )
                        cv2.ellipse(
                            ellipse_gate,
                            (
                                (int(round(last_ellipse_center[0])), int(round(last_ellipse_center[1]))),
                                gate_axes,
                                float(last_ellipse_angle if last_ellipse_angle is not None else 0.0),
                            ),
                            255,
                            -1,
                            cv2.LINE_AA,
                        )
                        gate_mask = cv2.bitwise_and(ball_patch, ellipse_gate)
                    elif prev_centroid is not None:
                        centroid_gate = np.zeros_like(club_mask)
                        local_radius = max(1, int(round(CLUB_BALL_CONTACT_UNION_MARGIN_PX)))
                        cv2.circle(
                            centroid_gate,
                            (int(round(prev_centroid[0])), int(round(prev_centroid[1]))),
                            local_radius,
                            255,
                            -1,
                            cv2.LINE_AA,
                        )
                        gate_mask = cv2.bitwise_and(ball_patch, centroid_gate)
                    elif centroid_mask is not None:
                        gate_mask = cv2.bitwise_and(ball_patch, centroid_mask)
                    else:
                        gate_mask = ball_patch
                    if gate_mask is not None:
                        club_mask = cv2.bitwise_or(club_mask, gate_mask)

                pending_sample: dict[str, object] | None = None
                pending_path_point: tuple[int, int] | None = None
                pending_u: float | None = None
                pending_v: float | None = None
                record_sample = False

                depth_cm: float | None = None
                club_width_px: float | None = None
                club_area_px: float | None = None
                leftmost_x: float | None = None
                club_visible_frac: float | None = None
                u: float | None = None
                v: float | None = None
                centroid_source = centroid_mask if centroid_mask is not None else club_mask
                try:
                    nz = np.column_stack(np.where(centroid_source > 0)) if centroid_source is not None else np.empty((0, 2), dtype=np.int32)
                except Exception:
                    nz = np.empty((0, 2), dtype=np.int32)
                if nz.size:
                    xs = nz[:, 1].astype(np.float64)
                    ys = nz[:, 0].astype(np.float64)
                    leftmost_x = float(xs.min())
                    rightmost_x = float(xs.max())
                    club_width_px = float(max(1.0, rightmost_x - leftmost_x + 1.0))
                    club_area_px = float(xs.size)
                    if frame_pixel_count and frame_pixel_count > 0.0:
                        club_visible_frac = float(xs.size) / float(frame_pixel_count)
                    weights = np.ones_like(xs, dtype=np.float64)
                    if CLUB_LEFT_WEIGHT_BIAS > 0.0 and club_width_px > 1.0:
                        span = (xs - leftmost_x) / max(1.0, club_width_px)
                        weights += float(CLUB_LEFT_WEIGHT_BIAS) * (1.0 - span)
                    weight_sum = float(weights.sum())
                    if weight_sum > 1e-6:
                        u = float(np.dot(xs, weights) / weight_sum)
                        v = float(np.dot(ys, weights) / weight_sum)
                    else:
                        u = float(xs.mean())
                        v = float(ys.mean())
                    baseline_px = max(1e-3, float(u) - leftmost_x)
                    try:
                        depth_cm = float(FOCAL_LENGTH * ACTUAL_BALL_RADIUS / baseline_px)
                    except Exception:
                        depth_cm = None
                if u is None or v is None:
                    m = cv2.moments(centroid_source, binaryImage=True) if centroid_source is not None else {"m00": 0.0}
                    if m["m00"] > 1e-6:
                        u = float(m["m10"] / m["m00"])
                        v = float(m["m01"] / m["m00"])
                if (u is None or v is None) and ball_contact_active:
                    fallback_center: tuple[float, float] | None = None
                    if last_ellipse_center is not None:
                        fallback_center = last_ellipse_center
                    elif prev_centroid is not None:
                        fallback_center = prev_centroid
                    elif ball_center_px is not None:
                        fallback_center = ball_center_px
                    if fallback_center is not None:
                        u = float(fallback_center[0])
                        v = float(fallback_center[1])
                frame_area = frame_pixel_count if frame_pixel_count else (float(h * w) if h and w else None)
                component_valid = True
                if club_area_px is not None:
                    min_area_req = CLUB_COMPONENT_MIN_AREA_PX
                    if frame_area:
                        min_area_req = max(min_area_req, float(frame_area) * CLUB_COMPONENT_MIN_AREA_FRAC)
                    if club_area_px < min_area_req:
                        component_valid = False
                if club_visible_frac is not None and club_visible_frac < CLUB_COMPONENT_MIN_VISIBLE_FRAC:
                    component_valid = False
                if club_width_px is not None and club_width_px < CLUB_COMPONENT_MIN_WIDTH_PX:
                    component_valid = False
                pre_contact_small_component = False
                if not ball_contact_occurred:
                    if club_area_px is not None and club_area_px < CLUB_PRECONTACT_MIN_COMPONENT_AREA_PX:
                        pre_contact_small_component = True
                    visible_floor = max(
                        CLUB_COMPONENT_MIN_VISIBLE_FRAC * 1.35,
                        CLUB_COMPONENT_MIN_VISIBLE_FRAC + 0.0004,
                    )
                    if club_visible_frac is not None and club_visible_frac < visible_floor:
                        pre_contact_small_component = True
                if pre_contact_small_component:
                    component_valid = False

                movement = None
                if prev_centroid is not None and u is not None and v is not None:
                    movement = math.hypot(u - prev_centroid[0], v - prev_centroid[1])

                reset_due_to_stationary = False
                if component_valid and path_points:
                    if movement is not None and movement >= CLUB_COMPONENT_MIN_MOVEMENT_PX:
                        club_stationary_streak = 0
                    else:
                        club_stationary_streak += 1
                    span_x = span_y = None
                    if (
                        club_path_min_u is not None
                        and club_path_max_u is not None
                        and club_path_min_v is not None
                        and club_path_max_v is not None
                        and u is not None
                        and v is not None
                    ):
                        span_x = max(club_path_max_u, u) - min(club_path_min_u, u)
                        span_y = max(club_path_max_v, v) - min(club_path_min_v, v)
                    span_small = (
                        span_x is not None
                        and span_y is not None
                        and len(path_points) >= CLUB_COMPONENT_STATIONARY_MAX
                        and span_x < CLUB_PATH_MIN_SPAN_X_PX
                        and span_y < CLUB_PATH_MIN_SPAN_Y_PX
                    )
                    if club_stationary_streak > CLUB_COMPONENT_STATIONARY_MAX and span_small:
                        component_valid = False
                        reset_due_to_stationary = True
                elif movement is not None and movement >= CLUB_COMPONENT_MIN_MOVEMENT_PX:
                    club_stationary_streak = 0

                if not component_valid:
                    club_small_component_streak += 1
                else:
                    club_small_component_streak = 0

                if reset_due_to_stationary:
                    if not club_path_confirmed:
                        club_pixels.clear()
                    path_points.clear()
                    club_path_min_u = club_path_max_u = None
                    club_path_min_v = club_path_max_v = None
                    club_path_confirmed = False
                    prev_centroid = None
                    prev_motion = None
                    prev_club_mask = None
                    club_stationary_streak = 0
                    club_small_component_streak = 0
                    club_depth_offset = None
                    club_depth_anchor_samples.clear()
                    club_depth_anchor_samples_collected = 0
                    club_sample_created = False
                    continue

                if not component_valid:
                    if pre_contact_small_component:
                        blackout_outlier_frames.add(frame_idx)
                    continue

                if club_mask is not None:
                    output_frame = cv2.bitwise_and(output_frame, output_frame, mask=club_mask)

                if u is not None and v is not None:
                    accept = True
                    if path_points:
                        pu, pv = path_points[-1]
                        if math.hypot(u - pu, v - pv) > MAX_TELEPORT_JUMP_PX:
                            accept = False
                    if accept:
                        if depth_cm is None:
                            try:
                                if leftmost_x is None and nz.size:
                                    xs = nz[:, 1].astype(np.float64)
                                    leftmost_x = float(xs.min())
                                if leftmost_x is not None:
                                    baseline_px = max(1e-3, float(u) - leftmost_x)
                                    depth_cm = float(FOCAL_LENGTH * ACTUAL_BALL_RADIUS / baseline_px)
                            except Exception:
                                depth_cm = None
                        if depth_cm is None and prev_club_depth is not None:
                            depth_cm = prev_club_depth
                        if depth_cm is not None and np.isfinite(depth_cm):
                            depth_cm = abs(float(depth_cm))
                            prev_club_depth = depth_cm
                        pending_u = float(u)
                        pending_v = float(v)
                        pending_sample = {
                            "time": float(t),
                            "u": pending_u,
                            "v": pending_v,
                            "depth_cm": depth_cm if depth_cm is not None else None,
                            "club_width_px": float(club_width_px) if club_width_px is not None else None,
                            "club_area_px": float(club_area_px) if club_area_px is not None else None,
                            "club_visible_frac": float(club_visible_frac) if club_visible_frac is not None else None,
                            "left_edge_x": float(leftmost_x) if leftmost_x is not None else None,
                            "frame_idx": frame_idx,
                        }
                        pending_path_point = (int(round(u)), int(round(v)))
                        record_sample = club_recording_enabled and not pre_contact_heavy_blackout
                        pending_sample["record"] = record_sample

                tracking_valid = pending_sample is not None
                small_ellipse_precontact = False
                ellipse_candidate: tuple[tuple[float, float], tuple[float, float], float] | None = None
                ellipse_center_payload: dict[str, float] | None = None
                if club_mask is not None and tracking_valid:
                    ellipse_mask = club_mask.copy()
                    mask_population = int(np.count_nonzero(ellipse_mask))
                    ellipse = None
                    dist: np.ndarray | None = None
                    dist_max = 0.0
                    refined_mask: np.ndarray | None = None
                    kernel_size = max(3, int(CLUBFACE_ELLIPSE_ERODE_SIZE))
                    erosion_kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
                    )
                    component_mask = ellipse_mask
                    component_labels: np.ndarray | None = None
                    component_count = 0
                    if mask_population >= 5:
                        try:
                            component_count, component_labels = cv2.connectedComponents(ellipse_mask)
                        except cv2.error:
                            component_labels = None
                            component_count = 0
                        if component_count <= 1:
                            component_labels = None
                    if mask_population >= CLUBFACE_ELLIPSE_REQUIRED_PIXELS:
                        try:
                            dist = cv2.distanceTransform(ellipse_mask, cv2.DIST_L2, 5)
                        except cv2.error:
                            dist = None
                    if dist is not None and dist.size:
                        dist_max = float(dist.max())
                        mask_bool = ellipse_mask > 0
                        dist_vals = dist[mask_bool]
                        if dist_vals.size:
                            percentile = float(CLUBFACE_ELLIPSE_CORE_PERCENTILE)
                            min_percentile = float(CLUBFACE_ELLIPSE_MIN_PERCENTILE)
                            while percentile >= min_percentile:
                                thresh_val = float(np.percentile(dist_vals, percentile))
                                core_threshold = max(CLUBFACE_ELLIPSE_CORE_MIN_PX, thresh_val)
                                core_mask = np.where(dist >= core_threshold, 255, 0).astype(np.uint8)
                                if int(np.count_nonzero(core_mask)) < 5:
                                    percentile -= 5.0
                                    continue
                                refined = cv2.erode(ellipse_mask, erosion_kernel, iterations=1)
                                refined = cv2.bitwise_and(refined, core_mask)
                                if int(np.count_nonzero(refined)) < 5:
                                    refined = core_mask
                                num, labels = cv2.connectedComponents(refined)
                                if num > 1:
                                    best_label = None
                                    best_score = -1.0
                                    for lbl in range(1, num):
                                        region = labels == lbl
                                        count = int(np.count_nonzero(region))
                                        if count < 5:
                                            continue
                                        score = float(dist[region].mean())
                                        if score > best_score:
                                            best_score = score
                                            best_label = lbl
                                    if best_label is not None:
                                        refined = np.where(labels == best_label, 255, 0).astype(np.uint8)
                                if int(np.count_nonzero(refined)) >= 5:
                                    refined_mask = refined
                                    break
                                percentile -= 5.0
                    if refined_mask is None and dist is not None and dist.size and dist_max > 0.0:
                        fallback_threshold = max(
                            CLUBFACE_ELLIPSE_CORE_MIN_PX,
                            dist_max * CLUBFACE_ELLIPSE_CORE_FRACTION,
                        )
                        fallback_mask = np.where(dist >= fallback_threshold, 255, 0).astype(np.uint8)
                        if int(np.count_nonzero(fallback_mask)) >= 5:
                            refined_mask = fallback_mask
                    if refined_mask is None:
                        eroded = cv2.erode(ellipse_mask, erosion_kernel, iterations=1)
                        refined_mask = eroded if int(np.count_nonzero(eroded)) >= 5 else ellipse_mask.copy()
                    else:
                        border_kernel = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE,
                            (
                                max(1, int(CLUBFACE_ELLIPSE_BORDER_KERNEL)),
                                max(1, int(CLUBFACE_ELLIPSE_BORDER_KERNEL)),
                            ),
                        )
                        dilated = cv2.dilate(refined_mask, border_kernel, iterations=1)
                        refined_mask = cv2.bitwise_and(dilated, ellipse_mask)

                    if (
                        component_labels is not None
                        and refined_mask is not None
                        and int(np.count_nonzero(refined_mask)) >= 5
                    ):
                        refined_bool = refined_mask > 0
                        best_label = None
                        best_overlap = 0
                        for lbl in range(1, component_count):
                            region_bool = component_labels == lbl
                            overlap = int(np.count_nonzero(refined_bool & region_bool))
                            if overlap > best_overlap:
                                best_overlap = overlap
                                best_label = lbl
                        if best_label is not None:
                            component_mask = np.where(component_labels == best_label, 255, 0).astype(np.uint8)
                        refined_mask = cv2.bitwise_and(refined_mask, component_mask)
                    else:
                        component_mask = ellipse_mask

                    top_left_mask: np.ndarray | None = None
                    bottom_right_mask: np.ndarray | None = None
                    component_points = np.column_stack(np.where(component_mask > 0))
                    if component_points.size:
                        xs = component_points[:, 1]
                        ys = component_points[:, 0]
                        min_x = int(xs.min())
                        max_x = int(xs.max())
                        min_y = int(ys.min())
                        max_y = int(ys.max())
                        width = max(1, max_x - min_x + 1)
                        height = max(1, max_y - min_y + 1)
                        left_frac = float(CLUBFACE_ELLIPSE_LEFT_BAND_FRAC)
                        top_frac = float(CLUBFACE_ELLIPSE_TOP_BAND_FRAC)
                        dist_component_vals = dist[ys, xs] if dist is not None and dist.size else None
                        for _ in range(6):
                            band_width = int(
                                min(
                                    CLUBFACE_ELLIPSE_LEFT_BAND_MAX,
                                    math.ceil(width * left_frac),
                                )
                            )
                            band_width = max(1, band_width)
                            x_limit = min(max_x, min_x + band_width)
                            top_height = int(math.ceil(height * top_frac))
                            top_height = max(1, top_height)
                            y_limit = min(max_y, min_y + top_height)
                            selector = (xs <= x_limit) & (ys <= y_limit)
                            if not selector.any():
                                left_frac = min(0.65, left_frac + 0.06)
                                top_frac = min(0.9, top_frac + 0.08)
                                continue
                            rows = ys[selector]
                            cols = xs[selector]
                            candidate = np.zeros_like(component_mask)
                            candidate[rows, cols] = 255
                            if dist is not None and dist.size:
                                local_vals = dist[rows, cols]
                                if local_vals.size:
                                    local_thresh = max(
                                        CLUBFACE_ELLIPSE_CORE_MIN_PX,
                                        np.percentile(
                                            local_vals,
                                            CLUBFACE_ELLIPSE_TOPLEFT_CORE_PERCENTILE,
                                        ),
                                    )
                                    keep_idx = local_vals >= local_thresh
                                    if int(np.count_nonzero(keep_idx)) >= 5:
                                        rows = rows[keep_idx]
                                        cols = cols[keep_idx]
                                        candidate[:] = 0
                                        candidate[rows, cols] = 255
                            if int(np.count_nonzero(candidate)) < 5:
                                left_frac = min(0.65, left_frac + 0.06)
                                top_frac = min(0.9, top_frac + 0.08)
                                continue
                            band_kernel = cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE,
                                (
                                    max(1, int(CLUBFACE_ELLIPSE_BORDER_KERNEL)),
                                    max(1, int(CLUBFACE_ELLIPSE_BORDER_KERNEL)),
                                ),
                            )
                            candidate = cv2.dilate(candidate, band_kernel, iterations=1)
                            candidate = cv2.bitwise_and(candidate, component_mask)
                            if int(np.count_nonzero(candidate)) >= 5:
                                top_left_mask = candidate
                                break
                            left_frac = min(0.65, left_frac + 0.06)
                            top_frac = min(0.9, top_frac + 0.08)

                        if top_left_mask is None and component_points.size:
                            order = np.argsort(xs.astype(float) + 0.25 * ys.astype(float))
                            count = max(5, int(math.ceil(order.size * 0.22)))
                            selected = order[:count]
                            slab_cols = xs[selected]
                            slab_rows = ys[selected]
                            left_edge = float(np.min(xs.astype(float)))
                            right_cap = left_edge + max(2.0, float(width) * 0.4)
                            mask_tmp = np.zeros_like(component_mask)
                            for r, c in zip(slab_rows, slab_cols):
                                if float(c) <= right_cap:
                                    mask_tmp[r, c] = 255
                            if int(np.count_nonzero(mask_tmp)) >= 5:
                                band_kernel = cv2.getStructuringElement(
                                    cv2.MORPH_ELLIPSE,
                                    (
                                        max(1, int(CLUBFACE_ELLIPSE_BORDER_KERNEL)),
                                        max(1, int(CLUBFACE_ELLIPSE_BORDER_KERNEL)),
                                    ),
                                )
                                mask_tmp = cv2.dilate(mask_tmp, band_kernel, iterations=1)
                                mask_tmp = cv2.bitwise_and(mask_tmp, component_mask)
                                if int(np.count_nonzero(mask_tmp)) >= 5:
                                    top_left_mask = mask_tmp

                        if dist_component_vals is not None and dist_component_vals.size:
                            moment = cv2.moments(component_mask, binaryImage=True)
                            if moment["m00"] > 1e-6:
                                cx = float(moment["m10"] / moment["m00"])
                                cy = float(moment["m01"] / moment["m00"])
                            else:
                                cx = float(xs.mean())
                                cy = float(ys.mean())
                            threshold = max(
                                CLUBFACE_ELLIPSE_BOTTOM_MIN_PX,
                                dist_max * CLUBFACE_ELLIPSE_BOTTOM_DIST_FRAC,
                            )
                            attempt_thresh = threshold
                            for _ in range(5):
                                selection = (
                                    (xs.astype(float) >= cx)
                                    & (ys.astype(float) >= cy)
                                    & (dist_component_vals >= attempt_thresh)
                                )
                                if int(np.count_nonzero(selection)) >= 5:
                                    rows = ys[selection]
                                    cols = xs[selection]
                                    candidate = np.zeros_like(component_mask)
                                    candidate[rows, cols] = 255
                                    spread_kernel = cv2.getStructuringElement(
                                        cv2.MORPH_ELLIPSE,
                                        (
                                            max(1, int(CLUBFACE_ELLIPSE_BORDER_KERNEL) + 1),
                                            max(1, int(CLUBFACE_ELLIPSE_BORDER_KERNEL) + 1),
                                        ),
                                    )
                                    candidate = cv2.dilate(candidate, spread_kernel, iterations=1)
                                    candidate = cv2.bitwise_and(candidate, component_mask)
                                    if int(np.count_nonzero(candidate)) >= 5:
                                        bottom_right_mask = candidate
                                        break
                                attempt_thresh *= 0.88

                    anchor_left_point: tuple[float, float] | None = None
                    if top_left_mask is not None and int(np.count_nonzero(top_left_mask)) >= 3:
                        tl_pts = cv2.findNonZero(top_left_mask)
                        if tl_pts is not None and len(tl_pts):
                            tl_arr = tl_pts.reshape(-1, 2).astype(float)
                            tl_x = tl_arr[:, 0]
                            tl_y = tl_arr[:, 1]
                            scores_left = tl_x + 0.18 * tl_y
                            best_idx = int(np.argmin(scores_left))
                            anchor_left_point = (float(tl_x[best_idx]), float(tl_y[best_idx]))
                    if anchor_left_point is None and component_points.size:
                        scores_left = xs.astype(float) + 0.2 * ys.astype(float)
                        best_idx = int(np.argmin(scores_left))
                        anchor_left_point = (float(xs[best_idx]), float(ys[best_idx]))

                    anchor_right_point: tuple[float, float] | None = None
                    if bottom_right_mask is not None and int(np.count_nonzero(bottom_right_mask)) >= 3:
                        br_pts = cv2.findNonZero(bottom_right_mask)
                        if br_pts is not None and len(br_pts):
                            br_arr = br_pts.reshape(-1, 2).astype(float)
                            br_x = br_arr[:, 0]
                            br_y = br_arr[:, 1]
                            scores_right = br_x + 0.6 * br_y
                            best_idx = int(np.argmax(scores_right))
                            anchor_right_point = (float(br_x[best_idx]), float(br_y[best_idx]))
                    if anchor_right_point is None and component_points.size:
                        scores_right = xs.astype(float) + 0.85 * ys.astype(float)
                        best_idx = int(np.argmax(scores_right))
                        anchor_right_point = (float(xs[best_idx]), float(ys[best_idx]))

                    ellipse_source_mask: np.ndarray | None = None
                    anchors_source_used = False
                    if top_left_mask is not None and int(np.count_nonzero(top_left_mask)) >= 5:
                        ellipse_source_mask = top_left_mask.copy()
                        anchors_source_used = True
                    if bottom_right_mask is not None and int(np.count_nonzero(bottom_right_mask)) >= 5:
                        if ellipse_source_mask is None:
                            ellipse_source_mask = bottom_right_mask.copy()
                        else:
                            ellipse_source_mask = cv2.bitwise_or(
                                ellipse_source_mask, bottom_right_mask
                            )
                        anchors_source_used = True
                    if ellipse_source_mask is None:
                        if refined_mask is not None:
                            ellipse_source_mask = refined_mask.copy()
                        else:
                            ellipse_source_mask = component_mask.copy()
                    elif not anchors_source_used and refined_mask is not None:
                        ellipse_source_mask = cv2.bitwise_or(
                            ellipse_source_mask, refined_mask
                        )
                    ellipse_source_mask = cv2.bitwise_and(
                        ellipse_source_mask, component_mask
                    )
                    if int(np.count_nonzero(ellipse_source_mask)) < 5:
                        ellipse_source_mask = component_mask.copy()

                    ellipse_points = cv2.findNonZero(ellipse_source_mask)
                    if ellipse_points is not None and len(ellipse_points) >= 5:
                        try:
                            ellipse = cv2.fitEllipse(ellipse_points)
                        except cv2.error:
                            ellipse = None

                    ellipse_from_anchors = False
                    anchor_major_value: float | None = None
                    anchor_minor_value: float | None = None
                    if ellipse is not None and anchor_left_point is not None and anchor_right_point is not None:
                        ax, ay = anchor_left_point
                        bx, by = anchor_right_point
                        center_anchor = ((ax + bx) / 2.0, (ay + by) / 2.0)
                        dx = bx - ax
                        dy = by - ay
                        anchor_distance = math.hypot(dx, dy)
                        anchor_angle = math.degrees(math.atan2(dy, dx))
                        axes_existing = ellipse[1]
                        major_existing = max(float(axes_existing[0]), float(axes_existing[1]))
                        minor_existing = max(4.0, min(float(axes_existing[0]), float(axes_existing[1])))
                        anchor_major = max(anchor_distance, major_existing)
                        if anchor_major < 4.0:
                            anchor_major = 4.0
                        if dist_max > 0.0:
                            dist_for_minor = dist_max * 1.25
                        else:
                            dist_for_minor = minor_existing
                        max_minor_allowed = anchor_major * 0.75
                        anchor_minor = max(
                            4.0,
                            min(
                                max_minor_allowed,
                                max(minor_existing, dist_for_minor),
                            ),
                        )
                        if anchor_minor > anchor_major:
                            anchor_minor, anchor_major = anchor_major, anchor_minor
                            anchor_angle = (anchor_angle + 90.0) % 180.0
                        ellipse = (
                            (float(center_anchor[0]), float(center_anchor[1])),
                            (float(anchor_major), float(anchor_minor)),
                            float(anchor_angle),
                        )
                        ellipse_from_anchors = True
                        anchor_major_value = float(anchor_major)
                        anchor_minor_value = float(anchor_minor)

                    ellipse_candidate: tuple[tuple[float, float], tuple[float, float], float] | None = None
                    if ellipse is not None:
                        axes = list(ellipse[1])
                        if dist_max > 0.0:
                            expand = min(
                                CLUBFACE_ELLIPSE_BORDER_EXPAND_MAX,
                                max(0.0, dist_max * CLUBFACE_ELLIPSE_BORDER_EXPAND_RATIO),
                            )
                            if ellipse_from_anchors and anchor_major_value is not None and anchor_minor_value is not None:
                                expand = min(
                                    expand,
                                    anchor_major_value * 0.2,
                                    anchor_minor_value * 0.2,
                                )
                            if expand > 0.0:
                                axes[0] = max(1e-4, axes[0] + 2.0 * expand)
                                axes[1] = max(1e-4, axes[1] + 2.0 * expand)
                        major_idx = 0 if axes[0] >= axes[1] else 1
                        minor_idx = 1 - major_idx
                        major_axis = axes[major_idx]
                        minor_axis = max(axes[minor_idx], 1e-4)
                        max_major_allowed = CLUBFACE_ELLIPSE_MAX_ASPECT * minor_axis
                        if major_axis > max_major_allowed:
                            axes[major_idx] = max_major_allowed
                        ellipse = (ellipse[0], tuple(axes), ellipse[2])
                        center_corr = (float(ellipse[0][0]), float(ellipse[0][1]))
                        axes_corr = (
                            float(max(1e-4, ellipse[1][0])),
                            float(max(1e-4, ellipse[1][1])),
                        )
                        if last_ellipse_axes is not None:
                            axes_corr = (
                                float(max(1e-4, 0.4 * axes_corr[0] + 0.6 * last_ellipse_axes[0])),
                                float(max(1e-4, 0.4 * axes_corr[1] + 0.6 * last_ellipse_axes[1])),
                            )
                        angle_corr = float(ellipse[2])
                        if last_ellipse_center is not None:
                            allowed_drift = CLUB_ELLIPSE_MAX_DRIFT_PX
                            if ball_contact_active:
                                allowed_drift = max(
                                    allowed_drift,
                                    (ball_radius_px if ball_radius_px is not None else 0.0) + CLUB_BALL_CONTACT_MARGIN_PX,
                                )
                            drift = math.hypot(
                                center_corr[0] - last_ellipse_center[0],
                                center_corr[1] - last_ellipse_center[1],
                            )
                            if drift > allowed_drift:
                                if ball_contact_active and ball_center_px is not None:
                                    blend = min(max(CLUB_ELLIPSE_CONTACT_BLEND, 0.0), 1.0)
                                    center_corr = (
                                        float((1.0 - blend) * last_ellipse_center[0] + blend * ball_center_px[0]),
                                        float((1.0 - blend) * last_ellipse_center[1] + blend * ball_center_px[1]),
                                    )
                                else:
                                    center_corr = (
                                        float(last_ellipse_center[0]),
                                        float(last_ellipse_center[1]),
                                    )
                                if last_ellipse_axes is not None:
                                    axes_corr = (
                                        float(max(1e-4, 0.4 * axes_corr[0] + 0.6 * last_ellipse_axes[0])),
                                        float(max(1e-4, 0.4 * axes_corr[1] + 0.6 * last_ellipse_axes[1])),
                                    )
                        major_axis = max(float(axes_corr[0]), float(axes_corr[1]))
                        minor_axis = min(float(axes_corr[0]), float(axes_corr[1]))
                        if (
                            not ball_contact_occurred
                            and (
                                major_axis < CLUB_PRECONTACT_MIN_ELLIPSE_MAJOR_PX
                                or minor_axis < CLUB_PRECONTACT_MIN_ELLIPSE_MINOR_PX
                            )
                        ):
                            tracking_valid = False
                            small_ellipse_precontact = True
                            last_ellipse_center = None
                            last_ellipse_axes = None
                            last_ellipse_angle = None
                            last_ellipse_frame_idx = -1
                        else:
                            ellipse_candidate = (center_corr, axes_corr, angle_corr)

                if (
                    tracking_valid
                    and ellipse_candidate is None
                    and last_ellipse_center is not None
                    and last_ellipse_axes is not None
                    and last_ellipse_frame_idx >= 0
                    and (frame_idx - last_ellipse_frame_idx) <= CLUB_ELLIPSE_HOLD_FRAMES
                ):
                    angle_keep = last_ellipse_angle if last_ellipse_angle is not None else 0.0
                    ellipse_candidate = (
                        (float(last_ellipse_center[0]), float(last_ellipse_center[1])),
                        (float(max(1e-4, last_ellipse_axes[0])), float(max(1e-4, last_ellipse_axes[1]))),
                        float(angle_keep),
                    )

                if ellipse_candidate is not None and tracking_valid:
                    last_ellipse_center = (
                        float(ellipse_candidate[0][0]),
                        float(ellipse_candidate[0][1]),
                    )
                    last_ellipse_axes = (
                        float(max(1e-4, ellipse_candidate[1][0])),
                        float(max(1e-4, ellipse_candidate[1][1])),
                    )
                    last_ellipse_angle = float(ellipse_candidate[2])
                    last_ellipse_frame_idx = frame_idx
                    ellipse_center_payload = {
                        "x": round(float(ellipse_candidate[0][0]), 2),
                        "y": round(float(ellipse_candidate[0][1]), 2),
                    }
                    cv2.ellipse(
                        output_frame,
                        ellipse_candidate,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )
                elif (
                    last_ellipse_center is not None
                    and last_ellipse_frame_idx >= 0
                    and (frame_idx - last_ellipse_frame_idx) > CLUB_ELLIPSE_HOLD_FRAMES
                ):
                    last_ellipse_center = None
                    last_ellipse_axes = None
                    last_ellipse_angle = None
                    last_ellipse_frame_idx = -1
                else:
                    tracking_valid = False

                if tracking_valid and ellipse_candidate is None and ellipse_center_payload is None:
                    tracking_valid = False

                if small_ellipse_precontact:
                    blackout_outlier_frames.add(frame_idx)
                    frame_blackout_heavy = True

                if (
                    not tracking_valid
                    and ball_contact_active
                    and last_ellipse_center is not None
                    and last_ellipse_axes is not None
                ):
                    synthetic_angle = float(last_ellipse_angle if last_ellipse_angle is not None else 0.0)
                    synthetic_axes = (
                        float(max(1.0, last_ellipse_axes[0])),
                        float(max(1.0, last_ellipse_axes[1])),
                    )
                    ellipse_candidate = (
                        (float(last_ellipse_center[0]), float(last_ellipse_center[1])),
                        synthetic_axes,
                        synthetic_angle,
                    )
                    ellipse_center_payload = {
                        "x": round(float(last_ellipse_center[0]), 2),
                        "y": round(float(last_ellipse_center[1]), 2),
                    }
                    if pending_sample is None:
                        pending_sample = {
                            "time": float(t),
                            "u": float(last_ellipse_center[0]),
                            "v": float(last_ellipse_center[1]),
                            "depth_cm": prev_club_depth if prev_club_depth is not None else None,
                            "club_width_px": None,
                            "club_area_px": None,
                            "club_visible_frac": None,
                            "left_edge_x": None,
                            "frame_idx": frame_idx,
                            "record": bool(club_recording_enabled),
                        }
                        pending_u = pending_sample["u"]
                        pending_v = pending_sample["v"]
                        pending_path_point = (
                            int(round(pending_u)),
                            int(round(pending_v)),
                        )
                    tracking_valid = bool(pending_sample.get("record", False))
                    record_sample = bool(pending_sample.get("record", False))

                if (
                    ellipse_candidate is not None
                    and pending_sample is not None
                ):
                    center_now = ellipse_candidate[0]
                    pending_u = float(center_now[0])
                    pending_v = float(center_now[1])
                    pending_sample["u"] = pending_u
                    pending_sample["v"] = pending_v
                    if pending_path_point is not None:
                        pending_path_point = (
                            int(round(pending_u)),
                            int(round(pending_v)),
                        )
                    else:
                        pending_path_point = (
                            int(round(pending_u)),
                            int(round(pending_v)),
                        )

                commit_sample = (
                    tracking_valid
                    and pending_sample is not None
                    and bool(pending_sample.get("record"))
                )
                if commit_sample and pending_sample is not None:
                    depth_cm_current = pending_sample.get("depth_cm")
                    raw_depth: float | None = None
                    adjusted_depth: float | None = None
                    if depth_cm_current is not None and np.isfinite(depth_cm_current):
                        raw_depth = float(depth_cm_current)
                        if club_depth_offset is None:
                            club_depth_anchor_samples.append(raw_depth)
                            if len(club_depth_anchor_samples) >= CLUB_DEPTH_ANCHOR_SAMPLES:
                                anchor_subset = club_depth_anchor_samples[:CLUB_DEPTH_ANCHOR_SAMPLES]
                                anchor_value = float(np.median(anchor_subset))
                                club_depth_offset = CLUB_DEPTH_ANCHOR_TARGET_CM - anchor_value
                                club_depth_anchor_samples_collected = len(anchor_subset)
                        if club_depth_offset is not None:
                            adjusted_depth = raw_depth + club_depth_offset
                        else:
                            adjusted_depth = CLUB_DEPTH_ANCHOR_TARGET_CM
                    elif club_depth_offset is not None and prev_club_depth is not None:
                        raw_depth = float(prev_club_depth)
                        adjusted_depth = raw_depth + club_depth_offset
                    if adjusted_depth is not None:
                        adjusted_depth = float(max(0.0, adjusted_depth))
                    z_value: float | None = None
                    if adjusted_depth is not None and np.isfinite(adjusted_depth):
                        z_value = float(np.clip(adjusted_depth, CLUB_DEPTH_MIN_CM, CLUB_DEPTH_MAX_CM))
                    club_entry = {
                        "time": round(float(pending_sample["time"]), 3),
                        "x": round(float(pending_sample["u"]), 2),
                        "y": round(float(pending_sample["v"]), 2),
                    }
                    if z_value is not None:
                        club_entry["z"] = round(float(z_value), 2)
                    if raw_depth is not None:
                        club_entry["_raw_depth"] = round(float(raw_depth), 2)
                    if adjusted_depth is not None:
                        club_entry["_depth_adjusted"] = round(float(adjusted_depth), 2)
                    if pending_sample.get("club_width_px") is not None:
                        club_entry["_width_px"] = float(pending_sample["club_width_px"])
                    if pending_sample.get("club_area_px") is not None:
                        club_entry["_area_px"] = float(pending_sample["club_area_px"])
                    if pending_sample.get("club_visible_frac") is not None:
                        club_entry["_visible_frac"] = float(pending_sample["club_visible_frac"])
                    if pending_sample.get("left_edge_x") is not None:
                        club_entry["_left_edge_x"] = float(pending_sample["left_edge_x"])
                    if ellipse_center_payload is not None:
                        club_entry["ellipse_center"] = dict(ellipse_center_payload)
                    club_entry["_frame"] = frame_idx
                    club_pixels.append(club_entry)
                    club_sample_created = True
                    if pending_path_point is None:
                        pending_path_point = (
                            int(round(pending_sample["u"])),
                            int(round(pending_sample["v"])),
                        )
                    path_points.append(pending_path_point)
                    u_float = float(pending_sample["u"])
                    v_float = float(pending_sample["v"])
                    if club_path_min_u is None or u_float < club_path_min_u:
                        club_path_min_u = u_float
                    if club_path_max_u is None or u_float > club_path_max_u:
                        club_path_max_u = u_float
                    if club_path_min_v is None or v_float < club_path_min_v:
                        club_path_min_v = v_float
                    if club_path_max_v is None or v_float > club_path_max_v:
                        club_path_max_v = v_float
                    span_x_now = (
                        club_path_max_u - club_path_min_u
                        if club_path_min_u is not None and club_path_max_u is not None
                        else None
                    )
                    span_y_now = (
                        club_path_max_v - club_path_min_v
                        if club_path_min_v is not None and club_path_max_v is not None
                        else None
                    )
                    if span_x_now is not None and span_y_now is not None:
                        if span_x_now >= CLUB_PATH_MIN_SPAN_X_PX or span_y_now >= CLUB_PATH_MIN_SPAN_Y_PX:
                            club_path_confirmed = True
                else:
                    club_sample_created = False

                if tracking_valid and pending_u is not None and pending_v is not None and club_mask is not None:
                    if prev_centroid is not None:
                        prev_motion = (pending_u - prev_centroid[0], pending_v - prev_centroid[1])
                    prev_centroid = (pending_u, pending_v)
                    if centroid_mask is not None:
                        prev_club_mask = centroid_mask.copy()
                    else:
                        prev_club_mask = club_mask.copy()
                clubface_time += time.perf_counter() - club_stage_start
        if ball_overlay is not None:
            cx_i, cy_i = ball_overlay["center"]
            radius_i = max(int(ball_overlay["radius"]), 1)
            cv2.circle(output_frame, (cx_i, cy_i), radius_i + 3, (0, 0, 0), -1, cv2.LINE_AA)
        if ball_prediction_overlay is not None:
            center_pred = ball_prediction_overlay["center"]
            radius_pred = max(int(ball_prediction_overlay["radius"]), 1)
            cv2.circle(output_frame, center_pred, radius_pred + 3, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(output_frame, center_pred, radius_pred + 5, (0, 255, 255), 1, cv2.LINE_AA)
        for text_meta in text_overlays:
            cv2.putText(
                output_frame,
                text_meta["value"],
                (int(text_meta["position"][0]), int(text_meta["position"][1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                float(text_meta.get("scale", 0.45)),
                text_meta.get("color", (255, 0, 0)),
                int(text_meta.get("thickness", 1)),
                cv2.LINE_AA,
            )

        frame_blackout_heavy = (
            frame_visible_fraction < CLUB_ALPHA_RELAX_VISIBLE_TRIGGER
        ) or pre_contact_heavy_blackout

        if frames_dir and inference_start <= frame_idx < inference_end:
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
            blackout_heavy = frame_visible_fraction < CLUB_ANNOTATION_MIN_VISIBLE_FRAC
            visible_fraction = frame_visible_fraction
            write_ok = cv2.imwrite(frame_path, output_frame)
            file_size = None
            if write_ok:
                try:
                    file_size = os.path.getsize(frame_path)
                except OSError:
                    file_size = None
                if (
                    file_size is not None
                    and file_size < CLUB_ANNOTATION_MIN_FILE_BYTES
                    and frame_visible_fraction < (CLUB_ANNOTATION_MIN_VISIBLE_FRAC * 6.0)
                ):
                    blackout_heavy = True
            else:
                blackout_heavy = True
            if blackout_heavy:
                blackout_outlier_frames.add(frame_idx)
                try:
                    os.remove(frame_path)
                except OSError:
                    pass
            else:
                annotated_visible_fractions[frame_idx] = visible_fraction
                if file_size is not None:
                    annotated_frame_sizes[frame_idx] = file_size

        relax_applicable = (
            in_window
            and (
                club_recording_enabled
                or len(club_pixels) < CLUB_ALPHA_RELAX_SAMPLE_TARGET
            )
        )
        if in_window and len(alpha_relax_info.setdefault("samples", [])) < 48:
            alpha_relax_info["samples"].append(
                {
                    "frame": frame_idx,
                    "visible": round(frame_visible_fraction, 6),
                    "club_samples": len(club_pixels),
                    "recording": bool(club_recording_enabled),
                    "miss": bool((not club_sample_created) or frame_blackout_heavy),
                    "relax_level": alpha_relax_level,
                }
            )
        if relax_applicable:
            miss_condition = (not club_sample_created) or frame_blackout_heavy
            if miss_condition:
                club_miss_streak += 1
            else:
                club_miss_streak = 0
            if (
                miss_condition
                and club_miss_streak >= CLUB_ALPHA_RELAX_STREAK
                and alpha_relax_level < CLUB_ALPHA_RELAX_MAX
            ):
                miss_count = club_miss_streak
                club_miss_streak = 0
                alpha_relax_level += 1
                prev_min_visible = float(alpha_mapper.min_visible_fraction)
                prev_threshold_scale = float(alpha_mapper.threshold_scale)
                prev_coverage = float(alpha_mapper.target_coverage)
                prev_guard = float(alpha_mapper.guard_margin)
                alpha_mapper.min_visible_fraction = min(
                    CLUB_ALPHA_RELAX_VISIBLE_CAP,
                    alpha_mapper.min_visible_fraction + CLUB_ALPHA_RELAX_VISIBLE_STEP,
                )
                alpha_mapper.threshold_scale = max(
                    CLUB_ALPHA_RELAX_THRESHOLD_SCALE_MIN,
                    alpha_mapper.threshold_scale * CLUB_ALPHA_RELAX_THRESHOLD_SCALE_FACTOR,
                )
                alpha_mapper.target_coverage = max(
                    CLUB_ALPHA_RELAX_MIN_COVERAGE,
                    alpha_mapper.target_coverage - CLUB_ALPHA_RELAX_COVERAGE_DROP,
                )
                alpha_mapper.guard_margin = max(
                    0.0, alpha_mapper.guard_margin - CLUB_ALPHA_RELAX_GUARD_MARGIN
                )
                alpha_relax_info["level"] = alpha_relax_level
                event = {
                    "frame": frame_idx,
                    "miss_streak": miss_count,
                    "visible_fraction": round(frame_visible_fraction, 6),
                    "club_samples": len(club_pixels),
                    "min_visible_before": round(prev_min_visible, 4),
                    "min_visible_after": round(alpha_mapper.min_visible_fraction, 4),
                    "threshold_scale_before": round(prev_threshold_scale, 4),
                    "threshold_scale_after": round(alpha_mapper.threshold_scale, 4),
                    "coverage_before": round(prev_coverage, 4),
                    "coverage_after": round(alpha_mapper.target_coverage, 4),
                    "guard_before": round(prev_guard, 4),
                    "guard_after": round(alpha_mapper.guard_margin, 4),
                    "level": alpha_relax_level,
                }
                alpha_relax_info["events"].append(event)
                print(
                    "Relaxed alpha mapping "
                    f"(level {alpha_relax_level}, frame {frame_idx}): "
                    f"visible_min {prev_min_visible:.3f}->{alpha_mapper.min_visible_fraction:.3f}, "
                    f"threshold_scale {prev_threshold_scale:.2f}->{alpha_mapper.threshold_scale:.2f}, "
                    f"coverage {prev_coverage:.2f}->{alpha_mapper.target_coverage:.2f}"
                )
        else:
            club_miss_streak = 0

        # No IR state to carry across frames
        frame_idx += 1

    cap.release()
    frame_loop_time = time.perf_counter() - frame_loop_start
    timings.add("frame_processing", frame_loop_time)
    post_start = time.perf_counter()

    if blackout_outlier_frames:
        removed_club_samples = 0
        filtered_club_pixels: list[dict[str, object]] = []
        for entry in club_pixels:
            frame_tag = entry.get("_frame")
            frame_key: int | None = None
            if frame_tag is not None:
                try:
                    frame_key = int(frame_tag)
                except (TypeError, ValueError):
                    frame_key = None
            if frame_key is not None and frame_key in blackout_outlier_frames:
                removed_club_samples += 1
                continue
            filtered_club_pixels.append(entry)
        if removed_club_samples:
            club_pixels = filtered_club_pixels
        blackout_filter_info["applied"] = True
        blackout_filter_info["frames_removed"] = len(blackout_outlier_frames)
        blackout_filter_info["club_samples_removed"] = removed_club_samples
        frame_list = sorted(blackout_outlier_frames)
        if frame_list:
            preview = frame_list[:12]
            detail = ", ".join(str(idx) for idx in preview)
            if len(preview) < len(frame_list):
                detail = f"{detail}, ..."
        else:
            detail = ""
        message = (
            f"Discarded {len(blackout_outlier_frames)} annotated frame(s) with excessive blackout"
        )
        if removed_club_samples:
            message += f"; removed {removed_club_samples} club sample(s)"
        if detail:
            message += f" (frames: {detail})"
        print(message)

    alpha_relax_info["applied"] = alpha_relax_level > 0
    alpha_relax_info["final_level"] = alpha_relax_level
    alpha_relax_info["final_min_visible_fraction"] = round(
        float(alpha_mapper.min_visible_fraction), 4
    )
    alpha_relax_info["final_threshold_scale"] = round(
        float(alpha_mapper.threshold_scale), 4
    )
    alpha_relax_info["final_target_coverage"] = round(
        float(alpha_mapper.target_coverage), 4
    )
    alpha_relax_info["final_guard_margin"] = round(
        float(alpha_mapper.guard_margin), 4
    )
    if alpha_relax_level > 0:
        print(
            "Alpha mapper relaxation summary: "
            f"applied {alpha_relax_level} time(s), "
            f"min_visible_fraction={alpha_relax_info['final_min_visible_fraction']}, "
            f"threshold_scale={alpha_relax_info['final_threshold_scale']}, "
            f"target_coverage={alpha_relax_info['final_target_coverage']}"
        )

    if club_pixels and (annotated_visible_fractions or annotated_frame_sizes):
        for entry in club_pixels:
            frame_tag = entry.get("_frame")
            if frame_tag is None:
                continue
            try:
                frame_key = int(frame_tag)
            except (TypeError, ValueError):
                continue
            visible_override = annotated_visible_fractions.get(frame_key)
            if visible_override is not None and np.isfinite(visible_override):
                entry["_visible_frac"] = float(visible_override)
            size_override = annotated_frame_sizes.get(frame_key)
            if size_override is not None:
                entry["_frame_size_bytes"] = int(size_override)

    ball_coords.sort(key=lambda c: c["time"])
    # Persist simple pixel-space club trajectory
    club_pixels.sort(key=lambda c: c["time"])
    original_club_pixels = [dict(entry) for entry in club_pixels]
    club_filter_info = {
        "applied": False,
        "total": len(club_pixels),
        "removed": 0,
        "width_min": None,
        "width_max": None,
        "area_min": None,
        "area_max": None,
        "depth_max": None,
        "depth_baseline_limit": None,
    }
    club_interpolation_info = {
        "added": 0,
        "start_frame": None,
        "target_frame": impact_frame_idx,
        "end_frame": impact_frame_idx,
    }
    club_depth_info = {
        "total": 0,
        "filled": 0,
        "fallback_used": False,
    }
    club_trim_info = {
        "removed": 0,
        "initial_removed": 0,
        "final_removed": 0,
        "min_frame": int(start_frame) if start_frame is not None else None,
        "max_frame": int(impact_frame_idx) if impact_frame_idx is not None else None,
    }
    if club_pixels:
        club_pixels, club_filter_info = filter_club_point_outliers(club_pixels)
        if club_filter_info.get("applied"):
            removed = club_filter_info.get("removed", 0)
            parts: list[str] = []
            width_min = club_filter_info.get("width_min")
            width_max = club_filter_info.get("width_max")
            area_min = club_filter_info.get("area_min")
            area_max = club_filter_info.get("area_max")
            depth_max = club_filter_info.get("depth_max")
            depth_baseline_limit = club_filter_info.get("depth_baseline_limit")
            visible_min = club_filter_info.get("visible_min")
            visible_max = club_filter_info.get("visible_max")
            if width_min is not None:
                parts.append(f"width<{width_min}")
            if width_max is not None:
                parts.append(f"width>{width_max}")
            if area_min is not None:
                parts.append(f"area<{area_min}")
            if area_max is not None:
                parts.append(f"area>{area_max}")
            if depth_max is not None:
                parts.append(f"depth>{depth_max}")
            if depth_baseline_limit is not None and depth_baseline_limit != depth_max:
                parts.append(f"depth>{depth_baseline_limit} vs baseline")
            if visible_min is not None:
                parts.append(f"visible_frac<{visible_min}")
            if visible_max is not None and visible_max < 1.0:
                parts.append(f"visible_frac>{visible_max}")
            clause = f" (limits: {', '.join(parts)})" if parts else ""
            print(f"Removed {removed} club outlier point(s){clause}")
            if frames_dir:
                remaining_frames: set[int] = set()
                for entry in club_pixels:
                    frame_tag = entry.get("_frame")
                    if frame_tag is None:
                        continue
                    try:
                        remaining_frames.add(int(frame_tag))
                    except (TypeError, ValueError):
                        continue
                visible_threshold = club_filter_info.get("visible_min")
                if visible_threshold is None:
                    visible_threshold = CLUB_ANNOTATION_MIN_VISIBLE_FRAC
                removed_frame_sample_counts: dict[int, int] = {}
                for entry in original_club_pixels:
                    frame_tag = entry.get("_frame")
                    if frame_tag is None:
                        continue
                    try:
                        frame_idx_local = int(frame_tag)
                    except (TypeError, ValueError):
                        continue
                    if frame_idx_local in remaining_frames:
                        continue
                    visible_metric: float | None = None
                    visible_val = entry.get("_visible_frac")
                    if visible_val is not None:
                        try:
                            visible_metric = float(visible_val)
                        except (TypeError, ValueError):
                            visible_metric = None
                    if visible_metric is None:
                        continue
                    if visible_threshold is not None and visible_metric > visible_threshold:
                        continue
                    removed_frame_sample_counts[frame_idx_local] = (
                        removed_frame_sample_counts.get(frame_idx_local, 0) + 1
                    )
                removal_targets = [
                    frame_idx_local
                    for frame_idx_local in removed_frame_sample_counts.keys()
                    if frame_idx_local not in blackout_outlier_frames
                ]
                if removal_targets:
                    removal_targets.sort()
                    preview = removal_targets[:12]
                    detail = ", ".join(str(idx) for idx in preview)
                    if len(preview) < len(removal_targets):
                        detail = f"{detail}, ..."
                    print(
                        "Discarded annotated frame(s) after club outlier filtering: "
                        f"{len(removal_targets)} frame(s)"
                        + (f" (frames: {detail})" if detail else "")
                    )
                    for frame_idx_local in removal_targets:
                        frame_path = os.path.join(frames_dir, f"frame_{frame_idx_local:04d}.png")
                        try:
                            os.remove(frame_path)
                        except OSError:
                            pass
                    blackout_outlier_frames.update(removal_targets)
                    removed_sample_total = sum(
                        removed_frame_sample_counts[frame_idx_local]
                        for frame_idx_local in removal_targets
                    )
                    blackout_filter_info["applied"] = True
                    blackout_filter_info["frames_removed"] = len(blackout_outlier_frames)
                    blackout_filter_info["club_samples_removed"] = (
                        blackout_filter_info.get("club_samples_removed", 0) + removed_sample_total
                    )
        frame_entries_map: dict[int, dict[str, object]] = {}
        frame_samples: list[int] = []
        for entry in club_pixels:
            frame_val = entry.get("_frame")
            if frame_val is None:
                continue
            try:
                frame_idx = int(frame_val)
            except (TypeError, ValueError):
                continue
            if frame_idx in frame_entries_map:
                continue
            frame_entries_map[frame_idx] = entry
            frame_samples.append(frame_idx)
        frame_samples.sort()

        good_start = None
        good_end = None
        if frame_samples:
            metric_keys = ["_raw_depth", "_width_px", "_area_px"]
            metric_stats: dict[str, tuple[float, float]] = {}
            for key in metric_keys:
                values = [
                    float(frame_entries_map[idx].get(key))
                    for idx in frame_samples
                    if frame_entries_map[idx].get(key) is not None
                    and _is_finite_number(frame_entries_map[idx].get(key))
                ]
                if len(values) >= 3:
                    arr = np.array(values, dtype=float)
                    median = float(np.median(arr))
                    mad = float(np.median(np.abs(arr - median)))
                    metric_stats[key] = (median, mad)

            def frame_quality(idx: int) -> float:
                entry = frame_entries_map.get(idx)
                if entry is None:
                    return 0.0
                total = 0.0
                used = 0
                for key, (median, mad) in metric_stats.items():
                    value = entry.get(key)
                    if value is None or not _is_finite_number(value):
                        continue
                    value_f = float(value)
                    scale = max(1e-3, mad * 1.4826, abs(median) * 0.1)
                    total += abs(value_f - median) / scale
                    used += 1
                if used == 0:
                    return 0.5
                avg = total / used
                return 1.0 / (1.0 + avg)

            quality_by_frame = {idx: frame_quality(idx) for idx in frame_samples}
            overall_min = frame_samples[0]
            overall_max = frame_samples[-1]
            window_span = int(CLUB_MAX_FRAME_WINDOW) if CLUB_MAX_FRAME_WINDOW else (overall_max - overall_min + 1)
            target_for_window: int | None = None
            if impact_frame_idx is not None:
                try:
                    target_for_window = int(impact_frame_idx)
                except (TypeError, ValueError):
                    target_for_window = None
            if target_for_window is None and frame_samples:
                target_for_window = frame_samples[-1]

            best_score = (-1.0, -1, -float("inf"), -float("inf"))
            best_start_candidate = overall_min
            for start_candidate in range(overall_min, overall_max + 1):
                end_candidate = start_candidate + window_span - 1
                quality_sum = 0.0
                count = 0
                for idx in frame_samples:
                    if idx < start_candidate:
                        continue
                    if idx > end_candidate:
                        break
                    quality_sum += quality_by_frame.get(idx, 0.0)
                    count += 1
                if count == 0:
                    continue
                avg_quality = quality_sum / count
                distance = 0.0
                if target_for_window is not None:
                    if end_candidate < target_for_window:
                        distance = target_for_window - end_candidate
                    elif start_candidate > target_for_window:
                        distance = start_candidate - target_for_window
                    else:
                        distance = 0.0
                score = (avg_quality, count, -float(distance), float(start_candidate))
                if score > best_score:
                    best_score = score
                    best_start_candidate = start_candidate
            best_end_candidate = best_start_candidate + window_span - 1
            good_start = next((idx for idx in frame_samples if idx >= best_start_candidate), frame_samples[0])
            good_end = next((idx for idx in reversed(frame_samples) if idx <= best_end_candidate), frame_samples[-1])

        if frame_samples and target_for_window is not None:
            target_for_window = int(target_for_window)
            span_limit = max(1, window_span)
            align_start = max(
                overall_min,
                min(target_for_window - span_limit + 1, overall_max - span_limit + 1),
            )
            align_end = align_start + span_limit - 1
            if align_end >= align_start:
                coverage = [
                    idx for idx in frame_samples if align_start <= idx <= align_end
                ]
                coverage_count = len(coverage)
                required = max(1, int(math.ceil(span_limit * CLUB_TARGET_ALIGN_MIN_COVERAGE)))
                if coverage_count >= required:
                    sum_quality = sum(quality_by_frame.get(idx, 0.0) for idx in coverage)
                    avg_quality = (sum_quality / coverage_count) if coverage_count else 0.0
                    align_score = (
                        avg_quality + CLUB_TARGET_ALIGN_BONUS,
                        coverage_count,
                        0.0,
                        float(align_start),
                    )
                    if align_score > best_score:
                        best_score = align_score
                        best_start_candidate = align_start

        if good_start is None and frame_samples:
            good_start = frame_samples[0]
        if good_end is None and frame_samples:
            good_end = frame_samples[-1]
        if good_end is not None and CLUB_FINAL_TRIM_MARGIN > 0:
            margin = int(CLUB_FINAL_TRIM_MARGIN)
            candidate_min = good_start if good_start is not None else good_end
            adjusted_end = max(int(candidate_min), int(good_end) - margin)
            if adjusted_end < int(good_end):
                good_end = adjusted_end
                club_trim_info["good_end_margin_applied"] = margin
        club_trim_info["good_start"] = good_start
        club_trim_info["good_end"] = good_end

        seed_start_frame = int(start_frame) if start_frame is not None else good_start
        club_trim_info["pre_fill_start"] = seed_start_frame

        trim_min_frame = good_start if good_start is not None else seed_start_frame
        initial_trim_max_frame = good_end

        club_pixels, initial_trim_info = trim_club_points_to_range(
            club_pixels,
            min_frame=trim_min_frame,
            max_frame=initial_trim_max_frame,
        )
        club_trim_info["initial_removed"] = initial_trim_info.get("removed", 0)
        club_trim_info["removed"] += club_trim_info["initial_removed"]
        if initial_trim_info.get("min_frame") is not None:
            club_trim_info["min_frame"] = initial_trim_info.get("min_frame")
        if initial_trim_info.get("max_frame") is not None:
            club_trim_info["max_frame"] = initial_trim_info.get("max_frame")

        interp_target_frame = None
        if impact_frame_idx is not None:
            try:
                interp_target_frame = int(impact_frame_idx)
            except (TypeError, ValueError):
                interp_target_frame = None
        if interp_target_frame is None:
            interp_target_frame = club_trim_info.get("max_frame")
        if interp_target_frame is not None and CLUB_FINAL_TRIM_MARGIN > 0:
            lower_bound = trim_min_frame
            if lower_bound is None:
                lower_bound = seed_start_frame
            if lower_bound is None:
                lower_bound = interp_target_frame
            adjusted_target = max(int(lower_bound), int(interp_target_frame) - int(CLUB_FINAL_TRIM_MARGIN))
            if adjusted_target < int(interp_target_frame):
                interp_target_frame = adjusted_target

        concavity_info = {
            "removed": 0,
            "max_score": 0.0,
        }
        if club_pixels:
            club_pixels, concavity_info = trim_tail_concavity_outliers(
                club_pixels,
                threshold=CLUB_TAIL_CONCAVITY_THRESHOLD,
                max_removals=CLUB_TAIL_CONCAVITY_MAX_REMOVALS,
                window=CLUB_TAIL_CONCAVITY_WINDOW,
                min_segment=CLUB_TAIL_CONCAVITY_MIN_SEGMENT,
            )
            if concavity_info.get("removed"):
                removed_concavity = concavity_info.get("removed", 0)
                club_trim_info["concavity_removed"] = removed_concavity
                club_trim_info["removed"] += removed_concavity
                print(
                    "Removed tail concavity outlier point(s): "
                    f"{removed_concavity} (max score {concavity_info.get('max_score', 0.0):.2f})"
                )
            if (
                original_club_pixels
                and len(club_pixels) <= 1
                and len(original_club_pixels) > len(club_pixels)
            ):
                print(
                    "Restoring raw club samples due to aggressive trimming; "
                    f"raw_count={len(original_club_pixels)}"
                )
                club_pixels = [dict(entry) for entry in original_club_pixels]
                club_trim_info["removed"] = 0
                club_trim_info["initial_removed"] = 0
                club_trim_info["final_removed"] = 0
                club_trim_info.pop("concavity_removed", None)
                concavity_info["reverted"] = True

        club_pixels, club_interpolation_info = interpolate_club_points(
            club_pixels,
            target_frame=interp_target_frame,
            fps=float(video_fps),
            start_frame=seed_start_frame,
        )
        if club_interpolation_info.get("added"):
            added = club_interpolation_info.get("added")
            tgt = club_interpolation_info.get("target_frame")
            print(f"Interpolated {added} club point(s) up to frame {tgt}")

        club_pixels, club_depth_info = enforce_club_depth_range(
            club_pixels,
            min_depth=CLUB_DEPTH_MIN_CM,
            max_depth=CLUB_DEPTH_MAX_CM,
        )
        if club_depth_info.get("filled") or club_depth_info.get("fallback_used"):
            print(
                "Adjusted club depth values: "
                f"filled={club_depth_info.get('filled')} fallback={club_depth_info.get('fallback_used')}"
            )

        club_pixels, final_trim_info = trim_club_points_to_range(
            club_pixels,
            min_frame=seed_start_frame,
            max_frame=interp_target_frame,
        )
        club_trim_info["final_removed"] = final_trim_info.get("removed", 0)
        club_trim_info["removed"] += club_trim_info["final_removed"]
        if final_trim_info.get("min_frame") is not None:
            club_trim_info["min_frame"] = final_trim_info.get("min_frame")
        if final_trim_info.get("max_frame") is not None:
            club_trim_info["max_frame"] = final_trim_info.get("max_frame")
        if club_trim_info.get("removed"):
            detail_parts = []
            if club_trim_info.get("initial_removed"):
                detail_parts.append(f"initial={club_trim_info['initial_removed']}")
            if club_trim_info.get("final_removed"):
                detail_parts.append(f"final={club_trim_info['final_removed']}")
            detail = f" ({', '.join(detail_parts)})" if detail_parts else ""
            print(
                "Trimmed club samples outside frame window: "
                f"removed={club_trim_info.get('removed')}" + detail
            )

    anchor_used = club_depth_anchor_samples_collected or len(club_depth_anchor_samples)
    club_depth_info["baseline_target"] = CLUB_DEPTH_ANCHOR_TARGET_CM
    club_depth_info["baseline_offset"] = (
        round(club_depth_offset, 2) if club_depth_offset is not None else None
    )
    club_depth_info["baseline_samples"] = int(anchor_used)
    smoothed_club_pixels, smoothing_info = smooth_sticker_pixels(club_pixels)
    if smoothing_info.get("applied"):
        club_pixels = smoothed_club_pixels
        print(
            "Smoothed sticker trajectory: "
            f"{smoothing_info['inliers']} inliers of {smoothing_info['total']} samples"
        )
    if club_pixels:
        for entry in club_pixels:
            z_value = entry.get("z")
            if not _is_finite_number(z_value):
                raw_depth_val = entry.get("_raw_depth")
                if _is_finite_number(raw_depth_val):
                    adjusted = float(raw_depth_val)
                    if club_depth_offset is not None:
                        adjusted += club_depth_offset
                    else:
                        adjusted = CLUB_DEPTH_ANCHOR_TARGET_CM
                    adjusted = max(0.0, adjusted)
                    entry["z"] = round(float(adjusted), 2)
                else:
                    entry.pop("z", None)
            else:
                entry["z"] = round(float(z_value), 2)
        for entry in club_pixels:
            entry.pop("ellipse_center", None)
            entry.pop("_left_edge_x", None)
            entry.pop("_width_px", None)
            entry.pop("_area_px", None)
            entry.pop("_visible_frac", None)
            entry.pop("_frame_size_bytes", None)
            entry.pop("_frame", None)
            entry.pop("_interpolated", None)
            entry.pop("_raw_depth", None)
            entry.pop("_depth_adjusted", None)
    if frames_dir:
        _annotate_sticker_frames(frames_dir, club_pixels)

    with open(ball_path, "w", encoding="utf-8") as f:
        json.dump(ball_coords, f, indent=2)
    with open(sticker_path, "w", encoding="utf-8") as f:
        json.dump(club_pixels, f, indent=2)
    post_time = time.perf_counter() - post_start
    timings.add("post_process", post_time)

    print(f"Saved {len(ball_coords)} ball points to {ball_path}")
    if club_pixels:
        print(f"Saved {len(club_pixels)} club path points to {sticker_path}")
    else:
        print("Warning: No club path points detected; sticker file left empty")
    print(f"Ball detection compile time: {ball_compile_time:.2f}s")
    print(f"Ball detection time: {ball_time:.2f}s")
    print(f"Clubface tracking time: {clubface_time:.2f}s")

    frame_other_time = max(0.0, frame_loop_time - ball_time - clubface_time)
    total_time = time.perf_counter() - total_start
    accounted = timings.total()
    unaccounted = max(0.0, total_time - accounted)
    print("Timing breakdown:")
    for name, duration in timings.items():
        pct = (duration / total_time * 100.0) if total_time > 0 else 0.0
        print(f"  {name:<24}: {duration:.2f}s ({pct:5.1f}%)")
    if frame_loop_time > 0:
        ball_pct = (ball_time / frame_loop_time * 100.0) if frame_loop_time > 0 else 0.0
        club_pct = (clubface_time / frame_loop_time * 100.0) if frame_loop_time > 0 else 0.0
        other_pct = (frame_other_time / frame_loop_time * 100.0) if frame_loop_time > 0 else 0.0
        print("  frame_processing details:")
        print(f"    ball_detection: {ball_time:.2f}s ({ball_pct:5.1f}%)")
        print(f"    club_tracking: {clubface_time:.2f}s ({club_pct:5.1f}%)")
        print(f"    frame_other:   {frame_other_time:.2f}s ({other_pct:5.1f}%)")
    if unaccounted > 0.005:
        pct = (unaccounted / total_time * 100.0) if total_time > 0 else 0.0
        print(f"  untracked: {unaccounted:.2f}s ({pct:5.1f}%)")
    print(f"  total: {total_time:.2f}s")
    return {
        "status": "ok",
        "video_path": video_path,
        "ball_path": ball_path,
        "sticker_path": sticker_path,
        "ball_points": len(ball_coords),
        "club_points": len(club_pixels),
        "club_filter": club_filter_info,
        "club_blackout": blackout_filter_info,
        "alpha_relax": alpha_relax_info,
        "club_interpolation": club_interpolation_info,
        "club_depth": club_depth_info,
        "club_trim": club_trim_info,
        "club_smoothing": smoothing_info,
        "impact_frame": impact_frame_idx,
        "impact_time": impact_time,
        "ball_compile_time": ball_compile_time,
        "ball_detection_time": ball_time,
        "clubface_time": clubface_time,
        "timings": {
            "total": total_time,
            "accounted": accounted,
            "unaccounted": unaccounted,
            "sections": timings.as_dict(),
            "frame": {
                "total": frame_loop_time,
                "ball_detection": ball_time,
                "club_tracking": clubface_time,
                "other": frame_other_time,
            },
        },
        "tail": {
            "ball_present": (tail_check.ball_present if tail_check else None),
            "hits": (tail_check.hits if tail_check else 0),
            "frames_checked": (tail_check.frames_checked if tail_check else 0),
            "scores": (tail_check.scores if tail_check else []),
            "frame_indices": (tail_check.frame_indices if tail_check else []),
            "score_threshold": tail_score_threshold,
            "min_hits": tail_min_hits,
        },
    }



if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "outdoor_130cm_3.mp4"
    ball_path = sys.argv[2] if len(sys.argv) > 2 else "ball_coords.json"
    sticker_path = sys.argv[3] if len(sys.argv) > 3 else "sticker_coords.json"
    frames_dir = sys.argv[4] if len(sys.argv) > 4 else "ball_frames"
    result = process_video(
        video_path,
        ball_path,
        sticker_path,
        frames_dir,
    )
    try:
        tail = result.get("tail", {}) if isinstance(result, dict) else {}
        print(
            "Tail ball present (CLI):",
            tail.get("ball_present"),
        )
    except Exception:
        pass
