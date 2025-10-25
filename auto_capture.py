import logging
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

from video_ball_detector import (
    TFLiteBallDetector,
    TailCheckResult,
    check_tail_for_ball,
)


logger = logging.getLogger(__name__)


@dataclass
class LowRateDetectionConfig:
    """Configuration for the low frame-rate watcher."""

    source_index: int = 0
    low_fps: float = 5.0
    max_wait_seconds: float = 90.0
    score_threshold: float = 0.25
    min_consecutive_hits: int = 2
    frame_resize: tuple[int, int] | None = None
    allow_video_loop: bool = True
    mock_video_path: Path | None = None


@dataclass
class HighSpeedCaptureConfig:
    """Parameters controlling the high frame-rate capture step."""

    duration_seconds: float = 5.0
    shutter_speed: int = 200
    width: int = 196
    height: int = 128
    framerate: int = 550
    camera_index: int = 0
    hflip: bool = True
    vflip: bool = True
    output_dir: Path = Path.home() / "Documents" / "webcamGolf"
    mock_sequence: tuple[Path, ...] = ()


@dataclass
class BallDetectionEvent:
    timestamp: float
    frame_index: int
    score: float
    frame: Optional[object]
    source: str


@dataclass
class CaptureCycleResult:
    final_video: Path
    tail: TailCheckResult
    attempts: int
    detection_event: Optional[BallDetectionEvent]
    all_videos: list[Path]


class BallWatcher:
    """Low frame-rate watcher that waits for the golf ball to appear."""

    def __init__(
        self,
        detector: TFLiteBallDetector,
        config: LowRateDetectionConfig,
    ) -> None:
        self.detector = detector
        self.config = config
        self._stop_event = threading.Event()
        self._event_queue: queue.Queue[BallDetectionEvent] = queue.Queue(maxsize=1)
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def wait_for_ball(self, timeout: Optional[float] = None) -> Optional[BallDetectionEvent]:
        if not self._thread or not self._thread.is_alive():
            self.start()
        try:
            return self._event_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _open_capture(self) -> cv2.VideoCapture:
        if self.config.mock_video_path:
            logger.debug("BallWatcher using mock source %s", self.config.mock_video_path)
            return cv2.VideoCapture(str(self.config.mock_video_path))
        return cv2.VideoCapture(self.config.source_index)

    def _run(self) -> None:
        cap = self._open_capture()
        if not cap.isOpened():
            logger.error("BallWatcher failed to open capture source")
            return

        fps_interval = 1.0 / self.config.low_fps if self.config.low_fps > 0 else 0.0
        start_time = time.monotonic()
        last_frame_time = 0.0
        hit_streak = 0
        frame_idx = 0
        video_looped = False
        status_interval = 3.0
        last_status_log = time.monotonic() - status_interval

        while not self._stop_event.is_set():
            if self.config.max_wait_seconds > 0:
                if (time.monotonic() - start_time) > self.config.max_wait_seconds:
                    logger.debug("BallWatcher timeout reached without detection")
                    break

            if fps_interval > 0.0:
                now = time.monotonic()
                sleep_for = fps_interval - (now - last_frame_time)
                if sleep_for > 0.0:
                    time.sleep(sleep_for)
                last_frame_time = time.monotonic()

            ok, frame = cap.read()
            if not ok or frame is None:
                if self.config.mock_video_path and self.config.allow_video_loop and not video_looped:
                    logger.debug("BallWatcher rewinding mock source")
                    cap.release()
                    cap = self._open_capture()
                    if not cap.isOpened():
                        logger.error("BallWatcher failed to reopen mock source")
                        break
                    video_looped = True
                    frame_idx = 0
                    continue
                break

            if self.config.frame_resize:
                frame = cv2.resize(frame, self.config.frame_resize, interpolation=cv2.INTER_AREA)

            detections = self.detector.detect(frame)
            best_score = max((det.get("score", 0.0) for det in detections), default=0.0)
            if best_score >= self.config.score_threshold:
                hit_streak += 1
                if hit_streak >= self.config.min_consecutive_hits:
                    event = BallDetectionEvent(
                        timestamp=time.time(),
                        frame_index=frame_idx,
                        score=float(best_score),
                        frame=frame.copy(),
                        source="mock" if self.config.mock_video_path else "camera",
                    )
                    logger.debug("BallWatcher detected ball with score %.3f", best_score)
                    self._put_event(event)
                    break
            else:
                hit_streak = 0

            now_status = time.monotonic()
            if now_status - last_status_log >= status_interval:
                logger.info(
                    "Low-rate capture active (frame=%d, best=%.3f, streak=%d)",
                    frame_idx,
                    best_score,
                    hit_streak,
                )
                last_status_log = now_status

            frame_idx += 1

        cap.release()

    def _put_event(self, event: BallDetectionEvent) -> None:
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            try:
                self._event_queue.get_nowait()
            except queue.Empty:
                pass
            self._event_queue.put_nowait(event)


class HighSpeedRecorder:
    """Wrapper that records high frame-rate clips using rpicam-vid or a mock sequence."""

    def __init__(self, config: HighSpeedCaptureConfig) -> None:
        self.config = config
        self._mock_queue: queue.Queue[Path] = queue.Queue()
        for path in config.mock_sequence:
            self._mock_queue.put(path)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def record_clip(self) -> Path:
        if not self._mock_queue.empty():
            clip = self._mock_queue.get()
            logger.debug("HighSpeedRecorder returning mock clip %s", clip)
            return clip

        output_path = self._next_output_path()
        cmd = self._build_command(output_path)
        logger.info("Recording high-speed clip to %s", output_path)
        status_interval = 1.0
        last_status_log = time.monotonic() - status_interval
        with subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as proc:
            try:
                while True:
                    retcode = proc.poll()
                    if retcode is not None:
                        if retcode != 0:
                            raise subprocess.CalledProcessError(retcode, cmd)
                        logger.info("High-speed recording complete: %s", output_path)
                        break
                    now_status = time.monotonic()
                    if now_status - last_status_log >= status_interval:
                        logger.info("High-speed recording in progress...")
                        last_status_log = now_status
                    time.sleep(0.5)
            except Exception:
                proc.kill()
                proc.wait()
                raise
        return output_path

    def _next_output_path(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        return self.config.output_dir / f"auto_{timestamp}.mp4"

    def _build_command(self, output_path: Path) -> list[str]:
        cmd = [
            "rpicam-vid",
            "--level",
            "4.2",
            "-t",
            f"{self.config.duration_seconds}s",
            "--camera",
            str(self.config.camera_index),
            "--width",
            str(self.config.width),
            "--height",
            str(self.config.height),
            "--no-raw",
            "--denoise",
            "cdn_off",
            "-o",
            str(output_path),
            "-n",
            "--shutter",
            str(self.config.shutter_speed),
            "--framerate",
            str(self.config.framerate),
        ]
        if self.config.hflip:
            cmd.append("--hflip")
        if self.config.vflip:
            cmd.append("--vflip")
        return cmd


class AutoCaptureManager:
    """Coordinate the low-fps watcher and high-speed capture loop."""

    def __init__(
        self,
        detector_model: str = "golf_ball_detector.tflite",
        low_config: Optional[LowRateDetectionConfig] = None,
        high_config: Optional[HighSpeedCaptureConfig] = None,
        tail_frames_to_check: int = 12,
        tail_stride: int = 1,
        tail_score_threshold: float = 0.25,
        tail_min_hits: int = 2,
        max_high_attempts: int = 3,
    ) -> None:
        env_low_source = os.environ.get("AUTO_CAPTURE_LOW_SOURCE")
        env_high_sequence = os.environ.get("AUTO_CAPTURE_HIGH_SEQUENCE")

        low_cfg = low_config or LowRateDetectionConfig()
        if env_low_source:
            low_cfg.mock_video_path = Path(env_low_source).expanduser()

        high_cfg = high_config or HighSpeedCaptureConfig()
        if env_high_sequence:
            paths = [Path(p.strip()).expanduser() for p in env_high_sequence.split(os.pathsep) if p.strip()]
            high_cfg = HighSpeedCaptureConfig(
                duration_seconds=high_cfg.duration_seconds,
                shutter_speed=high_cfg.shutter_speed,
                width=high_cfg.width,
                height=high_cfg.height,
                framerate=high_cfg.framerate,
                camera_index=high_cfg.camera_index,
                hflip=high_cfg.hflip,
                vflip=high_cfg.vflip,
                output_dir=high_cfg.output_dir,
                mock_sequence=tuple(paths),
            )

        self.detector = TFLiteBallDetector(detector_model, conf_threshold=0.01)
        self.low_config = low_cfg
        self.high_config = high_cfg
        self.tail_frames_to_check = tail_frames_to_check
        self.tail_stride = tail_stride
        self.tail_score_threshold = tail_score_threshold
        self.tail_min_hits = tail_min_hits
        self.max_high_attempts = max(1, max_high_attempts)
        self.watcher = BallWatcher(self.detector, self.low_config)
        self.recorder = HighSpeedRecorder(self.high_config)

    def acquire_clip(self) -> CaptureCycleResult:
        logger.info("Waiting for ball appearance via low-rate watcher")
        detection_event = self.watcher.wait_for_ball(timeout=self.low_config.max_wait_seconds)
        self.watcher.stop()
        if detection_event is None:
            raise TimeoutError("Ball detection timed out before capture trigger")
        attempts = 0
        captured: list[Path] = []
        tail_result: Optional[TailCheckResult] = None
        final_clip: Optional[Path] = None

        while attempts < self.max_high_attempts:
            attempts += 1
            video_path = self.recorder.record_clip()
            captured.append(video_path)
            tail_result = check_tail_for_ball(
                str(video_path),
                detector=self.detector,
                frames_to_check=self.tail_frames_to_check,
                stride=self.tail_stride,
                score_threshold=self.tail_score_threshold,
                min_hits=self.tail_min_hits,
            )
            logger.info(
                "Tail check attempt %d: hits=%d/%d present=%s",
                attempts,
                tail_result.hits,
                tail_result.frames_checked,
                tail_result.ball_present,
            )
            if not tail_result.ball_present:
                final_clip = video_path
                break

        if final_clip is None or tail_result is None:
            raise RuntimeError("Ball remained visible after maximum high-speed attempts; aborting.")

        return CaptureCycleResult(
            final_video=final_clip,
            tail=tail_result,
            attempts=attempts,
            detection_event=detection_event,
            all_videos=captured,
        )
