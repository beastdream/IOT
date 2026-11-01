"""Sequential video inference with bounded memory and guaranteed resource release."""

from collections.abc import Callable
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import time
from typing import Any

from smart_helmet.paths import PREDICTIONS_DIR

from .device import DeviceInfo, resolve_device, verify_inference_device
from .tracking import HelmetTracker, TrackingConfig
from .visualization import draw_detections, draw_tracks

LOGGER = logging.getLogger(__name__)
ProgressCallback = Callable[[int, int | None, float], None]


@dataclass(frozen=True)
class VideoInferenceResult:
    source: Path
    output: Path
    frames_processed: int
    total_frames: int | None
    detections: int
    fps: float
    source_fps: float
    device: str
    device_verified: bool
    audio_preserved: bool = False
    unique_tracks: int | None = None
    scene_cuts: int | None = None


def _opencv() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required for video inference. Install the project training "
            "extra with: python -m pip install -e \".[training]\""
        ) from error
    return cv2


def _make_output_path(
    source: Path,
    output: str | Path | None,
    overwrite: bool,
) -> tuple[Path, str]:
    if output is not None:
        target = Path(output).expanduser().resolve()
        if target == source:
            raise ValueError("Output video must not overwrite the source video.")
        suffix = target.suffix.lower()
        if suffix not in {".mp4", ".avi"}:
            raise ValueError("Output video must use the .mp4 or .avi extension.")
        if target.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {target}. Use --overwrite to replace it.")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target, "mp4v" if suffix == ".mp4" else "XVID"

    directory = PREDICTIONS_DIR / "video"
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / f"{source.stem}_helmet"
    target = directory / f"{source.stem}_helmet.mp4"
    counter = 1
    while target.exists():
        target = directory / f"{source.stem}_helmet_{counter}.mp4"
        counter += 1
    return target, "mp4v"


def _open_writer(cv2: Any, target: Path, codec: str, fps: float, size: tuple[int, int]) -> Any:
    """Prefer H.264 via Windows Media Foundation for .mp4 so common players can open it.

    MPEG-4 Part 2 (mp4v) does not play in browsers, VS Code or Windows Media Player;
    it remains the fallback when Media Foundation is unavailable.
    """
    if codec == "mp4v" and hasattr(cv2, "CAP_MSMF"):
        writer = cv2.VideoWriter(str(target), cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"avc1"), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    return cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*codec), fps, size)


def process_video(
    source: str | Path,
    model: Any,
    *,
    confidence: float = 0.25,
    image_size: int = 640,
    device: DeviceInfo | str = "auto",
    output: str | Path | None = None,
    max_frames: int | None = None,
    overwrite: bool = False,
    progress_callback: ProgressCallback | None = None,
    tracking: TrackingConfig | None = None,
) -> VideoInferenceResult:
    """Annotate a video sequentially; output video is silent (audio is not copied).

    With ``tracking``, detections run at the tracker's low threshold and each drawn
    object carries a persistent ID and a per-track smoothed class; ``confidence`` is
    then unused (apply it via ``TrackingConfig.with_confidence``).
    """
    video_path = Path(source).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if not 0 <= confidence <= 1:
        raise ValueError("Confidence threshold must be between 0 and 1.")
    if image_size <= 0:
        raise ValueError("Image size must be a positive integer.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("Maximum frame count must be a positive integer.")

    cv2 = _opencv()
    selection = device if isinstance(device, DeviceInfo) else resolve_device(device)
    target, codec = _make_output_path(video_path, output, overwrite)
    capture = cv2.VideoCapture(str(video_path))
    writer = None
    processed = 0
    detections = 0
    track_ids: set[int] = set()
    tracker = None
    actual_device = selection.device
    device_verified = False
    started = time.monotonic()
    output_existed = target.exists()
    completed = False
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open the input video: {video_path}")

        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("Input video does not report a valid FPS; cannot preserve playback speed.")
        raw_total = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = int(raw_total) if math.isfinite(raw_total) and raw_total > 0 else None
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"Input video contains no readable frames: {video_path}")

        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError("Input video returned a frame with invalid dimensions.")
        writer = _open_writer(cv2, target, codec, source_fps, (width, height))
        if not writer.isOpened():
            raise OSError(
                f"Could not initialize the video writer for {target}; "
                "check that an H.264/MP4V/XVID codec is available."
            )

        if tracking is not None:
            tracker = HelmetTracker(tracking, source_fps, cv2)
        threshold = tracking.track_low_thresh if tracking is not None else confidence
        LOGGER.info(
            "Starting video inference for %s (%s FPS, %s frames)",
            video_path, source_fps, total_frames if total_frames is not None else "unknown",
        )
        while ok and (max_frames is None or processed < max_frames):
            predictions = list(model.predict(
                source=frame,
                conf=threshold,
                imgsz=image_size,
                device=selection.device_argument,
                batch=1,
                verbose=False,
                **({"agnostic_nms": True} if tracking is not None and tracking.agnostic_nms else {}),
            ))
            if not predictions:
                raise RuntimeError(f"Detector returned no result for video frame {processed + 1}.")
            if tracker is None:
                annotated, count = draw_detections(frame, predictions[0], cv2)
            else:
                boxes = predictions[0].boxes
                objects = tracker.update(frame, None if boxes is None else boxes.cpu().numpy())
                track_ids.update(item.track_id for item in objects)
                annotated, count = draw_tracks(frame, objects, cv2)
            writer.write(annotated)
            detections += count
            processed += 1
            if processed == 1:
                actual_device, device_verified = verify_inference_device(model, selection)
            elapsed = max(time.monotonic() - started, 1e-9)
            if progress_callback is not None:
                progress_callback(processed, total_frames, processed / elapsed)
            if max_frames is not None and processed >= max_frames:
                break
            ok, frame = capture.read()

        writer.release()
        writer = None
        if not target.is_file() or target.stat().st_size == 0:
            raise OSError(f"Video writer produced no output file at {target}")
        completed = True
    finally:
        if writer is not None:
            writer.release()
        capture.release()
        if not completed and not output_existed and target.exists():
            try:
                target.unlink()
            except OSError:
                LOGGER.exception("Could not remove incomplete video output %s", target)

    elapsed = max(time.monotonic() - started, 1e-9)
    processing_fps = processed / elapsed
    LOGGER.info(
        "Video inference finished: %d frames, %.2f FPS, %d detections; output=%s; audio not preserved",
        processed, processing_fps, detections, target,
    )
    return VideoInferenceResult(
        video_path, target, processed, total_frames, detections, processing_fps,
        source_fps, actual_device, device_verified,
        unique_tracks=len(track_ids) if tracker is not None else None,
        scene_cuts=tracker.scene_cuts if tracker is not None else None,
    )
