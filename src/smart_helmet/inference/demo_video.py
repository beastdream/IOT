"""Build a local demo video from curated dataset images (no external footage).

Each image is letterboxed onto a fixed canvas and shown for a fixed duration with
a gentle centre zoom, so consecutive frames differ like a slowly moving camera.
TEST is locked and cannot be used as a source split.
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import random
from typing import Any

from smart_helmet.foundation import load_detection_config
from smart_helmet.paths import CURATED_V1_CONFIG, PREDICTIONS_DIR, PROJECT_ROOT

from .video_inference import _opencv

LOGGER = logging.getLogger(__name__)
ALLOWED_SPLITS = ("train", "val")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class DemoVideoResult:
    output: Path
    split: str
    images: int
    frames: int
    fps: float
    size: tuple[int, int]
    manifest: Path


def split_images(split: str, config_path: Path = CURATED_V1_CONFIG) -> list[Path]:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Split {split!r} is not allowed; TEST is locked. Use one of {ALLOWED_SPLITS}.")
    data = load_detection_config(config_path, PROJECT_ROOT)
    directory = Path(data[split])
    images = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No images found in {directory}")
    return images


def _letterbox(image: Any, width: int, height: int, cv2: Any, np: Any) -> tuple[Any, dict]:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    placement = dict(left=left, top=top, width=resized.shape[1], height=resized.shape[0])
    return canvas, placement


def _crop_window(width: int, height: int, factor: float) -> tuple[int, int, int, int]:
    if factor <= 1.0:
        return 0, 0, width, height
    crop_w, crop_h = round(width / factor), round(height / factor)
    return (width - crop_w) // 2, (height - crop_h) // 2, crop_w, crop_h


def _zoom(canvas: Any, factor: float, cv2: Any) -> Any:
    height, width = canvas.shape[:2]
    left, top, crop_w, crop_h = _crop_window(width, height, factor)
    if (crop_w, crop_h) == (width, height):
        return canvas
    crop = canvas[top:top + crop_h, left:left + crop_w]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)


def zoom_factor(index: int, frames_per_image: int, max_zoom: float) -> float:
    progress = index / max(frames_per_image - 1, 1)
    return 1.0 + (max_zoom - 1.0) * progress


def project_yolo_box(
    box: tuple[float, float, float, float],
    placement: dict,
    size: tuple[int, int],
    factor: float,
) -> tuple[float, float, float, float] | None:
    """Map a normalized YOLO box onto a demo frame; None when it leaves the frame."""
    width, height = size
    cx, cy, bw, bh = box
    x1 = placement["left"] + (cx - bw / 2) * placement["width"]
    y1 = placement["top"] + (cy - bh / 2) * placement["height"]
    x2 = placement["left"] + (cx + bw / 2) * placement["width"]
    y2 = placement["top"] + (cy + bh / 2) * placement["height"]
    left, top, crop_w, crop_h = _crop_window(width, height, factor)
    sx, sy = width / crop_w, height / crop_h
    x1, x2 = (x1 - left) * sx, (x2 - left) * sx
    y1, y2 = (y1 - top) * sy, (y2 - top) * sy
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(width), x2), min(float(height), y2)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return x1, y1, x2, y2


def label_path_for(image: Path) -> Path:
    return image.parent.parent / "labels" / f"{image.stem}.txt"


def _default_output(split: str) -> Path:
    directory = PREDICTIONS_DIR / "video_sources"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"demo_{split}.mp4"
    counter = 1
    while target.exists():
        target = directory / f"demo_{split}_{counter}.mp4"
        counter += 1
    return target


def build_demo_video(
    images: list[Path],
    *,
    split: str,
    output: str | Path | None = None,
    fps: float = 15.0,
    seconds_per_image: float = 1.0,
    size: tuple[int, int] = (416, 416),
    max_zoom: float = 1.1,
    limit: int | None = None,
    shuffle_seed: int | None = None,
    overwrite: bool = False,
) -> DemoVideoResult:
    """Write images as an MP4 slideshow; returns frame/image counts actually written."""
    if fps <= 0 or seconds_per_image <= 0:
        raise ValueError("FPS and seconds per image must be positive.")
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError("Video size must be positive.")
    if max_zoom < 1.0:
        raise ValueError("Maximum zoom must be at least 1.0.")
    if limit is not None and limit <= 0:
        raise ValueError("Image limit must be a positive integer.")
    selected = list(images)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(selected)
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No images selected for the demo video.")

    if output is None:
        target = _default_output(split)
    else:
        target = Path(output).expanduser().resolve()
        if target.suffix.lower() != ".mp4":
            raise ValueError("Demo video output must use the .mp4 extension.")
        if target.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {target}. Use --overwrite to replace it.")
        target.parent.mkdir(parents=True, exist_ok=True)

    cv2 = _opencv()
    import numpy as np

    width, height = size
    frames_per_image = max(1, round(fps * seconds_per_image))
    output_existed = target.exists()
    writer =cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    manifest_path = target.with_suffix(".json")
    written_frames = 0
    segments = []
    completed = False
    try:
        if not writer.isOpened():
            raise OSError(f"Could not initialize the video writer for {target}")
        for path in selected:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                LOGGER.warning("Skipping unreadable image %s", path)
                continue
            canvas, placement = _letterbox(image, width, height, cv2, np)
            segments.append(dict(image=str(path), label=str(label_path_for(path)),
                                 start_frame=written_frames, frames=frames_per_image, placement=placement))
            for index in range(frames_per_image):
                writer.write(_zoom(canvas, zoom_factor(index, frames_per_image, max_zoom), cv2))
                written_frames += 1
        if not segments:
            raise ValueError("None of the selected images could be read.")
        writer.release()
        if not target.is_file() or target.stat().st_size == 0:
            raise OSError(f"Video writer produced no output file at {target}")
        # Sidecar manifest lets tracking evaluation reconstruct exact per-frame ground truth.
        manifest_path.write_text(json.dumps(dict(
            video=str(target), split=split, fps=fps, size=[width, height], max_zoom=max_zoom,
            frames_per_image=frames_per_image, total_frames=written_frames, segments=segments,
        ), indent=2), encoding="utf-8")
        completed = True
    finally:
        writer.release()
        if not completed and not output_existed and target.exists():
            target.unlink()
    written_images = len(segments)

    LOGGER.info("Demo video written: %s (%d images, %d frames)", target, written_images, written_frames)
    return DemoVideoResult(target, split, written_images, written_frames, fps, size, manifest_path)
