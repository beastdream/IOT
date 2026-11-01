"""Run single-image helmet inference and save annotated output."""

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any

from smart_helmet.paths import PREDICTIONS_DIR

from .device import DeviceInfo, resolve_device, verify_inference_device
from .visualization import draw_detections

LOGGER = logging.getLogger(__name__)
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageInferenceResult:
    source: Path
    output: Path
    detections: int
    duration_seconds: float
    device: str
    device_verified: bool


def _opencv() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required for image inference. Install the project training "
            "extra with: python -m pip install -e \".[training]\""
        ) from error
    return cv2


def _output_path(source: Path, output: str | Path | None, overwrite: bool) -> Path:
    if output is not None:
        target = Path(output).expanduser().resolve()
        if target == source:
            raise ValueError("Output image must not overwrite the source image.")
        if target.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {target}. Use --overwrite to replace it.")
        if target.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError("Output image extension must be JPG, JPEG, PNG, BMP, or WEBP.")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    directory = PREDICTIONS_DIR / "image"
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / f"{source.stem}_helmet"
    target = directory / f"{source.stem}_helmet{source.suffix.lower()}"
    counter = 1
    while target.exists():
        target = directory / f"{source.stem}_helmet_{counter}{source.suffix.lower()}"
        counter += 1
    return target


def predict_image(
    source: str | Path,
    model: Any,
    *,
    confidence: float = 0.25,
    image_size: int = 640,
    device: DeviceInfo | str = "auto",
    output: str | Path | None = None,
    overwrite: bool = False,
) -> ImageInferenceResult:
    """Detect helmet status in one image and save its annotated copy."""
    image_path = Path(source).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {image_path.suffix or '(no extension)'}")
    if not 0 <= confidence <= 1:
        raise ValueError("Confidence threshold must be between 0 and 1.")
    if image_size <= 0:
        raise ValueError("Image size must be a positive integer.")

    cv2 = _opencv()
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"OpenCV could not decode the input image: {image_path}")
    selection = device if isinstance(device, DeviceInfo) else resolve_device(device)
    target = _output_path(image_path, output, overwrite)
    started = time.monotonic()
    LOGGER.info("Starting image inference for %s", image_path)
    predictions = list(model.predict(
        source=frame,
        conf=confidence,
        imgsz=image_size,
        device=selection.device_argument,
        batch=1,
        verbose=False,
    ))
    if not predictions:
        raise RuntimeError("The detector returned no result for the input image.")
    annotated, detection_count = draw_detections(frame, predictions[0], cv2)
    actual_device, verified = verify_inference_device(model, selection)
    if not cv2.imwrite(str(target), annotated):
        raise OSError(f"Could not save annotated image to {target}")
    duration = time.monotonic() - started
    LOGGER.info(
        "Image inference finished in %.3fs: %d detections; output=%s",
        duration, detection_count, target,
    )
    return ImageInferenceResult(
        image_path, target, detection_count, duration, actual_device, verified,
    )
