"""Render detector boxes using the project's fixed helmet class mapping."""

from typing import Any

import numpy as np

from . import CLASS_NAMES

CLASS_COLORS = {
    0: (0, 200, 0),    # With Helmet: Màu Xanh lá cây (Green in BGR)
    1: (0, 0, 255),    # Without Helmet: Màu Đỏ tươi (Red in BGR)
}


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def draw_detections(
    frame: np.ndarray,
    result: Any,
    cv2_module: Any = None,
) -> tuple[np.ndarray, int]:
    """Return an annotated copy and detection count; never reorder class labels."""
    cv2_module = cv2_module or _require_cv2()

    annotated = frame.copy()
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return annotated, 0

    coordinates = _array(boxes.xyxy).reshape((-1, 4))
    classes = _array(boxes.cls).reshape(-1)
    confidences = _array(boxes.conf).reshape(-1)
    if not (len(coordinates) == len(classes) == len(confidences)):
        raise ValueError("Model returned inconsistent box, class, and confidence counts.")

    for coordinates_row, raw_class, raw_confidence in zip(coordinates, classes, confidences):
        class_id = int(raw_class)
        if class_id not in CLASS_NAMES:
            raise ValueError(f"Model returned unsupported class ID {class_id}.")
        _draw_box(annotated, coordinates_row, class_id, f"{CLASS_NAMES[class_id]} {float(raw_confidence):.2f}",
                  cv2_module)
    return annotated, len(coordinates)


def draw_tracks(
    frame: np.ndarray,
    objects: Any,
    cv2_module: Any = None,
) -> tuple[np.ndarray, int]:
    """Draw tracked objects as compact '#ID confidence' labels; colour encodes the smoothed class.

    A class legend is drawn in the bottom-left corner, so crowded scenes stay legible.
    """
    cv2_module = cv2_module or _require_cv2()
    annotated = frame.copy()
    count = 0
    for item in objects:
        if item.class_id not in CLASS_NAMES:
            raise ValueError(f"Tracker returned unsupported class ID {item.class_id}.")
        label = f"ID:{item.track_id} | {CLASS_NAMES[item.class_id]} {item.confidence:.2f}"
        _draw_box(annotated, item.xyxy, item.class_id, label, cv2_module)
        count += 1
    _draw_legend(annotated, cv2_module)
    return annotated, count


def _draw_legend(annotated: np.ndarray, cv2_module: Any) -> None:
    height = annotated.shape[0]
    for row, class_id in enumerate(sorted(CLASS_NAMES, reverse=True)):
        (text_w, text_h), baseline = cv2_module.getTextSize(
            CLASS_NAMES[class_id], cv2_module.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        bottom = height - 4 - row * (text_h + baseline + 6)
        cv2_module.rectangle(annotated, (4, bottom - text_h - baseline - 2), (8 + text_w, bottom),
                             CLASS_COLORS[class_id], -1)
        cv2_module.putText(annotated, CLASS_NAMES[class_id], (6, bottom - baseline - 1),
                           cv2_module.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2_module.LINE_AA)


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required for drawing detections. Install the project "
            "training extra with: python -m pip install -e \".[training]\""
        ) from error
    return cv2


def _draw_box(annotated: np.ndarray, coordinates_row: Any, class_id: int, label: str, cv2_module: Any) -> None:
    height, width = annotated.shape[:2]
    x1, y1, x2, y2 = (
        int(round(float(value)))
        for value in coordinates_row
    )
    x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
    y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
    color = CLASS_COLORS[class_id]
    cv2_module.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    # Filled label background keeps text legible over busy scenes and neighbouring boxes.
    (text_w, text_h), baseline = cv2_module.getTextSize(label, cv2_module.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    top = y1 - text_h - baseline - 2 if y1 - text_h - baseline - 2 >= 0 else min(height - 1, y2 + 1)
    cv2_module.rectangle(annotated, (x1, top), (min(width - 1, x1 + text_w + 2), top + text_h + baseline + 2),
                         color, -1)
    cv2_module.putText(
        annotated,
        label,
        (x1 + 1, top + text_h + 1),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2_module.LINE_AA,
    )
