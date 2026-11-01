"""Identity/label stability evaluation on manifest-backed demo videos.

Ground truth is reconstructed exactly from the demo manifest (source YOLO labels
projected through the same letterbox and zoom used to render each frame).
Matching is class-agnostic Hungarian assignment at IoU >= 0.5 per frame.

Definitions:
- id_switches (CLEAR-MOT): a GT object is matched to a different output ID than
  at its previous match.
- id_transfers: an output ID is matched to a different GT object than at its
  previous match, i.e. the ID jumped onto another object; scene_id_transfers is the
  subset where the new object belongs to another scene (a similar object in a new
  image inheriting an old ID).
- class_flips: the displayed class of a GT object changes between consecutive
  matched frames.
"""

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from smart_helmet.inference.demo_video import ALLOWED_SPLITS, project_yolo_box, zoom_factor
from smart_helmet.inference.tracking import HelmetTracker, TrackingConfig


@dataclass(frozen=True)
class GroundTruth:
    key: tuple[int, int]  # (scene index, object index)
    class_id: int
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class Output:
    track_id: int | None
    class_id: int
    xyxy: tuple[float, float, float, float]


def load_ground_truth(manifest_path: str | Path) -> tuple[dict, list[list[GroundTruth]]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("split") not in ALLOWED_SPLITS:
        raise ValueError("Tracking evaluation accepts TRAIN/VALID demo videos only; TEST is locked.")
    size = tuple(manifest["size"])
    frames: list[list[GroundTruth]] = [[] for _ in range(manifest["total_frames"])]
    for scene, segment in enumerate(manifest["segments"]):
        objects = []
        label = Path(segment["label"])
        if label.is_file():
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 5:
                    objects.append((int(parts[0]), tuple(float(value) for value in parts[1:])))
        for index in range(segment["frames"]):
            factor = zoom_factor(index, manifest["frames_per_image"], manifest["max_zoom"])
            for number, (class_id, box) in enumerate(objects):
                xyxy = project_yolo_box(box, segment["placement"], size, factor)
                if xyxy is not None:
                    frames[segment["start_frame"] + index].append(GroundTruth((scene, number), class_id, xyxy))
    return manifest, frames


def _iou_matrix(first: list, second: list) -> np.ndarray:
    if not first or not second:
        return np.zeros((len(first), len(second)))
    a = np.asarray([item.xyxy for item in first], dtype=float)[:, None, :]
    b = np.asarray([item.xyxy for item in second], dtype=float)[None, :, :]
    width = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    height = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = width * height
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def compute_metrics(
    gt_frames: list[list[GroundTruth]],
    output_frames: list[list[Output]],
    iou_threshold: float = 0.5,
) -> dict:
    from scipy.optimize import linear_sum_assignment

    if len(gt_frames) != len(output_frames):
        raise ValueError("Ground truth and outputs cover different frame counts.")
    tracked = any(out.track_id is not None for frame in output_frames for out in frame)
    gt_total = matched = outputs = switches = transfers = scene_transfers = flips = 0
    class_correct = without_total = without_shown = 0
    last_id: dict[tuple[int, int], int] = {}
    last_gt: dict[int, tuple[int, int]] = {}
    last_class: dict[tuple[int, int], int] = {}
    ids_per_gt: dict[tuple[int, int], set] = defaultdict(set)
    gts_per_id: dict[int, set] = defaultdict(set)
    for gts, outs in zip(gt_frames, output_frames):
        gt_total += len(gts)
        outputs += len(outs)
        without_total += sum(gt.class_id == 1 for gt in gts)
        ious = _iou_matrix(gts, outs)
        pairs = []
        if ious.size:
            rows, cols = linear_sum_assignment(-ious)
            pairs = [(r, c) for r, c in zip(rows, cols) if ious[r, c] >= iou_threshold]
        matched += len(pairs)
        for row, col in pairs:
            gt, out = gts[row], outs[col]
            if out.track_id is not None:
                if gt.key in last_id and last_id[gt.key] != out.track_id:
                    switches += 1
                previous = last_gt.get(out.track_id)
                if previous is not None and previous != gt.key:
                    transfers += 1
                    scene_transfers += previous[0] != gt.key[0]
                last_id[gt.key] = out.track_id
                last_gt[out.track_id] = gt.key
                ids_per_gt[gt.key].add(out.track_id)
                gts_per_id[out.track_id].add(gt.key)
            if gt.key in last_class and last_class[gt.key] != out.class_id:
                flips += 1
            last_class[gt.key] = out.class_id
            class_correct += out.class_id == gt.class_id
            without_shown += gt.class_id == 1 and out.class_id == 1
    misses, false_positives = gt_total - matched, outputs - matched
    return dict(
        tracked=tracked,
        gt_boxes=gt_total, gt_objects=len({gt.key for frame in gt_frames for gt in frame}),
        recall=matched / gt_total if gt_total else 0.0,
        precision=matched / outputs if outputs else 0.0,
        false_positives=false_positives,
        id_switches=switches if tracked else None,
        id_transfers=transfers if tracked else None,
        scene_id_transfers=scene_transfers if tracked else None,
        ids_spanning_multiple_objects=sum(len(keys) > 1 for keys in gts_per_id.values()) if tracked else None,
        mean_ids_per_object=float(np.mean([len(ids) for ids in ids_per_gt.values()])) if ids_per_gt else None,
        mota=1 - (misses + false_positives + (switches if tracked else 0)) / gt_total if gt_total else None,
        class_flips=flips,
        class_accuracy=class_correct / matched if matched else 0.0,
        without_helmet_shown_recall=without_shown / without_total if without_total else 0.0,
    )


def run_sequence(
    video: str | Path,
    model: Any,
    *,
    config: TrackingConfig | None,
    confidence: float,
    image_size: int,
    device_argument: str,
    cv2: Any,
) -> tuple[list[list[Output]], dict]:
    """Run detection (and optional tracking) over every frame without writing video."""
    capture = cv2.VideoCapture(str(video))
    frames: list[list[Output]] = []
    tracker = None
    started = time.monotonic()
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open the video: {video}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if config is not None:
            tracker = HelmetTracker(config, fps, cv2)
        threshold = config.track_low_thresh if config is not None else confidence
        extra = {"agnostic_nms": True} if config is not None and config.agnostic_nms else {}
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            result = list(model.predict(source=frame, conf=threshold, imgsz=image_size,
                                        device=device_argument, batch=1, verbose=False, **extra))[0]
            if tracker is None:
                boxes = result.boxes
                outputs = [] if boxes is None else [
                    Output(None, int(c), tuple(float(v) for v in xyxy))
                    for xyxy, c in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy())
                ]
            else:
                boxes = None if result.boxes is None else result.boxes.cpu().numpy()
                outputs = [Output(obj.track_id, obj.class_id, obj.xyxy) for obj in tracker.update(frame, boxes)]
            frames.append(outputs)
    finally:
        capture.release()
    elapsed = max(time.monotonic() - started, 1e-9)
    return frames, dict(processing_fps=len(frames) / elapsed,
                        scene_cuts=tracker.scene_cuts if tracker is not None else None)
