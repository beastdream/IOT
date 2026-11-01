"""Identity tracking for video inference: stable IDs, scene-cut resets, class smoothing.

The Ultralytics ByteTrack/BoT-SORT association is reused, but this wrapper owns the
tracker lifecycle so that:
- display IDs are never reissued, even after the tracker is reset;
- a hard scene cut resets association, so a similar-looking object in the new
  scene cannot inherit an ID from the previous scene;
- lost-track memory is configured in seconds and converted using the source FPS;
- the shown helmet class is smoothed per track instead of flickering per frame;
- optional class-agnostic NMS keeps one box per head instead of overlapping
  "With Helmet" and "Without Helmet" boxes that would become two tracks.
"""

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import CLASS_NAMES

TRACKER_TYPES = ("bytetrack", "botsort")


@dataclass(frozen=True)
class TrackingConfig:
    tracker_type: str = "bytetrack"
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer_seconds: float = 1.0
    match_thresh: float = 0.8
    fuse_score: bool = True
    gmc_method: str = "sparseOptFlow"
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.8
    agnostic_nms: bool = False
    # Scene-cut reset; None disables. A cut is declared when the colour-histogram
    # correlation drops below the threshold or the mean grey-level change exceeds it.
    scene_cut_hist_corr: float | None = None
    scene_cut_mean_diff: float | None = None
    # Per-track class smoothing; None shows the raw per-frame class.
    class_smoothing: float | None = None
    class_hysteresis: float = 0.0

    def __post_init__(self) -> None:
        if self.tracker_type not in TRACKER_TYPES:
            raise ValueError(f"tracker_type must be one of {TRACKER_TYPES}")
        if not 0 <= self.track_low_thresh <= self.track_high_thresh <= 1:
            raise ValueError("Require 0 <= track_low_thresh <= track_high_thresh <= 1")
        if not 0 <= self.new_track_thresh <= 1 or not 0 < self.match_thresh <= 1:
            raise ValueError("new_track_thresh must be in [0, 1] and match_thresh in (0, 1]")
        if self.track_buffer_seconds < 0:
            raise ValueError("track_buffer_seconds must be non-negative")
        if self.scene_cut_hist_corr is not None and not -1 <= self.scene_cut_hist_corr <= 1:
            raise ValueError("scene_cut_hist_corr must be in [-1, 1]")
        if self.scene_cut_mean_diff is not None and not 0 < self.scene_cut_mean_diff <= 255:
            raise ValueError("scene_cut_mean_diff must be in (0, 255]")
        if self.class_smoothing is not None and not 0 <= self.class_smoothing < 1:
            raise ValueError("class_smoothing must be in [0, 1)")
        if self.class_hysteresis < 0:
            raise ValueError("class_hysteresis must be non-negative")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrackingConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Tracking config must be a mapping: {path}")
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown tracking config keys in {path}: {sorted(unknown)}")
        return cls(**data)

    def with_confidence(self, confidence: float) -> "TrackingConfig":
        """Apply a CLI confidence as the threshold for confident matches and new tracks."""
        return replace(self, track_high_thresh=confidence, new_track_thresh=confidence,
                       track_low_thresh=min(self.track_low_thresh, confidence))

    def framework_args(self, fps: float) -> dict:
        return dict(
            tracker_type=self.tracker_type, track_high_thresh=self.track_high_thresh,
            track_low_thresh=self.track_low_thresh, new_track_thresh=self.new_track_thresh,
            track_buffer=max(1, round(self.track_buffer_seconds * fps)), match_thresh=self.match_thresh,
            fuse_score=self.fuse_score, gmc_method=self.gmc_method, proximity_thresh=self.proximity_thresh,
            appearance_thresh=self.appearance_thresh, with_reid=False, model="auto",
        )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrackedObject:
    track_id: int
    xyxy: tuple[float, float, float, float]
    class_id: int
    raw_class_id: int
    confidence: float


class SceneCutDetector:
    """Detect hard cuts from consecutive downscaled frames."""

    def __init__(self, hist_corr: float | None, mean_diff: float | None, cv2: Any):
        self.hist_corr = hist_corr
        self.mean_diff = mean_diff
        self.cv2 = cv2
        self._previous: tuple[Any, Any] | None = None

    @property
    def enabled(self) -> bool:
        return self.hist_corr is not None or self.mean_diff is not None

    def is_cut(self, frame: np.ndarray) -> bool:
        if not self.enabled:
            return False
        cv2 = self.cv2
        small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        previous, self._previous = self._previous, (hist, grey)
        if previous is None:
            return False
        if self.hist_corr is not None and cv2.compareHist(previous[0], hist, cv2.HISTCMP_CORREL) < self.hist_corr:
            return True
        return self.mean_diff is not None and float(np.abs(grey - previous[1]).mean()) > self.mean_diff


class _ClassVote:
    """Confidence-weighted exponential vote with hysteresis for one track."""

    def __init__(self, class_id: int):
        self.scores = {class_id: 0.0 for class_id in CLASS_NAMES}
        self.shown = class_id

    def update(self, class_id: int, confidence: float, decay: float, hysteresis: float) -> int:
        for key in self.scores:
            self.scores[key] = decay * self.scores[key] + (1 - decay) * (confidence if key == class_id else 0.0)
        best = max(self.scores, key=self.scores.get)
        if best != self.shown and self.scores[best] > self.scores[self.shown] + hysteresis:
            self.shown = best
        return self.shown


class HelmetTracker:
    def __init__(self, config: TrackingConfig, fps: float, cv2: Any, tracker_factory: Any = None):
        if fps <= 0:
            raise ValueError("Tracking requires a positive FPS.")
        self.config = config
        self.fps = fps
        self._factory = tracker_factory or self._framework_tracker
        self._tracker = self._factory(config.framework_args(fps))
        self._cut = SceneCutDetector(config.scene_cut_hist_corr, config.scene_cut_mean_diff, cv2)
        self._generation = 0
        self._display_ids: dict[tuple[int, int], int] = {}
        self._next_id = 1
        self._votes: dict[int, _ClassVote] = {}
        self.scene_cuts = 0

    @staticmethod
    def _framework_tracker(args: dict) -> Any:
        from ultralytics.trackers import BOTSORT, BYTETracker
        from ultralytics.utils import IterableSimpleNamespace

        tracker_class = BYTETracker if args["tracker_type"] == "bytetrack" else BOTSORT
        return tracker_class(args=IterableSimpleNamespace(**args))

    def reset(self) -> None:
        # Framework reset restarts its IDs at 1; a new generation keeps display IDs unique.
        self._tracker.reset()
        self._generation += 1

    def update(self, frame: np.ndarray, boxes: Any) -> list[TrackedObject]:
        """Associate one frame's detections (numpy Boxes-like) and return confirmed tracks."""
        if self._cut.is_cut(frame):
            self.scene_cuts += 1
            self.reset()
        rows = self._tracker.update(boxes if boxes is not None else _EmptyBoxes(), frame)
        objects = []
        for row in np.asarray(rows).reshape((-1, 8)):
            x1, y1, x2, y2, raw_id, confidence, raw_class = row[:7]
            key = (self._generation, int(raw_id))
            if key not in self._display_ids:
                self._display_ids[key] = self._next_id
                self._next_id += 1
            track_id = self._display_ids[key]
            raw_class_id = int(raw_class)
            if raw_class_id not in CLASS_NAMES:
                raise ValueError(f"Tracker returned unsupported class ID {raw_class_id}.")
            class_id = raw_class_id
            if self.config.class_smoothing is not None:
                vote = self._votes.setdefault(track_id, _ClassVote(raw_class_id))
                class_id = vote.update(raw_class_id, float(confidence), self.config.class_smoothing,
                                       self.config.class_hysteresis)
            objects.append(TrackedObject(track_id, (float(x1), float(y1), float(x2), float(y2)),
                                         class_id, raw_class_id, float(confidence)))
        return objects


class _EmptyBoxes:
    conf = np.zeros(0, dtype=np.float32)
    cls = np.zeros(0, dtype=np.float32)
    xywh = np.zeros((0, 4), dtype=np.float32)
    xyxy = np.zeros((0, 4), dtype=np.float32)

    def __len__(self) -> int:
        return 0

    def __getitem__(self, _: Any) -> "_EmptyBoxes":
        return self
