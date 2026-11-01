import json
from types import SimpleNamespace

import numpy as np
import pytest

from smart_helmet.evaluation.tracking_eval import GroundTruth, Output, compute_metrics, load_ground_truth
from smart_helmet.inference.tracking import HelmetTracker, SceneCutDetector, TrackingConfig

cv2 = pytest.importorskip("cv2")
results = pytest.importorskip("ultralytics.engine.results")


def boxes(*rows, shape=(100, 100)):
    """rows: (x1, y1, x2, y2, conf, cls) -> numpy-backed Ultralytics Boxes."""
    data = np.asarray(rows, dtype=np.float32).reshape((-1, 6))
    return results.Boxes(data, shape).numpy()


class ScriptedTracker:
    """Framework stand-in returning scripted rows; raw IDs restart after reset like the real one."""

    def __init__(self, script):
        self.script = list(script)
        self.resets = 0

    def update(self, detections, image):
        return np.asarray(self.script.pop(0), dtype=float).reshape((-1, 8))

    def reset(self):
        self.resets += 1


def test_config_validation_yaml_and_fps_conversion(tmp_path):
    with pytest.raises(ValueError):
        TrackingConfig(tracker_type="sort")
    with pytest.raises(ValueError):
        TrackingConfig(track_low_thresh=0.5, track_high_thresh=0.3)
    path = tmp_path / "tracker.yaml"
    path.write_text("tracker_type: botsort\ntrack_buffer_seconds: 1.5\nunknown: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        TrackingConfig.from_yaml(path)
    path.write_text("tracker_type: botsort\ntrack_buffer_seconds: 1.5\n", encoding="utf-8")
    config = TrackingConfig.from_yaml(path)
    assert config.framework_args(30)["track_buffer"] == 45
    assert config.framework_args(10)["track_buffer"] == 15
    tuned = config.with_confidence(0.05)
    assert tuned.track_high_thresh == tuned.new_track_thresh == tuned.track_low_thresh == 0.05


def test_scene_cut_detector_flags_hard_cuts_only():
    detector = SceneCutDetector(0.6, 25, cv2)
    rng = np.random.default_rng(0)
    first = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert not detector.is_cut(first)
    assert not detector.is_cut(np.clip(first.astype(int) + 3, 0, 255).astype(np.uint8))
    assert detector.is_cut(255 - first)
    assert not SceneCutDetector(None, None, cv2).is_cut(first)


def test_display_ids_are_never_reused_after_scene_cut():
    frame_a = np.zeros((64, 64, 3), dtype=np.uint8)
    frame_b = np.full((64, 64, 3), 255, dtype=np.uint8)
    scripted = ScriptedTracker([
        [[0, 0, 10, 10, 1, 0.9, 0, 0]],
        [[0, 0, 10, 10, 1, 0.9, 0, 0]],
        [[0, 0, 10, 10, 1, 0.9, 0, 0]],  # framework restarts at raw ID 1 after reset
    ])
    config = TrackingConfig(scene_cut_hist_corr=0.6, scene_cut_mean_diff=25)
    tracker = HelmetTracker(config, 15, cv2, tracker_factory=lambda args: scripted)
    ids = [tracker.update(frame, None)[0].track_id for frame in (frame_a, frame_a, frame_b)]
    assert ids == [1, 1, 2]
    assert tracker.scene_cuts == 1 and scripted.resets == 1


def test_class_smoothing_ignores_single_frame_flip_but_follows_sustained_change():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    classes = [0, 0, 0, 1, 0, 1, 1, 1, 1]
    scripted = ScriptedTracker([[[0, 0, 10, 10, 7, 0.8, c, 0]] for c in classes])
    config = TrackingConfig(class_smoothing=0.7, class_hysteresis=0.05)
    tracker = HelmetTracker(config, 15, cv2, tracker_factory=lambda args: scripted)
    shown = [tracker.update(frame, None)[0].class_id for _ in classes]
    assert shown[:5] == [0, 0, 0, 0, 0]
    assert shown[-1] == 1
    raw = HelmetTracker(TrackingConfig(), 15, cv2,
                        tracker_factory=lambda args: ScriptedTracker([[[0, 0, 10, 10, 7, 0.8, 1, 0]]]))
    assert raw.update(frame, None)[0].class_id == 1


def test_real_bytetrack_keeps_id_for_moving_object_and_handles_empty_frames():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    tracker = HelmetTracker(TrackingConfig(), 15, cv2)
    first = tracker.update(frame, boxes((10, 10, 30, 30, 0.9, 0), (60, 60, 80, 80, 0.9, 1)))
    second = tracker.update(frame, boxes((12, 11, 32, 31, 0.9, 0), (61, 62, 81, 82, 0.9, 1)))
    assert {o.track_id for o in first} == {o.track_id for o in second} == {1, 2}
    by_class = {o.raw_class_id: o.track_id for o in second}
    assert by_class == {o.raw_class_id: o.track_id for o in first}
    assert tracker.update(frame, None) == []


def test_metrics_count_switches_transfers_and_flips():
    box = (0.0, 0.0, 10.0, 10.0)
    gt = [[GroundTruth((0, 0), 1, box)]] * 3 + [[GroundTruth((1, 0), 0, box)]] * 2
    outputs = [
        [Output(1, 1, box)], [Output(1, 0, box)], [Output(2, 1, box)],  # flip x2, switch 1->2
        [Output(2, 0, box)], [Output(2, 0, box)],                       # ID 2 jumps to a new scene
    ]
    metrics = compute_metrics(gt, outputs)
    assert metrics["id_switches"] == 1
    assert metrics["id_transfers"] == metrics["scene_id_transfers"] == 1
    assert metrics["ids_spanning_multiple_objects"] == 1
    assert metrics["class_flips"] == 2
    assert metrics["recall"] == metrics["precision"] == 1.0
    untracked = compute_metrics(gt, [[Output(None, g[0].class_id, box)] for g in gt])
    assert untracked["id_switches"] is None and untracked["class_accuracy"] == 1.0


def test_ground_truth_refuses_test_split(tmp_path):
    manifest = tmp_path / "demo.json"
    manifest.write_text(json.dumps(dict(split="test", size=[10, 10], total_frames=0, segments=[])),
                        encoding="utf-8")
    with pytest.raises(ValueError, match="TEST is locked"):
        load_ground_truth(manifest)


def test_process_video_with_tracking_draws_ids(tmp_path, monkeypatch):
    import sys
    from smart_helmet.inference import video_inference

    from test_inference import FakeCV2  # reuse the existing OpenCV fake

    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
    fake_cv2 = FakeCV2(frames)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(video_inference, "PREDICTIONS_DIR", tmp_path)

    class Model:
        predictor = SimpleNamespace(model=SimpleNamespace(device="cpu"))
        calls = []

        def predict(self, **kwargs):
            self.calls.append(kwargs)
            return [SimpleNamespace(boxes=results.Boxes(
                np.asarray([[10, 10, 30, 30, 0.9, 1]], dtype=np.float32), (100, 100)))]

    model = Model()
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"input")
    result = video_inference.process_video(source, model, device="cpu",
                                           tracking=TrackingConfig(agnostic_nms=True))
    assert result.unique_tracks == 1 and result.scene_cuts == 0
    assert all(call["conf"] == 0.1 and call["agnostic_nms"] for call in model.calls)
    box_labels = [label for label in fake_cv2.labels if label.startswith("#")]
    assert len(box_labels) == 3 and all(label.startswith("#1 0.9") for label in box_labels)
    assert {"With Helmet", "Without Helmet"} <= set(fake_cv2.labels)  # legend
