from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from smart_helmet.inference import CLASS_NAMES
from smart_helmet.inference.device import (
    DeviceUnavailableError,
    resolve_device,
    verify_inference_device,
)
from smart_helmet.inference.image_inference import predict_image
from smart_helmet.inference.model_loader import load_model
from smart_helmet.inference.video_inference import process_video


@pytest.fixture(autouse=True)
def isolated_predictions_dir(tmp_path, monkeypatch):
    # Default output paths must never write fake artifacts into results/predictions.
    directory = tmp_path / "predictions"
    monkeypatch.setattr("smart_helmet.inference.image_inference.PREDICTIONS_DIR", directory)
    monkeypatch.setattr("smart_helmet.inference.video_inference.PREDICTIONS_DIR", directory)
    return directory


class FakeCuda:
    def __init__(self, names=()):
        self.names = names

    def is_available(self):
        return bool(self.names)

    def device_count(self):
        return len(self.names)

    def get_device_name(self, index):
        return self.names[index]


def fake_torch(names=()):
    return SimpleNamespace(
        __version__="test-torch",
        version=SimpleNamespace(cuda="test-cuda" if names else None),
        cuda=FakeCuda(names),
    )


def test_inference_package_import_and_class_mapping():
    assert CLASS_NAMES == {0: "With Helmet", 1: "Without Helmet"}


def test_device_cpu_and_auto_fallback():
    torch = fake_torch()
    assert resolve_device("cpu", torch).device == "cpu"
    selected = resolve_device("auto", torch)
    assert selected.device == selected.device_argument == "cpu"
    assert selected.backend == "cpu"
    assert not selected.cuda_available


def test_device_auto_and_cuda_select_real_backend_when_available():
    torch = fake_torch(("NVIDIA Test GPU", "NVIDIA Second GPU"))
    assert resolve_device("auto", torch).device == "cuda:0"
    selection = resolve_device("cuda:1", torch)
    assert selection.device == "cuda:1"
    assert selection.device_argument == "1"
    assert selection.gpus[1].name == "NVIDIA Second GPU"


def test_cuda_request_fails_when_cuda_is_unavailable():
    with pytest.raises(DeviceUnavailableError, match="CUDA device 0 is unavailable"):
        resolve_device("cuda", fake_torch())


def test_cuda_runtime_device_is_verified_and_mismatch_is_rejected():
    selection = resolve_device("cuda", fake_torch(("NVIDIA Test GPU",)))
    model = SimpleNamespace(predictor=SimpleNamespace(model=SimpleNamespace(device="cuda:0")))
    assert verify_inference_device(model, selection) == ("cuda:0", True)
    model.predictor.model.device = "cpu"
    with pytest.raises(DeviceUnavailableError, match="not the requested"):
        verify_inference_device(model, selection)


def test_missing_weights_fail_without_model_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train the project baseline"):
        load_model(tmp_path / "missing.pt")


def test_model_loader_requires_the_exact_class_mapping(tmp_path, monkeypatch):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"checkpoint")

    class FakeYOLO:
        def __init__(self, path, task):
            self.names = {0: "Without Helmet", 1: "With Helmet"}

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    with pytest.raises(RuntimeError, match="class mapping must be"):
        load_model(weights)


class FakeBoxes:
    def __init__(self, classes=(1,), confidences=(0.91,)):
        self.xyxy = np.array([[2, 3, 18, 19] for _ in classes], dtype=float)
        self.cls = np.array(classes, dtype=float)
        self.conf = np.array(confidences, dtype=float)


class FakeModel:
    def __init__(self, classes=(1,)):
        self.classes = classes
        self.calls = []
        self.predictor = SimpleNamespace(model=SimpleNamespace(device="cpu"))

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(boxes=FakeBoxes(self.classes))]


class FakeCV2:
    IMREAD_COLOR = 1
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 1

    def __init__(self, frames=()):
        self.frames = list(frames)
        self.read_image = np.zeros((24, 32, 3), dtype=np.uint8)
        self.image_readable = True
        self.capture = None
        self.writer = None
        self.labels = []

    def imread(self, path, mode):
        return self.read_image if self.image_readable else None

    def imwrite(self, path, frame):
        Path(path).write_bytes(b"image")
        return True

    def rectangle(self, *args):
        return None

    def putText(self, image, text, *args):
        self.labels.append(text)

    def getTextSize(self, text, *args):
        return (6 * len(text), 10), 3

    def VideoCapture(self, path):
        self.capture = FakeCapture(self.frames)
        return self.capture

    def VideoWriter_fourcc(self, *codec):
        return 1

    def VideoWriter(self, path, codec, fps, size):
        self.writer = FakeWriter(path)
        return self.writer


class FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return 25.0 if prop == FakeCV2.CAP_PROP_FPS else len(self.frames)

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


class FakeWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.frames = []
        self.released = False

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True
        self.path.write_bytes(b"video")


def test_image_input_validation_and_threshold_and_class_mapping(tmp_path, monkeypatch):
    source = tmp_path / "helmet.PNG"
    source.write_bytes(b"input")
    fake_cv2 = FakeCV2()
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    model = FakeModel(classes=(1,))

    result = predict_image(
        source, model, confidence=0.43, image_size=320, device="cpu",
        output=tmp_path / "annotated.png",
    )
    assert result.detections == 1
    assert model.calls[0]["conf"] == 0.43
    assert model.calls[0]["imgsz"] == 320
    assert model.calls[0]["device"] == "cpu"
    assert fake_cv2.labels == ["Without Helmet 0.91"]
    assert result.output.is_file()
    assert result.device_verified

    with pytest.raises(FileNotFoundError):
        predict_image(tmp_path / "missing.jpg", model, device="cpu")
    fake_cv2.image_readable = False
    with pytest.raises(ValueError, match="could not decode"):
        predict_image(source, model, device="cpu", output=tmp_path / "bad.png")


def test_image_no_detections_is_a_valid_saved_result(tmp_path, monkeypatch):
    fake_cv2 = FakeCV2()
    fake_cv2.read_image = np.zeros((24, 32, 3), dtype=np.uint8)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    model = FakeModel(classes=())
    model.predict = lambda **kwargs: [SimpleNamespace(boxes=None)]
    source = tmp_path / "empty.jpg"
    source.write_bytes(b"input")
    result = predict_image(source, model, device="cpu")
    assert result.detections == 0
    assert result.output.is_file()


def test_video_input_validation_and_unopenable_video(tmp_path, monkeypatch):
    fake_cv2 = FakeCV2()
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    model = FakeModel()
    with pytest.raises(FileNotFoundError):
        process_video(tmp_path / "missing.mp4", model, device="cpu")

    source = tmp_path / "closed.mp4"
    source.write_bytes(b"input")

    class ClosedCapture(FakeCapture):
        def isOpened(self):
            return False

    fake_cv2.VideoCapture = lambda path: ClosedCapture([])
    with pytest.raises(ValueError, match="could not open"):
        process_video(source, model, device="cpu")
    assert fake_cv2.capture is None


def test_video_processes_frames_saves_output_and_releases_resources(tmp_path, monkeypatch):
    frames = [np.zeros((20, 30, 3), dtype=np.uint8) for _ in range(3)]
    fake_cv2 = FakeCV2(frames)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    model = FakeModel(classes=(0,))
    progress = []
    source = tmp_path / "sample.mov"
    source.write_bytes(b"input")

    result = process_video(
        source,
        model,
        confidence=0.62,
        image_size=416,
        device="cpu",
        max_frames=2,
        progress_callback=lambda current, total, fps: progress.append((current, total, fps)),
    )
    assert result.frames_processed == 2
    assert result.total_frames == 3
    assert result.detections == 2
    assert result.source_fps == 25.0
    assert result.output.suffix == ".mp4"
    assert result.output.read_bytes() == b"video"
    assert len(model.calls) == 2
    assert all(call["conf"] == 0.62 and call["imgsz"] == 416 for call in model.calls)
    assert [row[0] for row in progress] == [1, 2]
    assert fake_cv2.capture.released
    assert fake_cv2.writer.released
    assert not result.audio_preserved


def test_video_resources_are_released_when_inference_fails(tmp_path, monkeypatch):
    fake_cv2 = FakeCV2([np.zeros((20, 30, 3), dtype=np.uint8)])
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    source = tmp_path / "failure.avi"
    source.write_bytes(b"input")

    class BrokenModel(FakeModel):
        def predict(self, **kwargs):
            raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        process_video(source, BrokenModel(), device="cpu")
    assert fake_cv2.capture.released
    assert fake_cv2.writer.released
