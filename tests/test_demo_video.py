from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from smart_helmet.inference.demo_video import build_demo_video, split_images


def _write_images(directory: Path, count: int) -> list[Path]:
    paths = []
    for index in range(count):
        path = directory / f"img{index}.jpg"
        image = np.full((40, 60, 3), 40 * index, dtype=np.uint8)
        assert cv2.imwrite(str(path), image)
        paths.append(path)
    return paths


def test_test_split_is_locked():
    with pytest.raises(ValueError, match="TEST is locked"):
        split_images("test")


def test_demo_video_frames_size_and_limit(tmp_path):
    images = _write_images(tmp_path, 3)
    result = build_demo_video(
        images, split="val", output=tmp_path / "demo.mp4", fps=10, seconds_per_image=0.5,
        size=(64, 48), limit=2,
    )
    assert result.images == 2
    assert result.frames == 10
    capture = cv2.VideoCapture(str(result.output))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 48
    finally:
        capture.release()


def test_demo_video_refuses_overwrite_and_cleans_failed_output(tmp_path):
    images = _write_images(tmp_path, 1)
    target = tmp_path / "demo.mp4"
    target.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        build_demo_video(images, split="val", output=target)
    assert target.read_bytes() == b"existing"

    unreadable = tmp_path / "broken.jpg"
    unreadable.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="could be read"):
        build_demo_video([unreadable], split="val", output=tmp_path / "failed.mp4")
    assert not (tmp_path / "failed.mp4").exists()
