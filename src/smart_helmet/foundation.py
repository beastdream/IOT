"""Read-only foundation checks, not an annotation audit."""

from pathlib import Path

import yaml

from smart_helmet.paths import DATA_CONFIG, PROJECT_ROOT

EXPECTED_CLASSES = {0: "With Helmet", 1: "Without Helmet"}
EXPECTED_COUNTS = {"train": 1185, "val": 127, "test": 64}
SPLIT_FOLDERS = {"train": "train", "val": "valid", "test": "test"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_data_config(config_path: Path = DATA_CONFIG, root: Path = PROJECT_ROOT) -> dict:
    """Resolve local YAML relative to the checkout root, independently of cwd.

    The returned mapping has absolute paths suitable for downstream code.
    Native Ultralytics YAML use requires cwd=project root; no fallback is used.
    """
    with config_path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("Data config must be a mapping")
    if type(data.get("nc")) is not int or data["nc"] != 2:
        raise ValueError("nc must be the integer 2")
    names = data.get("names")
    if not isinstance(names, dict) or any(type(k) is not int for k in names):
        raise ValueError("names must map integer class IDs to class names")
    if names != EXPECTED_CLASSES:
        raise ValueError(f"Class mapping must remain {EXPECTED_CLASSES}")
    base = data.get("path")
    if not isinstance(base, str) or not base:
        raise ValueError("path must be a nonempty string")
    dataset_root = (root / base).resolve()
    if dataset_root != root.resolve():
        raise ValueError("Dataset root must remain the project root")
    data["path"] = str(dataset_root)
    for split, folder in SPLIT_FOLDERS.items():
        value = data.get(split)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Missing image path: {split}")
        resolved = (dataset_root / value).resolve()
        if resolved != (root / folder / "images").resolve():
            raise ValueError(f"Unexpected {split} path: {resolved}")
        data[split] = str(resolved)
    return data


def count_images(directory: Path) -> int:
    return sum(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in directory.iterdir())


def verify_foundation(root: Path = PROJECT_ROOT) -> bool:
    """Print checks and return success; never repair or modify the dataset."""
    print("=" * 40)
    print("PROJECT FOUNDATION VERIFICATION")
    print("=" * 40)
    checks = []

    def check(label: str, passed: bool) -> None:
        checks.append(passed)
        print(f"{label}: {'PASS' if passed else 'FAIL'}")

    check("Project root", root.is_dir())
    config = root / "configs" / "data.local.yaml"
    check("Local data config", config.is_file())
    check("Original export files present", all((root / n).is_file() for n in
          ("data.yaml", "README.dataset.txt", "README.roboflow.txt")))
    try:
        data = load_data_config(config, root)
        check("YAML loading and nc/classes", True)
        print("Class mapping:")
        for key, value in data["names"].items():
            print(f"{key} = {value}")
    except (OSError, ValueError, yaml.YAMLError) as error:
        check("YAML loading and nc/classes", False)
        print(f"  {error}")

    counts = {}
    for split, folder in SPLIT_FOLDERS.items():
        images, labels = root / folder / "images", root / folder / "labels"
        check(f"{folder.capitalize()} paths", images.is_dir() and labels.is_dir())
        try:
            counts[split] = count_images(images)
            check(f"{folder.capitalize()} image count ({counts[split]})", counts[split] == EXPECTED_COUNTS[split])
        except OSError as error:
            check(f"{folder.capitalize()} image count", False)
            print(f"  {error}")
    print("Images:")
    for split, folder in SPLIT_FOLDERS.items():
        print(f"{folder.capitalize()}: {counts.get(split, 'unavailable')}")
    print(f"Total: {sum(counts.values()) if len(counts) == 3 else 'unavailable'}")
    print("RESULT:")
    success = all(checks)
    print("PASS" if success else "FAIL")
    print("=" * 40)
    return success
