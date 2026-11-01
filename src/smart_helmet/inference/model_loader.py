"""Load only existing helmet detector weights; never download an unrelated model."""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import CLASS_NAMES

LOGGER = logging.getLogger(__name__)


def _validated_names(names: Any) -> dict[int, str]:
    if isinstance(names, Mapping):
        try:
            normalized = {int(key): str(value) for key, value in names.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("Model class names must use integer class IDs.") from error
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        normalized = {index: str(value) for index, value in enumerate(names)}
    else:
        raise ValueError("Model weights do not expose a supported class mapping.")
    if normalized != CLASS_NAMES:
        raise ValueError(
            f"Model class mapping must be {CLASS_NAMES}; found {normalized}."
        )
    return normalized


def load_model(weights: str | Path) -> Any:
    """Load a local Ultralytics detector whose labels match the dataset contract."""
    path = Path(weights).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {path}. Train the project baseline first "
            "with scripts/train_baseline.py and pass its best.pt checkpoint."
        )
    LOGGER.info("Loading helmet detector weights from %s", path)
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is required for inference. Install the project training "
            "extra with: python -m pip install -e \".[training]\""
        ) from error
    try:
        model = YOLO(str(path), task="detect")
        _validated_names(getattr(model, "names", None))
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        LOGGER.exception("Could not load compatible detector weights from %s", path)
        raise RuntimeError(f"Could not load a compatible helmet detector: {error}") from error
    LOGGER.info("Loaded detector with class mapping %s", CLASS_NAMES)
    return model
