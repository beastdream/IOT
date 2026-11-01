"""Run helmet-status detection on one image."""

import argparse
import logging
from pathlib import Path

from smart_helmet.inference.device import resolve_device
from smart_helmet.inference.image_inference import predict_image
from smart_helmet.inference.model_loader import load_model
from smart_helmet.paths import EXPERIMENTS_DIR

DEFAULT_WEIGHTS = EXPERIMENTS_DIR / "A_baseline" / "weights" / "best.pt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect helmet status in a local image.")
    parser.add_argument("--source", required=True, type=Path, help="Input image path.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Trained local best.pt.")
    parser.add_argument("--output", type=Path, help="Output image path; defaults to results/predictions/image.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold in [0, 1].")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an explicit output file.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        selection = resolve_device(args.device)
        logging.info("%s", selection.message)
        model = load_model(args.weights)
        result = predict_image(
            args.source,
            model,
            confidence=args.conf,
            image_size=args.imgsz,
            device=selection,
            output=args.output,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ImportError, OSError, RuntimeError, ValueError) as error:
        logging.error("Image inference failed: %s", error)
        return 1

    print(f"Detections: {result.detections}")
    if result.detections == 0:
        print("No helmet-status objects were detected.")
    print(f"Inference device: {result.device}" + (" (verified)" if result.device_verified else " (requested)"))
    print(f"Elapsed: {result.duration_seconds:.3f}s")
    print(f"Output: {result.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
