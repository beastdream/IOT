"""Run sequential helmet-status detection on a local video."""

import argparse
import logging
from pathlib import Path

from smart_helmet.inference.device import resolve_device
from smart_helmet.inference.model_loader import load_model
from smart_helmet.inference.tracking import TrackingConfig
from smart_helmet.inference.video_inference import process_video
from smart_helmet.paths import EXPERIMENTS_DIR, PROJECT_ROOT

DEFAULT_WEIGHTS = EXPERIMENTS_DIR / "A_baseline" / "weights" / "best.pt"
DEFAULT_TRACKER = PROJECT_ROOT / "configs" / "tracking" / "helmet_tracking.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect helmet status frame-by-frame in a video.")
    parser.add_argument("--source", required=True, type=Path, help="Input video path.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Trained local best.pt.")
    parser.add_argument("--output", type=Path, help="Output .mp4 or .avi; defaults to results/predictions/video.")
    parser.add_argument(
        "--conf", type=float,
        help="Confidence threshold in [0, 1]; default 0.25 without tracking, tracker config with tracking.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>.")
    parser.add_argument("--max-frames", type=int, help="Optional positive limit for processed frames.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an explicit output file.")
    parser.add_argument("--tracker", type=Path, default=DEFAULT_TRACKER, help="Tracking config YAML.")
    parser.add_argument("--no-track", action="store_true", help="Per-frame detection without persistent IDs.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        selection = resolve_device(args.device)
        logging.info("%s", selection.message)
        model = load_model(args.weights)
        tracking = None
        if not args.no_track:
            tracking = TrackingConfig.from_yaml(args.tracker)
            if args.conf is not None:
                tracking = tracking.with_confidence(args.conf)
            logging.info("Tracking with %s (%s)", args.tracker, tracking.tracker_type)

        def report_progress(current: int, total: int | None, fps: float) -> None:
            if current % 30 == 0 or current == total or (
                args.max_frames is not None and current == args.max_frames
            ):
                total_text = str(total) if total is not None else "unknown"
                print(
                    f"Frames: {current}/{total_text} | Processing FPS: {fps:.2f} | "
                    f"Device: {selection.device}",
                    flush=True,
                )

        result = process_video(
            args.source,
            model,
            confidence=args.conf if args.conf is not None else 0.25,
            image_size=args.imgsz,
            device=selection,
            output=args.output,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
            progress_callback=report_progress,
            tracking=tracking,
        )
    except (FileNotFoundError, FileExistsError, ImportError, OSError, RuntimeError, ValueError) as error:
        logging.error("Video inference failed: %s", error)
        return 1

    print(f"Frames processed: {result.frames_processed}/{result.total_frames or 'unknown'}")
    print(f"Detections: {result.detections}")
    if result.unique_tracks is not None:
        print(f"Tracked objects (unique IDs): {result.unique_tracks} | scene cuts reset: {result.scene_cuts}")
    if result.detections == 0:
        print("No helmet-status objects were detected in the processed frames.")
    print(f"Processing FPS: {result.fps:.2f} (source FPS: {result.source_fps:.2f})")
    print(f"Inference device: {result.device}" + (" (verified)" if result.device_verified else " (requested)"))
    print("Audio: not preserved by OpenCV output.")
    print(f"Output: {result.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
