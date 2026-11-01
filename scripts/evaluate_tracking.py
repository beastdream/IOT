"""Compare per-frame detection and tracking configs on a manifest-backed demo video."""

import argparse
import csv
import json
import logging
from pathlib import Path

from smart_helmet.evaluation.tracking_eval import compute_metrics, load_ground_truth, run_sequence
from smart_helmet.inference.device import resolve_device
from smart_helmet.inference.model_loader import load_model
from smart_helmet.inference.tracking import TrackingConfig
from smart_helmet.inference.video_inference import _opencv
from smart_helmet.paths import EVALUATION_DIR, EXPERIMENTS_DIR

DEFAULT_WEIGHTS = EXPERIMENTS_DIR / "A_baseline" / "weights" / "best.pt"
COLUMNS = ("config", "recall", "precision", "false_positives", "id_switches", "id_transfers",
           "scene_id_transfers", "ids_spanning_multiple_objects", "mean_ids_per_object", "mota",
           "class_flips", "class_accuracy", "without_helmet_shown_recall", "scene_cuts", "processing_fps")


def _fmt(value) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate ID and label stability against demo-video ground truth.")
    parser.add_argument("--manifest", required=True, type=Path, help="Demo video manifest (.json) from make_demo_video.py.")
    parser.add_argument("--config", type=Path, action="append", default=[], help="Tracking YAML; repeatable.")
    parser.add_argument("--no-per-frame", action="store_true", help="Skip the untracked per-frame baseline.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.25, help="Per-frame baseline confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, help="Defaults to results/evaluation/tracking/<video stem>.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        manifest, gt_frames = load_ground_truth(args.manifest)
        selection = resolve_device(args.device)
        model = load_model(args.weights)
        cv2 = _opencv()
        runs = ([] if args.no_per_frame else [("per_frame", None)]) + [
            (path.stem, TrackingConfig.from_yaml(path)) for path in args.config]
        if not runs:
            raise ValueError("Nothing to evaluate: pass --config or omit --no-per-frame.")
        rows = []
        for name, config in runs:
            logging.info("Evaluating %s", name)
            outputs, info = run_sequence(manifest["video"], model, config=config, confidence=args.conf,
                                         image_size=args.imgsz, device_argument=selection.device_argument, cv2=cv2)
            metrics = compute_metrics(gt_frames, outputs)
            rows.append(dict(config=name, **metrics, **info,
                             tracking_config=config.as_dict() if config is not None else None))
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError, KeyError) as error:
        logging.error("Tracking evaluation failed: %s", error)
        return 1

    output_dir = args.output_dir or EVALUATION_DIR / "tracking" / Path(manifest["video"]).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tracking_metrics.json").write_text(json.dumps(dict(
        manifest=str(args.manifest.resolve()), weights=str(args.weights.resolve()), imgsz=args.imgsz,
        per_frame_conf=args.conf, device=selection.device, test_status="LOCKED / NOT USED", runs=rows,
    ), indent=2), encoding="utf-8")
    with (output_dir / "tracking_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Video: {manifest['video']} | split={manifest['split']} | frames={manifest['total_frames']} | "
          f"scenes={len(manifest['segments'])} | GT objects={rows[0]['gt_objects']}")
    print(" | ".join(COLUMNS))
    for row in rows:
        print(" | ".join(_fmt(row.get(column)) for column in COLUMNS))
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
