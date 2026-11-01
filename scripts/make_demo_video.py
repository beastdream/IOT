"""Build a demo video from curated dataset images; TEST stays locked."""

import argparse
import logging
from pathlib import Path

from smart_helmet.inference.demo_video import ALLOWED_SPLITS, build_demo_video, split_images


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a local demo video from curated-v1 images.")
    parser.add_argument("--split", choices=ALLOWED_SPLITS, default="val", help="Source split (TEST is locked).")
    parser.add_argument("--output", type=Path, help="Output .mp4; defaults to results/predictions/video_sources.")
    parser.add_argument("--fps", type=float, default=15.0, help="Output frame rate.")
    parser.add_argument("--seconds-per-image", type=float, default=1.0, help="Display time per image.")
    parser.add_argument("--size", type=int, nargs=2, default=(416, 416), metavar=("W", "H"), help="Frame size.")
    parser.add_argument("--max-zoom", type=float, default=1.1, help="Centre zoom reached at the end of each image.")
    parser.add_argument("--limit", type=int, help="Optional maximum number of images.")
    parser.add_argument("--shuffle-seed", type=int, help="Shuffle images deterministically with this seed.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an explicit output file.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        result = build_demo_video(
            split_images(args.split),
            split=args.split,
            output=args.output,
            fps=args.fps,
            seconds_per_image=args.seconds_per_image,
            size=tuple(args.size),
            max_zoom=args.max_zoom,
            limit=args.limit,
            shuffle_seed=args.shuffle_seed,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ImportError, OSError, RuntimeError, ValueError) as error:
        logging.error("Demo video creation failed: %s", error)
        return 1

    print(f"Source split: {result.split} ({result.images} images)")
    print(f"Frames: {result.frames} at {result.fps:.2f} FPS ({result.frames / result.fps:.1f} s)")
    print(f"Size: {result.size[0]}x{result.size[1]}")
    print(f"Output: {result.output}")
    print(f"Ground-truth manifest: {result.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
