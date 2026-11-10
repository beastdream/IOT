"""Package the curated dataset into a zip file ready for Google Colab training."""

import argparse
import logging
from pathlib import Path
import shutil
import zipfile
import yaml

from smart_helmet.paths import CURATED_V1_DIR, PROJECT_ROOT


def package_dataset(output_zip: Path | None = None) -> Path:
    curated_dir = CURATED_V1_DIR
    if not curated_dir.is_dir():
        raise FileNotFoundError(f"Curated dataset directory not found at: {curated_dir}")

    if output_zip is None:
        output_zip = PROJECT_ROOT / "curated_v1_colab.zip"

    output_zip = output_zip.resolve()
    temp_yaml = PROJECT_ROOT / "data_colab.yaml"

    colab_yaml_content = {
        "path": "/content/dataset",
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 2,
        "names": {
            0: "With Helmet",
            1: "Without Helmet",
        },
    }

    with open(temp_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(colab_yaml_content, f, sort_keys=False)

    print(f"Creating zip file: {output_zip}...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(temp_yaml, arcname="data.yaml")
        for item in curated_dir.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(curated_dir)
                zipf.write(item, arcname=str(rel_path).replace("\\", "/"))

    if temp_yaml.exists():
        temp_yaml.unlink()

    file_size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"Successfully packaged dataset! File size: {file_size_mb:.2f} MB")
    print(f"Target location: {output_zip}")
    return output_zip


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Package curated_v1 dataset for Google Colab.")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Target zip path (defaults to curated_v1_colab.zip at project root).",
    )
    args = parser.parse_args()
    try:
        package_dataset(args.output)
        return 0
    except Exception as e:
        logging.error("Failed to package dataset: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
