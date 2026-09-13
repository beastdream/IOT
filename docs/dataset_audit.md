# Dataset audit

Run from the project root using the existing editable environment:

```powershell
.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e ".[test]"
.venv\Scripts\python.exe -m compileall src scripts tests
.venv\Scripts\python.exe scripts/audit_dataset.py
.venv\Scripts\python.exe -m pytest -q
```

`requirements.txt` delegates dependencies to pyproject.toml. Pillow and numpy are
the only additions for this phase. No model or training framework is imported.
The config loader and paths are shared with foundation. No Ultralytics fallback
is used. `run_audit(root, config_path, output)` also supports mini datasets with
the same layout; there is no hard-coded expected image count in the audit.

## Evidence

All output goes under `results/dataset_audit/`. The manifest covers supported
images, including unreadable images with blank unavailable metadata. Unsupported
files are logged in image_issues.csv. No bad file is repaired. CSVs always have
headers, including when there are no issues. Paths are relative to project root.

- dataset_manifest.csv: one row per image, hashes, source group, dimensions and counts.
- image_issues.csv / annotation_issues.csv: integrity findings with severity.
- bounding_boxes.csv: every numerically parseable, in-range bbox and its geometry.
- bbox_statistics.csv: class-specific width, height, area, aspect statistics and size counts.
- suspicious_bboxes.csv: one row per bbox/reason (not a semantic error verdict).
- class_distribution.csv: split and overall objects, images, ratios; pay special
  attention to Without Helmet object count, fraction and coverage.
- exact_duplicates.csv: SHA-256 groups, CRITICAL for cross-split groups.
- source_groups.csv / source_group_leakage.csv: basename-before-.rf groups, not proof of duplicates.
- near_duplicates.csv / near_duplicate_leakage.csv: image-similarity candidate pairs.
- image_resolutions.csv / negative_images.csv: dimensions and object-count categories.
- manual_review.csv: one row per reason/evidence; repeated images are intentional.
- dataset_audit_report.json: summary, thresholds, definitions, status and limitations.
- dataset_fingerprint.json: hashes before/after for every original split file and
  the three export documents. Equality and identical file sets are required.
- visual_review/: nine review categories of annotated copies, no confidence scores.

## Thresholds and limitations

YOLO coordinates must be finite; IDs must be integer 0/1 with unchanged names.
Frame-edge tolerance is 1e-6. Near-zero side means normalized width or height
below 1e-4. Tiny area <0.001; small [0.001,0.01); medium [0.01,0.1); large >=0.1.
Aspect ratio outside [0.2,5] needs review. Tiny/odd boxes are not deleted.
Non-416x416 resolution is a review signal, not an integrity error.

pHash uses a 32x32 grayscale image, DCT low 8x8 coefficients, median threshold
over the 63 AC coefficients, and fixed-zero DC bit. Distance 0 is an exact
perceptual-hash match, **not** proof of identical pixels. Distances 0–4 are
VERY_SIMILAR; 5–8 POSSIBLY_SIMILAR. These are initial review thresholds, not
calibrated duplicate probabilities. All image pairs are compared (about 946,000
pairs for 1376 images); only candidate pairs are stored. Filename similarity
does not affect pHash. Compression, crops and rotations can cause false negatives;
simple compositions can cause false positives. Exact-byte pairs may also appear
in the perceptual table; those counts should not be added together.

Near-duplicate groups are connected components, so not every member is necessarily
similar to every other member. Exact/source leakage counts are groups; perceptual
leakage counts are pairs. Cross-split perceptual candidates have HIGH severity,
within-split candidates REVIEW. Neither is automatically removed or relabeled.

Annotation counts include duplicate rows and numeric boxes with flagged edge
violations. Malformed/out-of-range rows are excluded from geometry statistics and
make status FAIL. Empty readable valid labels count as zero-object images; missing
or malformed labels do not. Empty labels alone cannot prove the scene has no heads.

Visual sampling uses seed 42, up to 24 images per category where available. Source
samples keep 2–3 group members together; near samples show up to 12 pairs side by
side with cross-split pairs first. Green is 0: With Helmet; red is 1: Without Helmet.
The manual manifest links all generated copies for each selected image. Many
candidates are intentionally not rendered. A rerun overwrites selected output
filenames but does not delete old output; the current CSV links are authoritative.

## Decision gate

Conservative policy: any config/structure or image/label numeric integrity ERROR
causes FAIL; a failed fingerprint or incomplete visualization also causes FAIL.
Other review evidence gives REVIEW_REQUIRED. PASS requires no remaining evidence
requiring review. CLI exits 1 for FAIL, 0 for a completed PASS or REVIEW_REQUIRED.
Review status is not a claim that annotations are semantically wrong or that
cleaning is authorized. Inspect candidates, record decisions separately, then
plan cleaning. This command never starts cleaning or training.
