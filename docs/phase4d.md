# Phase 4D: human review and sampling preparation

Run from the project environment:

```powershell
python scripts/review_baseline_failures.py --pending-only --open-image
python scripts/check_failure_review_status.py
python scripts/check_experiment_b_sampling.py
```

The review command reads the existing failure CSV, shows target GT and nearest
prediction details, and opens an individual GT/prediction overlay together with
the matching saved validation visualization, referencing existing images.
The existing `results/evaluation/A_baseline_val/failure_review/review_report.html`
also has individual green GT/orange nearest-prediction overlays. Browser-local
Phase 4C entries are not imported or interpreted as confirmed reviews.

Select multiple numbers separated by commas or spaces. UNKNOWN completes a
review and must be selected alone. SKIP leaves a pending case pending; revisiting
and skipping a completed case preserves its previous entry. QUIT, EOF and Ctrl-C
preserve entries already saved. Optional notes support commas and Unicode.
ANNOTATION_OK and ANNOTATION_SUSPICIOUS cannot be selected together.

Entries are stored separately in `human_review.csv`, keyed by a stable hash of
image path plus GT index. Every completed entry is atomically replaced on disk;
an existing CSV is backed up before the first save of each session. Concurrent
changes are rejected. The immutable automatic evidence CSV is never rewritten.
The status command refreshes CSV/JSON summaries and `experiment_b_readiness.md`.
Percentages explicitly use reviewed and total denominators; categories overlap.
Zero reviewed findings do not mean a cause is absent. All 50 final reviews,
including UNKNOWN, are required for design readiness. Unknowns or annotation
concerns keep endorsement pending; this phase never authorizes training.

## Runtime mechanism and inspected source

Inspected installed Ultralytics **8.4.128**:

- `ultralytics/models/yolo/detect/train.py`, `DetectionTrainer.get_dataloader`:
  builds the dataset, then delegates to `build_dataloader` for train/val.
- `ultralytics/data/build.py`, `build_dataloader`: chooses its own sampler;
  there is no custom sampler argument or supported class-weight YAML option.
  `InfiniteDataLoader` accepts DataLoader arguments including `sampler`, wraps
  the batch sampler for repeated epochs, and exposes `reset`.
- `ultralytics/data/base.py`, `BaseDataset.get_img_files`: supports image-list
  files but sorts the paths. Repeated manifests would alter dataset size,
  epoch length and optimizer-auto iteration calculations if simply appended.

Source SHA-256 hashes and package version are recorded in `sampling_plan.json`.
Chosen mechanism: prepared `TrainExposureSampler`, a DataLoader sampler protocol
implementation, with weighted draws **with replacement**, exactly 1,158 draws
per epoch. Its private Python RNG uses seed `42 + epoch`. The Python version is
recorded; use that version for exact replay. `set_epoch` allows explicit resume.
The JSON includes the ordered epoch-zero paths and inventory label hashes;
it is an audit artifact, not an Ultralytics training manifest.

Positive images have relative weight 1.573; negative images have weight 1.
This is a relative per-image exposure ratio, not an absolute 57.3% increase in
positive samples after normalization. Expected and realized object exposure
are reported separately. Mixed images increase both class exposures. Exposure
estimates count primary draws; baseline mosaic auxiliary image selection is
unchanged and not included in these estimates. No image files are copied.

## Future integration contract (not executed in this phase)

Implement a custom DetectionTrainer train loader using InfiniteDataLoader with
this sampler and `shuffle=False`. Validate the exact dataset.im_files identity
and order against the TRAIN inventory before assigning sampler indices; do not
assume framework ordering matches inventory ordering. Fail on removed, added or
reordered files unless rows are explicitly reordered and verified. Delegate
validation unchanged to the baseline loader. Restrict to the baseline single
device, workers=0, batch=4, rect=False and drop_last=False. Preserve collate,
worker seeding, loader generator, pinning, augmentation, cache-write redirection
and image-repair protection from the baseline. Check loader reset at close_mosaic
and resume/set_epoch behavior before authorizing a run. Do not replace the
sampler of a live persistent iterator.

The config is a **non-executable plan**, containing all baseline resolved args
and recorded optimizer groups. Constants: YOLO11n, 416, batch 4, workers 0, seed
42, 100 epochs, patience 20. Preserve optimizer=auto and verify its resolved AdamW
groups exactly match baseline; preserving epoch length avoids an unintended
change to auto optimizer selection. The only experimental change is TRAIN
sampling exposure. No custom trainer or training entry point is added now.

Sampling preparation can pass while human review and trainer integration remain
pending. TEST paths are never opened for model inference or selection. The
separate existing fingerprint/foundation checks may hash/count locked files only
to verify immutability. No TEST-derived success criteria are used.
