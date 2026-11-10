# Experiment B: class-aware TRAIN sampling

Only TRAIN image exposure changes from A_baseline. The configuration is
`configs/experiments/B_class_aware_sampling.yaml`: YOLO11n, imgsz 416, batch 4,
workers 0, seed 42, pretrained initialization, 100 epochs, patience 20.
The smoke run uses exactly two epochs; it still draws 1,158 samples per epoch
from all 1,158 TRAIN images and validates all 127 VALID images. No image copies,
label edits, split changes, new augmentation or TEST evaluation are involved.

```powershell
python -m compileall src scripts tests
python -m pytest -q
python scripts/check_failure_review_status.py
python scripts/check_experiment_b_sampling.py
python scripts/train_experiment_b.py --dry-run
python scripts/train_experiment_b.py --smoke-test
python scripts/check_experiment_b_readiness.py
```

The CLI accepts `--device`, `--batch`, `--workers`, `--epochs`, `--overwrite`,
`--dry-run`, and `--smoke-test`. Changes to the controlled constants are rejected;
`--epochs 2` is allowed with smoke, and `--epochs 100` with full training.
Auto chooses the installed CUDA:0; CPU is available for a dry run, but a CPU
smoke does not satisfy this machine's GPU readiness requirement.

## Integration and diagnostics

Installed Ultralytics 8.4.128 source was inspected: `Model.train(trainer=...)`
accepts a trainer factory. `DetectionTrainer.get_dataloader` is the extension
point. The local `ExperimentBTrainer` overrides only TRAIN loader construction;
VALID delegates to the framework loader unchanged. No installed source is edited
and sampler integration does not patch global state. The shared baseline
training lifecycle retains its scoped cache-write redirect and image-repair
guard to keep the dataset frozen.

`ClassAwareSampler` weights images with any class-1 box by 1.573 and all others by
1.0. Sampling is with replacement, with exactly 1,158 draws. It uses a private
Python RNG, seed `42 + epoch`, and maps inventory rows into the actual framework
image order. The explicit epoch callback prevents accidental epoch advancement
when `InfiniteDataLoader.reset` runs at close_mosaic. CPU tests cover both normal
epoch advancement and reset; no unit test trains a model.

The TRAIN loader preserves baseline collation, loader generator seeding,
pin-memory behavior, workers, batch size and final partial batch. Baseline
arguments are passed explicitly and checked after framework resolution.
The resolved AdamW initial parameter groups must exactly match A_baseline.
Baseline epoch length is retained, preserving optimizer-auto iteration logic.
All baseline augmentation settings, including mosaic, are unchanged.

`sampler_diagnostics.json` records sampler identity, replacement, weights,
counts, seed, index hash and class exposure. It starts with actual use=false;
only after an epoch does the trainer assert that the ordered primary image paths
observed in preprocessing and indices yielded by the attached sampler exactly
match the seeded draws. Finite completed batches and optimizer steps are also
required. Counts are diagnostic and never adjust the sampler. Mosaic auxiliary
images follow unchanged baseline behavior and are outside primary exposure
counts. Every completed epoch is recorded; top-level exposure describes epoch 0.

## Human evidence and interpretation

Readiness recomputes the human-review status from CSV. There are 49 ANNOTATION_OK
entries and one OTHER entry without notes, with no suspicious annotations.
The OTHER cause remains unresolved. The Phase 4D automatic gate requested a
design review for any OTHER entry. Phase 4E explicitly assesses sampling as a
controlled hypothesis using completed human review, low-confidence matches,
class confusion, and TRAIN underrepresentation together. It does not claim that
imbalance caused the errors or that OTHER has been explained. The assessment,
unresolved entries, human CSV hash and automatic evidence hash are saved as
`sampling_justification.json` with each run. Targeted augmentation or more real
TRAIN data remain alternatives; resolution is not changed.

## Output protection and full-run gate

Smoke writes only `results/experiments/B_class_aware_sampling_smoke/`; full
training writes only `results/experiments/B_class_aware_sampling/`. Nonempty
output directories, including an existing planning directory, require explicit
`--overwrite`. This archives the entire old directory beside it. A_baseline is
fingerprinted before and after training and is never an output destination.

The shared lifecycle stores experiment metadata, resolved config, environment,
dataset fingerprints and framework args. B adds its sampling plan, justification,
sampler diagnostics, implementation/source hashes, checkpoint hashes and
baseline comparison results. Raw and curated fingerprints are checked before
and after the run. Runtime data YAML omits TEST and download directives. The
trainer rejects TEST before framework dataset parsing and rejects any loader
path outside the exact curated TRAIN/VALID directories.

Readiness requires a matching successful two-epoch GPU smoke, both checkpoints,
all 127 VALID images, unchanged datasets/baseline, confirmed sampler use, and
matching code/config/review/source evidence. Full training is never launched by
a checker or dry run. The manual full command is:

```powershell
python scripts/train_experiment_b.py
```

Because the full B directory already contains the Phase 4D plan, its first actual
run requires explicit archival authorization via:

```powershell
python scripts/train_experiment_b.py --overwrite
```

This phase executes only the smoke command. Smoke metrics are integration
evidence, not the Experiment B result or a model-selection decision. TEST stays
locked throughout.
