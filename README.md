# SMART HELMET DETECTION SYSTEM

Current Phase: **IMAGE PHASE — PRE-TRAINING CURATION SANITY CHECK (3C)**.

Task: helmet status detection from head regions. Current classes are exactly
`0 = With Helmet` and `1 = Without Helmet`. See [dataset contract](docs/dataset_contract.md).

## Scope and status

Image dataset → Dataset Audit → Dataset Cleaning → Dataset Analysis → Baseline
Training → Experiments → Evaluation → Failure Analysis → Image Inference → Local Database.

Foundation, the read-only Dataset Audit pipeline and cleaning review preparation
are implemented, along with explicit P0-only curated builds after human review.
Video, model training, database and cloud integration are not implemented.

| Split | Images |
| --- | ---: |
| Train | 1185 |
| Valid | 127 |
| Test | 64 |
| Total | 1376 |

Dataset folders remain directly at the checkout root. Original `data.yaml`,
`README.dataset.txt`, `README.roboflow.txt`, images and labels are preserved.
Metrics mentioned in Roboflow README files are **not metrics of this project**.

## Setup and verification

Use Python 3.11 or newer. The inspected interpreter was Python 3.11.9.
From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import smart_helmet"
python -m compileall src scripts tests
python scripts/verify_foundation.py
python -m pytest -q
```

`pip install -e .` installs the src-layout package; `pip install -e ".[test]"`
also installs tests. `requirements.txt` delegates to that test extra, so there
is one dependency declaration in `pyproject.toml`. Foundation runtime only
needs PyYAML; audit additionally uses Pillow and numpy for decoding, overlays and
perceptual hashing. Pytest is a test dependency. Ultralytics and torch are not
used by the audit and remain outside the declared dependencies.
Observed existing versions were Ultralytics 8.4.128, torch 2.13.0, Pillow 12.2.0,
numpy 2.4.6, PyYAML 6.0.3 and pytest 9.1.1. This is an environment observation,
not a validated training stack or lockfile.

For this handoff, `.venv` was created with `--system-site-packages` to reuse
installed dependencies without downloads. Its seeded setuptools 65.5.0 failed
the editable build (`bdist_wheel` unavailable); removing only that venv-local
copy exposed the existing system setuptools 84.0.0. Installation then passed
with `python -m pip install --no-deps --no-build-isolation -e ".[test]"`.
The system Python installation was not modified. The normal setup above uses
pip build isolation to satisfy the declared build requirement automatically.

Verification is read-only, reports FAIL and returns exit code 1 for invalid
config, missing paths or unexpected image counts. It supports multiple image
extensions case-insensitively. It does not audit annotations or repair data.
“Original export files present” checks existence, not historical byte integrity;
the implementation handoff separately compares before/after SHA-256 hashes.

## Paths and Ultralytics compatibility

Use `configs/data.local.yaml`, not the original export YAML. It uses `path: .`
and `train/images`, `valid/images`, `test/images`, without `../` fallback.
For native Ultralytics YAML consumption, the working directory must be the
project root. In the inspected Ultralytics 8.4.128 resolver, an existing relative
`path` is interpreted from the working directory, not the config directory;
therefore `path: ..` here would be incorrect.

For Python code running from any working directory:

```python
from smart_helmet.foundation import load_data_config
from smart_helmet.paths import PROJECT_ROOT

data = load_data_config()  # Mapping with absolute root and split paths.
```

Paths derive from the installed source file location, with no machine-specific
root. This is a source-checkout/editable-install project: distributing a wheel
without the adjacent dataset is not supported. Do not pass this mapping blindly
to APIs that require a YAML filename; a future training entry point must use the
documented project-root cwd or explicitly serialize a resolved runtime config.

## Layout

```text
configs/                 Local dataset configuration
docs/                    Dataset contract
src/smart_helmet/        Paths and foundation verification
src/smart_helmet/dataset/ Reserved namespace for next phase
scripts/                 Verification entry point
tests/                   Foundation tests
results/
  dataset_audit/
  dataset_analysis/
  experiments/
  evaluation/
  predictions/
  failure_analysis/
train/, valid/, test/    Original dataset, unchanged
```

`.gitignore` excludes Python caches, environments, model weights and runtime
results while retaining result-folder placeholders. It does not ignore source,
config, tests, documentation or the dataset. Ignore rules do not remove files.

## Dataset Audit

Run `.venv\Scripts\python.exe scripts/audit_dataset.py` from the project root.
See [audit methodology and report guide](docs/dataset_audit.md) for thresholds,
output schemas and manual review instructions. Results are generated under
`results/dataset_audit/`; `REVIEW_REQUIRED` is a valid completed audit outcome.
The audit does not start cleaning or training.

## Cleaning review preparation (Phase 3A)

```powershell
.venv\Scripts\python.exe -m compileall src scripts tests
.venv\Scripts\python.exe scripts/prepare_cleaning_review.py
.venv\Scripts\python.exe scripts/validate_review_decisions.py
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts/verify_foundation.py
```

Open [the static review report](results/dataset_cleaning/review_report/index.html)
in a browser after preparation. It contains all cross-split comparisons (P0),
tiny-box cases with context zooms (P1), unusual resolutions (P2), and a fixed-seed
sample of up to 40 source groups (P3). It does not require reviewing the complete
audit's manual-evidence list. P0 pairs and P1 boxes remain separate decisions even
when the same image occurs in several cases; other image reasons are merged into
the evidence field.

Use the interactive P0 tool below to record human decisions safely in
`results/dataset_cleaning/review_decisions.csv`. Allowed values are documented in
`results/dataset_cleaning/cleaning_policy.md`.
The report is read-only. All template decisions start PENDING. The validator
accepts a valid pending file (exit 0), but reports READY TO APPLY CLEANING = NO
while P0 is PENDING or REVIEW_MORE. REVIEW_MORE is unresolved, not completed.
Other priorities may remain pending and are reported separately.

Preparation verifies the full raw dataset against the audit fingerprint before
and after generating artifacts. It uses audit CSVs rather than rerunning the
audit or applying changes. Outputs are under `results/dataset_cleaning/`:
review queue, source-group summary, conditional proposed actions, decisions,
review summary, policy, annotated comparisons and HTML. No additional dependency
is introduced. Dataset and original export files remain untouched.

Rerunning preparation generates stable review IDs from image and annotation
identity, deduplicates review units, and preserves existing decision values and
notes. Existing IDs that no longer match the queue, duplicate IDs or invalid
decisions cause an error without overwriting human input. Decisions entered while
preparation is running are also preserved, and the command asks for a rerun.
Current CSV-linked images are authoritative; old report images are not deleted.

`proposed_actions.csv` is suggestions only: approval is required, approved=false,
applied=false. A completed P0 review is a gate for planning the next phase, not
permission to execute exclusions automatically. Phase 3B below provides a separate
explicit build command. Preparation itself never applies decisions.

## Interactive P0 review

```powershell
.venv\Scripts\python.exe scripts/review_cleaning_cases.py --priority P0 --pending-only --open-image
.venv\Scripts\python.exe scripts/open_cleaning_review.py
```

The default priority is P0. `--open-image` opens each comparison in the default
image application; omit it to review using the printed path or HTML. The report
opener uses Python's webbrowser module and the default browser. No browser path
or additional dependency is required.

Choices: **1** KEEP_BOTH, **2** KEEP_A_EXCLUDE_B, **3** KEEP_B_EXCLUDE_A,
**4** REVIEW_MORE, **S** skip, **Q** quit. A/B refer to the displayed queue fields,
not a fixed train/evaluation ordering. The tool shows both splits, filenames,
hash distance, source groups, object/class counts and current decision/notes.
An empty notes response keeps the previous notes; entered text replaces notes.
Each saved decision persists immediately. Skipping, quitting or interrupting an
unsaved choice does not change that case. P1/P2/P3 decisions are never updated.

`--pending-only` resumes remaining PENDING cases. REVIEW_MORE is unresolved for
readiness but excluded from this filter; omit `--pending-only` to revisit it.
Progress reports completed, pending and REVIEW_MORE separately. The tool never
applies cleaning or marks proposed actions approved.

Before the first save in a session, an exact timestamped backup is created under
`results/dataset_cleaning/backups/`. No backup is needed for a skip/quit-only
session. Saves validate the full candidate CSV, write and validate a temporary
file in the same directory, then use atomic replacement and validate again.
Row order, other decisions and unchanged notes are preserved. Existing invalid
CSV, changed queue or detected external edits cause an error instead of silently
overwriting them. Avoid editing the decision CSV in another application while
reviewing. A file-lock/replace failure leaves the original in place.

The HTML generator now writes indented multiline markup and CSS. Regenerating
the report preserves existing decisions; the report remains a read-only viewer.

## Curated dataset v1 (Phase 3B)

```powershell
.venv\Scripts\python.exe scripts/validate_review_decisions.py
.venv\Scripts\python.exe scripts/apply_dataset_cleaning.py --dry-run
# Continue only when the dry-run reports PASS:
.venv\Scripts\python.exe scripts/apply_dataset_cleaning.py --apply
.venv\Scripts\python.exe scripts/audit_dataset.py --config configs/data.curated.v1.yaml --output results/dataset_audit/curated_v1
.venv\Scripts\python.exe scripts/check_training_readiness.py
.venv\Scripts\python.exe scripts/verify_foundation.py
```

The builder reads final human P0 decisions, never suggestions. Every image has
one final KEEP/EXCLUDE action. Conflicting KEEP and EXCLUDE votes abort the entire
build before copying; PENDING/REVIEW_MORE P0 also abort. Dry-run performs read-only
validation and prints unique exclusions, per-split counts and class counts.
KEEP_A/KEEP_B follows the recorded A/B order even when it differs from the original
evaluation-preference suggestion. Nothing is inferred from model performance.

Retained images and labels are copied with `shutil.copy2` to `data/curated/v1/`,
preserving original filenames and splits. No hardlinks, symlinks, resizing or
annotation edits are used. P1/P2/P3 create no additional exclusions: their data
is retained except where the same image is explicitly excluded by P0. Class IDs
and names remain unchanged. Raw folders and export documents remain immutable.

Copies are built in an owned staging directory and hash-checked before publishing
v1. A label-copy failure aborts, cleans staging and never reports a successful
partial build. Existing v1 is never silently merged; use `--apply --rebuild`
explicitly to replace only that v1 tree. Paths and reparse points are checked before
recursive removal. No raw directory is a rebuild target.

`configs/data.curated.v1.yaml` uses `path: data/curated/v1` and split-local image
paths. Run native YOLO commands from the project root. Python code uses
`load_detection_config(config, PROJECT_ROOT)` to resolve paths independently of
cwd; the original `load_data_config` remains strict for raw foundation checks.
The audit's default command still audits raw; `--config` and `--output` select the
curated version without changing the original audit outputs.

Build evidence lives in `results/dataset_cleaning/curated_v1_*`: a record for every
raw image, exclusions linked to human P0 reviews, summary, and hashes of every
curated image, label and config. Multiple contributing review IDs are pipe-separated
in manifests. The Phase 3A proposed-actions file stays historical suggestions;
the curated manifests are the authoritative record of what was actually applied.

Curated audit adds `cross_split_review.csv`, distinguishing the exact retained
REVIEWED_KEEP_BOTH_CANDIDATE from UNREVIEWED_CROSS_SPLIT_CANDIDATE. Review evidence
is tied to unchanged build input hashes and file provenance. Confirmed excluded
pairs cannot remain together in the built image set. An unreviewed cross-split
candidate blocks readiness; an exact pair explicitly reviewed KEEP_BOTH does not.

Readiness checks current decisions/action-map, raw immutability, curated config,
independent copied files, fingerprint, build provenance and a matching completed
audit. General audit status may remain REVIEW_REQUIRED for P1/P2/P3 observations;
these are warnings, not automatic baseline blockers. Changed decisions or stale
fingerprints/audit require rebuilding/re-auditing before readiness can pass.
Even READY FOR BASELINE TRAINING = YES does not start training.

## Pre-training split-priority sanity check (Phase 3C)

```powershell
.venv\Scripts\python.exe scripts/check_curation_policy.py
```

This read-only check traces every built exclusion to its P0 review and retained
partner. Split priority is TEST=3, VALID=2, TRAIN=1. A numerically valid final
decision can still reverse this policy. No human choice is changed, and no
dataset is rebuilt by the checker. Training readiness now includes this policy
gate and reports NO while mismatches or unresolved evidence remain.

Outputs under `results/dataset_cleaning/`:

- `pretraining_curation_sanity.csv`: one evidence row per contributing P0 pair.
- `test_exclusion_review.csv`: all contributing reviews for excluded test images.
- `pretraining_curation_summary.json`: distinct-image counts and status.
- `pretraining_review/`: annotated A/B comparisons with retained/excluded markers.
- `pretraining_curation_report.html`: formatted browser report covering test,
  problematic and all exclusions.

Multiple pair rows can reference one excluded image. Summary counts use unique
excluded images. `direction_status=POLICY_MISMATCH` means the higher-priority
split was excluded. With empty notes, `policy_status=NEEDS_REVIEW`: intent to
make an exception has not been established. Thus the direction-mismatch count
and needs-review count can overlap. Notes are displayed verbatim; the checker
does not infer approval of a policy exception from arbitrary text. Any documented
exception needs explicit human assessment rather than being silently accepted.

Revisit a completed P0 case safely, using an ID from the report:

```powershell
.venv\Scripts\python.exe scripts/review_cleaning_cases.py --review-id <ID> --open-image
```

`--review-id` overrides `--pending-only` for that case. It shows the existing
decision, keeps old notes when Enter is pressed, and uses the existing per-session
backup, atomic write and validator. No decision is entered automatically. For an
image participating in several pairs, review all related IDs; inconsistent
decisions will be rejected by the cleaning dry-run.

After human correction, these are **manual follow-up commands**, not commands
run automatically in Phase 3C:

```powershell
.venv\Scripts\python.exe scripts/validate_review_decisions.py
.venv\Scripts\python.exe scripts/apply_dataset_cleaning.py --dry-run
# Only after PASS and an explicit decision to rebuild:
.venv\Scripts\python.exe scripts/apply_dataset_cleaning.py --apply --rebuild
.venv\Scripts\python.exe scripts/audit_dataset.py --config configs/data.curated.v1.yaml --output results/dataset_audit/curated_v1
.venv\Scripts\python.exe scripts/check_curation_policy.py
.venv\Scripts\python.exe scripts/check_training_readiness.py
```

Changing a decision does not alter the already-built curated version; manifests
and readiness continue to detect the difference until a deliberate rebuild and
re-audit. No training is performed in this phase.

## Phase 4A Baseline Training

The baseline uses the installed Ultralytics detection framework and the official
YOLO11 nano checkpoint, `yolo11n.pt` (architecture verified against the installed
package; [official model documentation](https://docs.ultralytics.com/models/yolo11/)).
Install training dependencies if missing with `python -m pip install -e ".[training]"`.
The actual Python, PyTorch and Ultralytics versions are recorded for each experiment.
The verified initial environment uses Ultralytics 8.4.128 and PyTorch 2.13.0+cpu.

Run from the project root:

```powershell
.venv\Scripts\python.exe scripts/check_training_environment.py
.venv\Scripts\python.exe scripts/train_baseline.py --dry-run
.venv\Scripts\python.exe scripts/train_baseline.py --smoke-test
# Run manually only after the smoke test succeeds:
.venv\Scripts\python.exe scripts/train_baseline.py
```

`configs/baseline.yaml` configures the baseline: 416px, 100 epochs, patience 20,
seed 42, deterministic mode, pretrained weights, and two requested workers.
416px matches the predominant source resolution and keeps this baseline efficient.
A 640px tiny-object experiment is deferred. No custom augmentation is introduced.
**Baseline uses Ultralytics framework-default training augmentation for the installed version.**
The resolved framework arguments, including augmentation defaults, are saved in
`framework_resolved_args.yaml`, `args.yaml`, and experiment metadata.

`device: auto` selects the current CUDA GPU when available and falls back to CPU.
`batch: auto` uses supported single-GPU auto-batch; CPU uses a reported conservative
batch of 2. Override explicitly with `--device`, `--batch`, `--workers`, or `--epochs`.
For example, `--smoke-test --batch 2 --workers 0` avoids DataLoader multiprocessing.
Ultralytics may itself resolve CPU workers to zero; both requested and resolved
values are recorded. CUDA OOM is reported without silent retries; reduce batch
explicitly. Exact numerical reproducibility across devices/framework versions is
not guaranteed despite fixed seed and deterministic mode.

Smoke training uses two epochs, a 2% runtime TRAIN fraction when supported, and
full VALID validation. It does not create a new split or alter source files.
`--epochs` with `--smoke-test` is limited to two. Smoke metrics only establish that
loading, forward/backward, validation and checkpoint saving work.

Full output: `results/experiments/A_baseline/`.
Smoke output: `results/experiments/A_baseline_smoke/`.
Weights are in `weights/best.pt` and `weights/last.pt`. The full run fails if best.pt
is missing; last.pt is never relabeled as best.pt. Full training requires successful
smoke metadata matching the dataset fingerprint, model, image size and framework.
Existing outputs are protected: `--overwrite` explicitly archives the previous
experiment into a timestamped sibling before starting a clean output directory.
Alternatively set a new `experiment_name` in a config selected by `--config`.
Official downloaded weights are stored in `results/model_cache/`.

Curated v1 is frozen: no edits, moves, relabeling, resplitting, rebalancing, or on-disk
resizing. Readiness and the existing raw/curated fingerprints are verified before
training and again after training, including failure paths. Fingerprints and hashes
are stored in each experiment. Image caching is disabled; label-cache writes are
redirected into that experiment's `dataset_cache/`, including during final validation.

**TEST is locked until Final Evaluation.** The framework receives an experiment-local
YAML containing only the original TRAIN and VALID paths. TEST is omitted, and the
validation split is explicitly `val`. Do not use TEST to choose epochs, confidence,
augmentations, model sizes or hyperparameters. Validation metrics are not final TEST
metrics. This phase performs no standalone evaluation, video processing, failure
analysis, database or cloud work. Full training is a manual command, not part of
pipeline verification.

Experiment metadata also records the optimizer class and its initial parameter-group
settings after framework auto-selection, plus the actual two-class model parameter
count. Input defaults such as `optimizer: auto` are retained separately. The wrapper
disables the installed framework's first-epoch automatic OOM retries so a batch
change requires an explicit new invocation. Framework image-repair writes into raw
or curated image directories are refused. External analytics and experiment logger
callbacks are disabled for these local runs.
