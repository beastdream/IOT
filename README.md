# SMART HELMET DETECTION SYSTEM

Current Phase: **IMAGE PHASE — MANUAL REVIEW + CLEANING DECISION PREPARATION (3A)**.

Task: helmet status detection from head regions. Current classes are exactly
`0 = With Helmet` and `1 = Without Helmet`. See [dataset contract](docs/dataset_contract.md).

## Scope and status

Image dataset → Dataset Audit → Dataset Cleaning → Dataset Analysis → Baseline
Training → Experiments → Evaluation → Failure Analysis → Image Inference → Local Database.

Foundation, the read-only Dataset Audit pipeline and cleaning review preparation
are implemented. Applying cleaning must be started separately after human review. Video is not implemented.
There is no training, database, cloud integration or automatic cleaning.

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
permission to execute exclusions. No cleaning/apply function, curated dataset,
resplitting, rebalance, model training or predictions are implemented here.

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
