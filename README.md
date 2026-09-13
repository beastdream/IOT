# SMART HELMET DETECTION SYSTEM

Current Phase: **IMAGE PHASE — PROJECT FOUNDATION**.

Task: helmet status detection from head regions. Current classes are exactly
`0 = With Helmet` and `1 = Without Helmet`. See [dataset contract](docs/dataset_contract.md).

## Scope and status

Image dataset → Dataset Audit → Dataset Cleaning → Dataset Analysis → Baseline
Training → Experiments → Evaluation → Failure Analysis → Image Inference → Local Database.

Only foundation is implemented. Dataset Audit is the next phase and must be
started separately. Video is not implemented. There is no training, database,
cloud integration or automatic dataset cleaning in this phase.

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
needs PyYAML; pytest is a test dependency. Ultralytics, torch, Pillow and numpy
were considered but are not used by this phase's code, so they are not added.
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
