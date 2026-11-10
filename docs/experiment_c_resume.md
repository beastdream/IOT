# Resume interrupted Experiment C

From `D:\IOT` with the project environment activated:

```powershell
python scripts/train_experiment_c.py --resume --dry-run
python scripts/train_experiment_c.py --resume
```

Both commands validate the existing full run. Only the second continues training.
The only permitted checkpoint is
`results/experiments/C_hard_example_sampling/weights/last.pt`.
Missing, stripped, completed, inconsistent, or changed artifacts fail closed.
`--overwrite`, smoke mode, and training overrides are forbidden with `--resume`.
No new experiment directory is created, and A/B/mining artifacts are protected.

Validation compares the dataset and raw fingerprints, frozen Phase 4I plan and
human-review provenance, original sampler implementation and epoch draws,
baseline arguments, runtime versions and framework source, original output
paths, full VALID configuration, and TEST lock. CSV epochs must be contiguous
and agree exactly with checkpoint history; ambiguous partial writes require
manual investigation and are never silently truncated. Reported epochs are
one-based (checkpoint epoch 9 means completed epoch 10 and resume at epoch 11).

The pinned Ultralytics 8.4.128 resume loads saved EMA weights, optimizer state,
AMP scaler state (an empty dictionary is valid when AMP was disabled), and epoch.
Its epoch-based LambdaLR is reconstructed from the original arguments and
restored optimizer rates for legacy checkpoints. Existing early-stopping history
is recovered from saved detection fitness (mAP50–95); inconsistent history or
an already reached patience limit blocks resume. New checkpoints explicitly
include scheduler and early-stopping state and replace checkpoint files atomically.
The TRAIN sampler retains `seed + zero_based_epoch`, original weights, and
1158 draws with replacement. Previous completed sampler diagnostics are retained;
an uncheckpointed training-only epoch is replayed from the checkpoint.

This follows the framework's checkpoint semantics, not bitwise uninterrupted
training: the original checkpoints contain EMA weights and compressed optimizer
state, but do not contain augmentation RNG state or pending gradients.

A Windows OS file lock prevents simultaneous resume commands and releases on
shutdown. Pre-resume metadata/CSV/diagnostics and validation evidence are saved
inside C's `resume_history/`. `resume_runtime_verification.json` records restored
state checks before the first resumed batch. The existing results CSV is appended.
Original full-run metadata and start time are retained; final completion updates
the same `experiment.json`.
