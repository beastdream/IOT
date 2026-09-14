# Human policy correction

The current decisions exclude two TEST images through six P0 pairs. The grouped report is a read-only snapshot derived from current decisions; the existing sanity CSVs are checked for pair relationships and remain evidence of the built dataset.

Open `results/dataset_cleaning/policy_correction_report.html`, or run:

```powershell
.venv\Scripts\python.exe scripts/review_cleaning_cases.py --policy-mismatches --open-image
```

Every related pair is displayed before choices begin. Choose each record explicitly. No recommendation is preselected. A consistency warning lists all conflicting review IDs before a changed decision can be saved. Enter returns to choices; typing `SAVE` explicitly permits the individual save despite the warning, allowing sequential correction of an existing group. Use `R` after a group to revisit its records. For a conflict involving a record outside the group, revisit it with `--review-id ID`.

Notes are optional. `ACCEPT` explicitly accepts the displayed suggested note; Enter preserves existing notes. The first actual change creates one exact timestamped backup for the session, and each write uses the existing validated atomic replacement. Skip, quit and unchanged choices do not save.

After each group the preview counts unique images implied by current decisions. Mismatches count reversed split priority; needs-review counts unresolved decisions, missing notes on a reversed direction, or unknown/same-split exclusions. These counts can overlap. The report is a snapshot; rerunning the group command refreshes it.

Verify decisions without changing them:

```powershell
.venv\Scripts\python.exe scripts/check_decision_consistency.py
```

The checker requires 27 completed P0 decisions and reports conflicts, reversed priorities and evaluation exclusions. `REVIEW_REQUIRED` is an assessment, not a command execution failure. The existing `check_curation_policy.py` checks the built dataset, so its results may remain unresolved after human corrections until a separately authorized rebuild. This workflow does not rebuild or train.
