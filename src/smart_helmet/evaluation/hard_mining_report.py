"""Readable TRAIN-only mining definitions and design report."""
import json


def mining_report(summary):
    return '''# Frozen A TRAIN hard-example mining

TRAIN ONLY diagnostic analysis, not generalization performance. A was trained on
these images, so TRAIN inference is optimistic. Remaining errors motivate a
training-exposure hypothesis; they do not establish that under-exposure is the cause.

## Frozen method and difficulty definitions

Only A_baseline/best.pt, verified against prior checkpoint provenance. All 1,158
curated TRAIN images use imgsz=416, FP32, augmentation=false, NMS IoU=0.7,
max_det=300, batch=4, workers=0, rect=true and proposal floor=0.001. Operational
diagnostic confidence=0.25; production defaults are unchanged.

Predictions are matched deterministically one-to-one: confidence descending,
highest same-class IoU >=0.5, ties by lowest GT index. Correct matches at any
saved confidence precede wrong-class matching. Remaining wrong-class IoU >=0.5
pairs use the existing deterministic matching routine. Remaining same-class
localization proposals are assigned one-to-one by confidence then IoU.

- EASY_CORRECT: assigned correct-class IoU >=0.5 and confidence >=0.25.
- LOW_CONFIDENCE_CORRECT: assigned correct-class IoU >=0.5 at [0.001,0.25).
- CLASS_CONFUSION: no assigned correct match at the proposal floor, but an
  assigned wrong-class IoU >=0.5 proposal exists (possibly below 0.25).
- LOCALIZATION_FAILURE: remaining correct-class proposal assigned at 0.1 <= IoU <0.5.
- MISSED_NO_CORRECT_CLASS_PROPOSAL: no assigned match and no correct-class
  proposal at meaningful overlap (IoU >=0.1) anywhere in the saved proposals.
- MATCH_COMPETITION: no assigned proposal, but a meaningful correct-class
  proposal exists and was consumed by another GT. This extra category prevents
  shared proposals from being mislabeled as absence of a proposal.
- DUPLICATE_PREDICTION_CONTEXT: an additional, overlapping active correct-class
  proposal unused by all assignment stages. Each duplicate has one best-overlap
  GT owner. This is an overlapping context flag, not a second primary category.

The meaningful-overlap boundary 0.1 is a fixed diagnostic definition, not a
tuned threshold or validated causal distinction. Full proposals and best
correct-class IoUs are retained for review. An assigned low correct match takes
precedence over wrong-class overlaps; active wrong-class context is separately
recorded. Thus categories at the proposal floor are not operational 0.25 metrics.
Every GT has exactly one primary category, and no prediction is assigned to two
GTs. Geometric existence flags are diagnostic only, never additional TPs.

No-correct-overlap-at-0.5 and no-meaningful-proposal-at-0.1 are different concepts;
both flags are saved. Misses describe the saved post-NMS proposals down to 0.001,
not an assertion about all pre-NMS or sub-floor model outputs.

## Image tiers

Tier 3: any no-meaningful-proposal miss. Otherwise Tier 2: any class confusion or
match competition. Otherwise Tier 1: any low-confidence correct, localization
failure or duplicate context. Otherwise Tier 0: no hard GT. Crowded means >=3 GTs;
multi-class means both classes appear. Empty-GT images are Tier 0; unmatched
background predictions do not define GT difficulty. Without Helmet failures are
prioritized within each tier in the review queue; a separate flag supports audit.
No sampling weights are used in difficulty classification.

## TRAIN distribution

```json
'''+json.dumps(summary,indent=2)+'''
```

## Targeted manual sanity review

[Gallery](hard_example_gallery/index.html) shows original GT and frozen predictions,
class, confidence and IoU. Assigned low-confidence/localization proposals are shown
alongside all active predictions. Source images are never modified or copied as
an oversampled dataset; gallery panels are derived annotations only.

[Review queue](manual_review_queue.csv): up to 30 Tier 3, 15 Tier 2 and 8 Tier 1
images, prioritizing Without Helmet within tiers. Add up to two examples for each
observed primary hard category and three crowded hard scenes for coverage.
Categories with zero observations have no fabricated examples. Every item starts
PENDING; no automatic approval. Review verifies annotations, prediction semantics,
and automatic category logic before weights are finalized for implementation.
Not every TRAIN image requires review. REVIEW_REQUIRED is an acceptable completed
automatic-mining result, not a failed/incomplete inference run.

## Leakage and immutability

Every inference and sampled image resolves directly under curated v1 TRAIN/images;
symlinks/redirects outside that directory, VALID and TEST are rejected. GT labels
are checked under TRAIN/labels. No VALID identity, miss list or confidence is used
to define difficulty or weights. Prior VALID findings motivate the concept only.
Raw and curated split fingerprints are audited by byte hashing; this integrity
audit may read TEST bytes but never decodes them for inference/mining or produces
TEST predictions, metrics, sampling, threshold tuning or model selection.

A and dataset hashes are preserved. TEST STATUS: LOCKED / NOT USED. No C training
or smoke training, no trainer integration, and no production threshold change.

VALIDATION OVERUSE WARNING: the same 127-image VALID set has already supported
multiple exploratory analyses. Do not search C variants or tune weights on VALID.
Select at most one primary C configuration before training, unless a technical bug
invalidates it. A remains reference and B is not promoted.
'''
