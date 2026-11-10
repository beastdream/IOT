"""Frozen Phase 4I weights: no inference, tier calculation, or adaptive sampling."""
from collections import Counter
import csv
import json
import math
from pathlib import Path

from .class_aware_sampler import ClassAwareSampler, index_hash, ordered_inventory
from .experiment_b import file_hash
from .sampling import records, train_path, exposure
from .hard_sampling_plan import review_policy

PLAN = Path('results/hard_example_mining/A_baseline_train/sampling_plan.json')
TIERS = ('TIER_0_EASY', 'TIER_1_MODERATE', 'TIER_2_HARD', 'TIER_3_CRITICAL')
TIER_WEIGHTS = dict(zip(TIERS, (1.0, 1.3333333333333333, 1.6666666666666665, 2.0)))


def validate_mapping(root, plan, rows):
    required = dict(plan_stage='post_human_review', human_review_gate='PASS',
        human_review_pending=0, review_more=0, difficulty_source='TRAIN_ONLY',
        train_images=1158, effective_epoch_samples=1158, seed=42,
        sampling_strategy='runtime_weighted_sampling_with_replacement',
        valid_leakage=0, test_leakage=0, test_status='LOCKED')
    if any(plan.get(k) != v for k, v in required.items()):
        raise ValueError('Frozen post-human-review TRAIN-only plan required')
    if plan.get('proposed_tier_weights') != TIER_WEIGHTS:
        raise ValueError('Frozen tier weights changed')
    paths = [train_path(root, row['image']) for row in rows]
    names = {row['image'] for row in rows}
    if len(rows) != 1158 or len(set(paths)) != 1158 or set(plan['image_sampling_weights']) != names:
        raise ValueError('Every TRAIN image requires exactly one weight')
    excluded = set(plan['suspicious_excluded_images'] + plan['not_actually_hard_excluded_images'])
    if excluded != set(plan['excluded_from_upweighting']) or not excluded <= names:
        raise ValueError('Human exclusion mapping mismatch')
    eligible = {r['image'] for r in rows if r['tier'] != TIERS[0]} - excluded
    if eligible != set(plan['eligible_hard_images']) or len(plan['eligible_hard_images']) != len(eligible):
        raise ValueError('Eligible hard-image mapping mismatch')
    if dict(Counter(r['tier'] for r in rows)) != plan['tier_counts']:
        raise ValueError('Frozen tier inventory mismatch')
    for row in rows:
        expected = 1.0 if row['image'] in excluded else TIER_WEIGHTS[row['tier']]
        if plan['image_sampling_weights'][row['image']] != expected:
            raise ValueError('Weight mapping violates frozen tier or human exclusion')


def load_plan(root):
    root = Path(root).resolve()
    plan = json.loads((root/PLAN).read_text(encoding='utf-8'))
    difficulty = root/PLAN.parent/'train_image_difficulty.csv'
    if file_hash(difficulty) != plan['mining_image_csv_sha256']:
        raise ValueError('Frozen difficulty CSV changed')
    with difficulty.open(encoding='utf-8', newline='') as stream:
        tiers = list(csv.DictReader(stream))
    inventory = records(root)
    by_name = {r['image']: r for r in inventory}
    if len(tiers) != len(by_name) or {r['image'] for r in tiers} != set(by_name):
        raise ValueError('Difficulty CSV differs from physical TRAIN')
    rows = [dict(by_name[r['image']], tier=r['tier']) for r in tiers]
    validate_mapping(root, plan, rows)
    _, review = review_policy(root, rows)
    if any(plan.get(k) != v for k, v in review.items()):
        raise ValueError('Human review provenance changed')
    if len(plan['eligible_hard_images']) != 156 or len(plan['suspicious_excluded_images']) != 1:
        raise ValueError('Approved post-review population changed')
    provenance = json.loads((root/PLAN.parent/'provenance.json').read_text())
    if file_hash(root/'results/experiments/A_baseline/weights/best.pt') != provenance['checkpoint_sha256']:
        raise ValueError('Mining checkpoint provenance changed')
    if file_hash(root/'results/experiments/A_baseline/experiment.json') != plan['baseline_experiment_sha256']:
        raise ValueError('Baseline provenance changed')
    if file_hash(root/PLAN.parent/'train_predictions.json') != plan['mining_prediction_sha256']:
        raise ValueError('Frozen mining provenance changed')
    for name, digest in plan['pre_review_archive_sha256'].items():
        if file_hash(root/name) != digest:
            raise ValueError('Pre-review archive changed')
    return plan, rows


class HardExampleSampler(ClassAwareSampler):
    """Reuse B's epoch-explicit deterministic iterator with C's immutable weights."""
    def __init__(self, root, rows, plan):
        validate_mapping(root, plan, rows)
        self.rows = rows
        self.plan = plan
        self.weights = tuple(plan['image_sampling_weights'][r['image']] for r in rows)
        self.seed = 42
        self.epoch = 0
        self.num_samples = 1158
        self.iteration_count = 0
        self.yielded_indices = []

    def diagnostics(self, indices, actually_used=False):
        counts = Counter(indices)
        tiers = Counter(self.rows[i]['tier'] for i in indices)
        eligible = set(self.plan['eligible_hard_images'])
        hard = sum(self.rows[i]['image'] in eligible for i in indices)
        expected = self.plan['expected_tier_draws']
        comparison = {t: dict(expected=expected[t], actual=tiers[t],
            tolerance=max(5.0, 6*math.sqrt(expected[t]*(1-expected[t]/self.num_samples)))) for t in TIERS}
        plausible = all(abs(v['actual']-v['expected']) <= v['tolerance'] for v in comparison.values())
        if actually_used and (indices != self.indices() or not plausible):
            raise ValueError('Actual exposure contradicts deterministic frozen sampling plan')
        return dict(sampler_class=type(self).__name__, replacement=True, num_samples=1158, seed=42,
            tier_weights=self.plan['proposed_tier_weights'], train_images=len(self.rows),
            **{f'tier_{i}_images': self.plan['tier_counts'][t] for i,t in enumerate(TIERS)},
            eligible_hard_images=len(eligible), suspicious_excluded=len(self.plan['suspicious_excluded_images']),
            not_actually_hard_excluded=len(self.plan['not_actually_hard_excluded_images']),
            sampled_indices=list(indices), sampled_index_hash=index_hash(indices),
            ordered_train_images=[r['image'] for r in self.rows],
            sampled_tier_counts={t:tiers[t] for t in TIERS}, sampled_hard_image_count=hard,
            sampled_easy_image_count=tiers[TIERS[0]], sampled_baseline_weight_count=len(indices)-hard,
            sampled_unique_images=len(counts), maximum_sample_count=max(counts.values(), default=0),
            actual_sample_count=len(indices), estimated_object_exposure=exposure(self.rows, [counts[i] for i in range(len(self.rows))]),
            exposure_scope='Primary TRAIN draws; unchanged mosaic auxiliary draws excluded',
            plan_vs_actual=comparison, exposure_plausible=plausible if indices else None,
            sampler_actually_used=actually_used, epoch=self.epoch,
            valid_leakage=0, test_leakage=0, test_used=False)
