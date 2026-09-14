"""Evidence-based planning only; deliberately not a trainer configuration."""
import json
from pathlib import Path
import yaml

from smart_helmet.paths import PROJECT_ROOT
from .failure_review import EVALUATION


def recommend(summary, baseline):
    if summary.get('test_used') is not False or summary.get('test_status') != 'LOCKED / NOT USED' or summary.get('status') != 'PASS':
        raise ValueError('A passing, TEST-locked review is required')
    train = summary['train_distribution']; counts = train['objects']
    wh = counts['Without Helmet']; helmet = counts['With Helmet']
    cats = summary['categories']; n = summary['objects']
    evidence = f"{cats['LOW_CONFIDENCE']['count']}/{n} misses have one-to-one low-confidence same-class candidates; {cats['CLASS_CONFUSION']['count']}/{n} have wrong-class matches. TRAIN objects: With Helmet {helmet}, Without Helmet {wh}."
    composition = train['image_composition']
    positive = composition.get('without_helmet_only', 0)+composition.get('both', 0)
    negative = train['images']-positive
    # Equal probability mass for presence/absence strata; no arbitrary oversampling multiplier.
    weight = negative/positive if positive else None
    justified = weight is not None and weight > 1
    options = [
        dict(option='A. Targeted TRAIN augmentation', evidence_for=evidence,
             evidence_against='No human-confirmed blur, lighting, occlusion or view-angle coverage gap. Generic extra augmentation may obscure useful cues.',
             cost='One controlled retraining plus augmentation audit', expected_target_metric='Without Helmet recall and mAP50'),
        dict(option='B. Class-aware TRAIN image sampling', evidence_for=f'{positive}/{train["images"]} TRAIN images contain Without Helmet. '+evidence,
             evidence_against='Object imbalance alone is not proof of causation. Mixed-class images also increase With Helmet exposure; repeated images can overfit.',
             cost='Sampler implementation, exposure audit and one baseline-sized training run', expected_target_metric='Without Helmet recall with precision and overall mAP50 safeguards'),
        dict(option='C. More real Without Helmet TRAIN data', evidence_for='Additional independent examples could broaden coverage. '+evidence,
             evidence_against='Specific semantic coverage deficits are unconfirmed; requires new data version and leakage/annotation checks.',
             cost='Collection, annotation, audit and retraining; highest data cost', expected_target_metric='Without Helmet recall and mAP50'),
        dict(option='D. Higher resolution', evidence_for=f"Small/tiny missed boxes: {summary['sizes']}.",
             evidence_against='No tiny failures. Area association alone does not demonstrate insufficient pixel detail; no resolution ablation supports 640.',
             cost='Higher memory, training and inference latency', expected_target_metric='Small-object recall and localization AP'),
        dict(option='E. Threshold adjustment', evidence_for=evidence,
             evidence_against='Low-confidence candidates overlap categories and need rematching; lowering confidence may increase FP. It does not improve AP or demonstrate better training.',
             cost='Offline VALID precision/recall sweep; deployment threshold requires separate selection', expected_target_metric='Fixed-operating-point recall subject to precision constraint'),
        dict(option='F. Larger model', evidence_for='Recall and localization leave room for improvement.',
             evidence_against='No evidence isolates capacity as the limiting factor; changes compute/latency substantially.',
             cost='Higher memory, training time and inference latency', expected_target_metric='Without Helmet mAP50 and recall'),
    ]
    plan = dict(plan_only=True, training_authorized=False, baseline_reference='A_baseline',
        primary_problem='Without Helmet under-detection with low-confidence and class-confusion evidence; semantic causes unresolved',
        primary_metric='Without Helmet Recall', baseline_recall=summary['metrics']['per_class']['1']['recall'],
        proposed_change=dict(factor_group='TRAIN image sampling', strategy='Balance probability mass of images containing Without Helmet and images without it',
                             positive_image_weight=weight, other_image_weight=1.0, epoch_draws=train['images'],
                             replacement=True, seed=baseline['seed'], dataset_edits=False),
        constants_from_baseline={k: baseline[k] for k in ('model', 'imgsz', 'batch', 'workers', 'seed', 'epochs', 'patience')},
        additional_constants='Preserve exact baseline args, optimizer resolution, pretrained initialization, augmentations, split, epoch length, evaluator and thresholds; only sampler changes.',
        experiment_rationale=evidence+f' Image-presence weight {weight} balances the two strata; this tests exposure, not a proven cause.',
        expected_observation='Fewer Without Helmet misses at the unchanged operating point, improved framework recall and mAP50; improvement is not guaranteed.',
        success_criteria=[
            'Compare framework Without Helmet recall against 0.5304 using identical evaluation settings; also report fixed-confidence 0.25 metrics separately.',
            'Require recall improvement with a positive lower bound of a paired 95% image-bootstrap difference interval; resample whole images to preserve object dependence. This conventional uncertainty criterion avoids a fabricated effect-size cutoff.',
            'Conservative precision safeguard: Without Helmet precision must not fall below baseline 0.7344789657; overall mAP50 must not fall below 0.7547054629. Report paired uncertainty; inconclusive evidence is not a PASS.',
            'Report Without Helmet mAP50, overall recall, FP/FN counts, per-size and per-scene results; do not select a threshold to manufacture training gains.',
            'Treat one-seed results as provisional; training-run variability remains unmeasured. TEST stays locked through model selection.'
        ],
        higher_resolution_justified=False, test_status='LOCKED / NOT USED',
        ready_to_train=False,
        prerequisites=['Human review of all pending cases; flag suspicious annotations without editing VALID.',
                       'Implement and verify sampler, positive/negative exposure and unchanged epoch length in a future authorized training phase.',
                       'Resolve and preserve baseline optimizer configuration before training.'],
        recommendation='B. Class-aware TRAIN image sampling' if justified else 'Defer training intervention pending human coverage review')
    if not justified:
        plan['proposed_change'] = {'factor_group': 'none', 'strategy': 'Pending evidence'}
    return plan, options


def write_plan(path, plan):
    if plan.get('plan_only') is not True or plan.get('training_authorized') is not False:
        raise ValueError('Only non-executable plans may be serialized')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding='utf-8')


def run_plan(root=PROJECT_ROOT):
    root = Path(root); out = root/EVALUATION/'failure_review'
    summary = json.loads((out/'failure_review_summary.json').read_text())
    baseline = yaml.safe_load((root/'results/experiments/A_baseline/args.yaml').read_text())
    plan, options = recommend(summary, baseline)
    experiment = json.loads((root/'results/experiments/A_baseline/experiment.json').read_text())
    plan['resolved_optimizer_from_baseline'] = {k: experiment[k] for k in
        ('optimizer_class', 'optimizer_initial_parameter_groups')}
    plan['prerequisites'][-1] = 'Verify future training preserves the recorded AdamW parameter groups and all baseline settings except sampling.'
    write_plan(root/'configs/experiments/B_plan.yaml', plan)
    lines = ['# Experiment B recommendation', plan['recommendation'], plan['experiment_rationale'],
             '## Decision table (50 missed objects; categories overlap)', '| Category | Count | Percentage |', '|---|---:|---:|']
    lines += [f'| {k} | {v["count"]} | {v["percentage"]:.1f}% |' for k, v in summary['categories'].items()]
    lines += ['Visual category zeros mean UNASSIGNED, not absent. All 50 objects require human confirmation. Crowded means >=3 GT; multiple-object means >=2. Small includes tiny below 0.001 and small below 0.01 normalized area.',
              '## TRAIN versus VALID failures', 'Counts below are object-weighted size/scene distributions; image composition is image-weighted. Failure-only fractions cannot establish a coverage gap; full VALID denominators are included to assess selection effects.',
              '```json', json.dumps({k:summary[k] for k in ('train_distribution', 'validation_distribution', 'sizes', 'scene_objects', 'distribution_comparison')}, indent=2), '```',
              'The distributions measure geometric and count coverage only. They cannot establish missing lighting, blur, occlusion or viewpoint coverage. Review the HTML before selecting appearance-specific augmentation.',
              '## Candidate experiments']
    comparison = summary['distribution_comparison']
    small = comparison['size']['small']; crowded = comparison['scene']['three_or_more']
    lines.insert(-1, f"Small (not tiny) objects are {small['train_percentage']:.1f}% of TRAIN Without Helmet objects and {small['failure_percentage']:.1f}% of misses. Scenes with >=3 GT account for {crowded['train_percentage']:.1f}% of TRAIN Without Helmet objects and {crowded['failure_percentage']:.1f}% of misses. These forms are already common in TRAIN; their presence among failures does not establish missing coverage. Small-object VALID miss rate is {small['validation_miss_rate']:.1%}; crowded-scene VALID miss rate is {crowded['validation_miss_rate']:.1%}. No new evidence isolates resolution as the bottleneck.")
    for option in options:
        lines += ['### '+option['option']] + [f'**{k}:** {v}' for k, v in option.items() if k != 'option']
    lines += ['## Controlled plan', '```yaml', yaml.safe_dump(plan, sort_keys=False), '```',
              'Framework recall 0.5304 uses the framework-selected operating point. The 50 misses use confidence 0.25 and IoU 0.5; they are not interchangeable metrics.',
              'TEST USED: NO. TEST STATUS: LOCKED. READY TO TRAIN: NO; plan and human review prerequisites remain. No training was performed.']
    (out/'experiment_b_recommendation.md').write_text('\n\n'.join(lines), encoding='utf-8')
    print(plan['recommendation']); return plan
