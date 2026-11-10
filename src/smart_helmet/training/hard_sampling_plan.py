"""Design only: TRAIN difficulty exposure diagnostics, no sampler/trainer integration."""
from collections import Counter
import argparse
import csv
import json
import math
from pathlib import Path
import random

import yaml

from smart_helmet.paths import PROJECT_ROOT
from smart_helmet.dataset.duplicates import sha256_file
from smart_helmet.training.sampling import train_path
from smart_helmet.evaluation.hard_mining import OUT, TIERS, analyze, check
from smart_helmet.evaluation.hard_mining_report import mining_report
from smart_helmet.evaluation.train_human_review import ReviewStore, summarize, exclusive_lock, atomic_bytes

PLAN = Path('configs/experiments/C_hard_example_sampling_plan.yaml')
SEED=42


def choose_weights(rows):
    """One formula, no validation search: <=10% extra mass; relative weight <=2.

    Tier ranks 0/1/2/3 encode ordered severity only. The budget/cap are explicit
    conservative design choices, not learned optima or statistical guarantees.
    """
    counts=Counter(r['tier'] for r in rows)
    n=len(rows);severity=sum(i*counts[t] for i,t in enumerate(TIERS))
    step=min(.10*n/severity,1/3) if severity else 0.
    return {t:1+i*step for i,t in enumerate(TIERS)}


def dominance(n, probabilities, tier_counts, tier_fractions, expected_unique):
    maximum=max(probabilities)*n
    tiny_dominance=any(tier_counts[t]/n<=.05 and tier_fractions[t]>.20 for t in TIERS)
    if maximum>3 or tiny_dominance:
        return 'FAIL'
    if maximum>2 or expected_unique < .60*n:
        return 'WARNING'
    return 'PASS'


def build(rows,weights=None,seed=SEED,excluded_images=None):
    if not rows or len({r['image'] for r in rows})!=len(rows):raise ValueError('Unique nonempty TRAIN inventory required')
    if any(r['tier'] not in TIERS for r in rows): raise ValueError('Unknown difficulty tier')
    excluded=set(excluded_images or ())
    if excluded - {r['image'] for r in rows}:raise ValueError('Excluded image outside TRAIN inventory')
    weight_rows=[dict(r,tier=TIERS[0]) if r['image'] in excluded else r for r in rows]
    weights=weights or choose_weights(weight_rows)
    if set(weights)!=set(TIERS) or any(not math.isfinite(v) or v<=0 for v in weights.values()):
        raise ValueError('Finite positive tier weights required')
    n=len(rows);raw=[1.0 if r['image'] in excluded else weights[r['tier']] for r in rows];total=sum(raw);p=[v/total for v in raw]
    expected=[n*v for v in p];tier_counts=Counter(r['tier'] for r in rows)
    counts={t:tier_counts[t] for t in TIERS}
    fractions={t:sum(v for r,v in zip(rows,p) if r['tier']==t) for t in TIERS}
    indices=random.Random(seed).choices(range(n),weights=raw,k=n)
    repeat=random.Random(seed).choices(range(n),weights=raw,k=n)
    if indices!=repeat: raise ValueError('Non-deterministic plan')
    observed=Counter(indices)
    unique=sum(-math.expm1(n*math.log1p(-v)) if v<1 else 1 for v in p)
    top=sorted([dict(image=r['image'],tier=r['tier'],expected_draws=e,probability=v,
                     simulated_epoch_zero_count=observed[i]) for i,(r,e,v) in enumerate(zip(rows,expected,p))],
               key=lambda r:(-r['expected_draws'],r['image']))[:10]
    result = dict(train_images=n,effective_epoch_samples=n,seed=seed,epoch_seed='42 + zero-based epoch',
        sampling_strategy='runtime_weighted_sampling_with_replacement',plan_only=True,training_authorized=False,
        difficulty_source='TRAIN_ONLY',tier_counts=counts,proposed_tier_weights=weights,
        weight_rule='w_t=1+t*min(0.10*N/sum(n_t*t),1/3); t in {0,1,2,3}; if no hard images all weights 1',
        weight_rule_rationale='One fixed TRAIN-count-driven proposal: additional unnormalized mass <=10% of uniform mass and maximum relative per-image weight <=2. Severity ranks preserve tier order; no VALID tuning. These safety budgets are design assumptions, not empirically optimal weights.',
        extra_sampling_mass_fraction=(total-n)/n,expected_tier_fraction=fractions,
        expected_tier_draws={t:n*v for t,v in fractions.items()},
        expected_class_exposure={name:sum(r[key]*e for r,e in zip(rows,expected)) for name,key in
                                 [('With Helmet','with_helmet_gt_count'),('Without Helmet','without_helmet_gt_count')]},
        baseline_class_exposure={name:sum(r[key] for r in rows) for name,key in
                                 [('With Helmet','with_helmet_gt_count'),('Without Helmet','without_helmet_gt_count')]},
        expected_unique_images_per_epoch=unique,baseline_without_replacement_unique=n,
        uniform_replacement_expected_unique=n*(1-(1-1/n)**n),effective_sample_size=1/sum(v*v for v in p),
        maximum_expected_draws_per_image=max(expected),maximum_expected_additional_repeats=max(0,max(expected)-1),
        theoretical_maximum_sample_count_one_image=n,simulated_epoch_zero_maximum_sample_count=max(observed.values()),
        simulated_epoch_zero_unique_images=len(observed),top_10_most_frequently_expected_images=top,
        simulated_epoch_zero_images=[rows[i]['image'] for i in indices],
        exposure_scope='Primary image draws only; unchanged mosaic auxiliary draws are not included. Expected object exposure counts GT occurrences, not unique objects.',
        diversity_caveat='Replacement sampling reduces per-epoch unique images versus A without replacement; probabilities bound expected concentration, not realized repeat counts. No hard repeat cap is claimed.',
        within_tier_priority='Without Helmet hard objects prioritized for review; equal image weight within a tier avoids adding another class multiplier.',
        dominance_risk=dominance(n,p,counts,fractions,unique),
        dominance_rule='FAIL if maximum expected draws >3 or a <=5% tier exceeds 20% epoch mass; WARNING if maximum expected draws >2 or expected unique <60% of N; otherwise PASS. Heuristic safety guard, not a generalization guarantee.',
        deterministic='PASS',valid_leakage=0,test_leakage=0,no_disk_image_duplication=True)
    if excluded_images is not None:
        result.update(eligible_hard_images=[r['image'] for r in rows if r['tier']!=TIERS[0] and r['image'] not in excluded],
            excluded_from_upweighting=sorted(excluded),
            image_sampling_weights={r['image']:w for r,w in zip(rows,raw)},
            image_sampling_probabilities={r['image']:v for r,v in zip(rows,p)},
            weight_rule='w_t=1+t*min(0.10*N/sum(eligible_n_t*t),1/3); excluded images have weight 1; N includes all TRAIN images',
            within_tier_priority='Frozen tier weights apply except human ANNOTATION_SUSPICIOUS / NOT_ACTUALLY_HARD images, which have weight 1. Unreviewed images retain frozen automatic tiers; the targeted review queue is not the entire hard-image inventory.',
            exclusion_policy='No image or label is fixed, removed, or duplicated. Exclusions retain baseline unnormalized weight 1 (normalized expected draws may be below 1).')
    return result


def review_policy(root, rows):
    store=ReviewStore(root);status=summarize(store)
    if not status['ready']:
        raise ValueError('Human review gate failed: Pending and REVIEW_MORE must both be zero')
    inventory={r['image'] for r in rows}
    if len({r['image'] for r in store.rows})!=len(store.rows) or any(r['image'] not in inventory for r in store.rows):
        raise ValueError('Human review does not match TRAIN inventory')
    suspicious=sorted(r['image'] for r in store.rows if r['human_decision']=='ANNOTATION_SUSPICIOUS')
    not_hard=sorted(r['image'] for r in store.rows if r['human_decision']=='NOT_ACTUALLY_HARD')
    info=dict(plan_stage='post_human_review',human_review_total=status['total'],
        human_review_reviewed=status['reviewed'],human_review_pending=status['pending'],
        **{d.lower():n for d,n in status['counts'].items()},
        suspicious_excluded_from_upweighting=len(suspicious),suspicious_excluded_images=suspicious,
        not_actually_hard_excluded_images=not_hard,human_review_sha256=sha256_file(store.path),
        human_review_gate='PASS')
    return set(suspicious+not_hard),info


def archive_sources(root):
    return {root/OUT/'sampling_plan.json':root/OUT/'sampling_plan.pre_human_review.json',
            root/PLAN:(root/PLAN).with_name('C_hard_example_sampling_plan.pre_human_review.yaml'),
            root/OUT/'PHASE_4I_REPORT.md':root/OUT/'PHASE_4I_REPORT.pre_human_review.md'}


def validate_pre_review(root):
    sources=archive_sources(root)
    if any(not p.is_file() for p in sources):raise ValueError('Complete pre-review JSON/YAML/report required')
    old=json.loads((root/OUT/'sampling_plan.json').read_text(encoding='utf-8'))
    old_yaml=yaml.safe_load((root/PLAN).read_text(encoding='utf-8'))
    if old.get('plan_stage')=='post_human_review':raise ValueError('One post-human-review replacement already consumed')
    if old.get('plan_stage') not in (None,'pre_human_review') or old.get('experiment_name')!='C_hard_example_sampling':
        raise ValueError('Ambiguous pre-review plan')
    if old_yaml!={k:v for k,v in old.items() if k!='simulated_epoch_zero_images'}:
        raise ValueError('Pre-review YAML/JSON mismatch')
    for source,archive in sources.items():
        if archive.exists() and archive.read_bytes()!=source.read_bytes():
            raise ValueError('Ambiguous existing pre-review archive: '+str(archive))
    return sources


def prepare(root=PROJECT_ROOT,replan_after_review=False):
    root=Path(root)
    if not replan_after_review and ((root/PLAN).exists() or (root/OUT/'sampling_plan.json').exists()):
        raise ValueError('Preserve existing C plan; do not create tuned variants')
    with exclusive_lock(root/OUT/'c_plan.lock'):
        if replan_after_review:
            # Serialize review writes too: provenance and decisions are one snapshot.
            with exclusive_lock(root/OUT/'manual_review_queue.lock'):
                return _prepare(root,True)
        return _prepare(root,False)


def _prepare(root,replan_after_review):
    root=Path(root);out=root/OUT
    target=root/PLAN
    if not replan_after_review and (target.exists() or (out/'sampling_plan.json').exists()):raise ValueError('Preserve existing C plan; do not create tuned variants')
    if replan_after_review:
        if not summarize(ReviewStore(root))['ready']:
            raise ValueError('Human review gate failed: Pending and REVIEW_MORE must both be zero')
        sources=validate_pre_review(root)
    summary=check(root)
    cases=json.loads((out/'train_predictions.json').read_text())['cases']
    _,rows,_=analyze(cases)
    for r in rows:train_path(root,r['image'])
    if replan_after_review:
        if len(rows)!=1158:raise ValueError('Expected 1158 TRAIN images')
        old=json.loads((out/'sampling_plan.json').read_text(encoding='utf-8'))
        if any(old.get(k)!=v for k,v in build(rows).items()):
            raise ValueError('Pre-review plan does not reproduce from frozen TRAIN evidence')
        excluded,review_info=review_policy(root,rows)
        plan=build(rows,excluded_images=excluded)
        plan.update(review_info)
        plan.update(pre_review_plan_archive=(out/'sampling_plan.pre_human_review.json').relative_to(root).as_posix(),
                    pre_review_archive_sha256={a.relative_to(root).as_posix():sha256_file(s) for s,a in sources.items()})
    else:
        plan=build(rows)
    for name in plan['simulated_epoch_zero_images']:train_path(root,name)
    baseline_dir=root/'results/experiments/A_baseline'
    meta=json.loads((baseline_dir/'experiment.json').read_text())
    args=meta['resolved_training_arguments']
    constants=dict(model='YOLO11n',imgsz=416,batch=4,workers=0,seed=42,epochs=100,patience=20,pretrained=True)
    if any(args[k]!=v for k,v in constants.items() if k!='model') or Path(args['model']).name!='yolo11n.pt':
        raise ValueError('Baseline constants mismatch')
    provenance=json.loads((out/'provenance.json').read_text())
    plan.update(experiment_name='C_hard_example_sampling',baseline_reference='A_baseline',mining_model='A_baseline',
        baseline_constants=constants,baseline_constants_preserved=True,
        baseline_resolved_training_arguments=args,baseline_experiment_sha256=sha256_file(baseline_dir/'experiment.json'),
        baseline_actual_optimizer=meta['optimizer_class'],baseline_optimizer_initial_parameter_groups=meta['optimizer_initial_parameter_groups'],
        optimizer_preservation='Preserve actual A AdamW behavior and initial parameter groups, not just optimizer=auto text; future integration must verify resolved arguments before any training.',
        pretrained_preservation='Start C from the same original yolo11n.pt pretrained initialization as A; mining best.pt is NOT the C initialization.',
        tier_definitions={TIERS[0]:'No hard GT',TIERS[1]:'Low confidence / localization / duplicate context',
                          TIERS[2]:'Class confusion / match competition',TIERS[3]:'No meaningful correct-class proposal'},
        primary_hypothesis='A small set of persistently hard TRAIN images may be under-exposed relative to easy images. Increasing their runtime exposure may improve robustness more efficiently than weighting every Without Helmet image. Hypothesis, not proven.',
        primary_change='TRAIN sampling exposure based solely on frozen-A TRAIN difficulty',
        primary_metric=dict(name='Without Helmet Recall',baseline=.530435),
        secondary_metrics={'Without Helmet Precision':.734479,'Without Helmet mAP50':.641110,
                           'Overall Precision':.779936,'Overall mAP50':.754705},
        success_safeguards='Use the same A framework validation procedure first; require target recall improvement and report all precision/AP deltas. Conservative descriptive non-degradation checks (delta >=0) for listed safeguards define a candidate, not significance. Any decrease prevents automatic usefulness/promotion and requires explicit trade-off review; no post-hoc tolerance tuning. Also require improved recovery of prior VALID hard cases at unchanged diagnostic settings without hiding regressions. Do not train from VALID identities.',
        evaluation_policy='No confidence tuning alongside C. Same framework evaluation as A, same VALID split. TEST locked. At most one primary C configuration, except technical invalidation.',
        test_status='LOCKED',validation_overuse_warning=True,
        mining_prediction_sha256=provenance['prediction_sha256'],mining_image_csv_sha256=sha256_file(out/'train_image_difficulty.csv'),
        dataset_fingerprint=provenance['dataset_fingerprint'],raw_fingerprint=provenance['raw_fingerprint'],
        manual_review_status='COMPLETE' if replan_after_review else ('PENDING' if summary['manual_review_required'] else 'NOT_REQUIRED'),
        ready_for_experiment_c_implementation=(replan_after_review or not summary['manual_review_required']) and plan['dominance_risk']=='PASS',
        ready_for_experiment_c_training=False,
        implementation_gate='Complete targeted semantic review and resolve invalid annotations/mining bugs without modifying the frozen dataset; then verify sampler/trainer integration in the next phase. No trainer attached in Phase 4I.')
    if replan_after_review:
        plan['implementation_gate']='Human review complete; suspicious and NOT_ACTUALLY_HARD cases excluded from hard upweighting without changing annotations. Verify sampler/trainer integration in the next phase; no training authorized.'
    report=mining_report(summary)+'\n## Experiment C plan only\n\n'
    report+='**C_hard_example_sampling**: '+plan['primary_hypothesis']+'\n\n'
    report+='Only change: '+plan['primary_change']+'. '+plan['pretrained_preservation']+'\n\n'
    report+='### Proposed weights and exposure\n\n'+plan['weight_rule']+'\n\n'+plan['weight_rule_rationale']+'\n\n'
    fields=['tier_counts','proposed_tier_weights','effective_epoch_samples','expected_tier_fraction','expected_tier_draws',
            'baseline_class_exposure','expected_class_exposure','expected_unique_images_per_epoch','baseline_without_replacement_unique',
            'uniform_replacement_expected_unique','effective_sample_size','maximum_expected_draws_per_image',
            'maximum_expected_additional_repeats','simulated_epoch_zero_maximum_sample_count','simulated_epoch_zero_unique_images',
            'theoretical_maximum_sample_count_one_image','top_10_most_frequently_expected_images','dominance_risk']
    report+='```json\n'+json.dumps({k:plan[k] for k in fields},indent=2)+'\n```\n\n'
    report+=plan['dominance_rule']+'\n\n'+plan['diversity_caveat']+'\n\n'+plan['exposure_scope']+'\n\n'+plan['within_tier_priority']+'\n\n'
    report+='### Controlled design and safeguards\n\n```json\n'+json.dumps(constants,indent=2)+'\n```\n\n'
    report+=plan['optimizer_preservation']+' Full resolved A arguments and actual optimizer groups are in the plan; all augmentation, pretrained behavior and VALID split are inherited unchanged.\n\n'
    report+='Primary target: Without Helmet Recall >0.530435. Safeguards: Without Helmet Precision 0.734479, Without Helmet mAP50 0.641110, Overall Precision 0.779936, Overall mAP50 0.754705.\n\n'
    report+=plan['success_safeguards']+'\n\n'+plan['evaluation_policy']+'\n\n'
    report+='[Sampling diagnostics](sampling_plan.json). Plan config: `configs/experiments/C_hard_example_sampling_plan.yaml`.\n\n'
    report+=f"READY FOR EXPERIMENT C IMPLEMENTATION: {'YES' if plan['ready_for_experiment_c_implementation'] else 'NO'}. Manual review: {plan['manual_review_status']}. READY FOR EXPERIMENT C TRAINING: NO.\n\n"
    report+=plan['implementation_gate']+' TEST STATUS: LOCKED. Stop after Phase 4I.\n'
    if replan_after_review:
        report+='\n## Post-human-review provenance\n\nAutomatic mining status is historical; the human review gate is PASS.\n\n'
        report+='```json\n'+json.dumps({**review_info,'pre_review_plan_archive':plan['pre_review_plan_archive'],
            'pre_review_archive_sha256':plan['pre_review_archive_sha256'],
            'eligible_hard_image_count':len(plan['eligible_hard_images'])},indent=2)+'\n```\n'
        report+=plan['exclusion_policy']+'\n'
        # Validate all archives before any mutation, then publish complete copies.
        validate_pre_review(root)
        for source,archive in sources.items():
            if not archive.exists():atomic_bytes(archive,source.read_bytes(),create_only=True)
    payloads={out/'sampling_plan.json':(json.dumps(plan,indent=2)+'\n').encode(),
              target:yaml.safe_dump({k:v for k,v in plan.items() if k!='simulated_epoch_zero_images'},sort_keys=False).encode(),
              out/'PHASE_4I_REPORT.md':report.encode()}
    originals={path:path.read_bytes() if path.exists() else None for path in payloads}
    try:
        for path,data in payloads.items():atomic_bytes(path,data)
    except BaseException:
        for path,data in originals.items():
            if data is not None:atomic_bytes(path,data)
            elif path.exists():path.unlink()
        raise
    return plan


def verify(root=PROJECT_ROOT):
    root=Path(root);out=root/OUT;summary=check(root)
    plan=json.loads((out/'sampling_plan.json').read_text())
    if plan.get('experiment_name')!='C_hard_example_sampling':raise ValueError('Canonical C experiment name required')
    yaml_plan=yaml.safe_load((root/PLAN).read_text())
    if yaml_plan!={k:v for k,v in plan.items() if k!='simulated_epoch_zero_images'}: raise ValueError('YAML/JSON plan mismatch')
    cases=json.loads((out/'train_predictions.json').read_text())['cases'];_,rows,_=analyze(cases)
    if plan.get('plan_stage')=='post_human_review':
        excluded,info=review_policy(root,rows)
        if any(plan.get(k)!=v for k,v in info.items()):raise ValueError('Human review provenance drift')
        computed=build(rows,excluded_images=excluded)
        archives=archive_sources(root)
        expected_paths={a.relative_to(root).as_posix() for a in archives.values()}
        if set(plan['pre_review_archive_sha256'])!=expected_paths:raise ValueError('Archive manifest mismatch')
        for name,digest in plan['pre_review_archive_sha256'].items():
            if sha256_file(root/name)!=digest:raise ValueError('Pre-review archive drift')
        archived=json.loads((out/'sampling_plan.pre_human_review.json').read_text(encoding='utf-8'))
        archived_yaml=yaml.safe_load((root/PLAN).with_name('C_hard_example_sampling_plan.pre_human_review.yaml').read_text(encoding='utf-8'))
        if archived_yaml!={k:v for k,v in archived.items() if k!='simulated_epoch_zero_images'}:
            raise ValueError('Archived YAML/JSON mismatch')
        if any(archived.get(k)!=v for k,v in build(rows).items()):raise ValueError('Archived pre-review plan does not reproduce')
        if plan['pre_review_plan_archive']!=(OUT/'sampling_plan.pre_human_review.json').as_posix():raise ValueError('Archive path mismatch')
        if plan['manual_review_status']!='COMPLETE' or plan['ready_for_experiment_c_implementation']!=(computed['dominance_risk']=='PASS'):
            raise ValueError('Human review readiness mismatch')
    else:
        computed=build(rows)
    if any(plan[k]!=v for k,v in computed.items()):raise ValueError('Sampling plan does not reproduce')
    for name in [r['image'] for r in rows]+plan['simulated_epoch_zero_images']:train_path(root,name)
    if len(rows)!=1158 or len(plan['simulated_epoch_zero_images'])!=1158:raise ValueError('Epoch length mismatch')
    if sha256_file(out/'train_image_difficulty.csv')!=plan['mining_image_csv_sha256']:raise ValueError('Difficulty evidence drift')
    if sha256_file(root/'results/experiments/A_baseline/experiment.json')!=plan['baseline_experiment_sha256']:raise ValueError('Baseline argument drift')
    if plan['training_authorized'] or plan['ready_for_experiment_c_training']:raise ValueError('PLAN ONLY; no training')
    print('='*40+'\nEXPERIMENT C HARD SAMPLING PLAN\n'+'='*40)
    print('TRAIN images: 1158\nDifficulty source: TRAIN ONLY');print('Tier sizes:',plan['tier_counts'])
    print('Without Helmet hard images:',summary['without_helmet_hard_images'])
    print('Critical no-proposal images:',summary['critical_no_proposal_images'])
    print('Sampling weights:',plan['proposed_tier_weights']);print('Effective epoch samples: 1158')
    print('Expected tier exposure:',plan['expected_tier_draws']);print('Expected class exposure:',plan['expected_class_exposure'])
    print('Expected unique images:',plan['expected_unique_images_per_epoch'])
    print('Maximum expected draws:',plan['maximum_expected_draws_per_image'])
    print('Maximum sample count, simulated epoch:',plan['simulated_epoch_zero_maximum_sample_count'])
    print('VALID leakage: 0\nTEST leakage: 0\nDeterministic: PASS\nDominance risk:',plan['dominance_risk'])
    if plan.get('plan_stage')=='post_human_review':
        print('Plan stage: post_human_review\nHUMAN REVIEW: PASS')
        print('Reviewed:',plan['human_review_reviewed'],'Pending:',plan['human_review_pending'],'REVIEW_MORE:',plan['review_more'])
        print('ANNOTATION_SUSPICIOUS:',plan['annotation_suspicious'])
        print('Suspicious excluded from hard upweighting:',plan['suspicious_excluded_from_upweighting'])
        print('Eligible hard images:',len(plan['eligible_hard_images']))
        print('Pre-review plan preserved:',plan['pre_review_plan_archive'])
    print('READY FOR EXPERIMENT C IMPLEMENTATION:','YES' if plan['ready_for_experiment_c_implementation'] else 'NO')
    print('READY FOR EXPERIMENT C TRAINING: NO\nTEST STATUS: LOCKED\nNo training.')
    return plan


def main(argv=None,root=PROJECT_ROOT):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replan-after-review',action='store_true',
        help='Replace the pre-review plan once after completed human review; archive originals first.')
    args=parser.parse_args(argv)
    try:
        plan=prepare(root,replan_after_review=args.replan_after_review)
    except (OSError,ValueError,KeyError) as error:
        print('C PLAN: FAIL\n'+str(error));return 1
    print('C PLAN CREATED. Weights:',plan['proposed_tier_weights'])
    print('NO TRAINING. Review status:',plan['manual_review_status'])
    return 0
