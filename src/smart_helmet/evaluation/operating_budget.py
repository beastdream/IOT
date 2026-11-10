"""Phase 4H: frozen exploratory policies applied to saved VALID detections."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import yaml

from smart_helmet.paths import PROJECT_ROOT
from smart_helmet.dataset.audit import write_csv
from smart_helmet.dataset.duplicates import sha256_file
from smart_helmet.training.experiment import write_json
from .comparison import fingerprint
from .evaluator import validation_guard
from . import operating_point as op
from .failure_analysis import iou

MODES = ('STRICT', 'BALANCED', 'RECALL_ORIENTED')
CONSTRAINTS = ('minimum_without_helmet_precision', 'maximum_without_helmet_fp_per_image',
               'maximum_overall_fp_per_image')
TIEBREAK = ['higher_precision', 'lower_overall_fp', 'lower_without_helmet_fp', 'higher_threshold']


def parse_policy(path):
    policy = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(policy, dict):
        raise ValueError('Policy must be a mapping')
    validation_guard(policy.get('split'))
    required = dict(schema_version=1, label='EXPLORATORY VALIDATION BUDGETS',
                    test_status='LOCKED / NOT USED', primary_class='Without Helmet',
                    primary_objective='maximize_recall', production_policy=False,
                    selection_tiebreak=TIEBREAK, same_cost_max_fp_count_difference=1)
    for key, value in required.items():
        if policy.get(key) != value:
            raise ValueError('Invalid policy field: '+key)
    if set(policy.get('modes', {})) != set(MODES):
        raise ValueError('All three policy modes required')
    for mode in MODES:
        scenario = policy['modes'][mode]
        if not scenario.get('rationale') or set(scenario.get('constraints', {})) != set(CONSTRAINTS):
            raise ValueError('Explicit rationale and all constraints required')
        for key, v in scenario['constraints'].items():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError('Finite nonnegative numeric constraints required')
            if key == CONSTRAINTS[0] and v > 1:
                raise ValueError('Precision must be in [0,1]')
    return policy


def join_metrics(rows, thresholds, images=127):
    grouped = {}
    for row in rows:
        tag, threshold, scope = row['model'], row['threshold'], row['scope']
        if tag not in ('A', 'B') or threshold not in thresholds or scope not in op.SCOPES:
            raise ValueError('Foreign model, threshold or scope')
        key = (tag, threshold)
        group = grouped.setdefault(key, dict(model=tag, threshold=threshold))
        if scope in group:
            raise ValueError('Duplicate metric scope')
        if any(type(row[k]) is not int or row[k] < 0 for k in ('TP', 'FP', 'FN')):
            raise ValueError('Invalid counts')
        calculated = op.metrics(row['TP'], row['FP'], row['FN'], images)
        if any(not math.isfinite(row[k]) or abs(row[k]-calculated[k]) > 1e-12 for k in calculated):
            raise ValueError('Saved metric arithmetic mismatch')
        group[scope] = row
    if set(grouped) != {(tag,t) for tag in ('A', 'B') for t in thresholds}:
        raise ValueError('Incomplete paired threshold coverage')
    for group in grouped.values():
        if any(s not in group for s in op.SCOPES):
            raise ValueError('Missing class scope')
        if any(group['Overall'][k] != sum(group[s][k] for s in op.SCOPES[1:]) for k in ('TP','FP','FN')):
            raise ValueError('Overall counts do not equal class sums')
    return sorted(grouped.values(), key=lambda p: (p['model'], p['threshold']))


def feasible(point, constraints):
    wh, overall = point['Without Helmet'], point['Overall']
    return (wh['TP']+wh['FP'] > 0 and wh['precision'] >= constraints[CONSTRAINTS[0]] and
            wh['fp_per_image'] <= constraints[CONSTRAINTS[1]] and
            overall['fp_per_image'] <= constraints[CONSTRAINTS[2]])


def select(points, constraints):
    eligible = [p for p in points if feasible(p, constraints)]
    return max(eligible, key=lambda p: (p['Without Helmet']['recall'], p['Without Helmet']['precision'],
               -p['Overall']['FP'], -p['Without Helmet']['FP'], p['threshold']), default=None)


def same_cost(a, b, tolerance=1):
    return all(abs(a[s]['FP']-b[s]['FP']) <= tolerance for s in ('Without Helmet','Overall'))


def compare(a, b):
    if a is None and b is None:
        return 'NO_FEASIBLE_POINT'
    if a is None:
        return 'B_PREFERRED_UNDER_THIS_BUDGET'
    if b is None:
        return 'A_PREFERRED'
    if same_cost(a, b):
        aw, bw = a['Without Helmet'], b['Without Helmet']
        if aw['recall'] > bw['recall'] and aw['precision'] >= bw['precision']:
            return 'A_PREFERRED'
        if bw['recall'] > aw['recall'] and bw['precision'] >= aw['precision']:
            return 'B_PREFERRED_UNDER_THIS_BUDGET'
    return 'NO_CLEAR_ADVANTAGE'


def same_cost_comparison(a, b):
    # Deduplicate detection-count states to avoid weighting long confidence plateaus.
    def unique(points):
        result = {}
        for p in sorted(points, key=lambda p:p['threshold']):
            key = tuple(p[s][k] for s in op.SCOPES for k in ('TP','FP','FN'))
            result[key] = p
        return list(result.values())
    pairs = [(aa,bb) for aa in unique(a) for bb in unique(b) if same_cost(aa,bb)]
    counts = {key:0 for key in ('A_PREFERRED','B_PREFERRED_UNDER_THIS_BUDGET','NO_CLEAR_ADVANTAGE')}
    for aa, bb in pairs:
        counts[compare(aa,bb)] += 1
    pair = max(pairs, key=lambda ab: (min(p['Without Helmet']['recall'] for p in ab),
        max(p['Without Helmet']['recall'] for p in ab), -sum(p['Overall']['FP'] for p in ab),
        -sum(p['Without Helmet']['FP'] for p in ab), ab[0]['threshold'], ab[1]['threshold']), default=None)
    return dict(status='MATCHED_PAIR' if pair else 'NO_COMPARABLE_PAIR', pair_counts=counts,
                A=pair[0] if pair else None, B=pair[1] if pair else None,
                decision=compare(*pair) if pair else 'NO_CLEAR_ADVANTAGE')


def pareto(points):
    def vector(p):
        return (p['Without Helmet']['recall'], p['Without Helmet']['precision'], -p['Overall']['fp_per_image'])
    unique, multiplicity = {}, {}
    for p in sorted(points, key=lambda p:p['threshold']):
        v = vector(p); unique[v] = p; multiplicity[v] = multiplicity.get(v,0)+1
    return [dict(p, equivalent_threshold_count=multiplicity[v]) for v,p in sorted(unique.items())
            if not any(all(a >= b for a,b in zip(other,v)) and any(a>b for a,b in zip(other,v))
                       for other in unique if other != v)]


def diagnostics(cases, threshold, misses):
    counts = {cid:[0,0,0] for cid in (0,1)}
    result = dict(historical_recovered=0, historical_wrong_class=0, historical_still_missed=0,
                  class_confusion_count=0, without_helmet_class_confusion_count=0)
    for case in cases:
        numbers, states = op.outcomes(case, threshold)
        for cid in (0,1):
            counts[cid] = [a+b for a,b in zip(counts[cid], numbers[cid])]
        for index, state in enumerate(states):
            if state == 'CLASS_CONFUSION':
                result['class_confusion_count'] += 1
                result['without_helmet_class_confusion_count'] += int(case['ground_truth'][index]['class_id'] == 1)
            if (case['image'],index) in misses:
                key = {'CORRECT':'historical_recovered','CLASS_CONFUSION':'historical_wrong_class',
                       'MISSED':'historical_still_missed'}[state]
                result[key] += 1
    return result, counts


def flatten(point, policy, status='FEASIBLE'):
    row = dict(policy=policy, model=point['model'], status=status, threshold=point.get('threshold'))
    for scope, prefix in [('Without Helmet','wh'),('With Helmet','with'),('Overall','overall')]:
        for key in ('TP','FP','FN','precision','recall','F1','fp_per_image'):
            row[prefix+'_'+key] = point.get(scope,{}).get(key)
    for key in ('historical_recovered','historical_wrong_class','historical_still_missed',
                'class_confusion_count','without_helmet_class_confusion_count','equivalent_threshold_count'):
        row[key] = point.get(key)
    return row


def run(root=PROJECT_ROOT, split='val'):
    validation_guard(split)
    root = Path(root).resolve()
    policy_path = root/'configs/evaluation/operating_policy.yaml'
    policy = parse_policy(policy_path)
    policy_hash = sha256_file(policy_path)
    out = root/'results/evaluation/operating_budget_analysis'
    if out.resolve() != out or out.exists() and any(out.iterdir()):
        raise ValueError('Output must be new, empty and not redirected')
    out.mkdir(parents=True, exist_ok=True)
    (out/'operating_policy.yaml').write_bytes(policy_path.read_bytes())
    audit = dict(status='RUNNING', phase='4H', label='VALIDATION / EXPLORATORY',
                 policy_frozen_before_evaluation_utc=datetime.now(timezone.utc).isoformat(),
                 policy_sha256=policy_hash, test_status='LOCKED / NOT USED', test_used=False,
                 inference_executed=False, training_executed=False, production_changed=False)
    write_json(out/'reproducibility.json', audit)
    protected = [root/'results/evaluation/operating_point_analysis', root/'results/evaluation/A_vs_B',
                 root/'results/evaluation/A_baseline_val',root/'data/curated/v1/valid',root/'configs',
                 *[root/'results/experiments'/v for v in op.RUNS.values()]]
    before = {p.relative_to(root).as_posix():fingerprint(p) for p in protected}
    try:
        _, source, _, cases, misses = op.preflight(root, split)
        sweep_dir = root/'results/evaluation/operating_point_analysis'
        load = lambda p:json.loads(p.read_text(encoding='utf-8'))
        status = load(sweep_dir/'analysis_status.json')
        provenance = load(sweep_dir/'provenance.json')
        if status['status'] != 'PASS' or status['test_status'] != 'LOCKED / NOT USED' or status['validation_images'] != 127:
            raise ValueError('Passing TEST-locked 127-image validation sweep required')
        # The new authorized policy changes configs; all original data/model inputs must match.
        for name, expected in provenance['source_artifact_hashes'].items():
            if name == 'configs':
                continue
            if fingerprint(root/name) != expected:
                raise ValueError('Phase 4G input provenance drift: '+name)
        thresholds = provenance['thresholds']
        if len(thresholds) != 1005 or thresholds != list(op.refined_thresholds(cases)):
            raise ValueError('Exactly the complete 1005-point frozen sweep required')
        points = join_metrics(load(sweep_dir/'threshold_metrics.json'), thresholds)
        for p in points:
            for s,total in [('Overall',300),('With Helmet',185),('Without Helmet',115)]:
                if p[s]['TP']+p[s]['FN'] != total:
                    raise ValueError('Validation GT denominator mismatch')
        rows, results, frontiers, comparisons = [], {}, [], {}
        for mode in MODES:
            constraints = policy['modes'][mode]['constraints']
            chosen, eligible = {}, {}
            for tag in ('A','B'):
                pool = [p for p in points if p['model']==tag]
                eligible[tag] = [p for p in pool if feasible(p,constraints)]
                selected = select(pool,constraints)
                if selected:
                    diag, counts = diagnostics(cases[tag], selected['threshold'], misses)
                    for cid, scope in [(0,'With Helmet'),(1,'Without Helmet')]:
                        if counts[cid] != [selected[scope][k] for k in ('TP','FP','FN')]:
                            raise ValueError('Selected saved metrics differ from frozen predictions')
                    selected = dict(selected, **diag)
                chosen[tag] = selected
                rows.append(flatten(selected or dict(model=tag),mode,
                                    'FEASIBLE' if selected else 'NO_FEASIBLE_OPERATING_POINT'))
                frontiers.extend(flatten(p,mode) for p in pareto(eligible[tag]))
            results[mode] = dict(**chosen, decision=compare(chosen['A'],chosen['B']),
                                 model_status={tag:'FEASIBLE' if p else 'NO_FEASIBLE_OPERATING_POINT'
                                               for tag,p in chosen.items()},
                                 feasible_counts={tag:len(v) for tag,v in eligible.items()},
                                 optima_same_cost=bool(chosen['A'] and chosen['B'] and same_cost(chosen['A'],chosen['B'])))
            comparisons[mode] = same_cost_comparison(eligible['A'],eligible['B'])
        for tag in ('A','B'):
            frontiers.extend(flatten(p,'ALL_TESTED') for p in pareto([p for p in points if p['model']==tag]))
        # Geometric failure evidence uses all existing proposals down to 0.001;
        # it neither runs inference nor tunes an additional threshold.
        residuals = []
        for tag in ('A','B'):
            for case in cases[tag]:
                for index,gt in enumerate(case['ground_truth']):
                    if gt['class_id'] != 1:
                        continue
                    same = [p for p in case['predictions'] if p['class_id']==1 and iou(gt['bbox'],p['bbox']) >= .5]
                    wrong = [p for p in case['predictions'] if p['class_id']==0 and iou(gt['bbox'],p['bbox']) >= .5]
                    if not same:
                        residuals.append(dict(model=tag,image=case['image'],ground_truth_index=index,
                                              no_correct_class_proposal=True, wrong_class_overlap=bool(wrong),
                                              max_wrong_class_confidence=max((p['confidence'] for p in wrong),default=None)))
        write_json(out/'remaining_failure_evidence.json',residuals)
        write_json(out/'policy_results.json',results)
        write_csv(out/'policy_results.csv',rows,list(rows[0]))
        write_csv(out/'pareto_operating_points.csv',frontiers,list(frontiers[0]))
        write_json(out/'same_cost_comparisons.json',comparisons)
        from .operating_budget_report import write_report
        summary = write_report(out,policy,results,rows,frontiers,comparisons,residuals,
                               load(sweep_dir/'comparisons.json'),load(source/'comparison_metrics.json'))
        audit.update(status='PASS',validation_images=127,thresholds=1005,policy_results=6,
                     protected_input_hashes=before,checkpoint_sha256=provenance['checkpoint_sha256'],
                     source_code_sha256={p:sha256_file(root/p) for p in (
                         'src/smart_helmet/evaluation/operating_budget.py',
                         'src/smart_helmet/evaluation/operating_budget_report.py',
                         'scripts/analyze_operating_budgets.py','tests/test_operating_budget.py')},**summary)
    except BaseException as error:
        audit.update(status='FAIL',error=str(error)); raise
    finally:
        try:
            if sha256_file(policy_path) != policy_hash or sha256_file(out/'operating_policy.yaml') != policy_hash:
                raise ValueError('Predefined policy changed')
            if any(fingerprint(p) != before[p.relative_to(root).as_posix()] for p in protected):
                raise ValueError('Protected input changed')
            audit['protected_input_immutability'] = 'PASS'
        except BaseException as error:
            audit.update(status='FAIL',error=str(error)); raise
        finally:
            write_json(out/'reproducibility.json',audit)
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',default='val')
    args = parser.parse_args(argv)
    audit = run(split=args.split)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('protected_input_hashes','source_code_sha256')},indent=2))
    return 0
