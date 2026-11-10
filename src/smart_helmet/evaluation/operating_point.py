"""Phase 4G: confidence sweep over frozen, provenance-checked VALID predictions."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from smart_helmet.paths import PROJECT_ROOT
from smart_helmet.foundation import EXPECTED_CLASSES, IMAGE_EXTENSIONS
from smart_helmet.dataset.duplicates import sha256_file
from smart_helmet.dataset.audit import write_csv
from smart_helmet.training.experiment import write_json
from .comparison import RUNS, fingerprint, checkpoint_fingerprint, ensure_same_validation
from .evaluator import validation_guard
from .failure_review import guarded_image
from .failure_analysis import match_boxes, ground_truth

THRESHOLDS = tuple(i / 100 for i in range(5, 91, 5))
PRECISION_FLOOR = .734479
SCOPES = ('Overall', 'With Helmet', 'Without Helmet')


def refined_thresholds(cases):
    """All attainable cached operating states inside [0.05, 0.90], plus grids.

    With >= filtering, every distinct confidence is the inclusive right endpoint
    of an interval with unchanged predictions. Union across models keeps all
    thresholds common. The endpoints cover the truncated range boundaries.
    """
    return tuple(sorted({i/100 for i in range(5, 91)} | {
        p['confidence'] for cs in cases.values() for c in cs for p in c['predictions']
        if .05 <= p['confidence'] <= .9}))


def metrics(tp, fp, fn, images):
    if min(tp, fp, fn) < 0 or images <= 0:
        raise ValueError('Nonnegative counts and a positive image count required')
    return dict(TP=tp, FP=fp, FN=fn,
                precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0.,
                F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                fp_per_image=fp/images)


def outcomes(case, threshold):
    active = [p for p in case['predictions'] if p['confidence'] >= threshold]
    match = match_boxes(case['ground_truth'], active)
    confusion = {i for i, _, _ in match['confusions']}
    states = ['CORRECT' if i in match['matches'] else
              'CLASS_CONFUSION' if i in confusion else 'MISSED'
              for i in range(len(case['ground_truth']))]
    counts = {}
    for cid in EXPECTED_CLASSES:
        tp = sum(g['class_id'] == cid and states[i] == 'CORRECT'
                 for i, g in enumerate(case['ground_truth']))
        fn = sum(g['class_id'] == cid for g in case['ground_truth']) - tp
        fp = sum(active[j]['class_id'] == cid for j in match['extra'])
        counts[cid] = (tp, fp, fn)
    return counts, states


def sweep(cases, historical, thresholds=THRESHOLDS):
    thresholds = tuple(thresholds)
    if not thresholds or len(set(thresholds)) != len(thresholds) or any(
            not math.isfinite(t) or not .001 <= t <= 1 for t in thresholds):
        raise ValueError('Unique finite thresholds within cached confidence range required')
    rows, recovery, objects = [], [], []
    for tag in ('A', 'B'):
        if not cases[tag]:
            raise ValueError('Nonempty VALID cases required')
        identities = {(c['image'], i) for c in cases[tag]
                      for i, gt in enumerate(c['ground_truth']) if gt['class_id'] == 1}
        if not historical <= identities:
            raise ValueError('Historical identities must be Without Helmet VALID objects')
        for threshold in sorted(thresholds):
            totals = {cid: [0, 0, 0] for cid in EXPECTED_CLASSES}
            recovered = Counter()
            for case in cases[tag]:
                counts, states = outcomes(case, threshold)
                for cid, numbers in counts.items():
                    totals[cid] = [a+b for a, b in zip(totals[cid], numbers)]
                for i, state in enumerate(states):
                    if (case['image'], i) in historical:
                        recovered[state] += 1
                        objects.append(dict(model=tag, threshold=threshold, image=case['image'],
                                            ground_truth_index=i, outcome=state))
            for scope, ids in [('Overall', [0, 1]), ('With Helmet', [0]), ('Without Helmet', [1])]:
                nums = [sum(totals[cid][j] for cid in ids) for j in range(3)]
                rows.append(dict(model=tag, threshold=threshold, scope=scope,
                                 **metrics(*nums, len(cases[tag]))))
            if sum(recovered.values()) != len(historical):
                raise ValueError('Historical object counted more than once or missing')
            recovery.append(dict(model=tag, threshold=threshold, recovered=recovered['CORRECT'],
                                 still_missed=recovered['MISSED'], wrong_class=recovered['CLASS_CONFUSION'],
                                 still_not_correct=len(historical)-recovered['CORRECT']))
    return rows, recovery, objects


def best_point(rows, key='F1'):
    # Tie: higher recall, then precision, then higher confidence (deterministic).
    return max(rows, key=lambda r: (r[key], r['recall'], r['precision'], r['threshold']), default=None)


def precision_floor(rows, floor=PRECISION_FLOOR):
    if not 0 <= floor <= 1:
        raise ValueError('Precision floor must be in [0,1]')
    return {tag: best_point([r for r in rows if r['model'] == tag and
                            r['scope'] == 'Without Helmet' and r['TP']+r['FP'] > 0 and
                            r['precision'] >= floor], 'recall') for tag in ('A', 'B')}


def budget_comparison(rows, scope):
    groups = {tag: [r for r in rows if r['scope'] == scope and r['model'] == tag] for tag in ('A', 'B')}
    # Same *upper bound*, not necessarily the same realized FP count. Save slack explicitly.
    budgets = sorted({r['FP'] for r in groups['A']+groups['B']})
    frontier = []
    for budget in budgets:
        points = {tag: best_point([r for r in rr if r['FP'] <= budget], 'recall')
                  for tag, rr in groups.items()}
        frontier.append(dict(fp_budget=budget, **points,
                             slack={tag: budget-r['FP'] if r else None for tag, r in points.items()}))
    closest = []
    for a in groups['A']:
        b = min(groups['B'], key=lambda r: (abs(r['FP']-a['FP']), -r['recall'], -r['threshold']))
        closest.append(dict(A=a, B=b, fp_gap=b['FP']-a['FP'],
                            fp_per_image_gap=b['fp_per_image']-a['fp_per_image']))
    return dict(scope=scope, frontier=frontier, closest_pairs=closest)


def verify_checkpoint(path, expected):
    actual = checkpoint_fingerprint(path)
    if actual != expected:
        raise ValueError('Checkpoint provenance mismatch')
    return actual


def source_guard(provenance, status):
    if status.get('status') != 'PASS' or status.get('test_status') != 'LOCKED / NOT USED' or \
            provenance.get('test_status') != 'LOCKED / NOT USED':
        raise ValueError('Passing comparison with TEST LOCKED / NOT USED required')
    args = provenance['common_evaluation_args']
    for k, expected in dict(imgsz=416, iou=.7, max_det=300, augment=False, quantize=32, conf=.001).items():
        if args.get(k) != expected:
            raise ValueError('Frozen inference settings mismatch: '+k)
    if not args.get('device') or provenance['validation_images'] != 127:
        raise ValueError('Shared device and exactly 127 VALID images required')


def preflight(root, split):
    validation_guard(split)  # Reject TEST before any filesystem access.
    root = Path(root).resolve()
    source = root/'results/evaluation/A_vs_B'
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    provenance = load(source/'provenance.json')
    source_guard(provenance, load(source/'comparison_status.json'))
    training = {}
    for tag, run in RUNS.items():
        directory = root/'results/experiments'/run
        verify_checkpoint(directory/'weights/best.pt', provenance['checkpoint_sha256'][tag])
        training[tag] = fingerprint(directory)
        if training[tag] != provenance['training_artifact_hashes'][tag]:
            raise ValueError('Training artifacts differ from frozen comparison provenance')
    manifests = provenance['validation_manifest']
    expected = [r['image'] for r in manifests]
    if len(expected) != 127 or len(set(expected)) != 127:
        raise ValueError('Exactly 127 unique VALID images required')
    actual = {p.resolve() for p in (root/'data/curated/v1/valid/images').iterdir()
              if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
    guarded = [guarded_image(root, name) for name in expected]
    if len(actual) != 127 or set(guarded) != actual:
        raise ValueError('VALID manifest coverage mismatch')
    for entry, path in zip(manifests, guarded):
        if sha256_file(path) != entry['image_sha256'] or sha256_file(
                path.parent.parent/'labels'/(path.stem+'.txt')) != entry['label_sha256']:
            raise ValueError('Frozen VALID image/label hash mismatch')
    cases = {}
    for tag in ('A', 'B'):
        saved = load(source/(tag+'_validation_predictions.json'))
        if saved['checkpoint_sha256'] != provenance['checkpoint_sha256'][tag]:
            raise ValueError('Cached predictions checkpoint provenance mismatch')
        cases[tag] = saved['cases']
        for case in cases[tag]:
            path = guarded_image(root, case['image'])
            gt, w, h = ground_truth(path)
            if (gt, w, h) != (case['ground_truth'], case['width'], case['height']):
                raise ValueError('Cached VALID GT does not match current labels')
            if len(case['predictions']) > 300:
                raise ValueError('Cached predictions exceed max_det')
            for p in case['predictions']:
                if p['class_id'] not in EXPECTED_CLASSES or not .001 <= p['confidence'] <= 1 or \
                        len(p['bbox']) != 4 or not all(math.isfinite(v) for v in p['bbox']):
                    raise ValueError('Invalid cached prediction')
    ensure_same_validation(cases['A'], cases['B'], expected)
    historic_path = root/'results/evaluation/A_baseline_val/validation_predictions.json'
    if sha256_file(historic_path) != provenance['historic_predictions_sha256']:
        raise ValueError('Historical prediction hash mismatch')
    historic = load(historic_path)
    if historic['checkpoint_sha256'] != provenance['checkpoint_sha256']['A'] or \
            historic['dataset_fingerprint'] != provenance['dataset_fingerprint']:
        raise ValueError('Historical provenance mismatch')
    ensure_same_validation(historic['cases'], cases['A'], expected)
    misses = {(c['image'], i) for c in historic['cases'] for i in c['missed_gt']
              if c['ground_truth'][i]['class_id'] == 1}
    reproduced = {(c['image'], i) for c in cases['A'] for i, s in enumerate(outcomes(c, .25)[1])
                  if c['ground_truth'][i]['class_id'] == 1 and s != 'CORRECT'}
    if len(misses) != 50 or misses != reproduced:
        raise ValueError('Historical 50 miss identities not reproduced')
    return root, source, provenance, cases, misses


def plots(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    specs = [('without_helmet_precision_recall', 'Without Helmet', 'recall', 'precision'),
             ('without_helmet_f1_vs_confidence', 'Without Helmet', 'threshold', 'F1'),
             ('without_helmet_recall_vs_confidence', 'Without Helmet', 'threshold', 'recall'),
             ('without_helmet_precision_vs_confidence', 'Without Helmet', 'threshold', 'precision'),
             ('without_helmet_fp_per_image_vs_confidence', 'Without Helmet', 'threshold', 'fp_per_image'),
             ('overall_f1_vs_confidence', 'Overall', 'threshold', 'F1')]
    for name, scope, x, y in specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        for tag in ('A', 'B'):
            points = [r for r in rows if r['model'] == tag and r['scope'] == scope]
            ax.plot([r[x] for r in points], [r[y] for r in points], marker='o', markersize=2, label=tag)
        ax.set(xlim=(0, 1), ylim=(0, max(1, max(r[y] for r in rows if r['scope'] == scope)*1.05)
                                if y == 'fp_per_image' else 1), xlabel=x, ylabel=y,
               title=f'{scope}: {y} vs {x} (VALID only)')
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
        fig.savefig(out/(name+'.png'), dpi=150); plt.close(fig)


def run(root=PROJECT_ROOT, split='val'):
    root, source, provenance, cases, misses = preflight(root, split)
    out = root/'results/evaluation/operating_point_analysis'
    if out.resolve() != out or out.exists() and any(out.iterdir()):
        raise ValueError('Output must be empty and not redirected; preserve prior analysis')
    # No TEST paths are opened, enumerated, or hashed by this analysis.
    protected = [source, root/'results/evaluation/A_baseline_val', root/'data/curated/v1/valid',
                 root/'configs', *[root/'results/experiments'/v for v in RUNS.values()]]
    before = {p.relative_to(root).as_posix(): fingerprint(p) for p in protected}
    out.mkdir(parents=True, exist_ok=True)
    status = dict(status='RUNNING', test_status='LOCKED / NOT USED', test_used=False,
                  training_executed=False, validation_images=127)
    try:
        thresholds = refined_thresholds(cases)
        rows, recovery, objects = sweep(cases, misses, thresholds)
        for tag, expected in [('A', (65, 34, 50)), ('B', (45, 15, 70))]:
            r = next(r for r in rows if r['model'] == tag and r['scope'] == 'Without Helmet' and r['threshold'] == .25)
            if tuple(r[k] for k in ('TP', 'FP', 'FN')) != expected:
                raise ValueError('Fixed-0.25 diagnostic reproduction failed')
        for name, data in [('threshold_metrics', rows), ('historical_miss_recovery', recovery),
                           ('historical_miss_objects', objects)]:
            write_json(out/(name+'.json'), data)
            write_csv(out/(name+'.csv'), data, list(data[0]))
        analysis = dict(precision_floor=PRECISION_FLOOR, at_precision_floor=precision_floor(rows),
                        best_f1={tag: {scope: best_point([r for r in rows if r['model'] == tag and r['scope'] == scope])
                                       for scope in SCOPES} for tag in ('A', 'B')},
                        fp_budgets={scope: budget_comparison(rows, scope) for scope in ('Without Helmet', 'Overall')})
        write_json(out/'comparisons.json', analysis)
        plots(rows, out)
        write_json(out/'provenance.json', dict(source='Frozen Phase 4F post-NMS predictions; no new inference',
            checkpoint_sha256=provenance['checkpoint_sha256'],
            common_inference_settings=dict(provenance['common_evaluation_args'], rect=True),
            inference_environment=provenance['environment'], validation_manifest=provenance['validation_manifest'],
            class_mapping=EXPECTED_CLASSES, thresholds=thresholds, required_thresholds=THRESHOLDS, matching_iou=.5,
            analysis_code_sha256={name: sha256_file(root/name) for name in (
                'scripts/analyze_operating_points.py',
                'src/smart_helmet/evaluation/operating_point.py',
                'src/smart_helmet/evaluation/operating_point_report.py',
                'tests/test_operating_point.py')},
            precision_floor=PRECISION_FLOOR, source_artifact_hashes=before,
            test_status='LOCKED / NOT USED',
            cache_note='Prediction hashes recorded now; Phase 4F linked predictions to checkpoint hashes and common settings but did not publish independent prediction-file hashes.'))
        from .operating_point_report import write_report
        summary = write_report(rows, recovery, analysis, provenance, source, out)
        status.update(status='PASS', thresholds_tested=list(thresholds), threshold_rows=len(rows), **summary)
    except BaseException as error:
        status.update(status='FAIL', error=str(error)); raise
    finally:
        try:
            if any(fingerprint(p) != before[p.relative_to(root).as_posix()] for p in protected):
                raise ValueError('Protected artifacts changed during analysis')
            status['protected_artifact_immutability'] = 'PASS'
        except BaseException as error:
            status.update(status='FAIL', error=str(error)); raise
        finally:
            write_json(out/'analysis_status.json', status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', default='val')
    args = parser.parse_args(argv)
    status = run(split=args.split)
    print(json.dumps({k:v for k,v in status.items() if k != 'thresholds_tested'}, indent=2))
    print('THRESHOLDS TESTED:', len(status['thresholds_tested']))
    return 0
