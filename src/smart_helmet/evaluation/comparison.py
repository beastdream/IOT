"""Phase 4F: frozen A/B checkpoints, paired VALID evaluation, no training."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from smart_helmet.paths import PROJECT_ROOT
from smart_helmet.foundation import EXPECTED_CLASSES, IMAGE_EXTENSIONS
from smart_helmet.dataset.audit import write_csv
from smart_helmet.dataset.duplicates import sha256_file
from smart_helmet.training.environment import verify_dataset, check_unchanged, runtime_info
from smart_helmet.training.experiment import write_json, write_yaml, result_rows
from smart_helmet.training.trainer import frozen_cache
from .evaluator import validation_guard, check_model
from .failure_analysis import (analyze_image, ground_truth, match_boxes, iou, aggregate,
                               CONFIDENCE, MATCH_IOU, render_case)
from .failure_review import guarded_image
from .metrics import KEYS, FRAMEWORK_KEYS, serialize_metrics

RUNS = {'A': 'A_baseline', 'B': 'B_class_aware_sampling'}


def fingerprint(directory):
    return {p.relative_to(directory).as_posix(): sha256_file(p)
            for p in sorted(Path(directory).rglob('*')) if p.is_file()}


def checkpoint_fingerprint(path):
    path = Path(path)
    if not path.is_file() or not path.stat().st_size:
        raise ValueError('Nonempty best.pt required; no fallback')
    return sha256_file(path)


def metric_delta(a, b):
    return {key: b[key] - a[key] for key in KEYS}


def metric_comparison(metrics):
    rows = []
    for scope, cid in [('Overall', None), *[(name, str(c)) for c, name in EXPECTED_CLASSES.items()]]:
        values = {tag: m['overall'] if cid is None else m['per_class'][cid] for tag, m in metrics.items()}
        delta = metric_delta(values['A'], values['B'])
        for key in KEYS:
            rows.append(dict(scope=scope, metric=key, A=values['A'][key], B=values['B'][key], delta=delta[key]))
    return rows


def ensure_same_validation(a, b, expected):
    def indexed(cases):
        mapping = {c['image']: c for c in cases}
        if len(mapping) != len(cases) or set(mapping) != set(expected):
            raise ValueError('Same VALID set required: missing, duplicate or foreign image')
        return mapping
    aa, bb = indexed(a), indexed(b)
    for name in expected:
        for key in ('ground_truth', 'width', 'height'):
            if aa[name][key] != bb[name][key]:
                raise ValueError('Same VALID ground truth and dimensions required')
    return aa, bb


def object_outcomes(case):
    active = [dict(p, source_index=j) for j, p in enumerate(case['predictions']) if p['confidence'] >= CONFIDENCE]
    matches = match_boxes(case['ground_truth'], active)
    confusion = {i: (j, score) for i, j, score in matches['confusions']}
    outcomes = []
    for i in range(len(case['ground_truth'])):
        link = matches['matches'].get(i, confusion.get(i))
        status = 'CORRECT' if i in matches['matches'] else 'CLASS_CONFUSION' if i in confusion else 'MISSED'
        pred, score = (active[link[0]], link[1]) if link else (None, None)
        outcomes.append(dict(outcome=status, confidence=pred['confidence'] if pred else None,
                             predicted_class=EXPECTED_CLASSES[pred['class_id']] if pred else None,
                             iou=score, prediction_index=pred['source_index'] if pred else None))
    return outcomes, [active[j] for j in matches['extra']]


def transition(a, b):
    if a == 'CORRECT':
        return {'CORRECT': 'CORRECT_IN_BOTH', 'MISSED': 'CORRECT_IN_A_MISSED_IN_B',
                'CLASS_CONFUSION': 'CORRECT_IN_A_WRONG_IN_B'}[b]
    if b == 'CORRECT':
        return 'A_CLASS_CONFUSION_B_CORRECT' if a == 'CLASS_CONFUSION' else 'RECOVERED_BY_B'
    if a == 'MISSED' and b == 'CLASS_CONFUSION':
        return 'A_MISSED_B_CLASS_CONFUSION'
    return 'STILL_MISSED_BY_B'


def false_positive_transitions(a, b):
    # Same-image, same-class one-to-one maximum-overlap greedy association.
    # Persistent means spatially corresponding, not semantic identity certainty.
    edges = sorted([(iou(x['bbox'], y['bbox']), i, j) for i, x in enumerate(a)
                    for j, y in enumerate(b) if x['class_id'] == y['class_id']],
                   key=lambda x: (-x[0], x[1], x[2]))
    used_a, used_b, rows = set(), set(), []
    for score, i, j in edges:
        if score >= MATCH_IOU and i not in used_a and j not in used_b:
            used_a.add(i); used_b.add(j)
            rows.append(dict(category='PERSISTENT_FALSE_POSITIVE', A=a[i], B=b[j], cross_model_iou=score))
    rows += [dict(category='NEW_FALSE_POSITIVE_IN_B', A=None, B=p, cross_model_iou=None)
             for j, p in enumerate(b) if j not in used_b]
    rows += [dict(category='REMOVED_FALSE_POSITIVE_IN_B', A=p, B=None, cross_model_iou=None)
             for i, p in enumerate(a) if i not in used_a]
    return rows


def paired_analysis(cases, expected, historic, human):
    aa, bb = ensure_same_validation(cases['A'], cases['B'], expected)
    old, _ = ensure_same_validation(historic, cases['A'], expected)
    old_misses = {(name, i) for name, c in old.items() for i in c['missed_gt'] if c['ground_truth'][i]['class_id'] == 1}
    new_misses = {(name, i) for name, c in aa.items() for i in c['missed_gt'] if c['ground_truth'][i]['class_id'] == 1}
    if old_misses != new_misses or len(old_misses) != 50:
        raise ValueError('Independent A predictions did not reproduce the 50 historical missed objects')
    reviews = {(r['image'], int(r['gt_object_id'])): r for r in human}
    if len(reviews) != len(human) or set(reviews) != old_misses or any(r['reviewed'].lower() != 'true' for r in human):
        raise ValueError('Human review must cover exactly the 50 historical missed objects')
    objects, fps, recovery = [], [], []
    for name in sorted(expected):
        a, afp = object_outcomes(aa[name]); b, bfp = object_outcomes(bb[name])
        for i, gt in enumerate(aa[name]['ground_truth']):
            row = dict(image=name, ground_truth_index=i, gt_class=EXPECTED_CLASSES[gt['class_id']],
                       gt_bbox=gt['bbox'], transition_category=transition(a[i]['outcome'], b[i]['outcome']))
            row.update({tag+'_'+k: v for tag, item in [('A', a[i]), ('B', b[i])] for k, v in item.items()})
            objects.append(row)
            if (name, i) in old_misses:
                review = reviews[(name, i)]
                recovery.append(dict(row, human_categories=review['human_categories'], human_notes=review['notes']))
        fps.extend(dict(image=name, **row) for row in false_positive_transitions(afp, bfp))
    fp_counts = {}
    for cid, name in EXPECTED_CLASSES.items():
        chosen = [r for r in fps if (r['B'] or r['A'])['class_id'] == cid]
        counts = Counter(r['category'] for r in chosen)
        fp_counts[name] = dict(A=sum(r['A'] is not None for r in chosen), B=sum(r['B'] is not None for r in chosen),
            new=counts['NEW_FALSE_POSITIVE_IN_B'], removed=counts['REMOVED_FALSE_POSITIVE_IN_B'],
            persistent=counts['PERSISTENT_FALSE_POSITIVE'])
        assert fp_counts[name]['B'] - fp_counts[name]['A'] == fp_counts[name]['new'] - fp_counts[name]['removed']
    recovered = sum(r['B_outcome'] == 'CORRECT' for r in recovery)
    changed = sum(r['transition_category'] == 'A_MISSED_B_CLASS_CONFUSION' for r in recovery)
    human_counts = {}
    for category in sorted({c for r in recovery for c in r['human_categories'].split('|') if c}):
        group = [r for r in recovery if category in r['human_categories'].split('|')]
        human_counts[category] = dict(total=len(group), recovered=sum(r['B_outcome'] == 'CORRECT' for r in group))
    summary = dict(baseline_missed=len(recovery), recovered=recovered, still_not_correct=len(recovery)-recovered,
        still_missed_without_class_confusion=sum(r['B_outcome'] == 'MISSED' for r in recovery),
        B_class_confusion=sum(r['B_outcome'] == 'CLASS_CONFUSION' for r in recovery), changed_to_class_confusion=changed,
        transitions=dict(Counter(r['transition_category'] for r in objects)),
        transitions_per_class={name: dict(Counter(r['transition_category'] for r in objects if r['gt_class'] == name))
                               for name in EXPECTED_CLASSES.values()},
        recovery_transitions=dict(Counter(r['transition_category'] for r in recovery)),
        false_positives=fp_counts, human_categories=human_counts,
        fixed_threshold_metrics={tag: aggregate(cs)['per_class'] for tag, cs in cases.items()})
    return objects, recovery, fps, summary


def selection(rows):
    delta = {(r['scope'], r['metric']): r['delta'] for r in rows}
    target = delta['Without Helmet', 'recall'] > 0
    precision = delta['Without Helmet', 'precision'] >= 0 and delta['Overall', 'precision'] >= 0
    maps = all(delta['Overall', k] >= 0 for k in ('mAP50', 'mAP50-95'))
    wh_ap = delta['Without Helmet', 'mAP50'] >= 0
    decision = 'B_IS_CANDIDATE' if target and precision and maps and wh_ap else (
        'A_REMAINS_REFERENCE' if not target or (not precision and not maps) else 'TRADEOFF_REQUIRES_DECISION')
    return dict(primary_target_improved=target, precision_safeguard='PASS' if precision else 'FAIL',
                overall_map_safeguard='PASS' if maps else 'FAIL', without_helmet_map50_safeguard='PASS' if wh_ap else 'FAIL',
                model_decision=decision,
                policy='Conservative descriptive non-degradation safeguards (delta >= 0); no pre-registered tolerance or significance claim. Candidate requires recall gain and all safeguards. Recall gain with both precision and overall AP loss keeps A as reference.')


def training_comparison(root, metadata, checkpoint_metrics, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ultralytics.utils.metrics import Metric
    fields = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'val/box_loss',
              'val/cls_loss', 'val/dfl_loss', *FRAMEWORK_KEYS[2:]]
    summaries = {}; fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for tag, run in RUNS.items():
        rows = result_rows(root/'results/experiments'/run)
        scores = []
        for row in rows:
            metric = Metric(); metric.mean_results = lambda row=row: [float(row[k]) for k in FRAMEWORK_KEYS]
            scores.append(float(metric.fitness()))
        best = max(range(len(scores)), key=lambda i: scores[i])
        epochs = [int(row['epoch']) for row in rows]
        if epochs != list(range(1, len(rows)+1)) or len(rows) != metadata[tag]['epochs_completed']:
            raise ValueError('Inconsistent training epoch evidence')
        ckpt_fitness = float(checkpoint_metrics[tag]['fitness'])
        if abs(ckpt_fitness-scores[best]) > .001:
            raise ValueError('Checkpoint fitness does not confirm best CSV epoch')
        summaries[tag] = dict(best_epoch=epochs[best], stopped_epoch=epochs[-1],
            training_seconds=float(rows[-1]['time']), pipeline_seconds=metadata[tag]['duration_seconds'],
            checkpoint_fitness=ckpt_fitness, best_csv_fitness=scores[best], checkpoint_matches_best_csv=True,
            best_losses={k: float(rows[best][k]) for k in fields}, final_losses={k: float(rows[-1][k]) for k in fields},
            epochs=[{k: float(r[k]) for k in ['epoch', 'time', *fields]} for r in rows])
        for ax, key in zip(axes.flat, fields):
            line, = ax.plot(epochs, [float(r[key]) for r in rows], label=tag)
            ax.axvline(epochs[best], color=line.get_color(), linestyle='--', alpha=.5)
            ax.set_title(key); ax.set_xlabel('Epoch'); ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Frozen A/B TRAIN and VALID curves; dashed lines = best epochs')
    fig.tight_layout(); fig.savefig(out/'training_curves.png', dpi=150); plt.close(fig)
    write_json(out/'training_comparison.json', summaries)
    return summaries


def report_text(rows, summary, training, decision, provenance):
    deltas = {(r['scope'], r['metric']): r['delta'] for r in rows}
    wh_transitions = summary['transitions_per_class']['Without Helmet']
    wh_regressions = sum(wh_transitions.get(key, 0) for key in
                         ('CORRECT_IN_A_MISSED_IN_B', 'CORRECT_IN_A_WRONG_IN_B'))
    lines = ['# Controlled A vs B comparison',
        'TEST STATUS: LOCKED / NOT USED. No training performed. Both training directories and dataset are immutable.',
        f"Independent evaluation: {provenance['validation_images']} identical VALID images; imgsz=416, batch=4, workers=0, FP32, confidence floor=0.001, NMS IoU=0.7, max_det=300, no augmentation. Both use the same environment and args in provenance.json. AP uses framework IoU thresholds 0.50:0.05:0.95.",
        '## Metrics', '| Scope | Metric | A | B | B - A |', '|---|---|---:|---:|---:|']
    lines += [f"| {r['scope']} | {r['metric']} | {r['A']:.9f} | {r['B']:.9f} | {r['delta']:+.9f} |" for r in rows]
    lines += ['Precision and recall above use each model\'s framework-selected operating point. AP integrates rankings. They must not be equated with the fixed-0.25 object counts below.',
        '## Without Helmet recovery and regressions',
        f"The independent A inference reproduced the identities of all {summary['baseline_missed']} historical misses. Recovered by B: {summary['recovered']}; still not correct: {summary['still_not_correct']} (unmatched: {summary['still_missed_without_class_confusion']}; wrong class: {summary['B_class_confusion']}). Newly changed from unmatched to class confusion: {summary['changed_to_class_confusion']}.",
        'Each GT has one exclusive transition. RECOVERED_BY_B excludes A_CLASS_CONFUSION_B_CORRECT; total recovered combines both. STILL_MISSED_BY_B includes non-recovered cases except the explicitly separated A_MISSED_B_CLASS_CONFUSION. See both outcomes to distinguish persistent confusion. Confusion is one GT FN and one prediction FP, never an extra GT object.',
        'Recovery transition counts: '+json.dumps(summary['recovery_transitions']),
        'All-GT transition counts: '+json.dumps(summary['transitions']),
        'Without Helmet transition counts: '+json.dumps(wh_transitions),
        f"Without Helmet gains {summary['recovered']} formerly missed objects but loses {wh_regressions} formerly correct objects at confidence 0.25. Net correct-object change: {summary['recovered']-wh_regressions:+d}.",
        'Outcome confidence/class/IoU describe the assigned active prediction. Blank values mean no eligible active match, not zero confidence or absence of every low-confidence proposal; complete low-confidence predictions are preserved in the A/B prediction JSON files.',
        '[All GT transitions](object_transitions.csv) | [All 50 historical misses](without_helmet_recovery.csv)',
        '## False-positive cost',
        'Fixed diagnostic confidence >=0.25 and class-aware one-to-one GT matching IoU >=0.5, followed by wrong-class pairing. Unmatched active predictions include duplicates and localization errors. Cross-model FPs are associated one-to-one by same class, descending IoU >=0.5. New/removed are operational spatial categories; localization shifts may appear as one removal plus one addition.',
        '| Class | A FP | B FP | New B FP | Removed A FP | Persistent |', '|---|---:|---:|---:|---:|---:|']
    lines += [f"| {name} | {v['A']} | {v['B']} | {v['new']} | {v['removed']} | {v['persistent']} |" for name, v in summary['false_positives'].items()]
    totals = {k: sum(v[k] for v in summary['false_positives'].values()) for k in ('A','B','new','removed','persistent')}
    lines += [f"| Total | {totals['A']} | {totals['B']} | {totals['new']} | {totals['removed']} | {totals['persistent']} |"]
    lines += ['FP accounting satisfies B - A = new - removed per class. [FP transitions](false_positive_transitions.json) | [Annotated new B FP examples](new_false_positive_examples/manifest.json). Each example shows all GT and only newly unmatched B predictions, identified by their original prediction IDs in the manifest.',
        '## Fixed-threshold diagnostic metrics',
        '| Model | Class | TP | FN | FP | Precision | Recall |', '|---|---|---:|---:|---:|---:|---:|']
    for tag, classes in summary['fixed_threshold_metrics'].items():
        for cid, v in classes.items():
            lines.append(f"| {tag} | {EXPECTED_CLASSES[int(cid)]} | {v['true_positive_count']} | {v['false_negative_count']} | {v['false_positive_count']} | {v['precision']:.6f} | {v['recall']:.6f} |")
    lines += ['These common-threshold counts need not move in the same direction as framework-selected P/R. Fewer total FPs alone is not improvement when true positives also decline. The class distribution of FP cost must be considered.',
        '## Precision / recall trade-off and selection', decision['policy'],
        f"Without Helmet framework recall changes by {100*deltas['Without Helmet','recall']:+.4f} percentage points versus precision {100*deltas['Without Helmet','precision']:+.4f} points, mAP50 {100*deltas['Without Helmet','mAP50']:+.4f} points, and mAP50-95 {100*deltas['Without Helmet','mAP50-95']:+.4f} points.",
        f"Overall framework recall changes by {100*deltas['Overall','recall']:+.4f} percentage points versus precision {100*deltas['Overall','precision']:+.4f} points, mAP50 {100*deltas['Overall','mAP50']:+.4f} points, and mAP50-95 {100*deltas['Overall','mAP50-95']:+.4f} points.",
        'Primary target improved: '+str(decision['primary_target_improved'])+'. Precision safeguard: '+decision['precision_safeguard']+'. Overall mAP safeguard: '+decision['overall_map_safeguard']+'. Without Helmet mAP50 safeguard: '+decision['without_helmet_map50_safeguard']+'.',
        '**'+decision['model_decision']+'**. A recall increase alone does not establish a meaningful improvement. These are single-seed results on 127 images, without a deployment cost function or repeated-seed uncertainty estimate; no statistical or TEST generalization claim is made.',
        '## Human review', json.dumps(summary['human_categories']),
        'Categories are copied only from reviewed human entries. ANNOTATION_OK confirms annotation acceptability, not a failure cause; OTHER remains unspecified. Category totals can overlap. No occlusion, crowded-scene, or side-view recovery cause is inferred when absent.',
        '## Training comparison', '| Model | Best epoch | Stopped epoch | Training hours | Pipeline hours |', '|---|---:|---:|---:|---:|']
    lines += [f"| {tag} | {v['best_epoch']} | {v['stopped_epoch']} | {v['training_seconds']/3600:.4f} | {v['pipeline_seconds']/3600:.4f} |" for tag, v in training.items()]
    lines += ['| Loss | A best / final | B best / final |', '|---|---:|---:|']
    for field in ('train/box_loss','train/cls_loss','train/dfl_loss','val/box_loss','val/cls_loss','val/dfl_loss'):
        lines.append('| '+field+' | '+' | '.join(f"{training[tag]['best_losses'][field]:.5f} / {training[tag]['final_losses'][field]:.5f}" for tag in ('A','B'))+' |')
    lines += ['[Training curves](training_curves.png) | [Every epoch, best/final losses and times](training_comparison.json). Best epoch is selected with installed framework fitness and confirmed against checkpoint fitness. Earlier stopping or faster convergence does not prove better generalization.',
        '## One primary next experiment',
        'Plan a controlled confidence/operating-point analysis using frozen A and B checkpoints on VALID: compare Without Helmet recall at a common precision floor equal to A\'s reference precision and compare false positives per image. Vary only the confidence operating point, keep weights, data, resolution and NMS fixed, and retain A as reference until the trade-off is justified. This will test whether extra recall reflects a confidence shift rather than improved ranking. VALID-selected thresholds would remain exploratory; do not claim independent generalization or unlock TEST. Do not train C or change production thresholds.',
        '## Provenance', json.dumps(provenance['checkpoint_sha256']),
        '[Full provenance and frozen artifact hashes](provenance.json). TEST images/labels are never inference, matching, metric or selection inputs; whole-dataset hashing is used only for immutability auditing.',
        'READY FOR EXPERIMENT C PLANNING: YES. C EXECUTED: NO. TEST STATUS: LOCKED / NOT USED.']
    result = ''
    for index, line in enumerate(lines):
        separator = '\n' if index and line.startswith('|') and lines[index-1].startswith('|') and not line.startswith('| Loss') else '\n\n'
        result += (separator if index else '') + line
    return result+'\n'


def run(root=PROJECT_ROOT, split='val', device='auto'):
    validation_guard(split)
    root = Path(root).resolve(); out = root/'results/evaluation/A_vs_B'
    if out.resolve() != out or out.exists() and any(out.iterdir()):
        raise ValueError('Comparison output must be new, empty and not redirected; preserve prior results')
    directories = {tag: root/'results/experiments'/name for tag, name in RUNS.items()}
    checkpoints = {tag: d/'weights/best.pt' for tag, d in directories.items()}
    hashes = {tag: checkpoint_fingerprint(p) for tag, p in checkpoints.items()}
    frozen = {tag: fingerprint(d) for tag, d in directories.items()}
    metadata = {tag: json.loads((d/'experiment.json').read_text()) for tag, d in directories.items()}
    snapshot = verify_dataset(root); env = runtime_info(device)
    for m in metadata.values():
        if m['training_status'] != 'PASS' or m['test_status'] != 'LOCKED' or m['dataset_fingerprint'] != snapshot['curated_hash']:
            raise ValueError('Passing TEST-locked runs on the same dataset required')
        if m['ultralytics_version'] != env['ultralytics_version']:
            raise ValueError('Framework version mismatch')
    from smart_helmet.training.experiment_b_trainer import compare_baseline_arguments
    argument_check = compare_baseline_arguments(metadata['B']['resolved_training_arguments'], metadata['A']['resolved_training_arguments'])
    for key in ('optimizer_class', 'optimizer_initial_parameter_groups', 'parameter_count'):
        if metadata['A'][key] != metadata['B'][key]:
            raise ValueError('Training optimizer drift: '+key)
    historic_path = root/'results/evaluation/A_baseline_val/validation_predictions.json'
    review_path = root/'results/evaluation/A_baseline_val/failure_review/human_review.csv'
    historic = json.loads(historic_path.read_text())
    if historic['checkpoint_sha256'] != hashes['A'] or historic['dataset_fingerprint'] != snapshot['curated_hash']:
        raise ValueError('Historical A prediction provenance mismatch')
    expected = sorted(c['image'] for c in historic['cases'])
    images = [guarded_image(root, name) for name in expected]
    actual_images = sorted(p.resolve() for p in Path(snapshot['data']['val']).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if len(expected) != 127 or len(set(expected)) != 127 or set(images) != set(actual_images):
        raise ValueError('Exactly the same 127 VALID images required')
    with review_path.open(encoding='utf-8-sig', newline='') as f:
        human = list(csv.DictReader(f))
    historical_hash = sha256_file(historic_path)
    review_hash = sha256_file(review_path)
    out.mkdir(parents=True, exist_ok=True)
    status = dict(status='RUNNING', test_status='LOCKED / NOT USED', training_executed=False)
    try:
        write_yaml(out/'validation_data.yaml', {k: snapshot['data'][k] for k in ('path', 'train', 'val', 'nc', 'names')})
        common = dict(imgsz=416, batch=4, workers=0, device=env['device'], conf=.001, iou=.7,
                      max_det=300, quantize=32, augment=False)
        provenance = dict(checkpoint_sha256=hashes, training_artifact_hashes=frozen, environment=env,
            validation_images=127, validation_manifest=[dict(image=name, image_sha256=sha256_file(p),
                label_sha256=sha256_file(p.parent.parent/'labels'/(p.stem+'.txt'))) for name, p in zip(expected, images)],
            common_evaluation_args=common, dataset_fingerprint=snapshot['curated_hash'],
            historic_predictions_sha256=sha256_file(historic_path), human_review_sha256=sha256_file(review_path),
            test_status='LOCKED / NOT USED', baseline_argument_check=argument_check)
        write_json(out/'provenance.json', provenance)
        from ultralytics import YOLO
        metrics, cases, checkpoint_metrics = {}, {}, {}
        protected = [root/'data/curated/v1', root/'train', root/'valid', root/'test', *directories.values()]
        with frozen_cache(out/'dataset_cache', protected):
            for tag, checkpoint in checkpoints.items():
                print('Independent VALID evaluation: '+tag, flush=True)
                model = YOLO(str(checkpoint)); check_model(model)
                checkpoint_metrics[tag] = model.ckpt['train_metrics']
                args = dict(common, data=str(out/'validation_data.yaml'), split='val', plots=True,
                            save_json=False, save_txt=False, cache=False, project=str(out), name=tag, exist_ok=True)
                metrics[tag] = serialize_metrics(model.val(**args))
                write_json(out/(tag+'_metrics.json'), dict(metrics[tag], checkpoint_sha256=hashes[tag], evaluation_args=args))
                # New model avoids state carried over from validation into prediction.
                model = YOLO(str(checkpoint)); check_model(model)
                predictions = model.predict(source=[str(p) for p in images], **common, rect=True,
                                            save=False, verbose=False, stream=True)
                cs, seen = [], set()
                for result in predictions:
                    path = Path(result.path).resolve()
                    if path not in set(images) or path in seen:
                        raise ValueError('Non-VALID or duplicate prediction result')
                    seen.add(path); gt, w, h = ground_truth(path); boxes = result.boxes
                    pred = [dict(class_id=int(c), confidence=float(v), bbox=list(map(float, b)))
                            for b, c, v in zip(boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist())]
                    if any(p['class_id'] not in EXPECTED_CLASSES for p in pred):
                        raise ValueError('Unexpected predicted class')
                    cs.append(analyze_image(path.relative_to(root).as_posix(), gt, pred, w, h))
                if seen != set(images):
                    raise ValueError('Incomplete VALID inference')
                cases[tag] = sorted(cs, key=lambda c: c['image'])
                write_json(out/(tag+'_validation_predictions.json'), dict(checkpoint_sha256=hashes[tag], cases=cases[tag]))
        rows = metric_comparison(metrics)
        write_csv(out/'comparison_metrics.csv', rows, ['scope', 'metric', 'A', 'B', 'delta'])
        write_json(out/'comparison_metrics.json', rows)
        objects, recovery, fps, summary = paired_analysis(cases, expected, historic['cases'], human)
        for filename, data in [('object_transitions.csv', objects), ('without_helmet_recovery.csv', recovery)]:
            write_csv(out/filename, data, list(data[0]))
        write_json(out/'false_positive_transitions.json', fps)
        write_json(out/'transition_summary.json', summary)
        example_dir = out/'new_false_positive_examples'; example_dir.mkdir()
        manifest = []
        bb = {c['image']: c for c in cases['B']}
        for name in sorted({r['image'] for r in fps if r['category'] == 'NEW_FALSE_POSITIVE_IN_B'})[:12]:
            new = [r['B'] for r in fps if r['image'] == name and r['category'] == 'NEW_FALSE_POSITIVE_IN_B']
            case = dict(bb[name], predictions=new, events=[])
            target = example_dir/(Path(name).stem+'.png'); render_case(case, root, target)
            manifest.append(dict(image=name, visualization=target.relative_to(out).as_posix(), new_false_positives=new))
        write_json(example_dir/'manifest.json', manifest)
        training = training_comparison(root, metadata, checkpoint_metrics, out)
        decision = selection(rows)
        write_json(out/'model_decision.json', decision)
        (out/'A_vs_B_report.md').write_text(report_text(rows, summary, training, decision, provenance), encoding='utf-8')
        status.update(status='PASS', validation_images=127, **decision)
        print(json.dumps(dict(metrics=rows, recovery=summary, decision=decision), indent=2), flush=True)
    except BaseException as error:
        status.update(status='FAIL', error=str(error)); raise
    finally:
        try:
            check_unchanged(snapshot, root)
            if any(fingerprint(d) != frozen[tag] for tag, d in directories.items()):
                raise ValueError('Frozen A/B artifacts changed')
            if sha256_file(historic_path) != historical_hash or sha256_file(review_path) != review_hash:
                raise ValueError('Historical predictions or human review changed')
            status['dataset_immutability'] = status['training_artifact_immutability'] = 'PASS'
        except BaseException as error:
            status.update(status='FAIL', immutability_error=str(error)); raise
        finally:
            write_json(out/'comparison_status.json', status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', default='val'); parser.add_argument('--device', default='auto')
    args = parser.parse_args(argv)
    try:
        run(split=args.split, device=args.device)
        print('A VS B COMPARISON: PASS\nTEST: LOCKED / NOT USED\nTRAINING EXECUTED: NO')
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print('A VS B COMPARISON: FAIL\n'+str(error)); return 1
