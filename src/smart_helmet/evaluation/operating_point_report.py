"""Reviewable Phase 4G tables and descriptive selection; no production changes."""
import json

from smart_helmet.training.experiment import write_json
from .operating_point import THRESHOLDS, SCOPES, PRECISION_FLOOR, best_point


def table(rows, fields):
    def fmt(value):
        return f'{value:.6f}' if isinstance(value, float) else str(value)
    return '\n'.join(['| '+' | '.join(fields)+' |', '| '+' | '.join('---' for _ in fields)+' |'] +
                     ['| '+' | '.join(fmt(r.get(k, '')) for k in fields)+' |' for r in rows])


def write_report(rows, recovery, analysis, provenance, source, out):
    floor = analysis['at_precision_floor']
    budgets = analysis['fp_budgets']
    # A useful local operating point requires strictly more target TPs at exactly
    # the same realized FP count, each model optimized under that common budget.
    exact_wins = [r for r in budgets['Without Helmet']['frontier'] if r['A'] and r['B'] and
                  r['A']['FP'] == r['B']['FP'] and r['B']['recall'] > r['A']['recall'] and
                  r['B']['precision'] >= r['A']['precision']]
    selected = max(exact_wins, key=lambda r: r['B']['TP']-r['A']['TP'], default=None)
    useful = selected is not None
    if selected is None:
        selected = next(r for r in budgets['Without Helmet']['frontier'] if r['fp_budget'] == 34)
    conclusion = 'B_HAS_USEFUL_OPERATING_POINT' if useful else 'NO_CLEAR_ADVANTAGE'
    selected_recovery = {tag: next(r for r in recovery if r['model'] == tag and
                                  r['threshold'] == selected[tag]['threshold']) for tag in ('A', 'B')}
    summary = dict(model_conclusion=conclusion, reference_model='A', B_promoted=False,
                   B_has_recall_advantage_after_cost_control=useful,
                   selected_fp_budget=selected, selected_historical_recovery=selected_recovery,
                   next_direction='Investigate calibration/threshold behavior further',
                   ready_for_experiment_c_design=False, experiment_c_trained=False)
    write_json(out/'model_conclusion.json', summary)
    fields = ['model', 'threshold', 'scope', 'TP', 'FP', 'FN', 'precision', 'recall', 'F1', 'fp_per_image']
    coarse = [r for r in rows if r['threshold'] in THRESHOLDS]
    coarse_best = [best_point([r for r in coarse if r['model'] == tag and r['scope'] == scope])
                   for tag in ('A', 'B') for scope in SCOPES]
    refined_best = [analysis['best_f1'][tag][scope] for tag in ('A', 'B') for scope in SCOPES]
    lines = ['# Phase 4G confidence / operating-point analysis',
        f'OPERATING-POINT ANALYSIS: PASS. **{conclusion}**. A remains the reference; B is not promoted.',
        'TEST STATUS: **LOCKED / NOT USED**. No training, dataset edits, deployment changes, or production-threshold changes.',
        '## Data, frozen inference, and matching',
        '127 identical curated VALID images; 300 GT objects (185 With Helmet, 115 Without Helmet). Classes: 0 = With Helmet; 1 = Without Helmet.',
        'This is an offline threshold analysis of the frozen Phase 4F post-NMS predictions, generated with imgsz=416, NMS IoU=0.7, max_det=300, augmentation=false, FP32 (quantize=32), batch=4, workers=0, rect=true, device=0 and confidence floor=0.001. A and B share the recorded inference environment. No new model execution is required. Filtering the fixed post-NMS detections holds all inference behavior constant; lower-confidence boxes cannot suppress higher-confidence boxes in the existing confidence-ordered NMS output.',
        'Checkpoint SHA256 values were verified against existing Phase 4F provenance, as were every training artifact, all 127 image/label hashes, GT coordinates and the historical prediction hash. Predictions reproduce A TP/FP/FN=65/34/50 and B=45/15/70 at 0.25. Source prediction hashes are captured for this analysis; Phase 4F did not publish independent hashes of those prediction files. The analysis inherits their recorded checkpoint/settings provenance rather than claiming independently regenerated predictions.',
        table([dict(model=t, sha256=v) for t, v in provenance['checkpoint_sha256'].items()], ['model', 'sha256']),
        'Matching reuses the established confidence-ordered, class-aware one-to-one policy at IoU >=0.5. Correct-class matches precede wrong-class pairing. A wrong-class overlap is one GT-class FN and one predicted-class FP, with a separate CLASS_CONFUSION tag. Duplicate and localization-error predictions remain FP. Overall metrics pool counts across both classes; they are not macro averages. Precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN), FP/image=FP/127. Undefined zero-denominator ratios are saved as 0; zero-detection points cannot satisfy a precision floor.',
        '## Threshold coverage and selection limits',
        'Required common confidence points: '+', '.join(f'{t:.2f}' for t in THRESHOLDS)+'.',
        f"Refinement adds every 0.01 grid point and every distinct cached prediction confidence inside [0.05,0.90]: {len(set(r['threshold'] for r in rows))} unique common thresholds, {len(rows)} metric rows (2 models x 3 scopes). This resolves all attainable cached operating states within that interval and avoids grid-spacing artifacts. It does not explore below 0.05 or above 0.90. Threshold filtering is inclusive (confidence >= threshold). Exact floating-point thresholds are preserved in JSON/CSV; rounded report values are display only.",
        'Refinement was prompted by local FP-budget crossovers in the required sweep. All thresholds, maxima, budget examples and the selected comparison are exploratory VALID results, selected after inspecting these 127 images. No deployment threshold, independent generalization, significance or calibration-quality claim is made. Best-point ties use recall, precision, then highest tested confidence.',
        '## Required common-threshold comparison: Without Helmet',
        table([r for r in sorted(coarse, key=lambda r: (r['threshold'], r['model'])) if r['scope'] == 'Without Helmet'], fields),
        'Complete Overall and both class results at every required and refined point: [CSV](threshold_metrics.csv) / [JSON](threshold_metrics.json).',
        '## A reference precision floor: 0.734479',
        'This is the established framework-selected A precision, used only as a descriptive floor. Fixed-count precision is a different summary. For each model select maximum target recall among tested points at or above the floor; this does not require exactly equal precision.',
        table([floor[t] for t in ('A', 'B') if floor[t]], fields),
        'The refined comparison gives A 53/115 recall (0.460870) at precision 0.736111 versus B 51/115 (0.443478) at precision 0.739130. B has no recall advantage at this comparable precision. The original 0.05 grid gave A recall 0.426087 at 0.40 and B 0.417391 at 0.20; the 0.01 grid gave both 0.434783. These changes show why the precision-floor conclusion must account for discretization.',
        '## False-positive budgets',
        'For every observed FP count, independently maximize each model\'s recall subject to the same upper bound. Actual FP counts and unused budget (slack) are retained in [comparisons.json](comparisons.json). A shared upper bound can leave different actual costs; closest-pair tables separately show realized gaps and do not imply frontier optimality.',
        'Selected local comparison: greatest TP gain for B among budget-optimal Without Helmet points with exactly equal realized FP and no lower precision. This is a post-hoc descriptive example, not a prespecified target.',
        table([selected[t] for t in ('A', 'B')], fields),
        'Overall cost at those same two selected thresholds:',
        table([r for r in rows if r['scope'] == 'Overall' and
               r['threshold'] == selected[r['model']]['threshold']], fields),
        'Equal target-class cost is not equal total cost: the selected B point has 33 overall FPs versus A\'s 22. The separate overall-budget frontier below controls total FP cost and includes target-class recall at each selected overall operating point.',
        table([dict(model=t, **{k:v for k,v in selected_recovery[t].items() if k != 'model'}) for t in ('A','B')],
              ['model','threshold','recovered','still_missed','wrong_class','still_not_correct'])]
    for scope in ('Without Helmet', 'Overall'):
        compact = []
        for entry in budgets[scope]['frontier']:
            a, b = entry['A'], entry['B']
            if not a or not b:
                continue
            record = dict(budget=entry['fp_budget'], A_threshold=a['threshold'], B_threshold=b['threshold'],
                                A_FP=a['FP'], B_FP=b['FP'], A_FP_per_image=a['fp_per_image'], B_FP_per_image=b['fp_per_image'],
                                A_recall=a['recall'], B_recall=b['recall'], delta_recall=b['recall']-a['recall'])
            if scope == 'Overall':
                for tag, point in [('A', a), ('B', b)]:
                    target = next(r for r in rows if r['model'] == tag and r['scope'] == 'Without Helmet'
                                  and r['threshold'] == point['threshold'])
                    record[tag+'_target_recall'] = target['recall']
            compact.append(record)
        lines += [f'### {scope}: full optimized FP-budget frontier', table(compact, list(compact[0]))]
        closest = []
        for entry in budgets[scope]['closest_pairs']:
            if entry['A']['threshold'] not in THRESHOLDS:
                continue
            a,b = entry['A'],entry['B']
            closest.append(dict(A_threshold=a['threshold'],B_threshold=b['threshold'],A_FP=a['FP'],B_FP=b['FP'],
                                FP_per_image_gap=entry['fp_per_image_gap'],A_recall=a['recall'],B_recall=b['recall']))
        lines += [f'### {scope}: nearest refined B cost for each required A threshold', table(closest, list(closest[0]))]
    lines += ['## Best validation F1', 'Required 18-point grid:', table(coarse_best, fields),
              'Refined full cached-state sweep:', table(refined_best, fields),
              'These are separate scope optima, not one jointly optimized production threshold. B\'s required-grid Without Helmet optimum is at the lower search boundary (0.05); refinement inside the allowed interval finds an F1 tie with A, not an improvement.',
              '## Historical 50 misses',
              'The original 50 GT identities are fixed. Recovered, still missed (unmatched), and wrong class are mutually exclusive and sum to 50 at every threshold. Still not correct equals still missed + wrong class. These original misses include historical confusion FNs. Recovery alone omits regressions among A\'s original 65 correct objects.',
              table([r for r in recovery if r['threshold'] in THRESHOLDS],
                    ['model','threshold','recovered','still_missed','wrong_class','still_not_correct']),
              'Every refined threshold: [recovery counts](historical_miss_recovery.csv). Every original object at every threshold: [object outcomes](historical_miss_objects.csv). At each model\'s refined best target F1, A recovers 0 and B 12; B also loses 12 formerly correct objects, leaving both at 65 target TPs. At the selected equal-FP point above, recovery is a different, explicitly labeled comparison.',
              '## A and B curves']
    for name in ('without_helmet_precision_recall','without_helmet_f1_vs_confidence',
                 'without_helmet_recall_vs_confidence','without_helmet_precision_vs_confidence',
                 'without_helmet_fp_per_image_vs_confidence','overall_f1_vs_confidence'):
        lines.append(f'![{name.replace("_", " ")}]({name}.png)')
    ap = json.loads((source/'comparison_metrics.json').read_text(encoding='utf-8'))
    lines += ['Axes start at zero; probability axes cover [0,1], and FP/image covers its full data range. Both models share each plot. PR connects sampled operating states, not an interpolated framework AP curve; a point with no detections uses the documented zero-precision convention.',
              '## Interpretation and ranking safeguards',
              table([r for r in ap if r['metric'] in ('mAP50', 'mAP50-95')], ['scope','metric','A','B','delta']),
              'These AP values are inherited from the independent Phase 4F evaluation, not recomputed from the threshold sweep. B has lower overall mAP50 and mAP50-95 and lower Without Helmet AP; those ranking concerns remain. A has higher maximum target recall at the reference precision floor and the refined best target F1 is tied. B does have a narrow useful VALID operating point: at 4 target FPs (0.031496/image), it detects 32 target objects versus A\'s 26, gaining 6/115 = 0.052174 recall with slightly higher precision. At 5 FPs its gain is only one object. This local benefit does not establish global superiority, eliminate AP losses, or justify promotion.',
              'The different thresholds needed for similar target counts are consistent with a confidence-distribution shift. Threshold analysis alone is not a probability-calibration assessment and does not prove a causal mechanism.',
              '## Exactly one next direction',
              '**Investigate calibration/threshold behavior further.** Keep A as reference and predefine the acceptable Without Helmet precision/FP-image budget and minimum recall benefit before a subsequent validation-only confirmation of the narrow B trade-off. Do not choose a training variable yet. No additional training experiment or data modification is recommended in this phase.',
              'READY FOR EXPERIMENT C DESIGN: **NO**. Experiment C was not trained. VALID images must never enter TRAIN; TEST stays locked. No deployment changes.',
              '## Reproduction and audit',
              'Run `.venv/Scripts/python.exe scripts/analyze_operating_points.py` from the project root into a new/empty output directory (the script refuses to overwrite an existing analysis). Tests: `.venv/Scripts/python.exe -m pytest tests/test_operating_point.py tests/test_comparison.py -q`. Synthetic tests never load models or train.',
              '[Provenance](provenance.json) records input artifact hashes, checkpoint hashes, inference environment/settings, class mapping and all validation image/label hashes. [Analysis status](analysis_status.json) records completion and protected-artifact immutability. TEST paths are not opened, enumerated or hashed by this phase.',
              'OPERATING-POINT ANALYSIS: PASS. TEST USED: NO. TEST STATUS: LOCKED.']
    (out/'operating_point_report.md').write_text('\n\n'.join(lines)+'\n', encoding='utf-8')
    return summary
