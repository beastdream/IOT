"""Evidence-led report for fixed exploratory operating budgets."""
from .operating_budget import MODES, CONSTRAINTS, flatten


def table(rows, keys):
    def fmt(v):
        return '' if v is None else f'{v:.6f}' if isinstance(v,float) else str(v)
    return '\n'.join(['| '+' | '.join(keys)+' |','| '+' | '.join('---' for k in keys)+' |']+
                     ['| '+' | '.join(fmt(r.get(k)) for k in keys)+' |' for r in rows])


def write_report(out,policy,results,rows,frontiers,comparisons,residuals,prior,ap):
    counts = {tag:sum(r['model']==tag and r['policy']=='ALL_TESTED' for r in frontiers) for tag in ('A','B')}
    persistent = {tag:dict(no_correct_class_proposal=sum(r['model']==tag for r in residuals),
                           wrong_class_overlap=sum(r['model']==tag and r['wrong_class_overlap'] for r in residuals))
                  for tag in ('A','B')}
    consistent = all(results[m]['decision']=='B_PREFERRED_UNDER_THIS_BUDGET' and results[m]['optima_same_cost'] for m in MODES)
    # Promotion still requires an independent decision; this phase is descriptive.
    c_ready = persistent['A']['no_correct_class_proposal'] > 0 and persistent['B']['no_correct_class_proposal'] > 0
    summary = dict(B_consistently_better_under_controlled_cost=consistent,A_remains_reference=True,
                   validation_overuse_warning=True,recommended_next_action='EXPERIMENT_C_DESIGN' if c_ready else 'FREEZE_MODEL',
                   ready_for_experiment_c_design=c_ready,pareto_counts=counts,persistent_failure_counts=persistent,
                   policy_decisions={m:results[m]['decision'] for m in MODES},
                   proposed_change='Hard-example TRAIN sampling' if c_ready else None)
    budget_rows = [dict(policy=m,**policy['modes'][m]['constraints']) for m in MODES]
    lines = ['# Phase 4H predefined operating-budget analysis',
             '**OPERATING BUDGET ANALYSIS: PASS. VALIDATION / EXPLORATORY.**',
             '127 VALID images. TEST USED: NO. TEST STATUS: LOCKED / NOT USED. A remains reference; B is not promoted. No inference, training, weight edits, dataset edits, calibration transformations, or production confidence changes.',
             '## Policy definition before evaluation',
             '**EXPLORATORY VALIDATION BUDGETS**. No external deployment limits were supplied. The three round-number scenarios and decision/tie rules were saved before Phase 4H result evaluation. Earlier Phase 4G findings were known, so this is not an independent preregistration or held-out assessment. Limits were not adjusted after observing outcomes.',
             table(budget_rows,['policy',*CONSTRAINTS])]
    lines += [f"**{m}:** {policy['modes'][m]['rationale']}" for m in MODES]
    lines += ['The canonical config is `configs/evaluation/operating_policy.yaml`; an identical copy is [operating_policy.yaml](operating_policy.yaml). Its SHA256 and freeze timestamp are recorded in [reproducibility.json](reproducibility.json). No mode is a production policy.',
              'All three constraints are inclusive and must hold simultaneously. Maximize Without Helmet recall among feasible saved thresholds. Ties choose higher target precision, lower overall FP, lower target FP, then higher confidence. An empty feasible set is NO_FEASIBLE_OPERATING_POINT; limits are never weakened. Zero target detections are excluded because target precision is undefined (the saved sweep represents it as zero).',
              'Decision rule fixed in advance: compare independently optimized points only if both realized target FP and overall FP differ by at most one detection (1/127 = 0.007874 FP/image). Prefer strictly higher target recall only without lower target precision. Otherwise NO_CLEAR_ADVANTAGE. One feasible model wins on feasibility; neither yields NO_FEASIBLE_POINT. Shared upper limits alone do not establish equal realized cost.',
              '## Policy optima and decisions']
    for mode in MODES:
        rr = [r for r in rows if r['policy']==mode]
        lines += [f'### {mode}',
                  table(rr,['model','status','threshold','wh_TP','wh_FP','wh_FN','wh_precision','wh_recall','wh_F1','wh_fp_per_image']),
                  table(rr,['model','with_precision','with_recall','with_F1','overall_TP','overall_FP','overall_FN','overall_precision','overall_recall','overall_F1','overall_fp_per_image']),
                  table(rr,['model','historical_recovered','historical_wrong_class','historical_still_missed','class_confusion_count','without_helmet_class_confusion_count']),
                  f"Decision: **{results[mode]['decision']}**. Independent optima satisfy same-cost tolerance: **{results[mode]['optima_same_cost']}**. Feasible threshold counts: {results[mode]['feasible_counts']}."]
        pair = comparisons[mode]
        lines += [f"Matched-cost comparison: **{pair['status']}**; decision for the illustrative pair: **{pair['decision']}**."]
        if pair['A'] and pair['B']:
            lines += [table([flatten(pair[t],mode) for t in ('A','B')],
                            ['model','threshold','wh_precision','wh_recall','wh_FP','wh_fp_per_image','overall_FP','overall_fp_per_image'])]
        lines += ['Matched pairs use only points satisfying that policy and within one FP on BOTH costs. Selection maximizes the smaller recall, then larger recall, then minimizes combined overall/target FPs; final ties use higher A/B thresholds. It illustrates matched realized costs and does not replace each model\'s independent policy optimum.',
                  f"Across distinct feasible detection-count states, all comparable-pair decision counts: {pair['pair_counts']}. These counts are not statistical votes; pairs are correlated and may include dominated points. Complete pair summary: [same_cost_comparisons.json](same_cost_comparisons.json)."]
    lines += ['Historical counts refer exclusively to the original 50 GT identities; recovered + wrong class + still missed = 50. Class-confusion count covers all GT at the selected point; Without Helmet confusion covers all target GT, not just the historical misses. A confusion contributes one GT-class FN and one predicted-class FP. Full exact thresholds and metrics: [policy_results.csv](policy_results.csv) / [policy_results.json](policy_results.json). Rounded thresholds above are display only.',
              '## Constrained Pareto analysis',
              'A point dominates another only if its target recall and precision are both at least as high and its overall FP/image at most as high, with at least one strict improvement. No weighted/scalar score is used. Filter separately for each model after applying each policy. ALL_TESTED is an additional unconstrained reference within the saved sweep, not a fourth policy.',
              'Identical objective vectors count once; the highest tested threshold represents that vector. `equivalent_threshold_count` records the number of feasible thresholds represented. Counts therefore measure distinct objective vectors, not confidence-grid density.',
              table([dict(policy=m,**{t:sum(r['model']==t and r['policy']==m for r in frontiers) for t in ('A','B')})
                     for m in (*MODES,'ALL_TESTED')],['policy','A','B']),
              '[All non-dominated points](pareto_operating_points.csv). These within-model frontiers do not by themselves establish cross-model superiority.',
              '## Calibration observation and ranking',
              f"A best target F1 confidence is {prior['best_f1']['A']['Without Helmet']['threshold']:.9f} (approximately 0.263); B is {prior['best_f1']['B']['Without Helmet']['threshold']:.9f} (approximately 0.066). B exhibits a different score distribution / confidence-calibration behavior. Raw confidence values across models must not be treated as directly calibrated probabilities. No formal calibration metric was evaluated, so no claim about calibration quality is made. No temperature scaling or other calibration change was performed.",
              table([r for r in ap if r['metric'] in ('mAP50','mAP50-95')],['scope','metric','A','B','delta']),
              'AP is inherited from Phase 4F. B\'s lower overall and Without Helmet AP remains a ranking concern; isolated recall gains or lower thresholds do not justify replacing A.',
              '## Remaining failure mechanism and Experiment C gate',
              table([dict(model=t,**persistent[t]) for t in ('A','B')],['model','no_correct_class_proposal','wrong_class_overlap']),
              'For each target GT, inspect every saved post-NMS prediction down to the original 0.001 floor. The first column counts objects with no correct-class box at IoU >=0.5; the second is the subset having at least one wrong-class overlap. Each GT is counted once. These are proposal-level failure indicators, not the one-to-one confusion counts in the policy tables. [Object-level evidence](remaining_failure_evidence.json) preserves identities.',
              'Changing confidence within the saved detections cannot recover objects lacking any overlapping correct-class proposal. This is stronger than failure at one selected threshold, but it does not prove TRAIN coverage insufficiency or the cause of error. Predictions below 0.001, pre-NMS proposals, and different inference settings are outside the evidence.']
    if c_ready:
        lines += ['**RECOMMENDED NEXT ACTION: EXPERIMENT_C_DESIGN. READY FOR EXPERIMENT C DESIGN: YES.**',
                  '**Primary remaining problem:** persistent target misses, including wrong-class overlaps, that cannot be recovered by choosing a confidence threshold in the frozen inference output.',
                  '**Exactly one proposed intervention: hard-example TRAIN sampling.** Define difficulty from A\'s errors on TRAIN only using out-of-fold or another leakage-aware TRAIN-only procedure, then change only the sampling distribution from A\'s uniform sampling to error-weighted sampling. Keep architecture, resolution (416), augmentation, optimizer, epochs/early-stopping policy and other settings fixed. This is an error-driven intervention, not another generic class multiplier. Establish TRAIN evidence and freeze the rule before executing at most one controlled C experiment; no TRAIN analysis or training was performed in this phase.',
                  'The observed VALID failures justify designing a targeted experiment; they do not prove this sampling intervention will work. Never add VALID images to TRAIN or derive sampling weights from VALID object identities. No resolution increase or second intervention is proposed.']
    else:
        lines += ['**RECOMMENDED NEXT ACTION: FREEZE_MODEL. READY FOR EXPERIMENT C DESIGN: NO.** Retain A as the candidate and end exploratory threshold search; no production threshold is changed.']
    lines += ['## Validation overuse warning',
              '**VALIDATION OVERUSE WARNING: YES.** The same 127-image validation set has been used for multiple exploratory analyses. Further hyperparameter search risks validation overfitting. After this phase either freeze a candidate configuration or run at most one clearly motivated Experiment C before final TEST evaluation. Do not keep searching budgets or training variants to obtain a favorable validation result. This phase does not authorize unlocking TEST; final TEST evaluation remains a separate later action.',
              '## Reproducibility and completion',
              'Reuses all 1,005 saved Phase 4G thresholds and corresponding class/overall metrics. Saved metric arithmetic, full paired coverage and denominators are verified. Selected-point counts are recomputed from saved predictions using the unchanged matching policy to verify them; this is not model inference. Frozen checkpoint hashes, training artifacts, VALID image/label hashes and historical miss identities are verified by the existing preflight. Every protected input is hashed before and after this phase. TEST files are never enumerated, opened or hashed.',
              'Run `.venv/Scripts/python.exe scripts/analyze_operating_budgets.py` into a new/empty output directory; existing results are protected from overwrite. Tests: `.venv/Scripts/python.exe -m pytest tests/test_operating_budget.py -q`. Reproducibility metadata includes hashes for the policy, input artifacts and analysis code.',
              f"B CONSISTENTLY BETTER UNDER CONTROLLED COST: {'YES' if consistent else 'NO'}. A REMAINS REFERENCE: YES. TEST STATUS: LOCKED. Experiment C trained: NO."]
    (out/'operating_budget_report.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    return summary
