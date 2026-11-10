"""Audit Phase 4H artifacts against the fixed policy and frozen VALID predictions."""
import csv
import json
from pathlib import Path

from smart_helmet.paths import PROJECT_ROOT
from smart_helmet.dataset.duplicates import sha256_file
from smart_helmet.evaluation import operating_point as op
from smart_helmet.evaluation import operating_budget as budget
from smart_helmet.evaluation.comparison import fingerprint
from smart_helmet.training.experiment import write_json


def main():
    root=Path(PROJECT_ROOT)
    out=root/'results/evaluation/operating_budget_analysis'
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    audit=load(out/'reproducibility.json')
    policy=budget.parse_policy(root/'configs/evaluation/operating_policy.yaml')
    assert audit['policy_sha256']=='2673145792f20c103b545ff4c36f0356c29e9fba136c64041b24bb5d8254ad9f'
    assert sha256_file(out/'operating_policy.yaml')==audit['policy_sha256']
    assert sha256_file(root/'configs/evaluation/operating_policy.yaml')==audit['policy_sha256']
    for name, hashes in audit['protected_input_hashes'].items():
        assert fingerprint(root/name)==hashes, name
    _,_,_,cases,misses=op.preflight(root,'val')
    sweep=root/'results/evaluation/operating_point_analysis'
    saved=load(sweep/'threshold_metrics.json')
    thresholds=load(sweep/'provenance.json')['thresholds']
    reproduced,_,_=op.sweep(cases,misses,thresholds)
    assert reproduced==saved, 'Every saved metric must match frozen predictions'
    points=budget.join_metrics(saved,thresholds)
    results=load(out/'policy_results.json')
    matched=load(out/'same_cost_comparisons.json')
    with (out/'pareto_operating_points.csv').open(newline='',encoding='utf-8-sig') as f:
        frontiers=list(csv.DictReader(f))
    with (out/'policy_results.csv').open(newline='',encoding='utf-8-sig') as f:
        csv_rows=list(csv.DictReader(f))
    assert len(csv_rows)==6
    def eligible(p,c):
        wh=p['Without Helmet']; overall=p['Overall']
        return (wh['TP']+wh['FP']>0 and wh['precision']>=c['minimum_without_helmet_precision']
                and wh['fp_per_image']<=c['maximum_without_helmet_fp_per_image']
                and overall['fp_per_image']<=c['maximum_overall_fp_per_image'])
    for mode in budget.MODES:
        limits=policy['modes'][mode]['constraints']
        for tag in ('A','B'):
            pool=[p for p in points if p['model']==tag and eligible(p,limits)]
            expected=max(pool,key=lambda p:(p['Without Helmet']['recall'],p['Without Helmet']['precision'],
                         -p['Overall']['FP'],-p['Without Helmet']['FP'],p['threshold']),default=None)
            chosen=results[mode][tag]
            assert (chosen is None)==(expected is None)
            if chosen:
                assert all(chosen[k]==v for k,v in expected.items())
                diag,_=budget.diagnostics(cases[tag],chosen['threshold'],misses)
                assert all(chosen[k]==v for k,v in diag.items())
                assert sum(diag[k] for k in ('historical_recovered','historical_wrong_class','historical_still_missed'))==50
                flat=budget.flatten(chosen,mode)
                csv_row=next(r for r in csv_rows if r['model']==tag and r['policy']==mode)
                assert all(csv_row[k]==('' if v is None else str(v)) for k,v in flat.items())
        # Validate the complete deterministic same-cost selection, not just its gaps.
        pools={t:[p for p in points if p['model']==t and eligible(p,limits)] for t in ('A','B')}
        assert matched[mode]==budget.same_cost_comparison(pools['A'],pools['B'])
        assert results[mode]['decision']==budget.compare(results[mode]['A'],results[mode]['B'])
    # Independent set-based dominance audit of every constrained/unconstrained frontier.
    for mode in (*budget.MODES,'ALL_TESTED'):
        for tag in ('A','B'):
            pool=[p for p in points if p['model']==tag and (mode=='ALL_TESTED' or eligible(p,policy['modes'][mode]['constraints']))]
            def vector(p):
                return p['Without Helmet']['recall'],p['Without Helmet']['precision'],-p['Overall']['fp_per_image']
            vectors={vector(p) for p in pool}
            expected={v for v in vectors if not any(all(x>=y for x,y in zip(w,v)) and w!=v for w in vectors)}
            got=[r for r in frontiers if r['policy']==mode and r['model']==tag]
            assert {(float(r['wh_recall']),float(r['wh_precision']),-float(r['overall_fp_per_image'])) for r in got}==expected
            assert len(got)==len(expected)
    residuals=load(out/'remaining_failure_evidence.json')
    expected_ids=set()
    for tag,cs in cases.items():
        for c in cs:
            for i,gt in enumerate(c['ground_truth']):
                if gt['class_id']==1 and not any(p['class_id']==1 and budget.iou(gt['bbox'],p['bbox'])>=.5 for p in c['predictions']):
                    expected_ids.add((tag,c['image'],i))
    assert {(r['model'],r['image'],r['ground_truth_index']) for r in residuals}==expected_ids
    required=['operating_policy.yaml','policy_results.csv','policy_results.json','pareto_operating_points.csv',
              'operating_budget_report.md','reproducibility.json']
    assert all((out/f).stat().st_size>0 for f in required)
    report=(out/'operating_budget_report.md').read_text(encoding='utf-8')
    for phrase in ['VALIDATION / EXPLORATORY','EXPLORATORY VALIDATION BUDGETS','validation overfitting',
                   'No formal calibration metric','TEST STATUS: LOCKED','EXPERIMENT_C_DESIGN','hard-example TRAIN sampling']:
        assert phrase.lower() in report.lower(),phrase
    assert audit['status']=='PASS' and not audit['test_used'] and not audit['inference_executed'] and not audit['training_executed']
    write_json(out/'verification.json',dict(status='PASS',saved_metric_rows_verified=len(saved),
        policy_optima_verified=6,pareto_groups_verified=8,source_hashes_unchanged=True,
        frozen_policy_verified=True,matched_cost_selections_verified=3,residual_objects_verified=len(residuals),
        required_artifacts=required,test_status='LOCKED / NOT USED'))
    print('PASS: all 6030 metrics, 6 policy optima, 8 Pareto groups, 3 matched-cost selections, failure identities and protected hashes verified.')


if __name__=='__main__':
    main()
