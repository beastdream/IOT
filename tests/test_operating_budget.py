"""Synthetic operating-policy tests: no inference or model training."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock
import pytest
import yaml

from smart_helmet.evaluation import operating_budget as b
from smart_helmet.evaluation.operating_point import metrics


def point(tag='A', threshold=.2, tp=5, fp=1, overall_fp=2):
    return dict(model=tag,threshold=threshold,
                **{'Without Helmet':metrics(tp,fp,10-tp,10),
                   'With Helmet':metrics(5,overall_fp-fp,5,10),
                   'Overall':metrics(tp+5,overall_fp,15-tp,10)})


def constraints(precision=.5, wh=.5, overall=.5):
    return dict(zip(b.CONSTRAINTS,[precision,wh,overall]))


def test_policy_parsing(tmp_path):
    source=Path(__file__).parents[1]/'configs/evaluation/operating_policy.yaml'
    p=b.parse_policy(source)
    assert tuple(p['modes']) == b.MODES
    for change in ('missing','nan','negative','string','precision','test','production','mode'):
        bad=deepcopy(p)
        c=bad['modes']['STRICT']['constraints']
        if change=='missing': c.pop(b.CONSTRAINTS[0])
        if change=='nan': c[b.CONSTRAINTS[1]]=float('nan')
        if change=='negative': c[b.CONSTRAINTS[1]]=-1
        if change=='string': c[b.CONSTRAINTS[1]]='0.1'
        if change=='precision': c[b.CONSTRAINTS[0]]=1.1
        if change=='test': bad['split']='test'
        if change=='production': bad['production_policy']=True
        if change=='mode': bad['modes'].pop('STRICT')
        path=tmp_path/'policy.yaml';path.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError): b.parse_policy(path)


def test_constraints_inclusive_and_independent():
    p=point(tp=5,fp=5,overall_fp=5)
    assert b.feasible(p,constraints())
    assert not b.feasible(p,constraints(precision=.5001))
    assert not b.feasible(p,constraints(wh=.4999))
    assert not b.feasible(p,constraints(overall=.4999))
    assert not b.feasible(point(tp=0,fp=0),constraints(precision=0))


def test_no_feasible_and_no_relaxation():
    c=constraints(precision=1,wh=0,overall=0)
    original=deepcopy(c)
    assert b.select([point()],c) is None
    assert c==original
    assert b.compare(None,None)=='NO_FEASIBLE_POINT'
    assert b.compare(point(),None)=='A_PREFERRED'
    assert b.compare(None,point('B'))=='B_PREFERRED_UNDER_THIS_BUDGET'


def test_selection_recall_then_deterministic_tie():
    a=point(threshold=.1,tp=7,fp=2,overall_fp=3)
    better=point(threshold=.2,tp=7,fp=1,overall_fp=2)
    tied=point(threshold=.3,tp=7,fp=1,overall_fp=2)
    invalid=point(threshold=.05,tp=9,fp=9,overall_fp=9)
    for rows in ([a,better,tied,invalid],[invalid,tied,better,a]):
        assert b.select(rows,constraints())==tied


def test_both_costs_required_and_precision_safeguard():
    a=point(tp=5,fp=1,overall_fp=2)
    bb=point('B',tp=6,fp=1,overall_fp=4)
    assert not b.same_cost(a,bb)
    assert b.compare(a,bb)=='NO_CLEAR_ADVANTAGE'
    bb=point('B',tp=6,fp=1,overall_fp=3)
    assert b.same_cost(a,bb)
    assert b.compare(a,bb)=='B_PREFERRED_UNDER_THIS_BUDGET'
    bb=point('B',tp=6,fp=2,overall_fp=3)
    assert b.compare(a,bb)=='NO_CLEAR_ADVANTAGE'
    assert not b.same_cost(a,point('B',tp=6,fp=3,overall_fp=3))


def test_same_cost_pair_determinism_and_no_pair():
    a=[point(),point(threshold=.3,tp=6)]
    bb=[point('B',tp=7),point('B',threshold=.4,tp=7)]
    result=b.same_cost_comparison(a,bb)
    assert result==b.same_cost_comparison(list(reversed(a)),list(reversed(bb)))
    assert result['A']['threshold']==.3 and result['B']['threshold']==.4
    assert b.same_cost_comparison(a,[point('B',overall_fp=9)])['status']=='NO_COMPARABLE_PAIR'


def test_pareto_strict_dominance_tradeoff_duplicates():
    a=point(threshold=.1,tp=5,fp=1,overall_fp=3)
    dominated=point(threshold=.2,tp=4,fp=1,overall_fp=4)
    tradeoff=point(threshold=.3,tp=7,fp=3,overall_fp=5)
    duplicate=point(threshold=.4,tp=5,fp=1,overall_fp=3)
    result=b.pareto([a,dominated,tradeoff,duplicate])
    assert {p['threshold'] for p in result}=={.3,.4}
    assert next(p for p in result if p['threshold']==.4)['equivalent_threshold_count']==2
    assert b.pareto([])==[]


@pytest.mark.parametrize('split',['test','TEST','train','../test'])
def test_validation_guard_before_io(tmp_path,monkeypatch,split):
    spy=Mock(side_effect=AssertionError('Unexpected IO'))
    monkeypatch.setattr(b,'parse_policy',spy)
    with pytest.raises(ValueError,match='LOCKED'): b.run(tmp_path,split)
    spy.assert_not_called()


def test_join_metrics_completeness_arithmetic_and_duplicates():
    rows=[dict(model=t,threshold=.2,scope=s,**point(t)[s]) for t in ('A','B') for s in b.op.SCOPES]
    assert len(b.join_metrics(rows,[.2],10))==2
    for bad in [rows[:-1],rows+[rows[0]], [dict(rows[0],recall=.99),*rows[1:]]]:
        with pytest.raises(ValueError): b.join_metrics(bad,[.2],10)


def test_diagnostics_confusion_and_historical_ids():
    cases=[dict(image='v.jpg',ground_truth=[dict(class_id=1,bbox=[0,0,10,10])],
                predictions=[dict(class_id=0,bbox=[0,0,10,10],confidence=.8)])]
    diag,counts=b.diagnostics(cases,.25,{('v.jpg',0)})
    assert diag['class_confusion_count']==diag['historical_wrong_class']==1
    assert counts=={0:[0,1,0],1:[0,0,1]}


def test_constrained_pareto_excludes_high_recall_infeasible_point():
    pool=[point(tp=5),point(tp=9,fp=6,overall_fp=7)]
    allowed=[p for p in pool if b.feasible(p,constraints())]
    assert len(b.pareto(allowed))==1
    assert b.pareto(allowed)[0]['Without Helmet']['TP']==5


def test_same_cost_inclusive_count_boundary():
    a=point(tp=5,fp=1,overall_fp=2)
    assert b.same_cost(a,point('B',fp=2,overall_fp=3))
    assert not b.same_cost(a,point('B',fp=2,overall_fp=4))


def test_report_handles_no_feasible_and_freeze_action(tmp_path):
    from smart_helmet.evaluation.operating_budget_report import write_report
    policy=b.parse_policy(Path(__file__).parents[1]/'configs/evaluation/operating_policy.yaml')
    results={m:dict(A=None,B=None,decision='NO_FEASIBLE_POINT',optima_same_cost=False,
                   feasible_counts={'A':0,'B':0}) for m in b.MODES}
    rows=[b.flatten(dict(model=t),m,'NO_FEASIBLE_OPERATING_POINT') for m in b.MODES for t in ('A','B')]
    pairs={m:b.same_cost_comparison([],[]) for m in b.MODES}
    prior=dict(best_f1={t:{'Without Helmet':{'threshold':v}} for t,v in [('A',.263),('B',.066)]})
    summary=write_report(tmp_path,policy,results,rows,[],pairs,[],prior,[])
    assert summary['recommended_next_action']=='FREEZE_MODEL'
    text=(tmp_path/'operating_budget_report.md').read_text(encoding='utf-8')
    assert 'NO_FEASIBLE_OPERATING_POINT' in text and 'NO_FEASIBLE_POINT' in text
    assert 'TEST STATUS: LOCKED' in text
