"""Synthetic Phase 4G checks; no model loading, inference or training."""
from copy import deepcopy
import hashlib
from unittest.mock import Mock

import pytest

from smart_helmet.evaluation import operating_point as op
from smart_helmet.evaluation.failure_analysis import iou


def box(cls=1, conf=None, bbox=(0, 0, 10, 10)):
    b = dict(class_id=cls, bbox=list(bbox))
    if conf is not None:
        b['confidence'] = conf
    return b


def case(gt, pred):
    return dict(image='valid.jpg', ground_truth=gt, predictions=pred)


def point(tag, threshold, tp, fp, fn, scope='Without Helmet'):
    return dict(model=tag, threshold=threshold, scope=scope, **op.metrics(tp, fp, fn, 10))


def test_metrics_known_counts_and_zero():
    m = op.metrics(3, 1, 2, 2)
    assert m == pytest.approx(dict(TP=3, FP=1, FN=2, precision=.75, recall=.6, F1=2/3, fp_per_image=.5))
    assert op.metrics(0, 0, 2, 10)['precision'] == 0
    assert op.metrics(0, 0, 0, 10)['F1'] == 0
    with pytest.raises(ValueError):
        op.metrics(0, 0, 0, 0)


def test_iou_boundary_and_duplicate_matching():
    gt = box()
    assert iou(gt['bbox'], [0, 0, 20, 10]) == .5
    counts, states = op.outcomes(case([gt], [box(conf=.8, bbox=(0, 0, 20, 10)), box(conf=.7)]), .1)
    assert counts[1] == (1, 1, 0) and states == ['CORRECT']
    counts, states = op.outcomes(case([gt], [box(conf=.8, bbox=(0, 0, 20.01, 10))]), .1)
    assert counts[1] == (0, 1, 1) and states == ['MISSED']
    counts, states = op.outcomes(case([gt, gt], [box(conf=.8)]), .1)
    assert counts[1] == (1, 0, 1) and states == ['CORRECT', 'MISSED']


def test_wrong_class_and_correct_class_priority():
    c = case([box()], [box(cls=0, conf=.9)])
    counts, states = op.outcomes(c, .25)
    assert states == ['CLASS_CONFUSION']
    assert counts == {0: (0, 1, 0), 1: (0, 0, 1)}
    c['predictions'].append(box(conf=.4))
    assert op.outcomes(c, .25)[1] == ['CORRECT']


def test_sweep_boundary_historical_exclusivity_and_micro_overall():
    c = case([box(), box(bbox=(20, 0, 30, 10)), box(cls=0, bbox=(40, 0, 50, 10))],
             [box(conf=.1), box(cls=0, conf=.5, bbox=(20, 0, 30, 10))])
    cached = deepcopy(c)
    rows, rec, objects = op.sweep({'A': [c], 'B': [c]}, {('valid.jpg', 0), ('valid.jpg', 1)}, [.1, .15])
    assert len(rows) == 12 and len(objects) == 8
    assert c == cached
    assert rec[0] == dict(model='A', threshold=.1, recovered=1, still_missed=0, wrong_class=1, still_not_correct=1)
    assert rec[1]['recovered'] == 0 and rec[1]['still_missed'] == 1
    overall = rows[0]
    assert (overall['TP'], overall['FP'], overall['FN']) == (1, 1, 2)
    assert overall['precision'] == .5 and overall['recall'] == 1/3
    assert all(r['recovered']+r['still_missed']+r['wrong_class'] == 2 for r in rec)
    assert op.THRESHOLDS == tuple(i/20 for i in range(1, 19))


@pytest.mark.parametrize('thresholds', [[.1, .1], [0], [float('nan')], [1.1], []])
def test_invalid_sweep(thresholds):
    with pytest.raises(ValueError, match='threshold'):
        op.sweep({'A': [], 'B': []}, set(), thresholds)


def test_precision_floor_max_recall_inclusive_and_unattainable():
    rows = [point('A', .1, 6, 4, 4), point('A', .2, 4, 1, 6),
            point('A', .3, 3, 0, 7), point('B', .2, 7, 3, 3)]
    result = op.precision_floor(rows, .8)
    assert result['A']['threshold'] == .2 and result['B'] is None
    assert op.precision_floor([point('A', .9, 0, 0, 10)], 0)['A'] is None


def test_budget_upper_bound_closest_tie_and_slack():
    rows = [point('A', .1, 6, 4, 4), point('A', .2, 4, 1, 6),
            point('B', .1, 7, 5, 3), point('B', .2, 5, 3, 5)]
    result = op.budget_comparison(rows, 'Without Helmet')
    budget4 = next(r for r in result['frontier'] if r['fp_budget'] == 4)
    assert budget4['A']['recall'] == .6 and budget4['B']['recall'] == .5
    assert budget4['slack'] == {'A': 0, 'B': 1}
    assert result['frontier'][0]['B'] is None
    assert result['closest_pairs'][0]['B']['FP'] == 5
    assert result['closest_pairs'][0]['fp_per_image_gap'] == pytest.approx(.1)


@pytest.mark.parametrize('split', ['test', 'TEST', 'train', '../test'])
def test_test_lock_before_io(tmp_path, monkeypatch, split):
    spy = Mock(side_effect=AssertionError('Filesystem touched'))
    monkeypatch.setattr(op.Path, 'read_text', spy)
    with pytest.raises(ValueError, match='LOCKED'):
        op.run(tmp_path, split=split)
    spy.assert_not_called()


def test_validation_path_guard(tmp_path):
    for path in ['test/images/a.jpg', 'train/images/a.jpg', 'data/curated/v1/test/images/a.jpg', '../a.jpg']:
        with pytest.raises(ValueError, match='allowed split'):
            op.guarded_image(tmp_path, path)
    assert op.guarded_image(tmp_path, 'data/curated/v1/valid/images/a.jpg').name == 'a.jpg'


def test_checkpoint_provenance(tmp_path):
    path = tmp_path/'best.pt'
    with pytest.raises(ValueError):
        op.verify_checkpoint(path, 'bad')
    path.write_bytes(b'frozen')
    expected = hashlib.sha256(b'frozen').hexdigest()
    assert op.verify_checkpoint(path, expected) == expected
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='provenance'):
        op.verify_checkpoint(path, expected)


def test_source_guard():
    p = dict(test_status='LOCKED / NOT USED', validation_images=127,
             common_evaluation_args=dict(imgsz=416, iou=.7, max_det=300, augment=False,
                                         quantize=32, conf=.001, device='0'))
    status = dict(status='PASS', test_status='LOCKED / NOT USED')
    op.source_guard(p, status)
    for key, value in [('imgsz', 640), ('quantize', 16), ('conf', .25), ('augment', True)]:
        bad = deepcopy(p); bad['common_evaluation_args'][key] = value
        with pytest.raises(ValueError, match='settings'):
            op.source_guard(bad, status)
    with pytest.raises(ValueError, match='LOCKED'):
        op.source_guard(p, dict(status, test_status='UNLOCKED'))


def test_refinement_covers_every_cached_prediction_state():
    cases = {'A': [case([box()], [box(conf=.05123), box(conf=.61234), box(conf=.91)])],
             'B': [case([box()], [box(conf=.05321), box(conf=.02)])]}
    thresholds = op.refined_thresholds(cases)
    assert set(op.THRESHOLDS) <= set(thresholds)
    assert {.05123, .61234, .05321} <= set(thresholds)
    assert min(thresholds) == .05 and max(thresholds) == .9
    assert len(thresholds) == 89
    assert op.outcomes(cases['A'][0], .61234)[0][1] == (1, 1, 0)


def test_best_f1_tie_deterministic():
    rows = [point('A', .2, 6, 4, 4), point('A', .3, 6, 4, 4)]
    assert op.best_point(rows)['threshold'] == .3
