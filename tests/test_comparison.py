"""Synthetic Phase 4F checks: no training or model inference."""
from copy import deepcopy
import hashlib
from unittest.mock import Mock
import pytest
from smart_helmet.evaluation import comparison as c
from smart_helmet.evaluation.failure_analysis import analyze_image


def box(cls=1, confidence=None, bbox=(0, 0, 10, 10)):
    result = dict(class_id=cls, bbox=list(bbox))
    if confidence is not None:
        result['confidence'] = confidence
    return result


def test_metric_deltas_and_scope_mapping():
    a = dict(zip(c.KEYS, [.8, .5, .7, .4])); b = dict(zip(c.KEYS, [.7, .6, .65, .38]))
    assert c.metric_delta(a, b) == pytest.approx(dict(zip(c.KEYS, [-.1, .1, -.05, -.02])))
    metrics = {tag: dict(overall=v, per_class={'0': v, '1': v}) for tag, v in [('A', a), ('B', b)]}
    rows = c.metric_comparison(metrics)
    assert len(rows) == 12 and {r['scope'] for r in rows} == {'Overall', 'With Helmet', 'Without Helmet'}
    decision = c.selection(rows)
    assert decision['primary_target_improved']
    assert decision['precision_safeguard'] == decision['overall_map_safeguard'] == 'FAIL'
    assert decision['model_decision'] == 'A_REMAINS_REFERENCE'


@pytest.mark.parametrize('a,b,expected', [
    ('MISSED','CORRECT','RECOVERED_BY_B'),
    ('CLASS_CONFUSION','CORRECT','A_CLASS_CONFUSION_B_CORRECT'),
    ('MISSED','CLASS_CONFUSION','A_MISSED_B_CLASS_CONFUSION'),
    ('MISSED','MISSED','STILL_MISSED_BY_B'),
    ('CLASS_CONFUSION','MISSED','STILL_MISSED_BY_B'),
    ('CLASS_CONFUSION','CLASS_CONFUSION','STILL_MISSED_BY_B'),
    ('CORRECT','MISSED','CORRECT_IN_A_MISSED_IN_B'),
    ('CORRECT','CLASS_CONFUSION','CORRECT_IN_A_WRONG_IN_B'),
    ('CORRECT','CORRECT','CORRECT_IN_BOTH')])
def test_exclusive_object_transitions(a,b,expected):
    assert c.transition(a,b) == expected


def test_recovery_and_prediction_not_reused():
    case = analyze_image('v.jpg', [box(), box()], [box(confidence=.8)], 100, 100)
    outcomes, fps = c.object_outcomes(case)
    assert [r['outcome'] for r in outcomes] == ['CORRECT', 'MISSED']
    assert not fps
    wrong = analyze_image('v.jpg', [box()], [box(cls=0, confidence=.8)], 100, 100)
    outcomes, fps = c.object_outcomes(wrong)
    assert outcomes[0]['outcome'] == 'CLASS_CONFUSION' and len(fps) == 1
    assert outcomes[0]['iou'] == 1
    low = analyze_image('v.jpg', [box()], [box(confidence=.24)], 100, 100)
    assert c.object_outcomes(low)[0][0]['outcome'] == 'MISSED'


def test_new_and_removed_false_positives_one_to_one():
    a = [box(confidence=.8), box(cls=0, confidence=.9)]
    b = [box(confidence=.7), box(confidence=.6)]
    rows = c.false_positive_transitions(a,b)
    assert [r['category'] for r in rows] == ['PERSISTENT_FALSE_POSITIVE', 'NEW_FALSE_POSITIVE_IN_B', 'REMOVED_FALSE_POSITIVE_IN_B']
    assert sum(r['A'] is not None for r in rows) == len(a)
    assert sum(r['B'] is not None for r in rows) == len(b)


def test_same_validation_and_gt_enforcement():
    a = [analyze_image('v.jpg', [box()], [], 100, 100)]
    c.ensure_same_validation(a, deepcopy(a), ['v.jpg'])
    for b in [[], a+a, [dict(a[0], image='test.jpg')], [dict(a[0], ground_truth=[])], [dict(a[0], width=200)]]:
        with pytest.raises(ValueError, match='Same VALID'):
            c.ensure_same_validation(a,b,['v.jpg'])


@pytest.mark.parametrize('split', ['test','train','TEST'])
def test_test_lock_before_any_io(tmp_path, monkeypatch, split):
    verify = Mock(); monkeypatch.setattr(c, 'verify_dataset', verify)
    with pytest.raises(ValueError, match='LOCKED'):
        c.run(tmp_path, split=split)
    verify.assert_not_called()


def test_checkpoint_fingerprint(tmp_path):
    path = tmp_path/'best.pt'
    with pytest.raises(ValueError, match='best.pt'):
        c.checkpoint_fingerprint(path)
    path.write_bytes(b'')
    with pytest.raises(ValueError):
        c.checkpoint_fingerprint(path)
    path.write_bytes(b'checkpoint')
    before = c.checkpoint_fingerprint(path)
    assert before == hashlib.sha256(b'checkpoint').hexdigest()
    path.write_bytes(b'changed')
    assert before != c.checkpoint_fingerprint(path)


def paired_fixture():
    gt = [box(bbox=(i*20, 0, i*20+10, 10)) for i in range(50)]
    a = analyze_image('v.jpg', gt, [], 1100, 100)
    b = analyze_image('v.jpg', gt, [box(confidence=.8)], 1100, 100)
    human = [dict(image='v.jpg', gt_object_id=i, reviewed='true', human_categories='ANNOTATION_OK', notes='') for i in range(50)]
    return a,b,human


def test_all_50_recovery_identity_and_human_categories():
    a,b,human = paired_fixture()
    objects, recovery, fps, summary = c.paired_analysis({'A':[a], 'B':[b]}, ['v.jpg'], [a], human)
    assert len(objects) == len(recovery) == 50
    assert summary['recovered'] == 1 and summary['still_not_correct'] == 49
    assert summary['human_categories'] == {'ANNOTATION_OK': {'total':50, 'recovered':1}}
    assert not fps
    with pytest.raises(ValueError, match='Human review'):
        c.paired_analysis({'A':[a], 'B':[b]}, ['v.jpg'], [a], human[:-1])
    with pytest.raises(ValueError, match='historical'):
        c.paired_analysis({'A':[b], 'B':[b]}, ['v.jpg'], [a], human)


def test_report_generation(tmp_path):
    a,b,human = paired_fixture()
    summary = c.paired_analysis({'A':[a], 'B':[b]}, ['v.jpg'], [a], human)[3]
    v = dict(zip(c.KEYS, [.8,.5,.7,.4]))
    rows = c.metric_comparison({tag:dict(overall=v,per_class={'0':v,'1':v}) for tag in ('A','B')})
    losses = {k: 1.0 for k in ('train/box_loss','train/cls_loss','train/dfl_loss','val/box_loss','val/cls_loss','val/dfl_loss')}
    training = {tag:dict(best_epoch=1,stopped_epoch=2,training_seconds=100,pipeline_seconds=120,
                         best_losses=losses, final_losses=losses) for tag in ('A','B')}
    text = c.report_text(rows,summary,training,c.selection(rows),dict(validation_images=127,checkpoint_sha256={'A':'a','B':'b'}))
    path = tmp_path/'A_vs_B_report.md'; path.write_text(text)
    for expected in ['Without Helmet', 'False-positive cost', 'Training comparison', 'A_REMAINS_REFERENCE',
                     'confidence/operating-point', 'LOCKED / NOT USED', 'without_helmet_recovery.csv']:
        assert expected in path.read_text()
