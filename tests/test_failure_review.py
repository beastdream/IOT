import pytest
import yaml

from smart_helmet.evaluation.failure_analysis import analyze_image
from smart_helmet.evaluation.failure_review import (
    aggregate_review, category_mapping, distribution, guarded_image, review_rows,
    run_review, size_bucket, compare_distributions,
)
from smart_helmet.evaluation.experiment_plan import recommend, write_plan


def case(predictions=None):
    return analyze_image('data/curated/v1/valid/images/a.jpg',
                         [dict(class_id=1, bbox=[0, 0, 20, 20])], predictions or [], 100, 100)


def test_low_confidence_and_nearest():
    c = case([dict(class_id=1, bbox=[0, 0, 20, 20], confidence=.1)])
    row = review_rows([c])[0]
    assert row['categories'] == ['LOW_CONFIDENCE']
    assert row['iou'] == 1 and row['confidence'] == .1
    assert row['number_of_active_predictions'] == 0
    assert row['manual_review_required']
    assert row['bbox_width'] == 20


def test_confusion_and_low_confidence_overlap():
    c = case([dict(class_id=0, bbox=[0, 0, 20, 20], confidence=.9),
              dict(class_id=1, bbox=[0, 0, 20, 20], confidence=.1)])
    row = review_rows([c])[0]
    assert set(row['categories']) == {'CLASS_CONFUSION', 'LOW_CONFIDENCE'}
    assert row['nearest_prediction_class'] == 'With Helmet'
    assert aggregate_review([row])['categories']['CLASS_CONFUSION']['percentage'] == 100


def test_no_prediction_unknown_and_empty_aggregation():
    row = review_rows([case()])[0]
    assert row['nearest_prediction_bbox'] is None and row['iou'] is None
    assert row['categories'] == ['UNKNOWN']
    assert aggregate_review([])['objects'] == 0
    with pytest.raises(ValueError, match='Duplicate'):
        aggregate_review([row, row])


def test_distribution_and_scene_mapping():
    c = analyze_image('x', [dict(class_id=0, bbox=[0, 0, 30, 30]),
                            dict(class_id=1, bbox=[0, 0, 5, 5]),
                            dict(class_id=1, bbox=[20, 20, 60, 60])], [], 100, 100)
    d = distribution([c, case()])
    assert d['image_composition'] == {'both': 1, 'without_helmet_only': 1}
    assert d['objects'] == {'With Helmet': 1, 'Without Helmet': 3}
    assert d['without_helmet_sizes'] == {'tiny': 0, 'small': 1, 'medium': 1, 'large': 1}
    assert d['without_helmet_scene_objects'] == {'single': 1, 'two': 0, 'three_or_more': 2}
    assert category_mapping(c, 1) == ['SMALL_OBJECT', 'CROWDED_SCENE']
    assert not any(tag.endswith('CANDIDATE') for tag in category_mapping(c, 1))


@pytest.mark.parametrize('a,expected', [(0.0009, 'tiny'), (.001, 'small'), (.01, 'medium'), (.1, 'large')])
def test_size_boundaries(a, expected):
    assert size_bucket(a) == expected


@pytest.mark.parametrize('image', ['data/curated/v1/test/images/a.jpg',
                                  'data/curated/v1/train/images/a.jpg',
                                  '../test/images/a.jpg'])
def test_validation_path_guard(tmp_path, image):
    with pytest.raises(ValueError):
        guarded_image(tmp_path, image)


@pytest.mark.parametrize('split,status', [('test', 'LOCKED / NOT USED'), ('train', 'LOCKED / NOT USED'), ('val', 'UNLOCKED')])
def test_review_guards_before_io(tmp_path, split, status):
    with pytest.raises(ValueError):
        run_review(tmp_path, split, status)


def fixtures():
    summary = aggregate_review(review_rows([case()]))
    summary.update(status='PASS', test_used=False, test_status='LOCKED / NOT USED',
                   train_distribution=dict(images=10, image_composition={'without_helmet_only': 2, 'with_helmet_only': 8},
                                           objects={'With Helmet': 12, 'Without Helmet': 3}),
                   metrics={'per_class': {'1': {'recall': .5304}}})
    baseline = dict(model='yolo11n.pt', imgsz=416, batch=4, workers=0, seed=42, epochs=100, patience=20)
    return summary, baseline


def test_recommendation_and_serialization(tmp_path):
    summary, baseline = fixtures()
    plan, options = recommend(summary, baseline)
    assert len(options) == 6
    assert all({'evidence_for', 'evidence_against', 'cost', 'expected_target_metric'} <= o.keys() for o in options)
    assert plan['constants_from_baseline'] == baseline
    assert plan['proposed_change']['positive_image_weight'] == 4
    assert plan['higher_resolution_justified'] is False
    assert plan['training_authorized'] is False
    path = tmp_path/'B_plan.yaml'; write_plan(path, plan)
    assert yaml.safe_load(path.read_text()) == plan
    from smart_helmet.training.trainer import load_config
    with pytest.raises(ValueError, match='Baseline config'):
        load_config(path)
    plan['training_authorized'] = True
    with pytest.raises(ValueError):
        write_plan(path, plan)


def test_recommendation_does_not_force_sampling_without_imbalance():
    summary, baseline = fixtures()
    summary['train_distribution']['image_composition'] = {'without_helmet_only': 8, 'with_helmet_only': 2}
    plan, _ = recommend(summary, baseline)
    assert plan['proposed_change']['factor_group'] == 'none'


@pytest.mark.parametrize('key,value', [('test_used', True), ('test_status', 'UNLOCKED'), ('status', 'FAIL')])
def test_plan_test_lock_guard(key, value):
    summary, baseline = fixtures(); summary[key] = value
    with pytest.raises(ValueError):
        recommend(summary, baseline)


def test_comparison_uses_validation_denominators():
    c = case()
    s = aggregate_review(review_rows([c]))
    s.update(train_distribution=distribution([c]), validation_distribution=distribution([c, c]))
    comparison = compare_distributions(s)
    assert comparison['size']['medium']['validation_miss_rate'] == .5
    assert comparison['size']['tiny']['validation_miss_rate'] is None
    assert comparison['scene']['single']['train_percentage'] == 100
