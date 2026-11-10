import csv
import io
import json
from pathlib import Path

import pytest

from smart_helmet.evaluation.human_review import (
    FIELDS, ReviewStore, atomic_text, parse_choice, readiness, summarize,
)
from smart_helmet.training.sampling import (
    TrainExposureSampler, build_plan, exposure, records, train_path, weights,
)


@pytest.fixture
def inventory(tmp_path):
    base = tmp_path/'data/curated/v1/train'
    (base/'images').mkdir(parents=True)
    (base/'labels').mkdir()
    for name, labels in [('a','0 .5 .5 .2 .2\n'), ('b','1 .5 .5 .2 .2\n'),
                         ('c','0 .5 .5 .2 .2\n1 .5 .5 .2 .2\n')]:
        (base/'images'/f'{name}.jpg').write_bytes(b'fixture')
        (base/'labels'/f'{name}.txt').write_text(labels)
    return tmp_path, records(tmp_path)


def test_sampling_reproducibility_and_epoch_resume(inventory):
    root, rows = inventory
    first = TrainExposureSampler(root, rows)
    second = TrainExposureSampler(root, rows)
    sequence = [list(first) for _ in range(10)]
    assert sequence == [list(second) for _ in range(10)]
    second.set_epoch(4)
    assert list(second) == sequence[4]
    other = TrainExposureSampler(root, rows, seed=99)
    assert sequence != [list(other) for _ in range(10)]


def test_weighting_counts_and_expected_exposure(inventory):
    root, rows = inventory
    assert weights(rows) == [1, 1.573, 1.573]
    assert exposure(rows) == {'With Helmet':2, 'Without Helmet':2}
    plan = build_plan(root, rows)
    assert plan['effective_train_samples'] == 3
    assert plan['estimated_class_exposure_after_sampling']['Without Helmet'] == pytest.approx(3*2*1.573/(1+2*1.573))
    sampler = TrainExposureSampler(root, rows)
    draws = [i for _ in range(10000) for i in sampler]
    assert draws.count(1)/draws.count(0) == pytest.approx(1.573, rel=.06)


def test_no_image_duplication_or_mutation(inventory):
    root, rows = inventory
    before = {p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    build_plan(root, rows)
    assert before == {p:p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('split', ['valid', 'test', '../../../valid'])
def test_leakage_fails(inventory, split):
    root, rows = inventory
    bad = f'data/curated/v1/{split}/images/a.jpg'
    with pytest.raises(ValueError, match='leakage'):
        train_path(root, bad)
    with pytest.raises(ValueError, match='leakage'):
        TrainExposureSampler(root, [dict(image=bad, counts=[0,1])])


@pytest.mark.parametrize('multiplier', [0, -1, float('nan'), float('inf')])
def test_invalid_multiplier(inventory, multiplier):
    with pytest.raises(ValueError):
        weights(inventory[1], multiplier)


@pytest.mark.parametrize('choice', ['', 'U,1', '8,9', '1,1', 'INVALID', 'S,1'])
def test_invalid_review_choices(choice):
    with pytest.raises(ValueError):
        parse_choice(choice)


def cases(n=50):
    return [dict(review_id=str(i), image=f'valid/{i}.jpg', ground_truth_index=str(i)) for i in range(n)]


def test_store_resume_backup_and_unknown_completion(tmp_path):
    path = tmp_path/'human_review.csv'
    store = ReviewStore(path, cases())
    store.save()
    original = path.read_bytes()
    store.confirm('0', ['UNKNOWN'], 'comma, newline\nUnicode: mũ')
    store.confirm('1', ['OCCLUSION','ANNOTATION_OK'])
    backups = list(tmp_path.glob('*.bak'))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
    resumed = ReviewStore(path, cases())
    assert resumed.rows['0']['notes'] == 'comma, newline\nUnicode: mũ'
    summary = summarize(resumed)
    assert (summary['reviewed'], summary['pending']) == (2,48)
    assert summary['categories']['UNKNOWN']['count'] == 1
    assert not summary['ready_for_experiment_b_design']
    for row in resumed.rows.values():
        row.update(reviewed='true', human_categories='UNKNOWN')
    assert summarize(resumed)['ready_for_experiment_b_design']


def test_skip_does_not_complete(tmp_path):
    store = ReviewStore(tmp_path/'human.csv', cases())
    assert parse_choice('s') == 'S'
    assert summarize(store)['reviewed'] == 0
    assert parse_choice('1, 8') == ['OCCLUSION','ANNOTATION_OK']


def test_concurrent_edit_refused(tmp_path):
    path = tmp_path/'human.csv'
    store = ReviewStore(path, cases())
    store.save()
    other = ReviewStore(path, cases())
    store.confirm('0', ['UNKNOWN'])
    with pytest.raises(ValueError, match='another session'):
        other.confirm('1', ['BLUR'])


@pytest.mark.parametrize('mutation', ['duplicate', 'identity', 'category', 'boolean'])
def test_corrupt_review_refused(tmp_path, mutation):
    path = tmp_path/'human.csv'
    store = ReviewStore(path, cases())
    rows = list(store.rows.values())
    if mutation == 'duplicate': rows.append(rows[0])
    if mutation == 'identity': rows[0]['image'] = 'test/a.jpg'
    if mutation == 'category': rows[0].update(reviewed='true', human_categories='AUTOMATIC')
    if mutation == 'boolean': rows[0]['reviewed'] = 'yes'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        ReviewStore(path, cases())


def test_atomic_failure_keeps_previous_file(tmp_path, monkeypatch):
    path = tmp_path/'review.csv'
    path.write_text('before')
    def fail(*args):
        raise OSError('simulated replace failure')
    monkeypatch.setattr('smart_helmet.evaluation.human_review.os.replace', fail)
    with pytest.raises(OSError):
        atomic_text(path, 'after')
    assert path.read_text() == 'before'
    assert not list(tmp_path.glob('*.tmp'))


def test_readiness_uses_human_evidence(tmp_path):
    store = ReviewStore(tmp_path/'human.csv', cases())
    auto = dict(categories={'LOW_CONFIDENCE':{'count':30},'CLASS_CONFUSION':{'count':18}},
                train_distribution={'objects':{'Without Helmet':1038,'With Helmet':2009}})
    assert readiness(summarize(store), auto)[0] == 'REVIEW_REQUIRED'
    for row in store.rows.values():
        row.update(reviewed='true', human_categories='ANNOTATION_OK')
    assert readiness(summarize(store), auto)[0] == 'YES'
    store.rows['0']['human_categories'] = 'ANNOTATION_SUSPICIOUS'
    assert readiness(summarize(store), auto)[0] == 'REVIEW_REQUIRED'
    store.rows['0']['human_categories'] = 'UNKNOWN'
    assert readiness(summarize(store), auto)[0] == 'REVIEW_REQUIRED'
    store.rows['0']['human_categories'] = 'BLUR|ANNOTATION_OK'
    assert readiness(summarize(store), auto)[0] == 'REVIEW_REQUIRED'


def test_interactive_review_skip_quit_resume(tmp_path, monkeypatch):
    from smart_helmet.evaluation import human_review as review
    monkeypatch.setattr(review, 'read_cases', lambda root: cases())
    monkeypatch.setattr(review, 'display', lambda case: None)
    monkeypatch.setattr(review, 'reports', lambda root, store: summarize(store))
    answers = iter(['s', 'invalid', 'u', 'human note', 'q'])
    monkeypatch.setattr('builtins.input', lambda prompt: next(answers))
    assert review.main(['--pending-only'], root=tmp_path) == 0
    path = tmp_path/review.EVALUATION/'failure_review/human_review.csv'
    store = ReviewStore(path, cases())
    assert store.rows['0']['reviewed'] == 'false'
    assert store.rows['1']['human_categories'] == 'UNKNOWN'
    assert summarize(store)['reviewed'] == 1
    answers = iter(['8', 'confirmed', 'q'])
    assert review.main(['--pending-only'], root=tmp_path) == 0
    store = ReviewStore(path, cases())
    assert store.rows['0']['human_categories'] == 'ANNOTATION_OK'
    assert store.rows['1']['notes'] == 'human note'
    assert summarize(store)['reviewed'] == 2


def test_individual_visual_uses_existing_images(tmp_path, monkeypatch):
    from PIL import Image
    from smart_helmet.evaluation import human_review as review
    original = tmp_path/'data/curated/v1/valid/images/a.jpg'
    original.parent.mkdir(parents=True)
    Image.new('RGB', (10,10)).save(original)
    gallery = tmp_path/review.EVALUATION/'failures/without_helmet_missed/a.png'
    gallery.parent.mkdir(parents=True)
    Image.new('RGB', (10,10)).save(gallery)
    before = original.read_bytes()
    opened = []
    monkeypatch.setattr(review.webbrowser, 'open', opened.append)
    case = dict(review_id='one', image=original.relative_to(tmp_path).as_posix(),
                ground_truth_index='3', ground_truth_bbox='[1,1,4,4]', nearest_prediction_bbox='[1,1,5,5]')
    review.open_visual(tmp_path, case)
    page = (tmp_path/review.EVALUATION/'failure_review/current_human_review.html').read_text()
    assert 'Target GT #3' in page and '#00ff88' in page and '#ff9933' in page
    assert opened and original.read_bytes() == before
