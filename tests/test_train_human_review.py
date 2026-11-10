import csv
import json

import pytest

from smart_helmet.evaluation import train_human_review as h


@pytest.fixture
def project(tmp_path):
    out = tmp_path / h.OUT
    out.mkdir(parents=True)
    rows = []
    for i in range(3):
        image = f'data/curated/v1/train/images/{i}.jpg'
        path = tmp_path / image
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'synthetic image')
        rows.append(dict(review_id=str(i), image=image, tier='TIER_2_HARD',
                         automatic_reason='LOW_CONFIDENCE_CORRECT',
                         gt_summary=json.dumps([dict(class_id=1, bbox=[1, 2, 3, 4])]),
                         prediction_summary='[]', review_status='PENDING', notes='original note'))
    with (out / 'manual_review_queue.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path


def test_resume_preserves_reviews_notes_evidence_and_backup(project):
    store = h.ReviewStore(project)
    original = store.original
    evidence = [r.copy() for r in store.rows]
    assert not store.backup.exists()
    store.confirm('0', 'VALID_HARD_CASE', 'small head, occluded\nconfirmed')
    assert store.backup.read_bytes() == original
    completed = store.rows[0].copy()
    resumed = h.ReviewStore(project)
    assert [r['review_id'] for r in resumed.selected(True)] == ['1', '2']
    assert len(resumed.selected()) == 3
    resumed.confirm('1', 'ANNOTATION_SUSPICIOUS', 'bbox questionable')
    assert resumed.rows[0] == completed
    resumed.confirm('0', 'VALID_HARD_CASE')
    assert resumed.rows[0]['human_notes'] == completed['human_notes']
    assert resumed.backup.read_bytes() == original
    reloaded = h.ReviewStore(project)
    for before, after in zip(evidence, reloaded.rows):
        assert all(after[k] == v for k, v in before.items() if k not in h.HUMAN_FIELDS)
    assert reloaded.rows[0]['human_notes'] == 'small head, occluded\nconfirmed'


@pytest.mark.parametrize('value', ['', '0', '6', '1,2', 'yes', 'VALID_HARD_CASE', '1 2'])
def test_invalid_choices(value):
    with pytest.raises(ValueError):
        h.parse_choice(value)


def test_choices_and_invalid_save(project):
    assert [h.parse_choice(str(i)) for i in range(1, 5)] == list(h.DECISIONS)
    assert h.parse_choice('5') == 'SKIP'
    assert h.parse_choice(' q ') == 'QUIT'
    store = h.ReviewStore(project)
    with pytest.raises(ValueError):
        store.confirm('0', 'APPROVED')
    assert store.path.read_bytes() == store.original
    assert not store.backup.exists()


@pytest.mark.parametrize('value', ['data/curated/v1/valid/images/a.jpg',
                                 'data/curated/v1/test/images/a.jpg',
                                 'data/curated/v1/train/../valid/images/a.jpg',
                                 'data\\curated\\v1\\test\\images\\a.jpg',
                                 'data/curated/v1/train/images/test/a.jpg',
                                 '../outside.jpg'])
def test_path_rejection(project, value):
    with pytest.raises(ValueError, match='TRAIN-only'):
        h.train_path(project, value)


def test_train_path(project):
    assert h.train_path(project, 'data/curated/v1/train/images/0.jpg').is_file()


def test_status_and_review_more(project, capsys):
    store = h.ReviewStore(project)
    assert h.summarize(store)['pending'] == 3
    for i, decision in enumerate(['VALID_HARD_CASE', 'ANNOTATION_SUSPICIOUS', 'REVIEW_MORE']):
        store.confirm(str(i), decision)
    summary = h.summarize(store)
    assert summary['reviewed'] == 3 and summary['pending'] == 0
    assert summary['counts']['REVIEW_MORE'] == 1
    assert not summary['ready']
    store.confirm('2', 'NOT_ACTUALLY_HARD')
    assert h.summarize(store)['ready']  # Suspicious is a later planning concern.
    h.print_status(store)
    assert 'READY TO REPLAN C: YES' in capsys.readouterr().out


def test_atomic_failure_preserves_disk_and_memory(project, monkeypatch):
    store = h.ReviewStore(project)
    def fail(*args):
        raise OSError('simulated interrupted replace')
    monkeypatch.setattr(h.os, 'replace', fail)
    with pytest.raises(OSError):
        store.confirm('0', 'VALID_HARD_CASE')
    assert store.path.read_bytes() == store.original
    assert store.rows[0]['review_status'] == 'PENDING'
    assert store.backup.read_bytes() == store.original
    assert not list(store.path.parent.glob('*.tmp'))
    with h.exclusive_lock(store.path.with_suffix('.lock')):
        pass  # The failed save released the OS lock.


def test_stale_session_cannot_overwrite(project):
    first, stale = h.ReviewStore(project), h.ReviewStore(project)
    first.confirm('0', 'VALID_HARD_CASE')
    with pytest.raises(ValueError, match='another session'):
        stale.confirm('1', 'NOT_ACTUALLY_HARD')
    assert h.ReviewStore(project).rows[0]['human_decision'] == 'VALID_HARD_CASE'


def test_interactive_resume_skip_quit_and_status_read_only(project, monkeypatch):
    answers = iter(['bad', '1', 'occluded', '5', 'Q'])
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    assert h.main([], project) == 0
    assert h.summarize(h.ReviewStore(project))['reviewed'] == 1
    answers = iter(['2', 'bbox questionable'])
    def interrupted(_):
        try:
            return next(answers)
        except StopIteration:
            raise KeyboardInterrupt
    monkeypatch.setattr('builtins.input', interrupted)
    assert h.main(['--pending-only'], project) == 0
    store = h.ReviewStore(project)
    assert [r['review_id'] for r in store.selected(True)] == ['2']
    before = {p: p.read_bytes() for p in store.path.parent.iterdir() if p.is_file()}
    assert h.main(['--status'], project) == 0
    assert before == {p: p.read_bytes() for p in store.path.parent.iterdir() if p.is_file()}


def test_gallery_preference_fallback_and_escape(project):
    row = h.ReviewStore(project).rows[0]
    assert h.visual_path(project, row) == h.train_path(project, row['image'])
    gallery = project / h.OUT / 'hard_example_gallery/0.png'
    gallery.parent.mkdir()
    gallery.write_bytes(b'annotated')
    assert h.visual_path(project, row) == gallery
    with pytest.raises(ValueError, match='gallery'):
        h.visual_path(project, dict(row, gallery='../outside.png'))


@pytest.mark.parametrize('split', ['valid', 'test'])
def test_entire_queue_validated_before_any_decision(project, monkeypatch, split):
    store = h.ReviewStore(project)
    data = store.original.replace(b'train/images/2.jpg', f'{split}/images/2.jpg'.encode())
    store.path.write_bytes(data)
    monkeypatch.setattr('builtins.input', lambda _: pytest.fail('Must reject before input'))
    assert h.main([], project) == 1
    assert store.path.read_bytes() == data
    assert not store.backup.exists()


def test_review_more_filter_separate_from_pending(project):
    store = h.ReviewStore(project)
    store.confirm('0', 'VALID_HARD_CASE')
    store.confirm('1', 'REVIEW_MORE', 'class ambiguous')
    assert [r['review_id'] for r in store.selected(review_more=True)] == ['1']
    assert [r['review_id'] for r in store.selected(pending_only=True)] == ['2']
    with pytest.raises(ValueError):
        store.selected(pending_only=True, review_more=True)


@pytest.mark.parametrize('choice,decision', list(zip('1234', h.DECISIONS)))
@pytest.mark.parametrize('note', ['', 'closer inspection: bbox questionable'])
def test_review_more_redecision(project, monkeypatch, capsys, choice, decision, note):
    store = h.ReviewStore(project)
    store.confirm('0', 'VALID_HARD_CASE', 'completed note')
    store.confirm('1', 'REVIEW_MORE', 'class ambiguous')
    store.confirm('2', 'ANNOTATION_SUSPICIOUS', 'existing concern')
    before = h.ReviewStore(project).rows
    backup = store.backup.read_bytes()
    answers = iter([choice, note])
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    assert h.main(['--review-more'], project) == 0
    after = h.ReviewStore(project)
    assert after.rows[0] == before[0] and after.rows[2] == before[2]
    assert after.rows[1]['human_decision'] == decision
    assert after.rows[1]['review_status'] == 'REVIEWED'
    assert after.rows[1]['reviewed_at']
    assert after.rows[1]['human_notes'] == (note or 'class ambiguous')
    assert after.backup.read_bytes() == backup
    for original, updated in zip(before, after.rows):
        assert all(updated[k] == v for k, v in original.items() if k not in h.HUMAN_FIELDS)
    summary = h.summarize(after)
    assert summary['pending'] == 0
    assert summary['ready'] == (decision != 'REVIEW_MORE')
    output = capsys.readouterr().out
    assert 'human_notes: class ambiguous' in output
    assert 'Review 1/1' in output
    assert 'READY TO REPLAN C: ' + ('NO' if decision == 'REVIEW_MORE' else 'YES') in output
    assert ('Suspicious cases will be conservatively excluded from '
            'hard-example upweighting during C planning.') in output


@pytest.mark.parametrize('choice', ['5', 'Q'])
def test_review_more_skip_quit_preserves_queue(project, monkeypatch, choice):
    store = h.ReviewStore(project)
    store.confirm('0', 'REVIEW_MORE', 'keep this note')
    before = store.path.read_bytes()
    monkeypatch.setattr('builtins.input', lambda _: choice)
    assert h.main(['--review-more'], project) == 0
    assert store.path.read_bytes() == before


def test_review_more_open_image_and_immediate_save(project, monkeypatch):
    store = h.ReviewStore(project)
    store.confirm('0', 'REVIEW_MORE')
    store.confirm('1', 'REVIEW_MORE')
    opened = []
    monkeypatch.setattr(h.webbrowser, 'open', opened.append)
    answers = iter(['1', 'confirmed'])
    def answer(_):
        try:
            return next(answers)
        except StopIteration:
            # Before the next decision, the preceding one is already on disk.
            assert h.ReviewStore(project).rows[0]['human_decision'] == 'VALID_HARD_CASE'
            raise KeyboardInterrupt
    monkeypatch.setattr('builtins.input', answer)
    assert h.main(['--review-more', '--open-image'], project) == 0
    assert opened == [h.train_path(project, store.rows[i]['image']).as_uri() for i in (0, 1)]
    assert [r['review_id'] for r in h.ReviewStore(project).selected(review_more=True)] == ['1']


def test_review_more_help_and_conflicting_filters(project, capsys):
    with pytest.raises(SystemExit) as help_exit:
        h.main(['--help'], project)
    assert help_exit.value.code == 0
    assert '--review-more' in capsys.readouterr().out
    with pytest.raises(SystemExit) as conflict:
        h.main(['--pending-only', '--review-more'], project)
    assert conflict.value.code == 2


@pytest.mark.parametrize('split', ['valid', 'test'])
def test_review_more_rejects_non_train_before_open_or_input(project, monkeypatch, split):
    store = h.ReviewStore(project)
    store.confirm('0', 'REVIEW_MORE')
    data = store.path.read_bytes().replace(b'train/images/0.jpg', f'{split}/images/0.jpg'.encode())
    store.path.write_bytes(data)
    monkeypatch.setattr('builtins.input', lambda _: pytest.fail('No input allowed'))
    monkeypatch.setattr(h.webbrowser, 'open', lambda _: pytest.fail('No image opening allowed'))
    assert h.main(['--review-more', '--open-image'], project) == 1
    assert store.path.read_bytes() == data
