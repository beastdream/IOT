"""Interactive human decisions over saved TRAIN evidence; no inference or replanning."""
import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import tempfile
import webbrowser

from smart_helmet.paths import PROJECT_ROOT

OUT = Path('results/hard_example_mining/A_baseline_train')
DECISIONS = ('VALID_HARD_CASE', 'ANNOTATION_SUSPICIOUS', 'NOT_ACTUALLY_HARD', 'REVIEW_MORE')
HUMAN_FIELDS = ('review_status', 'human_decision', 'human_notes', 'reviewed_at')


@contextmanager
def exclusive_lock(path):
    # OS locks release even on process termination. Keep the lock file so
    # concurrent processes always lock the same file, including after restart.
    with path.open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def atomic_bytes(path, data, create_only=False):
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if create_only:
            # Publish a complete backup atomically, never replacing an old one.
            os.link(name, path)
        else:
            os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def train_path(root, value):
    root = Path(root).resolve()
    supplied = Path(value.replace('\\', '/'))
    if any(part.lower() in ('valid', 'test') for part in supplied.parts):
        raise ValueError('TRAIN-only guard: VALID/TEST path rejected')
    path = (root / supplied).resolve()
    train = root / 'data/curated/v1/train'
    if not path.is_relative_to(train) or any(p.lower() in ('valid', 'test') for p in path.relative_to(root).parts):
        raise ValueError('TRAIN-only guard: source is outside data/curated/v1/train')
    if not path.is_file():
        raise ValueError(f'Missing TRAIN image: {value}')
    return path


def parse_choice(value):
    value = value.strip().upper()
    if value == 'Q':
        return 'QUIT'
    if value == '5':
        return 'SKIP'
    if value in ('1', '2', '3', '4'):
        return DECISIONS[int(value) - 1]
    raise ValueError('Choose exactly 1, 2, 3, 4, 5 or Q.')


class ReviewStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.path = self.root / OUT / 'manual_review_queue.csv'
        self.backup = self.path.with_suffix('.backup.csv')
        self.original = self.path.read_bytes()
        reader = csv.DictReader(io.StringIO(self.original.decode('utf-8-sig'), newline=''))
        self.fields = reader.fieldnames or []
        required = {'review_id', 'image', 'tier', 'automatic_reason', 'gt_summary', 'prediction_summary', 'review_status'}
        if not required <= set(self.fields) or len(set(self.fields)) != len(self.fields):
            raise ValueError('Invalid review queue columns')
        self.rows = list(reader)
        seen = set()
        for row in self.rows:
            if None in row or any(v is None for v in row.values()):
                raise ValueError('Malformed review CSV')
            if not row['review_id'] or row['review_id'] in seen:
                raise ValueError('Missing/duplicate review ID')
            seen.add(row['review_id'])
            train_path(self.root, row['image'])
            status, decision = row['review_status'], row.get('human_decision', '')
            if status not in ('PENDING', 'REVIEWED') or (status == 'REVIEWED' and decision not in DECISIONS) or (status == 'PENDING' and decision):
                raise ValueError('Invalid human review state')

    def selected(self, pending_only=False, review_more=False):
        if pending_only and review_more:
            raise ValueError('Choose either pending-only or review-more, not both')
        return [r.copy() for r in self.rows
                if (not pending_only or r['review_status'] == 'PENDING')
                and (not review_more or r.get('human_decision') == 'REVIEW_MORE')]

    def confirm(self, review_id, decision, notes=''):
        if decision not in DECISIONS:
            raise ValueError('Invalid human decision')
        updated = [r.copy() for r in self.rows]
        row = next((r for r in updated if r['review_id'] == review_id), None)
        if row is None:
            raise ValueError('Unknown review ID')
        row.update(review_status='REVIEWED', human_decision=decision,
                   human_notes=notes if notes else row.get('human_notes', ''),
                   reviewed_at=datetime.now(timezone.utc).isoformat())
        fields = self.fields + [f for f in HUMAN_FIELDS if f not in self.fields]
        output = io.StringIO(newline='')
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(updated)
        data = output.getvalue().encode('utf-8')
        # Exclusive lock serializes writers from this tool. Byte comparison also
        # rejects stale sessions and external edits before replacing anything.
        lock = self.path.with_suffix('.lock')
        with exclusive_lock(lock):
            if self.path.read_bytes() != self.original:
                raise ValueError('Queue changed in another session; restart before saving')
            for entry in updated:
                train_path(self.root, entry['image'])
            if not self.backup.exists():
                atomic_bytes(self.backup, self.original, create_only=True)
            atomic_bytes(self.path, data)
        self.rows, self.fields, self.original = updated, fields, data


def summarize(store):
    counts = Counter(r.get('human_decision') for r in store.rows if r['review_status'] == 'REVIEWED')
    reviewed = sum(counts.values())
    pending = len(store.rows) - reviewed
    return dict(total=len(store.rows), reviewed=reviewed, pending=pending,
                counts={d: counts[d] for d in DECISIONS},
                ready=bool(store.rows) and pending == 0 and counts['REVIEW_MORE'] == 0)


def print_status(store):
    summary = summarize(store)
    print('=' * 40 + '\nHARD EXAMPLE HUMAN REVIEW\n' + '=' * 40)
    for label, key in [('Total review cases', 'total'), ('Reviewed', 'reviewed'), ('Pending', 'pending')]:
        print(f'{label}: {summary[key]}')
    for decision, count in summary['counts'].items():
        print(f'{decision}: {count}')
    print('TRAIN-only: PASS\nVALID used: 0\nTEST used: 0')
    print('READY TO REPLAN C: ' + ('YES' if summary['ready'] else 'NO'))
    print('Suspicious cases will be conservatively excluded from '
          'hard-example upweighting during C planning.')
    print('=' * 40)


def object_evidence(root):
    path = Path(root) / OUT / 'train_object_difficulty.csv'
    grouped = {}
    if path.exists():
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                train_path(root, row['image'])
                grouped.setdefault(row['image'], []).append(row)
    return grouped


def display(row, index, total, objects):
    print(f'\nReview {index}/{total}')
    for key in ('review_id', 'image', 'tier', 'automatic_reason', 'review_status', 'human_decision', 'notes', 'human_notes'):
        print(f'{key}: {row.get(key) or "(none)"}')
    print('GT class / bbox: ' + json.dumps(json.loads(row['gt_summary'])))
    print('Predicted class / confidence / bbox: ' + json.dumps(json.loads(row['prediction_summary'])))
    print('Object-level difficulty (saved evidence; no new matching):')
    for obj in objects:
        print(' | '.join(f'{k}: {obj.get(k) or "N/A"}' for k in
                        ('gt_index', 'class_name', 'gt_bbox', 'predicted_class', 'confidence', 'iou', 'category')))
    if not objects:
        print('Category / IoU: N/A (no saved object evidence)')


def visual_path(root, row):
    source = train_path(root, row['image'])
    out = Path(root).resolve() / OUT
    gallery = out / 'hard_example_gallery'
    candidate = (out / (row.get('gallery') or f'hard_example_gallery/{row["review_id"]}.png')).resolve()
    if not candidate.is_relative_to(gallery) or any(p.lower() in ('valid', 'test') for p in candidate.parts):
        raise ValueError('Unsafe gallery path')
    return candidate if candidate.is_file() else source


def main(argv=None, root=PROJECT_ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    filters = parser.add_mutually_exclusive_group()
    filters.add_argument('--pending-only', action='store_true',
                         help='Select only pending/unreviewed cases (excludes REVIEW_MORE).')
    filters.add_argument('--review-more', action='store_true',
                         help='Re-review only cases whose current human decision is REVIEW_MORE.')
    parser.add_argument('--open-image', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args(argv)
    try:
        store = ReviewStore(root)
        if args.status:
            print_status(store)
            return 0
        evidence = object_evidence(root)
        selected = store.selected(args.pending_only, args.review_more)
        try:
            for index, row in enumerate(selected, 1):
                display(row, index, len(selected), evidence.get(row['image'], []))
                if args.open_image:
                    webbrowser.open(visual_path(root, row).as_uri())
                print('1 = VALID_HARD_CASE (annotation correct; genuinely hard)\n'
                      '2 = ANNOTATION_SUSPICIOUS (GT questionable)\n'
                      '3 = NOT_ACTUALLY_HARD (mining conclusion unsupported)\n'
                      '4 = REVIEW_MORE (cannot decide yet)\n5 = SKIP\nQ = quit safely')
                while True:
                    try:
                        decision = parse_choice(input('Decision: '))
                        break
                    except ValueError as error:
                        print(error)
                if decision == 'QUIT':
                    break
                if decision == 'SKIP':
                    continue
                notes = input('Optional note (blank preserves existing notes; new text replaces human notes): ')
                store.confirm(row['review_id'], decision, notes)
                print('Saved.')
        except (EOFError, KeyboardInterrupt):
            print('\nStopped safely; completed decisions remain saved.')
        print_status(store)
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f'HARD EXAMPLE HUMAN REVIEW: FAIL\nTRAIN-only: FAIL or unverified\nREADY TO REPLAN C: NO\n{error}')
        return 1
