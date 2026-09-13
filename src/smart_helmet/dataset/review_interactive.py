"""Human-driven P0 decisions; never apply changes to images or labels."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
import uuid
import webbrowser

from smart_helmet.paths import DATASET_CLEANING_DIR
from .audit import write_csv
from .cleaning_review import read_csv, validate_decisions

CHOICES = {'1':'KEEP_BOTH', '2':'KEEP_A_EXCLUDE_B',
           '3':'KEEP_B_EXCLUDE_A', '4':'REVIEW_MORE', 'S':'SKIP', 'Q':'QUIT'}
DECISION_FIELDS = ['review_id', 'decision', 'notes']


def map_choice(value):
    return CHOICES.get(value.strip().upper())


def checked_decisions(queue, path):
    rows = read_csv(path)
    if any(set(r) != set(DECISION_FIELDS) for r in rows):
        raise ValueError('Decision CSV columns must be review_id, decision, notes')
    result = validate_decisions(queue, rows)
    if not result['valid']:
        raise ValueError('; '.join(result['errors']))
    return rows, result


class DecisionSession:
    """Validate a same-directory temporary file before atomic replacement.

    An optimistic byte check rejects external edits made since the case was shown.
    The backup is an exact copy before the first save, once per session.
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / 'review_decisions.csv'
        self.queue_path = self.directory / 'review_queue.csv'
        self.queue_bytes = self.queue_path.read_bytes()
        self.queue = read_csv(self.queue_path)
        self.backup = None
        self.refresh()

    def refresh(self):
        if self.queue_path.read_bytes() != self.queue_bytes:
            raise ValueError('Review queue changed. Restart the review tool.')
        before = self.path.read_bytes()
        rows, result = checked_decisions(self.queue, self.path)
        if self.path.read_bytes() != before:
            raise ValueError('Decision file changed during read. Retry the session.')
        self.expected_bytes = before
        self.decisions = rows
        return result

    def save(self, review_id, decision, notes=''):
        if self.queue_path.read_bytes() != self.queue_bytes or self.path.read_bytes() != self.expected_bytes:
            raise ValueError('External edit detected; original preserved. Restart the review tool.')
        case = next((r for r in self.queue if r['review_id'] == review_id), None)
        if case is None or case['priority'] != 'P0':
            raise ValueError('This tool only updates known P0 review IDs')
        updated = [dict(r) for r in self.decisions]
        target = next(r for r in updated if r['review_id'] == review_id)
        target['decision'] = decision
        if notes.strip():
            target['notes'] = notes
        validation = validate_decisions(self.queue, updated)
        if not validation['valid']:
            raise ValueError('; '.join(validation['errors']))
        descriptor, name = tempfile.mkstemp(prefix='.review_decisions_', suffix='.tmp', dir=self.directory)
        os.close(descriptor)
        temporary = Path(name)
        try:
            write_csv(temporary, updated, DECISION_FIELDS)
            checked_decisions(self.queue, temporary)
            with temporary.open('r+b') as stream:
                os.fsync(stream.fileno())
            if self.path.read_bytes() != self.expected_bytes or self.queue_path.read_bytes() != self.queue_bytes:
                raise ValueError('External edit detected; original preserved. Restart the review tool.')
            if self.backup is None:
                backup_dir = self.directory / 'backups'
                backup_dir.mkdir(exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
                self.backup = backup_dir / f'review_decisions_{stamp}_{uuid.uuid4().hex[:8]}.csv'
                with self.backup.open('xb') as stream:
                    stream.write(self.expected_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
        return self.refresh()  # Reuse the validator after every completed write.


def open_visualization(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Visualization missing: {path}')
    if os.name == 'nt':
        os.startfile(str(path))
    elif not webbrowser.open(path.as_uri()):
        raise OSError('Could not open visualization in the default application')


def open_report(directory=DATASET_CLEANING_DIR, output=print):
    path = (Path(directory) / 'review_report/index.html').resolve()
    if not path.is_file():
        output('Report missing. Run: .venv\\Scripts\\python.exe scripts/prepare_cleaning_review.py')
        return False
    if not webbrowser.open(path.as_uri(), new=2):
        output(f'Browser launch failed. Open this report manually: {path}')
        return False
    output(f'Opened report: {path}')
    return True


def show_progress(result, output):
    state = result['priorities']['P0']
    output(f"P0 completed: {state['completed']} / {state['total']}")
    output(f"P0 pending: {state['pending']}")
    output(f"REVIEW_MORE: {state['review_more']}")
    output('READY TO APPLY CLEANING: ' + ('YES' if result['ready_to_apply_cleaning'] else 'NO'))


def review_cases(directory=DATASET_CLEANING_DIR, pending_only=False, open_image=False,
                 input_fn=input, output=print, image_opener=None):
    session = DecisionSession(directory)
    decision_by_id = {r['review_id']:r for r in session.decisions}
    cases = [r for r in session.queue if r['priority'] == 'P0' and
             (not pending_only or decision_by_id[r['review_id']]['decision'] == 'PENDING')]
    opener = image_opener or open_visualization
    if not cases:
        output('No matching P0 cases. Use without --pending-only to revisit REVIEW_MORE or completed cases.')
    for index, case in enumerate(cases, 1):
        session.refresh()
        current = next(r for r in session.decisions if r['review_id'] == case['review_id'])
        if pending_only and current['decision'] != 'PENDING':
            continue
        output('=' * 40 + f'\nP0 REVIEW {index} / {len(cases)}\n' + '=' * 40)
        for key in ('review_id','priority','issue_type','split_a','image_a','split_b','image_b',
                    'phash_distance','source_group_a','source_group_b','annotation_count_a',
                    'annotation_count_b','classes_a','classes_b'):
            output(f'{key}: {case.get(key, "")}')
        visualization = (Path(directory) / case['visualization_path']).resolve()
        if not visualization.is_relative_to(Path(directory).resolve() / 'review_images'):
            raise ValueError('Visualization path must stay inside review_images/')
        output(f'Visualization: {visualization}')
        output(f"Current decision: {current['decision']}")
        output(f"Current notes: {current['notes']}")
        if open_image:
            try:
                opener(visualization)
            except OSError as error:
                output(f'Image opener failed: {error}. Open the displayed path manually.')
        for key, label in CHOICES.items():
            output(f'[{key}] {label}')
        while True:
            try:
                choice = map_choice(input_fn('Choice: '))
                if choice is None:
                    output('Invalid choice. Use 1, 2, 3, 4, S or Q.')
                    continue
                if choice == 'QUIT':
                    show_progress(session.refresh(), output)
                    return
                if choice == 'SKIP':
                    break
                notes = input_fn('Add notes? [Enter to keep existing notes]: ')
            except (EOFError, KeyboardInterrupt):
                output('\nStopped. Unsaved choice was not written; previous saves are preserved.')
                show_progress(session.refresh(), output)
                return
            result = session.save(case['review_id'], choice, notes)
            output(f"Saved {case['review_id']}. Backup: {session.backup}")
            show_progress(result, output)
            break
    show_progress(session.refresh(), output)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Interactive human P0 review. Saves decisions only; never applies cleaning.')
    parser.add_argument('--priority', choices=['P0'], default='P0', help='Priority to review (this phase supports P0 only)')
    parser.add_argument('--pending-only', action='store_true', help='Resume PENDING cases only; REVIEW_MORE is excluded')
    parser.add_argument('--open-image', action='store_true', help='Open each visualization in the default image application')
    args = parser.parse_args(argv)
    try:
        review_cases(pending_only=args.pending_only, open_image=args.open_image)
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f'P0 REVIEW TOOL: ERROR\n{error}')
        return 1
