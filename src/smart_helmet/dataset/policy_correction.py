"""Decision-only policy analysis and grouped human correction reports."""
from collections import defaultdict
from html import escape
from pathlib import Path

from smart_helmet.paths import DATASET_CLEANING_DIR, PROJECT_ROOT
from .cleaning_review import read_csv, validate_decisions
from .curation_policy import SPLIT_PRIORITY


def recommendation(case):
    a, b = (SPLIT_PRIORITY.get(case.get('split_'+s), 0) for s in ('a', 'b'))
    direction = 'KEEP_A_EXCLUDE_B' if a > b else 'KEEP_B_EXCLUDE_A' if b > a else 'REVIEW_MORE'
    return (f'If human confirms genuine leakage: {direction} (preserve TEST > VALID > TRAIN). '
            'If images are genuinely different: KEEP_BOTH. No option is selected automatically.')


def actions(case, decision):
    if decision == 'KEEP_BOTH':
        return [(case['image_a'], 'KEEP'), (case['image_b'], 'KEEP')]
    if decision in ('KEEP_A_EXCLUDE_B', 'KEEP_B_EXCLUDE_A'):
        return [(case['image_a'], 'KEEP' if decision == 'KEEP_A_EXCLUDE_B' else 'EXCLUDE'),
                (case['image_b'], 'EXCLUDE' if decision == 'KEEP_A_EXCLUDE_B' else 'KEEP')]
    return []


def analyze(queue, decisions, expected_total=27):
    validation = validate_decisions(queue, decisions)
    if not validation['valid']:
        raise ValueError('; '.join(validation['errors']))
    current = {r['review_id']: r for r in decisions}
    evidence = defaultdict(lambda: defaultdict(list))
    mismatches, needs, excluded = set(), set(), defaultdict(set)
    p0 = [r for r in queue if r['priority'] == 'P0']
    pending = more = completed = 0
    for case in p0:
        rid = case['review_id']; row = current[rid]; decision = row['decision']
        pending += decision == 'PENDING'; more += decision == 'REVIEW_MORE'
        completed += decision in ('KEEP_BOTH', 'KEEP_A_EXCLUDE_B', 'KEEP_B_EXCLUDE_A')
        for name, action in actions(case, decision):
            evidence[name][action].append(rid)
            if action == 'EXCLUDE':
                side = 'a' if name == case['image_a'] else 'b'
                other = 'b' if side == 'a' else 'a'
                split, paired = case['split_'+side], case['split_'+other]
                excluded[split].add(name)
                if SPLIT_PRIORITY.get(split, 0) > SPLIT_PRIORITY.get(paired, 0):
                    mismatches.add(name)
                    if not row['notes'].strip(): needs.add(name)
                if split not in SPLIT_PRIORITY or paired not in SPLIT_PRIORITY or split == paired:
                    needs.add(name)
        if decision in ('PENDING', 'REVIEW_MORE'):
            needs.update([case['image_a'], case['image_b']])
    conflicts = {name: dict(v) for name, v in evidence.items() if len(v) > 1}
    return dict(total=len(p0), completed=completed, pending=pending, review_more=more,
                conflicts=conflicts, mismatches=sorted(mismatches), needs_review=sorted(needs),
                excluded={s: sorted(excluded[s]) for s in SPLIT_PRIORITY},
                status='PASS' if len(p0) == expected_total and completed == len(p0) and not
                (conflicts or mismatches or needs or pending or more) else 'REVIEW_REQUIRED')


def mismatch_groups(queue, decisions):
    state = analyze(queue, decisions)
    groups = []
    for name in state['mismatches']:
        cases = [r for r in queue if r['priority'] == 'P0' and name in (r['image_a'], r['image_b'])]
        split = next(r['split_'+s] for r in cases for s in ('a', 'b') if r['image_'+s] == name)
        if split in ('test', 'valid'):
            groups.append(dict(image=name, split=split, cases=cases))
    return groups


def load_review(directory=DATASET_CLEANING_DIR):
    from .review_interactive import checked_decisions
    directory = Path(directory)
    queue = read_csv(directory/'review_queue.csv')
    decisions, _ = checked_decisions(queue, directory/'review_decisions.csv')
    lookup = {r['review_id']: r for r in queue}
    # These reports describe the existing build, not necessarily the latest decisions.
    # Verify their relationships, but derive active groups from current decisions.
    for filename, image_key, pair_key in (
        ('pretraining_curation_sanity.csv', 'excluded_image', 'retained_pair_image'),
        ('test_exclusion_review.csv', 'test_image', 'paired_image')):
        for row in read_csv(directory/filename):
            case = lookup.get(row['review_id'])
            if case is None or case['priority'] != 'P0' or {row[image_key], row[pair_key]} != {case['image_a'], case['image_b']}:
                raise ValueError(f'Invalid pair relationship in {filename}: {row["review_id"]}')
    return queue, decisions


def describe(case, decision):
    final = actions(case, decision['decision'])
    lines = [f'{key}: {case.get(key, "")}' for key in (
        'review_id', 'split_a', 'image_a', 'split_b', 'image_b', 'phash_distance',
        'source_group_a', 'source_group_b', 'annotation_count_a', 'annotation_count_b', 'classes_a', 'classes_b')]
    lines += [f'A split: {case["split_a"].upper()}', f'B split: {case["split_b"].upper()}',
              'Current decision: '+decision['decision'], 'Notes: '+decision['notes'],
              'Current excluded image: '+', '.join(n for n,a in final if a == 'EXCLUDE'),
              'Current retained image: '+', '.join(n for n,a in final if a == 'KEEP'),
              'Policy recommendation: '+recommendation(case)]
    return '\n'.join(lines)


def generate_report(directory=DATASET_CLEANING_DIR, root=PROJECT_ROOT):
    directory, root = Path(directory), Path(root).resolve()
    queue, decisions = load_review(directory)
    groups = mismatch_groups(queue, decisions)
    current = {r['review_id']: r for r in decisions}
    def picture(name):
        path = (root/name).resolve()
        if not path.is_relative_to(root): raise ValueError('Image path outside project')
        return f'<img loading="lazy" src="{escape(path.as_uri(), quote=True)}" alt="{escape(name, quote=True)}">'
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
             '<title>Human policy correction</title><style>body{font:16px system-ui;margin:24px}img{max-width:520px;max-height:420px}pre{white-space:pre-wrap;overflow-wrap:anywhere}article,section{border:1px solid #bbb;padding:16px;margin:16px 0}</style></head><body>',
             '<h1>Human policy correction</h1><p>Read-only snapshot of current decisions. Images show original pixels; annotation counts and classes are listed below. Confirm each pair in the CLI. No automatic selection, rebuild or training.</p>']
    for group in groups:
        parts += ['<section><h2>'+escape(group['split'].upper()+' image: '+group['image'])+'</h2>', picture(group['image'])]
        for case in group['cases']:
            other = case['image_b'] if case['image_a'] == group['image'] else case['image_a']
            parts += ['<article><pre>'+escape(describe(case, current[case['review_id']]))+'</pre>', picture(other), '</article>']
        parts.append('</section>')
    parts.append('</body></html>')
    path = directory/'policy_correction_report.html'
    path.write_text('\n'.join(parts), encoding='utf-8')
    return path, groups


def show_state(state, output=print):
    output('='*40+'\nP0 DECISION CONSISTENCY\n'+'='*40)
    for label, value in [('P0 total', state['total']), ('Completed', state['completed']),
                         ('PENDING', state['pending']), ('REVIEW_MORE', state['review_more']),
                         ('Contradictory image actions', len(state['conflicts'])),
                         ('Policy mismatches', len(state['mismatches'])),
                         ('Remaining needs-review', len(state['needs_review'])),
                         ('TEST images currently excluded', len(state['excluded']['test'])),
                         ('VALID images currently excluded', len(state['excluded']['valid']))]:
        output(f'{label}: {value}')
    for name, evidence in state['conflicts'].items(): output(f'{name}: {evidence}')
    output('Counts refer to unique images implied by current decisions; the built dataset is unchanged.')
    output('DECISION CONSISTENCY: '+state['status']+'\n'+'='*40)
