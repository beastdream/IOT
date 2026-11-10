"""Human-only annotations over immutable Phase 4C evidence; no inference."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import webbrowser

from smart_helmet.paths import PROJECT_ROOT
from .failure_review import EVALUATION, guarded_image

CATEGORIES = ('OCCLUSION', 'BLUR', 'LOW_LIGHT', 'SIDE_VIEW', 'BACK_VIEW',
              'CROWDED', 'BACKGROUND_COMPLEX', 'ANNOTATION_OK',
              'ANNOTATION_SUSPICIOUS', 'OTHER', 'UNKNOWN')
KEYS = dict(zip(('1','2','3','4','5','6','7','8','9','O','U'), CATEGORIES))
FIELDS = ('review_id', 'image', 'gt_object_id', 'human_categories', 'notes', 'reviewed')


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_cases(root):
    with (Path(root)/EVALUATION/'failure_review/failure_review.csv').open(encoding='utf-8-sig', newline='') as f:
        cases = list(csv.DictReader(f))
    seen = set()
    for case in cases:
        guarded_image(root, case['image'])
        if case['ground_truth_class'] != 'Without Helmet':
            raise ValueError('Unexpected ground-truth class')
        key = case['image']+':'+str(int(case['ground_truth_index']))
        case['review_id'] = hashlib.sha256(key.encode()).hexdigest()[:20]
        if key in seen:
            raise ValueError('Duplicate missed object')
        seen.add(key)
    if len(cases) != 50:
        raise ValueError('Expected exactly 50 baseline missed objects')
    return cases


def validate_categories(categories):
    if not categories or len(set(categories)) != len(categories) or set(categories)-set(CATEGORIES):
        raise ValueError('Choose one or more distinct listed categories')
    if 'UNKNOWN' in categories and len(categories) != 1:
        raise ValueError('UNKNOWN must be selected alone')
    if {'ANNOTATION_OK','ANNOTATION_SUSPICIOUS'} <= set(categories):
        raise ValueError('Annotation cannot be both OK and suspicious')
    return [c for c in CATEGORIES if c in categories]


def parse_choice(value):
    tokens = value.upper().replace(',', ' ').split()
    if tokens in (['S'], ['Q']):
        return tokens[0]
    try:
        return validate_categories([KEYS[t] if t in KEYS else t for t in tokens])
    except (KeyError, ValueError) as error:
        raise ValueError(str(error)) from error


class ReviewStore:
    def __init__(self, path, cases):
        self.path = Path(path)
        self.original = self.path.read_bytes() if self.path.exists() else None
        self.backed_up = False
        self.rows = {c['review_id']: dict(review_id=c['review_id'], image=c['image'],
            gt_object_id=c['ground_truth_index'], human_categories='', notes='', reviewed='false') for c in cases}
        if self.original is not None:
            reader = csv.DictReader(io.StringIO(self.original.decode('utf-8-sig')))
            if reader.fieldnames != list(FIELDS):
                raise ValueError('Invalid human review columns')
            seen = set()
            for row in reader:
                key = row['review_id']
                if key not in self.rows or key in seen:
                    raise ValueError('Unknown or duplicate review ID')
                seen.add(key)
                expected = self.rows[key]
                if any(row[k] != expected[k] for k in ('image', 'gt_object_id')):
                    raise ValueError('Review identity does not match frozen evidence')
                if row['reviewed'] not in ('true', 'false'):
                    raise ValueError('reviewed must be true or false')
                if row['reviewed'] == 'true':
                    validate_categories(row['human_categories'].split('|'))
                elif row['human_categories']:
                    raise ValueError('Pending rows cannot contain confirmed categories')
                self.rows[key] = row

    def save(self):
        current = self.path.read_bytes() if self.path.exists() else None
        if current != self.original:
            raise ValueError('Review changed in another session; restart before saving')
        if current is not None and not self.backed_up:
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            shutil.copy2(self.path, self.path.with_name(self.path.name+'.'+stamp+'.bak'))
            self.backed_up = True
        output = io.StringIO(newline='')
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(self.rows.values())
        atomic_text(self.path, output.getvalue())
        self.original = self.path.read_bytes()

    def confirm(self, key, categories, notes=''):
        categories = validate_categories(categories)
        self.rows[key].update(human_categories='|'.join(categories), notes=notes, reviewed='true')
        self.save()


def summarize(store):
    reviewed = [r for r in store.rows.values() if r['reviewed'] == 'true']
    counts = Counter(c for r in reviewed for c in r['human_categories'].split('|'))
    n = len(store.rows)
    return dict(total=n, reviewed=len(reviewed), pending=n-len(reviewed),
        ready_for_experiment_b_design=n == 50 and len(reviewed) == n,
        categories={c: dict(count=counts[c], percentage_of_reviewed=100*counts[c]/len(reviewed) if reviewed else None,
                            percentage_of_total=100*counts[c]/n if n else None) for c in CATEGORIES},
        counts_may_overlap=True, zero_means='No confirmed entries; not proof of absence', test_used=False)


def readiness(summary, automatic):
    cats = summary['categories']
    suspicious = cats['ANNOTATION_SUSPICIOUS']['count']
    unknown = cats['UNKNOWN']['count']
    # A conservative evidence gate, not a ratio-only decision or training authorization.
    auto = automatic['categories']
    train = automatic['train_distribution']
    evidence = auto['LOW_CONFIDENCE']['count'] + auto['CLASS_CONFUSION']['count'] > 0
    underrepresented = train['objects']['Without Helmet'] < train['objects']['With Helmet']
    decision = 'REVIEW_REQUIRED'
    obscured = any(cats[c]['count'] for c in ('OCCLUSION', 'BLUR', 'LOW_LIGHT', 'OTHER'))
    if summary['ready_for_experiment_b_design'] and not suspicious and not unknown and not obscured:
        decision = 'YES' if evidence and underrepresented else 'NO'
    lines = ['# Experiment B readiness', f'CLASS-AWARE SAMPLING STILL JUSTIFIED: {decision}',
        f"Human review: {summary['reviewed']}/{summary['total']}; pending: {summary['pending']}.",
        '## evidence_for_sampling',
        f"Automatic LOW_CONFIDENCE: {auto['LOW_CONFIDENCE']['count']}/50; CLASS_CONFUSION: {auto['CLASS_CONFUSION']['count']}/50 (overlap). TRAIN: {json.dumps(train)}.",
        'These observations support testing exposure as a hypothesis; they do not establish its cause.',
        '## evidence_against_sampling',
        'Repeated images may overfit; mixed-class images increase both classes. Geometry and scene-count evidence cannot establish semantic causes. Sampling cannot restore occluded or blurred information.',
        'Human evidence (counts overlap): '+json.dumps(cats),
        'Human occlusion, blur, lighting and viewpoint findings should guide a separate augmentation or data-coverage proposal. UNKNOWN entries complete review but leave the causal decision unresolved.',
        'Conservative decision rule: any occlusion, blur, low-light or OTHER finding requires a human comparison of sampling versus targeted alternatives; automatic evidence and class imbalance alone cannot settle that tradeoff.',
        '## annotation_quality_concerns',
        f'{suspicious} suspicious annotations. Any suspicious annotation blocks endorsement pending adjudication; labels remain frozen.',
        '## alternative_experiments',
        'Targeted TRAIN augmentation after visual confirmation; more independent TRAIN examples; a separate VALID threshold analysis with precision safeguards. No higher-resolution B: tiny failures=0; imgsz=416.',
        '## success_metrics',
        'Primary: Without Helmet recall > baseline 0.5304 with positive lower bound of a paired 95% image-bootstrap improvement interval. Conservative safeguards: Without Helmet precision >=0.7345 and overall mAP50 >=0.7547. Report Without Helmet mAP50 (baseline 0.6411) and overall recall (0.6814). Inconclusive evidence is not success. Keep evaluation settings fixed; never derive criteria from TEST.',
        'READY TO TRAIN EXPERIMENT B: NO. This phase authorizes planning only. Custom loader integration and baseline optimizer-group equivalence must be verified before a future authorized training phase.',
        'TEST USED: NO. TEST remains locked.']
    return decision, '\n\n'.join(lines)+'\n'


def reports(root, store):
    out = Path(root)/EVALUATION/'failure_review'
    summary = summarize(store)
    automatic = json.loads((out/'failure_review_summary.json').read_text(encoding='utf-8'))
    decision, report = readiness(summary, automatic)
    summary['sampling_justified'] = decision
    atomic_text(out/'human_review_summary.json', json.dumps(summary, indent=2)+'\n')
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=('category','count','percentage_of_reviewed','percentage_of_total'))
    writer.writeheader()
    writer.writerows(dict(category=c, **v) for c,v in summary['categories'].items())
    atomic_text(out/'human_review_summary.csv', buffer.getvalue())
    atomic_text(out/'experiment_b_readiness.md', report)
    return summary


def display(case, output=print):
    output('\nReview ID: '+case['review_id'])
    for key in ('image','ground_truth_index','ground_truth_bbox','nearest_prediction_bbox',
                'nearest_prediction_class','confidence','iou','bbox_normalized_area',
                'bbox_width','bbox_height','number_of_gt_objects','categories','automatic_evidence'):
        output(f'{key}: {case[key]}')
    output('Nearest means maximum IoU; zero overlap is not a match. Green=GT, orange=prediction in HTML.')


def open_visual(root, case):
    # Render a per-object overlay without copying or altering any dataset image.
    out = Path(root)/EVALUATION
    gallery = out/'failures/without_helmet_missed'/(Path(case['image']).stem+'.png')
    if not gallery.is_file():
        raise ValueError('Missing related validation visualization')
    from PIL import Image
    source = guarded_image(root, case['image'])
    with Image.open(source) as image:
        width, height = image.size
    boxes = ''
    for field, color in (('ground_truth_bbox', '#00ff88'), ('nearest_prediction_bbox', '#ff9933')):
        box = json.loads(case[field]) if case[field] else None
        if box:
            x, y, x2, y2 = box
            boxes += f'<rect x="{x}" y="{y}" width="{x2-x}" height="{y2-y}" fill="none" stroke="{color}" stroke-width="2"/>'
    page = out/'failure_review/current_human_review.html'
    atomic_text(page, '<!doctype html><html lang="en"><meta charset="utf-8"><title>Human failure review</title>'
        '<style>body{background:#101923;color:white;font:16px system-ui;padding:24px}svg,img{max-width:900px;width:100%}pre{white-space:pre-wrap}</style>'
        f'<h1>Review {html.escape(case["review_id"])}</h1><p>{html.escape(case["image"])}</p>'
        f'<p>Target GT #{html.escape(case["ground_truth_index"])}. Green: missed GT. Orange: nearest prediction (possibly zero overlap).</p>'
        f'<svg viewBox="0 0 {width} {height}"><image href="{source.as_uri()}" width="{width}" height="{height}"/>{boxes}</svg>'
        '<p>Record categories in the terminal. This page does not save review entries.</p>'
        f'<pre>{html.escape(json.dumps(case, indent=2))}</pre><h2>Related saved visualization</h2><img src="{gallery.resolve().as_uri()}" alt="Saved validation evidence"></html>')
    webbrowser.open(page.resolve().as_uri())


def main(argv=None, root=PROJECT_ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pending-only', action='store_true')
    parser.add_argument('--open-image', action='store_true')
    args = parser.parse_args(argv)
    try:
        cases = read_cases(root)
        store = ReviewStore(Path(root)/EVALUATION/'failure_review/human_review.csv', cases)
        if not store.path.exists():
            store.save()
        try:
            for case in cases:
                if args.pending_only and store.rows[case['review_id']]['reviewed'] == 'true':
                    continue
                display(case)
                print('Existing review: '+json.dumps(store.rows[case['review_id']]))
                if args.open_image:
                    open_visual(root, case)
                print(' '.join(f'[{k}] {v}' for k,v in KEYS.items())+' [S] SKIP [Q] QUIT')
                while True:
                    try:
                        selected = parse_choice(input('Choose categories (comma/space separated): '))
                        break
                    except ValueError as error:
                        print(error)
                if selected == 'Q':
                    break
                if selected == 'S':
                    continue
                notes = input('Optional notes: ')
                store.confirm(case['review_id'], selected, notes)
        except (EOFError, KeyboardInterrupt):
            print('\nReview stopped; completed entries saved.')
        reports(root, store)
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f'HUMAN REVIEW: FAIL\n{error}')
        return 1


def status_main(root=PROJECT_ROOT):
    try:
        cases = read_cases(root)
        store = ReviewStore(Path(root)/EVALUATION/'failure_review/human_review.csv', cases)
        if not store.path.exists():
            store.save()
        summary = reports(root, store)
        print('='*40+'\nBASELINE FAILURE HUMAN REVIEW\n'+'='*40)
        print(f"Total missed objects: {summary['total']}\nReviewed: {summary['reviewed']}\nPending: {summary['pending']}")
        print(f"Annotation suspicious: {summary['categories']['ANNOTATION_SUSPICIOUS']['count']}")
        print('READY FOR EXPERIMENT B DESIGN: '+('YES' if summary['ready_for_experiment_b_design'] else 'NO'))
        print('='*40)
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f'HUMAN REVIEW STATUS: FAIL\n{error}')
        return 1
