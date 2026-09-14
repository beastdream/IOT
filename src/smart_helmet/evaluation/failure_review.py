"""Offline Phase 4C: frozen validation predictions, TRAIN statistics, no inference."""
from collections import Counter
import csv
import html
import json
from pathlib import Path

from smart_helmet.paths import PROJECT_ROOT
from smart_helmet.foundation import EXPECTED_CLASSES, IMAGE_EXTENSIONS
from smart_helmet.dataset.duplicates import sha256_file
from smart_helmet.training.experiment import write_json
from .failure_analysis import analyze_image, area, ground_truth, iou

EVALUATION = Path('results/evaluation/A_baseline_val')
CATEGORIES = ('LOW_CONFIDENCE', 'CLASS_CONFUSION', 'OCCLUSION_CANDIDATE',
              'BLUR_CANDIDATE', 'LOW_LIGHT_CANDIDATE', 'SIDE_OR_BACK_VIEW_CANDIDATE',
              'SMALL_OBJECT', 'CROWDED_SCENE', 'BACKGROUND_COMPLEXITY',
              'ANNOTATION_REVIEW', 'UNKNOWN')
VISUAL = [c for c in CATEGORIES if c.endswith('_CANDIDATE')] + ['BACKGROUND_COMPLEXITY', 'ANNOTATION_REVIEW']


def guarded_image(root, image, split='val', test_status='LOCKED / NOT USED'):
    if split not in ('val', 'train') or test_status != 'LOCKED / NOT USED':
        raise ValueError('TEST must remain locked; only TRAIN and VALID are allowed')
    base = (Path(root)/'data/curated/v1'/('valid' if split == 'val' else 'train')/'images').resolve()
    path = (Path(root)/image).resolve()
    if path.parent != base or path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError('Image is outside the explicitly allowed split')
    label = path.parent.parent/'labels'/(path.stem+'.txt')
    if label.resolve().parent != (base.parent/'labels').resolve():
        raise ValueError('Label escapes allowed split')
    return path


def size_bucket(normalized_area):
    return next((n for n, t in zip(('tiny', 'small', 'medium'), (.001, .01, .1))
                 if normalized_area < t), 'large')


def category_mapping(case, index):
    tags = {e['error_type'] for e in case['events'] if e['ground_truth_index'] == index}
    result = [c for c in ('LOW_CONFIDENCE', 'CLASS_CONFUSION') if c in tags]
    if size_bucket(area(case['ground_truth'][index]['bbox'])/(case['width']*case['height'])) in ('tiny', 'small'):
        result.append('SMALL_OBJECT')
    # Explicit descriptive cutoff, not an inferred occlusion cause.
    if len(case['ground_truth']) >= 3:
        result.append('CROWDED_SCENE')
    return result or ['UNKNOWN']


def review_rows(cases):
    rows = []
    for case in cases:
        for index in case['missed_gt']:
            gt = case['ground_truth'][index]
            if gt['class_id'] != 1:
                continue
            box = gt['bbox']
            candidates = list(enumerate(case['predictions']))
            nearest = max(candidates, key=lambda p: (iou(box, p[1]['bbox']), p[1]['confidence'], -p[0]), default=None)
            j, pred = nearest if nearest else (None, None)
            evidence = [e for e in case['events'] if e['ground_truth_index'] == index
                        and e['error_type'] in ('LOW_CONFIDENCE', 'CLASS_CONFUSION')]
            rows.append(dict(image=case['image'], ground_truth_index=index, ground_truth_bbox=box,
                ground_truth_class='Without Helmet', nearest_prediction_index=j,
                nearest_prediction_bbox=pred['bbox'] if pred else None,
                nearest_prediction_class=EXPECTED_CLASSES[pred['class_id']] if pred else None,
                confidence=pred['confidence'] if pred else None, iou=iou(box, pred['bbox']) if pred else None,
                bbox_normalized_area=area(box)/(case['width']*case['height']),
                bbox_width=box[2]-box[0], bbox_height=box[3]-box[1],
                size_bucket=size_bucket(area(box)/(case['width']*case['height'])),
                number_of_gt_objects=len(case['ground_truth']), number_of_predictions=len(case['predictions']),
                number_of_active_predictions=sum(p['confidence'] >= .25 for p in case['predictions']),
                multiple_object_scene=len(case['ground_truth']) > 1, categories=category_mapping(case, index),
                automatic_evidence=evidence, candidate_causes=VISUAL,
                manual_review_required=True, manual_review_status='PENDING', manual_review_notes=''))
    return rows


def distribution(cases):
    images = Counter(); objects = Counter(); sizes = Counter(); scenes = Counter()
    for case in cases:
        ids = {g['class_id'] for g in case['ground_truth']}
        images['both' if ids == {0, 1} else 'with_helmet_only' if ids == {0} else 'without_helmet_only' if ids == {1} else 'empty'] += 1
        for gt in case['ground_truth']:
            objects[EXPECTED_CLASSES[gt['class_id']]] += 1
            if gt['class_id'] == 1:
                sizes[size_bucket(area(gt['bbox'])/(case['width']*case['height']))] += 1
                n = len(case['ground_truth'])
                scenes['single' if n == 1 else 'two' if n == 2 else 'three_or_more'] += 1
    return dict(images=len(cases), image_composition=dict(images), objects=dict(objects),
                without_helmet_sizes={s: sizes[s] for s in ('tiny', 'small', 'medium', 'large')},
                without_helmet_scene_objects={s: scenes[s] for s in ('single', 'two', 'three_or_more')})


def aggregate_review(rows):
    n = len(rows)
    if len({(r['image'], r['ground_truth_index']) for r in rows}) != n:
        raise ValueError('Duplicate missed object')
    counts = Counter(c for r in rows for c in r['categories'])
    return dict(objects=n, images=len({r['image'] for r in rows}),
                categories={c: dict(count=counts[c], percentage=100*counts[c]/n if n else 0) for c in CATEGORIES},
                sizes=dict(Counter(r['size_bucket'] for r in rows)),
                multiple_object_objects=sum(r['multiple_object_scene'] for r in rows),
                scene_objects=dict(Counter('single' if r['number_of_gt_objects'] == 1 else 'two' if r['number_of_gt_objects'] == 2 else 'three_or_more' for r in rows)),
                manual_review_required=sum(r['manual_review_required'] for r in rows),
                manual_visual_confirmed=0, category_counts_overlap=True)


def render_report(rows, out, root):
    sections = []
    for image in dict.fromkeys(r['image'] for r in rows):
        path = guarded_image(root, image)
        cases = [r for r in rows if r['image'] == image]
        # Inline SVG overlays preserve original pixels and show each missed object separately.
        from PIL import Image
        import base64
        with Image.open(path) as source:
            w, h = source.size
        mime = 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'
        uri = 'data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()
        body = []
        for r in cases:
            boxes = ''
            for box, color in ((r['ground_truth_bbox'], '#00ff88'), (r['nearest_prediction_bbox'], '#ff9933')):
                if box:
                    x, y, x2, y2 = box
                    boxes += f'<rect x="{x}" y="{y}" width="{x2-x}" height="{y2-y}" fill="none" stroke="{color}" stroke-width="2"/>'
            evidence = html.escape(json.dumps(r, indent=2))
            key = html.escape(image+':'+str(r['ground_truth_index']), quote=True)
            body.append(f'<article><h3>Missed GT #{r["ground_truth_index"]}: Without Helmet</h3><svg viewBox="0 0 {w} {h}"><image href="{uri}" width="{w}" height="{h}"/>{boxes}</svg><p>Green: GT; orange: nearest retained prediction (maximum IoU, possibly zero; not necessarily a match).</p><details open><summary>Evidence and candidate checklist</summary><pre>{evidence}</pre></details><label>Human status <select data-key="{key}:status"><option>PENDING</option><option>REVIEWED</option><option>ANNOTATION_REVIEW</option></select></label><label>Confirmed categories / notes <textarea data-key="{key}:notes"></textarea></label></article>')
        sections.append(f'<section><h2>{html.escape(path.name)}</h2><details><summary>Original image</summary><img src="{uri}" alt="Original validation image"></details>'+''.join(body)+'</section>')
    out.joinpath('review_report.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Baseline failure review</title><style>body{font:16px system-ui;max-width:1200px;margin:auto;padding:24px;background:#101923;color:#eee}section{border-top:2px solid #789;padding:16px 0}article{border:1px solid #567;padding:16px;margin:12px 0}svg,img{width:100%;max-width:650px}pre{white-space:pre-wrap;overflow-wrap:anywhere}label{display:block;margin:10px 0}textarea{width:95%;height:70px}</style><h1>50 missed Without Helmet objects · 28 VALID images</h1><p>Offline review, no inference. Candidate checklist is unconfirmed, not automatic findings. All semantic causes require human confirmation. Counts overlap. Manual entries persist in this browser only; export them for review. No labels are edited.</p><button onclick="downloadReview()">Export human review JSON</button>'''+''.join(sections)+'''<script>const fields=[...document.querySelectorAll('[data-key]')];for(const f of fields){f.value=localStorage.getItem('phase4c:'+f.dataset.key)||f.value;f.addEventListener('change',()=>localStorage.setItem('phase4c:'+f.dataset.key,f.value))}function downloadReview(){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(Object.fromEntries(fields.map(f=>[f.dataset.key,f.value])),null,2)],{type:'application/json'}));a.download='human_failure_review.json';a.click();URL.revokeObjectURL(a.href)}</script></html>''', encoding='utf-8')


def run_review(root=PROJECT_ROOT, split='val', test_status='LOCKED / NOT USED'):
    if split != 'val' or test_status != 'LOCKED / NOT USED':
        raise ValueError('Validation-only review; TEST locked')
    root = Path(root); source = root/EVALUATION
    saved = json.loads((source/'validation_predictions.json').read_text())
    metrics = json.loads((source/'metrics.json').read_text())
    checkpoint = root/'results/experiments/A_baseline/weights/best.pt'
    if sha256_file(checkpoint) != saved['checkpoint_sha256'] or saved['checkpoint_sha256'] != metrics['checkpoint_sha256']:
        raise ValueError('Checkpoint provenance mismatch')
    if saved['dataset_fingerprint'] != metrics['dataset_fingerprint']:
        raise ValueError('Prediction/metric dataset provenance mismatch')
    cases = []; before = {}
    def read_case(path, predictions):
        label = path.parent.parent/'labels'/(path.stem+'.txt')
        for p in (path, label): before[p] = sha256_file(p)
        gt, w, h = ground_truth(path)
        return analyze_image(path.relative_to(root).as_posix(), gt, predictions, w, h)
    for cached in saved['cases']:
        path = guarded_image(root, cached['image'])
        case = read_case(path, cached['predictions'])
        if case['ground_truth'] != cached['ground_truth'] or (case['width'], case['height']) != (cached['width'], cached['height']):
            raise ValueError('Curated VALID annotations differ from cached baseline')
        cases.append(case)
    rows = review_rows(cases)
    gallery = source/'failures/without_helmet_missed'
    expected_gallery = {Path(r['image']).stem+'.png' for r in rows}
    if {p.name for p in gallery.glob('*.png')} != expected_gallery:
        raise ValueError('Existing missed-image gallery does not match review cohort')
    with (source/'failure_cases.csv').open(encoding='utf-8-sig', newline='') as f:
        expected = {(r['image'], int(r['ground_truth_index'])) for r in csv.DictReader(f)
                    if r['error_type'] == 'FALSE_NEGATIVE' and r['ground_truth_class'] == 'Without Helmet'}
    if expected != {(r['image'], r['ground_truth_index']) for r in rows}:
        raise ValueError('Misses disagree with baseline failure CSV')
    summary = aggregate_review(rows)
    if (len(cases), summary['objects'], summary['images']) != (127, 50, 28):
        raise ValueError('Unexpected Phase 4C baseline cohort')
    train = []
    for path in sorted((root/'data/curated/v1/train/images').iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            train.append(read_case(guarded_image(root, path, 'train'), []))
    summary.update(train_distribution=distribution(train), validation_distribution=distribution(cases),
                   metrics=metrics, test_used=False, test_status=test_status, status='PASS',
                   policy=dict(confidence=.25, prediction_floor=.001, match_iou=.5,
                               nearest='Maximum IoU across retained post-NMS predictions; confidence breaks ties; zero overlap is not a match',
                               small='normalized area <0.01; tiny <0.001', crowded='3 or more GT objects; descriptive only',
                               manual='All visual candidates unconfirmed; zero assigned is not absence'),
                   provenance={p.name: sha256_file(p) for p in [checkpoint, source/'validation_predictions.json', source/'failure_cases.csv', source/'baseline_evaluation_report.md']})
    summary['gallery_verified_images'] = len(expected_gallery)
    summary['distribution_comparison'] = compare_distributions(summary)
    out = source/'failure_review'; out.mkdir(exist_ok=True)
    render_report(rows, out, root)
    if any(sha256_file(p) != digest for p, digest in before.items()):
        raise ValueError('TRAIN/VALID input changed during review')
    summary['read_input_immutability'] = 'PASS'
    with (out/'failure_review.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in rows)
    write_json(out/'failure_review_summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('status', 'objects', 'images', 'categories', 'train_distribution')}, indent=2))
    return summary


def compare_distributions(summary):
    """Use full validation denominators, rather than inferring risk from misses alone."""
    output = {}
    train = summary['train_distribution']; valid = summary['validation_distribution']
    for name, distribution_key, failure_key in (
            ('size', 'without_helmet_sizes', 'sizes'),
            ('scene', 'without_helmet_scene_objects', 'scene_objects')):
        output[name] = {}
        for bucket, count in train[distribution_key].items():
            denominator = valid[distribution_key][bucket]
            missed = summary[failure_key].get(bucket, 0)
            output[name][bucket] = dict(train_objects=count,
                train_percentage=100*count/train['objects']['Without Helmet'],
                validation_objects=denominator, missed_objects=missed,
                failure_percentage=100*missed/summary['objects'] if summary['objects'] else 0,
                validation_miss_rate=missed/denominator if denominator else None)
    return output
