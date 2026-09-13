"""Read-only YOLO dataset audit and evidence reports."""

from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics

from PIL import Image
import yaml

from smart_helmet.foundation import EXPECTED_CLASSES, IMAGE_EXTENSIONS, SPLIT_FOLDERS, load_detection_config
from smart_helmet.paths import PROJECT_ROOT, DATA_CONFIG, DATASET_AUDIT_DIR
from .duplicates import (sha256_file, source_group_id, perceptual_hash, grouped_records,
                         exact_duplicates, near_duplicates, connected_group_count)
from .visual_review import create_visual_review

EDGE_TOLERANCE = 1e-6
NEAR_ZERO_SIDE = 1e-4
AREA_THRESHOLDS = (0.001, 0.01, 0.1)
ASPECT_LIMITS = (0.2, 5.0)


def parse_yolo_line(line: str):
    """Return (numeric box or None, issues); warnings never correct input."""
    issues = []
    fields = line.split()
    if len(fields) != 5:
        return None, [('FIELD_COUNT', 'ERROR', f'Expected 5 fields, got {len(fields)}')]
    try:
        values = [float(v) for v in fields]
    except ValueError:
        return None, [('NON_NUMERIC', 'ERROR', line)]
    if not all(math.isfinite(v) for v in values):
        return None, [('NON_FINITE', 'ERROR', line)]
    k, x, y, w, h = values
    if not k.is_integer():
        return None, [('CLASS_NOT_INTEGER', 'ERROR', str(k))]
    if int(k) not in EXPECTED_CLASSES:
        return None, [('INVALID_CLASS_ID', 'ERROR', str(k))]
    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
        return None, [('INVALID_COORDINATES', 'ERROR', line)]
    if x - w / 2 < -EDGE_TOLERANCE or y - h / 2 < -EDGE_TOLERANCE or x + w / 2 > 1 + EDGE_TOLERANCE or y + h / 2 > 1 + EDGE_TOLERANCE:
        issues.append(('OUT_OF_BOUNDS', 'ERROR', line))
    if min(w, h) < NEAR_ZERO_SIDE:
        issues.append(('NEAR_ZERO', 'REVIEW', f'minimum normalized side < {NEAR_ZERO_SIDE}'))
    area = w * h
    category = next((name for name, threshold in zip(('tiny', 'small', 'medium'), AREA_THRESHOLDS) if area < threshold), 'large')
    return dict(class_id=int(k), x=x, y=y, normalized_width=w, normalized_height=h,
                normalized_area=area, aspect_ratio=w / h, size_category=category), issues


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(root: Path) -> dict:
    files = [root / n for n in ('data.yaml', 'README.dataset.txt', 'README.roboflow.txt')]
    for folder in SPLIT_FOLDERS.values():
        files.extend(p for p in (root / folder).rglob('*') if p.is_file())
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(files) if p.is_file()}


def calculate_status(fatal: bool, image_issues: list, annotation_issues: list, review: list) -> str:
    # Conservative gate: any structural/numeric integrity error blocks cleaning decisions.
    if fatal or any(i['severity'] in ('ERROR', 'CRITICAL') for i in image_issues + annotation_issues):
        return 'FAIL'
    return 'REVIEW_REQUIRED' if review else 'PASS'


def audit_fingerprint(root, config, config_path):
    dataset_root = Path(config['path']) if config else root
    hashes = fingerprint(dataset_root)
    if dataset_root == root:
        return hashes
    result = {(dataset_root / name).relative_to(root).as_posix(): digest for name,digest in hashes.items()}
    result[Path(config_path).resolve().relative_to(root).as_posix()] = sha256_file(Path(config_path))
    return result


def run_audit(root=PROJECT_ROOT, config_path=DATA_CONFIG, output=DATASET_AUDIT_DIR, visuals=True):
    root, output = Path(root).resolve(), Path(output).resolve()
    # Only allow generated reports below results, never alongside source data.
    if not output.is_relative_to(root / 'results') or output == root / 'results':
        raise ValueError('Audit output must be a subdirectory of project results/')
    output.mkdir(parents=True, exist_ok=True)
    manifest, image_issues, annotation_issues, boxes, suspicious, manual = [], [], [], [], [], []
    fatal = []
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    try:
        config = load_detection_config(config_path, root)
    except (OSError, ValueError, yaml.YAMLError) as error:
        config = None
        fatal.append(f'Config: {error}')
    if config and Path(config['path']) != root and output == root / 'results/dataset_audit':
        raise ValueError('Custom dataset requires a separate --output; raw audit evidence must not be overwritten')
    try:
        before = audit_fingerprint(root, config, config_path)
    except OSError as error:
        before = None
        fatal.append(f'Pre-audit fingerprint unavailable: {error}')

    def relative(path):
        return path.relative_to(root).as_posix()

    def image_issue(split, image, label, kind, details, severity='ERROR'):
        image_issues.append(dict(split=split, image=image, label=label, issue_type=kind, severity=severity, details=details))

    def review(split, image, reason, details, severity='REVIEW'):
        manual.append(dict(split=split, image_path=image, reason=reason, severity=severity,
                           details=details, visualization_path='', manual_review_required=True))

    if config:
        for key, split in SPLIT_FOLDERS.items():
            images = Path(config[key]); labels = images.parent / 'labels'
            if not images.is_dir() or not labels.is_dir():
                fatal.append(f'Missing split directories: {split}')
                image_issue(split, relative(images), relative(labels), 'MISSING_DIRECTORY', 'Images and labels directories required')
            if not images.is_dir():
                continue
            image_files = sorted(p for p in images.iterdir() if p.is_file())
            supported = [p for p in image_files if p.suffix.lower() in IMAGE_EXTENSIONS]
            stems = Counter(p.stem for p in supported)
            label_files = sorted(labels.glob('*.txt')) if labels.is_dir() else []
            for label in label_files:
                if label.stem not in stems:
                    image_issue(split, '', relative(label), 'ORPHAN_LABEL', 'No corresponding supported image')
            for path in image_files:
                name = relative(path); label = labels / (path.stem + '.txt'); lname = relative(label)
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    image_issue(split, name, lname, 'UNSUPPORTED_EXTENSION', path.suffix, 'REVIEW')
                    review(split, name, 'POSSIBLE_ANNOTATION_ISSUE', 'Unsupported file in image directory')
                    continue
                if stems[path.stem] > 1:
                    image_issue(split, name, lname, 'AMBIGUOUS_LABEL_PAIRING', 'Multiple images share the same label stem')
                row = dict(split=split, image_path=name, label_path=lname, filename=path.name,
                           source_group_id=source_group_id(path.name), image_width=None, image_height=None,
                           file_size=None, sha256='', perceptual_hash='', number_of_objects=0,
                           with_helmet_count=0, without_helmet_count=0, decode_ok=False,
                           label_ok=True, annotation_lines=0)
                manifest.append(row)
                try:
                    row['file_size'] = path.stat().st_size
                    row['sha256'] = sha256_file(path)
                    with Image.open(path) as im:
                        im.load()
                        if min(im.size) <= 0:
                            raise ValueError('Non-positive image dimension')
                        row['image_width'], row['image_height'] = im.size
                        row['perceptual_hash'] = perceptual_hash(im)
                        row['decode_ok'] = True
                except (OSError, ValueError, Image.DecompressionBombError) as error:
                    image_issue(split, name, lname, 'IMAGE_UNREADABLE', str(error))
                    review(split, name, 'POSSIBLE_ANNOTATION_ISSUE', 'Image cannot be decoded', 'HIGH')
                if row['decode_ok'] and (row['image_width'], row['image_height']) != (416, 416):
                    review(split, name, 'UNUSUAL_RESOLUTION', f"{row['image_width']}x{row['image_height']}; not automatically an error")
                if not label.is_file():
                    row['label_ok'] = False
                    image_issue(split, name, lname, 'MISSING_LABEL', 'Expected label does not exist')
                    continue
                try:
                    lines = label.read_text(encoding='utf-8-sig').splitlines()
                except (OSError, UnicodeError) as error:
                    row['label_ok'] = False
                    image_issue(split, name, lname, 'LABEL_UNREADABLE', str(error))
                    continue
                seen = set()
                for n, line in enumerate(lines, 1):
                    if not line.strip():
                        continue
                    row['annotation_lines'] += 1
                    box, issues = parse_yolo_line(line)
                    canonical = tuple(float(v) for v in line.split()) if box else line.strip()
                    if canonical in seen:
                        issues.append(('DUPLICATE_ANNOTATION', 'REVIEW', 'Repeated numeric annotation in the same label'))
                    seen.add(canonical)
                    for kind, severity, details in issues:
                        annotation_issues.append(dict(split=split, image=name, label=lname, line_number=n,
                                                      issue_type=kind, severity=severity, details=details))
                        review(split, name, 'POSSIBLE_ANNOTATION_ISSUE', f'Line {n}: {kind}', severity)
                    if any(i[1] == 'ERROR' for i in issues):
                        row['label_ok'] = False
                    if box is None:
                        continue
                    box.update(split=split, image=name, label=lname, line_number=n)
                    boxes.append(box)
                    row['number_of_objects'] += 1
                    row['with_helmet_count' if box['class_id'] == 0 else 'without_helmet_count'] += 1
                    reasons = []
                    if box['size_category'] == 'tiny':
                        reasons.append('TINY_BBOX')
                    if not ASPECT_LIMITS[0] <= box['aspect_ratio'] <= ASPECT_LIMITS[1]:
                        reasons.append('EXTREME_ASPECT_RATIO')
                    if any(i[0] == 'NEAR_ZERO' for i in issues):
                        reasons.append('NEAR_ZERO')
                    for reason in reasons:
                        suspicious.append(dict(**box, reason=reason, severity='REVIEW', manual_review_required=True))
                        review(split, name, reason, f"Line {n}; area={box['normalized_area']:.8g}; aspect={box['aspect_ratio']:.5g}")
                if row['number_of_objects'] > 1:
                    review(split, name, 'MULTIPLE_OBJECT_COMPLEX_SCENE', f"{row['number_of_objects']} parsed objects; check missed heads")
            if not supported:
                fatal.append(f'Empty image split: {split}')

    print(f'Scanned {len(manifest)} images. Comparing content hashes...', flush=True)
    exact = exact_duplicates(manifest)
    source = grouped_records(manifest, 'source_group_id')
    source_rows = []
    for group in source:
        splits = sorted({r['split'] for r in group}); cross = len(splits) > 1
        source_rows.append(dict(source_group_id=group[0]['source_group_id'], number_of_images=len(group),
                                splits='|'.join(splits), cross_split=cross, severity='HIGH' if cross else 'INFO',
                                images='|'.join(r['image_path'] for r in group)))
        if len(group) > 1:
            for r in group:
                review(r['split'], r['image_path'], 'SOURCE_VARIANTS', f"{r['source_group_id']}: {len(group)} files; source-name signal only")
                if cross:
                    review(r['split'], r['image_path'], 'CROSS_SPLIT_SIMILARITY', 'Shared source group across splits', 'HIGH')
    for group in exact:
        for name in group['images'].split('|'):
            review(next(r['split'] for r in manifest if r['image_path'] == name), name, 'EXACT_DUPLICATE', f"SHA-256 group: {group['sha256']}", group['severity'])
    near = near_duplicates(manifest)
    curated = bool(config and Path(config['path']) == root / 'data/curated/v1')
    reviewed_leakage = []
    if curated:
        from .curation import classify_curated_pairs
        try:
            reviewed_leakage = classify_curated_pairs(root, near)
        except (OSError, ValueError, KeyError) as error:
            fatal.append(f'Curated review provenance: {error}')
    reviewed_pair_keys = {tuple(sorted((r['image_a'],r['image_b']))) for r in reviewed_leakage
                          if r['review_status'] == 'REVIEWED_KEEP_BOTH_CANDIDATE'}
    for pair in near:
        if tuple(sorted((pair['image_a'],pair['image_b']))) in reviewed_pair_keys:
            continue
        reason = 'CROSS_SPLIT_SIMILARITY' if pair['cross_split'] else 'NEAR_DUPLICATE'
        for side, other in (('a', 'b'), ('b', 'a')):
            review(pair[f'split_{side}'], pair[f'image_{side}'], reason,
                   f"pHash distance {pair['distance']} to {pair[f'image_{other}']}; candidate only", pair['severity'])

    distribution = []
    for split in ('train', 'valid', 'test', 'overall'):
        selected = [r for r in manifest if split == 'overall' or r['split'] == split]
        n = len(selected); a = sum(r['with_helmet_count'] for r in selected); b = sum(r['without_helmet_count'] for r in selected)
        distribution.append(dict(split=split, number_of_images=n, number_of_annotations=a+b,
                                 with_helmet_objects=a, without_helmet_objects=b,
                                 images_containing_with_helmet=sum(r['with_helmet_count'] > 0 for r in selected),
                                 images_containing_without_helmet=sum(r['without_helmet_count'] > 0 for r in selected),
                                 objects_per_image=(a+b)/n if n else None,
                                 with_to_without_ratio=a/b if b else None,
                                 without_helmet_percentage=100*b/(a+b) if a+b else None))
    statistics_rows = []
    for k, name in EXPECTED_CLASSES.items():
        selected = [b for b in boxes if b['class_id'] == k]
        for metric in ('normalized_width', 'normalized_height', 'normalized_area', 'aspect_ratio'):
            values = [b[metric] for b in selected]
            statistics_rows.append(dict(class_id=k, class_name=name, metric=metric, count=len(values),
                                       minimum=min(values) if values else None, maximum=max(values) if values else None,
                                       mean=statistics.mean(values) if values else None,
                                       median=statistics.median(values) if values else None,
                                       **{size: sum(b['size_category'] == size for b in selected) for size in ('tiny','small','medium','large')}))
    resolutions = Counter(f"{r['image_width']}x{r['image_height']}" for r in manifest if r['decode_ok'])
    viz = {}
    if visuals:
        print('Generating deterministic ground-truth review copies...', flush=True)
        try:
            viz = create_visual_review(manifest, boxes, source, near, root, output)
        except (OSError, ValueError) as error:
            fatal.append(f'Visual review incomplete: {error}')
    for row in manual:
        row['visualization_path'] = '|'.join(viz.get(row['image_path'], []))
    try:
        after = audit_fingerprint(root, config, config_path)
        preserved = before is not None and before == after
    except OSError as error:
        after = None; preserved = False; fatal.append(f'Post-audit fingerprint unavailable: {error}')
    if not preserved:
        fatal.append('Dataset immutability could not be confirmed')

    img_counts = Counter(i['issue_type'] for i in image_issues)
    ann_counts = Counter(i['issue_type'] for i in annotation_issues)
    invalid_lines = {(i['label'], i['line_number']) for i in annotation_issues if i['severity'] == 'ERROR'}
    objects = Counter(b['class_id'] for b in boxes)
    size_counts = Counter(b['size_category'] for b in boxes)
    negative = {}
    for split in ('train', 'valid', 'test', 'overall'):
        rows = [r for r in manifest if split == 'overall' or r['split'] == split]
        known = [r for r in rows if r['label_ok']]
        negative[split] = dict(images_with_zero_objects=sum(r['number_of_objects'] == 0 for r in known),
                               images_with_one_object=sum(r['number_of_objects'] == 1 for r in known),
                               images_with_multiple_objects=sum(r['number_of_objects'] > 1 for r in known),
                               images_with_unknown_or_invalid_labels=len(rows)-len(known))
    report = dict(
        config_path=str(config_path.resolve()),
        dataset=dict(total_images=len(manifest), **{f'{s}_images': sum(r['split'] == s for r in manifest) for s in ('train','valid','test')},
                     total_annotations=len(boxes), total_nonblank_annotation_lines=sum(r['annotation_lines'] for r in manifest)),
        classes=dict(mapping=EXPECTED_CLASSES, counts={k: objects[k] for k in EXPECTED_CLASSES},
                     percentages={k:100*objects[k]/len(boxes) if boxes else 0 for k in EXPECTED_CLASSES}),
        images=dict(corrupted=img_counts['IMAGE_UNREADABLE'], missing_labels=img_counts['MISSING_LABEL'], orphan_labels=img_counts['ORPHAN_LABEL'],
                    unusual_resolutions=sum(r['decode_ok'] and (r['image_width'],r['image_height']) != (416,416) for r in manifest), issue_counts=dict(img_counts)),
        annotations=dict(invalid=len(invalid_lines), out_of_bounds=ann_counts['OUT_OF_BOUNDS'], near_zero=ann_counts['NEAR_ZERO'],
                         suspicious=len({(b['label'],b['line_number']) for b in suspicious}), issue_counts=dict(ann_counts)),
        duplicates=dict(exact_groups=len(exact), exact_cross_split=sum(g['cross_split'] for g in exact),
                        near_duplicate_groups=connected_group_count(near), near_duplicate_pairs=len(near),
                        near_duplicate_cross_split=sum(p['cross_split'] for p in near), source_variant_groups=sum(len(g)>1 for g in source)),
        source_groups=dict(number_of_source_groups=len(source), groups_with_multiple_images=sum(len(g)>1 for g in source),
                           largest_group_size=max((len(g) for g in source),default=0), extra_files=sum(len(g)-1 for g in source),
                           cross_split_groups=sum(g['cross_split'] for g in source_rows)),
        bbox={s:size_counts[s] for s in ('tiny','small','medium','large')}, image_resolutions=dict(resolutions), negative_images=negative,
        manual_review=dict(count=len(manual), unique_images=len({r['image_path'] for r in manual}),
                           visualized_unique_images=len(viz), manual_review_required=bool(manual)),
        immutability=dict(status='PASS' if preserved else 'FAIL', files_checked=len(before or {})),
        thresholds=dict(edge_tolerance=EDGE_TOLERANCE, near_zero_side=NEAR_ZERO_SIDE, area_thresholds=AREA_THRESHOLDS,
                        extreme_aspect_ratio=ASPECT_LIMITS, phash_very_similar_max=4, phash_possibly_similar_max=8, seed=42),
        methodology=dict(phash='32x32 grayscale DCT; top-left 8x8, median of AC coefficients, DC bit zero; 63 effective bits',
                         near_groups='Connected components of all candidate pairs; not confirmed duplicate equivalence classes',
                         cross_split_units='Exact/source counts are groups; near cross-split count is pairs',
                         annotation_counts='Numerically parseable in-range boxes including flagged edge violations and duplicate rows; malformed rows excluded',
                         negative_images='Empty readable valid labels only; absence of targets is not visually confirmed',
                         manual_count='One record per reason/evidence; unique_images counts distinct candidates',
                         status_policy='Any structure/config/image/annotation ERROR => FAIL; candidates => REVIEW_REQUIRED; otherwise PASS'),
        fatal_errors=fatal,
        status=calculate_status(bool(fatal), image_issues, annotation_issues, manual))
    if curated:
        report['cross_split_review'] = dict(
            reviewed_keep_both=sum(r['review_status'] == 'REVIEWED_KEEP_BOTH_CANDIDATE' for r in reviewed_leakage),
            unreviewed=sum(r['review_status'] == 'UNREVIEWED_CROSS_SPLIT_CANDIDATE' for r in reviewed_leakage))

    tables = {
        'dataset_manifest': (manifest, 'split image_path label_path filename source_group_id image_width image_height file_size sha256 perceptual_hash number_of_objects with_helmet_count without_helmet_count decode_ok label_ok annotation_lines'),
        'image_issues': (image_issues, 'split image label issue_type severity details'),
        'annotation_issues': (annotation_issues, 'split image label line_number issue_type severity details'),
        'bbox_statistics': (statistics_rows, 'class_id class_name metric count minimum maximum mean median tiny small medium large'),
        'bounding_boxes': (boxes, 'split image label line_number class_id x y normalized_width normalized_height normalized_area aspect_ratio size_category'),
        'suspicious_bboxes': (suspicious, 'split image label line_number class_id normalized_width normalized_height normalized_area aspect_ratio reason severity manual_review_required'),
        'class_distribution': (distribution, 'split number_of_images number_of_annotations with_helmet_objects without_helmet_objects images_containing_with_helmet images_containing_without_helmet objects_per_image with_to_without_ratio without_helmet_percentage'),
        'exact_duplicates': (exact, 'sha256 count splits cross_split severity images'),
        'source_groups': (source_rows, 'source_group_id number_of_images splits cross_split severity images'),
        'source_group_leakage': ([g for g in source_rows if g['cross_split']], 'source_group_id number_of_images splits cross_split severity images'),
        'near_duplicates': (near, 'image_a image_b split_a split_b distance similarity perceptual_match cross_split severity'),
        'near_duplicate_leakage': ([p for p in near if p['cross_split']], 'image_a image_b split_a split_b distance similarity perceptual_match cross_split severity'),
        'image_resolutions': ([dict(resolution=k,count=v) for k,v in sorted(resolutions.items())], 'resolution count'),
        'manual_review': (manual, 'split image_path reason severity details visualization_path manual_review_required'),
        'negative_images': ([dict(split=k,**v) for k,v in negative.items()], 'split images_with_zero_objects images_with_one_object images_with_multiple_objects images_with_unknown_or_invalid_labels'),
    }
    for name, (rows, fields) in tables.items():
        write_csv(output / f'{name}.csv', rows, fields.split())
    if curated:
        write_csv(output / 'cross_split_review.csv', reviewed_leakage,
                  'image_a image_b split_a split_b distance review_status review_id'.split())
    (output / 'dataset_fingerprint.json').write_text(json.dumps(dict(before=before, after=after, status=report['immutability']['status']), indent=2), encoding='utf-8')
    (output / 'dataset_audit_report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


def print_report(report):
    print('=' * 43 + '\nSMART HELMET DATASET AUDIT\n' + '=' * 43)
    print(f"Dataset config: {report.get('config_path', DATA_CONFIG)}")
    for title, values in (
        ('Images', report['dataset']), ('Annotations', report['classes']),
        ('Integrity', dict(**report['images'], invalid_annotations=report['annotations']['invalid'])),
        ('Bounding Boxes', report['bbox']), ('Duplicates', report['duplicates']),
        ('Manual Review', report['manual_review']),
    ):
        print(title + '\n' + '-' * 43)
        for key, value in values.items():
            print(f'{key}: {value}')
    print('AUDIT STATUS:\n' + report['status'] + '\n' + '=' * 43)
