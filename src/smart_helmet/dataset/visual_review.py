"""Deterministic ground-truth review copies. Original images are read only."""

from collections import defaultdict
from pathlib import Path
import random

from PIL import Image, ImageDraw

from smart_helmet.foundation import EXPECTED_CLASSES

REVIEW_SEED = 42
SAMPLES_PER_CATEGORY = 24
COLORS = {0: '#00b85c', 1: '#ff483c'}


def draw_annotation(image_path: Path, boxes: list[dict]) -> Image.Image:
    with Image.open(image_path) as source:
        canvas = source.convert('RGB')
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    label_regions = []
    for box in boxes:
        k, x, y, bw, bh = (box[n] for n in ('class_id', 'x', 'y', 'normalized_width', 'normalized_height'))
        coords = ((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h)
        draw.rectangle(coords, outline=COLORS[k], width=2)
        label = f'{k}: {EXPECTED_CLASSES[k]}'
        text_width = draw.textbbox((0, 0), label)[2]
        tx = max(0, min(coords[0], w - text_width))
        ty = max(0, min(coords[1] - 13, h - 13))
        background = draw.textbbox((tx, ty), label)
        for offset in range(0, h, 13):
            candidate_y = (int(ty) + offset) % max(1, h - 13)
            candidate = draw.textbbox((tx, candidate_y), label)
            if not any(candidate[0] < r[2] and candidate[2] > r[0] and
                       candidate[1] < r[3] and candidate[3] > r[1] for r in label_regions):
                ty, background = candidate_y, candidate
                break
        label_regions.append(background)
        draw.rectangle(background, fill='black')
        draw.text((tx, ty), label, fill=COLORS[k])
    return canvas


def create_visual_review(records, boxes, source_groups, pairs, root, output, limit=SAMPLES_PER_CATEGORY):
    rng = random.Random(REVIEW_SEED)
    available = {r['image_path']: r for r in records if r['decode_ok']}
    by_image = defaultdict(list)
    for box in boxes:
        by_image[box['image']].append(box)

    def choose(values, count=limit):
        values = sorted(set(values))
        return rng.sample(values, min(count, len(values)))

    categories = {}
    for split in ('train', 'valid', 'test'):
        categories[f'random_{split}_samples'] = choose([p for p, r in available.items() if r['split'] == split])
    categories['tiny_bbox_samples'] = choose([b['image'] for b in boxes if b['size_category'] == 'tiny' and b['image'] in available])
    categories['without_helmet_samples'] = choose([p for p, r in available.items() if r['without_helmet_count']])
    categories['multi_object_samples'] = choose([p for p, r in available.items() if r['number_of_objects'] > 1])
    categories['unusual_resolution_samples'] = choose([p for p, r in available.items() if (r['image_width'], r['image_height']) != (416, 416)])
    variants = [g for g in source_groups if len(g) > 1]
    rng.shuffle(variants)
    categories['source_group_samples'] = []
    for group in variants:
        members = [r['image_path'] for r in group if r['image_path'] in available]
        if len(members) > 1:
            remaining = limit - len(categories['source_group_samples'])
            if remaining < 2:
                break
            categories['source_group_samples'].extend(members[:min(3, remaining)])
    paths = defaultdict(list)
    for category, selected in categories.items():
        directory = output / 'visual_review' / category
        directory.mkdir(parents=True, exist_ok=True)
        for n, name in enumerate(selected):
            target = directory / f'{n:02d}_{Path(name).name}.png'
            draw_annotation(root / name, by_image[name]).save(target)
            paths[name].append(target.relative_to(root).as_posix())
    directory = output / 'visual_review' / 'near_duplicate_samples'
    directory.mkdir(parents=True, exist_ok=True)
    # Cross-split pairs have priority, then fixed-seed within-split candidates.
    ordered = list(pairs)
    rng.shuffle(ordered)
    ordered.sort(key=lambda p: not p['cross_split'])
    for n, pair in enumerate(ordered[:limit // 2]):
        left, right = pair['image_a'], pair['image_b']
        images = [draw_annotation(root / p, by_image[p]) for p in (left, right)]
        for im in images:
            im.thumbnail((600, 600))
        canvas = Image.new('RGB', (sum(im.width for im in images), max(im.height for im in images) + 42), 'white')
        x = 0
        for im in images:
            canvas.paste(im, (x, 42)); x += im.width
        ImageDraw.Draw(canvas).text((5, 4), f"{pair['split_a']} | {pair['split_b']}  pHash distance={pair['distance']}\nGround truth; similarity candidate, not confirmed duplicate", fill='black')
        target = directory / f'pair_{n:02d}.png'
        canvas.save(target)
        for p in (left, right):
            paths[p].append(target.relative_to(root).as_posix())
    return paths
