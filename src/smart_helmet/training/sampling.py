"""Prepared runtime sampler and audit only. No model or training entry point."""
from collections import Counter
import hashlib
from importlib.metadata import distribution
import json
import math
from pathlib import Path
import random
import sys

import yaml

from smart_helmet.paths import PROJECT_ROOT
from smart_helmet.foundation import IMAGE_EXTENSIONS
from smart_helmet.evaluation.human_review import atomic_text

MULTIPLIER = 1.573
SEED = 42
NAMES = ('With Helmet', 'Without Helmet')
OUT = Path('results/experiments/B_class_aware_sampling')


def train_path(root, value):
    root = Path(root).resolve()
    base = root/'data/curated/v1/train/images'
    # Reject a symlinked train directory too, not only individual escaping files.
    path = (root/value).resolve()
    if base.resolve() != base or path.parent != base or path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError('Sampling leakage: path must belong to curated v1 TRAIN images')
    if not path.is_file():
        raise ValueError('Missing TRAIN image')
    return path


def records(root):
    root = Path(root).resolve()
    result = []
    for image in sorted((root/'data/curated/v1/train/images').iterdir()):
        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image = train_path(root, image)
        label = image.parent.parent/'labels'/(image.stem+'.txt')
        if label.resolve().parent != root/'data/curated/v1/train/labels':
            raise ValueError('TRAIN label escapes curated TRAIN')
        counts = [0, 0]
        for line in label.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5 or fields[0] not in ('0', '1'):
                raise ValueError('Invalid TRAIN annotation')
            coords = [float(v) for v in fields[1:]]
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in coords) or min(coords[2:]) <= 0:
                raise ValueError('Invalid TRAIN box')
            counts[int(fields[0])] += 1
        result.append(dict(image=image.relative_to(root).as_posix(), counts=counts,
                           label_sha256=hashlib.sha256(label.read_bytes()).hexdigest()))
    if not result:
        raise ValueError('Empty TRAIN')
    return result


def weights(rows, multiplier=MULTIPLIER):
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('Multiplier must be positive and finite')
    return [multiplier if row['counts'][1] else 1.0 for row in rows]


class TrainExposureSampler:
    """PyTorch sampler protocol, using a private reproducible Python RNG.

    Each iteration advances one epoch. set_epoch supports explicit resume.
    Feed indices in dataset.im_files order after checking path identity.
    Prepared for a future custom InfiniteDataLoader; no trainer is invoked here.
    """
    def __init__(self, root, rows, seed=SEED, multiplier=MULTIPLIER):
        if not rows or len({r['image'] for r in rows}) != len(rows):
            raise ValueError('Expected unique nonempty TRAIN inventory')
        for row in rows:
            train_path(root, row['image'])
        self.weights = weights(rows, multiplier)
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.weights)

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError('Invalid epoch')
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed+self.epoch)
        self.epoch += 1
        return iter(rng.choices(range(len(self)), weights=self.weights, k=len(self)))


def exposure(rows, factors=None):
    factors = factors if factors is not None else [1]*len(rows)
    return {name: sum(r['counts'][i]*f for r,f in zip(rows, factors)) for i,name in enumerate(NAMES)}


def build_plan(root, rows, seed=SEED, multiplier=MULTIPLIER):
    sampler = TrainExposureSampler(root, rows, seed, multiplier)
    indices = list(sampler)
    repeated = list(TrainExposureSampler(root, rows, seed, multiplier))
    if indices != repeated:
        raise ValueError('Non-deterministic sampler')
    paths = [rows[i]['image'] for i in indices]
    for path in paths:
        train_path(root, path)
    w = weights(rows, multiplier)
    normalized = [len(rows)*v/sum(w) for v in w]
    realized = Counter(indices)
    return dict(strategy='runtime_weighted_sampling_with_replacement', seed=seed, multiplier=multiplier,
        original_train_images=len(rows), effective_train_samples=len(indices),
        images_with_without_helmet=sum(r['counts'][1] > 0 for r in rows),
        images_without_without_helmet=sum(r['counts'][1] == 0 for r in rows),
        class_counts_before_sampling=exposure(rows),
        estimated_class_exposure_after_sampling=exposure(rows, normalized),
        realized_epoch_zero_class_exposure=exposure(rows, [realized[i] for i in range(len(rows))]),
        expected_positive_image_draws=sum(f for r,f in zip(rows,normalized) if r['counts'][1]),
        realized_positive_image_draws=sum(rows[i]['counts'][1] > 0 for i in indices),
        weight_semantics='1.573 is positive/negative per-image relative weight, not an absolute 1.573x count increase. Normalize to unchanged epoch length.',
        exposure_scope='Primary sampler image draws only; mosaic auxiliary draws follow unchanged baseline augmentation and are not included.',
        replacement=True, epoch_seed='42 + zero-based epoch', python_version=sys.version.split()[0],
        sampled_paths_epoch_zero=paths, train_inventory=rows,
        valid_leakage=0, test_leakage=0, train_only='PASS', deterministic='PASS',
        no_disk_image_duplication=True, test_used=False, plan_only=True,
        training_authorized=False, ready_to_train=False)


def source_evidence():
    package = distribution('ultralytics')
    if package.version != '8.4.128':
        raise ValueError('Sampling integration was inspected only for Ultralytics 8.4.128')
    files = ('ultralytics/data/build.py', 'ultralytics/models/yolo/detect/train.py',
             'ultralytics/data/base.py', 'ultralytics/engine/trainer.py', 'ultralytics/engine/model.py')
    return dict(version=package.version, source_sha256={f: hashlib.sha256(Path(package.locate_file(f)).read_bytes()).hexdigest() for f in files})


def prepare(root=PROJECT_ROOT):
    root = Path(root)
    rows = records(root)
    plan = build_plan(root, rows)
    if (len(rows), plan['images_with_without_helmet'], plan['class_counts_before_sampling']) != (1158, 450, {'With Helmet':2009,'Without Helmet':1038}):
        raise ValueError('TRAIN inventory differs from verified baseline')
    baseline = yaml.safe_load((root/'results/experiments/A_baseline/args.yaml').read_text(encoding='utf-8'))
    constants = dict(imgsz=416, batch=4, workers=0, seed=42, epochs=100, patience=20)
    if any(baseline[k] != v for k,v in constants.items()) or Path(baseline['model']).name != 'yolo11n.pt':
        raise ValueError('Baseline constants differ')
    experiment = json.loads((root/'results/experiments/A_baseline/experiment.json').read_text(encoding='utf-8'))
    plan['ultralytics_source_inspection'] = source_evidence()
    plan['baseline_constants_preserved'] = True
    plan['dataset_fingerprint'] = experiment['dataset_fingerprint']
    plan['integration_status'] = 'PREPARED; custom trainer integration must be verified in a future authorized phase'
    plan['baseline_args_sha256'] = hashlib.sha256((root/'results/experiments/A_baseline/args.yaml').read_bytes()).hexdigest()
    # Phase 4E owns the executable config. A checker must not replace it with
    # the earlier planning-only schema or overwrite an actual training record.
    from .experiment_b import load_config, CONFIG
    load_config(root/CONFIG)
    plan['integration_status'] = 'LOCAL CUSTOM DETECTION TRAINER; see smoke readiness checker'
    if not (root/OUT/'experiment.json').exists():
        atomic_text(root/OUT/'sampling_plan.json', json.dumps(plan, indent=2)+'\n')
    return plan


def main():
    try:
        plan = prepare()
        print('='*40+'\nEXPERIMENT B SAMPLING\n'+'='*40)
        for label, key in [('Strategy','strategy'), ('Multiplier','multiplier'),
            ('Original TRAIN images','original_train_images'), ('Images containing Without Helmet','images_with_without_helmet'),
            ('Effective samples','effective_train_samples'), ('Original exposure','class_counts_before_sampling'),
            ('Estimated sampled exposure','estimated_class_exposure_after_sampling'),
            ('VALID leakage','valid_leakage'), ('TEST leakage','test_leakage'), ('DETERMINISTIC','deterministic')]:
            print(f'{label}: {plan[key]}')
        print('READY: YES (sampling plan only)\nREADY TO TRAIN: NO\n'+'='*40)
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f'EXPERIMENT B SAMPLING: FAIL\n{error}\nREADY: NO')
        return 1
