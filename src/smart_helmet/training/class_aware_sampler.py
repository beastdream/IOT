"""TRAIN-only weighted sampler, ordered against the actual framework dataset."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .sampling import MULTIPLIER, SEED, TrainExposureSampler, exposure, train_path


def ordered_inventory(root, rows, image_files):
    inventory = {str(train_path(root, row['image'])): row for row in rows}
    paths = [str(train_path(root, path)) for path in image_files]
    if len(inventory) != len(rows) or len(paths) != len(set(paths)) or set(paths) != set(inventory):
        raise ValueError('Framework TRAIN inventory differs from frozen sampling inventory')
    return [inventory[p] for p in paths]


def index_hash(indices):
    return hashlib.sha256(json.dumps(indices, separators=(',', ':')).encode()).hexdigest()


class ClassAwareSampler(TrainExposureSampler):
    """An epoch-explicit sampler: iterator resets replay the same epoch.

    Trainer's epoch-start callback sets the epoch; unlike implicit advancement,
    this remains deterministic when InfiniteDataLoader.reset closes mosaic.
    """
    replacement = True

    def __init__(self, root, rows, seed=SEED, multiplier=MULTIPLIER):
        super().__init__(root, rows, seed, multiplier)
        self.rows = rows
        self.multiplier = multiplier
        self.num_samples = len(rows)
        self.iteration_count = 0
        self.yielded_indices = []

    def indices(self, epoch=None):
        rng = random.Random(self.seed + (self.epoch if epoch is None else epoch))
        return rng.choices(range(len(self)), weights=self.weights, k=len(self))

    def __iter__(self):
        self.iteration_count += 1
        self.yielded_indices = []
        for index in self.indices():
            self.yielded_indices.append(index)
            yield index

    def diagnostics(self, indices, actually_used=False):
        counts = Counter(indices)
        positives = sum(self.rows[i]['counts'][1] > 0 for i in indices)
        return dict(sampler_class=type(self).__name__, replacement=self.replacement,
            num_samples=self.num_samples, seed=self.seed, multiplier=self.multiplier,
            unique_train_images=len(self.rows),
            images_with_without_helmet=sum(r['counts'][1] > 0 for r in self.rows),
            sampled_without_helmet_image_count=positives,
            sampled_other_image_count=len(indices)-positives, sampled_index_hash=index_hash(indices),
            sampled_unique_images=len(counts), actual_sample_count=len(indices),
            estimated_object_exposure=exposure(self.rows, [counts[i] for i in range(len(self.rows))]),
            exposure_scope='Primary TRAIN draws; unchanged mosaic auxiliary draws excluded',
            sampler_actually_used=actually_used, epoch=self.epoch,
            valid_leakage=0, test_leakage=0, test_used=False)
