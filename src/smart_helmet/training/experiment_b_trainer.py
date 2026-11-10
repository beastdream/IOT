"""Local Ultralytics 8.4.128 extension: only TRAIN sampling differs."""
import json
from pathlib import Path

import torch
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK

from smart_helmet.evaluation.human_review import atomic_text
from .class_aware_sampler import ClassAwareSampler, ordered_inventory
from .trainer import training_details


def weighted_loader(dataset, sampler, batch=4, device='cpu'):
    if batch != 4:
        raise ValueError('Experiment B requires batch=4')
    # Same generator formula, collate, workers, pinning and drop_last as installed
    # build_dataloader for baseline single-device / workers=0 / compile=False.
    seed = torch.initial_seed() - RANK - 1
    generator = torch.Generator().manual_seed((6148914691236517205 + RANK + seed) % (1 << 64))
    return InfiniteDataLoader(dataset, batch_size=min(batch, len(dataset)), shuffle=False,
        sampler=sampler, num_workers=0, prefetch_factor=None,
        pin_memory=str(device).split(':')[0] == 'cuda',
        collate_fn=getattr(dataset, 'collate_fn', None), worker_init_fn=seed_worker,
        generator=generator, drop_last=False)


def compare_baseline_arguments(actual, baseline, smoke=False):
    # Dataset paths are independently checked by get_dataset/get_dataloader.
    exceptions = {'data','project','name','save_dir'}
    if smoke:
        exceptions.add('epochs')
        if actual.get('epochs') != 2:
            raise ValueError('Smoke requires exactly 2 epochs')
    fields = {k:dict(baseline=baseline.get(k), actual=actual.get(k),
                    status='EXPECTED DIFFERENCE' if k in exceptions and baseline.get(k) != actual.get(k)
                    else 'MATCH' if k in baseline and k in actual and baseline[k] == actual[k] else 'DRIFT')
              for k in sorted(set(baseline)|set(actual))}
    mismatches = {k:(v['baseline'], v['actual']) for k,v in fields.items()
                  if v['status'] == 'DRIFT'}
    if mismatches:
        raise ValueError('Baseline argument drift: '+json.dumps(mismatches, default=str))
    return dict(result='PASS', fields=fields, unexpected_drift=mismatches,
                sampling_strategy=dict(baseline='normal', actual='class-aware weighted sampling',
                                       status='EXPECTED DIFFERENCE'))


class ExperimentBTrainer(DetectionTrainer):
    def __init__(self, *args, project_root, inventory, baseline_args, baseline_optimizer, smoke=False, **kwargs):
        self.project_root = Path(project_root).resolve()
        self.inventory = inventory
        self.baseline_args = baseline_args
        self.baseline_optimizer = baseline_optimizer
        self.smoke_run = smoke
        self.exposure_epochs = []
        self.observed_indices = []
        self.completed_optimizer_steps = 0
        self.completed_batches = 0
        super().__init__(*args, **kwargs)
        self.add_callback('on_pretrain_routine_end', self.verify_attachment)
        self.add_callback('on_train_epoch_start', self.start_sampling_epoch)
        self.add_callback('on_train_batch_end', self.record_completed_batch)
        self.add_callback('on_train_epoch_end', self.finish_sampling_epoch)

    def get_dataset(self):
        # Validate before framework parsing can take its val-or-test fallback.
        import yaml
        data = yaml.safe_load(Path(self.args.data).read_text(encoding='utf-8'))
        if 'test' in data or 'download' in data or self.args.split != 'val':
            raise ValueError('TEST locked: runtime YAML must omit test/download and select val')
        base = self.project_root/'data/curated/v1'
        for split, folder in (('train','train'), ('val','valid')):
            if Path(data[split]).resolve() != base/folder/'images':
                raise ValueError('Unexpected TRAIN/VALID dataset path')
        return super().get_dataset()

    def get_dataloader(self, dataset_path, batch_size=4, rank=-1, mode='train'):
        if mode not in ('train','val'):
            raise ValueError('TEST locked: loader mode must be train or val')
        expected = self.project_root/'data/curated/v1'/('train' if mode == 'train' else 'valid')/'images'
        if Path(dataset_path).resolve() != expected:
            raise ValueError('TRAIN/VALID loader path mismatch; TEST is locked')
        if mode == 'val':
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        if rank != -1 or self.args.workers != 0 or self.args.rect or self.args.compile or self.args.fraction != 1:
            raise ValueError('Experiment B requires single-device full TRAIN, workers=0, rect=False, compile=False')
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        rows = ordered_inventory(self.project_root, self.inventory, dataset.im_files)
        self.class_aware_sampler = ClassAwareSampler(self.project_root, rows)
        self.path_to_index = {str((self.project_root/r['image']).resolve()): i for i,r in enumerate(rows)}
        return weighted_loader(dataset, self.class_aware_sampler, batch_size, self.device)

    def verify_attachment(self, trainer):
        if self.train_loader.sampler is not self.class_aware_sampler:
            raise ValueError('Custom sampler is not attached to TRAIN loader')
        if isinstance(self.test_loader.sampler, ClassAwareSampler):
            raise ValueError('Validation must not use class-aware sampling')
        report = compare_baseline_arguments(vars(self.args), self.baseline_args, self.smoke_run)
        actual = json.loads(json.dumps(training_details(self)))
        for key, value in self.baseline_optimizer.items():
            if actual[key] != value:
                raise ValueError('Baseline optimizer mismatch: '+key)
        report['optimizer'] = dict(status='MATCH', **actual)
        atomic_text(self.save_dir/'baseline_argument_comparison.json', json.dumps(report, indent=2)+'\n')
        self.save_diagnostics(self.class_aware_sampler.diagnostics([], False))

    def start_sampling_epoch(self, trainer):
        self.class_aware_sampler.set_epoch(self.epoch)
        self.observed_indices = []
        # workers=0 means there are no prefetched samples; the exhausted iterator
        # advances to the newly set epoch. Framework mosaic reset replays it.

    def preprocess_batch(self, batch):
        for path in batch['im_file']:
            key = str(Path(path).resolve())
            if key not in self.path_to_index:
                raise ValueError('Non-TRAIN primary image reached optimizer batch')
            self.observed_indices.append(self.path_to_index[key])
        return super().preprocess_batch(batch)

    def optimizer_step(self):
        super().optimizer_step()
        self.completed_optimizer_steps += 1

    def record_completed_batch(self, trainer):
        if not torch.isfinite(self.loss).all():
            raise ValueError('Nonfinite training loss')
        self.completed_batches += 1

    def finish_sampling_epoch(self, trainer):
        expected = self.class_aware_sampler.indices()
        if self.observed_indices != expected or self.class_aware_sampler.yielded_indices != expected:
            raise ValueError('Actual optimizer batch order differs from weighted sampler')
        if self.train_loader.sampler is not self.class_aware_sampler or self.completed_optimizer_steps < 1:
            raise ValueError('Sampler attachment or backward/optimizer evidence missing')
        diagnostic = self.class_aware_sampler.diagnostics(self.observed_indices, True)
        diagnostic.update(completed_batches=self.completed_batches,
                          completed_optimizer_steps=self.completed_optimizer_steps,
                          device=str(self.device), sampler_attached=True)
        self.exposure_epochs.append(diagnostic)
        self.save_diagnostics(self.exposure_epochs[0])

    def save_diagnostics(self, first):
        payload = dict(first, epochs=self.exposure_epochs,
            baseline_constants_preserved=True, optimizer_matches_baseline=True)
        atomic_text(self.save_dir/'sampler_diagnostics.json', json.dumps(payload, indent=2)+'\n')
