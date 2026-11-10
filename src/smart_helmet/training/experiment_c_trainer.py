"""C inherits B's attachment, batch-order, optimizer, and TEST-lock checks."""
import json
from pathlib import Path

from .experiment_b_trainer import ExperimentBTrainer, weighted_loader
from .hard_example_sampler import HardExampleSampler, ordered_inventory
from .experiment import write_json


class ExperimentCTrainer(ExperimentBTrainer):
    def __init__(self, *args, sampling_plan, **kwargs):
        self.sampling_plan = sampling_plan
        super().__init__(*args, **kwargs)

    def get_dataloader(self, dataset_path, batch_size=4, rank=-1, mode='train'):
        if mode != 'train':
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        expected = self.project_root/'data/curated/v1/train/images'
        if Path(dataset_path).resolve() != expected:
            raise ValueError('TRAIN loader path mismatch; TEST locked')
        if rank != -1 or self.args.workers != 0 or self.args.rect or self.args.compile or self.args.fraction != 1:
            raise ValueError('C requires single-device full TRAIN, workers=0, rect=False, compile=False')
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        rows = ordered_inventory(self.project_root, self.inventory, dataset.im_files)
        # Keep the inherited callback attribute; it now holds the C sampler.
        self.class_aware_sampler = HardExampleSampler(self.project_root, rows, self.sampling_plan)
        self.path_to_index = {str((self.project_root/r['image']).resolve()): i for i,r in enumerate(rows)}
        return weighted_loader(dataset, self.class_aware_sampler, batch_size, self.device)

    def verify_attachment(self, trainer):
        super().verify_attachment(trainer)
        path = self.save_dir/'baseline_argument_comparison.json'
        report = json.loads(path.read_text())
        report['sampling_strategy']['actual'] = 'frozen TRAIN hard-example weighted sampling'
        write_json(path, report)
