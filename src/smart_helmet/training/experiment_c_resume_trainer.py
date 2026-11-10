"""Pinned framework resume with C's original TRAIN sampler and VALID guards."""
from copy import deepcopy
from unittest.mock import patch
from uuid import uuid4

import torch
from ultralytics.cfg import get_cfg

from .experiment_c_trainer import ExperimentCTrainer
from .experiment_c_resume import check_arguments, atomic_json
from .trainer import training_details


def state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        return isinstance(actual, torch.Tensor) and torch.equal(actual.cpu(), expected.cpu())
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(state_equal(actual[k], v) for k,v in expected.items())
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(state_equal(a,b) for a,b in zip(actual, expected))
    return actual == expected


class ResumeExperimentCTrainer(ExperimentCTrainer):
    def __init__(self, *args, resume_context, **kwargs):
        self.resume_context = resume_context
        super().__init__(*args, **kwargs)
        self.exposure_epochs = deepcopy(resume_context['epochs'])
        self.completed_batches = self.exposure_epochs[-1]['completed_batches']
        self.completed_optimizer_steps = self.exposure_epochs[-1]['completed_optimizer_steps']

    def check_resume(self, overrides):
        last = self.resume_context['out']/'weights/last.pt'
        if overrides != {'resume':str(last)} or not last.is_file():
            raise ValueError('Only the validated C last.pt may be resumed')
        saved = self.resume_context['checkpoint']['train_args']
        check_arguments(saved, self.baseline_args, self.resume_context['out'])
        self.args = get_cfg(saved)
        self.args.model = self.args.resume = str(last)
        self.resume = True

    def resume_training(self, checkpoint):
        if checkpoint is None or checkpoint['epoch']+1 != self.resume_context['completed']:
            raise ValueError('Checkpoint epoch changed during setup')
        # Scheduler setup has now supplied initial_lr, as in the baseline report.
        # Compare BEFORE loading the progressed optimizer's LR/state.
        import json
        actual = json.loads(json.dumps(training_details(self)))
        if any(actual[k] != v for k,v in self.baseline_optimizer.items()):
            raise ValueError('Baseline optimizer initialization mismatch')
        super().resume_training(checkpoint)
        if not state_equal(self.optimizer.state_dict(), checkpoint['optimizer']):
            raise ValueError('Optimizer state was not restored exactly')
        if not state_equal(self.scaler.state_dict(), checkpoint['scaler']):
            raise ValueError('AMP scaler state was not restored')
        self.stopper.__dict__.update(self.resume_context['stopper'])

    def verify_attachment(self, trainer):
        context = self.resume_context
        if self.save_dir.resolve() != context['out'] or self.start_epoch != context['completed']:
            raise ValueError('Resume changed output directory or epoch')
        if self.train_loader.sampler is not self.class_aware_sampler or isinstance(self.test_loader.sampler, type(self.class_aware_sampler)):
            raise ValueError('TRAIN sampler attachment / VALID isolation failed')
        report = check_arguments(vars(self.args), self.baseline_args, context['out'])
        saved = context['checkpoint'].get('c_resume_state', {}).get('scheduler')
        if saved is not None:
            self.scheduler.load_state_dict(saved)
        else:
            # LambdaLR is epoch-only in the pinned framework; resume_training already
            # restored optimizer group LRs. Retain its original total-epoch lambda.
            self.scheduler.last_epoch = self.start_epoch-1
            self.scheduler._step_count = self.start_epoch+1
            self.scheduler._last_lr = [g['lr'] for g in self.optimizer.param_groups]
        if self.scheduler.last_epoch != self.start_epoch-1:
            raise ValueError('Scheduler epoch restoration failed')
        self.class_aware_sampler.set_epoch(self.start_epoch)
        self.train_loader.reset()
        atomic_json(self.save_dir/'resume_runtime_verification.json', dict(
            result='PASS', optimizer_state='RESTORED', scaler_state='RESTORED',
            scheduler_state=self.scheduler.state_dict(), early_stopping=vars(self.stopper),
            start_epoch_zero_based=self.start_epoch, sampler_seed_policy='seed + zero_based_epoch',
            baseline_arguments=report['result'], test_status='LOCKED'))

    def save_model(self):
        """Add resumable control state to the SAME checkpoint, then atomically replace."""
        original_save = torch.save
        def save(payload, *args, **kwargs):
            payload['c_resume_state'] = dict(scheduler=self.scheduler.state_dict(),
                                             early_stopping=vars(self.stopper).copy())
            return original_save(payload, *args, **kwargs)
        last, best = self.last, self.best
        suffix = '.'+uuid4().hex+'.pending'
        self.last, self.best = last.with_suffix(suffix), best.with_suffix(suffix)
        try:
            with patch.object(torch, 'save', save):
                result = super().save_model()
            if self.last.is_file():
                self.last.replace(last)
            if self.best.is_file():
                self.best.replace(best)
            return result
        finally:
            self.last, self.best = last, best
