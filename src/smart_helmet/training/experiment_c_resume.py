"""Fail-closed continuation of the owned full C run; never prepare a new output."""
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import yaml

from smart_helmet.paths import PROJECT_ROOT
from . import experiment_c as pipeline
from .environment import check_unchanged
from .experiment import output_path
from .experiment_b import file_hash, tree_hashes, baseline_runtime_signature
from .hard_example_sampler import PLAN, HardExampleSampler, ordered_inventory
from .sampling import source_evidence


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    from smart_helmet.evaluation.human_review import atomic_text
    atomic_text(Path(path), json.dumps(value, indent=2, default=str)+'\n')


def recover_epochs(path, checkpoint, requested):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(stream)]
    epoch = checkpoint.get('epoch')
    if type(epoch) is not int or epoch < 0:
        raise ValueError('Checkpoint is completed/stripped or has no recoverable epoch')
    completed = epoch + 1  # checkpoint is zero-based; CSV and display are one-based
    if not 0 < completed < requested:
        raise ValueError('Run has no remaining epochs')
    if [int(r['epoch']) for r in rows] != list(range(1, completed+1)):
        raise ValueError('CSV/checkpoint epoch disagreement; refusing to truncate or duplicate epochs')
    saved = checkpoint.get('train_results', {})
    if saved.get('epoch') != list(range(1, completed+1)):
        raise ValueError('Checkpoint history does not cover completed epochs')
    for key, values in saved.items():
        if len(values) != completed or any(not math.isfinite(float(v)) or float(r[key]) != float(v)
                                          for r, v in zip(rows, values)):
            raise ValueError('CSV/checkpoint history mismatch: '+key)
    return completed


def stopping_state(checkpoint, patience, completed):
    """Pinned 8.4.128 detection fitness is saved mAP50-95, including its rounding."""
    saved = checkpoint.get('c_resume_state', {}).get('early_stopping')
    best, best_epoch = 0.0, 0
    for epoch, fitness in enumerate(checkpoint['train_results']['metrics/mAP50-95(B)'], 1):
        if fitness > best or best == 0:
            best, best_epoch = fitness, epoch
    expected = dict(best_fitness=best, best_epoch=best_epoch, patience=patience,
                    possible_stop=completed-best_epoch >= patience-1)
    if best != checkpoint['best_fitness'] or (saved is not None and saved != expected):
        raise ValueError('Early-stopping history is ambiguous or inconsistent')
    if completed-best_epoch >= patience:
        raise ValueError('Early stopping already reached; run is finished')
    return saved or expected


def check_arguments(actual, baseline, out):
    from .experiment_b_trainer import compare_baseline_arguments
    expected_paths = dict(data=out/'training_data.yaml', project=out.parent, save_dir=out)
    if any(Path(actual[k]).resolve() != p for k, p in expected_paths.items()) or actual['name'] != pipeline.NAME:
        raise ValueError('Resume output/data path differs from the original C run')
    last = out/'weights/last.pt'
    if actual.get('resume') not in (False, str(last)):
        raise ValueError('Unexpected resume checkpoint')
    if Path(actual['model']).resolve() not in (Path(baseline['model']).resolve(), last):
        raise ValueError('Unexpected model checkpoint')
    # Only transport fields may differ on a subsequent resume, never training controls.
    normalized = dict(actual, model=baseline['model'], resume=baseline['resume'])
    return compare_baseline_arguments(normalized, baseline)


def validate_resume(root=PROJECT_ROOT, report=None):
    root = Path(root).resolve()
    out = output_path(root, pipeline.NAME)
    report = report if report is not None else {}
    last = out/'weights/last.pt'
    report['CHECKPOINT'] = str(last)
    if not last.is_file() or last.resolve() != last or not last.stat().st_size:
        raise ValueError('Missing or unsafe last.pt; no fallback is allowed')
    meta = read_json(out/'experiment.json')
    if meta.get('training_status') not in ('RUNNING', 'INTERRUPTED', 'FAIL') or meta['experiment_name'] != pipeline.NAME:
        raise ValueError('No interrupted full C run detected')
    import torch
    checkpoint = torch.load(last, map_location='cpu', weights_only=False)
    completed = recover_epochs(out/'results.csv', checkpoint, meta['epochs_requested'])
    report.update({'INTERRUPTED RUN DETECTED': 'YES', 'LAST COMPLETED EPOCH': completed,
                   'RESUME FROM EPOCH': completed+1})
    if not (checkpoint.get('optimizer') or {}).get('state') or checkpoint.get('ema') is None or not isinstance(checkpoint.get('scaler'), dict):
        raise ValueError('Checkpoint lacks optimizer/EMA/scaler state')
    state = pipeline.preflight(root=root)
    cfg, env, snapshot, rows, plan, baseline_args, baseline = state
    fingerprint = read_json(out/'dataset_fingerprint.json')
    if (meta['dataset_fingerprint'] != snapshot['curated_hash'] or
            fingerprint != {k:snapshot[k] for k in ('curated','raw','curated_hash','raw_hash')}):
        raise ValueError('Dataset fingerprint mismatch with interrupted run')
    report['DATASET FINGERPRINT'] = 'PASS'
    if (read_json(out/'sampling_plan_snapshot.json') != plan or
            file_hash(root/PLAN) != meta['sampling_plan_sha256'] or
            meta['human_review_sha256'] != plan['human_review_sha256']):
        raise ValueError('Phase 4I sampling plan mismatch')
    report['SAMPLING PLAN'] = 'PASS'
    if any(meta[k] != env[k] for k in ('torch_version','ultralytics_version')) or checkpoint['version'] != env['ultralytics_version']:
        raise ValueError('Resume runtime version changed')
    if meta['ultralytics_source_inspection'] != source_evidence():
        raise ValueError('Ultralytics source changed')
    # Only C lifecycle routing is changed by this resume feature. Sampler, baseline,
    # dataset and framework behavior must still match the interrupted implementation.
    hashes = pipeline.implementation_hashes(root)
    for name, digest in meta['implementation_sha256'].items():
        if name != 'experiment_c.py' and hashes.get(name) != digest:
            raise ValueError('Original training implementation changed: '+name)
    if meta['baseline_runtime_sha256'] != baseline_runtime_signature(baseline_args, baseline):
        raise ValueError('Baseline reference changed')
    for actual in (checkpoint['train_args'], meta['resolved_training_arguments'],
                   yaml.safe_load((out/'args.yaml').read_text()),
                   yaml.safe_load((out/'framework_resolved_args.yaml').read_text())):
        check_arguments(actual, baseline_args, out)
    original_cfg = yaml.safe_load((out/'resolved_config.yaml').read_text())
    if original_cfg != cfg:
        raise ValueError('Sampler configuration or controlled run configuration changed')
    report['BASELINE ARGUMENTS'] = 'PASS'
    runtime = yaml.safe_load((out/'training_data.yaml').read_text())
    expected_data = {k:snapshot['data'][k] for k in ('path','train','val','nc','names')}
    if runtime != expected_data or meta['test_status'] != 'LOCKED' or meta['validation_scope'] != 'Full VALID split':
        raise ValueError('TEST lock / unchanged VALID configuration failed')
    diagnostic = read_json(out/'sampler_diagnostics.json')
    epochs = diagnostic['epochs']
    if len(epochs) not in (completed, completed+1):
        raise ValueError('Sampler epoch evidence missing')
    sampler = HardExampleSampler(root, ordered_inventory(root, rows, diagnostic['ordered_train_images']), plan)
    for i, observed in enumerate(epochs):
        sampler.set_epoch(i)
        expected = sampler.diagnostics(sampler.indices(), True)
        if any(observed.get(k) != v for k,v in expected.items()):
            raise ValueError('Sampler configuration, seed policy or leakage mismatch')
        if not observed['sampler_attached'] or observed['completed_batches'] != (i+1)*290 or observed['device'] != 'cuda:0':
            raise ValueError('Sampler runtime evidence mismatch')
    if epochs[completed-1]['completed_optimizer_steps'] != checkpoint['updates']:
        raise ValueError('Optimizer progress disagrees with sampler evidence')
    stopper = stopping_state(checkpoint, cfg['patience'], completed)
    scheduler = checkpoint.get('c_resume_state', {}).get('scheduler')
    if scheduler is not None and (scheduler.get('last_epoch') != completed-1 or
            scheduler.get('base_lrs') != [g['initial_lr'] for g in checkpoint['optimizer']['param_groups']] or
            scheduler.get('_last_lr') != [g['lr'] for g in checkpoint['optimizer']['param_groups']]):
        raise ValueError('Scheduler state disagrees with checkpoint epoch/optimizer')
    report.update({'SAMPLER CONFIGURATION':'PASS', 'VALID LEAKAGE':0, 'TEST LEAKAGE':0,
                   'TEST STATUS':'LOCKED', 'READY TO RESUME':'YES'})
    return dict(state=state, out=out, metadata=meta, completed=completed,
                epochs=epochs[:completed], stopper=stopper, checkpoint=checkpoint,
                checkpoint_sha256=file_hash(last), report=report)


def print_report(report):
    for key in ('INTERRUPTED RUN DETECTED','LAST COMPLETED EPOCH','RESUME FROM EPOCH','CHECKPOINT',
                'DATASET FINGERPRINT','SAMPLING PLAN','BASELINE ARGUMENTS','SAMPLER CONFIGURATION',
                'VALID LEAKAGE','TEST LEAKAGE','TEST STATUS','READY TO RESUME'):
        print(f'{key}:\n{report.get(key, "NOT VERIFIED")}\n', flush=True)


def execute_resume(validated, root=PROJECT_ROOT):
    """Caller holds the run lock. Revalidate all inputs immediately before mutation."""
    root = Path(root).resolve()
    current = validate_resume(root)
    if current['checkpoint_sha256'] != validated['checkpoint_sha256']:
        raise ValueError('Checkpoint changed after validation')
    from .experiment_c_resume_trainer import ResumeExperimentCTrainer
    from .trainer import frozen_cache, disable_oom_retry
    cfg, env, snapshot, rows, plan, args, baseline = current['state']
    out, meta = current['out'], current['metadata']
    protected = [root/'results/experiments/A_baseline', root/'results/experiments/B_class_aware_sampling', root/PLAN.parent]
    before = {str(p):tree_hashes(p) for p in protected}
    # A durable audit copy preserves all pre-resume metadata without touching A/B.
    audit = out/'resume_history'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    audit.mkdir(parents=True, exist_ok=False)
    import shutil
    for name in ('experiment.json','args.yaml','results.csv','sampler_diagnostics.json'):
        shutil.copy2(out/name, audit/name)
    atomic_json(audit/'validation.json', dict(current['report'], checkpoint_sha256=current['checkpoint_sha256']))
    meta['training_status'] = 'RUNNING'
    # Prior failures remain in the audited metadata, not as the current run error.
    meta.pop('error', None)
    meta.setdefault('resume_history', []).append(dict(from_epoch=current['completed']+1,
        checkpoint=str(out/'weights/last.pt'), checkpoint_sha256=current['checkpoint_sha256'],
        time=datetime.now(timezone.utc).isoformat()))
    atomic_json(out/'experiment.json', meta)
    try:
        check_unchanged(snapshot, root)
        with frozen_cache(out/'dataset_cache', [root/'data/curated/v1',root/'train',root/'valid',root/'test']):
            trainer = ResumeExperimentCTrainer(project_root=root, inventory=rows, sampling_plan=plan,
                baseline_args=args, baseline_optimizer={k:baseline[k] for k in
                    ('optimizer_class','optimizer_initial_parameter_groups','parameter_count')},
                resume_context=current, overrides=dict(resume=str(out/'weights/last.pt')))
            trainer.add_callback('on_train_epoch_start', disable_oom_retry)
            trainer.train()
        if trainer.validator.seen != 127 or len(trainer.exposure_epochs) != trainer.epoch+1:
            raise ValueError('Full VALID or sampler completion evidence missing')
        if not all(math.isfinite(float(v)) for v in trainer.metrics.values()):
            raise ValueError('Nonfinite validation metrics')
        meta.update(training_status='PASS', epochs_completed=trainer.epoch+1,
            best_model_path=str(trainer.best), last_model_path=str(trainer.last),
            validation_metrics=trainer.metrics, validation_images=127, sampler_actually_used=True,
            effective_train_samples=1158, completed_optimizer_steps=trainer.completed_optimizer_steps)
    except BaseException as error:
        meta.update(training_status='INTERRUPTED', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            meta['dataset_immutability'] = check_unchanged(snapshot, root)
            if any(tree_hashes(p) != before[str(p)] for p in protected):
                raise ValueError('Protected A/B/mining artifacts changed')
            meta['protected_artifacts_immutability'] = 'PASS'
        except BaseException:
            meta['training_status'] = 'FAIL'
            raise
        finally:
            meta['end_time'] = datetime.now(timezone.utc).isoformat()
            atomic_json(out/'experiment.json', meta)
    return meta


def resume_main(dry_run=False, root=PROJECT_ROOT):
    report = {'INTERRUPTED RUN DETECTED':'NO', 'READY TO RESUME':'NO'}
    # OS lock is released even on process death or machine shutdown; no stale PID lock.
    out = output_path(root, pipeline.NAME)
    try:
        if not out.is_dir():
            raise ValueError('Existing C output is missing')
        import msvcrt
        with (out/'.resume.lock').open('a+b') as lock:
            lock.seek(0)
            if not lock.read(1):
                lock.write(b'0'); lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                validated = validate_resume(root, report)
                print_report(report)
                if not dry_run:
                    execute_resume(validated, root)
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, ImportError, TypeError, AssertionError) as error:
        report['READY TO RESUME'] = 'NO'
        print_report(report)
        print(f'EXPERIMENT C RESUME: FAIL\n{error}', flush=True)
        return 1
