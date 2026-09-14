"""Configurable YOLO baseline; TRAIN/VALID only, no dataset mutations."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from unittest.mock import patch

import yaml

from smart_helmet.paths import PROJECT_ROOT
from .environment import runtime_info, verify_dataset, resolve_batch, check_unchanged
from .experiment import output_path, prepare_output, write_json, write_yaml, discover_checkpoints, result_rows

AUGMENTATION_POLICY = 'Baseline uses Ultralytics framework-default training augmentation for the installed version.'


def load_config(path, overrides=None):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))
    required = {'experiment_name','data','model','imgsz','epochs','patience','seed','deterministic',
                'device','workers','cache','pretrained','batch','smoke_epochs','smoke_fraction'}
    if not isinstance(cfg, dict) or set(cfg) != required:
        raise ValueError('Baseline config must contain exactly: '+', '.join(sorted(required)))
    cfg.update({k:v for k,v in (overrides or {}).items() if v is not None})
    if cfg['model'] != 'yolo11n.pt': raise ValueError('This baseline supports the verified official yolo11n.pt detection checkpoint')
    if cfg['imgsz'] != 416: raise ValueError('Phase 4A baseline requires imgsz=416')
    if cfg['cache'] is not False or cfg['pretrained'] is not True or cfg['deterministic'] is not True:
        raise ValueError('Baseline requires cache=false, pretrained=true, deterministic=true')
    for key in ('epochs','patience','workers','seed','smoke_epochs'):
        if type(cfg[key]) is not int or cfg[key] < (1 if key in ('epochs','smoke_epochs') else 0):
            raise ValueError(f'Invalid integer {key}')
    if cfg['smoke_epochs'] > 2: raise ValueError('Smoke test is limited to two epochs')
    if not isinstance(cfg['smoke_fraction'], (int,float)) or not 0 < cfg['smoke_fraction'] <= 1:
        raise ValueError('Invalid smoke_fraction')
    return cfg


def preflight(config, overrides=None, smoke=False, root=PROJECT_ROOT):
    root = Path(root).resolve()
    cfg = load_config(config, overrides)
    env = runtime_info(cfg['device'])
    snapshot = verify_dataset(root, root/cfg['data'])
    batch, strategy = resolve_batch(cfg['batch'], env['device'], env['auto_batch_supported'])
    fraction = cfg['smoke_fraction'] if smoke and env['fraction_supported'] else 1.0
    resolved = dict(cfg, device=env['device'], batch=batch, batch_strategy=strategy,
                    epochs=cfg['smoke_epochs'] if smoke else cfg['epochs'], fraction=fraction,
                    data=snapshot['config'], output=str(output_path(root,cfg['experiment_name'],smoke)),
                    experiment_type='smoke' if smoke else 'baseline', cli_overrides=overrides or {},
                    augmentation_policy=AUGMENTATION_POLICY, test_status='LOCKED',
                    validation_scope='Full VALID split', dataset_fingerprint=snapshot['curated_hash'])
    return resolved, env, snapshot


def print_plan(cfg, env, snapshot):
    print('='*40+'\nBASELINE TRAINING DRY RUN\n'+'='*40)
    for key in ('experiment_name','model','data','imgsz','epochs','batch','batch_strategy','workers','seed','device','fraction','output'):
        print(f'{key}: {cfg[key]}')
    print(env['device_message'])
    print('Train: '+snapshot['data']['train'])
    print('Validation: '+snapshot['data']['val']+' (full VALID split)')
    print('Test: LOCKED\nDataset fingerprint: PASS\nTraining readiness: YES\nREADY: YES\n'+'='*40)


@contextmanager
def frozen_cache(cache_dir, protected_roots=()):
    """Redirect label caches; cache=False separately disables image caching.

    Label scanning occurs in the trainer process before Windows DataLoader workers
    start. Keep the redirect active through final validation too.
    """
    from ultralytics.data import dataset
    from ultralytics.utils import callbacks, checks
    from PIL import Image
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    original = dataset.save_dataset_cache_file
    original_save = Image.Image.save
    protected = [Path(p).resolve() for p in protected_roots]
    def guarded_save(image, fp, *args, **kwargs):
        target = fp if isinstance(fp, (str,Path)) else getattr(fp,'name',None)
        if target and any(Path(target).resolve().is_relative_to(p) for p in protected):
            raise ValueError('Frozen dataset: framework image repair/write refused')
        return original_save(image,fp,*args,**kwargs)
    def redirected(prefix, path, content, version):
        from hashlib import sha256
        name = sha256(str(Path(path).resolve()).encode()).hexdigest()[:16]+'.cache'
        return original(prefix, cache_dir/name, content, version)
    # Local training only: omit analytics and external experiment integrations.
    with patch.object(dataset, 'save_dataset_cache_file', redirected), \
         patch.object(callbacks, 'add_integration_callbacks', lambda _: None), \
         patch.object(checks, 'check_pip_update_available', lambda: None), \
         patch.object(Image.Image, 'save', guarded_save):
        yield


def disable_oom_retry(trainer):
    # Installed 8.4.128 retries when this counter is <3. Fail instead of changing
    # the batch during an epoch; the caller reports how to retry explicitly.
    if hasattr(trainer, '_oom_retries'):
        trainer._oom_retries = 3


def training_details(trainer):
    """Record actual optimizer settings, beyond framework arguments such as 'auto'."""
    return dict(parameter_count=sum(p.numel() for p in trainer.model.parameters()),
                optimizer_class=type(trainer.optimizer).__name__,
                optimizer_initial_parameter_groups=[
                    {k:v for k,v in group.items() if k != 'params'}
                    for group in trainer.optimizer.param_groups])


def require_smoke(cfg, env, root):
    smoke_path = output_path(root,cfg['experiment_name'],True)
    file = smoke_path/'experiment.json'
    if not file.is_file(): raise ValueError('Run --smoke-test successfully before full baseline training')
    smoke = json.loads(file.read_text(encoding='utf-8'))
    for key, expected in [('training_status','PASS'),('dataset_fingerprint',cfg['dataset_fingerprint']),
                          ('model_checkpoint',cfg['model']),('ultralytics_version',env['ultralytics_version']),
                          ('imgsz',cfg['imgsz']),('dataset_immutability','PASS')]:
        if smoke.get(key) != expected: raise ValueError(f'Smoke prerequisite mismatch: {key}; rerun --smoke-test')
    discover_checkpoints(smoke_path, smoke=True)


def train(cfg, env, snapshot, overwrite=False, root=PROJECT_ROOT, model_factory=None, cache_context=None):
    root = Path(root).resolve()
    smoke = cfg['experiment_type'] == 'smoke'
    if not smoke: require_smoke(cfg,env,root)
    out = Path(cfg['output'])
    if out != output_path(root,cfg['experiment_name'],smoke):
        raise ValueError('Output differs from the owned experiment directory')
    prepare_output(out,overwrite)
    start = time.monotonic()
    metadata = dict(experiment_name=out.name, experiment_type=cfg['experiment_type'],
        model_checkpoint=cfg['model'], ultralytics_version=env['ultralytics_version'],
        torch_version=env['torch_version'], python_version=env['python_version'], device=cfg['device'],
        gpu_name=env['gpu_name'], dataset_config=cfg['data'], dataset_version='curated_v1',
        dataset_fingerprint=snapshot['curated_hash'], class_mapping=snapshot['data']['names'],
        imgsz=cfg['imgsz'], epochs_requested=cfg['epochs'], epochs_completed=0, batch=cfg['batch'],
        workers=cfg['workers'], seed=cfg['seed'], deterministic=cfg['deterministic'], pretrained=cfg['pretrained'],
        start_time=datetime.now(timezone.utc).isoformat(), end_time=None, duration_seconds=None,
        best_model_path=None,last_model_path=None,training_status='RUNNING',test_status='LOCKED',
        augmentation_policy=AUGMENTATION_POLICY, validation_scope='Full VALID split', train_fraction=cfg['fraction'])
    write_json(out/'environment.json',env)
    write_json(out/'dataset_fingerprint.json',dict(curated=snapshot['curated'],raw=snapshot['raw'],
               curated_hash=snapshot['curated_hash'],raw_hash=snapshot['raw_hash']))
    write_yaml(out/'resolved_config.yaml',cfg)
    write_json(out/'experiment.json',metadata)
    # Framework-facing YAML deliberately omits TEST and any download directives.
    runtime_data = {k:snapshot['data'][k] for k in ('path','train','val','nc','names')}
    write_yaml(out/'training_data.yaml',runtime_data)
    arguments = {k:cfg[k] for k in ('imgsz','epochs','patience','seed','deterministic','device','workers','cache','pretrained','batch','fraction')}
    arguments.update(data=str(out/'training_data.yaml'), project=str(out.parent),name=out.name,
                     exist_ok=True, val=True, split='val', save=True, task='detect')
    try:
        # Recheck immediately before any model/data loading, including programmatic callers.
        check_unchanged(snapshot,root)
        if model_factory is None:
            from ultralytics import YOLO
            model_factory = YOLO
        weights = root/'results/model_cache'/cfg['model']
        weights.parent.mkdir(parents=True,exist_ok=True)
        print(f'Checkpoint: {weights} (official download if absent)')
        print(env['device_message']); print(cfg['batch_strategy'])
        print(f'TRAIN fraction={cfg["fraction"]}; full VALID; TEST LOCKED; workers requested={cfg["workers"]}')
        model = model_factory(str(weights))
        metadata['pretrained_checkpoint_parameter_count'] = sum(p.numel() for p in model.model.parameters())
        def record_arguments(trainer):
            resolved_args = json.loads(json.dumps(vars(trainer.args),default=str))
            if resolved_args.get('split') != 'val': raise ValueError('TEST is locked; validation split must be val')
            write_yaml(out/'framework_resolved_args.yaml',resolved_args)
            metadata['resolved_training_arguments'] = resolved_args
            metadata['batch'] = trainer.batch_size
            metadata['workers_resolved'] = trainer.args.workers
            metadata.update(training_details(trainer))
            print(f'Resolved batch: {trainer.batch_size}; resolved workers: {trainer.args.workers}')
            write_json(out/'experiment.json',metadata)
        model.add_callback('on_pretrain_routine_end', record_arguments)
        model.add_callback('on_train_epoch_start', disable_oom_retry)
        context = cache_context or (lambda path: frozen_cache(path,
            [root/'data/curated/v1', root/'train', root/'valid', root/'test']))
        with context(out/'dataset_cache'):
            metrics = model.train(**arguments)
        metadata['best_model_path'],metadata['last_model_path'] = discover_checkpoints(out,smoke)
        rows = result_rows(out)
        metadata['epochs_completed'] = len(rows)
        if not rows: raise ValueError('Training produced no epoch results')
        values = getattr(metrics,'results_dict',None)
        metadata['validation_metrics'] = values if values is not None else {k:v for k,v in rows[-1].items() if k.startswith('metrics/')}
        if not metadata['validation_metrics']: raise ValueError('No validation metrics were produced')
        metadata['training_status'] = 'PASS'
    except BaseException as error:
        metadata['training_status'] = 'FAIL'
        metadata['error'] = f'{type(error).__name__}: {error}'
        if 'out of memory' in str(error).lower():
            print('CUDA OOM: reduce --batch explicitly. No automatic retry or config changes.')
        print('If Windows DataLoader multiprocessing failed, retry explicitly with --workers 0.')
        raise
    finally:
        try:
            metadata['dataset_immutability'] = check_unchanged(snapshot,root)
        except BaseException as error:
            metadata['dataset_immutability'] = 'FAIL'; metadata['training_status'] = 'FAIL'
            metadata['immutability_error'] = str(error)
            raise
        finally:
            if (out/'results.csv').is_file():
                metadata['epochs_completed'] = len(result_rows(out))
            metadata['end_time'] = datetime.now(timezone.utc).isoformat()
            metadata['duration_seconds'] = round(time.monotonic()-start,3)
            write_json(out/'experiment.json',metadata)
    print(f'{"SMOKE TEST" if smoke else "FULL BASELINE TRAINING"}: PASS\nOutput: {out}')
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description='Frozen curated-v1 YOLO baseline. TEST remains locked.')
    parser.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/baseline.yaml')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--smoke-test',action='store_true')
    parser.add_argument('--device')
    parser.add_argument('--batch')
    parser.add_argument('--workers',type=int)
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args(argv)
    overrides={k:getattr(args,k) for k in ('device','batch','workers','epochs') if getattr(args,k) is not None}
    if args.smoke_test and args.epochs is not None:
        overrides['smoke_epochs']=args.epochs
    try:
        cfg,env,snapshot=preflight(args.config,overrides,args.smoke_test)
        print_plan(cfg,env,snapshot)
        if not args.dry_run:
            train(cfg,env,snapshot,args.overwrite)
        return 0
    except (ImportError,OSError,ValueError,RuntimeError,KeyError) as error:
        print(f'BASELINE PIPELINE: FAIL\n{error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
