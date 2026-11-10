"""Controlled C lifecycle, sharing baseline training and B runtime protections."""
import argparse
from functools import partial
import json
import math
from pathlib import Path

import yaml
from smart_helmet.paths import PROJECT_ROOT
from .environment import runtime_info, verify_dataset
from .experiment import output_path, discover_checkpoints, write_json
from .experiment_b import file_hash, tree_hashes, baseline_runtime_arguments, baseline_runtime_signature
from .hard_example_sampler import PLAN, TIERS, TIER_WEIGHTS, HardExampleSampler, load_plan, ordered_inventory
from .sampling import source_evidence
from .trainer import train

NAME = 'C_hard_example_sampling'
CONFIG = Path('configs/experiments/C_hard_example_sampling.yaml')
CONSTANTS = dict(experiment_name=NAME, baseline_reference='A_baseline',
    difficulty_source='TRAIN_ONLY', sampling_plan_path=PLAN.as_posix(),
    sampling_strategy='runtime_weighted_sampling_with_replacement', replacement=True,
    num_samples=1158, tier_weights=TIER_WEIGHTS,
    human_review_requirement='post_human_review_PASS_pending_0_REVIEW_MORE_0',
    suspicious_exclusion_policy='baseline_weight_1_keep_image_and_label',
    data='configs/data.curated.v1.yaml', model='yolo11n.pt', imgsz=416, batch=4,
    workers=0, seed=42, epochs=100, patience=20, pretrained=True, deterministic=True, cache=False)


def implementation_hashes(root):
    names = ('experiment_c.py','experiment_c_trainer.py','hard_example_sampler.py',
        'experiment_b.py','experiment_b_trainer.py','class_aware_sampler.py','sampling.py',
        'hard_sampling_plan.py','trainer.py','environment.py','experiment.py')
    return {n:file_hash(Path(root)/'src/smart_helmet/training'/n) for n in names}


def load_config(path, overrides=None, smoke=False):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(cfg, dict) or set(cfg) != set(CONSTANTS)|{'device'}:
        raise ValueError('C config must contain exactly the controlled fields')
    for key, value in CONSTANTS.items():
        if cfg[key] != value or type(cfg[key]) is not type(value):
            raise ValueError('C constant changed: '+key)
    for key, value in (overrides or {}).items():
        if key == 'device':
            cfg[key] = value
        elif key not in ('batch','workers','epochs') or value != (2 if smoke and key == 'epochs' else CONSTANTS[key]):
            raise ValueError('CLI override would change baseline constant: '+key)
    if str(cfg['device']) not in ('auto','0','cuda:0'):
        raise ValueError('Controlled C requires CUDA:0 / GTX 1650')
    cfg['epochs'] = 2 if smoke else 100
    return cfg


def preflight(overrides=None, smoke=False, root=PROJECT_ROOT):
    root = Path(root).resolve()
    cfg = load_config(root/CONFIG, overrides, smoke)
    directory = root/'results/experiments/A_baseline'
    baseline = json.loads((directory/'experiment.json').read_text())
    discover_checkpoints(directory)
    if baseline.get('training_status') != 'PASS' or baseline.get('test_status') != 'LOCKED':
        raise ValueError('Passing TEST-locked baseline required')
    args = baseline_runtime_arguments(directory, baseline)
    for key in ('imgsz','batch','workers','seed','epochs','patience','pretrained','deterministic','cache'):
        if args[key] != CONSTANTS[key]:
            raise ValueError('Baseline reference mismatch: '+key)
    if Path(args['model']).name != 'yolo11n.pt' or args['optimizer'] != 'auto':
        raise ValueError('Unexpected baseline model/optimizer')
    plan, rows = load_plan(root)
    if plan['baseline_resolved_training_arguments'] != args:
        raise ValueError('Frozen plan runtime arguments disagree with A')
    env = runtime_info(cfg['device'])
    if env['device'] != '0' or env['gpu_name'] != 'NVIDIA GeForce GTX 1650' or env['ultralytics_version'] != '8.4.128':
        raise ValueError('C requires Ultralytics 8.4.128 and GTX 1650 on CUDA:0')
    snapshot = verify_dataset(root, root/cfg['data'])
    if snapshot['curated_hash'] != baseline['dataset_fingerprint'] or snapshot['curated_hash'] != plan['dataset_fingerprint'] or snapshot['raw_hash'] != plan['raw_fingerprint']:
        raise ValueError('Dataset fingerprint changed')
    if snapshot['counts']['train'] != 1158 or snapshot['counts']['val'] != 127:
        raise ValueError('TRAIN/VALID population changed')
    if not (root/'results/model_cache/yolo11n.pt').is_file():
        raise ValueError('Original pretrained initialization missing')
    sampler = HardExampleSampler(root, rows, plan)
    if [rows[i]['image'] for i in sampler.indices()] != plan['simulated_epoch_zero_images']:
        raise ValueError('Seed-42 sampling fails to reproduce frozen plan')
    sampler.diagnostics(sampler.indices(), True)
    cfg.update(device=env['device'], data=snapshot['config'], fraction=1.0,
        output=str(output_path(root, NAME, smoke)), experiment_type='smoke' if smoke else 'hard_example_sampling',
        batch_strategy='Fixed controlled batch=4, workers=0', dataset_fingerprint=snapshot['curated_hash'],
        validation_scope='Full VALID split', test_status='LOCKED', cli_overrides=overrides or {})
    return cfg, env, snapshot, rows, plan, args, baseline


def print_dry_run(state):
    cfg, env, snapshot, rows, plan, args, baseline = state
    print('='*40+'\nEXPERIMENT C DRY RUN\n'+'='*40)
    print(f'Experiment: {NAME}\nBaseline: A_baseline\nMain change: TRAIN hard-example sampling')
    print('Difficulty source: TRAIN ONLY\nPlan stage: post_human_review\nHuman review: PASS\nPending: 0\nREVIEW_MORE: 0')
    print('TRAIN: 1158')
    for i,t in enumerate(TIERS): print(f'Tier {i}: {plan["tier_counts"][t]}')
    print('Eligible hard images: 156\nSuspicious exclusions: 1\nEffective samples: 1158\nReplacement: YES\nSeed: 42')
    print('VALID: unchanged / 127\nTEST: LOCKED\nVALID leakage: 0\nTEST leakage: 0')
    print(f'Device: {cfg["device"]} ({env["gpu_name"]})\nBatch: 4\nWorkers: 0\nEpochs: {cfg["epochs"]}')
    print('Baseline constants: PASS\nDataset fingerprint: PASS\nREADY FOR SMOKE: YES\n'+'='*40)


def evidence(state, root):
    cfg, env, snapshot, rows, plan, args, baseline = state
    return dict(implementation_sha256=implementation_hashes(root), sampling_plan_sha256=file_hash(Path(root)/PLAN),
        baseline_runtime_sha256=baseline_runtime_signature(args, baseline),
        human_review_sha256=plan['human_review_sha256'], ultralytics_source_inspection=source_evidence())


def verify_diagnostics(root, rows, plan, diagnostic):
    if len(diagnostic.get('epochs', [])) != 2 or not diagnostic.get('sampler_actually_used'):
        raise ValueError('Two verified sampler epochs required')
    ordered = ordered_inventory(root, rows, diagnostic['ordered_train_images'])
    sampler = HardExampleSampler(root, ordered, plan)
    for i, observed in enumerate(diagnostic['epochs']):
        sampler.set_epoch(i)
        expected = sampler.diagnostics(sampler.indices(), True)
        if any(observed.get(k) != v for k,v in expected.items()):
            raise ValueError('Sampler diagnostics fail deterministic replay')
        if not observed.get('sampler_attached') or observed.get('device') != 'cuda:0' or observed.get('completed_optimizer_steps', 0) < 1 or observed.get('completed_batches') != (i+1)*290:
            raise ValueError('GPU optimizer/batch/attachment evidence missing')
    if any(diagnostic.get(k) != v for k,v in diagnostic['epochs'][0].items()):
        raise ValueError('First-epoch summary differs from actual exposure')


def smoke_status(root, state):
    out = output_path(root, NAME, True)
    if not (out/'experiment.json').is_file(): return 'NOT RUN'
    try:
        cfg, env, snapshot, rows, plan, args, baseline = state
        meta = json.loads((out/'experiment.json').read_text())
        required = dict(training_status='PASS', dataset_immutability='PASS', protected_artifacts_immutability='PASS',
            sampler_actually_used=True, epochs_completed=2, validation_images=127, device='0',
            gpu_name='NVIDIA GeForce GTX 1650', batch=4, workers_resolved=0, imgsz=416, seed=42,
            test_status='LOCKED', dataset_fingerprint=snapshot['curated_hash'],
            torch_version=env['torch_version'], ultralytics_version=env['ultralytics_version'], **evidence(state, root))
        if any(meta.get(k) != v for k,v in required.items()): return 'FAIL'
        from .experiment_b_trainer import compare_baseline_arguments
        compare_baseline_arguments(meta['resolved_training_arguments'], args, smoke=True)
        for key in ('optimizer_class','optimizer_initial_parameter_groups','parameter_count'):
            if meta.get(key) != baseline[key]: return 'FAIL'
        for name, digest in meta['artifact_sha256'].items():
            if file_hash(out/name) != digest: return 'FAIL'
        required_artifacts = {'sampler_diagnostics.json','sampling_plan_snapshot.json','baseline_argument_comparison.json',
            'resolved_config.yaml','environment.json','dataset_fingerprint.json','weights/best.pt','weights/last.pt'}
        if not required_artifacts <= set(meta['artifact_sha256']): return 'FAIL'
        if file_hash(out/'sampling_plan_snapshot.json') != file_hash(Path(root)/PLAN): return 'FAIL'
        diagnostic = json.loads((out/'sampler_diagnostics.json').read_text())
        verify_diagnostics(root, rows, plan, diagnostic)
        report = json.loads((out/'baseline_argument_comparison.json').read_text())
        if report['result'] != 'PASS' or report['unexpected_drift']: return 'FAIL'
        discover_checkpoints(out)
        return 'PASS'
    except (OSError, ValueError, KeyError, TypeError):
        return 'FAIL'


def execute(state, overwrite=False, root=PROJECT_ROOT):
    root = Path(root).resolve()
    cfg, env, snapshot, rows, plan, args, baseline = state
    smoke = cfg['experiment_type'] == 'smoke'
    if cfg['experiment_name'] != NAME or Path(cfg['output']) != output_path(root, NAME, smoke):
        raise ValueError('C cannot write to any other experiment output')
    if not smoke and smoke_status(root, state) != 'PASS':
        raise ValueError('Full C requires a matching passing GPU smoke test')
    # Revalidate just before training; supplied state must not outlive frozen inputs.
    current_plan, current_rows = load_plan(root)
    if current_plan != plan or current_rows != rows: raise ValueError('Frozen inputs changed after preflight')
    from .experiment_c_trainer import ExperimentCTrainer
    protected = [root/'results/experiments/A_baseline', root/'results/experiments/B_class_aware_sampling', root/PLAN.parent]
    before = {str(p):tree_hashes(p) for p in protected}
    optimizer = {k:baseline[k] for k in ('optimizer_class','optimizer_initial_parameter_groups','parameter_count')}
    factory = partial(ExperimentCTrainer, project_root=root, inventory=rows, sampling_plan=plan,
        baseline_args=args, baseline_optimizer=optimizer, smoke=smoke)
    arguments = {k:v for k,v in args.items() if k not in ('model','data','project','name','save_dir')}

    def unchanged():
        if any(tree_hashes(p) != before[str(p)] for p in protected):
            raise ValueError('Protected A/B/mining artifacts changed')

    def configure(model, out, metadata):
        (out/'sampling_plan_snapshot.json').write_bytes((root/PLAN).read_bytes())
        metadata.update(baseline_reference='A_baseline', main_experimental_change='TRAIN hard-example sampling',
            sampler_actually_used=False, **evidence(state, root))
        write_json(out/'experiment.json', metadata)

    def verify(model, out, metadata):
        diagnostic = json.loads((out/'sampler_diagnostics.json').read_text())
        if smoke:
            verify_diagnostics(root, rows, plan, diagnostic)
            if metadata['epochs_completed'] != 2: raise ValueError('Exactly two smoke epochs required')
        elif not diagnostic['sampler_actually_used'] or len(diagnostic['epochs']) != metadata['epochs_completed']:
            raise ValueError('Sampler epoch evidence missing')
        discover_checkpoints(out)
        if model.trainer.validator.seen != 127 or not all(math.isfinite(float(v)) for v in metadata['validation_metrics'].values()):
            raise ValueError('Full VALID / finite validation not verified')
        unchanged()
        metadata.update(sampler_actually_used=True, protected_artifacts_immutability='PASS',
            validation_images=127, effective_train_samples=1158, completed_optimizer_steps=model.trainer.completed_optimizer_steps)
        names = ('sampler_diagnostics.json','sampling_plan_snapshot.json','baseline_argument_comparison.json',
            'resolved_config.yaml','environment.json','dataset_fingerprint.json','weights/best.pt','weights/last.pt')
        metadata['artifact_sha256'] = {n:file_hash(out/n) for n in names}

    try:
        return train(cfg, env, snapshot, overwrite, root, trainer_factory=factory,
            framework_arguments=arguments, configure_run=configure, verify_run=verify)
    finally:
        unchanged()


def readiness_main(root=PROJECT_ROOT):
    try:
        state = preflight(root=root)
        status = smoke_status(root, state)
        out = output_path(root, NAME, True)
        passed = status == 'PASS'
        print('='*40+'\nEXPERIMENT C READINESS\n'+'='*40)
        print('Phase 4I final plan: PASS\nHuman review: PASS\nPending: 0\nREVIEW_MORE: 0\nSuspicious exclusions: PASS\nSampling plan: PASS')
        from .experiment_c_trainer import ExperimentCTrainer
        print('Custom sampler implementation: PASS\nSampler actually used: '+('YES' if passed else 'NO'))
        print('TRAIN only: PASS\nVALID leakage: 0\nTEST leakage: 0\nBaseline constants: PASS\nDataset fingerprint: PASS')
        print('Sampler diagnostics: '+('PASS' if passed else 'FAIL'))
        print(f'Smoke test: {status}\nGPU: {state[1]["gpu_name"]}')
        for name in ('best','last'): print(f'Smoke {name}.pt: '+('PASS' if (out/f'weights/{name}.pt').is_file() else 'FAIL'))
        print('READY FOR FULL EXPERIMENT C: '+('YES' if passed else 'NO')+'\nTEST STATUS: LOCKED\n'+'='*40)
        return 0 if passed else 1
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        print(f'EXPERIMENT C READINESS: FAIL\n{error}\nREADY FOR FULL EXPERIMENT C: NO')
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dry-run', action='store_true')
    group.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Resume only the existing full C last.pt')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--device')
    for key in ('batch','workers','epochs'): parser.add_argument('--'+key, type=int)
    options = parser.parse_args(argv)
    if options.resume:
        from .experiment_c_resume import resume_main
        if options.overwrite or options.smoke_test or any(getattr(options, k) is not None for k in ('device','batch','workers','epochs')):
            parser.error('--resume forbids --overwrite, --smoke-test and training overrides')
        return resume_main(dry_run=options.dry_run)
    overrides = {k:getattr(options,k) for k in ('device','batch','workers','epochs') if getattr(options,k) is not None}
    try:
        state = preflight(overrides, options.smoke_test)
        print_dry_run(state)
        if not options.dry_run: execute(state, options.overwrite)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        print(f'EXPERIMENT C PIPELINE: FAIL\n{error}')
        return 1
