"""Controlled Experiment B orchestration, reusing the baseline lifecycle."""
import argparse
from functools import partial
import hashlib
import json
import math
from pathlib import Path

import yaml

from smart_helmet.paths import PROJECT_ROOT
from smart_helmet.evaluation.human_review import EVALUATION, ReviewStore, read_cases, summarize, atomic_text
from .environment import runtime_info, verify_dataset
from .experiment import output_path, discover_checkpoints, write_json
from .sampling import records, build_plan, source_evidence
from .trainer import train

NAME = 'B_class_aware_sampling'
CONFIG = Path('configs/experiments/B_class_aware_sampling.yaml')
CONSTANTS = dict(experiment_name=NAME, baseline_reference='A_baseline',
    data='configs/data.curated.v1.yaml', model='yolo11n.pt', imgsz=416,
    batch=4, workers=0, seed=42, epochs=100, patience=20, pretrained=True,
    deterministic=True, cache=False, sampling_strategy='runtime_weighted_sampling_with_replacement',
    without_helmet_multiplier=1.573, replacement=True, num_samples=1158)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def tree_hashes(path):
    return {p.relative_to(path).as_posix():file_hash(p) for p in sorted(Path(path).rglob('*')) if p.is_file()}


def baseline_runtime_arguments(baseline, metadata):
    """Use post-optimizer arguments, never the earlier framework args.yaml."""
    saved = yaml.safe_load((Path(baseline)/'args.yaml').read_text(encoding='utf-8'))
    runtime = yaml.safe_load((Path(baseline)/'framework_resolved_args.yaml').read_text(encoding='utf-8'))
    if not isinstance(runtime, dict) or set(runtime) != set(saved):
        raise ValueError('Incomplete baseline runtime arguments')
    if runtime != metadata.get('resolved_training_arguments'):
        raise ValueError('Baseline runtime argument sources disagree')
    return dict(runtime)


def baseline_runtime_signature(args, metadata):
    evidence = dict(arguments=args, **{k:metadata[k] for k in
        ('optimizer_class','optimizer_initial_parameter_groups','parameter_count')})
    return hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()


def implementation_hashes(root):
    return {name:file_hash(Path(root)/'src/smart_helmet/training'/name) for name in
            ('experiment_b.py','experiment_b_trainer.py','class_aware_sampler.py','sampling.py','trainer.py','environment.py')}


def load_config(path, overrides=None, smoke=False):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(cfg, dict) or set(cfg) != set(CONSTANTS)|{'device'}:
        raise ValueError('Experiment B config must contain exactly the documented controlled fields')
    for key, value in CONSTANTS.items():
        if cfg[key] != value or type(cfg[key]) is not type(value):
            raise ValueError('Experiment B constant changed: '+key)
    overrides = overrides or {}
    for key, value in overrides.items():
        if key == 'device':
            cfg[key] = value
        elif key not in ('batch','workers','epochs') or int(value) != (2 if smoke and key == 'epochs' else CONSTANTS[key]):
            raise ValueError('CLI override would change baseline constant: '+key)
    if cfg['device'] not in ('auto','0','cuda:0','cpu'):
        raise ValueError('Use auto, CUDA:0, or CPU; multi-device training is not controlled')
    cfg['epochs'] = 2 if smoke else 100
    return cfg


def human_assessment(root):
    out = Path(root)/EVALUATION/'failure_review'
    store = ReviewStore(out/'human_review.csv', read_cases(root))
    summary = summarize(store)
    automatic = json.loads((out/'failure_review_summary.json').read_text(encoding='utf-8'))
    train_stats = automatic['train_distribution']
    cats = summary['categories']
    passed = (summary['ready_for_experiment_b_design'] and
        cats['ANNOTATION_SUSPICIOUS']['count'] == 0 and
        automatic['categories']['LOW_CONFIDENCE']['count'] > 0 and
        automatic['categories']['CLASS_CONFUSION']['count'] > 0 and
        train_stats['objects']['Without Helmet'] < train_stats['objects']['With Helmet'])
    rationale = (
        'Design assessment: PASS for a controlled exposure experiment, not a proven causal remedy. '
        'The completed human review has no suspicious annotations; low-confidence and wrong-class matches '
        'provide a model-learning hypothesis in addition to TRAIN underrepresentation. '
        'Human OTHER/UNKNOWN entries remain unresolved and are not relabeled or inferred. '
        'Repeated examples may overfit and mixed images increase both class exposures. '
        'Sampling cannot recover obscured visual information. Alternatives are targeted augmentation '
        'after cause confirmation, new independent TRAIN data, or a separate VALID operating-point study. '
        'Resolution and augmentation remain unchanged. This explicit Phase 4E design assessment '
        'supersedes the Phase 4D automatic request for design review; it does not change human entries.'
    ) if passed else 'FAIL: completed review without annotation concerns and supporting model evidence required.'
    return dict(status='PASS' if passed else 'FAIL', human_review='PASS' if summary['ready_for_experiment_b_design'] else 'FAIL',
        summary=summary, rationale=rationale,
        unresolved_entries=[r for r in store.rows.values() if any(c in r['human_categories'].split('|') for c in ('OTHER','UNKNOWN'))],
        human_review_sha256=file_hash(out/'human_review.csv'), automatic_summary_sha256=file_hash(out/'failure_review_summary.json'))


def preflight(overrides=None, smoke=False, root=PROJECT_ROOT):
    root = Path(root).resolve()
    cfg = load_config(root/CONFIG, overrides, smoke)
    baseline = root/'results/experiments/A_baseline'
    metadata = json.loads((baseline/'experiment.json').read_text(encoding='utf-8'))
    if metadata.get('training_status') != 'PASS' or metadata.get('test_status') != 'LOCKED':
        raise ValueError('Passing TEST-locked baseline required')
    discover_checkpoints(baseline)
    args = baseline_runtime_arguments(baseline, metadata)
    for key in ('imgsz','batch','workers','seed','epochs','patience','pretrained','deterministic','cache'):
        if args[key] != CONSTANTS[key]:
            raise ValueError('Baseline reference mismatch: '+key)
    if Path(args['model']).name != 'yolo11n.pt' or args['optimizer'] != 'auto':
        raise ValueError('Unexpected baseline model/optimizer')
    assessment = human_assessment(root)
    if assessment['status'] != 'PASS':
        raise ValueError(assessment['rationale'])
    env = runtime_info(cfg['device'])
    if env['device'] not in ('0','cpu') or env['ultralytics_version'] != '8.4.128':
        raise ValueError('Unsupported device or Ultralytics version')
    snapshot = verify_dataset(root, root/cfg['data'])
    if snapshot['curated_hash'] != metadata['dataset_fingerprint']:
        raise ValueError('Dataset differs from baseline fingerprint')
    rows = records(root)
    plan = build_plan(root, rows)
    if plan['original_train_images'] != 1158 or plan['images_with_without_helmet'] != 450 or plan['class_counts_before_sampling'] != {'With Helmet':2009,'Without Helmet':1038}:
        raise ValueError('Unexpected TRAIN population')
    saved_plan = json.loads((root/'results/experiments'/NAME/'sampling_plan.json').read_text(encoding='utf-8'))
    for key in ('strategy','seed','multiplier','effective_train_samples','train_inventory','sampled_paths_epoch_zero'):
        if saved_plan.get(key) != plan[key]:
            raise ValueError('Stored sampling plan differs from current TRAIN plan: '+key)
    plan.update(ultralytics_source_inspection=source_evidence(), dataset_fingerprint=snapshot['curated_hash'],
        integration_status='LOCAL CUSTOM DETECTION TRAINER', baseline_constants_preserved=True)
    if not (root/'results/model_cache/yolo11n.pt').is_file():
        raise ValueError('Baseline pretrained initialization checkpoint missing')
    cfg.update(device=env['device'], data=snapshot['config'], fraction=1.0,
        output=str(output_path(root, NAME, smoke)), experiment_type='smoke' if smoke else 'class_aware_sampling',
        batch_strategy='Fixed controlled batch=4, workers=0', dataset_fingerprint=snapshot['curated_hash'],
        validation_scope='Full VALID split', test_status='LOCKED', cli_overrides=overrides or {})
    return cfg, env, snapshot, rows, plan, args, metadata, assessment


def print_dry_run(state):
    cfg, env, snapshot, rows, plan, args, metadata, assessment = state
    print('='*40+'\nEXPERIMENT B DRY RUN\n'+'='*40)
    print(f'Experiment: {NAME}\nBaseline: A_baseline\nMain change: class-aware TRAIN sampling\nMultiplier: 1.573')
    print(f'TRAIN: {len(rows)}\nWithout Helmet-containing images: {plan["images_with_without_helmet"]}')
    print(f'VALID: unchanged ({snapshot["counts"]["val"]})\nTEST: LOCKED\nLeakage: 0 / 0')
    print(f'Device: {cfg["device"]} ({env["gpu_name"]})\nBatch: 4\nWorkers: 0\nEpochs: {cfg["epochs"]}')
    print('Human failure review: PASS\nSampling justification: PASS\nDataset fingerprint: PASS\nREADY: YES\n'+'='*40)


def smoke_status(root, state):
    root = Path(root)
    out = output_path(root, NAME, True)
    if not (out/'experiment.json').is_file():
        return 'NOT RUN'
    try:
        meta = json.loads((out/'experiment.json').read_text(encoding='utf-8'))
        diagnostic = json.loads((out/'sampler_diagnostics.json').read_text(encoding='utf-8'))
        cfg, env, snapshot, rows, plan, args, baseline, assessment = state
        required = dict(training_status='PASS', dataset_immutability='PASS', baseline_immutability='PASS',
            sampler_actually_used=True, optimizer_matches_baseline=True, epochs_completed=2,
            batch=4, workers_resolved=0, imgsz=416, seed=42, test_status='LOCKED',
            torch_version=env['torch_version'], ultralytics_version=env['ultralytics_version'],
            dataset_fingerprint=snapshot['curated_hash'], device='0',
            implementation_sha256=implementation_hashes(root),
            baseline_args_sha256=file_hash(root/'results/experiments/A_baseline/args.yaml'),
            baseline_runtime_sha256=baseline_runtime_signature(args, baseline),
            human_review_sha256=assessment['human_review_sha256'])
        if any(meta.get(k) != v for k,v in required.items()):
            return 'FAIL'
        from .experiment_b_trainer import compare_baseline_arguments
        compare_baseline_arguments(meta['resolved_training_arguments'], args, smoke=True)
        for key in ('optimizer_class','optimizer_initial_parameter_groups','parameter_count'):
            if meta.get(key) != baseline[key]:
                return 'FAIL'
        report = json.loads((out/'baseline_argument_comparison.json').read_text(encoding='utf-8'))
        if report.get('result') != 'PASS' or meta.get('baseline_argument_report_sha256') != file_hash(out/'baseline_argument_comparison.json'):
            return 'FAIL'
        if diagnostic.get('sampler_actually_used') is not True or len(diagnostic.get('epochs', [])) != 2:
            return 'FAIL'
        for key, expected in dict(sampler_class='ClassAwareSampler', replacement=True,
                                  num_samples=1158, seed=42, multiplier=1.573,
                                  unique_train_images=1158, images_with_without_helmet=450,
                                  valid_leakage=0, test_leakage=0).items():
            if diagnostic.get(key) != expected:
                return 'FAIL'
        if meta.get('sampler_diagnostics_sha256') != file_hash(out/'sampler_diagnostics.json'):
            return 'FAIL'
        if meta.get('ultralytics_source_inspection') != source_evidence():
            return 'FAIL'
        for epoch in diagnostic['epochs']:
            if epoch['actual_sample_count'] != 1158 or not epoch['sampler_actually_used'] or not epoch['completed_optimizer_steps']:
                return 'FAIL'
        best, last = discover_checkpoints(out)
        if meta.get('checkpoint_sha256') != {'best':file_hash(best),'last':file_hash(last)}:
            return 'FAIL'
        return 'PASS'
    except (OSError, ValueError, KeyError, TypeError):
        return 'FAIL'


def execute(state, overwrite=False, root=PROJECT_ROOT):
    root = Path(root).resolve()
    cfg, env, snapshot, rows, plan, baseline_args, baseline, assessment = state
    smoke = cfg['experiment_type'] == 'smoke'
    if cfg['experiment_name'] != NAME or Path(cfg['output']) != output_path(root, NAME, smoke):
        raise ValueError('Experiment B cannot write to baseline or any other experiment output')
    if not smoke and smoke_status(root, state) != 'PASS':
        raise ValueError('Full Experiment B requires a matching passing GPU smoke test')
    from .experiment_b_trainer import ExperimentBTrainer
    baseline_dir = root/'results/experiments/A_baseline'
    before = tree_hashes(baseline_dir)
    optimizer = {k:baseline[k] for k in ('optimizer_class','optimizer_initial_parameter_groups','parameter_count')}
    factory = partial(ExperimentBTrainer, project_root=root, inventory=rows,
        baseline_args=baseline_args, baseline_optimizer=optimizer, smoke=smoke)
    arguments = {k:v for k,v in baseline_args.items() if k not in ('model','data','project','name','save_dir')}

    def configure(model, out, metadata):
        metadata.update(baseline_reference='A_baseline', main_experimental_change='class-aware sampling',
            sampling_strategy=cfg['sampling_strategy'], multiplier=1.573,
            sampler_actually_used=False, implementation_sha256=implementation_hashes(root),
            ultralytics_source_inspection=source_evidence(),
            baseline_args_sha256=file_hash(baseline_dir/'args.yaml'), human_review_sha256=assessment['human_review_sha256'])
        metadata['baseline_runtime_sha256'] = baseline_runtime_signature(baseline_args, baseline)
        write_json(out/'experiment.json', metadata)
        write_json(out/'sampling_plan.json', dict(plan, plan_only=False, training_authorized=True))
        write_json(out/'sampling_justification.json', assessment)

    def verify(model, out, metadata):
        trainer = model.trainer
        diagnostic = json.loads((out/'sampler_diagnostics.json').read_text(encoding='utf-8'))
        if not diagnostic['sampler_actually_used'] or len(diagnostic['epochs']) != metadata['epochs_completed']:
            raise ValueError('Actual epoch sampler usage was not verified')
        if smoke and metadata['epochs_completed'] != 2:
            raise ValueError('Smoke must complete exactly two epochs')
        best, last = discover_checkpoints(out)
        if not all(math.isfinite(float(v)) for v in metadata['validation_metrics'].values()):
            raise ValueError('Nonfinite validation results')
        if trainer.validator.seen != snapshot['counts']['val']:
            raise ValueError('Full VALID coverage not verified')
        if tree_hashes(baseline_dir) != before:
            raise ValueError('A_baseline artifacts changed')
        metadata.update(sampler_actually_used=True, optimizer_matches_baseline=True,
            baseline_constants_preserved=True, baseline_immutability='PASS',
            validation_images=trainer.validator.seen, effective_train_samples=1158,
            completed_optimizer_steps=trainer.completed_optimizer_steps,
            sampler_diagnostics_sha256=file_hash(out/'sampler_diagnostics.json'),
            checkpoint_sha256={'best':file_hash(best),'last':file_hash(last)})
        metadata['baseline_argument_report_sha256'] = file_hash(out/'baseline_argument_comparison.json')

    try:
        return train(cfg, env, snapshot, overwrite, root,
            trainer_factory=factory, framework_arguments=arguments, configure_run=configure, verify_run=verify)
    finally:
        if tree_hashes(baseline_dir) != before:
            raise ValueError('A_baseline artifacts changed')


def readiness_main(root=PROJECT_ROOT):
    try:
        state = preflight(root=root)
        status = smoke_status(root, state)
        print('='*40+'\nEXPERIMENT B READINESS\n'+'='*40)
        print('Human failure review: PASS\nSampling justification: PASS\nSampling plan: PASS')
        from .experiment_b_trainer import ExperimentBTrainer
        print('Custom sampler implementation: PASS\nTRAIN only: PASS\nVALID leakage: 0\nTEST leakage: 0')
        print(f'Baseline constants: PASS\nDataset fingerprint: PASS\nSmoke test: {status}')
        print('READY FOR FULL EXPERIMENT B: '+('YES' if status == 'PASS' else 'NO')+'\n'+'='*40)
        return 0 if status in ('PASS','NOT RUN') else 1
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        print(f'EXPERIMENT B READINESS: FAIL\n{error}\nREADY FOR FULL EXPERIMENT B: NO')
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dry-run', action='store_true')
    group.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--device')
    for key in ('batch','workers','epochs'):
        parser.add_argument('--'+key, type=int)
    options = parser.parse_args(argv)
    overrides = {k:getattr(options,k) for k in ('device','batch','workers','epochs') if getattr(options,k) is not None}
    try:
        state = preflight(overrides, options.smoke_test)
        print_dry_run(state)
        if not options.dry_run:
            execute(state, options.overwrite)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        print(f'EXPERIMENT B PIPELINE: FAIL\n{error}')
        return 1
