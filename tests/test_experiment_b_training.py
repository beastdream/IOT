"""CPU loader integration and safeguards; no models, training or downloads."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from smart_helmet.paths import PROJECT_ROOT
from smart_helmet.training import experiment_b as pipeline
from smart_helmet.training.class_aware_sampler import ClassAwareSampler, ordered_inventory, index_hash
from smart_helmet.training.experiment import output_path, prepare_output
from smart_helmet.training.sampling import records


@pytest.fixture(autouse=True)
def isolated_framework_import():
    # Ultralytics replaces PIL.Image.open at import. Do not let collection-time
    # imports change unrelated dataset-audit tests or trigger decoder installs.
    from PIL import Image
    original_open = Image.open
    global ExperimentBTrainer, weighted_loader, compare_baseline_arguments
    try:
        from smart_helmet.training.experiment_b_trainer import (
            ExperimentBTrainer, weighted_loader, compare_baseline_arguments,
        )
        yield
    finally:
        Image.open = original_open


@pytest.fixture
def sample(tmp_path):
    base = tmp_path/'data/curated/v1/train'
    (base/'images').mkdir(parents=True)
    (base/'labels').mkdir()
    for i in range(18):
        (base/'images'/f'{i:02}.jpg').write_bytes(b'not decoded in loader tests')
        labels = '0 .5 .5 .2 .2\n' + ('1 .5 .5 .1 .1\n' if i%3 == 0 else '')
        (base/'labels'/f'{i:02}.txt').write_text(labels)
    return tmp_path, records(tmp_path)


class IndexDataset:
    def __init__(self, paths):
        self.im_files = paths
    def __len__(self):
        return len(self.im_files)
    def __getitem__(self, index):
        return index


def test_actual_infinite_loader_uses_sampler_and_reset(sample):
    root, rows = sample
    sampler = ClassAwareSampler(root, rows)
    dataset = IndexDataset([str(root/r['image']) for r in rows])
    loader = weighted_loader(dataset, sampler)
    assert loader.sampler is sampler and loader.num_workers == 0
    assert sampler.replacement and len(sampler) == len(rows)
    for epoch in range(3):
        sampler.set_epoch(epoch)
        if epoch == 2:
            loader.reset()  # same hook used by framework close_mosaic
        actual = [int(i) for batch in loader for i in batch]
        assert actual == sampler.indices(epoch) == sampler.yielded_indices
    loader.close()


def test_seed_weights_replacement_and_diagnostics(sample):
    root, rows = sample
    first = ClassAwareSampler(root, rows)
    second = ClassAwareSampler(root, rows)
    assert first.weights == [1.573 if i%3 == 0 else 1.0 for i in range(18)]
    assert first.indices() == second.indices()
    assert first.indices() != ClassAwareSampler(root, rows, seed=43).indices()
    assert len(set(first.indices())) < len(rows)  # duplicates are draws, not files
    diagnostic = first.diagnostics(first.indices(), True)
    assert diagnostic['sampled_index_hash'] == index_hash(first.indices())
    assert diagnostic['sampled_without_helmet_image_count'] + diagnostic['sampled_other_image_count'] == 18
    assert diagnostic['estimated_object_exposure']['With Helmet'] == 18
    assert diagnostic['estimated_object_exposure']['Without Helmet'] == diagnostic['sampled_without_helmet_image_count']


def test_framework_order_is_mapped_without_assumption(sample):
    root, rows = sample
    paths = [root/r['image'] for r in rows][::-1]
    assert ordered_inventory(root, rows, paths) == rows[::-1]
    with pytest.raises(ValueError):
        ordered_inventory(root, rows, paths[:-1])
    with pytest.raises(ValueError):
        ordered_inventory(root, rows, paths+[paths[0]])


@pytest.mark.parametrize('split', ['valid','test'])
def test_foreign_split_rejected_before_loading(sample, split):
    root, rows = sample
    bad = dict(rows[0], image=f'data/curated/v1/{split}/images/a.jpg')
    with pytest.raises(ValueError, match='leakage'):
        ClassAwareSampler(root, [bad])
    trainer = object.__new__(ExperimentBTrainer)
    trainer.project_root = root
    with pytest.raises(ValueError):
        trainer.get_dataloader(root/f'data/curated/v1/{split}/images', mode='train')
    with pytest.raises(ValueError, match='TEST locked'):
        trainer.get_dataloader(root/'data/curated/v1/test/images', mode='test')


def test_custom_trainer_loader_extension(sample, monkeypatch):
    root, rows = sample
    trainer = object.__new__(ExperimentBTrainer)
    trainer.project_root = root
    trainer.inventory = rows
    trainer.args = SimpleNamespace(workers=0, rect=False, compile=False, fraction=1)
    trainer.device = torch.device('cpu')
    dataset = IndexDataset([str(root/r['image']) for r in rows][::-1])
    monkeypatch.setattr(trainer, 'build_dataset', lambda *args: dataset)
    loader = trainer.get_dataloader(root/'data/curated/v1/train/images')
    assert loader.sampler is trainer.class_aware_sampler
    assert trainer.class_aware_sampler.rows == rows[::-1]
    assert [int(i) for batch in loader for i in batch] == trainer.class_aware_sampler.indices()
    loader.close()


def test_validation_delegates_unchanged(sample, monkeypatch):
    from ultralytics.models.yolo.detect import DetectionTrainer
    root, rows = sample
    trainer = object.__new__(ExperimentBTrainer)
    trainer.project_root = root
    calls = []
    monkeypatch.setattr(DetectionTrainer, 'get_dataloader', lambda self,*args: calls.append(args) or 'VALID')
    path = root/'data/curated/v1/valid/images'
    assert trainer.get_dataloader(path, 8, -1, 'val') == 'VALID'
    assert calls == [(path,8,-1,'val')]


def test_config_constants_and_cli_restrictions(tmp_path):
    path = PROJECT_ROOT/pipeline.CONFIG
    cfg = pipeline.load_config(path)
    assert cfg['batch'] == 4 and cfg['workers'] == 0 and cfg['epochs'] == 100
    assert pipeline.load_config(path, {'epochs':2}, True)['epochs'] == 2
    for override in ({'batch':8}, {'workers':2}, {'epochs':3}, {'device':'0,1'}):
        with pytest.raises(ValueError):
            pipeline.load_config(path, override)
    content = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    content['imgsz'] = 640
    altered = tmp_path/'altered.yaml'
    altered.write_text(yaml.safe_dump(content))
    with pytest.raises(ValueError):
        pipeline.load_config(altered)


def test_argument_and_optimizer_controls():
    base = dict(epochs=100, imgsz=416, mosaic=1.0, optimizer='auto', batch=4)
    compare_baseline_arguments(dict(base), base)
    compare_baseline_arguments(dict(base, epochs=2), base, smoke=True)
    for key,value in [('imgsz',640),('mosaic',0),('optimizer','SGD'),('batch',8)]:
        with pytest.raises(ValueError, match='drift'):
            compare_baseline_arguments(dict(base, **{key:value}), base)


def test_output_separation_and_archival(tmp_path):
    baseline = output_path(tmp_path, 'A_baseline')
    baseline.mkdir(parents=True)
    (baseline/'original.txt').write_text('preserve')
    full = output_path(tmp_path, pipeline.NAME)
    smoke = output_path(tmp_path, pipeline.NAME, True)
    assert len({baseline,full,smoke}) == 3
    prepare_output(smoke)
    (smoke/'experiment.json').write_text('{}')
    with pytest.raises(ValueError, match='overwrite'):
        prepare_output(smoke)
    prepare_output(smoke, True)
    assert list(smoke.parent.glob(pipeline.NAME+'_smoke_archive_*'))
    assert (baseline/'original.txt').read_text() == 'preserve'


def test_runtime_yaml_test_lock_before_framework(tmp_path):
    trainer = object.__new__(ExperimentBTrainer)
    path = tmp_path/'data.yaml'
    path.write_text('test: test/images\n')
    trainer.project_root = tmp_path
    trainer.args = SimpleNamespace(data=str(path), split='val')
    with pytest.raises(ValueError, match='TEST locked'):
        trainer.get_dataset()


def test_dry_run_never_trains(monkeypatch):
    monkeypatch.setattr(pipeline, 'preflight', lambda *args: 'state')
    monkeypatch.setattr(pipeline, 'print_dry_run', lambda *args: None)
    monkeypatch.setattr(pipeline, 'execute', lambda *args: pytest.fail('Dry run trained'))
    assert pipeline.main(['--dry-run']) == 0


def test_epoch_diagnostics_require_actual_order(sample, tmp_path):
    root, rows = sample
    trainer = object.__new__(ExperimentBTrainer)
    trainer.class_aware_sampler = ClassAwareSampler(root, rows)
    trainer.train_loader = SimpleNamespace(sampler=trainer.class_aware_sampler)
    trainer.observed_indices = list(trainer.class_aware_sampler)
    trainer.completed_optimizer_steps = 1
    trainer.completed_batches = 5
    trainer.device = torch.device('cpu')
    trainer.exposure_epochs = []
    trainer.save_dir = tmp_path
    trainer.finish_sampling_epoch(trainer)
    saved = json.loads((tmp_path/'sampler_diagnostics.json').read_text())
    assert saved['sampler_actually_used'] and saved['sampler_class'] == 'ClassAwareSampler'
    assert len(saved['epochs']) == 1
    trainer.observed_indices = trainer.observed_indices[::-1]
    with pytest.raises(ValueError, match='order'):
        trainer.finish_sampling_epoch(trainer)


def test_failed_or_missing_smoke_not_ready(tmp_path):
    assert pipeline.smoke_status(tmp_path, None) == 'NOT RUN'
    out = output_path(tmp_path, pipeline.NAME, True)
    out.mkdir(parents=True)
    (out/'experiment.json').write_text('{"training_status":"FAIL"}')
    assert pipeline.smoke_status(tmp_path, None) == 'FAIL'


def test_experiment_b_metadata_hooks_and_output_guard(tmp_path, monkeypatch):
    baseline_dir = tmp_path/'results/experiments/A_baseline'
    baseline_dir.mkdir(parents=True)
    (baseline_dir/'args.yaml').write_text('optimizer: auto\n')
    monkeypatch.setattr(pipeline, 'implementation_hashes', lambda root: {'code':'hash'})
    monkeypatch.setattr(pipeline, 'source_evidence', lambda: {'version':'8.4.128'})
    cfg = dict(experiment_name=pipeline.NAME, experiment_type='smoke',
               output=str(output_path(tmp_path, pipeline.NAME, True)), sampling_strategy='runtime_weighted_sampling_with_replacement')
    baseline = dict(optimizer_class='AdamW', optimizer_initial_parameter_groups=[], parameter_count=2590230)
    assessment = dict(status='PASS', human_review_sha256='reviewhash')
    state = (cfg, {}, {}, [], {'strategy':'runtime_weighted_sampling_with_replacement'}, {}, baseline, assessment)
    def fake_train(*args, **kwargs):
        out = Path(cfg['output'])
        out.mkdir(parents=True)
        metadata = {}
        kwargs['configure_run'](None, out, metadata)
        assert kwargs['trainer_factory'].func is ExperimentBTrainer
        assert metadata['baseline_reference'] == 'A_baseline'
        assert metadata['main_experimental_change'] == 'class-aware sampling'
        assert metadata['sampler_actually_used'] is False
        assert metadata['human_review_sha256'] == 'reviewhash'
        assert (out/'sampling_plan.json').is_file()
        assert (out/'sampling_justification.json').is_file()
        return metadata
    monkeypatch.setattr(pipeline, 'train', fake_train)
    result = pipeline.execute(state, root=tmp_path)
    assert result['implementation_sha256'] == {'code':'hash'}
    cfg['experiment_name'] = 'A_baseline'
    cfg['output'] = str(baseline_dir)
    with pytest.raises(ValueError, match='cannot write to baseline'):
        pipeline.execute(state, root=tmp_path)

@pytest.mark.parametrize('value', [0.0, 0.1, 0.025])
def test_baseline_runtime_inheritance(tmp_path, value):
    saved = dict(warmup_bias_lr=0.1, optimizer='auto', lr0=0.01, momentum=0.937)
    runtime = dict(saved, warmup_bias_lr=value)
    (tmp_path/'args.yaml').write_text(yaml.safe_dump(saved))
    (tmp_path/'framework_resolved_args.yaml').write_text(yaml.safe_dump(runtime))
    metadata = dict(resolved_training_arguments=runtime)
    assert pipeline.baseline_runtime_arguments(tmp_path, metadata) == runtime
    metadata['resolved_training_arguments'] = dict(runtime, lr0=0.2)
    with pytest.raises(ValueError, match='disagree'):
        pipeline.baseline_runtime_arguments(tmp_path, metadata)


@pytest.mark.parametrize('key', ['warmup_bias_lr','lr0','lrf','momentum','warmup_epochs',
                                'warmup_momentum','weight_decay','optimizer','model','device'])
def test_runtime_optimizer_argument_drift_rejected(key):
    baseline_dir = PROJECT_ROOT/'results/experiments/A_baseline'
    metadata = json.loads((baseline_dir/'experiment.json').read_text())
    baseline = pipeline.baseline_runtime_arguments(baseline_dir, metadata)
    report = compare_baseline_arguments(dict(baseline), baseline)
    assert report['fields'][key]['status'] == 'MATCH'
    with pytest.raises(ValueError, match='drift'):
        compare_baseline_arguments(dict(baseline, **{key:'unexpected'}), baseline)


def test_expected_output_and_sampling_differences():
    base = dict(name='A_baseline', project='old', save_dir='old/A', epochs=100,
                warmup_bias_lr=0.0, model='yolo11n.pt')
    actual = dict(base, name=pipeline.NAME, project='new', save_dir='new/B', epochs=2)
    report = compare_baseline_arguments(actual, base, smoke=True)
    assert report['result'] == 'PASS'
    assert report['sampling_strategy']['status'] == 'EXPECTED DIFFERENCE'
    assert report['fields']['name']['status'] == 'EXPECTED DIFFERENCE'
    with pytest.raises(ValueError, match='drift'):
        compare_baseline_arguments(dict(actual, unknown_training_option=True), base, smoke=True)


def test_full_execution_blocked_after_failed_smoke(tmp_path, monkeypatch):
    cfg = dict(experiment_name=pipeline.NAME, experiment_type='class_aware_sampling',
               output=str(output_path(tmp_path, pipeline.NAME)))
    state = (cfg, {}, {}, [], {}, {}, {}, {})
    monkeypatch.setattr(pipeline, 'smoke_status', lambda *args: 'FAIL')
    monkeypatch.setattr(pipeline, 'train', lambda *args, **kwargs: pytest.fail('Full training started'))
    with pytest.raises(ValueError, match='matching passing GPU smoke'):
        pipeline.execute(state, root=tmp_path)


def test_readiness_requires_matching_passing_smoke(tmp_path, monkeypatch):
    out = output_path(tmp_path, pipeline.NAME, True)
    out.mkdir(parents=True)
    args = dict(epochs=100, warmup_bias_lr=0.0)
    baseline = dict(optimizer_class='AdamW', optimizer_initial_parameter_groups=[], parameter_count=1)
    state = ({}, dict(torch_version='torch', ultralytics_version='8.4.128'),
             dict(curated_hash='dataset'), [], {}, args, baseline, dict(human_review_sha256='review'))
    monkeypatch.setattr(pipeline, 'file_hash', lambda path: 'hash')
    monkeypatch.setattr(pipeline, 'implementation_hashes', lambda root: {'code':'hash'})
    monkeypatch.setattr(pipeline, 'source_evidence', lambda: {})
    monkeypatch.setattr(pipeline, 'discover_checkpoints', lambda path: ('best', 'last'))
    meta = dict(training_status='PASS', dataset_immutability='PASS', baseline_immutability='PASS',
        sampler_actually_used=True, optimizer_matches_baseline=True, epochs_completed=2,
        batch=4, workers_resolved=0, imgsz=416, seed=42, test_status='LOCKED',
        torch_version='torch', ultralytics_version='8.4.128', dataset_fingerprint='dataset', device='0',
        implementation_sha256={'code':'hash'}, baseline_args_sha256='hash',
        baseline_runtime_sha256=pipeline.baseline_runtime_signature(args, baseline),
        human_review_sha256='review', resolved_training_arguments=dict(args, epochs=2),
        sampler_diagnostics_sha256='hash', ultralytics_source_inspection={},
        checkpoint_sha256={'best':'hash','last':'hash'}, baseline_argument_report_sha256='hash', **baseline)
    diagnostic = dict(sampler_actually_used=True, sampler_class='ClassAwareSampler', replacement=True,
        num_samples=1158, seed=42, multiplier=1.573, unique_train_images=1158,
        images_with_without_helmet=450, valid_leakage=0, test_leakage=0,
        epochs=[dict(actual_sample_count=1158, sampler_actually_used=True, completed_optimizer_steps=1)]*2)
    (out/'sampler_diagnostics.json').write_text(json.dumps(diagnostic))
    (out/'baseline_argument_comparison.json').write_text('{"result":"PASS"}')
    def status(candidate):
        (out/'experiment.json').write_text(json.dumps(candidate))
        return pipeline.smoke_status(tmp_path, state)
    assert status(meta) == 'PASS'
    for changes in [dict(training_status='FAIL'), dict(baseline_runtime_sha256='stale'),
                    dict(implementation_sha256={}), dict(device='cpu'), dict(optimizer_class='SGD'),
                    dict(resolved_training_arguments=dict(args, epochs=2, warmup_bias_lr=0.1))]:
        assert status(dict(meta, **changes)) == 'FAIL'
