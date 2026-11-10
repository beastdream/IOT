"""Frozen C sampling and real CPU DataLoader tests; never train a model."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from smart_helmet.paths import PROJECT_ROOT
from smart_helmet.training import experiment_c as pipeline
from smart_helmet.training.hard_example_sampler import (
    PLAN, TIERS, TIER_WEIGHTS, HardExampleSampler, load_plan, validate_mapping, ordered_inventory,
)
from smart_helmet.training.experiment import output_path, prepare_output


@pytest.fixture(autouse=True)
def isolated_framework_import():
    from PIL import Image
    original = Image.open
    global ExperimentCTrainer, compare_baseline_arguments
    try:
        from smart_helmet.training.experiment_c_trainer import ExperimentCTrainer
        from smart_helmet.training.experiment_b_trainer import compare_baseline_arguments
        yield
    finally:
        Image.open = original


@pytest.fixture(scope='module')
def frozen():
    return load_plan(PROJECT_ROOT)


def test_frozen_plan_load_and_exact_simulation(frozen):
    plan, rows = frozen
    sampler = HardExampleSampler(PROJECT_ROOT, rows, plan)
    assert sampler.num_samples == len(sampler) == 1158
    assert sampler.replacement and sampler.seed == 42
    assert [rows[i]['image'] for i in sampler.indices()] == plan['simulated_epoch_zero_images']
    assert len(set(sampler.indices())) < 1158
    assert sampler.indices() == HardExampleSampler(PROJECT_ROOT, rows, plan).indices()
    assert sampler.indices(1) != sampler.indices(0)
    for r,w in zip(rows,sampler.weights):
        assert w == (1.0 if r['image'] in plan['excluded_from_upweighting'] else TIER_WEIGHTS[r['tier']])


@pytest.mark.parametrize('key,value', [('plan_stage','pre_human_review'),('human_review_gate','FAIL'),
    ('human_review_pending',1),('review_more',1),('difficulty_source','VALID'),('seed',43),
    ('effective_epoch_samples',1159),('test_status','OPEN')])
def test_invalid_plan_gates(frozen,key,value):
    plan,rows = frozen
    with pytest.raises(ValueError): validate_mapping(PROJECT_ROOT,dict(plan,**{key:value}),rows)


def test_weight_and_eligibility_rejections(frozen):
    plan,rows = frozen
    for mutation in ('missing','suspicious','eligible','tier'):
        bad = deepcopy(plan)
        if mutation == 'missing': bad['image_sampling_weights'].pop(rows[0]['image'])
        if mutation == 'suspicious': bad['image_sampling_weights'][bad['suspicious_excluded_images'][0]] = 2
        if mutation == 'eligible': bad['eligible_hard_images'].pop()
        if mutation == 'tier': bad['proposed_tier_weights'][TIERS[1]] = 2
        with pytest.raises(ValueError): validate_mapping(PROJECT_ROOT,bad,rows)


def test_not_actually_hard_baseline_exposure(frozen):
    plan,rows = frozen
    changed = deepcopy(plan)
    name = changed['eligible_hard_images'].pop()
    changed['not_actually_hard_excluded_images'].append(name)
    changed['excluded_from_upweighting'].append(name)
    changed['image_sampling_weights'][name] = 1.0
    sampler = HardExampleSampler(PROJECT_ROOT,rows,changed)
    assert sampler.weights[next(i for i,r in enumerate(rows) if r['image']==name)] == 1.0
    changed['image_sampling_weights'][name] = 2.0
    with pytest.raises(ValueError,match='exclusion'): validate_mapping(PROJECT_ROOT,changed,rows)


@pytest.mark.parametrize('split',['valid','test'])
def test_train_restriction_and_test_lock(frozen,split):
    plan,rows = frozen
    bad = [dict(rows[0],image=f'data/curated/v1/{split}/images/x.jpg')]+rows[1:]
    with pytest.raises(ValueError,match='leakage'): validate_mapping(PROJECT_ROOT,plan,bad)
    trainer = object.__new__(ExperimentCTrainer)
    trainer.project_root = PROJECT_ROOT
    with pytest.raises(ValueError): trainer.get_dataloader(PROJECT_ROOT/f'data/curated/v1/{split}/images')
    with pytest.raises(ValueError,match='TEST locked'): trainer.get_dataloader('anything',mode='test')


class IndexDataset:
    def __init__(self,paths): self.im_files=paths
    def __len__(self): return len(self.im_files)
    def __getitem__(self,index): return index


def test_c_trainer_attaches_sampler_and_observes_actual_order(frozen,tmp_path,monkeypatch):
    plan,rows=frozen
    trainer=object.__new__(ExperimentCTrainer)
    trainer.project_root=PROJECT_ROOT
    trainer.inventory=rows
    trainer.sampling_plan=plan
    trainer.args=SimpleNamespace(workers=0,rect=False,compile=False,fraction=1)
    trainer.device='cpu'
    trainer.save_dir=tmp_path
    trainer.exposure_epochs=[]
    trainer.completed_optimizer_steps=1
    trainer.completed_batches=290
    dataset=IndexDataset([str(PROJECT_ROOT/r['image']) for r in rows][::-1])
    monkeypatch.setattr(trainer,'build_dataset',lambda *args:dataset)
    loader=trainer.get_dataloader(PROJECT_ROOT/'data/curated/v1/train/images')
    trainer.train_loader=loader
    sampler=trainer.class_aware_sampler
    assert type(sampler) is HardExampleSampler and loader.sampler is sampler
    assert sampler.rows == rows[::-1]
    for epoch in range(3):
        trainer.epoch=epoch
        trainer.start_sampling_epoch(trainer)
        if epoch==2: loader.reset()
        trainer.observed_indices=[int(i) for batch in loader for i in batch]
        assert trainer.observed_indices == sampler.indices() == sampler.yielded_indices
        trainer.finish_sampling_epoch(trainer)
    saved=json.loads((tmp_path/'sampler_diagnostics.json').read_text())
    assert saved['sampler_actually_used'] and len(saved['epochs'])==3
    assert sum(saved['sampled_tier_counts'].values())==1158
    assert saved['exposure_plausible'] and saved['maximum_sample_count']>1
    trainer.observed_indices.reverse()
    with pytest.raises(ValueError,match='order'):trainer.finish_sampling_epoch(trainer)
    loader.close()


def test_valid_loader_unchanged(monkeypatch):
    from ultralytics.models.yolo.detect import DetectionTrainer
    trainer=object.__new__(ExperimentCTrainer)
    trainer.project_root=PROJECT_ROOT
    monkeypatch.setattr(DetectionTrainer,'get_dataloader',lambda *args:'unchanged')
    assert trainer.get_dataloader(PROJECT_ROOT/'data/curated/v1/valid/images',mode='val')=='unchanged'


@pytest.mark.parametrize('key',['model','imgsz','batch','workers','seed','optimizer','lr0','lrf','momentum',
    'weight_decay','warmup_epochs','warmup_momentum','warmup_bias_lr','mosaic','mixup','patience','pretrained'])
def test_runtime_arguments_and_drift(key):
    directory=PROJECT_ROOT/'results/experiments/A_baseline'
    meta=json.loads((directory/'experiment.json').read_text())
    args=pipeline.baseline_runtime_arguments(directory,meta)
    actual=dict(args,epochs=2,name=pipeline.NAME+'_smoke')
    assert compare_baseline_arguments(actual,args,smoke=True)['result']=='PASS'
    with pytest.raises(ValueError,match='drift'):
        compare_baseline_arguments(dict(actual,**{key:'drift'}),args,smoke=True)


def test_cli_constants_and_output_protection(tmp_path):
    config=PROJECT_ROOT/pipeline.CONFIG
    assert pipeline.load_config(config)['epochs']==100
    assert pipeline.load_config(config,{'epochs':2},True)['epochs']==2
    for override in ({'epochs':2},{'batch':8},{'workers':2},{'device':'cpu'}):
        with pytest.raises(ValueError):pipeline.load_config(config,override)
    out=output_path(tmp_path,pipeline.NAME,True)
    assert out != output_path(tmp_path,pipeline.NAME)
    prepare_output(out)
    (out/'evidence').write_text('keep')
    with pytest.raises(ValueError):prepare_output(out)
    prepare_output(out,True)
    assert list(out.parent.glob(out.name+'_archive_*/evidence'))[0].read_text()=='keep'
    cfg=dict(experiment_name='A_baseline',experiment_type='smoke',output=str(out))
    with pytest.raises(ValueError,match='other experiment'):pipeline.execute((cfg,{}, {},[],{},{},{}),root=tmp_path)


def test_full_requires_passing_smoke_and_readiness_fails_closed(tmp_path,monkeypatch):
    assert pipeline.smoke_status(tmp_path,None)=='NOT RUN'
    out=output_path(tmp_path,pipeline.NAME,True)
    out.mkdir(parents=True)
    (out/'experiment.json').write_text('{"training_status":"FAIL"}')
    assert pipeline.smoke_status(tmp_path,None)=='FAIL'
    cfg=dict(experiment_name=pipeline.NAME,experiment_type='hard_example_sampling',output=str(output_path(tmp_path,pipeline.NAME)))
    monkeypatch.setattr(pipeline,'train',lambda *a,**kw:pytest.fail('Training forbidden'))
    with pytest.raises(ValueError,match='passing GPU smoke'):
        pipeline.execute((cfg,{},{},[],{},{},{}),root=tmp_path)


def test_diagnostic_replay_rejects_tampering(frozen):
    plan,rows=frozen
    sampler=HardExampleSampler(PROJECT_ROOT,rows,plan)
    epochs=[]
    for i in range(2):
        sampler.set_epoch(i)
        epochs.append(dict(sampler.diagnostics(sampler.indices(),True),sampler_attached=True,
            completed_optimizer_steps=i+1,completed_batches=(i+1)*290,device='cuda:0'))
    diagnostic=dict(epochs[0],epochs=epochs)
    pipeline.verify_diagnostics(PROJECT_ROOT,rows,plan,diagnostic)
    for key,value in [('sampled_index_hash','fake'),('suspicious_excluded',0),('completed_optimizer_steps',0)]:
        bad=deepcopy(diagnostic)
        bad['epochs'][0][key]=value
        with pytest.raises(ValueError):pipeline.verify_diagnostics(PROJECT_ROOT,rows,plan,bad)


def test_dataset_immutability_guard_and_runtime_yaml(tmp_path,monkeypatch):
    from smart_helmet.training import environment
    monkeypatch.setattr(environment,'verify_curated_fingerprint',lambda root:{'changed':1})
    with pytest.raises(ValueError,match='Dataset changed'):
        environment.check_unchanged(dict(curated={},raw={}),tmp_path)
    trainer=object.__new__(ExperimentCTrainer)
    trainer.project_root=tmp_path
    data=tmp_path/'runtime.yaml'
    data.write_text(yaml.safe_dump(dict(test='locked',train='train',val='valid')))
    trainer.args=SimpleNamespace(data=str(data),split='val')
    with pytest.raises(ValueError,match='TEST locked'):trainer.get_dataset()


@pytest.mark.parametrize('status,ready,code',[('PASS','YES',0),('FAIL','NO',1),('NOT RUN','NO',1)])
def test_readiness_calculation(tmp_path,monkeypatch,capsys,status,ready,code):
    monkeypatch.setattr(pipeline,'preflight',lambda **kw:({},dict(gpu_name='GPU')))
    monkeypatch.setattr(pipeline,'smoke_status',lambda *args:status)
    assert pipeline.readiness_main(tmp_path)==code
    assert f'READY FOR FULL EXPERIMENT C: {ready}' in capsys.readouterr().out


def test_smoke_status_requires_current_artifacts(tmp_path,monkeypatch,frozen):
    plan,rows=frozen
    args=dict(epochs=100,optimizer='auto')
    baseline=dict(optimizer_class='AdamW',optimizer_initial_parameter_groups=[],parameter_count=1)
    env=dict(torch_version='torch',ultralytics_version='8.4.128')
    state=({},env,dict(curated_hash='curated'),rows,plan,args,baseline)
    out=output_path(tmp_path,pipeline.NAME,True)
    (out/'weights').mkdir(parents=True)
    files=['sampler_diagnostics.json','sampling_plan_snapshot.json','baseline_argument_comparison.json',
        'resolved_config.yaml','environment.json','dataset_fingerprint.json','weights/best.pt','weights/last.pt']
    for name in files:(out/name).write_text('{}')
    (tmp_path/PLAN).parent.mkdir(parents=True)
    (tmp_path/PLAN).write_text('{}')
    (out/'baseline_argument_comparison.json').write_text('{"result":"PASS","unexpected_drift":{}}')
    monkeypatch.setattr(pipeline,'evidence',lambda *a:dict(sampling_plan_sha256='frozen'))
    monkeypatch.setattr(pipeline,'verify_diagnostics',lambda *a:None)
    meta=dict(training_status='PASS',dataset_immutability='PASS',protected_artifacts_immutability='PASS',
        sampler_actually_used=True,epochs_completed=2,validation_images=127,device='0',
        gpu_name='NVIDIA GeForce GTX 1650',batch=4,workers_resolved=0,imgsz=416,seed=42,
        test_status='LOCKED',dataset_fingerprint='curated',**env,**baseline,sampling_plan_sha256='frozen',
        resolved_training_arguments=dict(args,epochs=2),artifact_sha256={n:pipeline.file_hash(out/n) for n in files})
    def status(changes):
        (out/'experiment.json').write_text(json.dumps(dict(meta,**changes)))
        return pipeline.smoke_status(tmp_path,state)
    assert status({})=='PASS'
    for changes in (dict(device='cpu'),dict(epochs_completed=1),dict(sampling_plan_sha256='stale'),
                    dict(dataset_immutability='FAIL'),dict(artifact_sha256={}),dict(optimizer_class='SGD')):
        assert status(changes)=='FAIL'
    (out/'weights/best.pt').write_text('tampered')
    assert status({})=='FAIL'
