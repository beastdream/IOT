"""Resume safety tests. Read real provenance; never train or mutate experiment artifacts."""
from copy import deepcopy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from smart_helmet.paths import PROJECT_ROOT
from smart_helmet.training import experiment_c as pipeline
from smart_helmet.training import experiment_c_resume as resume


@pytest.fixture(scope='module')
def interrupted():
    out = PROJECT_ROOT/'results/experiments'/pipeline.NAME
    # Use the immutable first two epochs, even after the real run has completed
    # and Ultralytics strips the final checkpoint's optimizer and epoch.
    with (out/'results.csv').open(newline='') as stream:
        history = list(csv.DictReader(stream))[:2]
    results = {k:[float(row[k]) for row in history] for k in history[0]}
    diagnostic = resume.read_json(out/'sampler_diagnostics.json')
    meta = resume.read_json(out/'experiment.json')
    checkpoint = dict(epoch=1, train_results=results, optimizer=dict(state={0:{}}), ema=object(), scaler={},
        train_args=meta['resolved_training_arguments'], version=meta['ultralytics_version'],
        updates=diagnostic['epochs'][1]['completed_optimizer_steps'],
        best_fitness=max(results['metrics/mAP50-95(B)']))
    return pipeline.preflight(), checkpoint


@pytest.fixture
def fast_validation(interrupted, monkeypatch):
    state, checkpoint = interrupted
    monkeypatch.setattr(pipeline, 'preflight', lambda **kw:state)
    monkeypatch.setattr(torch, 'load', lambda *a, **kw:checkpoint)
    monkeypatch.setattr(resume, 'recover_epochs', lambda *a:2)
    def interrupted_meta(meta):
        meta['training_status'] = 'RUNNING'
    mutate_json(monkeypatch, 'experiment.json', interrupted_meta)
    mutate_json(monkeypatch, 'sampler_diagnostics.json', lambda v:v.update(epochs=v['epochs'][:2]))
    return state, checkpoint


def mutate_json(monkeypatch, filename, mutate):
    original = resume.read_json
    def read(path):
        value = original(path)
        if Path(path).name == filename:
            mutate(value)
        return value
    monkeypatch.setattr(resume, 'read_json', read)


def test_missing_last_pt_never_falls_back(tmp_path):
    out = tmp_path/'results/experiments'/pipeline.NAME/'weights'
    out.mkdir(parents=True)
    (out/'best.pt').write_text('not a resume checkpoint')
    with pytest.raises(ValueError, match='last.pt'):
        resume.validate_resume(tmp_path)


def test_safe_resume_and_interrupted_detection(fast_validation):
    report = {}
    result = resume.validate_resume(report=report)
    completed = result['checkpoint']['epoch']+1
    assert result['completed'] == completed
    assert report['INTERRUPTED RUN DETECTED'] == 'YES'
    assert report['LAST COMPLETED EPOCH'] == completed
    assert report['RESUME FROM EPOCH'] == completed+1
    assert report['READY TO RESUME'] == 'YES'
    assert report['TEST STATUS'] == 'LOCKED'
    assert len(result['epochs']) == completed
    assert result['stopper']['best_epoch'] > 0


@pytest.mark.parametrize('filename,field,value,message', [
    ('experiment.json','dataset_fingerprint','changed','fingerprint'),
    ('sampling_plan_snapshot.json','seed',43,'plan mismatch'),
    ('experiment.json','test_status','OPEN','TEST lock'),
    ('experiment.json','training_status','PASS','interrupted'),
])
def test_provenance_failure(fast_validation, monkeypatch, filename, field, value, message):
    mutate_json(monkeypatch, filename, lambda v:v.update({field:value}))
    with pytest.raises(ValueError, match=message):
        resume.validate_resume()


@pytest.mark.parametrize('field,value', [('seed',43),('test_leakage',1),('valid_leakage',1),('epoch',99)])
def test_sampler_seed_and_leakage_rejected(fast_validation, monkeypatch, field, value):
    mutate_json(monkeypatch, 'sampler_diagnostics.json', lambda v:v['epochs'][0].update({field:value}))
    with pytest.raises(ValueError, match='Sampler configuration'):
        resume.validate_resume()


def test_epoch_recovery_and_disagreement(tmp_path):
    path = tmp_path/'results.csv'
    path.write_text('epoch,metric\n1,0.1\n2,0.2\n')
    checkpoint = dict(epoch=1, train_results=dict(epoch=[1,2], metric=[0.1,0.2]))
    assert resume.recover_epochs(path, checkpoint, 100) == 2
    for epoch in (-1, 0, 2, 99):
        with pytest.raises(ValueError):
            resume.recover_epochs(path, dict(checkpoint, epoch=epoch), 100)
    path.write_text('epoch,metric\n1,0.1\n1,0.2\n')
    with pytest.raises(ValueError, match='disagreement'):
        resume.recover_epochs(path, checkpoint, 100)


def test_baseline_drift_and_foreign_output_rejected(fast_validation):
    state, checkpoint = fast_validation
    baseline = state[5]
    out = PROJECT_ROOT/'results/experiments'/pipeline.NAME
    for field, value in (('batch',8), ('split','test'), ('save_dir',str(out.parent/'A_baseline')),
                         ('resume',str(out/'weights/best.pt')), ('epochs',101)):
        with pytest.raises(ValueError):
            resume.check_arguments(dict(checkpoint['train_args'], **{field:value}), baseline, out)


def test_missing_optimizer_refuses_resume(fast_validation, monkeypatch):
    _, checkpoint = fast_validation
    monkeypatch.setattr(torch,'load',lambda *a,**kw:dict(checkpoint,optimizer=None))
    with pytest.raises(ValueError, match='optimizer'):
        resume.validate_resume()


def test_early_stopping_preserved_and_completed_run_rejected():
    cp = dict(best_fitness=0.4, train_results={'metrics/mAP50-95(B)':[0.2,0.4,0.3]})
    state = resume.stopping_state(cp, 20, 3)
    assert state == dict(best_fitness=0.4,best_epoch=2,patience=20,possible_stop=False)
    cp['c_resume_state'] = dict(early_stopping=state)
    assert resume.stopping_state(cp, 20, 3) == state
    with pytest.raises(ValueError, match='already reached'):
        resume.stopping_state({k:v for k,v in cp.items() if k != 'c_resume_state'}, 1, 3)
    cp['best_fitness'] = 0.5
    with pytest.raises(ValueError, match='ambiguous'):
        resume.stopping_state(cp, 20, 3)


@pytest.mark.parametrize('option', ['--overwrite','--smoke-test','--epochs=100','--device=0','--batch=4'])
def test_resume_cli_rejects_overrides(option):
    with pytest.raises(SystemExit):
        pipeline.main(['--resume', option])


def test_cli_validation_failure_cannot_train(tmp_path, monkeypatch, capsys):
    (tmp_path/'results/experiments'/pipeline.NAME).mkdir(parents=True)
    monkeypatch.setattr(resume, 'execute_resume', lambda *a:pytest.fail('Training forbidden'))
    assert resume.resume_main(root=tmp_path) == 1
    assert 'READY TO RESUME:\nNO' in capsys.readouterr().out


def test_cli_dry_run_cannot_train(tmp_path, monkeypatch):
    (tmp_path/'results/experiments'/pipeline.NAME).mkdir(parents=True)
    monkeypatch.setattr(resume, 'validate_resume', lambda *a:{})
    monkeypatch.setattr(resume, 'execute_resume', lambda *a:pytest.fail('Training forbidden'))
    assert resume.resume_main(dry_run=True, root=tmp_path) == 0


def test_resume_runtime_restores_optimizer_scaler_and_stopper(monkeypatch):
    from smart_helmet.training import experiment_c_resume_trainer as module
    from smart_helmet.training.experiment_c_resume_trainer import ResumeExperimentCTrainer, state_equal
    from smart_helmet.training.experiment_c_trainer import ExperimentCTrainer
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter])
    torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch:1.0)
    parameter.sum().backward(); optimizer.step()
    optimizer.param_groups[0]['lr'] = 0.0005
    saved = deepcopy(optimizer.state_dict())
    fresh_parameter = torch.nn.Parameter(torch.zeros(2))
    fresh = torch.optim.AdamW([fresh_parameter])
    torch.optim.lr_scheduler.LambdaLR(fresh, lambda epoch:1.0)
    obj = object.__new__(ResumeExperimentCTrainer)
    obj.optimizer = fresh
    obj.model = SimpleNamespace(parameters=lambda:[fresh_parameter])
    obj.baseline_optimizer = json.loads(json.dumps(module.training_details(obj)))
    obj.scaler = SimpleNamespace(state_dict=lambda:{})
    obj.stopper = SimpleNamespace()
    obj.resume_context = dict(completed=10, stopper=dict(best_epoch=9, best_fitness=0.4, patience=20, possible_stop=False))
    def framework(self, checkpoint):
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch']+1
    monkeypatch.setattr(ExperimentCTrainer, 'resume_training', framework)
    obj.resume_training(dict(epoch=9, optimizer=saved, scaler={}))
    assert state_equal(fresh.state_dict(), saved)
    assert obj.start_epoch == 10 and obj.stopper.best_epoch == 9


def test_scheduler_and_sampler_resume_next_epoch(tmp_path, monkeypatch):
    from smart_helmet.training import experiment_c_resume_trainer as module
    obj = object.__new__(module.ResumeExperimentCTrainer)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=0.001667)
    schedule = lambda epoch:max(1-epoch/100,0)*0.99+0.01
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    for _ in range(10):
        optimizer.step(); scheduler.step()
    obj.optimizer = optimizer
    obj.scheduler = scheduler
    obj.start_epoch = 10
    obj.save_dir = tmp_path
    obj.args = SimpleNamespace()
    obj.baseline_args = {}
    obj.stopper = SimpleNamespace(best_epoch=9)
    class Sampler:
        def set_epoch(self, epoch): self.epoch=epoch
    obj.class_aware_sampler = Sampler()
    obj.train_loader = SimpleNamespace(sampler=obj.class_aware_sampler, reset=lambda:None)
    obj.test_loader = SimpleNamespace(sampler=object())
    obj.resume_context = dict(out=tmp_path,completed=10,checkpoint={})
    monkeypatch.setattr(module,'check_arguments',lambda *a:dict(result='PASS'))
    obj.verify_attachment(obj)
    assert scheduler.last_epoch == 9 and obj.class_aware_sampler.epoch == 10
    optimizer.step(); scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.001667*schedule(10))


def test_saved_checkpoint_contains_control_state_and_atomic_last(tmp_path, monkeypatch):
    from smart_helmet.training.experiment_c_resume_trainer import ResumeExperimentCTrainer
    from smart_helmet.training.experiment_c_trainer import ExperimentCTrainer
    obj = object.__new__(ResumeExperimentCTrainer)
    obj.last, obj.best = tmp_path/'last.pt', tmp_path/'best.pt'
    obj.last.write_bytes(b'old')
    obj.scheduler = SimpleNamespace(state_dict=lambda:dict(last_epoch=10))
    obj.stopper = SimpleNamespace(best_epoch=9,patience=20)
    def framework(self):
        import io
        stream=io.BytesIO()
        torch.save(dict(epoch=10),stream)
        self.last.write_bytes(stream.getvalue())
        return True
    monkeypatch.setattr(ExperimentCTrainer,'save_model',framework)
    assert obj.save_model()
    saved=torch.load(obj.last, weights_only=True)
    assert saved['c_resume_state']['scheduler']['last_epoch']==10
    assert saved['c_resume_state']['early_stopping']['best_epoch']==9
    assert not list(tmp_path.glob('*.pending'))
