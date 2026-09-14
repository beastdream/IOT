"""No real training, downloads, or real dataset writes in these tests."""
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import yaml

from smart_helmet.training import environment as env, trainer as tr, experiment as ex
from smart_helmet.paths import PROJECT_ROOT


def config_file(tmp_path):
    cfg=yaml.safe_load((PROJECT_ROOT/'configs/baseline.yaml').read_text(encoding='utf-8-sig'))
    path=tmp_path/'baseline.yaml'; path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.fixture
def setup(tmp_path,monkeypatch):
    runtime=dict(python_version='3.11',torch_version='mock',ultralytics_version='mock',gpu_name=None,
                 device='cpu',device_message='CUDA unavailable -> using CPU.',auto_batch_supported=True,fraction_supported=True)
    snapshot=dict(config=str(tmp_path/'configs/data.curated.v1.yaml'),
                  data=dict(path=str(tmp_path/'data/curated/v1'),train='train/images',val='valid/images',test='test/images',nc=2,names={0:'With Helmet',1:'Without Helmet'}),
                  counts={'train':100,'val':10,'test':10},curated={'image':'hash'},raw={'raw':'hash'},curated_hash='frozen',raw_hash='raw')
    monkeypatch.setattr(tr,'runtime_info',lambda *_:runtime)
    monkeypatch.setattr(tr,'verify_dataset',lambda *_:snapshot)
    monkeypatch.setattr(tr,'check_unchanged',lambda *_:'PASS')
    return config_file(tmp_path),runtime,snapshot


def test_config_loading_and_policy(tmp_path):
    path=config_file(tmp_path)
    cfg=tr.load_config(path,{'batch':'4','workers':0})
    assert cfg['imgsz']==416 and cfg['epochs']==100 and cfg['batch']=='4' and cfg['workers']==0
    for overrides in ({'imgsz':640},{'cache':True},{'model':'yolo11m.pt'},{'workers':-1},{'smoke_epochs':100}):
        with pytest.raises(ValueError):tr.load_config(path,overrides)


def test_device_auto_cpu_gpu_and_batch():
    torch=Mock();torch.cuda.is_available.return_value=False
    assert env.resolve_device('auto',torch)[0]=='cpu'
    assert env.resolve_batch('auto','cpu',True)[0]==2
    torch.cuda.is_available.return_value=True;torch.cuda.current_device.return_value=1;torch.cuda.device_count.return_value=2
    assert env.resolve_device('auto',torch)[0]=='1'
    assert env.resolve_batch('auto','1',True)[0]==-1
    assert env.resolve_batch('auto','1',False)[0]==2
    assert env.resolve_batch('8','cpu',True)[0]==8
    with pytest.raises(ValueError):env.resolve_device('3',torch)
    with pytest.raises(ValueError):env.resolve_batch('-1','cpu',True)


def test_preflight_overrides_and_separation(setup,tmp_path):
    path,_,_=setup
    full,_,_=tr.preflight(path,{'batch':'4','workers':0},root=tmp_path)
    smoke,_,_=tr.preflight(path,smoke=True,root=tmp_path)
    assert full['batch']==4 and full['workers']==0
    assert full['cli_overrides']=={'batch':'4','workers':0}
    assert Path(full['output']).name=='A_baseline'
    assert Path(smoke['output']).name=='A_baseline_smoke'
    assert full['fraction']==1 and smoke['fraction']==0.02 and smoke['epochs']==2


def test_dry_run_does_not_construct_model(setup,monkeypatch,capsys,tmp_path):
    path,_,_=setup
    train=Mock();monkeypatch.setattr(tr,'train',train)
    assert tr.main(['--config',str(path),'--dry-run','--batch','3','--workers','0'])==0
    train.assert_not_called()
    assert 'Test: LOCKED' in capsys.readouterr().out
    assert not (tmp_path/'results').exists()


class FakeModel:
    instances=[]
    missing_best=False
    def __init__(self,path):
        self.model=SimpleNamespace(parameters=lambda:[]);self.callbacks={};self.calls=[]
        self.instances.append(self)
    def add_callback(self,event,fn):self.callbacks[event]=fn
    def val(self,**kwargs):raise AssertionError('No standalone/test evaluation permitted')
    def train(self,**kwargs):
        self.calls.append(kwargs)
        data=yaml.safe_load(Path(kwargs['data']).read_text())
        assert 'test' not in data and 'download' not in data
        assert kwargs['split']=='val' and kwargs['val'] is True
        out=Path(kwargs['project'])/kwargs['name'];(out/'weights').mkdir()
        (out/'weights/last.pt').write_bytes(b'fake')
        if not self.missing_best:(out/'weights/best.pt').write_bytes(b'fake')
        (out/'results.csv').write_text('epoch,metrics/mAP50(B)\n1,0.5\n2,0.6\n')
        ex.write_yaml(out/'args.yaml',kwargs)
        self.callbacks['on_pretrain_routine_end'](SimpleNamespace(args=SimpleNamespace(**kwargs),batch_size=kwargs['batch'],
            model=self.model,optimizer=SimpleNamespace(param_groups=[{'lr':0.001667,'params':[]}])) )
        return SimpleNamespace(results_dict={'metrics/mAP50(B)':0.6})


def run_fake(setup,tmp_path,smoke=True,factory=FakeModel):
    path,_,_=setup
    cfg,runtime,snapshot=tr.preflight(path,smoke=smoke,root=tmp_path)
    return tr.train(cfg,runtime,snapshot,root=tmp_path,model_factory=factory,cache_context=lambda _:nullcontext())


def test_smoke_then_full_metadata_checkpoints_test_lock(setup,tmp_path):
    smoke=run_fake(setup,tmp_path)
    full=run_fake(setup,tmp_path,False)
    assert smoke['training_status']==full['training_status']=='PASS'
    assert full['epochs_requested']==100 and full['epochs_completed']==2
    assert full['test_status']=='LOCKED' and full['dataset_immutability']=='PASS'
    for key in ('model_checkpoint','ultralytics_version','torch_version','python_version','device','gpu_name','dataset_config','dataset_version','dataset_fingerprint','class_mapping','imgsz','batch','workers','seed','deterministic','pretrained','start_time','end_time','duration_seconds','best_model_path','last_model_path'):
        assert key in full
    out=tmp_path/'results/experiments/A_baseline'
    for name in ('experiment.json','resolved_config.yaml','environment.json','dataset_fingerprint.json','results.csv','args.yaml','framework_resolved_args.yaml'):
        assert (out/name).is_file()
    assert yaml.safe_load((out/'framework_resolved_args.yaml').read_text())['split']=='val'
    assert full['validation_metrics']=={'metrics/mAP50(B)':0.6}
    assert full['optimizer_initial_parameter_groups']==[{'lr':0.001667}]


def test_full_requires_matching_smoke(setup,tmp_path):
    with pytest.raises(ValueError,match='smoke-test'):run_fake(setup,tmp_path,False)
    run_fake(setup,tmp_path)
    path=tmp_path/'results/experiments/A_baseline_smoke/experiment.json'
    data=json.loads(path.read_text());data['dataset_fingerprint']='changed';ex.write_json(path,data)
    with pytest.raises(ValueError,match='prerequisite mismatch'):run_fake(setup,tmp_path,False)


def test_missing_best_is_failure(setup,tmp_path):
    run_fake(setup,tmp_path)
    class Missing(FakeModel):missing_best=True
    with pytest.raises(ValueError,match='best.pt missing'):run_fake(setup,tmp_path,False,Missing)
    metadata=json.loads((tmp_path/'results/experiments/A_baseline/experiment.json').read_text())
    assert metadata['training_status']=='FAIL' and metadata['dataset_immutability']=='PASS'


def test_overwrite_archives_preserves_artifacts(tmp_path):
    out=ex.output_path(tmp_path,'A_baseline');ex.prepare_output(out)
    (out/'results.csv').write_text('original')
    with pytest.raises(ValueError,match='overwrite'):ex.prepare_output(out)
    ex.prepare_output(out,True)
    assert not list(out.iterdir())
    archive=list(out.parent.glob('A_baseline_archive_*'))
    assert len(archive)==1 and (archive[0]/'results.csv').read_text()=='original'
    with pytest.raises(ValueError):ex.output_path(tmp_path,'../escape')


def test_fingerprint_verification_detects_mutation(monkeypatch,tmp_path):
    snapshot={'curated':{'a':'1'},'raw':{'b':'2'}}
    monkeypatch.setattr(env,'verify_curated_fingerprint',lambda _:snapshot['curated'])
    monkeypatch.setattr(env,'raw_snapshot',lambda _:snapshot['raw'])
    assert env.check_unchanged(snapshot,tmp_path)=='PASS'
    monkeypatch.setattr(env,'verify_curated_fingerprint',lambda _:{'a':'changed'})
    with pytest.raises(ValueError,match='changed'):env.check_unchanged(snapshot,tmp_path)


def test_cache_redirect_preserves_dataset(tmp_path,monkeypatch):
    from ultralytics.data import dataset
    saved=[]
    def saver(prefix,path,content,version):
        content['version']=version;saved.append(path);path.write_text('cache')
    monkeypatch.setattr(dataset,'save_dataset_cache_file',saver)
    frozen=tmp_path/'frozen';frozen.mkdir()
    with tr.frozen_cache(tmp_path/'experiment_cache'):
        content={};dataset.save_dataset_cache_file('',frozen/'labels.cache',content,'1')
    assert not list(frozen.iterdir()) and saved[0].parent==tmp_path/'experiment_cache'
    assert content['version']=='1' and dataset.save_dataset_cache_file is saver


def test_windows_entrypoint_guard():
    import ast
    tree=ast.parse((PROJECT_ROOT/'scripts/train_baseline.py').read_text(encoding='utf-8-sig'))
    guarded=[n for n in tree.body if isinstance(n,ast.If) and '__name__' in ast.unparse(n.test)]
    assert len(guarded)==1 and 'freeze_support()' in ast.unparse(guarded[0])


def test_frozen_image_repair_refused(tmp_path):
    from PIL import Image
    frozen=tmp_path/'frozen';frozen.mkdir()
    with tr.frozen_cache(tmp_path/'cache',[frozen]):
        with pytest.raises(ValueError,match='Frozen dataset'):
            Image.new('RGB',(8,8)).save(frozen/'image.jpg')
        Image.new('RGB',(8,8)).save(tmp_path/'plot.jpg')
    assert not list(frozen.iterdir()) and (tmp_path/'plot.jpg').exists()


def test_no_silent_oom_retry():
    trainer=SimpleNamespace(_oom_retries=0)
    tr.disable_oom_retry(trainer)
    assert trainer._oom_retries==3
