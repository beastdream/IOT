"""Synthetic/temp data only; no GPU inference or model downloads in tests."""
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from PIL import Image

from smart_helmet.evaluation import metrics as mt, evaluator as ev, failure_analysis as fa
from smart_helmet.foundation import EXPECTED_CLASSES
from smart_helmet.training.experiment import write_json


def box(c=1,b=(0,0,10,10),conf=None):
    value=dict(class_id=c,bbox=list(b))
    if conf is not None:value['confidence']=conf
    return value


def fake_metrics():
    return SimpleNamespace(names=EXPECTED_CLASSES,box=SimpleNamespace(ap_class_index=[1,0],
        mean_results=lambda:[.75,.65,.7,.4],class_result=lambda i:[.7,.5,.6,.3] if i==0 else [.8,.8,.8,.5]))


def test_metric_serialization_mapping_and_comparison(tmp_path):
    result=mt.serialize_metrics(fake_metrics())
    assert result['per_class']['1']['recall']==.5
    assert result['per_class']['0']['mAP50']==.8
    mt.save_metrics(tmp_path,result)
    assert json.loads((tmp_path/'metrics.json').read_text())==result
    assert 'Without Helmet' in (tmp_path/'metrics.csv').read_text()
    comparison=mt.compare_training(result,{'validation_metrics':dict(zip(mt.FRAMEWORK_KEYS,[.1,.65,.7,.4]))})
    assert len(comparison['warnings'])==1
    with pytest.raises(ValueError):mt.metric_values([float('nan'),.5,.5,.5])


@pytest.mark.parametrize('split',['test','train','valid','TEST'])
def test_validation_only_guard(split,monkeypatch,tmp_path):
    verify=Mock();monkeypatch.setattr(ev,'verify_dataset',verify)
    with pytest.raises(ValueError,match='LOCKED'):ev.preflight(tmp_path,split)
    verify.assert_not_called()
    assert ev.main(['--split',split])==1
    assert fa.main(['--split',split])==1


def test_only_best_checkpoint_no_fallback(tmp_path):
    directory=tmp_path/'results/experiments/A_baseline/weights';directory.mkdir(parents=True)
    (directory/'last.pt').write_bytes(b'fake')
    with pytest.raises(ValueError,match='no fallback'):ev.preflight(tmp_path)


def test_iou_and_one_to_one_confidence_order():
    assert fa.iou([0,0,10,10],[0,0,10,10])==1
    assert fa.iou([0,0,10,10],[20,20,30,30])==0
    assert fa.iou([0,0,10,10],[0,0,5,10])==.5
    match=fa.match_boxes([box()],[box(conf=.6),box(conf=.9)])
    assert match['matches'][0][0]==1 and match['extra']==[0]
    assert fa.match_boxes([box()],[box(b=(0,0,5,10),conf=.9)])['matches']


def test_false_negative_false_positive_and_confusion():
    missed=fa.analyze_image('val.jpg',[box()],[],100,100)
    assert missed['events'][0]['error_type']=='FALSE_NEGATIVE'
    extra=fa.analyze_image('val.jpg',[],[box(conf=.8)],100,100)
    assert extra['events'][0]['error_type']=='FALSE_POSITIVE'
    confused=fa.analyze_image('val.jpg',[box()],[box(c=0,conf=.8)],100,100)
    assert {r['error_type'] for r in confused['events']}=={'FALSE_NEGATIVE','FALSE_POSITIVE','CLASS_CONFUSION'}
    assert confused['per_class']['1']['false_negative_count']==1
    assert confused['per_class']['0']['false_positive_count']==1
    # A valid same-class prediction takes precedence over a higher-confidence wrong class.
    correct=fa.analyze_image('val.jpg',[box()],[box(c=0,conf=.9),box(conf=.8)],100,100)
    assert correct['per_class']['1']['true_positive_count']==1
    assert not any(e['error_type']=='CLASS_CONFUSION' for e in correct['events'])


def test_tiny_low_confidence_and_aggregate():
    case=fa.analyze_image('val.jpg',[box(b=(0,0,1,1))],[box(b=(0,0,1,1),conf=.2)],100,100)
    assert {e['error_type'] for e in case['events']}=={'FALSE_NEGATIVE','LOW_CONFIDENCE','TINY_OBJECT_FAILURE'}
    summary=fa.aggregate([case]);wh=summary['without_helmet']
    assert wh['false_negative_groups']['tiny_bbox']==wh['false_negative_groups']['low_confidence']==1
    assert wh['manual_review_required'] and wh['false_negative_groups']['occlusion_candidate'] is None
    assert wh['recall']==0 and summary['confidence_distribution']['<0.25']==1
    correct=fa.analyze_image('val.jpg',[box()],[box(conf=.4)],100,100)
    assert correct['per_class']['1']['false_negative_count']==0
    assert [e['error_type'] for e in correct['events']]==['LOW_CONFIDENCE']


def test_low_prediction_not_reused_for_two_gt():
    case=fa.analyze_image('val.jpg',[box(),box()],[box(conf=.2)],100,100)
    assert len(case['low_confidence_missed_gt'])==1
    assert sum(e['error_type']=='MULTIPLE_OBJECT_SCENE' for e in case['events'])==1


def test_render_copies_and_report(tmp_path):
    image=tmp_path/'val.jpg';Image.new('RGB',(100,100),'white').save(image);before=image.read_bytes()
    case=fa.analyze_image('val.jpg',[box()],[box(c=0,conf=.8)],100,100)
    fa.render_case(case,tmp_path,tmp_path/'copies/val.png')
    assert image.read_bytes()==before
    with Image.open(tmp_path/'copies/val.png') as result:assert result.width==220
    summary=fa.aggregate([case]);metrics=mt.serialize_metrics(fake_metrics())
    metrics.update(model='best.pt',checkpoint_sha256='hash',dataset_fingerprint='frozen',validation_images=1,training_comparison={})
    training=dict(best_epoch=36,checkpoint_matches_best_csv=True,epochs_completed=56,epochs_requested=100,
                  early_stopping_consistent=True,patience=20,training_loop_seconds=5280,pipeline_duration_seconds=5338,trends={})
    fa.write_report(tmp_path,metrics,training,summary)
    text=(tmp_path/'baseline_evaluation_report.md').read_text(encoding='utf-8')
    assert 'OBSERVED PROBLEM' in text and 'PROPOSED EXPERIMENT' in text
    assert 'LOCKED / NOT USED' in text and 'Without Helmet' in text


def test_training_summary_uses_fitness_and_checkpoint(tmp_path):
    (tmp_path/'results.csv').write_text('epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,val/box_loss,val/cls_loss,val/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n1,20,2,2,2,2,2,2,.7,.7,.8,.5\n2,40,1,1,1,2,2,2,.7,.7,.9,.4\n')
    exp=dict(epochs_requested=10,training_status='PASS',resolved_training_arguments={'patience':1},duration_seconds=45)
    result=mt.training_summary(tmp_path,exp,lambda values:values[-1],checkpoint_metrics={'fitness':.5})
    assert result['best_epoch']==1 and result['epochs_completed']==2
    assert result['checkpoint_matches_best_csv'] and result['early_stopping_consistent']
    assert result['trends']['train/box_loss']['last']==1


def test_sample_selection_unique_mixed_cases():
    cases=[fa.analyze_image(f'{i}.jpg',[box(c=i%2)],[] if i%3 else [box(c=i%2,conf=.9)],100,100) for i in range(30)]
    samples=fa.select_samples(cases)
    assert len(samples)==len({c['image'] for c in samples})==20
    assert any(c['matched_gt'] for c in samples) and any(c['missed_gt'] for c in samples)


def test_evaluation_calls_val_only_and_preserves_training(tmp_path,monkeypatch):
    out=tmp_path/'results/evaluation/A_baseline_val';train=tmp_path/'results/experiments/A_baseline'
    (train/'weights').mkdir(parents=True);checkpoint=train/'weights/best.pt';checkpoint.write_bytes(b'fake')
    training_results='epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,val/box_loss,val/cls_loss,val/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n1,20,1,1,1,1,1,1,.75,.65,.7,.4\n'
    (train/'results.csv').write_text(training_results)
    snapshot=dict(data=dict(path='curated',train='train/images',val='valid/images',test='test/images',nc=2,names=EXPECTED_CLASSES),curated_hash='frozen')
    exp=dict(epochs_requested=100,training_status='PASS',resolved_training_arguments={'patience':20},duration_seconds=25,
             validation_metrics=dict(zip(mt.FRAMEWORK_KEYS,[.75,.65,.7,.4])))
    state=dict(root=tmp_path,checkpoint=checkpoint,snapshot=snapshot,experiment=exp,output=out,images=[tmp_path/'valid/a.jpg'],environment={'device':'cpu'})
    monkeypatch.setattr(ev,'preflight',lambda *args:state)
    monkeypatch.setattr(ev,'verify_unchanged',lambda *_:dict(raw='PASS',curated='PASS',training_artifacts='PASS'))
    monkeypatch.setattr(ev,'protection',lambda _:nullcontext())
    calls=[]
    class Fake:
        names=EXPECTED_CLASSES;ckpt={'train_metrics':{'fitness':.4}}
        def __init__(self,path):assert path==str(checkpoint)
        def val(self,**kwargs):
            import yaml
            calls.append(kwargs);assert kwargs['split']=='val'
            assert kwargs['quantize']==32
            assert 'test' not in yaml.safe_load(Path(kwargs['data']).read_text())
            for name in ['BoxPR_curve.png','BoxF1_curve.png','BoxP_curve.png','BoxR_curve.png','confusion_matrix.png','confusion_matrix_normalized.png']:
                (out/name).write_bytes(b'framework-plot')
            return fake_metrics()
        def train(self,**kwargs):raise AssertionError('No training permitted')
    ev.evaluate(tmp_path,model_factory=Fake)
    assert len(calls)==1 and (train/'results.csv').read_text()==training_results
    assert (out/'PR_curve.png').read_bytes()==(out/'BoxPR_curve.png').read_bytes()
    assert json.loads((out/'evaluation_status.json').read_text())['status']=='PASS'


def test_confused_fp_area_refers_to_prediction():
    case=fa.analyze_image('val.jpg',[box()],[box(c=0,b=(0,0,20,10),conf=.8)],100,100)
    events={e['error_type']:e for e in case['events']}
    assert events['FALSE_POSITIVE']['bbox_area']==200
    assert events['FALSE_NEGATIVE']['bbox_area']==100
