"""Synthetic TRAIN difficulty tests; never execute model inference or training."""
from copy import deepcopy
from unittest.mock import Mock
import pytest
from smart_helmet.evaluation import hard_mining as h


def gt(cls=1,bbox=(0,0,10,10)):
    return dict(class_id=cls,bbox=list(bbox))


def pred(cls=1,conf=.8,bbox=(0,0,10,10)):
    return dict(gt(cls,bbox),confidence=conf)


def case(gg=None,pp=None):
    return dict(image='data/curated/v1/train/images/a.jpg',ground_truth=gg or [gt()],predictions=pp or [])


@pytest.mark.parametrize('split',['val','valid','test','TEST','../train'])
def test_train_only_before_io(tmp_path,monkeypatch,split):
    spy=Mock();monkeypatch.setattr(h,'records',spy)
    with pytest.raises(ValueError,match='TRAIN ONLY'):h.run(tmp_path,split)
    spy.assert_not_called()


def test_lock_guard():
    with pytest.raises(ValueError,match='LOCKED'):h.guard(test_status='UNLOCKED')


@pytest.mark.parametrize('predictions,category',[
    ([pred()], 'EASY_CORRECT'),([pred(conf=.25)],'EASY_CORRECT'),
    ([pred(conf=.001)],'LOW_CONFIDENCE_CORRECT'),([pred(cls=0)],'CLASS_CONFUSION'),
    ([], 'MISSED_NO_CORRECT_CLASS_PROPOSAL'),
    ([pred(bbox=(0,0,25,10))],'LOCALIZATION_FAILURE'),
    ([pred(bbox=(100,100,110,110))],'MISSED_NO_CORRECT_CLASS_PROPOSAL')])
def test_primary_categories(predictions,category):
    assert h.classify(case(pp=predictions))[0]['category']==category


def test_one_to_one_competition_and_low_precedence():
    oo=h.classify(case([gt(),gt()],[pred()]))
    assert [r['category'] for r in oo]==['EASY_CORRECT','MATCH_COMPETITION']
    assert len({r['prediction_index'] for r in oo if r['prediction_index'] is not None})==1
    r=h.classify(case(pp=[pred(cls=0),pred(conf=.1)]))[0]
    assert r['category']=='LOW_CONFIDENCE_CORRECT' and r['active_wrong_class_overlap']


def test_duplicate_context_and_tier():
    c=case(pp=[pred(),pred(conf=.7)])
    oo=h.classify(c)
    assert oo[0]['duplicate_context']=='DUPLICATE_PREDICTION_CONTEXT'
    assert h.image_row(c,oo)['tier']==h.TIERS[1]


def test_tiers_target_aggregation_and_stable_ids():
    cc=case([gt(0),gt(1,bbox=(20,0,30,10)),gt(1,bbox=(40,0,50,10))],[pred(0)])
    oo,images,s=h.analyze([cc])
    assert images[0]['tier']==h.TIERS[3]
    assert images[0]['hard_without_helmet_count']==2
    assert images[0]['crowded'] and images[0]['multi_class']
    assert h.analyze([deepcopy(cc)])==(oo,images,s)
    assert len({r['object_id'] for r in oo})==3
    assert h.stable_id(cc['image'],0)==oo[0]['object_id']


def test_train_paths_and_label_escape(tmp_path):
    for value in ['data/curated/v1/valid/images/a.jpg','data/curated/v1/test/images/a.jpg','../a.jpg']:
        with pytest.raises(ValueError,match='leakage'):h.train_path(tmp_path,value)
    p=tmp_path/'data/curated/v1/train/images/a.jpg';p.parent.mkdir(parents=True);p.write_bytes(b'x')
    assert h.train_path(tmp_path,p)==p


def test_matching_iou_boundaries():
    assert h.classify(case(pp=[pred(bbox=(0,0,20,10))]))[0]['category']=='EASY_CORRECT'
    assert h.classify(case(pp=[pred(bbox=(0,0,100,10))]))[0]['category']=='LOCALIZATION_FAILURE'
    assert h.classify(case(pp=[pred(bbox=(0,0,101,10))]))[0]['category']=='MISSED_NO_CORRECT_CLASS_PROPOSAL'


def test_explicit_inference_chunks_prevent_full_list_batch():
    model=Mock();model.predict.side_effect=lambda source,**kwargs:iter(source)
    assert list(h.predict_batches(model,list(range(10)),{'batch':4}))==list(map(str,range(10)))
    assert [len(call.kwargs['source']) for call in model.predict.call_args_list]==[4,4,2]


def test_stable_case_order_and_review_sample():
    a=case(pp=[pred(conf=.1)]); b=case(pp=[]); b['image']='data/curated/v1/train/images/b.jpg'
    assert h.analyze([b,a])==h.analyze([a,b])
    objects,images,_=h.analyze([a,b])
    sample=h.review_sample(images,objects)
    assert {r['image'] for r in sample}=={a['image'],b['image']}
