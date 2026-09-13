import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest
import yaml

from smart_helmet.dataset.audit import run_audit, fingerprint, write_csv
from smart_helmet.dataset.cleaning_review import read_csv
from smart_helmet.dataset.curation import (action_map, build_plan, apply_cleaning, training_readiness,
    curated_fingerprint, verify_curated_fingerprint, classify_curated_pairs, safe_remove_owned)
from smart_helmet.foundation import load_detection_config, load_data_config


def map_fixture(decision):
    records={'train/images/a.png':{'split':'train'},'valid/images/b.png':{'split':'valid'},'test/images/c.png':{'split':'test'}}
    queue=[dict(review_id='p0',priority='P0',image_a='train/images/a.png',image_b='valid/images/b.png')]
    decisions=[dict(review_id='p0',decision=decision,notes='human')]
    return records,queue,decisions


@pytest.mark.parametrize('decision,actions', [
    ('KEEP_BOTH',['KEEP','KEEP','KEEP']),('KEEP_A_EXCLUDE_B',['KEEP','EXCLUDE','KEEP']),
    ('KEEP_B_EXCLUDE_A',['EXCLUDE','KEEP','KEEP']),
])
def test_action_map(decision,actions):
    records,queue,decisions=map_fixture(decision)
    result,_,_=action_map(records,queue,decisions)
    assert [result[n]['action'] for n in records]==actions


@pytest.mark.parametrize('decision',['PENDING','REVIEW_MORE'])
def test_unfinished_aborts(decision):
    with pytest.raises(ValueError,match='P0_INCOMPLETE'):action_map(*map_fixture(decision))


def test_multiple_pairs_and_conflict():
    records,queue,decisions=map_fixture('KEEP_B_EXCLUDE_A')
    queue.append(dict(review_id='other',priority='P0',image_a='train/images/a.png',image_b='test/images/c.png'))
    decisions.append(dict(review_id='other',decision='KEEP_B_EXCLUDE_A',notes=''))
    result,_,_=action_map(records,queue,decisions)
    assert result['train/images/a.png']['action']=='EXCLUDE'
    assert len(result['train/images/a.png']['evidence'])==2
    decisions[1]['decision']='KEEP_BOTH'
    with pytest.raises(ValueError,match='DECISION_CONFLICT'):action_map(records,queue,decisions)


def fixture_dataset(root,decision='KEEP_B_EXCLUDE_A'):
    (root/'configs').mkdir()
    cfg=dict(path='.',train='train/images',val='valid/images',test='test/images',nc=2,names={0:'With Helmet',1:'Without Helmet'})
    config=root/'configs/data.local.yaml';config.write_text(yaml.safe_dump(cfg))
    for n in ('data.yaml','README.dataset.txt','README.roboflow.txt'):(root/n).write_text('original')
    rng=np.random.default_rng(9)
    shared=Image.fromarray(rng.integers(0,256,(80,100,3),dtype=np.uint8))
    for split,names in [('train',['a','independent_train']),('valid',['b']),('test',['c'])]:
        (root/split/'images').mkdir(parents=True);(root/split/'labels').mkdir()
        for name in names:
            im=shared if name in ['a','b'] else Image.fromarray(rng.integers(0,256,(80,100,3),dtype=np.uint8))
            im.save(root/split/'images'/f'{name}.png')
            (root/split/'labels'/f'{name}.txt').write_text('0 .5 .5 .2 .2\n1 .2 .2 .02 .02\n')
    out=root/'results/dataset_audit';run_audit(root,config,out,visuals=False)
    records={r['image_path']:r for r in read_csv(out/'dataset_manifest.csv')}
    queue=[]
    for i,p in enumerate(read_csv(out/'near_duplicate_leakage.csv')):
        queue.append(dict(review_id=f'p0_{i}',priority='P0',image_a=p['image_a'],image_b=p['image_b'],split_a=p['split_a'],split_b=p['split_b'],
                          sha256_a=records[p['image_a']]['sha256'],sha256_b=records[p['image_b']]['sha256']))
    assert len(queue)==1
    qfields=list(queue[0]);queue.append({k:('p1' if k=='review_id' else 'P1' if k=='priority' else '') for k in qfields})
    decisions=[dict(review_id=r['review_id'],decision=decision if r['priority']=='P0' else 'PENDING',notes='human reviewed' if r['priority']=='P0' else '') for r in queue]
    cleaning=root/'results/dataset_cleaning';cleaning.mkdir()
    write_csv(cleaning/'review_queue.csv',queue,qfields)
    write_csv(cleaning/'review_decisions.csv',decisions,['review_id','decision','notes'])
    return out,cleaning


def tree_bytes(root):return {p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_dry_run_no_changes(tmp_path):
    fixture_dataset(tmp_path);before=tree_bytes(tmp_path)
    plan=build_plan(tmp_path)
    assert plan['summary']['cleaning']['unique_images_excluded']==1
    assert plan['summary']['curated']['total_images']==3
    assert tree_bytes(tmp_path)==before
    assert not (tmp_path/'data').exists()


def test_build_manifest_pairing_config_fingerprint_and_readiness(tmp_path):
    _,results=fixture_dataset(tmp_path);raw=fingerprint(tmp_path)
    summary=apply_cleaning(tmp_path)
    assert summary['status']=='PASS' and summary['curated']['total_images']==3
    target=tmp_path/'data/curated/v1'
    assert not (target/'train/images/a.png').exists()
    assert not (target/'train/labels/a.txt').exists()
    assert (target/'valid/images/b.png').is_file() and (target/'valid/labels/b.txt').is_file()
    assert fingerprint(tmp_path)==raw
    manifest=read_csv(results/'curated_v1_manifest.csv')
    assert len(manifest)==4
    excluded=[r for r in manifest if r['action']=='EXCLUDE']
    assert len(excluded)==1 and excluded[0]['curated_image']==excluded[0]['curated_label']==''
    assert read_csv(results/'curated_v1_exclusions.csv')[0]['human_reviewed']=='TRUE'
    for row in manifest:
        if row['action']=='KEEP':
            assert (tmp_path/row['original_image']).read_bytes()==(tmp_path/row['curated_image']).read_bytes()
            assert (tmp_path/row['original_label']).read_bytes()==(tmp_path/row['curated_label']).read_bytes()
            assert not (tmp_path/row['original_image']).samefile(tmp_path/row['curated_image'])
    cfg=tmp_path/'configs/data.curated.v1.yaml';data=load_detection_config(cfg,tmp_path)
    assert Path(data['path'])==target and data['names']=={0:'With Helmet',1:'Without Helmet'}
    with pytest.raises(ValueError):load_data_config(cfg,tmp_path)
    assert len(verify_curated_fingerprint(tmp_path))==7
    assert not training_readiness(tmp_path)['ready_for_baseline_training']  # No re-audit yet.
    audit=run_audit(tmp_path,cfg,tmp_path/'results/dataset_audit/curated_v1',visuals=False)
    assert audit['dataset']['total_images']==3
    assert audit['cross_split_review']['unreviewed']==0
    state=training_readiness(tmp_path)
    assert state['ready_for_baseline_training'],state
    assert state['pending_reviews']['P1']==1
    (target/'test/labels/c.txt').write_text('0 .1 .1 .1 .1')
    assert not training_readiness(tmp_path)['ready_for_baseline_training']
    assert fingerprint(tmp_path)==raw


def test_reviewed_keep_both_is_not_unreviewed_blocker(tmp_path):
    fixture_dataset(tmp_path,'KEEP_BOTH');apply_cleaning(tmp_path)
    out=tmp_path/'results/dataset_audit/curated_v1'
    report=run_audit(tmp_path,tmp_path/'configs/data.curated.v1.yaml',out,visuals=False)
    assert report['cross_split_review']==dict(reviewed_keep_both=1,unreviewed=0)
    assert read_csv(out/'cross_split_review.csv')[0]['review_status']=='REVIEWED_KEEP_BOTH_CANDIDATE'
    state=training_readiness(tmp_path);assert state['ready_for_baseline_training'],state
    hypothetical=[dict(image_a='data/curated/v1/train/images/independent_train.png',image_b='data/curated/v1/test/images/c.png',split_a='train',split_b='test',distance=2)]
    assert classify_curated_pairs(tmp_path,hypothetical)[0]['review_status']=='UNREVIEWED_CROSS_SPLIT_CANDIDATE'


def test_label_copy_failure_never_publishes_partial(tmp_path,monkeypatch):
    fixture_dataset(tmp_path);raw=fingerprint(tmp_path);copy=shutil.copy2
    def fail_label(source,dest):
        if Path(source).suffix=='.txt':raise OSError('label copy failed')
        return copy(source,dest)
    monkeypatch.setattr(shutil,'copy2',fail_label)
    with pytest.raises(OSError):apply_cleaning(tmp_path)
    assert not (tmp_path/'data/curated/v1').exists()
    assert not (tmp_path/'results/dataset_cleaning/curated_v1_summary.json').exists()
    assert fingerprint(tmp_path)==raw
    assert not list((tmp_path/'data/curated').glob('.v1-stage-*'))


def test_existing_build_requires_explicit_rebuild(tmp_path):
    fixture_dataset(tmp_path);apply_cleaning(tmp_path);raw=fingerprint(tmp_path)
    with pytest.raises(ValueError,match='--rebuild'):apply_cleaning(tmp_path)
    apply_cleaning(tmp_path,rebuild=True)
    assert fingerprint(tmp_path)==raw
    assert verify_curated_fingerprint(tmp_path)==curated_fingerprint(tmp_path)
    with pytest.raises(ValueError):safe_remove_owned(tmp_path/'train',tmp_path/'data/curated','v1')
    assert (tmp_path/'train').is_dir()


def test_custom_audit_cannot_overwrite_raw_evidence(tmp_path):
    audit,_=fixture_dataset(tmp_path);apply_cleaning(tmp_path)
    before=(audit/'dataset_fingerprint.json').read_bytes()
    with pytest.raises(ValueError,match='separate --output'):
        run_audit(tmp_path,tmp_path/'configs/data.curated.v1.yaml',audit,visuals=False)
    assert (audit/'dataset_fingerprint.json').read_bytes()==before
