import csv
from html.parser import HTMLParser
import json

import numpy as np
from PIL import Image
import pytest
import yaml

from smart_helmet.dataset.audit import run_audit
from smart_helmet.dataset.cleaning_review import (
    PRIORITIES, ALLOWED, build_queue, merge_decisions, policy_suggestion,
    validate_decisions, prepare_review, read_csv, numeric_boxes,
)


def small_queue():
    return [dict(review_id=f'{priority}_case',priority=priority) for priority in ALLOWED]


@pytest.mark.parametrize('a,b,expected', [
    ('train','valid','KEEP_B_EXCLUDE_A'),('valid','train','KEEP_A_EXCLUDE_B'),
    ('train','test','KEEP_B_EXCLUDE_A'),('test','train','KEEP_A_EXCLUDE_B'),
    ('valid','test','KEEP_B_EXCLUDE_A'),('test','valid','KEEP_A_EXCLUDE_B'),
])
def test_policy(a,b,expected):
    assert policy_suggestion(a,b)==expected


def test_invalid_pair_policy():
    with pytest.raises(ValueError):
        policy_suggestion('train','train')


def test_priorities_and_pending_gate():
    assert PRIORITIES=={'CROSS_SPLIT_NEAR_DUPLICATE':'P0','TINY_BBOX':'P1',
                        'UNUSUAL_RESOLUTION':'P2','SOURCE_VARIANT_REVIEW':'P3'}
    queue=small_queue();decisions=merge_decisions(queue,[])
    state=validate_decisions(queue,decisions)
    assert state['valid'] and not state['ready_to_apply_cleaning']
    assert state['blocking_reviews_pending']==1
    decisions[0]['decision']='REVIEW_MORE'
    state=validate_decisions(queue,decisions)
    assert state['priorities']['P0']['review_more']==1
    assert not state['ready_to_apply_cleaning']
    decisions[0]['decision']='KEEP_BOTH'
    state=validate_decisions(queue,decisions)
    assert state['ready_to_apply_cleaning']
    assert state['priorities']['P1']['pending']==1


@pytest.mark.parametrize('priority,bad', [('P0','KEEP'),('P1','KEEP_ALL'),('P2','CHECK_BBOX'),('P3','DELETE'),('P0','')])
def test_invalid_decision_rejected(priority,bad):
    queue=small_queue();decisions=merge_decisions(queue,[])
    next(r for r in decisions if r['review_id']==f'{priority}_case')['decision']=bad
    assert not validate_decisions(queue,decisions)['valid']
    with pytest.raises(ValueError):
        merge_decisions(queue,decisions)


def test_duplicate_unknown_and_missing_ids():
    queue=small_queue();decisions=merge_decisions(queue,[])
    assert not validate_decisions(queue,decisions+[decisions[0]])['valid']
    assert not validate_decisions(queue,decisions[:-1])['valid']
    decisions.append(dict(review_id='UNKNOWN',decision='PENDING',notes=''))
    assert not validate_decisions(queue,decisions)['valid']


def test_preserve_notes_and_append_new_case():
    queue=small_queue()
    existing=[dict(review_id='P0_case',decision='KEEP_BOTH',notes='Distinct event; reviewed, do not overwrite.')]
    merged=merge_decisions(queue,existing)
    assert merged[0]==existing[0]
    assert all(r['decision']=='PENDING' for r in merged[1:])


def make_audited_dataset(root):
    (root/'configs').mkdir()
    config=dict(path='.',train='train/images',val='valid/images',test='test/images',nc=2,
                names={0:'With Helmet',1:'Without Helmet'})
    (root/'configs/data.local.yaml').write_text(yaml.safe_dump(config))
    for name in ('data.yaml','README.dataset.txt','README.roboflow.txt'):
        (root/name).write_text('original')
    rng=np.random.default_rng(5)
    shared=Image.fromarray(rng.integers(0,256,(80,100,3),dtype=np.uint8))
    other=Image.fromarray(rng.integers(0,256,(80,100,3),dtype=np.uint8))
    for split,names in [('train',['source.rf.aa.png','source.rf.bb.png']),('valid',['evaluation.png']),('test',['independent.png'])]:
        (root/split/'images').mkdir(parents=True)
        (root/split/'labels').mkdir()
        for name in names:
            (other if split=='test' else shared).save(root/split/'images'/name)
            (root/split/'labels'/name.replace('.png','.txt')).write_text('0 .5 .5 .2 .2\n1 .8 .8 .02 .02\n')
    audit=root/'results/dataset_audit'
    run_audit(root,root/'configs/data.local.yaml',audit,visuals=False)
    return audit


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__();self.images=[];self.articles=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='img':self.images.append(attrs['src'])
        if tag=='article':self.articles.append(attrs['id'])


def test_integration_idempotent_decisions_and_html(tmp_path):
    audit=make_audited_dataset(tmp_path);out=tmp_path/'results/dataset_cleaning'
    summary=prepare_review(tmp_path,audit,out)
    queue=read_csv(out/'review_queue.csv')
    assert len(queue)==len({r['review_id'] for r in queue})
    assert summary['P0']['cross_split_pairs']==2
    assert summary['P1']['tiny_bbox_cases']==4
    assert summary['P2']['unusual_resolution_images']==4
    assert summary['P3']['sampled_groups']==1
    assert summary['immutability']['status']=='PASS'
    parser=ReportParser();parser.feed((out/'review_report/index.html').read_text())
    assert set(parser.articles)=={r['review_id'] for r in queue}
    assert len(parser.images)==len(queue)
    for src in parser.images:
        assert (out/'review_report'/src).is_file()
    for r in read_csv(out/'proposed_actions.csv'):
        assert r['requires_human_approval']=='true' and r['approved']==r['applied']=='false'
        if r['priority']=='P0':assert r['current_split']=='train'
    decision_file=out/'review_decisions.csv'
    decisions=read_csv(decision_file);decisions[0]['decision']='KEEP_BOTH';decisions[0]['notes']='Human note with comma, and Unicode: xem ảnh.'
    with decision_file.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['review_id','decision','notes']);writer.writeheader();writer.writerows(decisions)
    before=decision_file.read_bytes();queue_before=(out/'review_queue.csv').read_bytes()
    image_hashes={p.name:p.read_bytes() for p in (out/'review_images').rglob('*.png')}
    rerun=prepare_review(tmp_path,audit,out)
    assert decision_file.read_bytes()==before
    assert (out/'review_queue.csv').read_bytes()==queue_before
    assert image_hashes=={p.name:p.read_bytes() for p in (out/'review_images').rglob('*.png')}
    assert rerun['decision_status']['priorities']['P0']['completed']==1
    assert rerun['decision_status']['priorities']['P0']['pending']==1


def test_queue_deduplication_and_input_order(tmp_path):
    audit=make_audited_dataset(tmp_path)
    manifest=read_csv(audit/'dataset_manifest.csv');boxes=numeric_boxes(read_csv(audit/'bounding_boxes.csv'))
    pairs=read_csv(audit/'near_duplicate_leakage.csv');manual=read_csv(audit/'manual_review.csv')
    queue,*_=build_queue(manifest,boxes,pairs,manual)
    duplicate,*_=build_queue(list(reversed(manifest)),list(reversed(boxes)),pairs+pairs,manual+manual)
    assert queue==duplicate


def test_changed_dataset_fails_without_overwriting_decisions(tmp_path):
    audit=make_audited_dataset(tmp_path);out=tmp_path/'results/dataset_cleaning'
    out.mkdir();decision=out/'review_decisions.csv';decision.write_text('human content')
    (tmp_path/'train/labels/source.rf.aa.txt').write_text('changed')
    with pytest.raises(ValueError,match='fingerprint'):
        prepare_review(tmp_path,audit,out)
    assert decision.read_text()=='human content'
    assert not (out/'review_queue.csv').exists()


def test_output_guard(tmp_path):
    with pytest.raises(ValueError):
        prepare_review(tmp_path,tmp_path/'results/audit',tmp_path/'train/images')
