from pathlib import Path
from unittest.mock import Mock

import pytest

from smart_helmet.dataset.audit import write_csv
from smart_helmet.dataset.cleaning_review import read_csv
from smart_helmet.dataset.curation_policy import (SPLIT_PRIORITY, direction_status, analyze_policy,
    test_exclusion_rows as make_test_rows, policy_html)
from smart_helmet.dataset.review_interactive import DecisionSession, review_cases
from test_review_interactive import make_review


@pytest.mark.parametrize('excluded,kept,expected', [
    ('train','valid','OK'),('train','test','OK'),('valid','test','OK'),
    ('valid','train','POLICY_MISMATCH'),('test','valid','POLICY_MISMATCH'),('test','train','POLICY_MISMATCH'),
])
def test_split_policy(excluded,kept,expected):
    assert SPLIT_PRIORITY=={'test':3,'valid':2,'train':1}
    assert direction_status(excluded,kept)==expected


def policy_fixture(root):
    p=root/'results/dataset_cleaning';p.mkdir(parents=True)
    queue=[];decisions=[]
    for i in range(3):
        queue.append(dict(review_id=f'case_{i}',priority='P0',image_a='test/images/a.png',image_b=f'train/images/b{i}.png',
                          split_a='test',split_b='train',phash_distance='0',source_group_a='a',source_group_b='b',
                          annotation_count_a='1',annotation_count_b='1',classes_a='With Helmet',classes_b='With Helmet'))
        decisions.append(dict(review_id=f'case_{i}',decision='KEEP_B_EXCLUDE_A',notes=''))
    write_csv(p/'review_queue.csv',queue,list(queue[0]))
    write_csv(p/'review_decisions.csv',decisions,list(decisions[0]))
    manifest=[dict(original_image='test/images/a.png',action='EXCLUDE')]+[dict(original_image=f'train/images/b{i}.png',action='KEEP') for i in range(3)]
    write_csv(p/'curated_v1_manifest.csv',manifest,list(manifest[0]))
    exclusions=[dict(image='test/images/a.png',original_split='test',review_id='case_0|case_1|case_2',decision='KEEP_B_EXCLUDE_A|KEEP_B_EXCLUDE_A|KEEP_B_EXCLUDE_A')]
    write_csv(p/'curated_v1_exclusions.csv',exclusions,list(exclusions[0]))
    return p


def test_unique_test_exclusions_and_pair_evidence(tmp_path):
    p=policy_fixture(tmp_path);before={f.name:f.read_bytes() for f in p.iterdir()}
    rows,summary=analyze_policy(tmp_path)
    assert len(rows)==3 and summary['unique_exclusions']==1
    assert summary['policy_mismatch']==summary['needs_review']==1
    assert summary['status']=='REVIEW_REQUIRED'
    special=make_test_rows(rows)
    assert len(special)==3 and len({r['test_image'] for r in special})==1
    assert all(r['policy_status']=='NEEDS_REVIEW' and r['direction_status']=='POLICY_MISMATCH' for r in special)
    html=policy_html(rows,summary)
    assert 'TEST EXCLUSIONS' in html and 'POLICY MISMATCHES' in html and 'ALL EXCLUSIONS' in html
    assert len(html.splitlines())>30 and '--review-id case_0' in html
    assert before=={f.name:f.read_bytes() for f in p.iterdir()}


def test_changed_current_decision_needs_rebuild_review_only(tmp_path):
    p=policy_fixture(tmp_path);decisions=read_csv(p/'review_decisions.csv');decisions[0]['decision']='KEEP_BOTH'
    write_csv(p/'review_decisions.csv',decisions,list(decisions[0]))
    rows,_=analyze_policy(tmp_path)
    assert rows[0]['policy_status']=='NEEDS_REVIEW'
    assert 'differs' in rows[0]['explanation']


def test_revisit_completed_id_retains_atomic_backup_and_notes(tmp_path,monkeypatch):
    make_review(tmp_path)
    initial=DecisionSession(tmp_path);initial.save('P0_b','KEEP_BOTH','original human note')
    before=(tmp_path/'review_decisions.csv').read_bytes()
    calls=[];replace=__import__('os').replace
    def watched(source,destination):
        calls.append((source,destination));replace(source,destination)
    monkeypatch.setattr('smart_helmet.dataset.review_interactive.os.replace',watched)
    choices=iter(['2','']);log=[];opener=Mock()
    review_cases(tmp_path,review_id='P0_b',pending_only=True,open_image=True,
                 input_fn=lambda _:next(choices),output=log.append,image_opener=opener)
    rows=read_csv(tmp_path/'review_decisions.csv')
    assert rows[0]['decision']=='KEEP_A_EXCLUDE_B' and rows[0]['notes']=='original human note'
    assert len(calls)==1
    assert any(p.read_bytes()==before for p in (tmp_path/'backups').glob('*.csv'))
    assert any('Current decision: KEEP_BOTH' in s for s in log)
    assert any('P0 REVIEW 1 / 1' in s for s in log)
    assert all(r['decision']=='PENDING' for r in rows[1:])
    opener.assert_called_once()


def test_unknown_review_id_cannot_write(tmp_path):
    make_review(tmp_path);before=(tmp_path/'review_decisions.csv').read_bytes()
    with pytest.raises(ValueError,match='Unknown P0'):
        review_cases(tmp_path,review_id='missing',input_fn=lambda _:pytest.fail('must not prompt'))
    assert (tmp_path/'review_decisions.csv').read_bytes()==before
