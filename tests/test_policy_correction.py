from pathlib import Path
import pytest
from smart_helmet.dataset import policy_correction as pc
from smart_helmet.dataset import review_interactive as ui
from smart_helmet.dataset.audit import write_csv


def fixture(tmp_path, reverse=False):
    queue = [dict(review_id=f'P0_{i}', priority='P0', image_a='test/images/e.jpg',
                  image_b=f'train/images/{i}.jpg', split_a='test', split_b='train',
                  phash_distance=str(i), source_group_a='eval', source_group_b='train',
                  annotation_count_a='2', annotation_count_b='2', classes_a='helmet', classes_b='helmet') for i in range(3)]
    if reverse:
        for r in queue:
            for key in ('image', 'split'):
                r[key+'_a'], r[key+'_b'] = r[key+'_b'], r[key+'_a']
    decisions = [dict(review_id=r['review_id'], decision='KEEP_A_EXCLUDE_B' if reverse else 'KEEP_B_EXCLUDE_A', notes='') for r in queue]
    write_csv(tmp_path/'review_queue.csv', queue, list(queue[0]))
    write_csv(tmp_path/'review_decisions.csv', decisions, ui.DECISION_FIELDS)
    sanity = [dict(review_id=r['review_id'], excluded_image='test/images/e.jpg', retained_pair_image=f'train/images/{i}.jpg') for i,r in enumerate(queue)]
    tests = [dict(review_id=r['review_id'], test_image=r['excluded_image'], paired_image=r['retained_pair_image']) for r in sanity]
    write_csv(tmp_path/'pretraining_curation_sanity.csv', sanity, list(sanity[0]))
    write_csv(tmp_path/'test_exclusion_review.csv', tests, list(tests[0]))
    return queue, decisions


@pytest.mark.parametrize('reverse,expected', [(False, 'KEEP_A_EXCLUDE_B'), (True, 'KEEP_B_EXCLUDE_A')])
def test_grouping_orientation_filter_report(tmp_path, reverse, expected):
    q,d = fixture(tmp_path, reverse)
    assert expected in pc.recommendation(q[0])
    assert 'KEEP_BOTH' in pc.recommendation(q[0])
    groups = pc.mismatch_groups(q,d)
    assert len(groups) == 1 and len(groups[0]['cases']) == 3
    path,_ = pc.generate_report(tmp_path, tmp_path)
    html = path.read_text(encoding='utf-8')
    assert html.count('<img ') == 4
    for r in q: assert r['review_id'] in html
    for r in d: r['decision'] = 'KEEP_BOTH'
    assert pc.mismatch_groups(q,d) == []
    state = pc.analyze(q,d,3)
    assert not state['conflicts'] and state['status'] == 'PASS'


def test_conflicting_and_nonconflicting_multiple_pairs(tmp_path):
    q,d = fixture(tmp_path)
    assert not pc.analyze(q,d)['conflicts']
    d[0]['decision'] = 'KEEP_A_EXCLUDE_B'
    conflict = pc.analyze(q,d)['conflicts']['test/images/e.jpg']
    assert conflict == {'KEEP':['P0_0'], 'EXCLUDE':['P0_1','P0_2']}
    for r in d: r['decision'] = 'KEEP_A_EXCLUDE_B'
    assert pc.analyze(q,d,3)['status'] == 'PASS'


def test_group_skip_and_warning_return_preserve_bytes(tmp_path):
    fixture(tmp_path); path=tmp_path/'review_decisions.csv'; before=path.read_bytes()
    log=[]; answers=iter(['2','','s','s','s',''])
    def respond(prompt):
        if 'P0_0 choice' in prompt:
            assert all(any(f'review_id: P0_{i}' in line for line in log) for i in range(3))
        return next(answers)
    ui.review_policy_groups(tmp_path, input_fn=respond, output=log.append)
    assert any('DECISION CONSISTENCY WARNING' in line for line in log)
    assert path.read_bytes() == before and not (tmp_path/'backups').exists()


def test_group_explicit_saves_backup_and_notes(tmp_path):
    fixture(tmp_path); before=(tmp_path/'review_decisions.csv').read_bytes()
    answers=iter(['2','SAVE','ACCEPT','2','SAVE','','2','',''])
    ui.review_policy_groups(tmp_path,input_fn=lambda _:next(answers),output=lambda _:None)
    backups=list((tmp_path/'backups').glob('*.csv'))
    assert len(backups)==1 and backups[0].read_bytes()==before
    q,d=pc.load_review(tmp_path)
    assert pc.analyze(q,d,3)['status']=='PASS'
    assert d[0]['notes']=='Confirmed same scene/image; preserve TEST over TRAIN.'
    assert d[1]['notes']==''
    assert not list(tmp_path.glob('.review_decisions_*.tmp'))


def test_stale_reports_do_not_reintroduce_corrected_groups(tmp_path):
    q,d=fixture(tmp_path)
    for r in d:r['decision']='KEEP_BOTH'
    write_csv(tmp_path/'review_decisions.csv',d,ui.DECISION_FIELDS)
    assert pc.mismatch_groups(*pc.load_review(tmp_path))==[]


def test_invalid_evidence_and_decisions(tmp_path):
    fixture(tmp_path)
    (tmp_path/'test_exclusion_review.csv').write_text('review_id,test_image,paired_image\nunknown,x,y\n')
    with pytest.raises(ValueError,match='relationship'):pc.load_review(tmp_path)


def test_noop_save_has_no_backup(tmp_path):
    fixture(tmp_path); session=ui.DecisionSession(tmp_path)
    before=session.path.read_bytes()
    session.save('P0_0','KEEP_B_EXCLUDE_A')
    assert session.path.read_bytes()==before and session.backup is None


def test_duplicate_decision_columns_rejected(tmp_path):
    fixture(tmp_path)
    (tmp_path/'review_decisions.csv').write_text('review_id,decision,notes,notes\nP0_0,KEEP_BOTH,,\n')
    with pytest.raises(ValueError,match='columns'):ui.DecisionSession(tmp_path)
