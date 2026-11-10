"""Sampling design tests operate on manifests only; no trainer integration."""
from copy import deepcopy
import csv
import json
import yaml
import pytest
from smart_helmet.training import hard_sampling_plan as p
from smart_helmet.evaluation import hard_mining as h


def rows(n=100):
    return [dict(image=f'data/curated/v1/train/images/{i:04d}.jpg',tier=h.TIERS[min(i//25,3)],
                 with_helmet_gt_count=1,without_helmet_gt_count=int(i%2==0)) for i in range(n)]


def test_weights_derived_from_tier_sizes_and_budget():
    rr=rows();w=p.choose_weights(rr)
    assert w[h.TIERS[0]]==1
    assert list(w.values())==sorted(w.values())
    assert max(w.values())<=2
    assert sum(w[r['tier']]-1 for r in rr)<=.10*len(rr)+1e-10
    rare=rows(1000)
    for r in rare:r['tier']=h.TIERS[0]
    rare[0]['tier']=h.TIERS[3]
    assert p.choose_weights(rare)[h.TIERS[3]]==2


def test_epoch_size_class_exposure_and_determinism():
    rr=rows();plan=p.build(rr)
    assert plan==p.build(deepcopy(rr))
    assert plan['effective_epoch_samples']==len(plan['simulated_epoch_zero_images'])==100
    assert sum(plan['expected_tier_draws'].values())==pytest.approx(100)
    assert sum(plan['expected_tier_fraction'].values())==pytest.approx(1)
    assert plan['expected_class_exposure']['With Helmet']==pytest.approx(100)
    assert plan['baseline_class_exposure']=={'With Helmet':100,'Without Helmet':50}
    assert plan['maximum_expected_draws_per_image']<=2
    assert 0<plan['expected_unique_images_per_epoch']<100
    assert plan['dominance_risk']=='PASS'


def test_dominance_detection_and_no_files_created(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    rr=rows()
    for r in rr:r['tier']=h.TIERS[0]
    rr[0]['tier']=h.TIERS[3]
    w={t:1 for t in h.TIERS};w[h.TIERS[3]]=1000
    plan=p.build(rr,w)
    assert plan['dominance_risk']=='FAIL'
    assert list(tmp_path.iterdir())==[]
    assert plan['no_disk_image_duplication'] and not plan['training_authorized']


def test_uniform_plan_and_empty_rejection():
    rr=rows()
    for r in rr:r['tier']=h.TIERS[0]
    plan=p.build(rr)
    assert set(plan['proposed_tier_weights'].values())=={1}
    assert plan['expected_unique_images_per_epoch']==pytest.approx(plan['uniform_replacement_expected_unique'])
    with pytest.raises(ValueError):p.build([])
    with pytest.raises(ValueError):p.build(rr+rr[:1])


def test_immutability_guard(tmp_path,monkeypatch):
    from smart_helmet.training import environment as e
    snapshot={'curated':{'a':'hash'},'raw':{'b':'hash'}}
    monkeypatch.setattr(e,'verify_curated_fingerprint',lambda root:{'a':'changed'})
    monkeypatch.setattr(e,'raw_snapshot',lambda root:{'b':'hash'})
    with pytest.raises(ValueError,match='Dataset changed'):e.check_unchanged(snapshot,tmp_path)
    monkeypatch.setattr(e,'verify_curated_fingerprint',lambda root:{'a':'hash'})
    assert e.check_unchanged(snapshot,tmp_path)=='PASS'


@pytest.fixture
def planning_project(tmp_path,monkeypatch):
    from smart_helmet.evaluation.train_human_review import ReviewStore
    rr=rows(1158)
    out=tmp_path/p.OUT;out.mkdir(parents=True)
    (tmp_path/p.PLAN).parent.mkdir(parents=True)
    for row in rr:
        image=tmp_path/row['image'];image.parent.mkdir(parents=True,exist_ok=True)
        image.write_bytes(b'synthetic TRAIN image')
    label=tmp_path/'data/curated/v1/train/labels/frozen.txt'
    label.parent.mkdir();label.write_text('1 0.5 0.5 0.2 0.2\n')
    summary=dict(manual_review_required=True,without_helmet_hard_images=100,
                 critical_no_proposal_images=10,status='REVIEW_REQUIRED')
    monkeypatch.setattr(p,'check',lambda root:summary)
    monkeypatch.setattr(p,'analyze',lambda cases:([],deepcopy(rr),summary))
    (out/'train_predictions.json').write_text('{"cases": []}')
    (out/'train_image_difficulty.csv').write_text('frozen TRAIN evidence')
    (out/'provenance.json').write_text(json.dumps(dict(prediction_sha256='frozen',dataset_fingerprint='dataset',raw_fingerprint='raw')))
    baseline=tmp_path/'results/experiments/A_baseline';baseline.mkdir(parents=True)
    args=dict(model='yolo11n.pt',imgsz=416,batch=4,workers=0,seed=42,epochs=100,patience=20,pretrained=True)
    (baseline/'experiment.json').write_text(json.dumps(dict(resolved_training_arguments=args,
        optimizer_class='AdamW',optimizer_initial_parameter_groups=[])))
    queue=[dict(review_id=str(i),image=r['image'],tier=r['tier'],automatic_reason='saved',
                gt_summary='[]',prediction_summary='[]',review_status='REVIEWED',
                human_decision='ANNOTATION_SUSPICIOUS' if i==29 else 'VALID_HARD_CASE',
                human_notes='original notes',reviewed_at='2026-10-04T00:00:00+00:00') for i,r in enumerate(rr[:30])]
    with (out/'manual_review_queue.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(queue[0]));writer.writeheader();writer.writerows(queue)
    p.prepare(tmp_path)
    return tmp_path,rr,ReviewStore(tmp_path)


def test_ordinary_overwrite_and_cli_default_protected(planning_project):
    root,_,_=planning_project
    before={s:s.read_bytes() for s in p.archive_sources(root)}
    with pytest.raises(ValueError,match='Preserve existing C plan; do not create tuned variants'):
        p.prepare(root)
    assert p.main([],root)==1
    assert all(s.read_bytes()==v for s,v in before.items())


@pytest.mark.parametrize('state',['PENDING','REVIEW_MORE'])
def test_incomplete_review_blocks_without_archiving(planning_project,state):
    root,_,store=planning_project
    if state=='REVIEW_MORE':store.confirm('0',state)
    else:
        data=store.path.read_text().replace('REVIEWED,VALID_HARD_CASE','PENDING,',1)
        store.path.write_text(data)
    before={s:s.read_bytes() for s in p.archive_sources(root)}
    with pytest.raises(ValueError,match='Human review gate failed'):
        p.prepare(root,replan_after_review=True)
    assert all(s.read_bytes()==v for s,v in before.items())
    assert all(not a.exists() for a in p.archive_sources(root).values())


def test_post_review_archives_once_reproduces_and_preserves_dataset(planning_project):
    root,rr,store=planning_project
    originals={s:s.read_bytes() for s in p.archive_sources(root)}
    dataset={f:f.read_bytes() for f in (root/'data').rglob('*') if f.is_file()}
    review=store.path.read_bytes()
    assert p.main(['--replan-after-review'],root)==0
    plan=p.verify(root)
    assert plan['plan_stage']=='post_human_review'
    assert plan['human_review_total']==30 and plan['valid_hard_case']==29
    assert plan['annotation_suspicious']==plan['suspicious_excluded_from_upweighting']==1
    assert plan['review_more']==plan['human_review_pending']==0
    suspicious=rr[29]['image']
    assert suspicious not in plan['eligible_hard_images']
    assert rr[25]['image'] in plan['eligible_hard_images']
    assert plan['image_sampling_weights'][suspicious]==1
    assert plan['image_sampling_probabilities'][suspicious]*1158<=1
    assert plan['effective_epoch_samples']==len(plan['simulated_epoch_zero_images'])==1158
    assert plan['seed']==42 and plan['valid_leakage']==plan['test_leakage']==0
    assert plan['ready_for_experiment_c_implementation'] and not plan['ready_for_experiment_c_training']
    assert plan['experiment_name']=='C_hard_example_sampling'
    for source,archive in p.archive_sources(root).items():assert archive.read_bytes()==originals[source]
    assert all(f.read_bytes()==v for f,v in dataset.items())
    assert store.path.read_bytes()==review
    assert not any(list(root.rglob(f'C{i}*')) for i in (1,2,3))
    with pytest.raises(ValueError,match='already consumed'):p.prepare(root,replan_after_review=True)
    assert p.verify(root)==plan


@pytest.mark.parametrize('same',[True,False])
def test_existing_archive_verified_never_overwritten(planning_project,same):
    root,_,_=planning_project
    source,archive=next(iter(p.archive_sources(root).items()))
    original=source.read_bytes() if same else b'ambiguous archive'
    archive.write_bytes(original)
    if same:p.prepare(root,replan_after_review=True)
    else:
        before=source.read_bytes()
        with pytest.raises(ValueError,match='Ambiguous existing'):p.prepare(root,replan_after_review=True)
        assert source.read_bytes()==before
    assert archive.read_bytes()==original


def test_not_hard_excluded_and_review_drift_detected(planning_project):
    root,rr,store=planning_project
    store.confirm('28','NOT_ACTUALLY_HARD')
    plan=p.prepare(root,replan_after_review=True)
    assert plan['image_sampling_weights'][rr[28]['image']]==1
    assert rr[28]['image'] not in plan['eligible_hard_images']
    store.confirm('28','VALID_HARD_CASE')
    with pytest.raises(ValueError,match='Human review provenance drift'):p.verify(root)


@pytest.mark.parametrize('split',['valid','test'])
def test_post_review_rejects_foreign_paths(planning_project,split):
    root,_,store=planning_project
    data=store.path.read_text().replace('train/images/0029.jpg',f'{split}/images/0029.jpg')
    store.path.write_text(data)
    with pytest.raises(ValueError,match='TRAIN-only'):p.prepare(root,replan_after_review=True)
    assert all(not a.exists() for a in p.archive_sources(root).values())


def test_publication_failure_rolls_back_and_allows_safe_retry(planning_project,monkeypatch):
    root,_,_=planning_project
    before={s:s.read_bytes() for s in p.archive_sources(root)}
    atomic=p.atomic_bytes
    failed=False
    def fail_once(path,data,**kwargs):
        nonlocal failed
        if path==root/p.PLAN and not failed:
            failed=True
            raise OSError('simulated interrupted publication')
        return atomic(path,data,**kwargs)
    monkeypatch.setattr(p,'atomic_bytes',fail_once)
    with pytest.raises(OSError):p.prepare(root,replan_after_review=True)
    assert all(s.read_bytes()==v for s,v in before.items())
    assert p.prepare(root,replan_after_review=True)['plan_stage']=='post_human_review'


def test_planner_help_does_not_prepare(monkeypatch,capsys):
    monkeypatch.setattr(p,'prepare',lambda *a,**k:pytest.fail('Help must not mutate'))
    with pytest.raises(SystemExit) as result:p.main(['--help'])
    assert result.value.code==0
    assert '--replan-after-review' in capsys.readouterr().out
