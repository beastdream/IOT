"""Curated v1: approved P0 decisions only, independent copies, verifiable provenance."""

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import shutil
import tempfile

import yaml

from smart_helmet.foundation import EXPECTED_CLASSES, SPLIT_FOLDERS, load_data_config, load_detection_config
from smart_helmet.paths import PROJECT_ROOT
from .audit import fingerprint, parse_yolo_line, write_csv
from .cleaning_review import read_csv, validate_decisions
from .duplicates import sha256_file

FINAL_DECISIONS = {'KEEP_BOTH','KEEP_A_EXCLUDE_B','KEEP_B_EXCLUDE_A'}


def locations(root):
    return (root/'data/curated/v1', root/'configs/data.curated.v1.yaml', root/'results/dataset_cleaning')


def raw_snapshot(root):
    saved=json.loads((root/'results/dataset_audit/dataset_fingerprint.json').read_text(encoding='utf-8'))
    current=fingerprint(root)
    if saved.get('status')!='PASS' or current!=saved['before'] or current!=saved['after']:
        raise ValueError('RAW_DATASET_CHANGED: fingerprint differs from raw audit')
    return current


def action_map(records, queue, decisions):
    validation=validate_decisions(queue,decisions)
    if not validation['valid']:
        raise ValueError('INVALID_DECISIONS: '+'; '.join(validation['errors']))
    lookup={r['review_id']:r for r in queue}
    final={r['review_id']:r['decision'] for r in decisions}
    p0=[r for r in queue if r['priority']=='P0']
    if any(final[r['review_id']] not in FINAL_DECISIONS for r in p0):
        raise ValueError('P0_INCOMPLETE: PENDING or REVIEW_MORE; no curated dataset may be created')
    votes=defaultdict(list)
    for case in p0:
        rid=case['review_id'];decision=final[rid]
        a,b=case['image_a'],case['image_b']
        if a==b or a not in records or b not in records:
            raise ValueError(f'Invalid P0 image reference: {rid}')
        if records[a]['split']==records[b]['split']:
            raise ValueError(f'P0 must be cross-split: {rid}')
        for side,name,other in [('A',a,b),('B',b,a)]:
            exclude=(decision=='KEEP_A_EXCLUDE_B' and side=='B') or (decision=='KEEP_B_EXCLUDE_A' and side=='A')
            votes[name].append(dict(action='EXCLUDE' if exclude else 'KEEP',review_id=rid,decision=decision,paired_with=other))
    conflicts={name:items for name,items in votes.items() if len({v['action'] for v in items})>1}
    if conflicts:
        raise ValueError('DECISION_CONFLICT: '+json.dumps(conflicts))
    result={name:dict(action=votes[name][0]['action'] if votes[name] else 'KEEP',evidence=votes[name]) for name in records}
    return result,validation,Counter(final[r['review_id']] for r in p0)


def dataset_totals(records):
    rows=list(records)
    classes={str(k):sum(int(r['with_helmet_count' if k==0 else 'without_helmet_count']) for r in rows) for k in EXPECTED_CLASSES}
    return dict(**{f'{s}_images':sum(r['split']==s for r in rows) for s in SPLIT_FOLDERS.values()},
                total_images=len(rows),annotations=sum(classes.values()),class_counts=classes)


def build_plan(root=PROJECT_ROOT):
    """Pure read-only validation shared by --dry-run, --apply and readiness."""
    root=Path(root).resolve();target,config,results=locations(root)
    load_data_config(root/'configs/data.local.yaml',root)
    queue=read_csv(results/'review_queue.csv');decisions=read_csv(results/'review_decisions.csv')
    records_list=read_csv(root/'results/dataset_audit/dataset_manifest.csv')
    records={r['image_path']:r for r in records_list}
    if len(records)!=len(records_list):raise ValueError('Duplicate raw manifest image')
    actions,validation,counts=action_map(records,queue,decisions)
    raw=raw_snapshot(root)
    expected_images=set()
    for folder in SPLIT_FOLDERS.values():
        expected_images.update(p.relative_to(root).as_posix() for p in (root/folder/'images').iterdir() if p.is_file())
    if set(records)!=expected_images:raise ValueError('Raw manifest does not cover exactly the original image set')
    for name,row in records.items():
        image=Path(name);label=Path(row['label_path']);split=row['split']
        if split not in SPLIT_FOLDERS.values() or image.parts[:2]!=(split,'images') or label!=Path(split)/'labels'/(image.stem+'.txt'):
            raise ValueError(f'Invalid image-label pairing: {name}')
        for path in (image,label):
            if not (root/path).resolve().is_relative_to(root) or path.as_posix() not in raw:
                raise ValueError(f'Unfingerprinted raw path: {path}')
        if raw[name]!=row['sha256']:raise ValueError(f'Stale manifest hash: {name}')
        classes=Counter()
        for line in (root/label).read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():continue
            box,issues=parse_yolo_line(line)
            if box is None or any(v[1]=='ERROR' for v in issues):raise ValueError(f'Invalid raw annotation: {label}')
            classes[box['class_id']]+=1
        if classes[0]!=int(row['with_helmet_count']) or classes[1]!=int(row['without_helmet_count']):
            raise ValueError(f'Stale annotation counts: {name}')
    # Validate review coverage and attachment to the unchanged audited images.
    p0=[r for r in queue if r['priority']=='P0']
    audit_pairs=read_csv(root/'results/dataset_audit/near_duplicate_leakage.csv')
    keys=lambda rows:{tuple(sorted((r['image_a'],r['image_b']))) for r in rows}
    if keys(p0)!=keys(audit_pairs):raise ValueError('P0 queue does not cover raw cross-split audit candidates')
    for r in p0:
        for side in ('a','b'):
            row=records[r['image_'+side]]
            if r['split_'+side]!=row['split'] or r['sha256_'+side]!=row['sha256']:
                raise ValueError('P0 evidence does not match the raw manifest')
    kept=[r for n,r in records.items() if actions[n]['action']=='KEEP']
    original=dataset_totals(records.values());curated=dataset_totals(kept)
    if any(curated[f'{s}_images']==0 for s in SPLIT_FOLDERS.values()):raise ValueError('Cleaning would empty a split')
    exclusions=Counter(records[n]['split'] for n,a in actions.items() if a['action']=='EXCLUDE')
    inputs=[results/'review_queue.csv',results/'review_decisions.csv',root/'results/dataset_audit/dataset_manifest.csv',
            root/'results/dataset_audit/near_duplicate_leakage.csv',root/'results/dataset_audit/dataset_fingerprint.json']
    return dict(root=root,target=target,config=config,results=results,records=records,actions=actions,raw=raw,
                summary=dict(dataset_version='curated_v1',status='PLANNED',original=original,curated=curated,
                             cleaning=dict(p0_decisions=sum(counts.values()),keep_both=counts['KEEP_BOTH'],
                                           keep_a_exclude_b=counts['KEEP_A_EXCLUDE_B'],keep_b_exclude_a=counts['KEEP_B_EXCLUDE_A'],
                                           unique_images_excluded=sum(exclusions.values()),exclusion_by_split={s:exclusions[s] for s in SPLIT_FOLDERS.values()}),
                             pending_reviews={p:validation['priorities'][p]['pending'] for p in ('P1','P2','P3')},
                             other_unresolved_reviews={p:validation['priorities'][p]['review_more'] for p in ('P1','P2','P3')},
                             policy='Only final human P0 decisions applied. P1/P2/P3 cause no additional exclusion, relabel, resize or resplit; images overlapping P0 exclusions follow P0.',
                             input_sha256={p.relative_to(root).as_posix():sha256_file(p) for p in inputs}))


def print_plan(plan):
    s=plan['summary'];print('='*40+'\nCURATED DATASET V1 - DRY RUN\n'+'='*40)
    print(f"P0 decisions: {s['cleaning']['p0_decisions']} / {s['cleaning']['p0_decisions']} completed\nDecision conflicts: 0")
    print(f"Original images: {s['original']['total_images']}\nImages to keep: {s['curated']['total_images']}\nImages to exclude: {s['cleaning']['unique_images_excluded']}")
    for split in SPLIT_FOLDERS.values():
        print(f"{split}: original={s['original'][split+'_images']} excluded={s['cleaning']['exclusion_by_split'][split]} curated={s['curated'][split+'_images']}")
    for k,name in EXPECTED_CLASSES.items():print(f"{name}: {s['curated']['class_counts'][str(k)]}")
    print('Exclusions:')
    for name,a in plan['actions'].items():
        if a['action']=='EXCLUDE':print(name+' | '+','.join(e['review_id'] for e in a['evidence']))
    print('RESULT:\nPASS\n'+'='*40)


def reject_reparse(path):
    if path.exists() and (path.is_symlink() or getattr(path.lstat(),'st_file_attributes',0)&0x400):
        raise ValueError(f'Reparse point is not permitted in curated build paths: {path}')


def safe_remove_owned(path, parent, expected_name=None):
    """Never delete a computed tree before verifying its absolute scope."""
    path=Path(path);parent=Path(parent)
    if path.resolve().parent!=parent.resolve() or (expected_name and path.name!=expected_name):
        raise ValueError('Unsafe curated removal target')
    reject_reparse(parent);reject_reparse(path)
    if path.exists():
        for p in path.rglob('*'):reject_reparse(p)
        shutil.rmtree(path)


def curated_fingerprint(root):
    target,config,_=locations(root)
    if not target.is_dir() or not config.is_file():raise ValueError('Curated dataset/config missing')
    files=sorted(p for p in target.rglob('*') if p.is_file())+[config]
    return {p.relative_to(root).as_posix():sha256_file(p) for p in files}


def verify_curated_fingerprint(root):
    _,_,results=locations(root)
    saved=json.loads((results/'curated_v1_fingerprint.json').read_text(encoding='utf-8'))
    actual=curated_fingerprint(root)
    if saved.get('status')!='PASS' or actual!=saved['files']:raise ValueError('Curated fingerprint mismatch')
    return actual


def apply_cleaning(root=PROJECT_ROOT,rebuild=False):
    plan=build_plan(root);root=plan['root'];target=plan['target'];config=plan['config'];results=plan['results']
    # No merge. --rebuild only ever replaces this exact v1 directory.
    for p in (root/'data',target.parent,target):reject_reparse(p)
    if target.resolve()!=root/'data/curated/v1':raise ValueError('Unsafe curated destination')
    if target.exists() and not rebuild:raise ValueError('Curated v1 already exists; use explicit --apply --rebuild')
    target.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.v1-stage-',dir=target.parent))
    manifest=[];exclusions=[]
    try:
        for folder in SPLIT_FOLDERS.values():
            for kind in ('images','labels'):(stage/folder/kind).mkdir(parents=True)
        for name,row in plan['records'].items():
            info=plan['actions'][name];keep=info['action']=='KEEP';label=row['label_path'];split=row['split']
            evidence=info['evidence'];rid='|'.join(e['review_id'] for e in evidence);decision='|'.join(e['decision'] for e in evidence)
            reason='Human-reviewed P0 decision' if evidence else 'Retained without changes; no P0 exclusion'
            dest_image=(target/name).relative_to(root).as_posix() if keep else ''
            dest_label=(target/label).relative_to(root).as_posix() if keep else ''
            if keep:
                for source in (name,label):
                    shutil.copy2(root/source,stage/source)
                    if sha256_file(stage/source)!=plan['raw'][source] or os.path.samefile(root/source,stage/source):
                        raise ValueError(f'Independent copy verification failed: {source}')
            else:
                exclusions.append(dict(image=name,label=label,original_split=split,review_id=rid,decision=decision,reason=reason,
                                       paired_with='|'.join(e['paired_with'] for e in evidence),
                                       paired_split='|'.join(plan['records'][e['paired_with']]['split'] for e in evidence),human_reviewed='TRUE'))
            manifest.append(dict(original_split=split,original_image=name,original_label=label,curated_split=split if keep else '',
                                 curated_image=dest_image,curated_label=dest_label,action=info['action'],reason=reason,review_id=rid,decision=decision,
                                 source_sha256=plan['raw'][name],curated_sha256=plan['raw'][name] if keep else '',
                                 source_label_sha256=plan['raw'][label],curated_label_sha256=plan['raw'][label] if keep else ''))
        if raw_snapshot(root)!=plan['raw']:raise ValueError('Raw changed during build')
        for name,digest in plan['summary']['input_sha256'].items():
            if sha256_file(root/name)!=digest:raise ValueError('Review/audit evidence changed during build')
        if target.exists():safe_remove_owned(target,target.parent,'v1')
        stage.rename(target)
        cfg=dict(path='data/curated/v1',train='train/images',val='valid/images',test='test/images',nc=2,names=EXPECTED_CLASSES)
        config.write_text('# Native YOLO use: run from project root. No fallback paths.\n'+yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
        loaded=load_detection_config(config,root)
        if not all(Path(loaded[k]).is_dir() for k in SPLIT_FOLDERS):raise ValueError('Curated config path missing')
        write_csv(results/'curated_v1_manifest.csv',manifest,
                  'original_split original_image original_label curated_split curated_image curated_label action reason review_id decision source_sha256 curated_sha256 source_label_sha256 curated_label_sha256'.split())
        write_csv(results/'curated_v1_exclusions.csv',exclusions,
                  'image label original_split review_id decision reason paired_with paired_split human_reviewed'.split())
        fp=curated_fingerprint(root)
        (results/'curated_v1_fingerprint.json').write_text(json.dumps(dict(dataset_version='curated_v1',status='PASS',files=fp),indent=2),encoding='utf-8')
        if raw_snapshot(root)!=plan['raw']:raise ValueError('RAW_DATASET_CHANGED after apply')
        summary=plan['summary'];summary.update(status='PASS',raw_immutability='PASS',copy_method='shutil.copy2',curated_fingerprint_files=len(fp))
        summary['artifact_sha256']={n:sha256_file(results/n) for n in ('curated_v1_manifest.csv','curated_v1_exclusions.csv','curated_v1_fingerprint.json')}
        (results/'curated_v1_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        return summary
    finally:
        if stage.exists():safe_remove_owned(stage,target.parent)


def classify_curated_pairs(root,pairs):
    """Only the exact retained pair with KEEP_BOTH and matching build provenance is reviewed."""
    _,_,results=locations(root)
    summary=json.loads((results/'curated_v1_summary.json').read_text(encoding='utf-8'))
    if summary['status']!='PASS':raise ValueError('No successful curated build')
    for name,digest in summary['input_sha256'].items():
        if sha256_file(root/name)!=digest:raise ValueError('Build decisions/provenance changed; rebuild required')
    for name,digest in summary['artifact_sha256'].items():
        if sha256_file(results/name)!=digest:raise ValueError('Build artifact changed')
    verify_curated_fingerprint(root)
    manifest=read_csv(results/'curated_v1_manifest.csv')
    raw_by_curated={r['curated_image']:r['original_image'] for r in manifest if r['action']=='KEEP'}
    queue=read_csv(results/'review_queue.csv');decisions=read_csv(results/'review_decisions.csv')
    validation=validate_decisions(queue,decisions)
    if not validation['valid'] or not validation['ready_to_apply_cleaning']:raise ValueError('P0 reviews invalid or incomplete')
    choices={r['review_id']:r['decision'] for r in decisions}
    kept_pairs={tuple(sorted((r['image_a'],r['image_b']))):r['review_id'] for r in queue if r['priority']=='P0' and choices[r['review_id']]=='KEEP_BOTH'}
    result=[]
    for pair in pairs:
        if pair['split_a']==pair['split_b']:continue
        names=tuple(sorted((raw_by_curated[pair['image_a']],raw_by_curated[pair['image_b']])))
        rid=kept_pairs.get(names,'')
        result.append(dict(**{k:pair[k] for k in ('image_a','image_b','split_a','split_b','distance')},
                           review_status='REVIEWED_KEEP_BOTH_CANDIDATE' if rid else 'UNREVIEWED_CROSS_SPLIT_CANDIDATE',review_id=rid))
    return result


def training_readiness(root=PROJECT_ROOT):
    root=Path(root).resolve();target,config,results=locations(root);checks={};errors=[];pending={}
    try:
        plan=build_plan(root);pending=plan['summary']['pending_reviews']
        checks['P0 cleaning']=True;checks['RAW DATASET IMMUTABLE']=True
        loaded=load_detection_config(config,root)
        checks['Curated dataset']=Path(loaded['path'])==target and all(Path(loaded[k]).is_dir() for k in SPLIT_FOLDERS)
        checks['Class mapping']=loaded['names']==EXPECTED_CLASSES
        current=verify_curated_fingerprint(root);checks['Curated fingerprint']=True
        manifest=read_csv(results/'curated_v1_manifest.csv')
        if {r['original_image'] for r in manifest}!=set(plan['records']) or len(manifest)!=len(plan['records']):
            raise ValueError('Build manifest coverage mismatch')
        expected_files={config.relative_to(root).as_posix()}
        for row in manifest:
            name=row['original_image'];action=plan['actions'][name]['action']
            if row['action']!=action:raise ValueError('Build does not match current P0 action map')
            for source,dest in ((name,row['curated_image']),(row['original_label'],row['curated_label'])):
                if action=='KEEP':
                    expected=(target/source).relative_to(root).as_posix()
                    if dest!=expected or current.get(dest)!=plan['raw'][source]:raise ValueError('Curated copy/provenance mismatch')
                    if os.path.samefile(root/source,root/dest):raise ValueError('Curated file is not independent')
                    expected_files.add(dest)
                elif dest or (target/source).exists():raise ValueError('Excluded file still present')
        if set(current)!=expected_files:raise ValueError('Unexpected curated files')
        audit_dir=root/'results/dataset_audit/curated_v1'
        report=json.loads((audit_dir/'dataset_audit_report.json').read_text(encoding='utf-8'))
        audit_fp=json.loads((audit_dir/'dataset_fingerprint.json').read_text(encoding='utf-8'))
        if current!=audit_fp['before'] or current!=audit_fp['after'] or audit_fp['status']!='PASS':raise ValueError('Curated audit is stale')
        if report['status']=='FAIL' or report['fatal_errors']:raise ValueError('Curated audit failed')
        expected=plan['summary']['curated']
        if report['dataset']['total_images']!=expected['total_images'] or report['dataset']['total_annotations']!=expected['annotations']:
            raise ValueError('Curated audit counts mismatch')
        checks['Dataset integrity']=not (report['images']['corrupted'] or report['images']['missing_labels'] or report['images']['orphan_labels'] or report['annotations']['invalid'])
        pairs=read_csv(audit_dir/'near_duplicate_leakage.csv')
        classified=classify_curated_pairs(root,pairs)
        unresolved=sum(r['review_status']=='UNREVIEWED_CROSS_SPLIT_CANDIDATE' for r in classified)
        checks['Unreviewed leakage']=unresolved==0
        # Numeric validity alone must not hide an incomplete audit artifact set.
        for filename in ('image_issues.csv','annotation_issues.csv','exact_duplicates.csv','source_groups.csv','image_resolutions.csv','bbox_statistics.csv'):
            read_csv(audit_dir/filename)
        checks['Curated audit complete']=True
    except (OSError,ValueError,KeyError) as error:
        errors.append(str(error))
    ready=not errors and bool(checks) and all(checks.values())
    return dict(checks=checks,errors=errors,pending_reviews=pending,ready_for_baseline_training=ready)
