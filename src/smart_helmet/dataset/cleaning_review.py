"""Prepare review decisions from audit evidence; never apply cleaning."""

from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import random

from smart_helmet.foundation import EXPECTED_CLASSES, load_data_config
from smart_helmet.paths import PROJECT_ROOT, DATASET_AUDIT_DIR, DATASET_CLEANING_DIR
from .audit import fingerprint, write_csv
from .duplicates import hamming_distance, sha256_file
from .cleaning_visuals import FOLDERS, generate_comparison, render_html

PRIORITIES = {'CROSS_SPLIT_NEAR_DUPLICATE':'P0','TINY_BBOX':'P1',
              'UNUSUAL_RESOLUTION':'P2','SOURCE_VARIANT_REVIEW':'P3'}
ALLOWED = {
    'P0': {'PENDING','KEEP_BOTH','KEEP_A_EXCLUDE_B','KEEP_B_EXCLUDE_A','REVIEW_MORE'},
    'P1': {'PENDING','KEEP','CHECK_BBOX','REVIEW_MORE'},
    'P2': {'PENDING','KEEP','REVIEW_MORE'},
    'P3': {'PENDING','KEEP_ALL','REVIEW_MORE'},
}
SEED = 42
SOURCE_SAMPLE_SIZE = 40
QUEUE_FIELDS = ('review_id priority issue_type split_a image_a split_b image_b phash_a phash_b phash_distance '
                'source_group_a source_group_b sha256_a sha256_b annotation_count_a annotation_count_b classes_a classes_b '
                'split image_path label_path class_id class_name normalized_width normalized_height normalized_area bbox_line_number '
                'source_group_id members suggested_action decision review_notes reason evidence visualization_path').split()
ACTION_FIELDS = 'review_id priority issue_type target_image target_label current_split suggested_action reason requires_human_approval approved applied'.split()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f'Missing or duplicate CSV headers: {path}')
        rows = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError(f'Malformed CSV row: {path}')
        return rows


def stable_id(priority, identity):
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return f'{priority}_{digest}'


def policy_suggestion(split_a, split_b):
    rank = {'train':0,'valid':1,'test':2}
    if split_a not in rank or split_b not in rank or split_a == split_b:
        raise ValueError('P0 requires two different known splits')
    return 'KEEP_A_EXCLUDE_B' if rank[split_a] > rank[split_b] else 'KEEP_B_EXCLUDE_A'


def numeric_boxes(rows):
    result = []
    for row in rows:
        r = dict(row)
        for key in ('class_id','line_number'):
            r[key] = int(r[key])
        for key in ('x','y','normalized_width','normalized_height','normalized_area','aspect_ratio'):
            r[key] = float(r[key])
        result.append(r)
    return result


def build_queue(manifest, boxes, leakage, manual, sample_size=SOURCE_SAMPLE_SIZE):
    records = {r['image_path']: r for r in manifest}
    if len(records) != len(manifest):
        raise ValueError('Duplicate images in audit manifest')
    by_image = defaultdict(list)
    for b in boxes:
        if b['image'] not in records:
            raise ValueError('BBox references an unknown image')
        by_image[b['image']].append(b)
    evidence = defaultdict(set)
    for r in manual:
        evidence[r['image_path']].add(r['reason'])

    def signature(name):
        return sorted(tuple(b[k] for k in ('class_id','x','y','normalized_width','normalized_height')) for b in by_image[name])

    def identity(name):
        return [name,records[name]['sha256'],signature(name)]

    def class_counts(name):
        counts = Counter(b['class_id'] for b in by_image[name])
        return {EXPECTED_CLASSES[k]:counts[k] for k in EXPECTED_CLASSES}

    queue = []
    identities = set()

    def add(kind, members, key, suggested, reason, **extra):
        priority = PRIORITIES[kind]
        rid = stable_id(priority,key)
        if rid in identities:
            return
        identities.add(rid)
        merged = sorted(set().union(*(evidence[name] for name in members)))
        queue.append(dict(review_id=rid, priority=priority, issue_type=kind, members=members,
                          suggested_action=suggested, decision='PENDING', review_notes='', reason=reason,
                          evidence='; '.join(merged), visualization_path='', **extra))

    for pair in sorted(leakage, key=lambda r: tuple(sorted((r['image_a'],r['image_b'])))):
        left, right = sorted((pair['image_a'],pair['image_b']))
        a, b = records[left], records[right]
        distance = hamming_distance(a['perceptual_hash'],b['perceptual_hash'])
        if distance != int(pair['distance']):
            raise ValueError('Audit pair distance does not match manifest hashes')
        suggestion = policy_suggestion(a['split'],b['split'])
        add('CROSS_SPLIT_NEAR_DUPLICATE',[left,right],[identity(left),identity(right)],suggestion,
            'Only if human confirms genuine leakage: preserve the higher-priority evaluation split (test > valid > train). Otherwise KEEP_BOTH or REVIEW_MORE.',
            split_a=a['split'], image_a=left, split_b=b['split'], image_b=right,
            phash_a=a['perceptual_hash'], phash_b=b['perceptual_hash'], phash_distance=distance,
            source_group_a=a['source_group_id'],source_group_b=b['source_group_id'],
            sha256_a=a['sha256'],sha256_b=b['sha256'],
            annotation_count_a=len(by_image[left]),annotation_count_b=len(by_image[right]),
            classes_a=json.dumps(class_counts(left)),classes_b=json.dumps(class_counts(right)))
    for box in sorted(boxes,key=lambda b:(b['image'],b['line_number'])):
        if box['size_category'] != 'tiny':
            continue
        name = box['image'];r = records[name]
        add('TINY_BBOX',[name],[identity(name),box['line_number']], 'CHECK_BBOX',
            'Inspect head context and box placement. Small area alone is never a reason to delete.',
            split=r['split'],image_path=name,label_path=r['label_path'],class_id=box['class_id'],
            class_name=EXPECTED_CLASSES[box['class_id']], normalized_width=box['normalized_width'],
            normalized_height=box['normalized_height'],normalized_area=box['normalized_area'],bbox_line_number=box['line_number'])
    for name,r in sorted(records.items()):
        if (int(r['image_width']),int(r['image_height'])) != (416,416):
            add('UNUSUAL_RESOLUTION',[name],identity(name),'KEEP_IF_ANNOTATION_CORRECT',
                f"Resolution {r['image_width']}x{r['image_height']}; no resize or exclusion suggested solely for dimensions.",
                split=r['split'],image_path=name,label_path=r['label_path'])

    groups = defaultdict(list)
    for name,r in sorted(records.items()):
        groups[r['source_group_id']].append(name)
    group_rows = []
    for gid,members in sorted(groups.items()):
        if len(members) < 2:
            continue
        distances = [dict(image_a=a,image_b=b,distance=hamming_distance(records[a]['perceptual_hash'],records[b]['perceptual_hash']))
                     for a,b in itertools.combinations(members,2)]
        sigs = [signature(n) for n in members]
        counts = [class_counts(n) for n in members]
        group_rows.append(dict(source_group_id=gid,split='|'.join(sorted({records[n]['split'] for n in members})),group_size=len(members),
                               images=members,phash_distances=distances,annotation_counts=[len(by_image[n]) for n in members],classes=counts,
                               annotation_difference=any(s != sigs[0] for s in sigs[1:]),
                               class_count_difference=any(c != counts[0] for c in counts[1:]),
                               minimum_phash_distance=min(d['distance'] for d in distances)))
    # Fixed-seed tie breaking, round-robin across four selection criteria.
    rng = random.Random(SEED)
    shuffled = list(group_rows);rng.shuffle(shuffled)
    criteria = {
        'largest_groups':sorted(shuffled,key=lambda g:-g['group_size']),
        'lowest_phash_distance':sorted(shuffled,key=lambda g:g['minimum_phash_distance']),
        'annotation_difference':[g for g in shuffled if g['annotation_difference']],
        'class_count_difference':[g for g in shuffled if g['class_count_difference']],
    }
    chosen = {};cursors = {k:0 for k in criteria}
    while len(chosen) < min(sample_size,len(group_rows)):
        progress = False
        for criterion,candidates in criteria.items():
            while cursors[criterion] < len(candidates) and candidates[cursors[criterion]]['source_group_id'] in chosen:
                cursors[criterion] += 1
            if cursors[criterion] < len(candidates) and len(chosen) < sample_size:
                group = candidates[cursors[criterion]];cursors[criterion] += 1
                chosen[group['source_group_id']] = (group,criterion);progress = True
        if not progress:
            break
    for group,criterion in chosen.values():
        members = group['images']
        add('SOURCE_VARIANT_REVIEW',members,[identity(n) for n in members],'KEEP_ALL',
            f'Informational same-split sample selected by {criterion}; do not automatically remove variants.',
            split=group['split'],source_group_id=group['source_group_id'])
    queue.sort(key=lambda r:(r['priority'],r['review_id']))
    return queue,group_rows,records,by_image


def validate_decisions(queue, decisions, require_complete=True):
    lookup = {r['review_id']:r for r in queue};errors=[];seen={}
    if len(lookup) != len(queue):
        errors.append('Duplicate review_id in queue')
    for row in queue:
        if row['priority'] not in ALLOWED:
            errors.append(f"Unknown priority: {row['priority']}")
    for row in decisions:
        rid=row.get('review_id');decision=row.get('decision')
        if rid in seen:
            errors.append(f'Duplicate decision review_id: {rid}')
        seen[rid]=decision
        if rid not in lookup:
            errors.append(f'Unknown review_id: {rid}')
        elif decision not in ALLOWED.get(lookup[rid]['priority'],set()):
            errors.append(f'Invalid decision for {rid}: {decision}')
    missing=set(lookup)-set(seen)
    if require_complete and missing:
        errors.append(f'Missing decisions for {len(missing)} review IDs')
    result={}
    for priority in ALLOWED:
        ids=[r['review_id'] for r in queue if r['priority']==priority]
        pending=sum(seen.get(rid,'PENDING')=='PENDING' for rid in ids)
        more=sum(seen.get(rid)=='REVIEW_MORE' for rid in ids)
        completed=sum(seen.get(rid) in ALLOWED[priority]-{'PENDING','REVIEW_MORE'} for rid in ids)
        result[priority]=dict(total=len(ids),completed=completed,pending=pending,review_more=more)
    blocking=result['P0']['pending']+result['P0']['review_more']
    return dict(valid=not errors,errors=errors,priorities=result,blocking_p0_pending=blocking>0,
                blocking_reviews_pending=blocking,ready_to_apply_cleaning=not errors and blocking==0)


def merge_decisions(queue, existing):
    validation=validate_decisions(queue,existing,require_complete=False)
    if not validation['valid']:
        raise ValueError('Existing decisions preserved without overwrite: '+'; '.join(validation['errors']))
    old={r['review_id']:r for r in existing}
    return [dict(old[r['review_id']]) if r['review_id'] in old else
            dict(review_id=r['review_id'],decision='PENDING',notes='') for r in queue]


def print_decision_status(result):
    print('='*40+'\nDATASET REVIEW DECISIONS\n'+'='*40)
    for priority,values in result['priorities'].items():
        print(priority+':')
        for key,value in values.items():
            print(f'{key.capitalize()}: {value}')
    for error in result['errors']:
        print('ERROR: '+error)
    print('BLOCKING P0 PENDING:\n'+('YES' if result['blocking_p0_pending'] else 'NO'))
    print('READY TO APPLY CLEANING:\n'+('YES' if result['ready_to_apply_cleaning'] else 'NO'))
    print('VALIDATION: '+('PASS' if result['valid'] else 'FAIL')+'\n'+'='*40)


POLICY = '''# Cleaning policy — suggestions only

A. Raw dataset is immutable: original images, labels, data.yaml and Roboflow README files stay unchanged.
B. Never automatically change semantic labels or class IDs: 0 = With Helmet; 1 = Without Helmet.
C. Never automatically delete a box because it is tiny.
D. Never remove an image merely because its resolution differs from 416x416; no resize is proposed here.
E. Future exact-byte duplicates may be handled by an explicitly approved deterministic policy. The current audit found zero. Nothing is applied in Phase 3A.
F. Near duplicates always need human review before exclusion. pHash distance zero is not pixel identity.
G. Confirmed genuine cross-split leakage must be resolved before final baseline training.
H. If genuine leakage is confirmed, prefer KEEP VALID / EXCLUDE TRAIN for train-valid, KEEP TEST / EXCLUDE TRAIN for train-test, KEEP TEST / EXCLUDE VALID for valid-test. These are conditional suggestions only.
I. Do not rebalance by duplicating Without Helmet in cleaning.
J. Never change the test set based on model performance.
K. Same-source variants within train are not automatically leakage.
L. Validation/test diversity matters more than preserving near-identical variants.
M. Every future dataset change must have a manifest linking raw source, reviewed decision, action and resulting version.

## Human judgement

Confirm genuine leakage when evidence shows the same source/event, scene, near frame, person and composition, with only minor resize/brightness/crop/augmentation differences. Visual resemblance or similar people alone is not enough. For different scenes/people or independent frames/events choose KEEP_BOTH; if uncertain choose REVIEW_MORE.

## Workload and priorities

P0 BLOCKING: all cross-split candidate pairs. P1 HIGH: each tiny box. P2 MEDIUM: each unusual-resolution image. P3 INFORMATIONAL: a seed-42 sample of same-split source groups. P3 does not block baseline by itself. Evidence is merged by image and bounded review unit, not copied from all manual-evidence rows. P0 stays pair-based and P1 stays bbox-based to preserve distinct decisions.

P3 sample target is 40 groups (or all when fewer), selected round-robin across largest groups, lowest pHash distances, annotation differences and class-count differences. Empty strata are reported rather than invented. Numeric annotation signatures include class and box coordinates; a difference is not proof of an error.

## Decisions and future actions

Edit only review_decisions.csv: review_id, decision, notes. review_queue.csv decisions start PENDING and are template fields; the decision CSV is the authority. Stable IDs include image identity and annotation signatures. Rerunning preparation preserves existing decision values and notes; unknown/duplicate/invalid IDs cause failure rather than deleting human input.

P0: PENDING, KEEP_BOTH, KEEP_A_EXCLUDE_B, KEEP_B_EXCLUDE_A, REVIEW_MORE.
P1: PENDING, KEEP, CHECK_BBOX, REVIEW_MORE.
P2: PENDING, KEEP, REVIEW_MORE.
P3: PENDING, KEEP_ALL, REVIEW_MORE.

P1 suggestions are KEEP / CHECK_BBOX / CHECK_MISSING_CONTEXT; default CHECK_BBOX. P2 suggestion KEEP_IF_ANNOTATION_CORRECT. P3 suggestion KEEP_ALL.

The validator reports REVIEW_MORE separately; it remains unresolved for P0 and blocks readiness just like PENDING. Other priorities may remain pending, and their counts must stay visible. READY TO APPLY CLEANING is a review-completeness gate only, not an instruction or authorization to execute changes. No apply function is provided. Proposed actions always remain requires_human_approval=true, approved=false, applied=false in this phase, even after a decision is entered.

Group-valued CSV fields (members, images, distances, counts, classes) contain JSON. A P3 proposed-action target may contain pipe-separated member paths; it is informational, not an executable filesystem argument. HTML is read-only; click comparisons for full size. Rebuild it with prepare after queue changes, and use the validator for current decision state.
'''


def prepare_review(root=PROJECT_ROOT,audit_dir=DATASET_AUDIT_DIR,output=DATASET_CLEANING_DIR):
    root=Path(root).resolve();audit_dir=Path(audit_dir).resolve();output=Path(output).resolve()
    if not output.is_relative_to(root/'results') or output==root/'results' or output==audit_dir or output.is_relative_to(audit_dir):
        raise ValueError('Cleaning review output must be a separate subdirectory under results/')
    load_data_config(root/'configs/data.local.yaml',root)
    saved=json.loads((audit_dir/'dataset_fingerprint.json').read_text(encoding='utf-8'))
    current=fingerprint(root)
    if saved.get('status')!='PASS' or current!=saved['before'] or current!=saved['after']:
        raise ValueError('FAIL: dataset fingerprint differs from audit; no preparation permitted')
    report=json.loads((audit_dir/'dataset_audit_report.json').read_text(encoding='utf-8'))
    if report['status']=='FAIL' or report['classes']['mapping']!={str(k):v for k,v in EXPECTED_CLASSES.items()}:
        raise ValueError('Audit integrity or class mapping is invalid')
    manifest=read_csv(audit_dir/'dataset_manifest.csv')
    boxes=numeric_boxes(read_csv(audit_dir/'bounding_boxes.csv'))
    leakage=read_csv(audit_dir/'near_duplicate_leakage.csv')
    manual=read_csv(audit_dir/'manual_review.csv')
    if len(manifest)!=report['dataset']['total_images'] or len(boxes)!=report['dataset']['total_annotations']:
        raise ValueError('Audit tables do not reconcile with summary')
    for row in manifest:
        for key in ('image_path','label_path'):
            path=(root/row[key]).resolve()
            if not path.is_relative_to(root) or row[key] not in current:
                raise ValueError('Audit references a path outside fingerprinted dataset')
        if current[row['image_path']]!=row['sha256']:
            raise ValueError('Manifest image hash differs from current dataset')
    queue,groups,records,by_image=build_queue(manifest,boxes,leakage,manual)
    # A cross-split source group must not be silently treated as informational.
    if any('|' in g['split'] for g in groups):
        raise ValueError('Unexpected cross-split source groups: extend P0 review coverage before proceeding')
    path=output/'review_decisions.csv'
    previous=path.read_bytes() if path.exists() else None
    existing=read_csv(path) if previous is not None else []
    if existing and any(set(r)!={'review_id','decision','notes'} for r in existing):
        raise ValueError('Existing decision columns must be review_id, decision, notes; file preserved')
    decisions=merge_decisions(queue,existing)
    state=validate_decisions(queue,decisions)
    output.mkdir(parents=True,exist_ok=True)
    (output/'review_report').mkdir(exist_ok=True)
    for folder in FOLDERS.values():
        (output/'review_images'/folder).mkdir(parents=True,exist_ok=True)
    print(f'Preparing {len(queue)} bounded review cases...',flush=True)
    actions=[]
    for row in queue:
        row['visualization_path']=generate_comparison(row,records,by_image,root,output)
        if row['priority']=='P0':
            targets=[row['image_b'] if row['suggested_action']=='KEEP_A_EXCLUDE_B' else row['image_a']]
        else:
            targets=row['members']
        actions.append(dict(review_id=row['review_id'],priority=row['priority'],issue_type=row['issue_type'],
                            target_image='|'.join(targets),target_label='|'.join(records[n]['label_path'] for n in targets),
                            current_split='|'.join(sorted({records[n]['split'] for n in targets})),
                            suggested_action=row['suggested_action'],reason=row['reason'],
                            requires_human_approval='true',approved='false',applied='false'))
    after=fingerprint(root)
    if current!=after:
        raise ValueError('FAIL: dataset changed during preparation')
    if (path.read_bytes() if path.exists() else None)!=previous:
        raise ValueError('Decisions changed during preparation; preserved. Rerun to merge.')
    # Preserve bytes for existing rows when no new review IDs are needed.
    if previous is None or {r['review_id'] for r in existing}!={r['review_id'] for r in decisions}:
        write_csv(path,decisions,['review_id','decision','notes'])
    queue_csv=[dict(r,members=json.dumps(r['members'])) for r in queue]
    write_csv(output/'review_queue.csv',queue_csv,QUEUE_FIELDS)
    write_csv(output/'proposed_actions.csv',actions,ACTION_FIELDS)
    group_csv=[{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in g.items()} for g in groups]
    write_csv(output/'source_variant_groups.csv',group_csv,
              'source_group_id split group_size images phash_distances annotation_counts classes annotation_difference class_count_difference minimum_phash_distance'.split())
    counts=Counter(r['priority'] for r in queue)
    summary=dict(status='PASS',P0=dict(cross_split_pairs=counts['P0'],affected_images=len({n for r in queue if r['priority']=='P0' for n in r['members']})),
                 P1=dict(tiny_bbox_cases=counts['P1']),P2=dict(unusual_resolution_images=counts['P2']),
                 P3=dict(source_groups_total=len(groups),sampled_groups=counts['P3'],seed=SEED,
                         annotation_difference_groups=sum(g['annotation_difference'] for g in groups),
                         class_count_difference_groups=sum(g['class_count_difference'] for g in groups)),
                 review_queue_size=len(queue),proposed_actions=len(actions),blocking_reviews_pending=state['blocking_reviews_pending'],
                 decision_status=state,immutability=dict(status='PASS',files_checked=len(current)),
                 dataset=report['dataset'],classes=report['classes']['mapping'],
                 input_evidence_sha256={p.name:sha256_file(p) for p in audit_dir.glob('*.csv')},
                 ready_for_baseline_training=False)
    (output/'cleaning_policy.md').write_text(POLICY,encoding='utf-8')
    (output/'review_report/index.html').write_text(render_html(queue,summary),encoding='utf-8')
    (output/'review_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('status','P0','P1','P2','P3','review_queue_size','immutability')},indent=2))
    return summary
