"""Read-only sanity check of built exclusions against evaluation split priority."""

from collections import Counter, defaultdict
from html import escape
import json
from pathlib import Path

from smart_helmet.paths import PROJECT_ROOT
from .audit import write_csv
from .cleaning_review import read_csv, numeric_boxes, validate_decisions
from .cleaning_visuals import panel, join_panels
from .visual_review import draw_annotation

SPLIT_PRIORITY = {'train':1, 'valid':2, 'test':3}
FIELDS = ('review_id excluded_image excluded_split retained_pair_image retained_split human_decision '
          'phash_distance source_group_excluded source_group_retained annotation_count_excluded annotation_count_retained '
          'classes_excluded classes_retained notes split_priority_excluded split_priority_retained '
          'direction_status policy_status explanation visualization_path').split()
TEST_FIELDS = ('review_id test_image paired_image paired_split decision phash_distance annotation_count_test '
               'annotation_count_pair classes_test classes_pair policy_status direction_status visualization_path notes').split()


def direction_status(excluded,retained):
    if excluded not in SPLIT_PRIORITY or retained not in SPLIT_PRIORITY or excluded==retained:
        return 'NEEDS_REVIEW'
    return 'OK' if SPLIT_PRIORITY[excluded]<SPLIT_PRIORITY[retained] else 'POLICY_MISMATCH'


def analyze_policy(root=PROJECT_ROOT):
    """One row per contributing review pair; summary counts distinct excluded images."""
    root=Path(root);p=root/'results/dataset_cleaning'
    queue=read_csv(p/'review_queue.csv');decisions=read_csv(p/'review_decisions.csv')
    validation=validate_decisions(queue,decisions)
    if not validation['valid']:raise ValueError('; '.join(validation['errors']))
    q={r['review_id']:r for r in queue};d={r['review_id']:r for r in decisions}
    manifest=read_csv(p/'curated_v1_manifest.csv');exclusions=read_csv(p/'curated_v1_exclusions.csv')
    m={r['original_image']:r for r in manifest}
    if len(m)!=len(manifest):raise ValueError('Duplicate original image in curated manifest')
    excluded={r['image'] for r in exclusions}
    if len(excluded)!=len(exclusions) or excluded!={r['original_image'] for r in manifest if r['action']=='EXCLUDE'}:
        raise ValueError('Exclusion manifest coverage mismatch')
    rows=[]
    for item in exclusions:
        image=item['image'];ids=item['review_id'].split('|');built=item['decision'].split('|')
        if len(ids)!=len(built):raise ValueError('Exclusion decision evidence is not aligned')
        for rid,built_decision in zip(ids,built):
            if rid not in q or q[rid]['priority']!='P0':raise ValueError('Exclusion lacks P0 review evidence')
            case=q[rid];decision=d[rid]
            side='a' if case['image_a']==image else 'b' if case['image_b']==image else None
            if side is None:raise ValueError('Excluded image is not part of its review pair')
            other='b' if side=='a' else 'a';retained=case['image_'+other]
            if retained not in m:raise ValueError('Unknown retained pair image')
            es,rs=item['original_split'],case['split_'+other]
            direction=direction_status(es,rs);notes=decision['notes'];status=direction
            explanation='Excluded lower-priority split; direction follows test > valid > train.'
            expected='KEEP_B_EXCLUDE_A' if side=='a' else 'KEEP_A_EXCLUDE_B'
            if built_decision!=expected or decision['decision']!=built_decision or m[retained]['action']!='KEEP':
                status='NEEDS_REVIEW';explanation='Current decision, built action or retained partner differs; no automatic rebuild.'
            elif direction=='POLICY_MISMATCH':
                status='NEEDS_REVIEW' if not notes.strip() else 'POLICY_MISMATCH'
                explanation=('Higher-priority evaluation image was excluded. Notes are empty; intentional exception is not established.'
                             if not notes.strip() else 'Direction reverses split priority. Notes are shown verbatim for human exception assessment; free text is not automatically approved.')
            elif direction=='NEEDS_REVIEW':explanation='Unknown or same split; evidence needs review.'
            rows.append(dict(review_id=rid,excluded_image=image,excluded_split=es,retained_pair_image=retained,retained_split=rs,
                             human_decision=decision['decision'],phash_distance=case.get('phash_distance',''),
                             source_group_excluded=case.get('source_group_'+side,''),source_group_retained=case.get('source_group_'+other,''),
                             annotation_count_excluded=case.get('annotation_count_'+side,''),annotation_count_retained=case.get('annotation_count_'+other,''),
                             classes_excluded=case.get('classes_'+side,''),classes_retained=case.get('classes_'+other,''),notes=notes,
                             split_priority_excluded=SPLIT_PRIORITY.get(es,0),split_priority_retained=SPLIT_PRIORITY.get(rs,0),
                             direction_status=direction,policy_status=status,explanation=explanation,visualization_path=''))
    by_split=Counter(r['original_split'] for r in exclusions)
    mismatches={r['excluded_image'] for r in rows if r['direction_status']=='POLICY_MISMATCH'}
    needs={r['excluded_image'] for r in rows if r['policy_status']=='NEEDS_REVIEW'}
    non_ok={r['excluded_image'] for r in rows if r['policy_status']!='OK'}
    summary=dict(total_p0_decisions=sum(r['priority']=='P0' for r in queue),unique_exclusions=len(excluded),
                 exclusion_by_split={s:by_split[s] for s in SPLIT_PRIORITY},policy_ok=len(excluded-non_ok),
                 policy_mismatch=len(mismatches),needs_review=len(needs),evidence_rows=len(rows),
                 mismatch_review_ids=sorted({r['review_id'] for r in rows if r['policy_status']!='OK'}),
                 count_definition='Counts are unique excluded images. policy_mismatch counts reversed directions; needs_review counts unresolved evidence, so these counts can overlap.',
                 status='REVIEW_REQUIRED' if non_ok or mismatches else 'PASS')
    return rows,summary


def test_exclusion_rows(rows):
    return [dict(review_id=r['review_id'],test_image=r['excluded_image'],paired_image=r['retained_pair_image'],paired_split=r['retained_split'],
                 decision=r['human_decision'],phash_distance=r['phash_distance'],annotation_count_test=r['annotation_count_excluded'],
                 annotation_count_pair=r['annotation_count_retained'],classes_test=r['classes_excluded'],classes_pair=r['classes_retained'],
                 policy_status=r['policy_status'],direction_status=r['direction_status'],visualization_path=r['visualization_path'],notes=r['notes'])
            for r in rows if r['excluded_split']=='test']


def policy_html(rows,summary):
    parts=['<!doctype html>','<html lang="en">','  <head>','    <meta charset="utf-8">',
           '    <meta name="viewport" content="width=device-width, initial-scale=1">',
           '    <title>PRE-TRAINING CURATION SANITY CHECK</title>',
           '    <style>body{font:16px/1.5 system-ui;max-width:1250px;margin:auto;padding:24px;background:#f4f6f8}article{background:white;border:1px solid #ccd3dc;padding:16px;margin:16px 0}img{max-width:100%}pre,p{overflow-wrap:anywhere;white-space:pre-wrap}.warning{color:#a33200}</style>',
           '  </head>','  <body>','    <h1>PRE-TRAINING CURATION SANITY CHECK</h1>',
           '    <p>No decisions or datasets were changed. Review direction independently of numeric integrity.</p>',
           '    <pre>'+escape(json.dumps(summary,indent=2))+'</pre>']
    for title,selected in [('TEST EXCLUSIONS',[r for r in rows if r['excluded_split']=='test']),
                            ('POLICY MISMATCHES',[r for r in rows if r['policy_status']!='OK']),('ALL EXCLUSIONS',rows)]:
        parts.extend(['    <section>','      <h2>'+title+'</h2>'])
        for r in selected:
            parts.extend(['      <article>','        <h3>'+escape(r['review_id'])+'</h3>',
                          '        <p>EXCLUDED: '+escape(r['excluded_image'])+'\nRETAINED: '+escape(r['retained_pair_image'])+'</p>',
                          '        <p class="warning">'+escape(r['direction_status']+' / '+r['policy_status']+' — '+r['explanation'])+'</p>',
                          '        <p>Decision: '+escape(r['human_decision'])+' | Notes: '+escape(r['notes'] or '(empty)')+'</p>',
                          '        <pre>.venv\\Scripts\\python.exe scripts/review_cleaning_cases.py --review-id '+escape(r['review_id'])+' --open-image</pre>'])
            if r['visualization_path']:
                src=escape(r['visualization_path'],quote=True)
                parts.append(f'        <a href="{src}"><img loading="lazy" src="{src}" alt="{escape(r["review_id"])} policy direction"></a>')
            parts.append('      </article>')
        parts.append('    </section>')
    parts.extend(['  </body>','</html>',''])
    return '\n'.join(parts)


def generate_policy_report(root=PROJECT_ROOT):
    from .curation import raw_snapshot, verify_curated_fingerprint, locations
    root=Path(root).resolve();raw=raw_snapshot(root);curated=verify_curated_fingerprint(root)
    _,_,p=locations(root)
    protected={n:(p/n).read_bytes() for n in ('review_queue.csv','review_decisions.csv','curated_v1_manifest.csv','curated_v1_exclusions.csv','curated_v1_summary.json')}
    rows,summary=analyze_policy(root)
    boxes=defaultdict(list)
    for b in numeric_boxes(read_csv(root/'results/dataset_audit/bounding_boxes.csv')):boxes[b['image']].append(b)
    queue={r['review_id']:r for r in read_csv(p/'review_queue.csv')}
    out=p/'pretraining_review';out.mkdir(exist_ok=True)
    for r in rows:
        if r['policy_status']=='OK':continue
        case=queue[r['review_id']];panels=[]
        for side in ('a','b'):
            name=case['image_'+side];split=case['split_'+side]
            label='EXCLUDED' if name==r['excluded_image'] else 'RETAINED'
            title=f"{r['review_id']} | {side.upper()} | {split.upper()} | priority {SPLIT_PRIORITY[split]} | {label}\n{Path(name).name}\npHash distance={r['phash_distance']} | human decision={r['human_decision']}\n{r['direction_status']} / {r['policy_status']} | Ground Truth only"
            panels.append(panel(draw_annotation(root/name,boxes[name]),title))
        path=out/(r['review_id']+'.png');join_panels(panels).save(path)
        r['visualization_path']=path.relative_to(p).as_posix()
    if raw_snapshot(root)!=raw or verify_curated_fingerprint(root)!=curated:raise ValueError('Dataset changed during sanity check')
    if any((p/n).read_bytes()!=data for n,data in protected.items()):raise ValueError('Review/build evidence changed during check')
    write_csv(p/'pretraining_curation_sanity.csv',rows,FIELDS)
    write_csv(p/'test_exclusion_review.csv',test_exclusion_rows(rows),TEST_FIELDS)
    summary.update(raw_immutability='PASS',curated_fingerprint_unchanged='PASS',curated_dataset_modified=False)
    (p/'pretraining_curation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (p/'pretraining_curation_report.html').write_text(policy_html(rows,summary),encoding='utf-8')
    return summary
