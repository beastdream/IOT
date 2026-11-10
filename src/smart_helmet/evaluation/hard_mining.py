"""Phase 4I: frozen A TRAIN inference and deterministic difficulty evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from smart_helmet.paths import PROJECT_ROOT
from smart_helmet.foundation import EXPECTED_CLASSES
from smart_helmet.dataset.audit import write_csv
from smart_helmet.dataset.duplicates import sha256_file
from smart_helmet.training.environment import verify_dataset, check_unchanged, runtime_info
from smart_helmet.training.experiment import write_json
from smart_helmet.training.sampling import records, train_path
from smart_helmet.training.trainer import frozen_cache
from .comparison import fingerprint, checkpoint_fingerprint
from .evaluator import check_model
from .failure_analysis import ground_truth, iou, match_boxes

OUT = Path('results/hard_example_mining/A_baseline_train')
CATEGORIES = ('EASY_CORRECT','LOW_CONFIDENCE_CORRECT','CLASS_CONFUSION',
              'MISSED_NO_CORRECT_CLASS_PROPOSAL','LOCALIZATION_FAILURE','MATCH_COMPETITION')
TIERS = ('TIER_0_EASY','TIER_1_MODERATE','TIER_2_HARD','TIER_3_CRITICAL')
FLOOR = .001
CONF = .25
LOCAL_IOU = .1


def guard(split='train', test_status='LOCKED'):
    if split != 'train' or test_status != 'LOCKED':
        raise ValueError('TRAIN ONLY mining; VALID rejected; TEST LOCKED')


def stable_id(image, index=None):
    key = image if index is None else image+':'+str(index)
    return hashlib.sha256(key.encode()).hexdigest()[:20]


def predict_batches(model, images, args):
    # Ultralytics treats a Python list as LoadPilAndNumpy with one source batch;
    # explicit chunks bound device memory regardless of the batch argument.
    for start in range(0,len(images),4):
        yield from model.predict(source=[str(p) for p in images[start:start+4]],**args)


def classify(case):
    """One-to-one correct matches at floor, then wrong class, then localization.

    Low correct matches take precedence over wrong-class proposals. Active wrong
    overlap remains a separate context flag. Diagnostic geometry never counts as
    a TP. A shared proposal consumed elsewhere produces MATCH_COMPETITION.
    """
    gt, pred = case['ground_truth'], case['predictions']
    if any(p['class_id'] not in EXPECTED_CLASSES or not FLOOR <= p['confidence'] <= 1 for p in pred):
        raise ValueError('Invalid proposal class/confidence')
    match = match_boxes(gt,pred)
    assigned = dict(match['matches'])
    categories = {i:('EASY_CORRECT' if pred[j]['confidence'] >= CONF else 'LOW_CONFIDENCE_CORRECT')
                  for i,(j,_) in assigned.items()}
    for i,j,score in match['confusions']:
        assigned[i]=(j,score); categories[i]='CLASS_CONFUSION'
    used={j for j,_ in assigned.values()}
    # Greedy descending confidence, then best overlap/lowest GT index.
    for j in sorted(range(len(pred)),key=lambda j:(-pred[j]['confidence'],j)):
        if j in used:
            continue
        options=[(iou(g['bbox'],pred[j]['bbox']),i) for i,g in enumerate(gt)
                 if i not in assigned and g['class_id']==pred[j]['class_id']]
        if options:
            score,i=max(options,key=lambda x:(x[0],-x[1]))
            if LOCAL_IOU <= score < .5:
                assigned[i]=(j,score); categories[i]='LOCALIZATION_FAILURE'; used.add(j)
    duplicate = Counter()
    duplicate_ids = {}
    # Each extra active prediction has at most one duplicate-context GT owner.
    for j,p in enumerate(pred):
        if j in used or p['confidence'] < CONF:
            continue
        candidates=[(iou(g['bbox'],p['bbox']),i) for i,g in enumerate(gt)
                    if i in match['matches'] and g['class_id']==p['class_id']]
        if candidates:
            score,i=max(candidates,key=lambda x:(x[0],-x[1]))
            if score >= .5:
                duplicate[i]+=1; duplicate_ids[j]=i
    objects=[]
    for i,g in enumerate(gt):
        same=[(iou(g['bbox'],p['bbox']),j) for j,p in enumerate(pred) if p['class_id']==g['class_id']]
        best=max(same,default=(0,None),key=lambda x:(x[0],-x[1] if x[1] is not None else 0))
        if i not in categories:
            categories[i]='MATCH_COMPETITION' if best[0]>=LOCAL_IOU else 'MISSED_NO_CORRECT_CLASS_PROPOSAL'
        link=assigned.get(i)
        p=pred[link[0]] if link else None
        objects.append(dict(object_id=stable_id(case['image'],i),image=case['image'],gt_index=i,
            class_id=g['class_id'],class_name=EXPECTED_CLASSES[g['class_id']],gt_bbox=g['bbox'],
            category=categories[i],hard=categories[i]!='EASY_CORRECT' or duplicate[i]>0,
            prediction_index=link[0] if link else None,predicted_class=p['class_id'] if p else None,
            confidence=p['confidence'] if p else None,iou=link[1] if link else None,
            correct_class_overlap_at_floor=best[0]>=.5,meaningful_correct_class_overlap=best[0]>=LOCAL_IOU,
            best_correct_class_iou=best[0],best_correct_class_prediction_index=best[1],
            active_wrong_class_overlap=any(p['class_id']!=g['class_id'] and p['confidence']>=CONF
                                          and iou(g['bbox'],p['bbox'])>=.5 for p in pred),
            duplicate_prediction_count=duplicate[i],duplicate_context='DUPLICATE_PREDICTION_CONTEXT' if duplicate[i] else ''))
    return objects


def image_row(case, objects):
    counts=Counter(r['category'] for r in objects)
    if counts['MISSED_NO_CORRECT_CLASS_PROPOSAL']:
        tier=TIERS[3]
    elif counts['CLASS_CONFUSION'] or counts['MATCH_COMPETITION']:
        tier=TIERS[2]
    elif any(r['hard'] for r in objects):
        tier=TIERS[1]
    else:
        tier=TIERS[0]
    classes=Counter(r['class_id'] for r in objects)
    return dict(image_id=stable_id(case['image']),image=case['image'],gt_object_count=len(objects),
                with_helmet_gt_count=classes[0],without_helmet_gt_count=classes[1],
                hard_object_count=sum(r['hard'] for r in objects),
                hard_without_helmet_count=sum(r['hard'] and r['class_id']==1 for r in objects),
                class_confusion_count=counts['CLASS_CONFUSION'],low_confidence_correct_count=counts['LOW_CONFIDENCE_CORRECT'],
                no_proposal_miss_count=counts['MISSED_NO_CORRECT_CLASS_PROPOSAL'],
                localization_failure_count=counts['LOCALIZATION_FAILURE'],match_competition_count=counts['MATCH_COMPETITION'],
                duplicate_context_count=sum(bool(r['duplicate_prediction_count']) for r in objects),
                crowded=len(objects)>=3,multi_class=bool(classes[0] and classes[1]),tier=tier,
                without_helmet_priority=any(r['hard'] and r['class_id']==1 for r in objects))


def analyze(cases):
    objects,images=[],[]
    for case in sorted(cases,key=lambda c:c['image']):
        oo=classify(case); objects.extend(oo); images.append(image_row(case,oo))
    summary=dict(train_images=len(images),gt_objects=len(objects),
                 class_counts={name:sum(o['class_id']==cid for o in objects) for cid,name in EXPECTED_CLASSES.items()},
                 tier_counts={t:sum(r['tier']==t for r in images) for t in TIERS},
                 without_helmet_difficulty={cat:sum(r['class_id']==1 and r['category']==cat for r in objects) for cat in CATEGORIES},
                 all_object_difficulty={cat:sum(r['category']==cat for r in objects) for cat in CATEGORIES},
                 without_helmet_hard_images=sum(r['hard_without_helmet_count']>0 for r in images),
                 critical_no_proposal_images=sum(r['tier']==TIERS[3] for r in images),
                 valid_images_used=0,test_images_used=0,mining_source='TRAIN ONLY')
    return objects,images,summary


def review_sample(images,objects):
    # Bounded targeted queue; no automatic semantic approval.
    chosen={}
    for tier,limit in [(TIERS[3],30),(TIERS[2],15),(TIERS[1],8)]:
        pool=sorted([r for r in images if r['tier']==tier],
                    key=lambda r:(-int(r['without_helmet_priority']),-r['hard_without_helmet_count'],r['image']))
        for row in pool[:limit]: chosen[row['image']]=row
    # Ensure gallery/queue covers each observed reason and crowded hard scenes.
    for category in CATEGORIES[1:]:
        candidates=sorted({o['image'] for o in objects if o['category']==category})
        for name in candidates[:2]: chosen[name]=next(r for r in images if r['image']==name)
    for row in sorted([r for r in images if r['crowded'] and r['hard_object_count']],key=lambda r:r['image'])[:3]:
        chosen[row['image']]=row
    return list(chosen.values())


def render_gallery(root,out,cases,images,objects):
    from PIL import Image,ImageDraw,ImageFont
    import html
    directory=out/'hard_example_gallery'; directory.mkdir(exist_ok=True)
    indexed={c['image']:c for c in cases}
    queue=[]; cards=[]
    for row in review_sample(images,objects):
        case=indexed[row['image']]; oo=[r for r in objects if r['image']==row['image']]
        image=train_path(root,row['image'])
        with Image.open(image) as f: source=f.convert('RGB')
        w,h=source.size
        associated={r['prediction_index'] for r in oo if r['prediction_index'] is not None}
        shown=[(j,p) for j,p in enumerate(case['predictions']) if p['confidence']>=CONF or j in associated]
        lines=[f"GT {r['gt_index']}: {r['class_name']} | {r['category']} | pred={r['prediction_index']} conf={r['confidence']} IoU={r['iou']}" for r in oo]
        lines += [f"P{j}: {EXPECTED_CLASSES[p['class_id']]} conf={p['confidence']:.6f} max GT IoU={max((iou(g['bbox'],p['bbox']) for g in case['ground_truth']),default=0):.4f}" for j,p in shown]
        canvas=Image.new('RGB',(max(w*2+20,1000),h+80+len(lines)*19),'#17202b');draw=ImageDraw.Draw(canvas)
        try: font=ImageFont.truetype('arial.ttf',13)
        except OSError: font=ImageFont.load_default()
        canvas.paste(source,(0,35));canvas.paste(source,(w+20,35))
        draw.text((5,5),'TRAIN ground truth (green)',font=font,fill='white')
        draw.text((w+25,5),'A predictions: active orange / assigned low yellow',font=font,fill='white')
        for offset,items,color in [(0,list(enumerate(case['ground_truth'])),'#55ee99'),(w+20,shown,'#ffa43b')]:
            for j,b in items:
                x1,y1,x2,y2=b['bbox']; col='#ffeb50' if b.get('confidence',1)<CONF else color
                draw.rectangle((x1+offset,y1+35,x2+offset,y2+35),outline=col,width=2)
                draw.text((max(offset,x1+offset),max(35,y1+35)),str(j),font=font,fill=col)
        for n,line in enumerate(lines): draw.text((5,h+45+n*19),line,font=font,fill='white')
        target=directory/(row['image_id']+'.png'); canvas.save(target)
        reasons='|'.join(sorted({r['category'] for r in oo if r['category']!='EASY_CORRECT'} |
                               ({'DUPLICATE_PREDICTION_CONTEXT'} if any(r['duplicate_prediction_count'] for r in oo) else set())))
        queue.append(dict(review_id=row['image_id'],image=row['image'],tier=row['tier'],automatic_reason=reasons,
            gt_summary=json.dumps(case['ground_truth']),prediction_summary=json.dumps([dict(index=j,**p) for j,p in shown]),
            review_status='PENDING',notes='',gallery='hard_example_gallery/'+target.name))
        cards.append(f'<section><h2>{html.escape(row["tier"]+" "+row["image"])}</h2><p>{html.escape(reasons)}</p><a href="{target.name}"><img style="max-width:100%" src="{target.name}"></a></section>')
    (directory/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>TRAIN hard examples</title><h1>TRAIN ONLY — manual review PENDING</h1>'+''.join(cards),encoding='utf-8')
    write_csv(out/'manual_review_queue.csv',queue,list(queue[0]) if queue else ['review_id','image','tier','automatic_reason','gt_summary','prediction_summary','review_status','notes'])
    return queue


def run(root=PROJECT_ROOT,split='train'):
    guard(split)
    root=Path(root).resolve();out=root/OUT
    if out.resolve()!=out or out.exists() and any(out.iterdir()):
        raise ValueError('Mining output must be new, empty and not redirected')
    inventory=records(root)
    if len(inventory)!=1158: raise ValueError('Exactly 1158 curated TRAIN images required')
    images=[train_path(root,r['image']) for r in inventory]
    directory=root/'results/experiments/A_baseline'; checkpoint=directory/'weights/best.pt'
    metadata=json.loads((directory/'experiment.json').read_text())
    reference=json.loads((root/'results/evaluation/A_vs_B/provenance.json').read_text())
    digest=checkpoint_fingerprint(checkpoint)
    if digest!=reference['checkpoint_sha256']['A']: raise ValueError('Checkpoint provenance mismatch')
    frozen=fingerprint(directory)
    if frozen!=reference['training_artifact_hashes']['A']: raise ValueError('A artifacts changed')
    snapshot=verify_dataset(root)  # Hash-only integrity audit across splits, never mining input.
    if snapshot['curated_hash']!=metadata['dataset_fingerprint']: raise ValueError('Dataset provenance mismatch')
    env=runtime_info(reference['common_evaluation_args']['device'])
    for key in ('ultralytics_version','torch_version','cuda_version','device'):
        if env[key]!=reference['environment'][key]: raise ValueError('Model environment mismatch: '+key)
    out.mkdir(parents=True)
    args=dict(imgsz=416,batch=4,workers=0,device=env['device'],conf=FLOOR,iou=.7,max_det=300,
              quantize=32,augment=False,rect=True,save=False,verbose=False,stream=True)
    provenance=dict(status='RUNNING',reference_model='A_baseline',checkpoint=str(checkpoint),checkpoint_sha256=digest,
                    dataset_fingerprint=snapshot['curated_hash'],raw_fingerprint=snapshot['raw_hash'],environment=env,
                    inference_args=args,diagnostic_confidence=CONF,match_iou=.5,meaningful_overlap_iou=LOCAL_IOU,
                    train_manifest=[dict(r,image_sha256=sha256_file(p)) for r,p in zip(inventory,images)],
                    frozen_A_hashes=frozen,test_status='LOCKED',valid_images_used=0,test_images_used=0,
                    source_batching='Explicit four-image path chunks; final chunk may be smaller',
                    training_executed=False,integrity_scope='All-split byte hashing only; VALID/TEST never inference, mining or sampling inputs')
    write_json(out/'provenance.json',provenance)
    try:
        from ultralytics import YOLO
        cases=[];seen=set();allowed=set(images)
        with frozen_cache(out/'dataset_cache',[root/'data',root/'train',root/'valid',root/'test',directory]):
            model=YOLO(str(checkpoint));check_model(model)
            for result in predict_batches(model,images,args):
                path=train_path(root,Path(result.path))
                if path not in allowed or path in seen: raise ValueError('Foreign/duplicate TRAIN prediction')
                seen.add(path); gt,w,h=ground_truth(path);boxes=result.boxes
                predictions=[dict(class_id=int(c),confidence=float(v),bbox=list(map(float,b)))
                             for b,c,v in zip(boxes.xyxy.cpu().tolist(),boxes.cls.cpu().tolist(),boxes.conf.cpu().tolist())]
                cases.append(dict(image=path.relative_to(root).as_posix(),ground_truth=gt,predictions=predictions,width=w,height=h))
                if len(cases)%100==0: print(f'TRAIN inference: {len(cases)}/1158',flush=True)
        if seen!=allowed: raise ValueError('Incomplete TRAIN inference')
        cases.sort(key=lambda c:c['image'])
        write_json(out/'train_predictions.json',dict(checkpoint_sha256=digest,cases=cases))
        objects,rows,summary=analyze(cases)
        for filename,data in [('train_object_difficulty.csv',objects),('train_image_difficulty.csv',rows),
                              ('without_helmet_hard_examples.csv',[r for r in objects if r['class_id']==1 and r['hard']])]:
            write_csv(out/filename,data,list(data[0]) if data else list(objects[0]))
        queue=render_gallery(root,out,cases,rows,objects)
        summary.update(manual_review_required=bool(queue),manual_review_queue_size=len(queue),
                       status='REVIEW_REQUIRED' if queue else 'PASS',checkpoint_provenance='PASS',raw_integrity='PASS',curated_integrity='PASS')
        write_json(out/'hard_example_summary.json',summary)
        from .hard_mining_report import mining_report
        (out/'hard_example_report.md').write_text(mining_report(summary),encoding='utf-8')
        provenance.update(status=summary['status'],prediction_sha256=sha256_file(out/'train_predictions.json'),
                          artifact_sha256={f:sha256_file(out/f) for f in ('train_object_difficulty.csv','train_image_difficulty.csv','without_helmet_hard_examples.csv')})
    except BaseException as error:
        provenance.update(status='FAIL',error=str(error));raise
    finally:
        try:
            check_unchanged(snapshot,root)
            if fingerprint(directory)!=frozen: raise ValueError('A artifacts changed during mining')
            provenance.update(raw_integrity='PASS',curated_integrity='PASS',A_immutability='PASS')
        except BaseException as error:
            provenance.update(status='FAIL',error=str(error));raise
        finally: write_json(out/'provenance.json',provenance)
    return summary


def check(root=PROJECT_ROOT):
    root=Path(root);out=root/OUT
    p=json.loads((out/'provenance.json').read_text())
    if p['status'] not in ('PASS','REVIEW_REQUIRED'): raise ValueError('Mining incomplete/failed')
    guard('train',p['test_status'])
    if checkpoint_fingerprint(root/'results/experiments/A_baseline/weights/best.pt')!=p['checkpoint_sha256']:
        raise ValueError('Checkpoint provenance mismatch')
    if fingerprint(root/'results/experiments/A_baseline')!=p['frozen_A_hashes']: raise ValueError('A artifacts changed')
    snap=verify_dataset(root)
    if snap['curated_hash']!=p['dataset_fingerprint'] or snap['raw_hash']!=p['raw_fingerprint']:
        raise ValueError('Dataset fingerprint mismatch')
    if sha256_file(out/'train_predictions.json')!=p['prediction_sha256']: raise ValueError('Prediction cache changed')
    for name,digest in p['artifact_sha256'].items():
        if sha256_file(out/name)!=digest: raise ValueError('Mining artifact changed: '+name)
    cached=json.loads((out/'train_predictions.json').read_text())
    cases=cached['cases'];inventory=records(root)
    if cached['checkpoint_sha256']!=p['checkpoint_sha256'] or len(cases)!=1158 or len({c['image'] for c in cases})!=1158:
        raise ValueError('Incomplete mining/provenance')
    if {c['image'] for c in cases}!={r['image'] for r in inventory}: raise ValueError('TRAIN coverage mismatch')
    for c in cases:
        path=train_path(root,c['image']);gt,w,h=ground_truth(path)
        if (gt,w,h)!=(c['ground_truth'],c['width'],c['height']): raise ValueError('TRAIN GT mismatch')
    objects,rows,summary=analyze(cases)
    saved=json.loads((out/'hard_example_summary.json').read_text())
    if any(saved[k]!=v for k,v in summary.items()): raise ValueError('Mining reproducibility failed')
    import csv
    for name,expected in [('train_object_difficulty.csv',objects),('train_image_difficulty.csv',rows),
                         ('without_helmet_hard_examples.csv',[r for r in objects if r['class_id']==1 and r['hard']])]:
        with (out/name).open(newline='',encoding='utf-8-sig') as f: actual=list(csv.DictReader(f))
        if len(actual)!=len(expected) or any(any(a[k]!=('' if v is None else str(v)) for k,v in e.items()) for a,e in zip(actual,expected)):
            raise ValueError('Difficulty CSV does not reproduce')
    with (out/'manual_review_queue.csv').open(newline='',encoding='utf-8-sig') as f:queue=list(csv.DictReader(f))
    expected_review=review_sample(rows,objects)
    if {r['image'] for r in queue}!={r['image'] for r in expected_review} or len(queue)!=len(expected_review):
        raise ValueError('Targeted review queue coverage mismatch')
    for r in queue:
        train_path(root,r['image'])
        if r['review_id']!=stable_id(r['image']) or r['review_status'] not in ('PENDING','APPROVED','REJECTED','REVIEWED'):
            raise ValueError('Invalid review entry')
        gallery=(out/r['gallery']).resolve()
        if gallery.parent!=(out/'hard_example_gallery').resolve() or not gallery.is_file():raise ValueError('Missing/redirected gallery')
    print('='*40+'\nTRAIN HARD-EXAMPLE MINING\n'+'='*40)
    print('Reference model: A_baseline\nTRAIN images analyzed: 1158\nGT objects:',summary['gt_objects'])
    print('Class counts:',summary['class_counts']);print('Image tiers:',summary['tier_counts'])
    print('Without Helmet:',summary['without_helmet_difficulty'])
    print('Mining source: TRAIN ONLY\nVALID images used: 0\nTEST images used: 0\nCheckpoint provenance: PASS\nDataset fingerprint: PASS')
    print('AUTOMATIC MINING STATUS:',saved['status'])
    from .train_human_review import ReviewStore, summarize
    review=summarize(ReviewStore(root))
    print('HUMAN REVIEW GATE:', 'PASS' if review['ready'] else 'REVIEW_REQUIRED')
    print('READY TO REPLAN C:', 'YES' if review['ready'] else 'NO')
    return saved


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--split',default='train')
    args=parser.parse_args(argv);print(json.dumps(run(split=args.split),indent=2));return 0
