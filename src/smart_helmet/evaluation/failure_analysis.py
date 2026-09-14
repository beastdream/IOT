"""Fixed-threshold, one-to-one VALID diagnostics. Not framework AP or production tuning."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from smart_helmet.paths import PROJECT_ROOT
from smart_helmet.foundation import EXPECTED_CLASSES
from smart_helmet.dataset.audit import AREA_THRESHOLDS, parse_yolo_line, write_csv
from smart_helmet.dataset.duplicates import sha256_file
from smart_helmet.training.experiment import write_json
from .evaluator import preflight, protection, check_model, verify_unchanged

CONFIDENCE = 0.25
PREDICTION_FLOOR = 0.001
MATCH_IOU = 0.5
TINY_AREA = AREA_THRESHOLDS[0]
FIELDS = ['image','error_type','ground_truth_class','predicted_class','confidence','iou','bbox_area',
          'tiny_object','notes','ground_truth_index','prediction_index','normalized_area']
FOLDERS = {'FALSE_NEGATIVE':'false_negatives','FALSE_POSITIVE':'false_positives',
           'CLASS_CONFUSION':'class_confusion','LOW_CONFIDENCE':'low_confidence',
           'TINY_OBJECT_FAILURE':'tiny_object_failures','MULTIPLE_OBJECT_SCENE':'multiple_object_scenes'}


def area(box):
    return max(0,box[2]-box[0])*max(0,box[3]-box[1])


def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=area(a)+area(b)-intersection
    return intersection/union if union else 0.0


def match_boxes(ground_truth, predictions, threshold=MATCH_IOU):
    """Confidence-ordered class-aware greedy matching; then wrong-class unmatched pairs.

    Every prediction/GT participates at most once in each stage. Correct class
    matches take precedence. Confusions remain FN for the GT class and FP for the
    predicted class; the confusion category is an additional diagnostic tag.
    """
    matched={};used=set()
    order=sorted(range(len(predictions)),key=lambda j:(-predictions[j]['confidence'],j))
    for j in order:
        candidates=[(iou(g['bbox'],predictions[j]['bbox']),i) for i,g in enumerate(ground_truth)
                    if i not in matched and g['class_id']==predictions[j]['class_id']]
        if candidates:
            score,i=max(candidates,key=lambda x:(x[0],-x[1]))
            if score>=threshold: matched[i]=(j,score);used.add(j)
    missed=[i for i in range(len(ground_truth)) if i not in matched]
    extra=[j for j in range(len(predictions)) if j not in used]
    edges=sorted([(iou(ground_truth[i]['bbox'],predictions[j]['bbox']),i,j) for i in missed for j in extra
                  if ground_truth[i]['class_id']!=predictions[j]['class_id']],reverse=True)
    confusions=[];taken_g=set();taken_p=set()
    for score,i,j in edges:
        if score>=threshold and i not in taken_g and j not in taken_p:
            confusions.append((i,j,score));taken_g.add(i);taken_p.add(j)
    return dict(matches=matched,missed=missed,extra=extra,confusions=confusions)


def analyze_image(image, gt, predictions, width, height):
    active=[dict(p,source_index=j) for j,p in enumerate(predictions) if p['confidence']>=CONFIDENCE]
    low=[dict(p,source_index=j) for j,p in enumerate(predictions) if PREDICTION_FLOOR<=p['confidence']<CONFIDENCE]
    match=match_boxes(gt,active)
    low_match=match_boxes([gt[i] for i in match['missed']],low)['matches']
    low_by_gt={match['missed'][i]:(low[j],score) for i,(j,score) in low_match.items()}
    confusion_g={i:(j,score) for i,j,score in match['confusions']}
    confusion_p={j:(i,score) for i,j,score in match['confusions']}
    events=[]
    def event(kind,i=None,p=None,score=0,notes=''):
        box=p['bbox'] if kind=='FALSE_POSITIVE' and p else gt[i]['bbox'] if i is not None else p['bbox'] if p else [0,0,0,0]
        pixels=area(box);normalized=pixels/(width*height)
        events.append(dict(image=image,error_type=kind,ground_truth_class=EXPECTED_CLASSES[gt[i]['class_id']] if i is not None else '',
            predicted_class=EXPECTED_CLASSES[p['class_id']] if p else '',confidence=p['confidence'] if p else '',
            iou=score,bbox_area=pixels,normalized_area=normalized,tiny_object=bool(pixels and normalized<TINY_AREA),notes=notes,
            ground_truth_index=i if i is not None else '',prediction_index=p['source_index'] if p else ''))
    for i in match['missed']:
        candidate=active[confusion_g[i][0]] if i in confusion_g else None
        score=confusion_g[i][1] if i in confusion_g else max((iou(gt[i]['bbox'],p['bbox']) for p in active),default=0)
        flags=[]
        if i in confusion_g:flags.append('Wrong-class overlap; also counted as CLASS_CONFUSION and predicted-class FP.')
        if i in low_by_gt:flags.append('Same-class candidate below diagnostic confidence threshold.')
        if len(gt)>1:flags.append('Multiple GT objects; no causal inference.')
        event('FALSE_NEGATIVE',i,candidate,score,' '.join(flags)+' Manual review required.')
        if area(gt[i]['bbox'])/(width*height)<TINY_AREA:
            event('TINY_OBJECT_FAILURE',i,candidate,score,'Missed GT below audited normalized-area threshold; overlaps FN category.')
        if i in low_by_gt:
            p,score=low_by_gt[i]
            event('LOW_CONFIDENCE',i,p,score,'Below 0.25; same-class IoU >=0.5 candidate for this FN. Diagnostic only.')
    for j in match['extra']:
        linked=confusion_p.get(j)
        event('FALSE_POSITIVE',linked[0] if linked else None,active[j],linked[1] if linked else max((iou(g['bbox'],active[j]['bbox']) for g in gt),default=0),
              'Wrong-class pair; also counted as confusion and GT-class FN.' if linked else 'Unmatched active prediction, including duplicate/localization errors.')
    for i,j,score in match['confusions']:
        event('CLASS_CONFUSION',i,active[j],score,'IoU >=0.5, wrong class. Counts as one FN and one FP, not a TP.')
    for i,(j,score) in match['matches'].items():
        if active[j]['confidence']<0.5:
            event('LOW_CONFIDENCE',i,active[j],score,'Correct match with confidence in [0.25,0.50); not a false negative.')
    if len(gt)>1:event('MULTIPLE_OBJECT_SCENE',notes=f'{len(gt)} GT objects. Scene context; not itself an error.')
    per_class={}
    for c in EXPECTED_CLASSES:
        total=sum(g['class_id']==c for g in gt)
        tp=sum(gt[i]['class_id']==c for i in match['matches'])
        fn=total-tp;fp=sum(active[j]['class_id']==c for j in match['extra'])
        per_class[str(c)]=dict(ground_truth_count=total,true_positive_count=tp,false_negative_count=fn,false_positive_count=fp)
    return dict(image=image,ground_truth=gt,predictions=predictions,events=events,per_class=per_class,
                matched_gt=sorted(match['matches']),missed_gt=match['missed'],low_confidence_missed_gt=sorted(low_by_gt),
                width=width,height=height)


def ground_truth(path):
    with Image.open(path) as source: width,height=source.size
    label=path.parent.parent/'labels'/(path.stem+'.txt')
    boxes=[]
    for line in label.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():continue
        b,issues=parse_yolo_line(line)
        if b is None or any(x[1]=='ERROR' for x in issues):raise ValueError(f'Invalid GT: {label}')
        x,y,w,h=b['x'],b['y'],b['normalized_width'],b['normalized_height']
        boxes.append(dict(class_id=b['class_id'],bbox=[(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height]))
    return boxes,width,height


def render_case(case, root, target):
    """GT and predictions in separate panels; legend includes all drawn box IDs."""
    with Image.open(Path(root)/case['image']) as image: source=image.convert('RGB')
    low_ids={e['prediction_index'] for e in case['events'] if e['error_type']=='LOW_CONFIDENCE'}
    predictions=[(j,p) for j,p in enumerate(case['predictions']) if p['confidence']>=CONFIDENCE or j in low_ids]
    w,h=source.size
    try:font=ImageFont.truetype('arial.ttf',12)
    except OSError:font=ImageFont.load_default()
    count=max(len(case['ground_truth']),len(predictions),1)
    canvas=Image.new('RGB',(w*2+20,h+80+count*17),'#151c25')
    canvas.paste(source,(0,40));canvas.paste(source,(w+20,40));draw=ImageDraw.Draw(canvas)
    draw.text((5,6),'GROUND TRUTH (green)',font=font,fill='#54ed9a')
    draw.text((w+25,6),'PREDICTIONS (orange); low-confidence candidates (yellow)',font=font,fill='#ffb35c')
    for panel,boxes,color in [(0,list(enumerate(case['ground_truth'])),'#00d878'),(w+20,predictions,'#ff8300')]:
        for n,(j,b) in enumerate(boxes):
            x1,y1,x2,y2=b['bbox'];chosen='#ffe766' if b.get('confidence',1)<CONFIDENCE else color
            draw.rectangle((x1+panel,y1+40,x2+panel,y2+40),outline=chosen,width=2)
            draw.text((max(panel,x1+panel),max(40,y1+40)),str(j),font=font,fill='black',stroke_width=1,stroke_fill=chosen)
            label=f'{j}: {EXPECTED_CLASSES[b["class_id"]]}'
            if 'confidence' in b:label+=f' | conf={b["confidence"]:.3f}'
            draw.text((panel+4,h+47+n*17),label,font=font,fill=chosen)
    draw.text((4,canvas.height-24),'VALID only | diagnostic conf >=0.25 | matching IoU >=0.5 | GT coordinates unchanged',font=font,fill='white')
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True);canvas.save(target)


def select_samples(cases,limit=20):
    selected=[]
    predicates=[lambda c:1 in [c['ground_truth'][i]['class_id'] for i in c['matched_gt']],
                lambda c:0 in [c['ground_truth'][i]['class_id'] for i in c['matched_gt']],
                lambda c:bool(c['missed_gt']),lambda c:len(c['ground_truth'])>1,
                lambda c:not c['missed_gt'] and not any(e['error_type']=='FALSE_POSITIVE' for e in c['events']),
                lambda c:bool(c['low_confidence_missed_gt'])]
    # Round-robin strata, unique images, deterministic source order.
    pools=[[c for c in cases if predicate(c)] for predicate in predicates]
    while len(selected)<min(limit,len(cases)):
        changed=False
        for pool in pools+[cases]:
            candidate=next((c for c in pool if c['image'] not in {s['image'] for s in selected}),None)
            if candidate:
                selected.append(candidate);changed=True
                if len(selected)==min(limit,len(cases)):break
        if not changed:break
    return selected


def aggregate(cases):
    counts=Counter(e['error_type'] for c in cases for e in c['events'])
    totals={str(k):{field:sum(c['per_class'][str(k)][field] for c in cases)
                   for field in ('ground_truth_count','true_positive_count','false_negative_count','false_positive_count')} for k in EXPECTED_CLASSES}
    for value in totals.values():
        tp,fp,fn=(value[k] for k in ('true_positive_count','false_positive_count','false_negative_count'))
        value.update(precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn) if tp+fn else 0)
    wh=totals['1'].copy();tiny=low=multiple=0
    for c in cases:
        for i in c['missed_gt']:
            if c['ground_truth'][i]['class_id']!=1:continue
            tiny+=area(c['ground_truth'][i]['bbox'])/(c['width']*c['height'])<TINY_AREA
            low+=i in c['low_confidence_missed_gt'];multiple+=len(c['ground_truth'])>1
    wh.update(false_negative_groups=dict(tiny_bbox=tiny,low_confidence=low,multiple_objects=multiple,occlusion_candidate=None),
              group_counts_overlap=True,manual_review_required=bool(wh['false_negative_count']),
              note='Occlusion is not inferred from geometry. Groups are associations, not established causes.',
              confidence_threshold=CONFIDENCE,matching_iou=MATCH_IOU)
    confidence={label:0 for label in ('<0.25','0.25-0.50','0.50-0.75','>=0.75')}
    for c in cases:
        for p in c['predictions']:
            value=p['confidence'];confidence['<0.25' if value<.25 else '0.25-0.50' if value<.5 else '0.50-0.75' if value<.75 else '>=0.75']+=1
    return dict(validation_images=len(cases),counts={k:counts[k] for k in FOLDERS},per_class=totals,
                without_helmet=wh,confidence_distribution=confidence,
                matching_policy='Confidence-ordered, class-aware, one-to-one greedy IoU >=0.5; unmatched wrong-class overlaps tagged as confusion and remain FN+FP.',
                confidence_threshold=CONFIDENCE,prediction_floor=PREDICTION_FLOOR,nms_iou=0.7,max_det=300,
                tiny_normalized_area_threshold=TINY_AREA,category_counts_overlap=True,test_status='LOCKED / NOT USED')


def write_report(out, metrics, training, summary):
    wh=summary['without_helmet'];c=summary['counts'];overall=metrics['overall'];trends=training['trends']
    table=['| Class | Precision | Recall | AP50 | AP50-95 |','|---|---:|---:|---:|---:|']
    for v in metrics['per_class'].values():table.append(f'| {v["class_name"]} | {v["precision"]:.4f} | {v["recall"]:.4f} | {v["mAP50"]:.4f} | {v["mAP50-95"]:.4f} |')
    recommendations=[]
    if wh['false_negative_count']:
        recommendations.append(f'OBSERVED PROBLEM: {wh["false_negative_count"]}/{wh["ground_truth_count"]} Without Helmet GT objects missed at diagnostic confidence 0.25. PROPOSED EXPERIMENT: manually review the missed-object gallery, then test targeted TRAIN-only additions or class-aware sampling if the review confirms coverage gaps. No causal claim or guaranteed improvement.')
    if wh['false_negative_groups']['tiny_bbox']:
        recommendations.append(f'OBSERVED PROBLEM: {wh["false_negative_groups"]["tiny_bbox"]} Without Helmet misses have normalized area <{TINY_AREA}. PROPOSED EXPERIMENT: compare 416 vs 640 on VALID with the same split and seed, measuring recall and runtime; do not change TEST.')
    else:recommendations.append('OBSERVED PROBLEM: no Without Helmet misses meet the audited tiny-area cutoff. Higher resolution is not justified specifically by tiny-object evidence in this run; review small objects before proposing it.')
    if wh['false_negative_groups']['low_confidence']:
        recommendations.append(f'OBSERVED PROBLEM: {wh["false_negative_groups"]["low_confidence"]} misses have same-class low-confidence overlapping candidates. PROPOSED EXPERIMENT: review confidence calibration and image appearance on VALID; test one evidence-supported TRAIN augmentation at a time. Keep the production threshold unchanged.')
    if c['CLASS_CONFUSION']:
        recommendations.append(f'OBSERVED PROBLEM: {c["CLASS_CONFUSION"]} wrong-class overlapping pairs. PROPOSED EXPERIMENT: review their GT and visible head regions for label ambiguity or missing TRAIN coverage before changing training.')
    lines=['# Baseline validation evaluation',
        f'Model: `{metrics["model"]}` (SHA-256 `{metrics["checkpoint_sha256"]}`).',
        f'Dataset: curated_v1; fingerprint `{metrics["dataset_fingerprint"]}`; {metrics["validation_images"]} VALID images.',
        f'Best epoch: **{training["best_epoch"]}**; checkpoint/CSV agreement: {training["checkpoint_matches_best_csv"]}; completed {training["epochs_completed"]}/{training["epochs_requested"]}; early stopping consistent with patience {training["patience"]}: {training["early_stopping_consistent"]}.',
        f'Training-loop duration: {training["training_loop_seconds"]/3600:.3f} hours; whole pipeline including setup/final validation/fingerprints: {training["pipeline_duration_seconds"]/3600:.3f} hours.',
        '## Validation metrics',
        'Overall: '+', '.join(f'{k}={v:.4f}' for k,v in overall.items())+'.', '\n'.join(table),
        'Precision/recall above use the framework-selected operating point; AP integrates confidence rankings. They are not the fixed-threshold diagnostic metrics below.',
        'Independent comparison against final training metrics: '+json.dumps(metrics['training_comparison']),
        metrics.get('precision_policy','FP32 evaluation.'),
        '## Training curves', '[Training curves](training_curves.png); complete epoch values and first/last-five loss summaries in [training_summary.json](training_summary.json).',
        '\n'.join(f'- {k}: first-five mean {v["first_five_mean"]:.4f}, last-five mean {v["last_five_mean"]:.4f}; best-epoch {v["best_epoch_value"]:.4f}, final {v["last"]:.4f}.' for k,v in trends.items()),
        'Lower training losses alone do not establish better generalization. Compare VALID losses and AP before diagnosing overfitting.',
        '## Strengths and weaknesses',
        f'With Helmet recall={metrics["per_class"]["0"]["recall"]:.3f}; Without Helmet recall={metrics["per_class"]["1"]["recall"]:.3f}. The class difference is observed on VALID, not a statement about TEST performance.',
        f'The gap between mAP50 ({overall["mAP50"]:.3f}) and mAP50-95 ({overall["mAP50-95"]:.3f}) motivates review of localization at stricter IoUs; it does not identify a semantic cause.',
        '## Fixed-threshold failure analysis',
        summary['matching_policy'],
        f'Diagnostic confidence >=0.25; inference floor 0.001, class-aware NMS IoU 0.7, max_det=300 per image. Tiny means bbox area/image area <{TINY_AREA}, reusing the dataset audit definition. Pixel bbox area and normalized area are stored separately: FP rows use the predicted box, GT-associated rows use the GT box.',
        'Counts overlap: CLASS_CONFUSION is included in FN and FP. TINY_OBJECT_FAILURE tags missed tiny GT. MULTIPLE_OBJECT_SCENE describes context, not an error. LOW_CONFIDENCE denotes correct matches in [0.25,0.5), or one-to-one same-class candidates below 0.25 for missed GT.',
        json.dumps(c),
        '## Without Helmet',
        f'GT={wh["ground_truth_count"]}; TP={wh["true_positive_count"]}; FN={wh["false_negative_count"]}; FP={wh["false_positive_count"]}; diagnostic precision={wh["precision"]:.4f}; recall={wh["recall"]:.4f}.',
        'Missed-object associations (overlapping): '+json.dumps(wh['false_negative_groups'])+'. Occlusion remains unassigned; manual review required for uncertain causes.',
        '[Without Helmet missed gallery](failures/without_helmet_missed/) | [All failure rows](failure_cases.csv) | [20 validation samples](sample_predictions/)',
        '## Confidence distribution',json.dumps(summary['confidence_distribution']),
        'Counts cover post-NMS predictions retained above 0.001, capped at 300 per image. They are not all raw proposals. No production threshold has been changed.',
        '## Next experiments',*recommendations,
        'Recommendations are proposals only. Evaluate one change at a time on the unchanged VALID split, recording the same metric definitions and compute cost. No Experiment B was trained.',
        '## Framework plots',
        '[Confusion matrix](confusion_matrix.png) | [Normalized matrix](confusion_matrix_normalized.png) | [PR](PR_curve.png) | [F1](F1_curve.png) | [Precision-confidence](P_curve.png) | [Recall-confidence](R_curve.png)',
        'The framework confusion matrix may use its own confidence/IoU filtering, so its cells need not equal the diagnostic counts. Original Box-prefixed curve files are preserved; conventional names are byte-identical aliases.',
        'Metric API reference: [Ultralytics validation](https://docs.ultralytics.com/modes/val/).',
        '**TEST STATUS: LOCKED / NOT USED.** Raw/curated/training artifact immutability is recorded in `evaluation_status.json` and `failure_analysis_status.json`.']
    (Path(out)/'baseline_evaluation_report.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')


def analyze(root=PROJECT_ROOT,split='val',device='auto',overwrite=False,model_factory=None):
    state=preflight(root,split,device);out=state['output']
    metrics=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
    evaluation_status=json.loads((out/'evaluation_status.json').read_text(encoding='utf-8'))
    if evaluation_status['status']!='PASS' or metrics['checkpoint_sha256']!=sha256_file(state['checkpoint']) or metrics['dataset_fingerprint']!=state['snapshot']['curated_hash']:
        raise ValueError('Run current best.pt validation successfully before failure analysis')
    owned=['failures','sample_predictions','failure_cases.csv','without_helmet_analysis.json','failure_summary.json',
           'validation_predictions.json','sample_manifest.json','baseline_evaluation_report.md','failure_analysis_status.json']
    existing=[out/name for name in owned if (out/name).exists()]
    if existing:
        if not overwrite:raise ValueError('Failure artifacts already exist. Use --overwrite to archive them.')
        archive=out/('analysis_archive_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'));archive.mkdir()
        for path in existing:
            if path.resolve().parent!=out or path.is_symlink():raise ValueError('Unsafe analysis archive target')
            path.rename(archive/path.name)
    status=dict(status='RUNNING',test_status='LOCKED / NOT USED')
    try:
        if model_factory is None:
            from ultralytics import YOLO
            model_factory=YOLO
        cases=[];seen=set();allowed={p.resolve() for p in state['images']}
        with protection(state):
            model=model_factory(str(state['checkpoint']));check_model(model)
            predictions=model.predict(source=[str(p) for p in state['images']],imgsz=416,batch=4,workers=0,
                device=state['environment']['device'],conf=PREDICTION_FLOOR,iou=.7,max_det=300,
                quantize=32,
                rect=True,augment=False,save=False,verbose=False,stream=True)
            for result in predictions:
                path=Path(result.path).resolve()
                if path not in allowed or path in seen:raise ValueError('Unexpected/duplicate inference image; VALID only')
                seen.add(path)
                gt,w,h=ground_truth(path)
                boxes=result.boxes
                pred=[dict(class_id=int(c),confidence=float(v),bbox=[float(x) for x in b])
                      for b,c,v in zip(boxes.xyxy.cpu().tolist(),boxes.cls.cpu().tolist(),boxes.conf.cpu().tolist())]
                if any(p['class_id'] not in EXPECTED_CLASSES for p in pred):raise ValueError('Unexpected prediction class')
                cases.append(analyze_image(path.relative_to(state['root']).as_posix(),gt,pred,w,h))
        if seen!=allowed:raise ValueError('Inference did not cover every VALID image')
        cases.sort(key=lambda c:c['image'])
        summary=aggregate(cases)
        for folder in [*FOLDERS.values(),'without_helmet_missed']:(out/'failures'/folder).mkdir(parents=True,exist_ok=True)
        for case in cases:
            folders={FOLDERS[e['error_type']] for e in case['events']}
            if any(case['ground_truth'][i]['class_id']==1 for i in case['missed_gt']):folders.add('without_helmet_missed')
            for folder in sorted(folders):render_case(case,state['root'],out/'failures'/folder/(Path(case['image']).stem+'.png'))
        samples=select_samples(cases)
        for case in samples:render_case(case,state['root'],out/'sample_predictions'/(Path(case['image']).stem+'.png'))
        write_json(out/'sample_manifest.json',[dict(image=c['image'],categories=sorted({e['error_type'] for e in c['events']}),
                   matched_classes=sorted({c['ground_truth'][i]['class_id'] for i in c['matched_gt']})) for c in samples])
        write_json(out/'validation_predictions.json',dict(checkpoint_sha256=metrics['checkpoint_sha256'],dataset_fingerprint=metrics['dataset_fingerprint'],cases=cases))
        write_csv(out/'failure_cases.csv',[e for c in cases for e in c['events']],FIELDS)
        write_json(out/'without_helmet_analysis.json',summary['without_helmet'])
        write_json(out/'failure_summary.json',summary)
        training=json.loads((out/'training_summary.json').read_text(encoding='utf-8'))
        write_report(out,metrics,training,summary)
        status.update(status='PASS',validation_images=len(cases),samples=len(samples))
        print(json.dumps(summary,indent=2))
    except BaseException as error:
        status.update(status='FAIL',error=str(error));raise
    finally:
        try:status['immutability']=verify_unchanged(state)
        except BaseException as error:
            status.update(status='FAIL',immutability_error=str(error));raise
        finally:write_json(out/'failure_analysis_status.json',status)
    print('BASELINE FAILURE ANALYSIS: PASS\nTEST: LOCKED / NOT USED')
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description='Diagnostic baseline failures on VALID only; no threshold tuning.')
    parser.add_argument('--split',default='val');parser.add_argument('--device',default='auto');parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args(argv)
    try:analyze(split=args.split,device=args.device,overwrite=args.overwrite);return 0
    except (OSError,ValueError,RuntimeError,KeyError) as error:
        print(f'BASELINE FAILURE ANALYSIS: FAIL\n{error}');return 1
