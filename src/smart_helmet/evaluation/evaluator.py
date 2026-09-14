"""Evaluate exactly A_baseline/best.pt on curated VALID. No TEST/model override."""
import argparse
import json
from pathlib import Path
import shutil

from smart_helmet.paths import PROJECT_ROOT
from smart_helmet.foundation import EXPECTED_CLASSES, IMAGE_EXTENSIONS
from smart_helmet.dataset.duplicates import sha256_file
from smart_helmet.training.environment import verify_dataset, check_unchanged, runtime_info
from smart_helmet.training.trainer import frozen_cache
from smart_helmet.training.experiment import prepare_output, write_json, write_yaml
from .metrics import serialize_metrics, save_metrics, compare_training, training_summary


def validation_guard(split):
    if split != 'val': raise ValueError('Phase 4B is VALIDATION ONLY; TEST is LOCKED')


def training_fingerprint(root):
    directory = Path(root)/'results/experiments/A_baseline'
    return {p.relative_to(directory).as_posix():sha256_file(p) for p in sorted(directory.rglob('*')) if p.is_file()}


def preflight(root=PROJECT_ROOT,split='val',device='auto'):
    validation_guard(split)
    root=Path(root).resolve()
    checkpoint=root/'results/experiments/A_baseline/weights/best.pt'
    if not checkpoint.is_file() or not checkpoint.stat().st_size:
        raise ValueError('Required A_baseline/weights/best.pt is missing; no fallback allowed')
    snapshot=verify_dataset(root)
    experiment=json.loads((checkpoint.parent.parent/'experiment.json').read_text(encoding='utf-8'))
    if experiment['training_status']!='PASS' or experiment['dataset_fingerprint']!=snapshot['curated_hash']:
        raise ValueError('Completed baseline does not match current frozen dataset')
    env=runtime_info(device)
    directory=Path(snapshot['data']['val']).resolve()
    images=sorted(p.resolve() for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if len(images)!=snapshot['counts']['val'] or any(p.parent!=directory for p in images):
        raise ValueError('Validation image coverage/path mismatch')
    out=root/'results/evaluation/A_baseline_val'
    if out.resolve()!=out: raise ValueError('Evaluation output must not use a symlink/reparse redirect')
    return dict(root=root,checkpoint=checkpoint,snapshot=snapshot,experiment=experiment,environment=env,
                images=images,output=out,training_hashes=training_fingerprint(root))


def verify_unchanged(state):
    check_unchanged(state['snapshot'],state['root'])
    if training_fingerprint(state['root'])!=state['training_hashes']:
        raise ValueError('Baseline training artifacts changed during evaluation')
    return dict(raw='PASS',curated='PASS',training_artifacts='PASS',test_status='LOCKED / NOT USED')


def protection(state):
    root=state['root']
    return frozen_cache(state['output']/'dataset_cache',
                        [root/'data/curated/v1',root/'train',root/'valid',root/'test'])


def check_model(model):
    if {int(k):v for k,v in model.names.items()}!=EXPECTED_CLASSES:
        raise ValueError('Checkpoint class mapping differs from required labels')


def evaluate(root=PROJECT_ROOT,split='val',device='auto',overwrite=False,model_factory=None):
    state=preflight(root,split,device);out=state['output']
    prepare_output(out,overwrite)
    status=dict(status='RUNNING',test_status='LOCKED / NOT USED')
    try:
        data=state['snapshot']['data']
        # TRAIN is present only to satisfy the framework YAML schema. TEST is absent.
        write_yaml(out/'validation_data.yaml',{k:data[k] for k in ('path','train','val','nc','names')})
        write_json(out/'dataset_fingerprint.json',state['snapshot'])
        args=dict(data=str(out/'validation_data.yaml'),split='val',imgsz=416,batch=4,workers=0,
                  device=state['environment']['device'],conf=0.001,iou=0.7,max_det=300,
                  quantize=32,
                  plots=True,save_json=False,save_txt=False,cache=False,augment=False,
                  project=str(out.parent),name=out.name,exist_ok=True)
        write_yaml(out/'evaluation_args.yaml',args)
        if model_factory is None:
            from ultralytics import YOLO
            model_factory=YOLO
        with protection(state):
            model=model_factory(str(state['checkpoint']));check_model(model)
            result=model.val(**args)
        metrics=serialize_metrics(result)
        metrics.update(model=str(state['checkpoint']),checkpoint_sha256=sha256_file(state['checkpoint']),
            dataset_version='curated_v1',dataset_fingerprint=state['snapshot']['curated_hash'],
            validation_images=len(state['images']),test_status='LOCKED / NOT USED',
            environment=state['environment'],evaluation_args=args)
        metrics['precision_policy']='FP32 on CPU/CUDA. Installed Ultralytics disables AMP training on GTX 1650 due to possible zero-mAP/NaN results; evaluation must not force FP16.'
        metrics['training_comparison']=compare_training(metrics,state['experiment'])
        save_metrics(out,metrics)
        from ultralytics.utils.metrics import Metric
        def fitness(values):
            metric=Metric();metric.mean_results=lambda:values
            return metric.fitness()
        summary=training_summary(state['checkpoint'].parent.parent,state['experiment'],fitness,out,
                                 model.ckpt.get('train_metrics',{}))
        # Preserve framework filenames, and provide the requested conventional aliases.
        curves={}
        for name in ('PR_curve.png','F1_curve.png','P_curve.png','R_curve.png'):
            source=out/('Box'+name) if (out/('Box'+name)).is_file() else out/name
            if not source.is_file(): raise ValueError(f'Framework curve missing: {name}')
            if source.name!=name: shutil.copy2(source,out/name)
            curves[name]=source.name
        for name in ('confusion_matrix.png','confusion_matrix_normalized.png'):
            if not (out/name).is_file(): raise ValueError(f'Framework artifact missing: {name}')
        write_json(out/'curve_artifacts.json',curves)
        status.update(status='PASS',best_epoch=summary['best_epoch'],validation_images=len(state['images']))
        print(json.dumps(dict(overall=metrics['overall'],per_class=metrics['per_class'],
                              best_epoch=summary['best_epoch'],warnings=metrics['training_comparison']['warnings']),indent=2))
    except BaseException as error:
        status.update(status='FAIL',error=str(error));raise
    finally:
        try: status['immutability']=verify_unchanged(state)
        except BaseException as error:
            status.update(status='FAIL',immutability_error=str(error));raise
        finally: write_json(out/'evaluation_status.json',status)
    print('BASELINE VALIDATION EVALUATION: PASS\nTEST: LOCKED / NOT USED')
    return metrics


def main(argv=None):
    parser=argparse.ArgumentParser(description='Evaluate only baseline best.pt on curated VALID; TEST locked.')
    parser.add_argument('--split',default='val')
    parser.add_argument('--device',default='auto')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args(argv)
    try:
        evaluate(split=args.split,device=args.device,overwrite=args.overwrite);return 0
    except (OSError,ValueError,RuntimeError,KeyError) as error:
        print(f'BASELINE VALIDATION EVALUATION: FAIL\n{error}');return 1
