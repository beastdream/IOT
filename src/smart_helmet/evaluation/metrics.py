"""Framework metric serialization and evidence-based training summaries."""
import math
from pathlib import Path
import statistics

from smart_helmet.foundation import EXPECTED_CLASSES
from smart_helmet.dataset.audit import write_csv
from smart_helmet.training.experiment import result_rows, write_json

KEYS = ('precision','recall','mAP50','mAP50-95')
FRAMEWORK_KEYS = ('metrics/precision(B)','metrics/recall(B)','metrics/mAP50(B)','metrics/mAP50-95(B)')


def metric_values(values):
    result = dict(zip(KEYS, map(float, values)))
    if len(result) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in result.values()):
        raise ValueError('Invalid framework metrics')
    return result


def serialize_metrics(metrics):
    if {int(k):v for k,v in metrics.names.items()} != EXPECTED_CLASSES:
        raise ValueError('Unexpected model class mapping')
    indices = {int(c):i for i,c in enumerate(metrics.box.ap_class_index)}
    if set(indices) != set(EXPECTED_CLASSES):
        raise ValueError('Both validation classes must have framework metrics')
    return dict(overall=metric_values(metrics.box.mean_results()),
                per_class={str(c):dict(class_name=name, **metric_values(metrics.box.class_result(indices[c])))
                           for c,name in EXPECTED_CLASSES.items()})


def save_metrics(output, metrics):
    output = Path(output)
    write_json(output/'metrics.json',metrics)
    rows = [dict(scope='overall',class_id='',class_name='All',**metrics['overall'])]
    rows += [dict(scope='class',class_id=c,**v) for c,v in metrics['per_class'].items()]
    write_csv(output/'metrics.csv',rows,['scope','class_id','class_name',*KEYS])


def compare_training(metrics, experiment, tolerance=0.03):
    saved = experiment.get('validation_metrics',{})
    differences = {k:metrics['overall'][k]-float(saved[f]) for k,f in zip(KEYS,FRAMEWORK_KEYS) if f in saved}
    warnings = [f'{k} differs from final training validation by {d:+.4f} (absolute tolerance {tolerance}).'
                for k,d in differences.items() if abs(d)>tolerance]
    if len(differences)!=4: warnings.append('Final training validation metadata is incomplete.')
    return dict(absolute_warning_tolerance=tolerance,differences=differences,warnings=warnings,
                note='Independent VALID evaluation; precision/recall are framework operating-point metrics, not fixed-0.25 diagnostic counts.')


def training_summary(directory, experiment, fitness_fn, output=None, checkpoint_metrics=None):
    rows = result_rows(directory)
    if not rows: raise ValueError('Training results.csv is empty')
    epochs = [int(r['epoch']) for r in rows]
    if epochs != list(range(1,len(rows)+1)): raise ValueError('Training epoch sequence is not contiguous')
    scores = [fitness_fn([float(r[k]) for k in FRAMEWORK_KEYS]) for r in rows]
    best_index = max(range(len(rows)),key=lambda i:scores[i])
    best = epochs[best_index]
    requested = int(experiment['epochs_requested'])
    patience = int(experiment['resolved_training_arguments']['patience'])
    early = len(rows)<requested and len(rows)-best>=patience and experiment['training_status']=='PASS'
    fields = ['train/box_loss','train/cls_loss','train/dfl_loss','val/box_loss','val/cls_loss','val/dfl_loss',*FRAMEWORK_KEYS[2:]]
    trends = {}
    for key in fields:
        values = [float(r[key]) for r in rows]
        trends[key] = dict(first=values[0],last=values[-1],best_epoch_value=values[best_index],
                           first_five_mean=statistics.mean(values[:5]),last_five_mean=statistics.mean(values[-5:]),
                           minimum=min(values),maximum=max(values))
    checkpoint_metrics = checkpoint_metrics or {}
    saved_fitness = checkpoint_metrics.get('fitness')
    checkpoint_matches = saved_fitness is not None and abs(float(saved_fitness)-scores[best_index])<0.001
    summary = dict(best_epoch=best,best_epoch_expected=36,best_epoch_confirmed=best==36 and checkpoint_matches,
        best_fitness=scores[best_index],checkpoint_fitness=saved_fitness,checkpoint_matches_best_csv=checkpoint_matches,
        criterion='Installed Ultralytics Metric.fitness applied to per-epoch validation metrics',
        epochs_completed=len(rows),epochs_requested=requested,early_stopping_consistent=early,patience=patience,
        training_loop_seconds=float(rows[-1]['time']),pipeline_duration_seconds=experiment['duration_seconds'],
        trends=trends,epochs=[dict(epoch=e,**{k:float(r[k]) for k in fields}) for e,r in zip(epochs,rows)],
        warnings=[])
    if best!=36: summary['warnings'].append(f'Observed best epoch {best}, not the expected 36.')
    if len(rows)!=56: summary['warnings'].append(f'Observed {len(rows)} completed epochs, not the expected 56.')
    if not checkpoint_matches: summary['warnings'].append('Checkpoint fitness does not confirm CSV best epoch; review required.')
    if output:
        output = Path(output)
        write_json(output/'training_summary.json',summary)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2,4,figsize=(16,7))
        for axis,key in zip(axes.flat,fields):
            axis.plot(epochs,[float(r[key]) for r in rows]);axis.axvline(best,color='orange',linestyle='--')
            axis.set_title(key);axis.set_xlabel('Epoch');axis.grid(alpha=.25)
        fig.suptitle(f'TRAIN / VALID only | Best epoch {best} | Completed {len(rows)}')
        fig.tight_layout();fig.savefig(output/'training_curves.png',dpi=160);plt.close(fig)
    return summary
