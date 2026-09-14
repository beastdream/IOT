"""Read-only environment and frozen dataset preflight; no checkpoint loading."""
import hashlib
import json
import platform
from pathlib import Path

from smart_helmet.paths import PROJECT_ROOT
from smart_helmet.foundation import load_detection_config, count_images
from smart_helmet.dataset.curation import training_readiness, verify_curated_fingerprint, raw_snapshot


def digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def resolve_device(requested, torch):
    available = torch.cuda.is_available()
    if str(requested) == 'auto':
        if not available:
            return 'cpu', 'CUDA unavailable -> using CPU.'
        return str(torch.cuda.current_device()), 'Using the current CUDA device.'
    if str(requested) == 'cpu':
        return 'cpu', 'CPU explicitly selected.'
    value = str(requested).removeprefix('cuda:')
    if not value.isdigit() or not available or int(value) >= torch.cuda.device_count():
        raise ValueError('Requested CUDA device unavailable. Use --device cpu or auto; single GPU only.')
    return value, 'Explicit CUDA device.'


def resolve_batch(value, device, auto_supported):
    if str(value) == 'auto':
        if device != 'cpu' and auto_supported:
            return -1, 'Ultralytics single-GPU auto-batch (-1); actual batch recorded after setup.'
        return 2, 'Conservative fixed batch 2 (CPU or no supported GPU auto-batch). Override with --batch.'
    try:
        batch = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError('Batch must be auto or a positive integer') from error
    if str(batch) != str(value) or batch < 1:
        raise ValueError('Batch must be auto or a positive integer')
    return batch, f'Explicit fixed batch {batch}.'


def runtime_info(device='auto'):
    import torch
    import ultralytics
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from ultralytics.data import dataset
    resolved, message = resolve_device(device, torch)
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append(dict(index=index, name=props.name, memory_bytes=props.total_memory))
    package = Path(ultralytics.__file__).parent
    if not (package/'cfg/models/11/yolo11.yaml').is_file():
        raise ValueError('Installed Ultralytics lacks official YOLO11 architecture; choose a supported official nano model explicitly.')
    if not hasattr(dataset, 'save_dataset_cache_file'):
        raise ValueError('Installed label-cache API unsupported; frozen dataset protection cannot be guaranteed.')
    return dict(python_version=platform.python_version(), torch_version=torch.__version__,
                ultralytics_version=ultralytics.__version__, cuda_available=torch.cuda.is_available(),
                cuda_version=torch.version.cuda, gpu_count=len(gpus), gpus=gpus,
                gpu_name=next((g['name'] for g in gpus if str(g['index']) == resolved), None),
                device=resolved, device_message=message, auto_batch_supported=hasattr(BaseTrainer,'auto_batch'),
                fraction_supported='fraction' in DEFAULT_CFG_DICT, model_checkpoint='yolo11n.pt')


def verify_dataset(root=PROJECT_ROOT, config=None):
    root = Path(root).resolve()
    config = Path(config or root/'configs/data.curated.v1.yaml').resolve()
    if config != root/'configs/data.curated.v1.yaml':
        raise ValueError('Baseline must use configs/data.curated.v1.yaml')
    data = load_detection_config(config, root)
    if Path(data['path']) != root/'data/curated/v1':
        raise ValueError('Baseline dataset must be curated v1')
    readiness = training_readiness(root)
    if not readiness['ready_for_baseline_training']:
        raise ValueError('Training readiness failed: '+json.dumps(readiness))
    curated = verify_curated_fingerprint(root)
    raw = raw_snapshot(root)
    counts = {s: count_images(Path(data[s])) for s in ('train','val','test')}
    if not all(counts.values()):
        raise ValueError('Empty dataset split')
    return dict(config=str(config), data=data, counts=counts, readiness=readiness,
                curated=curated, raw=raw, curated_hash=digest(curated), raw_hash=digest(raw))


def check_unchanged(snapshot, root=PROJECT_ROOT):
    if verify_curated_fingerprint(Path(root)) != snapshot['curated'] or raw_snapshot(Path(root)) != snapshot['raw']:
        raise ValueError('Dataset changed during training')
    return 'PASS'


def show_environment(runtime, snapshot, output=print):
    output('='*40+'\nTRAINING ENVIRONMENT\n'+'='*40)
    for key, value in runtime.items(): output(f'{key}: {value}')
    output(f'Project root: {Path(snapshot["config"]).parent.parent}')
    output(f'Dataset config: {snapshot["config"]}')
    for split in ('train','val','test'):
        output(f'{split}: {snapshot["data"][split]} | {snapshot["counts"][split]} images | '+('LOCKED' if split=='test' else 'PASS'))
    output(f'Class mapping: {snapshot["data"]["names"]}')
    output('Curated fingerprint: PASS\nTraining readiness: YES\nREADY FOR BASELINE PIPELINE: YES\n'+'='*40)


def main():
    try:
        show_environment(runtime_info(), verify_dataset())
        return 0
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        print(f'TRAINING ENVIRONMENT: FAIL\n{error}\nREADY FOR BASELINE PIPELINE: NO')
        return 1
