"""Experiment output ownership and serializable provenance."""
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import yaml


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')


def write_yaml(path, data):
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def output_path(root, name, smoke=False):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name) or name.endswith('_smoke'):
        raise ValueError('Use a simple experiment name without a _smoke suffix')
    parent = Path(root).resolve()/'results/experiments'
    path = parent/(name+'_smoke' if smoke else name)
    if path.resolve().parent != parent or path.is_symlink():
        raise ValueError('Experiment path escapes its expected directory')
    return path


def prepare_output(path, overwrite=False):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise ValueError(f'Experiment already exists: {path}. Use --overwrite or another experiment_name.')
        # Preserve all old artifacts in a sibling archive; never merge old and new results.
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        path.rename(path.with_name(path.name+'_archive_'+stamp))
    path.mkdir(parents=True, exist_ok=True)


def discover_checkpoints(path, smoke=False):
    best, last = Path(path)/'weights/best.pt', Path(path)/'weights/last.pt'
    if not last.is_file() or last.stat().st_size == 0:
        raise ValueError('Training failed: missing/nonempty last.pt required')
    if not smoke and (not best.is_file() or best.stat().st_size == 0):
        raise ValueError('FULL BASELINE TRAINING STATUS = FAIL: best.pt missing; last.pt is not a substitute')
    return str(best) if best.is_file() else None, str(last)


def result_rows(path):
    with (Path(path)/'results.csv').open(newline='', encoding='utf-8-sig') as stream:
        return [{k.strip():v.strip() for k,v in row.items()} for row in csv.DictReader(stream)]
