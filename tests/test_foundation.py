from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from smart_helmet import paths
from smart_helmet.foundation import (
    EXPECTED_CLASSES, EXPECTED_COUNTS, SPLIT_FOLDERS,
    count_images, load_data_config, verify_foundation,
)


def test_paths_ignore_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert paths.PROJECT_ROOT == Path(__file__).resolve().parents[1]
    assert paths.DATA_CONFIG == paths.PROJECT_ROOT / 'configs/data.local.yaml'
    assert paths.ORIGINAL_DATA_CONFIG == paths.PROJECT_ROOT / 'data.yaml'
    for name, relative in {
        'TRAIN_IMAGES_DIR': 'train/images', 'TRAIN_LABELS_DIR': 'train/labels',
        'VALID_IMAGES_DIR': 'valid/images', 'VALID_LABELS_DIR': 'valid/labels',
        'TEST_IMAGES_DIR': 'test/images', 'TEST_LABELS_DIR': 'test/labels',
        'RESULTS_DIR': 'results', 'DATASET_AUDIT_DIR': 'results/dataset_audit',
        'DATASET_ANALYSIS_DIR': 'results/dataset_analysis',
        'EXPERIMENTS_DIR': 'results/experiments', 'EVALUATION_DIR': 'results/evaluation',
        'PREDICTIONS_DIR': 'results/predictions',
        'FAILURE_ANALYSIS_DIR': 'results/failure_analysis',
    }.items():
        assert getattr(paths, name) == paths.PROJECT_ROOT / relative
        assert getattr(paths, name).is_dir()
    data = load_data_config()
    assert Path(data['path']) == paths.PROJECT_ROOT
    for split, folder in SPLIT_FOLDERS.items():
        assert Path(data[split]) == paths.PROJECT_ROOT / folder / 'images'


def test_yaml_and_original_mapping():
    local = yaml.safe_load(paths.DATA_CONFIG.read_text(encoding='utf-8'))
    original = yaml.safe_load(paths.ORIGINAL_DATA_CONFIG.read_text(encoding='utf-8'))
    assert local['nc'] == original['nc'] == 2
    assert local['names'] == dict(enumerate(original['names'])) == EXPECTED_CLASSES
    assert local['path'] == '.'
    # Native Ultralytics primary paths with cwd=project root; no ../ fallback.
    for split, folder in SPLIT_FOLDERS.items():
        assert local[split] == f'{folder}/images'
        assert (paths.PROJECT_ROOT / local['path'] / local[split]).is_dir()


@pytest.mark.parametrize('split,folder', SPLIT_FOLDERS.items())
def test_split_counts_and_labels(split, folder):
    assert (paths.PROJECT_ROOT / folder / 'labels').is_dir()
    assert count_images(paths.PROJECT_ROOT / folder / 'images') == EXPECTED_COUNTS[split]


def test_multiple_image_extensions(tmp_path):
    for name in ['a.jpg', 'b.JPEG', 'c.PNG', 'note.txt']:
        (tmp_path / name).touch()
    (tmp_path / 'directory.jpg').mkdir()
    assert count_images(tmp_path) == 3


@pytest.mark.parametrize('change', [
    {'nc': 3}, {'nc': 2.0}, {'names': {0: 'Without Helmet', 1: 'With Helmet'}},
    {'names': {0: 'helmet', 1: 'no_helmet'}}, {'train': 'missing/images'},
    {'path': '..'},
])
def test_invalid_config_rejected(tmp_path, change):
    data = yaml.safe_load(paths.DATA_CONFIG.read_text(encoding='utf-8'))
    data.update(change)
    config = tmp_path / 'data.yaml'
    config.write_text(yaml.safe_dump(data), encoding='utf-8')
    with pytest.raises(ValueError):
        load_data_config(config)


def test_malformed_yaml_rejected(tmp_path):
    config = tmp_path / 'data.yaml'
    config.write_text('names: [', encoding='utf-8')
    with pytest.raises(yaml.YAMLError):
        load_data_config(config)


def test_wrong_counts_fail_without_repair(tmp_path, capsys):
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/data.local.yaml').write_bytes(paths.DATA_CONFIG.read_bytes())
    for name in ['data.yaml', 'README.dataset.txt', 'README.roboflow.txt']:
        (tmp_path / name).write_text('fixture', encoding='utf-8')
    for folder in SPLIT_FOLDERS.values():
        (tmp_path / folder / 'images').mkdir(parents=True)
        (tmp_path / folder / 'labels').mkdir()
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert verify_foundation(tmp_path) is False
    assert 'RESULT:\nFAIL' in capsys.readouterr().out
    after = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert before == after
    assert all(count_images(tmp_path / f / 'images') == 0 for f in SPLIT_FOLDERS.values())


def test_missing_project_fails(tmp_path, capsys):
    assert verify_foundation(tmp_path / 'missing') is False
    assert 'RESULT:\nFAIL' in capsys.readouterr().out


def test_cli_outside_project_root(tmp_path):
    result = subprocess.run(
        [sys.executable, str(paths.PROJECT_ROOT / 'scripts/verify_foundation.py')],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'RESULT:\nPASS' in result.stdout
