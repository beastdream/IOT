import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from smart_helmet.dataset.audit import parse_yolo_line, run_audit, calculate_status
from smart_helmet.dataset.duplicates import (source_group_id, sha256_file, perceptual_hash,
    hamming_distance, similarity, exact_duplicates, near_duplicates, connected_group_count)
from smart_helmet.dataset.visual_review import draw_annotation


@pytest.mark.parametrize('line,issue', [
    ('0 .5 .5 .1', 'FIELD_COUNT'), ('0 a .5 .1 .1', 'NON_NUMERIC'),
    ('0 nan .5 .1 .1', 'NON_FINITE'), ('0 .5 inf .1 .1', 'NON_FINITE'),
    ('0.5 .5 .5 .1 .1', 'CLASS_NOT_INTEGER'), ('2 .5 .5 .1 .1', 'INVALID_CLASS_ID'),
    ('-1 .5 .5 .1 .1', 'INVALID_CLASS_ID'), ('0 .5 .5 0 .1', 'INVALID_COORDINATES'),
    ('0 1.1 .5 .1 .1', 'INVALID_COORDINATES'), ('0 .5 .5 -.1 .1', 'INVALID_COORDINATES'),
    ('0 .5 .5 .1 1.1', 'INVALID_COORDINATES'),
])
def test_invalid_line(line, issue):
    box, issues = parse_yolo_line(line)
    assert box is None
    assert issues[0][0] == issue


def test_geometry_and_tolerance():
    box, issues = parse_yolo_line('1 .5 .5 .2 .4')
    assert issues == []
    assert box['class_id'] == 1
    assert box['normalized_area'] == pytest.approx(.08)
    assert box['aspect_ratio'] == pytest.approx(.5)
    assert box['size_category'] == 'medium'
    assert parse_yolo_line('0 .1 .5 .200001 .1')[1] == []
    assert parse_yolo_line('0 .1 .5 .20001 .1')[1][0][0] == 'OUT_OF_BOUNDS'
    assert parse_yolo_line('0 .5 .5 .00001 .1')[1][0][0] == 'NEAR_ZERO'


@pytest.mark.parametrize('name,expected', [
    ('frame.png.rf.abcdef123.jpg', 'frame.png'), ('frame.rf.123.PNG', 'frame'),
    ('frame.jpg', 'frame'), ('some.rf.text.jpg', 'some.rf.text'),
])
def test_source_id(name, expected):
    assert source_group_id(name) == expected


def test_hashes_and_cross_split(tmp_path):
    rng = np.random.default_rng(17)
    im = Image.fromarray(rng.integers(0,256,(64,64,3),dtype=np.uint8))
    a = tmp_path / 'a.png'; b = tmp_path / 'b.png'
    im.save(a); b.write_bytes(a.read_bytes())
    assert sha256_file(a) == sha256_file(b)
    ph = perceptual_hash(im)
    assert hamming_distance(ph, perceptual_hash(im.copy())) == 0
    assert similarity(0) == similarity(4) == 'VERY_SIMILAR'
    assert similarity(5) == similarity(8) == 'POSSIBLY_SIMILAR'
    assert similarity(9) is None
    records = [dict(image_path='train/a.png',split='train',sha256=sha256_file(a),perceptual_hash=ph),
               dict(image_path='test/b.png',split='test',sha256=sha256_file(b),perceptual_hash=ph)]
    assert exact_duplicates(records)[0]['severity'] == 'CRITICAL'
    pairs = near_duplicates(records)
    assert len(pairs) == 1 and pairs[0]['cross_split'] and pairs[0]['severity'] == 'HIGH'
    assert connected_group_count(pairs) == 1
    records[1]['split'] = 'train'
    assert exact_duplicates(records)[0]['severity'] == 'REVIEW'
    assert near_duplicates(records)[0]['severity'] == 'REVIEW'
    records[1]['perceptual_hash'] = f'{int(ph,16) ^ ((1 << 12)-1):016x}'
    assert near_duplicates(records) == []


def mini_dataset(root):
    (root / 'configs').mkdir()
    config = dict(path='.', train='train/images', val='valid/images', test='test/images',
                  nc=2, names={0:'With Helmet',1:'Without Helmet'})
    (root / 'configs/data.local.yaml').write_text(yaml.safe_dump(config))
    for n in ('data.yaml','README.dataset.txt','README.roboflow.txt'):
        (root / n).write_text('original')
    rng = np.random.default_rng(5)
    for split in ('train','valid','test'):
        (root / split / 'images').mkdir(parents=True)
        (root / split / 'labels').mkdir()
        im = Image.fromarray(rng.integers(0,256,(80,100,3),dtype=np.uint8))
        im.save(root / split / 'images/frame.png')
        (root / split / 'labels/frame.txt').write_text('0 .5 .5 .2 .2\n1 .2 .2 .02 .02\n')
    return root / 'configs/data.local.yaml'


def test_manifest_reports_and_visuals(tmp_path):
    config = mini_dataset(tmp_path)
    output = tmp_path / 'results/audit'
    report = run_audit(tmp_path,config,output)
    assert report['dataset']['total_images'] == 3
    assert report['dataset']['total_annotations'] == 6
    assert report['classes']['counts'] == {0:3,1:3}
    assert report['bbox']['tiny'] == 3
    assert report['status'] == 'REVIEW_REQUIRED'
    assert report['immutability'] == {'status':'PASS','files_checked':9}
    assert report['source_groups']['cross_split_groups'] == 1
    rows = list(csv.DictReader((output / 'dataset_manifest.csv').open()))
    assert len(rows) == 3
    assert all(r['sha256'] and r['perceptual_hash'] and r['image_width'] == '100' for r in rows)
    for filename in ('image_issues','annotation_issues','bbox_statistics','suspicious_bboxes',
                     'class_distribution','exact_duplicates','source_groups','source_group_leakage',
                     'near_duplicates','near_duplicate_leakage','image_resolutions','manual_review'):
        assert (output / f'{filename}.csv').is_file()
    assert len(list((output / 'visual_review').iterdir())) == 9
    rendered = next((output / 'visual_review/random_train_samples').glob('*.png'))
    with Image.open(rendered) as im:
        assert im.size == (100,80)
    saved = json.loads((output / 'dataset_audit_report.json').read_text())
    assert saved['immutability']['status'] == 'PASS'


def test_corruption_and_bad_labels_do_not_abort(tmp_path):
    config = mini_dataset(tmp_path)
    (tmp_path / 'train/images/frame.png').write_bytes(b'broken')
    (tmp_path / 'train/labels/frame.txt').write_text('0 .5 .5 .2 .2\n0 .5 .5 .2 .2\n2 .5 .5 .1 .1\n')
    (tmp_path / 'valid/labels/frame.txt').unlink()
    (tmp_path / 'test/labels/orphan.txt').write_text('0 .5 .5 .1 .1')
    output = tmp_path / 'results/audit'
    report = run_audit(tmp_path,config,output,visuals=False)
    assert report['status'] == 'FAIL'
    assert report['images']['corrupted'] == 1
    assert report['images']['missing_labels'] == 1
    assert report['images']['orphan_labels'] == 1
    assert report['annotations']['invalid'] == 1
    assert report['annotations']['issue_counts']['DUPLICATE_ANNOTATION'] == 1
    assert report['dataset']['total_images'] == 3
    assert report['negative_images']['overall']['images_with_zero_objects'] == 0
    assert report['immutability']['status'] == 'PASS'


def test_empty_label_not_missing_label(tmp_path):
    config = mini_dataset(tmp_path)
    (tmp_path / 'train/labels/frame.txt').write_text(' \n')
    report = run_audit(tmp_path,config,tmp_path/'results/audit',visuals=False)
    assert report['negative_images']['overall']['images_with_zero_objects'] == 1
    assert report['images']['missing_labels'] == 0


def test_config_failure_and_output_guard(tmp_path):
    config = mini_dataset(tmp_path)
    config.write_text('names: [')
    report = run_audit(tmp_path,config,tmp_path/'results/audit',visuals=False)
    assert report['status'] == 'FAIL'
    assert report['fatal_errors']
    with pytest.raises(ValueError):
        run_audit(tmp_path,config,tmp_path/'train/images')


def test_status_rules():
    assert calculate_status(False,[],[],[]) == 'PASS'
    assert calculate_status(False,[],[],[{}]) == 'REVIEW_REQUIRED'
    assert calculate_status(True,[],[],[]) == 'FAIL'
    assert calculate_status(False,[{'severity':'ERROR'}],[],[]) == 'FAIL'
    assert calculate_status(False,[],[{'severity':'ERROR'}],[]) == 'FAIL'


def test_rendered_box_has_class_color(tmp_path):
    path = tmp_path/'im.png'; Image.new('RGB',(100,100),'white').save(path)
    box, _ = parse_yolo_line('1 .5 .5 .4 .4')
    canvas = draw_annotation(path,[box])
    assert canvas.getpixel((30,50)) == (255,72,60)
    with Image.open(path) as original:
        assert original.getpixel((30,50)) == (255,255,255)
