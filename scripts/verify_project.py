"""Verify distribution hashes, archived training data and the paired-run evidence."""
import ast
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def rows(archive, name):
    return list(csv.DictReader(io.StringIO(archive.read(name).decode('utf-8'))))


def verify():
    manifest = json.loads((ROOT / 'provenance/project_manifest.json').read_text())
    for name, meta in manifest['files'].items():
        p = ROOT / name
        assert p.is_file(), name
        assert p.stat().st_size == meta['bytes'] and digest(p.read_bytes()) == meta['sha256'], name
    source_map = json.loads((ROOT / 'provenance/source_map.json').read_text())
    for record in source_map:
        assert digest((ROOT / record['project_path']).read_bytes()) == record['sha256'], record
    for p in ROOT.rglob('*.py'):
        if not any(s in p.parts for s in ['unpacked', '.git', '__pycache__']):
            if 'vendor' in p.parts and sys.version_info < (3, 12):
                continue  # Upstream uses PEP 701 f-strings; its runtime is Python 3.12.
            ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
    result = json.loads((ROOT / 'results/comparison.json').read_text())
    for key in ['paired_initialization_verified', 'paired4epoch_batches_LRs_verified',
                'shared_actual_training_params_verified', 'query_gallery_verified', 'all_three_retrained']:
        assert result[key] is True, key
    assert result['generation_seed'] == 43
    paired = json.loads((ROOT / 'results/paired_training_audit.json').read_text())
    initial = paired['baseline']['manifest']['initial_model_sha256']
    baseline_epochs = paired['baseline']['epochs']
    for arm in ['baseline', 'original', 'new_method']:
        group = result['groups'][arm]
        assert group['hitk']['epoch'] == 4 and group['hitk']['queries'] == 5779
        assert paired[arm]['manifest']['initial_model_sha256'] == initial
        assert len(paired[arm]['epochs']) == 4
        for actual, reference in zip(paired[arm]['epochs'], baseline_epochs):
            for key in ['epoch', 'samples', 'num_batches', 'nominal_schedule_epochs', 'batch_order_sha256', 'lr_last']:
                assert actual[key] == reference[key], (arm, key)
    with zipfile.ZipFile(ROOT / 'artifacts/reference_inputs.zip') as reference, \
         zipfile.ZipFile(ROOT / 'artifacts/experiment_20261101.zip') as experiment:
        for archive in [reference, experiment]:
            assert archive.testzip() is None
            for info in archive.infolist():
                p = Path(info.filename)
                assert not p.is_absolute() and '..' not in p.parts
                assert not stat.S_ISLNK(info.external_attr >> 16)
        original_bytes = reference.read('wbd_data/victim/attack_train_wbd_100k.csv')
        assert digest(original_bytes) == '007723441ebe295c382b66057b533b0823b49f35b3058a0f704032c5e59283b8'
        clean = rows(reference, 'wbd_data/victim/clean_train_100k.csv')
        original = list(csv.DictReader(io.StringIO(original_bytes.decode())))
        assert len(clean) == 100000 and len(original) == 100100
        clean_remaining = Counter((x['path'], x['caption']) for x in clean)
        clean_indices = []
        poison_indices = []
        for i, row in enumerate(original):
            key = row['path'], row['caption']
            if clean_remaining[key]:
                clean_indices.append(i); clean_remaining[key] -= 1
            else:
                poison_indices.append(i)
        assert not any(clean_remaining.values()) and len(poison_indices) == 100
        paths = {'baseline': 'datasets/baseline.csv',
                 'original': 'original/selected/attack_train_100k.csv',
                 'new_method': 'new_method/selected/attack_train_100k.csv'}
        for arm, relative in paths.items():
            content = experiment.read('full_repeat_01/' + relative)
            assert digest(content) == result['groups'][arm]['dataset_audit']['sha256'], arm
            actual = list(csv.DictReader(io.StringIO(content.decode())))
            assert len(actual) == 100100
            assert all(x['path'] == y['path'] for x, y in zip(actual, original)), arm
            assert all(actual[i] == original[i] for i in clean_indices), arm
            assert sorted(Counter(actual[i]['path'] for i in poison_indices).values()) == [5] * 20
            assert all(actual[i]['caption'].endswith(' zx') for i in poison_indices), arm
        for arm in ['original', 'new_method']:
            generated = rows(experiment, f'full_repeat_01/{arm}/generated/candidates.csv')
            assert len(generated) == 400
            assert Counter(int(x['source_index']) for x in generated) == Counter({i: 2 for i in range(200)})
            chosen = rows(experiment, f'full_repeat_01/{arm}/selected/selected.csv')
            assert len(chosen) == 100 and sorted(Counter(x['category'] for x in chosen).values()) == [5] * 20
    print('PASS: distribution hashes, source maps, Python syntax, both 400-candidate runs,')
    print('three datasets, clean rows/poison positions, shared initialization and 4-epoch batch/LR evidence.')
    print('This does not rerun model inference or verify external image/model bytes.')
    if sys.version_info < (3, 12):
        print('Vendor syntax check skipped on Python <3.12; vendor bytes were hash-verified.')


if __name__ == '__main__':
    verify()
