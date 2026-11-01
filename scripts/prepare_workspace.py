"""Prepare a NEW source workspace without modifying archived CSVs or starting training.

Frozen paths require Linux HOME=/home/million and the historical destination.
The script refuses an existing destination and only links external image/model assets.
"""
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = Path('/home/million/ToxicTextCLIP_W-BD')
ASSETS = ['wbd_data/victim/clean_train_images', 'wbd_data/victim/val_images',
          'wbd_data/cc12m_boat/images', 'wbd_data/nonboat20/images',
          'datasets/coco', 'models/checkpoints/epoch_14.pth']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unpacked', type=Path, default=ROOT / 'unpacked')
    parser.add_argument('--assets-root', type=Path, required=True,
                        help='Separate external assets directory preserving historical relative paths')
    parser.add_argument('--destination', type=Path, default=HISTORICAL)
    args = parser.parse_args()
    destination = args.destination.absolute()
    if destination != HISTORICAL or Path.home() != Path('/home/million'):
        parser.error('Frozen source and CSV paths require HOME=/home/million and destination=' + str(HISTORICAL))
    if destination.exists() or destination.is_symlink():
        parser.error('Refusing existing workspace; do not overwrite the original experiment')
    inputs = args.unpacked / 'reference_inputs'
    if not (inputs / 'wbd_data/victim/clean_train_100k.csv').is_file():
        parser.error('Run unpack_artifacts.py first')
    assets = args.assets_root.resolve()
    missing = [name for name in ASSETS if not (assets / name).exists()]
    if missing:
        parser.error('Missing external assets: ' + ', '.join(missing))
    if not (Path.home() / '.cache/clip/ViT-B-32.pt').is_file():
        parser.error('Missing ~/.cache/clip/ViT-B-32.pt; see docs/ASSETS.md')
    shutil.copytree(ROOT / 'vendor/toxictextclip', destination)
    shutil.copytree(inputs, destination, dirs_exist_ok=True)
    code = destination / 'wbd_data/gda_project_code'
    shutil.copytree(ROOT / 'pipeline', code)
    shutil.copytree(ROOT / 'automatic_detail_method', code / 'automatic_detail_method')
    # Archived completed outputs stay outside the NEW workspace. Copy datasets
    # to a clearly named read-only reference area for optional text-fixed training.
    reference = destination / 'wbd_data/gda_completed_inputs'
    reference.mkdir()
    completed = args.unpacked / 'experiment_20261101/full_repeat_01'
    for arm, source in [('baseline', completed / 'datasets/baseline.csv'),
                        ('original', completed / 'original/selected/attack_train_100k.csv'),
                        ('new_method', completed / 'new_method/selected/attack_train_100k.csv')]:
        shutil.copyfile(source, reference / (arm + '.csv'))
    for name in ASSETS:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(assets / name, target_is_directory=(assets / name).is_dir())
    (destination / 'gda_workspace_provenance.json').write_text(json.dumps({
        'project': str(ROOT), 'assets_root': str(assets),
        'csv_bytes_rewritten': False, 'training_started': False,
        'note': 'Prepared for an explicit NEW run; historical coordination scripts are not auto-executed.'
    }, indent=2) + '\n')
    print('Prepared:', destination)
    print('No generation, training or evaluation started.')


if __name__ == '__main__':
    main()
