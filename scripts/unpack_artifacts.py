"""Safely extract the two research archives; never run the experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def unpack(output):
    manifest = json.loads((ROOT / 'provenance/project_manifest.json').read_text())
    archives = [('experiment_20261101.zip', 'experiment_20261101'),
                ('reference_inputs.zip', 'reference_inputs')]
    # Check every archive and every target before writing any files.
    for name, target in archives:
        path = ROOT / 'artifacts' / name
        expected = manifest['files']['artifacts/' + name]['sha256']
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Archive checksum mismatch: ' + name)
        if (output / target).exists():
            raise FileExistsError(output / target)
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                p = Path(info.filename)
                if p.is_absolute() or '..' in p.parts or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError('Unsafe archive path: ' + info.filename)
    for name, target in archives:
        folder = output / target
        folder.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(ROOT / 'artifacts' / name) as archive:
            archive.extractall(folder)
        print(folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'unpacked')
    args = parser.parse_args()
    unpack(args.output.resolve())
