"""Import the course-provided CSVs from the original archive without extracting books."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    targets = {'simpsons_dataset.csv': 'simpsons_dataset.csv', 'IMDB Dataset_small.csv': 'imdb_reviews.csv'}
    destination = ROOT / 'data' / 'external'
    destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    with zipfile.ZipFile(args.archive) as archive:
        for source, target in targets.items():
            candidates = [n for n in archive.namelist() if Path(n).name == source and '__MACOSX' not in n]
            if len(candidates) != 1:
                raise ValueError(f'Expected one {source}, found {len(candidates)}')
            data = archive.read(candidates[0])
            path = destination / target
            path.write_bytes(data)
            manifest.append({'file': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(data).hexdigest()})
            print(f'Imported {path.relative_to(ROOT)}')
    (destination / 'import_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
